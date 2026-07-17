"""Stage 2 v2: Target speaker extraction (verification-first architecture).

Flow:
1. UniSE TSE (optional; can reuse cached tse_v1.wav)
2. VAD on TSE → active intervals on original timeline
3. Cut candidate clips from original cleaned audio
4. Speaker verification (embedding similarity vs reference)
5. Quality gate (duration / loudness / normalize)
6. Optional per-clip ASR (no diarization) + text filter
7. Merge accepted clips
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.task_state import TaskState
from src.pipeline.components.unise_tse import run_unise_tse
from src.pipeline.components.vad_detection import detect_active_segments
from src.pipeline.components.speaker_verification import verify_clips
from src.pipeline.components.quality_gate import apply_quality_gate
from src.pipeline.components.clip_asr import transcribe_clips
from src.audio_utils import cut_segment, merge_with_gaps


DEFAULT_COMPONENTS = [
    "unise_tse_v1",
    "vad_segments",
    "speaker_verify",
    "quality_gate",
    "clip_asr",
    "merge_output",
]


def _split_long_intervals(
    intervals: List[Tuple[float, float]],
    max_sec: float,
) -> List[Tuple[float, float]]:
    """Split intervals longer than max_sec into fixed-size chunks."""
    if max_sec <= 0:
        return intervals
    out: List[Tuple[float, float]] = []
    for start, end in intervals:
        if end - start <= max_sec:
            out.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_sec, end)
            if chunk_end - cursor >= 0.3:
                out.append((cursor, chunk_end))
            cursor = chunk_end
    return out


def run_stage2_v2(
    task_state: TaskState,
    cleaned_wavs: List[str | Path],
    reference_dir: str | Path,
    output_dir: str | Path,
    components: Optional[List[str]] = None,
    unise_dir: Optional[str | Path] = None,
    unise_ckpt_path: Optional[str | Path] = None,
    segment_seconds: float = 360.0,
    # Kept for API compatibility with v1 configs / main.py
    asr_speaker_count: Optional[int] = None,
    srt_model: str = "qwen3.6-max",
    run_unise_v2: bool = False,
    asr_gap_sec: float = 2.0,
    clip_padding_sec: float = 0.1,
    min_clip_duration_sec: float = 1.5,
    merge_gap_sec: float = 2.0,
    # v2-specific
    max_clip_duration_sec: float = 12.0,
    vad_method: str = "auto",
    vad_top_db: float = 35.0,
    vad_min_silence_sec: float = 0.3,
    speaker_threshold: float = 0.45,
    speaker_backend: str = "auto",
    speaker_device: str = "cpu",
    normalize_loudness: bool = True,
    target_lufs: float = -23.0,
    skip_asr: bool = False,
    cached_tse_paths: Optional[List[str | Path]] = None,
    **_ignored,
) -> List[Path]:
    """Run Stage 2 v2 (VAD + ECAPA verification) speaker extraction.

    Args:
        cached_tse_paths: Optional precomputed TSE WAVs aligned 1:1 with
            cleaned_wavs. When provided and unise_tse_v1 is disabled (or used
            as override), UniSE is skipped.
    """
    _ = (asr_speaker_count, srt_model, run_unise_v2, asr_gap_sec)  # compat

    components = components or DEFAULT_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)

    reference_files = sorted(
        p for p in reference_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
    )
    if not reference_files:
        raise ValueError(f"No reference audio files found in {reference_dir}")

    task_state.set_inputs({
        "cleaned_wavs": [str(p) for p in cleaned_wavs],
        "reference_dir": str(reference_dir),
        "pipeline_version": "v2",
    })
    task_state.mark_started("stage2")

    all_clips: List[Path] = []

    for episode_idx, cleaned_wav in enumerate(cleaned_wavs):
        cleaned_wav = Path(cleaned_wav)
        ep_output_dir = output_dir / f"episode_{episode_idx:03d}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)
        ep_prefix = f"ep{episode_idx:03d}"

        # -----------------------------------------------------------------
        # 1. UniSE TSE (or reuse cache)
        # -----------------------------------------------------------------
        tse_v1_path = cleaned_wav
        cached = None
        if cached_tse_paths and episode_idx < len(cached_tse_paths):
            cached = Path(cached_tse_paths[episode_idx])

        if "unise_tse_v1" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_unise_tse_v1")
            tse_v1_path = ep_output_dir / f"{ep_prefix}_tse_v1.wav"
            if cached is not None and cached.exists():
                shutil.copy(str(cached), str(tse_v1_path))
                print(f"[stage2] Reusing cached TSE: {cached}")
            else:
                if unise_dir is None or unise_ckpt_path is None:
                    raise ValueError(
                        "unise_tse_v1 enabled but unise_dir/unise_ckpt_path missing "
                        "and no cached_tse_paths provided"
                    )
                run_unise_tse(
                    input_path=cleaned_wav,
                    reference_path=reference_files[0],
                    output_path=tse_v1_path,
                    unise_dir=unise_dir,
                    ckpt_path=unise_ckpt_path,
                    segment_seconds=segment_seconds,
                )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_unise_tse_v1",
                outputs={"tse_v1_path": str(tse_v1_path)},
            )
        elif cached is not None and cached.exists():
            tse_v1_path = ep_output_dir / f"{ep_prefix}_tse_v1.wav"
            shutil.copy(str(cached), str(tse_v1_path))
            print(f"[stage2] Using cached TSE (unise disabled): {cached}")

        # -----------------------------------------------------------------
        # 2. VAD → original-timeline intervals → cut candidates
        # -----------------------------------------------------------------
        candidates: List[Path] = []
        candidate_meta: List[Dict[str, Any]] = []

        if "vad_segments" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_vad_segments")
            vad_map_path = ep_output_dir / f"{ep_prefix}_vad_map.json"
            vad_map = detect_active_segments(
                audio_path=tse_v1_path,
                method=vad_method,
                min_silence_sec=vad_min_silence_sec,
                top_db=vad_top_db,
                map_path=vad_map_path,
            )
            intervals = [
                (s["start_sec"], s["end_sec"]) for s in vad_map["segments"]
            ]
            intervals = _split_long_intervals(intervals, max_clip_duration_sec)

            cand_dir = ep_output_dir / "candidates"
            cand_dir.mkdir(parents=True, exist_ok=True)
            candidates = []
            candidate_meta = []
            for i, (start, end) in enumerate(intervals):
                padded_start = max(0.0, start - clip_padding_sec)
                padded_end = end + clip_padding_sec
                if padded_end - padded_start < 0.3:
                    continue
                out = cand_dir / (
                    f"{ep_prefix}_cand_{i:04d}_{padded_start:.3f}_{padded_end:.3f}.wav"
                )
                cut_segment(cleaned_wav, out, padded_start, padded_end)
                candidates.append(out)
                candidate_meta.append({
                    "path": str(out),
                    "original_start_sec": round(start, 4),
                    "original_end_sec": round(end, 4),
                    "padded_start_sec": round(padded_start, 4),
                    "padded_end_sec": round(padded_end, 4),
                })
            meta_path = ep_output_dir / f"{ep_prefix}_candidates_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"candidates": candidate_meta}, f, indent=2, ensure_ascii=False)

            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_vad_segments",
                outputs={
                    "vad_map_path": str(vad_map_path),
                    "num_candidates": len(candidates),
                    "candidates_dir": str(cand_dir),
                },
            )
        else:
            # No VAD: treat whole file as one candidate (smoke / debug).
            cand_dir = ep_output_dir / "candidates"
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand = cand_dir / f"{ep_prefix}_cand_0000_full.wav"
            shutil.copy(str(cleaned_wav), str(cand))
            candidates = [cand]

        # -----------------------------------------------------------------
        # 3. Speaker verification
        # -----------------------------------------------------------------
        verified_paths: List[Path] = candidates
        verified_meta: List[Dict[str, Any]] = candidate_meta

        if "speaker_verify" in components and candidates:
            task_state.mark_started("stage2", step=f"{ep_prefix}_speaker_verify")
            results_path = ep_output_dir / f"{ep_prefix}_speaker_scores.json"
            verify_result = verify_clips(
                clip_paths=candidates,
                reference_dir=reference_dir,
                threshold=speaker_threshold,
                backend=speaker_backend,
                device=speaker_device,
                results_path=results_path,
            )
            accepted_set = {a["path"] for a in verify_result["accepted"]}
            verified_paths = [Path(p) for p in candidates if str(p) in accepted_set]
            if candidate_meta:
                verified_meta = [
                    m for m in candidate_meta if m["path"] in accepted_set
                ]
            else:
                verified_meta = [{"path": str(p)} for p in verified_paths]

            # Copy accepted into verified/ for clarity.
            verified_dir = ep_output_dir / "verified"
            verified_dir.mkdir(parents=True, exist_ok=True)
            copied: List[Path] = []
            for p in verified_paths:
                dst = verified_dir / p.name
                shutil.copy(str(p), str(dst))
                copied.append(dst)
            # Remap meta paths to verified copies.
            name_to_meta = {Path(m["path"]).name: m for m in verified_meta}
            verified_paths = copied
            verified_meta = []
            for p in verified_paths:
                m = dict(name_to_meta.get(p.name, {}))
                m["path"] = str(p)
                verified_meta.append(m)

            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_speaker_verify",
                outputs={
                    "scores_path": str(results_path),
                    "num_verified": len(verified_paths),
                    "backend": verify_result.get("backend"),
                },
            )

        # -----------------------------------------------------------------
        # 4. Quality gate
        # -----------------------------------------------------------------
        final_paths: List[Path] = verified_paths
        final_meta: List[Dict[str, Any]] = verified_meta

        if "quality_gate" in components and verified_paths:
            task_state.mark_started("stage2", step=f"{ep_prefix}_quality_gate")
            qc_dir = ep_output_dir / "clips"
            qc_result = apply_quality_gate(
                clip_paths=verified_paths,
                output_dir=qc_dir,
                min_duration_sec=min_clip_duration_sec,
                max_duration_sec=max_clip_duration_sec,
                normalize_loudness=normalize_loudness,
                target_lufs=target_lufs,
                results_path=ep_output_dir / f"{ep_prefix}_quality.json",
            )
            final_paths = [Path(a["path"]) for a in qc_result["accepted"]]
            # Preserve original timeline meta by matching source stem.
            src_to_meta = {
                Path(m["path"]).stem: m for m in verified_meta
            }
            remapped: List[Dict[str, Any]] = []
            for a in qc_result["accepted"]:
                src_stem = Path(a["source"]).stem
                m = dict(src_to_meta.get(src_stem, {}))
                m["path"] = a["path"]
                m["duration_sec"] = a.get("duration_sec")
                remapped.append(m)
            final_meta = remapped
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_quality_gate",
                outputs={
                    "clips_dir": str(qc_dir),
                    "num_clips": len(final_paths),
                },
            )
        elif verified_paths:
            # Still place into clips/ even without quality_gate.
            clips_dir = ep_output_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            copied = []
            for p in verified_paths:
                dst = clips_dir / p.name
                shutil.copy(str(p), str(dst))
                copied.append(dst)
            final_paths = copied

        # -----------------------------------------------------------------
        # 5. Optional per-clip ASR
        # -----------------------------------------------------------------
        if "clip_asr" in components and final_paths and not skip_asr:
            task_state.mark_started("stage2", step=f"{ep_prefix}_clip_asr")
            asr_result = transcribe_clips(
                clip_paths=final_paths,
                output_json_path=ep_output_dir / f"{ep_prefix}_clip_asr.json",
                output_srt_path=ep_output_dir / f"{ep_prefix}_cleaned.srt",
                filter_meaningless=True,
                clip_meta=final_meta,
            )
            # Drop clips whose text was filtered as meaningless (if ASR succeeded).
            kept_paths = {
                c["path"] for c in asr_result["clips"]
                if not c.get("filtered") and c.get("text") and not c.get("error")
            }
            # If ASR failed entirely for everything, keep audio clips.
            if kept_paths:
                final_paths = [p for p in final_paths if str(p) in kept_paths]
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_clip_asr",
                outputs={
                    "num_transcribed": asr_result["num_input"],
                    "num_kept": asr_result["num_kept"],
                },
            )

        # -----------------------------------------------------------------
        # 6. Merge
        # -----------------------------------------------------------------
        if "merge_output" in components and final_paths:
            task_state.mark_started("stage2", step=f"{ep_prefix}_merge_output")
            merged_path = ep_output_dir / f"{ep_prefix}_merged_output.wav"
            merge_with_gaps(
                clip_paths=final_paths,
                output_path=merged_path,
                gap_sec=merge_gap_sec,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_merge_output",
                outputs={"merged_output_path": str(merged_path)},
            )

        all_clips.extend(final_paths)

        summary = {
            "episode": episode_idx,
            "num_final_clips": len(final_paths),
            "clips": [str(p) for p in final_paths],
        }
        with open(ep_output_dir / f"{ep_prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    task_state.mark_done(
        "stage2",
        outputs={"final_clips": [str(p) for p in all_clips]},
    )
    return all_clips
