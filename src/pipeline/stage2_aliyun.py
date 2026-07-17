"""Stage 2: Target speaker extraction pipeline.

Orchestrates:
1. UniSE TSE (initial pass).
2. Silence removal with position mapping.
3. Build ASR input from original BGM-removed audio segments (with gaps).
4. Aliyun ASR speaker diarization.
5. pyannote embedding verification against reference audio.
6. Qwen SRT cleaning (target speaker from step 5 when available).
7. Map timestamps back to original audio and cut target clips.
8. Optional second UniSE TSE pass on the clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.task_state import TaskState
from src.pipeline.components.unise_tse import run_unise_tse
from src.pipeline.components.silence_removal import remove_silence
from src.pipeline.components.aliyun_asr import transcribe_with_diarization
from src.pipeline.components.srt_cleaning import clean_srt, parse_srt
from src.pipeline.components.speaker_verify import (
    DEFAULT_EMBEDDING_MODEL,
    identify_target_speaker,
)
from src.pipeline.components.audio_mapping import (
    build_asr_input,
    cut_by_timestamps,
    map_asr_to_original,
    rebuild_from_srt,
)
from src.audio_utils import merge_with_gaps


DEFAULT_COMPONENTS = [
    "unise_tse_v1",
    "silence_removal",
    "aliyun_asr",
    "speaker_verify",
    "srt_cleaning",
    "rebuild_clips",
    "merge_output",
]


def run_stage2_aliyun(
    task_state: TaskState,
    cleaned_wavs: List[str | Path],
    reference_dir: str | Path,
    output_dir: str | Path,
    components: Optional[List[str]] = None,
    unise_dir: Optional[str | Path] = None,
    unise_ckpt_path: Optional[str | Path] = None,
    segment_seconds: float = 360.0,
    asr_speaker_count: Optional[int] = None,
    srt_model: str = "qwen3.6-max",
    run_unise_v2: bool = False,
    asr_gap_sec: float = 2.0,
    clip_padding_sec: float = 0.15,
    min_clip_duration_sec: float = 0.1,
    merge_gap_sec: float = 2.0,
    speaker_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    speaker_verify_samples: int = 8,
    **_ignored,
) -> List[Path]:
    """Run Stage 2 legacy Aliyun ASR + diarization pipeline (fallback).

    Args:
        task_state: TaskState for checkpointing.
        cleaned_wavs: List of cleaned episode WAVs from Stage 1.
        reference_dir: Directory containing reference audio for the target speaker.
        output_dir: Directory for Stage 2 outputs.
        components: Enabled components.
        unise_dir: Path to unified-audio/QuarkAudio-UniSE.
        unise_ckpt_path: Path to UniSE checkpoint.
        segment_seconds: Chunk length for UniSE / ASR.
        asr_speaker_count: Expected speaker count hint for ASR.
        srt_model: Qwen model name for SRT cleaning.
        run_unise_v2: Whether to run a second UniSE pass on final clips.
        asr_gap_sec: Silence gap between segments in ASR input audio (seconds).
        clip_padding_sec: Padding added to each clip when cutting (seconds).
        min_clip_duration_sec: Minimum clip duration; shorter clips are skipped.
        merge_gap_sec: Silence gap between clips in merged output audio (seconds).
        speaker_embedding_model: pyannote embedding model id for speaker verify.
        speaker_verify_samples: Number of clips sampled per diarization speaker.

    Returns:
        List of final target-speaker clip WAV paths.
    """
    components = components or DEFAULT_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(reference_dir)
    reference_files = sorted([p for p in reference_dir.iterdir() if p.is_file()])
    if not reference_files:
        raise ValueError(f"No reference audio files found in {reference_dir}")
    reference_path = reference_files[0]

    task_state.set_inputs({
        "cleaned_wavs": [str(p) for p in cleaned_wavs],
        "reference_dir": str(reference_dir),
        "reference_path": str(reference_path),
    })
    task_state.mark_started("stage2")

    all_clips: List[Path] = []

    for episode_idx, cleaned_wav in enumerate(cleaned_wavs):
        cleaned_wav = Path(cleaned_wav)
        ep_output_dir = output_dir / f"episode_{episode_idx:03d}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        ep_prefix = f"ep{episode_idx:03d}"

        # Step 1: UniSE TSE (always run when enabled)
        tse_v1_path = cleaned_wav
        if "unise_tse_v1" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_unise_tse_v1")
            if unise_dir is None or unise_ckpt_path is None:
                raise ValueError("unise_tse_v1 enabled but unise_dir or unise_ckpt_path missing")
            tse_v1_path = ep_output_dir / f"{ep_prefix}_tse_v1.wav"
            run_unise_tse(
                input_path=cleaned_wav,
                reference_path=reference_path,
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

        # Step 2: silence removal (always run when enabled)
        silence_out = tse_v1_path
        silence_map = None
        if "silence_removal" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_silence_removal")
            silence_out = ep_output_dir / f"{ep_prefix}_silence_removed.wav"
            map_path = ep_output_dir / f"{ep_prefix}_silence_map.json"
            _, silence_map = remove_silence(
                input_path=tse_v1_path,
                output_path=silence_out,
                map_path=map_path,
                sr=16000,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_silence_removal",
                outputs={
                    "silence_removed_path": str(silence_out),
                    "silence_map_path": str(map_path),
                },
            )

        # Early exit: if no ASR/SRT/rebuild components are enabled, return the
        # best audio we have so far (silence-removed or TSE output).
        if not any(c in components for c in ("aliyun_asr", "srt_cleaning", "rebuild_clips")):
            all_clips.append(silence_out)
            continue

        # Step 3: Build ASR input from original audio segments.
        # Use TSE output to detect non-silent regions (target speaker active),
        # then extract corresponding segments from original audio.
        # TSE preserves original timeline length, so timestamps match directly.
        asr_input_path = ep_output_dir / f"{ep_prefix}_asr_input.wav"
        asr_map = None
        if tse_v1_path != cleaned_wav:
            task_state.mark_started("stage2", step=f"{ep_prefix}_build_asr_input")
            asr_map = build_asr_input(
                original_audio_path=cleaned_wav,
                tse_audio_path=tse_v1_path,
                output_path=asr_input_path,
                gap_sec=asr_gap_sec,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_build_asr_input",
                outputs={"asr_input_path": str(asr_input_path)},
            )
        else:
            # No TSE: use original audio directly for ASR.
            asr_input_path = cleaned_wav

        # Step 4: Aliyun ASR with diarization
        if "aliyun_asr" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_aliyun_asr")
            srt_path = ep_output_dir / f"{ep_prefix}_diarization.srt"
            transcribe_with_diarization(
                audio_path=asr_input_path,
                output_srt_path=srt_path,
                speaker_count=asr_speaker_count,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_aliyun_asr",
                outputs={"srt_path": str(srt_path)},
            )
        else:
            srt_path = next(ep_output_dir.glob("*.srt"), None)
            if srt_path is None:
                raise FileNotFoundError(f"aliyun_asr disabled but no SRT found in {ep_output_dir}")

        # Step 5: identify target speaker via reference embedding similarity.
        target_speaker: Optional[str] = None
        if "speaker_verify" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_speaker_verify")
            scores_path = ep_output_dir / f"{ep_prefix}_speaker_scores.json"
            try:
                verify_result = identify_target_speaker(
                    srt_path=srt_path,
                    audio_path=asr_input_path,
                    reference_path=reference_path,
                    output_scores_path=scores_path,
                    embedding_model=speaker_embedding_model,
                    samples_per_speaker=speaker_verify_samples,
                )
                target_speaker = verify_result["target_speaker"]
                task_state.mark_done(
                    "stage2",
                    step=f"{ep_prefix}_speaker_verify",
                    outputs={
                        "target_speaker": target_speaker,
                        "speaker_scores_path": str(scores_path),
                        "centroid_scores": verify_result.get("centroid_scores", {}),
                    },
                )
            except Exception as exc:
                # Fall back to duration-based main-speaker heuristic in clean_srt.
                task_state.mark_done(
                    "stage2",
                    step=f"{ep_prefix}_speaker_verify",
                    outputs={
                        "target_speaker": None,
                        "error": str(exc),
                        "fallback": "duration_heuristic",
                    },
                )
                print(
                    f"[stage2] speaker_verify failed ({exc}); "
                    "falling back to longest-duration speaker."
                )

        # Step 6: SRT cleaning
        if "srt_cleaning" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_srt_cleaning")
            cleaned_srt_path = ep_output_dir / f"{ep_prefix}_cleaned.srt"
            clean_srt(
                input_srt_path=srt_path,
                output_srt_path=cleaned_srt_path,
                model=srt_model,
                use_llm=True,
                main_speaker=target_speaker,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_srt_cleaning",
                outputs={
                    "cleaned_srt_path": str(cleaned_srt_path),
                    "main_speaker": target_speaker,
                },
            )
        else:
            cleaned_srt_path = srt_path

        # Step 6: rebuild clips from cleaned SRT
        if "rebuild_clips" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_rebuild_clips")
            srt_entries = parse_srt(cleaned_srt_path.read_text(encoding="utf-8"))
            timestamps = [(e["begin_ms"] / 1000.0, e["end_ms"] / 1000.0) for e in srt_entries]
            clips_dir = ep_output_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            if asr_map is not None:
                # Map ASR timestamps back to original timeline using asr_map.
                original_timestamps = map_asr_to_original(timestamps, asr_map)
                clips = cut_by_timestamps(
                    cleaned_wav,
                    original_timestamps,
                    clips_dir,
                    prefix=f"{ep_prefix}_target_clip",
                    min_duration_sec=min_clip_duration_sec,
                    padding_sec=clip_padding_sec,
                )
            elif silence_map is not None:
                # Fallback: use silence_map for mapping (old behavior).
                clips = rebuild_from_srt(
                    original_audio_path=cleaned_wav,
                    srt_timestamps=timestamps,
                    silence_map=silence_map,
                    output_dir=clips_dir,
                    prefix=f"{ep_prefix}_target_clip",
                    min_duration_sec=min_clip_duration_sec,
                    padding_sec=clip_padding_sec,
                )
            else:
                # No mapping available: cut directly.
                clips = cut_by_timestamps(
                    cleaned_wav,
                    timestamps,
                    clips_dir,
                    prefix=f"{ep_prefix}_target_clip",
                    min_duration_sec=min_clip_duration_sec,
                    padding_sec=clip_padding_sec,
                )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_rebuild_clips",
                outputs={"clips_dir": str(clips_dir), "clips": [str(p) for p in clips]},
            )

            # Step 7: optional second UniSE TSE pass on the clips
            if run_unise_v2:
                task_state.mark_started("stage2", step=f"{ep_prefix}_unise_tse_v2")
                v2_dir = ep_output_dir / "clips_v2"
                v2_dir.mkdir(parents=True, exist_ok=True)
                v2_clips: List[Path] = []
                for idx, clip in enumerate(clips):
                    v2_out = v2_dir / f"{clip.stem}_v2.wav"
                    run_unise_tse(
                        input_path=clip,
                        reference_path=reference_path,
                        output_path=v2_out,
                        unise_dir=unise_dir,
                        ckpt_path=unise_ckpt_path,
                        segment_seconds=segment_seconds,
                    )
                    v2_clips.append(v2_out)
                clips = v2_clips
                task_state.mark_done(
                    "stage2",
                    step=f"{ep_prefix}_unise_tse_v2",
                    outputs={"v2_clips": [str(p) for p in v2_clips]},
                )

            # Step 8: optional merge all clips into a single output audio with gaps.
            if "merge_output" in components and clips:
                task_state.mark_started("stage2", step=f"{ep_prefix}_merge_output")
                merged_path = ep_output_dir / f"{ep_prefix}_merged_output.wav"
                merge_with_gaps(
                    clip_paths=clips,
                    output_path=merged_path,
                    gap_sec=merge_gap_sec,
                )
                task_state.mark_done(
                    "stage2",
                    step=f"{ep_prefix}_merge_output",
                    outputs={"merged_output_path": str(merged_path)},
                )

            all_clips.extend(clips)
        else:
            # If rebuild disabled, treat silence-removed audio as the output.
            all_clips.append(silence_out)

    final_outputs = [str(p) for p in all_clips]
    task_state.mark_done("stage2", outputs={"final_clips": final_outputs})
    return all_clips
