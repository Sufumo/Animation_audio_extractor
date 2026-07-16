"""Stage 2: Target speaker extraction pipeline.

Orchestrates:
1. UniSE TSE (initial pass).
2. Silence removal with position mapping.
3. Aliyun ASR speaker diarization.
4. Qwen SRT cleaning.
5. Reverse-map timestamps to original audio and cut target clips.
6. Optional second UniSE TSE pass on the clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.task_state import TaskState
from src.pipeline.components.unise_tse import run_unise_tse
from src.pipeline.components.silence_removal import remove_silence
from src.pipeline.components.aliyun_asr import transcribe_with_diarization
from src.pipeline.components.srt_cleaning import clean_srt, parse_srt
from src.pipeline.components.audio_mapping import rebuild_from_srt


DEFAULT_COMPONENTS = [
    "unise_tse_v1",
    "silence_removal",
    "aliyun_asr",
    "srt_cleaning",
    "rebuild_clips",
]


def run_stage2(
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
) -> List[Path]:
    """Run Stage 2 speaker extraction.

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

        # Step 1: initial UniSE TSE
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
        else:
            tse_v1_path = cleaned_wav

        # Step 2: silence removal
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
        else:
            silence_out = tse_v1_path
            silence_map = None

        # Step 3: Aliyun ASR with diarization
        if "aliyun_asr" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_aliyun_asr")
            srt_path = ep_output_dir / f"{ep_prefix}_diarization.srt"
            transcribe_with_diarization(
                audio_path=silence_out,
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

        # Step 4: SRT cleaning
        if "srt_cleaning" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_srt_cleaning")
            cleaned_srt_path = ep_output_dir / f"{ep_prefix}_cleaned.srt"
            clean_srt(
                input_srt_path=srt_path,
                output_srt_path=cleaned_srt_path,
                model=srt_model,
                use_llm=True,
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_srt_cleaning",
                outputs={"cleaned_srt_path": str(cleaned_srt_path)},
            )
        else:
            cleaned_srt_path = srt_path

        # Step 5: rebuild clips from cleaned SRT
        if "rebuild_clips" in components:
            task_state.mark_started("stage2", step=f"{ep_prefix}_rebuild_clips")
            if silence_map is None:
                raise ValueError("rebuild_clips requires silence_map from silence_removal")
            srt_entries = parse_srt(cleaned_srt_path.read_text(encoding="utf-8"))
            timestamps = [(e["begin_ms"] / 1000.0, e["end_ms"] / 1000.0) for e in srt_entries]
            clips_dir = ep_output_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            clips = rebuild_from_srt(
                original_audio_path=cleaned_wav,
                srt_timestamps=timestamps,
                silence_map=silence_map,
                output_dir=clips_dir,
                prefix=f"{ep_prefix}_target_clip",
            )
            task_state.mark_done(
                "stage2",
                step=f"{ep_prefix}_rebuild_clips",
                outputs={"clips_dir": str(clips_dir), "clips": [str(p) for p in clips]},
            )

            # Step 6: optional second UniSE pass
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

            all_clips.extend(clips)
        else:
            # If rebuild disabled, treat silence-removed audio as the output.
            all_clips.append(silence_out)

    final_outputs = [str(p) for p in all_clips]
    task_state.mark_done("stage2", outputs={"final_clips": final_outputs})
    return all_clips
