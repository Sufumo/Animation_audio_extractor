"""Stage 1: Data cleaning pipeline.

Orchestrates:
1. Convert source MP4/MP3 to WAV.
2. Remove OP/ED segments.
3. Remove background music (keep vocals).

Each step is optional and can be enabled/disabled via the components list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.task_state import TaskState
from src.pipeline.components.mp4_to_wav import convert_sources
from src.pipeline.components.oped_removal import remove_oped_segments
from src.pipeline.components.bgm_removal import remove_background_music


DEFAULT_COMPONENTS = ["mp4_to_wav", "oped_removal", "bgm_removal"]


def run_stage1(
    task_state: TaskState,
    source_dir: str | Path,
    output_dir: str | Path,
    oped_dir: Optional[str | Path] = None,
    reference_dir: Optional[str | Path] = None,
    components: Optional[List[str]] = None,
    melband_dir: Optional[str | Path] = None,
    melband_model_path: Optional[str | Path] = None,
    melband_config_path: Optional[str | Path] = None,
    bgm_segment_seconds: float = 360.0,
    keep_instrumental: bool = False,
    sample_rate: int = 44100,
    mono: bool = False,
    bit_depth: int = 16,
) -> List[Path]:
    """Run Stage 1 data cleaning and return list of cleaned episode WAVs.

    Args:
        task_state: TaskState instance for checkpointing.
        source_dir: Directory containing episode MP4/MP3/WAV files.
        output_dir: Directory where cleaned WAVs will be saved.
        oped_dir: Directory containing OP/ED audio files.
        reference_dir: Ignored in Stage 1; reserved for Stage 2.
        components: List of enabled components. Defaults to all.
        melband_dir: Path to Mel-Band-Roformer project root (required if
            bgm_removal is enabled).
        melband_model_path: Path to Mel-Band-Roformer .ckpt (required if
            bgm_removal is enabled).
        melband_config_path: Optional Mel-Band-Roformer config YAML.
        bgm_segment_seconds: Chunk length for BGM removal.
        keep_instrumental: Whether to keep instrumental outputs.
        sample_rate: Target sample rate for WAV conversion.
        mono: Whether to downmix to mono.
        bit_depth: 16 or 24.

    Returns:
        List of cleaned episode WAV paths.
    """
    components = components or DEFAULT_COMPONENTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_state.set_inputs({
        "source_dir": str(source_dir),
        "oped_dir": str(oped_dir) if oped_dir else None,
        "reference_dir": str(reference_dir) if reference_dir else None,
    })

    # Track Stage 1 as a whole.
    task_state.mark_started("stage1")

    # Step 1: Convert sources to WAV
    if "mp4_to_wav" in components:
        task_state.mark_started("stage1", step="mp4_to_wav")
        wav_dir = output_dir / "01_wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_files = convert_sources(
            source_dir,
            wav_dir,
            sample_rate=sample_rate,
            mono=mono,
            bit_depth=bit_depth,
        )
        task_state.mark_done(
            "stage1",
            step="mp4_to_wav",
            outputs={"wav_files": [str(p) for p in wav_files]},
        )
    else:
        wav_dir = output_dir / "01_wav"
        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError("mp4_to_wav disabled but no pre-existing WAV files found")

    # Step 2: Remove OP/ED
    if "oped_removal" in components:
        task_state.mark_started("stage1", step="oped_removal")
        if oped_dir is None:
            raise ValueError("oped_removal enabled but oped_dir not provided")
        oped_out_dir = output_dir / "02_no_oped"
        oped_out_dir.mkdir(parents=True, exist_ok=True)
        oped_removed_files: List[Path] = []
        removed_segments_map: Dict[str, List[Dict]] = {}
        for wav in wav_files:
            out = oped_out_dir / f"{wav.stem}_no_oped.wav"
            _, removed = remove_oped_segments(wav, oped_dir, out)
            oped_removed_files.append(out)
            removed_segments_map[wav.name] = removed
        task_state.mark_done(
            "stage1",
            step="oped_removal",
            outputs={
                "oped_removed_files": [str(p) for p in oped_removed_files],
                "removed_segments": removed_segments_map,
            },
        )
    else:
        oped_removed_files = wav_files

    # Step 3: Remove background music
    if "bgm_removal" in components:
        task_state.mark_started("stage1", step="bgm_removal")
        if melband_dir is None or melband_model_path is None:
            raise ValueError("bgm_removal enabled but melband_dir or melband_model_path not provided")
        bgm_out_dir = output_dir / "03_cleaned"
        bgm_out_dir.mkdir(parents=True, exist_ok=True)
        cleaned_files: List[Path] = []
        for wav in oped_removed_files:
            vocals_path = bgm_out_dir / f"{wav.stem}_cleaned.wav"
            remove_background_music(
                input_path=wav,
                output_path=vocals_path,
                melband_dir=melband_dir,
                model_path=melband_model_path,
                config_path=melband_config_path,
                segment_seconds=bgm_segment_seconds,
                keep_instrumental=keep_instrumental,
            )
            cleaned_files.append(vocals_path)
        task_state.mark_done(
            "stage1",
            step="bgm_removal",
            outputs={"cleaned_files": [str(p) for p in cleaned_files]},
        )
    else:
        cleaned_files = oped_removed_files

    final_outputs = [str(p) for p in cleaned_files]
    task_state.mark_done("stage1", outputs={"cleaned_wavs": final_outputs})
    return cleaned_files
