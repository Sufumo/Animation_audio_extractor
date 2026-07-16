"""Background music removal using Mel-Band-Roformer Vocal Model.

This component wraps the external Mel-Band-Roformer project. Long audio is
split into chunks (default 360s) to avoid out-of-memory issues, then vocals are
extracted per chunk and merged back losslessly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.audio_utils import (
    convert_to_wav,
    cut_segment,
    get_duration,
    merge_audio,
    split_audio,
)


DEFAULT_SEGMENT_SECONDS = 360.0


def _call_melband_roformer(
    melband_dir: Path,
    config_path: Path,
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    device_ids: Optional[List[int]] = None,
) -> None:
    """Run Mel-Band-Roformer inference.py on all WAV files in input_dir."""
    inference_script = melband_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Mel-Band-Roformer inference.py not found: {inference_script}")

    cmd = [
        sys.executable,
        str(inference_script),
        "--config_path", str(config_path),
        "--model_path", str(model_path),
        "--input_folder", str(input_dir),
        "--store_dir", str(output_dir),
    ]
    if device_ids is not None:
        cmd += ["--device_ids"] + [str(d) for d in device_ids]

    subprocess.run(cmd, cwd=str(melband_dir), check=True)


def remove_background_music(
    input_path: str | Path,
    output_path: str | Path,
    melband_dir: str | Path,
    model_path: str | Path,
    config_path: Optional[str | Path] = None,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    keep_instrumental: bool = False,
    device_ids: Optional[List[int]] = None,
) -> Dict[str, Path]:
    """Remove background music from an audio file, keeping vocals.

    Args:
        input_path: Input audio/video file.
        output_path: Where to save the vocals WAV.
        melband_dir: Path to Mel-Band-Roformer-Vocal-Model project root.
        model_path: Path to the .ckpt model file.
        config_path: Optional config YAML. Defaults to
            `<melband_dir>/configs/config_vocals_mel_band_roformer.yaml`.
        segment_seconds: Split long audio into chunks of this length (seconds).
        keep_instrumental: If True, also return the instrumental file path.
        device_ids: GPU device IDs; None uses default behavior of inference.py.

    Returns:
        Dictionary with at least key `"vocals"` and optionally `"instrumental"`.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    melband_dir = Path(melband_dir)
    model_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path is None:
        config_path = melband_dir / "configs" / "config_vocals_mel_band_roformer.yaml"
    config_path = Path(config_path)

    duration = get_duration(input_path)

    with tempfile.TemporaryDirectory(prefix="bgm_removal_") as tmp:
        tmp_dir = Path(tmp)
        input_wav = tmp_dir / "input.wav"
        convert_to_wav(input_path, input_wav, sample_rate=44100, mono=False, bit_depth=16)

        if duration <= segment_seconds:
            chunks = [input_wav]
            chunk_dir = tmp_dir / "chunks"
            chunk_dir.mkdir()
            chunk_input = chunk_dir / "input_000.wav"
            shutil.copy(str(input_wav), str(chunk_input))
        else:
            chunk_dir = tmp_dir / "chunks"
            chunks = split_audio(
                input_wav,
                chunk_dir,
                segment_seconds=segment_seconds,
                prefix="input",
                sample_rate=44100,
                mono=False,
                bit_depth=16,
            )

        roformer_input = tmp_dir / "roformer_input"
        roformer_input.mkdir()
        for chunk in chunks:
            shutil.copy(str(chunk), str(roformer_input / chunk.name))

        roformer_output = tmp_dir / "roformer_output"
        roformer_output.mkdir()

        _call_melband_roformer(
            melband_dir=melband_dir,
            config_path=config_path,
            model_path=model_path,
            input_dir=roformer_input,
            output_dir=roformer_output,
            device_ids=device_ids,
        )

        vocal_segments = sorted(roformer_output.glob("*_vocals.wav"))
        if not vocal_segments:
            raise RuntimeError("Mel-Band-Roformer did not produce any *_vocals.wav files")

        # Rename to consistent order and merge losslessly.
        merge_audio(vocal_segments, output_path, concat_with_copy=True)

        result: Dict[str, Path] = {"vocals": output_path}

        if keep_instrumental:
            instrumental_segments = sorted(roformer_output.glob("*_instrumental.wav"))
            if instrumental_segments:
                instrumental_path = output_path.parent / f"{output_path.stem}_instrumental.wav"
                merge_audio(instrumental_segments, instrumental_path, concat_with_copy=True)
                result["instrumental"] = instrumental_path

    return result
