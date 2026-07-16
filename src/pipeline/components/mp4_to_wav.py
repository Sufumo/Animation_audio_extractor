"""Convert animation source files (MP4/MP3/WAV) to uniform WAV files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.audio_utils import convert_to_wav


SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mp3", ".wav", ".flac", ".m4a", ".aac"}


def collect_source_files(source_dir: str | Path) -> List[Path]:
    """Collect all supported media files in source_dir (sorted by name)."""
    source_dir = Path(source_dir)
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def convert_sources(
    source_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int = 44100,
    mono: bool = False,
    bit_depth: int = 16,
) -> List[Path]:
    """Convert all supported source files to WAV.

    Output filenames preserve the original basename but with `.wav` extension.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_source_files(source_dir)
    outputs: List[Path] = []
    for src in files:
        out = output_dir / f"{src.stem}.wav"
        convert_to_wav(src, out, sample_rate=sample_rate, mono=mono, bit_depth=bit_depth)
        outputs.append(out)
    return outputs
