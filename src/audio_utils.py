"""High-quality audio utilities built on ffmpeg.

Design goals:
- Avoid unnecessary re-encoding / resampling.
- Use PCM WAV (s16le / s24le) for intermediate files.
- Use ffmpeg concat demuxer with stream copy when merging segments that share
  the same codec, sample rate, and channel layout.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _run_ffmpeg(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with given arguments."""
    cmd = ["ffmpeg", "-y"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _run_ffprobe(path: str | Path) -> Dict[str, Any]:
    """Run ffprobe and return JSON stream info."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_media_info(path: str | Path) -> Dict[str, Any]:
    """Return ffprobe JSON info for a media file."""
    return _run_ffprobe(path)


def get_duration(path: str | Path) -> float:
    """Return media duration in seconds."""
    info = _run_ffprobe(path)
    # Try format duration first, then stream duration.
    duration = info.get("format", {}).get("duration")
    if duration is None:
        for stream in info.get("streams", []):
            duration = stream.get("duration")
            if duration is not None:
                break
    if duration is None:
        raise RuntimeError(f"Could not determine duration for {path}")
    return float(duration)


def get_audio_sample_rate(path: str | Path) -> int:
    """Return audio sample rate, or 0 if no audio stream found."""
    info = _run_ffprobe(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return int(stream.get("sample_rate", 0))
    return 0


def convert_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: Optional[int] = None,
    mono: bool = False,
    bit_depth: int = 16,
) -> Path:
    """Convert any media file to WAV.

    If the input is already a WAV with matching parameters, this function will
    attempt to copy the audio stream (no re-encode). Otherwise it re-encodes to
    PCM using the requested bit depth.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acodec = "pcm_s16le" if bit_depth == 16 else "pcm_s24le"

    # If input is wav and no transformation needed, copy audio stream.
    if input_path.suffix.lower() == ".wav" and sample_rate is None and not mono:
        info = _run_ffprobe(input_path)
        audio_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None
        )
        if audio_stream and audio_stream.get("codec_name") in ("pcm_s16le", "pcm_s24le"):
            _run_ffmpeg([
                "-i", str(input_path),
                "-vn", "-acodec", "copy",
                str(output_path),
            ])
            return output_path

    args = ["-i", str(input_path), "-vn"]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    if mono:
        args += ["-ac", "1"]
    args += ["-acodec", acodec, str(output_path)]
    _run_ffmpeg(args)
    return output_path


def split_audio(
    input_path: str | Path,
    output_dir: str | Path,
    segment_seconds: float,
    suffix_fmt: str = "%03d",
    prefix: str = "segment",
    sample_rate: Optional[int] = None,
    mono: bool = False,
    bit_depth: int = 16,
) -> List[Path]:
    """Split audio into fixed-length segments using ffmpeg segment muxer.

    Returns the list of generated segment paths in order.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acodec = "pcm_s16le" if bit_depth == 16 else "pcm_s24le"
    output_pattern = output_dir / f"{prefix}_{suffix_fmt}.wav"

    args = [
        "-i", str(input_path),
        "-vn",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
    ]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    if mono:
        args += ["-ac", "1"]
    args += ["-acodec", acodec, str(output_pattern)]

    _run_ffmpeg(args)

    segments = sorted(output_dir.glob(f"{prefix}_*.wav"))
    return segments


def merge_audio(
    segment_paths: List[str | Path],
    output_path: str | Path,
    concat_with_copy: bool = True,
) -> Path:
    """Merge multiple audio files into one.

    When `concat_with_copy` is True, ffmpeg concat demuxer with stream copy is
    used (fastest and lossless), requiring all segments share codec/sample
    rate/channels. If False, segments are re-encoded into a single PCM WAV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_paths = [Path(p) for p in segment_paths]
    if not segment_paths:
        raise ValueError("segment_paths is empty")

    if concat_with_copy:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for seg in segment_paths:
                f.write(f"file '{seg.resolve()}'\n")
            concat_list = f.name
        try:
            _run_ffmpeg([
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(output_path),
            ])
        finally:
            os.unlink(concat_list)
    else:
        inputs: List[str] = []
        for seg in segment_paths:
            inputs += ["-i", str(seg)]
        filter_complex = "".join(f"[{i}:a:0]" for i in range(len(segment_paths)))
        filter_complex += f"concat=n={len(segment_paths)}:v=0:a=1[outa]"
        _run_ffmpeg(
            inputs
            + [
                "-filter_complex", filter_complex,
                "-map", "[outa]",
                "-acodec", "pcm_s16le",
                str(output_path),
            ]
        )
    return output_path


def resample(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int,
    bit_depth: int = 16,
) -> Path:
    """Resample audio to a target sample rate using high-quality soxr."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    acodec = "pcm_s16le" if bit_depth == 16 else "pcm_s24le"
    _run_ffmpeg([
        "-i", str(input_path),
        "-vn",
        "-ar", str(target_sr),
        "-acodec", acodec,
        "-af", "aresample=resampler=soxr:precision=28",
        str(output_path),
    ])
    return output_path


def cut_segment(
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
    copy: bool = False,
) -> Path:
    """Extract a [start, end) segment from an audio file (seconds).

    When `copy` is True, stream copy is used (requires segment boundaries to
    align with codec frames, may be slightly imprecise). When False, the output
    is re-encoded precisely.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-i", str(input_path),
        "-vn",
        "-ss", str(start),
        "-to", str(end),
    ]
    if copy:
        args += ["-c", "copy"]
    else:
        args += ["-acodec", "pcm_s16le"]
    args += [str(output_path)]
    _run_ffmpeg(args)
    return output_path


def apply_gain(
    input_path: str | Path,
    output_path: str | Path,
    gain_db: float,
) -> Path:
    """Apply volume gain in dB. Output is PCM WAV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "-i", str(input_path),
        "-vn",
        "-af", f"volume={gain_db}dB",
        "-acodec", "pcm_s16le",
        str(output_path),
    ])
    return output_path


def validate_pcm_wav(path: str | Path) -> Tuple[bool, Optional[str]]:
    """Check whether file is a PCM WAV. Returns (ok, reason_or_none)."""
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    try:
        info = _run_ffprobe(path)
    except Exception as e:
        return False, f"ffprobe failed: {e}"
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if audio_stream is None:
        return False, "no audio stream"
    if audio_stream.get("codec_name") not in ("pcm_s16le", "pcm_s24le"):
        return False, f"not PCM: {audio_stream.get('codec_name')}"
    return True, None


# Small hack to avoid forward-reference annotation issues in older Python.
from typing import Any

__all__ = [
    "get_media_info",
    "get_duration",
    "get_audio_sample_rate",
    "convert_to_wav",
    "split_audio",
    "merge_audio",
    "resample",
    "cut_segment",
    "apply_gain",
    "validate_pcm_wav",
]
