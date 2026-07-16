"""Generic audio mapping utilities for split-then-merge pipelines.

Provides forward splitting (record original positions) and reverse merging
(restore original timeline). Used by Stage 2 to map SRT timestamps back to
the original cleaned audio.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.audio_utils import cut_segment, get_duration, merge_audio, split_audio


def forward_split(
    input_path: str | Path,
    output_dir: str | Path,
    segment_seconds: float,
    prefix: str = "seg",
    map_filename: str = "forward_map.json",
    sample_rate: Optional[int] = None,
    mono: Optional[bool] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Split audio into fixed-length segments and record original positions.

    Returns:
        (segment_paths, map_dict) where map_dict contains the original start/end
        time of each segment.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    if mono is not None:
        kwargs["mono"] = mono

    segments = split_audio(
        input_path,
        output_dir,
        segment_seconds=segment_seconds,
        prefix=prefix,
        **kwargs,
    )

    map_entries = []
    cursor = 0.0
    for idx, seg in enumerate(segments):
        duration = get_duration(seg)
        map_entries.append({
            "index": idx,
            "filename": seg.name,
            "original_start_sec": round(cursor, 4),
            "original_end_sec": round(cursor + duration, 4),
            "duration_sec": round(duration, 4),
        })
        cursor += duration

    map_data = {
        "input_file": str(input_path),
        "segment_seconds": segment_seconds,
        "segments": map_entries,
    }
    map_path = output_dir / map_filename
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False)

    return segments, map_data


def reverse_merge(
    map_data: Dict[str, Any],
    segment_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Merge segments back into a single timeline using a forward map.

    This is mainly for verification; actual Stage 2 uses the timestamps from
    the cleaned SRT to cut the original audio directly.
    """
    segment_dir = Path(segment_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = [segment_dir / seg["filename"] for seg in map_data["segments"]]
    merge_audio(segments, output_path, concat_with_copy=True)
    return output_path


def cut_by_timestamps(
    input_path: str | Path,
    timestamps: List[Tuple[float, float]],
    output_dir: str | Path,
    prefix: str = "clip",
) -> List[Path]:
    """Cut an audio file into clips according to (start_sec, end_sec) pairs."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips: List[Path] = []
    for idx, (start, end) in enumerate(timestamps):
        out = output_dir / f"{prefix}_{idx:04d}_{start:.3f}_{end:.3f}.wav"
        cut_segment(input_path, out, start, end)
        clips.append(out)
    return clips


def map_srt_to_original(
    srt_timestamps: List[Tuple[float, float]],
    silence_map: Dict[str, Any],
    forward_map: Optional[Dict[str, Any]] = None,
) -> List[Tuple[float, float]]:
    """Map timestamps from the silence-removed audio back to the original audio.

    Args:
        srt_timestamps: Timestamps relative to the silence-removed audio.
        silence_map: Map produced by silence_removal.remove_silence.
        forward_map: Optional forward split map if the silence-removed audio was
            further split before ASR.

    Returns:
        List of (original_start_sec, original_end_sec) timestamps.
    """
    silence_segments = silence_map.get("segments", [])
    if not silence_segments:
        return srt_timestamps

    def _map_one(t: float) -> float:
        # Find the silence-removed segment that contains t.
        for seg in silence_segments:
            out_start = seg["output_start_sec"]
            out_end = seg["output_end_sec"]
            if out_start <= t <= out_end:
                offset_in_seg = t - out_start
                return seg["original_start_sec"] + offset_in_seg
        # Fallback: extrapolate from last segment.
        last = silence_segments[-1]
        return last["original_start_sec"] + (t - last["output_start_sec"])

    if forward_map is not None:
        # First map from ASR-split audio back to silence-removed audio, then to original.
        forward_segments = forward_map.get("segments", [])

        def _map_through_forward(t: float) -> float:
            for seg in forward_segments:
                if seg["original_start_sec"] <= t <= seg["original_end_sec"]:
                    return _map_one(seg["original_start_sec"] + (t - seg["original_start_sec"]))
            return _map_one(t)

        mapper = _map_through_forward
    else:
        mapper = _map_one

    return [(mapper(start), mapper(end)) for start, end in srt_timestamps]


def rebuild_from_srt(
    original_audio_path: str | Path,
    srt_timestamps: List[Tuple[float, float]],
    silence_map: Dict[str, Any],
    output_dir: str | Path,
    forward_map: Optional[Dict[str, Any]] = None,
    prefix: str = "target_clip",
) -> List[Path]:
    """Cut original audio into clips using SRT timestamps mapped back to original timeline."""
    original_audio_path = Path(original_audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_timestamps = map_srt_to_original(srt_timestamps, silence_map, forward_map)
    return cut_by_timestamps(original_audio_path, original_timestamps, output_dir, prefix=prefix)


# Forward reference helper
from typing import Any

__all__ = [
    "forward_split",
    "reverse_merge",
    "cut_by_timestamps",
    "map_srt_to_original",
    "rebuild_from_srt",
]
