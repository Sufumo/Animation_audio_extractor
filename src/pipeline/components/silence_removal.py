"""Silence removal with precise position mapping.

Removes silent regions from an audio file while keeping a JSON map that records
where each output segment originated in the original audio timeline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from src.audio_utils import cut_segment, get_duration, merge_audio


def detect_non_silent_regions(
    wav: np.ndarray,
    sr: int,
    top_db: int = 40,
    min_silence_sec: float = 0.3,
) -> np.ndarray:
    """Return non-silent intervals as a NumPy array of [[start, end], ...] in samples."""
    intervals = librosa.effects.split(wav, top_db=top_db, ref=np.max)
    if len(intervals) == 0:
        return np.array([[0, len(wav)]])

    # Merge intervals separated by very short silence to avoid over-fragmentation.
    min_gap_samples = int(min_silence_sec * sr)
    merged = [intervals[0].tolist()]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return np.array(merged)


def remove_silence(
    input_path: str | Path,
    output_path: str | Path,
    map_path: str | Path,
    sr: int = 16000,
    top_db: int = 40,
    min_silence_sec: float = 0.3,
) -> Tuple[Path, Dict[str, Any]]:
    """Remove silence and write a position map.

    Returns:
        (output_path, map_dict) where map_dict contains:
        - sample_rate: int
        - original_duration_sec: float
        - segments: list of {index, original_start_sec, original_end_sec,
          output_start_sec, output_end_sec, duration_sec}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    map_path = Path(map_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    wav, sr_loaded = librosa.load(str(input_path), sr=sr, mono=True)
    original_duration = float(len(wav)) / sr

    intervals = detect_non_silent_regions(wav, sr, top_db=top_db, min_silence_sec=min_silence_sec)

    if len(intervals) == 1:
        # Only one non-silent segment: extract it directly.
        start_sec = float(intervals[0][0]) / sr
        end_sec = float(intervals[0][1]) / sr
        cut_segment(input_path, output_path, start_sec, end_sec)
        segments = [{
            "index": 0,
            "original_start_sec": round(start_sec, 4),
            "original_end_sec": round(end_sec, 4),
            "output_start_sec": 0.0,
            "output_end_sec": round(end_sec - start_sec, 4),
            "duration_sec": round(end_sec - start_sec, 4),
        }]
    else:
        # Multiple segments: cut each and merge with copy.
        temp_dir = output_path.parent / ".silence_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cuts: List[Path] = []
        segments = []
        output_cursor = 0.0
        for idx, (start_sample, end_sample) in enumerate(intervals):
            start_sec = float(start_sample) / sr
            end_sec = float(end_sample) / sr
            duration = end_sec - start_sec
            cut_path = temp_dir / f"speech_{idx:04d}.wav"
            cut_segment(input_path, cut_path, start_sec, end_sec)
            cuts.append(cut_path)
            segments.append({
                "index": idx,
                "original_start_sec": round(start_sec, 4),
                "original_end_sec": round(end_sec, 4),
                "output_start_sec": round(output_cursor, 4),
                "output_end_sec": round(output_cursor + duration, 4),
                "duration_sec": round(duration, 4),
            })
            output_cursor += duration
        merge_audio(cuts, output_path, concat_with_copy=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    map_data = {
        "sample_rate": sr,
        "original_duration_sec": round(original_duration, 4),
        "output_duration_sec": round(output_cursor, 4),
        "segments": segments,
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False)

    return output_path, map_data


# Forward reference helper
from typing import Any

__all__ = ["detect_non_silent_regions", "remove_silence"]
