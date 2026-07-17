"""Generic audio mapping utilities for split-then-merge pipelines.

Provides forward splitting (record original positions) and reverse merging
(restore original timeline). Used by Stage 2 to map SRT timestamps back to
the original cleaned audio.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.audio_utils import cut_segment, get_audio_channels, get_audio_sample_rate, get_duration, merge_audio, split_audio


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
    min_duration_sec: float = 0.1,
    padding_sec: float = 0.0,
) -> List[Path]:
    """Cut an audio file into clips according to (start_sec, end_sec) pairs.

    Args:
        min_duration_sec: Skip clips shorter than this (default 0.1s).
        padding_sec: Extend each clip by this many seconds on both sides.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips: List[Path] = []
    for idx, (start, end) in enumerate(timestamps):
        # Apply padding
        padded_start = max(0.0, start - padding_sec)
        padded_end = end + padding_sec

        # Skip zero/negative duration or too-short clips
        if padded_end - padded_start < min_duration_sec:
            continue

        out = output_dir / f"{prefix}_{idx:04d}_{padded_start:.3f}_{padded_end:.3f}.wav"
        cut_segment(input_path, out, padded_start, padded_end)
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

    # forward_split preserves the compressed timeline and adds no gaps, so it
    # does not alter these coordinates. Keep the argument for API compatibility.
    _ = forward_map

    mapped: List[Tuple[float, float]] = []
    for start, end in srt_timestamps:
        if end <= start:
            continue

        for seg in silence_segments:
            out_start = seg["output_start_sec"]
            out_end = seg["output_end_sec"]
            overlap_start = max(start, out_start)
            overlap_end = min(end, out_end)
            if overlap_end <= overlap_start:
                continue

            original_start = seg["original_start_sec"] + (overlap_start - out_start)
            original_end = seg["original_start_sec"] + (overlap_end - out_start)
            mapped.append((original_start, original_end))

    return mapped


def _create_silence_wav(output_path: Path, duration_sec: float, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Create a silent WAV file using ffmpeg anullsrc."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels >= 2 else "mono"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl={layout}",
        "-t", str(duration_sec),
        "-acodec", "pcm_s16le",
        str(output_path),
    ], capture_output=True, check=True)
    return output_path


def build_asr_input(
    original_audio_path: str | Path,
    tse_audio_path: str | Path,
    output_path: str | Path,
    gap_sec: float = 2.0,
    sample_rate: Optional[int] = None,
    top_db: int = 40,
    min_silence_sec: float = 0.3,
) -> Dict[str, Any]:
    """Build ASR input by detecting non-silent regions in TSE output.

    UniSE TSE suppresses non-target speakers (they become near-silent).
    By detecting non-silent regions in the TSE output, we can identify
    time segments where the target speaker is active. Since TSE preserves
    the original timeline length, these timestamps can be used directly on
    the original audio to extract segments for ASR diarization.

    This avoids sending the entire audio to Aliyun ASR, saving cost and
    improving accuracy by focusing only on segments likely to contain
    the target speaker.

    Args:
        original_audio_path: Path to the original BGM-removed audio.
        tse_audio_path: Path to the UniSE TSE output (same length as original).
        output_path: Where to save the ASR input audio.
        gap_sec: Silence gap between segments (seconds).
        sample_rate: Sample rate for the output audio. If None, auto-detect
            from the original audio file.
        top_db: Threshold (in dB) for silence detection on TSE output.
        min_silence_sec: Minimum silence length; shorter gaps merge adjacent
            speech regions.

    Returns:
        asr_map dict with segments recording original and ASR input positions.
    """
    import librosa
    import numpy as np

    original_audio_path = Path(original_audio_path)
    tse_audio_path = Path(tse_audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect sample rate from original audio if not specified.
    if sample_rate is None:
        sample_rate = get_audio_sample_rate(original_audio_path)
        if sample_rate <= 0:
            sample_rate = 16000

    # Load TSE output and detect non-silent intervals.
    wav, sr = librosa.load(str(tse_audio_path), sr=16000, mono=True)
    intervals = librosa.effects.split(wav, top_db=top_db, ref=np.max)

    if len(intervals) == 0:
        # Entirely silent after TSE: nothing to process.
        # Create empty output.
        _create_silence_wav(output_path, 0.1, sample_rate, channels=1)
        return {
            "gap_sec": 0.0,
            "sample_rate": sample_rate,
            "channels": 1,
            "total_duration_sec": 0.1,
            "segments": [],
        }

    # Merge intervals separated by very short silence.
    min_gap_samples = int(min_silence_sec * sr)
    merged = [intervals[0].tolist()]
    for start_sample, end_sample in intervals[1:]:
        if start_sample - merged[-1][1] <= min_gap_samples:
            merged[-1][1] = end_sample
        else:
            merged.append([start_sample, end_sample])

    temp_dir = output_path.parent / ".asr_build_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create mono silence gap file (Aliyun diarization requires mono).
        silence_path = temp_dir / "silence.wav"
        _create_silence_wav(silence_path, gap_sec, sample_rate, channels=1)

        # Extract corresponding segments from original audio and convert to mono.
        concat_entries: List[Path] = []
        asr_segments: List[Dict[str, Any]] = []
        asr_cursor = 0.0

        for idx, (start_sample, end_sample) in enumerate(merged):
            start_sec = float(start_sample) / sr
            end_sec = float(end_sample) / sr
            duration = end_sec - start_sec

            # Cut segment from original audio, then convert to mono.
            seg_path_raw = temp_dir / f"seg_{idx:04d}_raw.wav"
            seg_path_mono = temp_dir / f"seg_{idx:04d}.wav"
            cut_segment(original_audio_path, seg_path_raw, start_sec, end_sec)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(seg_path_raw),
                "-vn", "-ac", "1",
                "-acodec", "pcm_s16le",
                str(seg_path_mono),
            ], capture_output=True, check=True)
            seg_path_raw.unlink(missing_ok=True)
            concat_entries.append(seg_path_mono)

            asr_segments.append({
                "index": idx,
                "original_start_sec": round(start_sec, 4),
                "original_end_sec": round(end_sec, 4),
                "asr_start_sec": round(asr_cursor, 4),
                "asr_end_sec": round(asr_cursor + duration, 4),
            })
            asr_cursor += duration

            # Add gap after each segment (except the last one).
            if idx < len(merged) - 1:
                concat_entries.append(silence_path)
                asr_cursor += gap_sec

        # Merge all segments and gaps (mono output).
        merge_audio(concat_entries, output_path, concat_with_copy=True)

        asr_map = {
            "gap_sec": gap_sec,
            "sample_rate": sample_rate,
            "channels": 1,
            "total_duration_sec": round(asr_cursor, 4),
            "segments": asr_segments,
        }

        # Save map for debugging.
        map_path = output_path.with_suffix(".asr_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(asr_map, f, indent=2, ensure_ascii=False)

        return asr_map
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def map_asr_to_original(
    asr_timestamps: List[Tuple[float, float]],
    asr_map: Dict[str, Any],
) -> List[Tuple[float, float]]:
    """Map ASR ranges back to the original timeline.

    A single ASR utterance may span several source segments because the ASR
    can treat the artificial silence gaps as an intra-sentence pause.  Such a
    range must be split at every map boundary.  Mapping only its two endpoints
    would create one continuous range on the original timeline and include all
    audio between otherwise non-contiguous source segments.

    Args:
        asr_timestamps: Timestamps relative to the ASR input audio.
        asr_map: Map produced by build_asr_input.

    Returns:
        Original ranges intersecting real ASR-input segments. Parts falling in
        artificial silence gaps are omitted. One input range can therefore
        produce zero, one, or multiple output ranges.
    """
    segments = asr_map.get("segments", [])
    if not segments:
        return asr_timestamps

    mapped: List[Tuple[float, float]] = []
    for start, end in asr_timestamps:
        if end <= start:
            continue

        for seg in segments:
            asr_start = seg["asr_start_sec"]
            asr_end = seg["asr_end_sec"]
            overlap_start = max(start, asr_start)
            overlap_end = min(end, asr_end)
            if overlap_end <= overlap_start:
                continue

            original_start = seg["original_start_sec"] + (overlap_start - asr_start)
            original_end = seg["original_start_sec"] + (overlap_end - asr_start)
            mapped.append((original_start, original_end))

    return mapped


def rebuild_from_srt(
    original_audio_path: str | Path,
    srt_timestamps: List[Tuple[float, float]],
    silence_map: Dict[str, Any],
    output_dir: str | Path,
    forward_map: Optional[Dict[str, Any]] = None,
    prefix: str = "target_clip",
    min_duration_sec: float = 0.1,
    padding_sec: float = 0.0,
) -> List[Path]:
    """Cut original audio into clips using SRT timestamps mapped back to original timeline.

    Args:
        min_duration_sec: Skip clips shorter than this.
        padding_sec: Extend each clip by this many seconds on both sides.
    """
    original_audio_path = Path(original_audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_timestamps = map_srt_to_original(srt_timestamps, silence_map, forward_map)
    return cut_by_timestamps(
        original_audio_path,
        original_timestamps,
        output_dir,
        prefix=prefix,
        min_duration_sec=min_duration_sec,
        padding_sec=padding_sec,
    )


# Forward reference helper
from typing import Any

__all__ = [
    "forward_split",
    "reverse_merge",
    "cut_by_timestamps",
    "map_srt_to_original",
    "rebuild_from_srt",
    "build_asr_input",
    "map_asr_to_original",
]
