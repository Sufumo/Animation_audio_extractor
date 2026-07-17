"""Voice activity detection for target-speaker active segments.

Primary: Silero VAD (speech/non-speech classifier).
Fallback: energy-based detection with absolute dBFS threshold (avoids
librosa.effects.split's ref=np.max pitfall).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SAMPLE_RATE = 16000


def _merge_intervals(
    intervals: List[Tuple[float, float]],
    min_silence_sec: float,
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= min_silence_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(float(s), float(e)) for s, e in merged]


def detect_with_silero(
    wav: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    min_speech_sec: float = 0.25,
    min_silence_sec: float = 0.3,
) -> List[Tuple[float, float]]:
    """Run Silero VAD and return (start_sec, end_sec) speech intervals."""
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = torch.from_numpy(wav.astype(np.float32))
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_sec * 1000),
        min_silence_duration_ms=int(min_silence_sec * 1000),
    )
    intervals = [
        (t["start"] / sr, t["end"] / sr)
        for t in timestamps
    ]
    return _merge_intervals(intervals, min_silence_sec)


def detect_with_energy(
    wav: np.ndarray,
    sr: int,
    top_db: float = 35.0,
    min_silence_sec: float = 0.3,
    abs_dbfs_floor: float = -45.0,
) -> List[Tuple[float, float]]:
    """Energy VAD using a fixed absolute floor instead of ref=np.max.

    A frame is speech if its RMS is above max(abs_floor, peak - top_db).
    This prevents a single shout from raising the whole-file threshold.
    """
    import librosa

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)

    peak = float(np.max(db)) if len(db) else -80.0
    threshold = max(abs_dbfs_floor, peak - top_db)
    speech_mask = db >= threshold

    intervals: List[Tuple[float, float]] = []
    in_speech = False
    start_frame = 0
    for i, flag in enumerate(speech_mask):
        if flag and not in_speech:
            in_speech = True
            start_frame = i
        elif not flag and in_speech:
            in_speech = False
            start = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            if end > start:
                intervals.append((start, end))
    if in_speech:
        start = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end = librosa.frames_to_time(len(speech_mask), sr=sr, hop_length=hop_length)
        if end > start:
            intervals.append((start, end))

    return _merge_intervals(intervals, min_silence_sec)


def detect_active_segments(
    audio_path: str | Path,
    method: str = "auto",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_silence_sec: float = 0.3,
    min_speech_sec: float = 0.25,
    top_db: float = 35.0,
    abs_dbfs_floor: float = -45.0,
    silero_threshold: float = 0.5,
    map_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Detect speech-active intervals on an audio file (typically TSE output).

    Args:
        method: "silero", "energy", or "auto" (try silero, fall back to energy).

    Returns:
        Map dict with segments [{index, start_sec, end_sec, duration_sec}, ...].
    """
    import librosa

    audio_path = Path(audio_path)
    wav, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    used_method = method
    intervals: List[Tuple[float, float]] = []

    if method in ("silero", "auto"):
        try:
            intervals = detect_with_silero(
                wav, sr,
                threshold=silero_threshold,
                min_speech_sec=min_speech_sec,
                min_silence_sec=min_silence_sec,
            )
            used_method = "silero"
        except Exception as e:
            if method == "silero":
                raise
            print(f"[vad] Silero unavailable ({e}); falling back to energy VAD.")
            used_method = "energy"

    if used_method == "energy" or (method == "energy"):
        intervals = detect_with_energy(
            wav, sr,
            top_db=top_db,
            min_silence_sec=min_silence_sec,
            abs_dbfs_floor=abs_dbfs_floor,
        )
        used_method = "energy"

    # Drop intervals shorter than min_speech_sec.
    intervals = [(s, e) for s, e in intervals if (e - s) >= min_speech_sec]

    segments = []
    for idx, (start, end) in enumerate(intervals):
        segments.append({
            "index": idx,
            "start_sec": round(start, 4),
            "end_sec": round(end, 4),
            "duration_sec": round(end - start, 4),
        })

    result = {
        "method": used_method,
        "sample_rate": sr,
        "audio_path": str(audio_path),
        "original_duration_sec": round(float(len(wav)) / sr, 4),
        "num_segments": len(segments),
        "segments": segments,
    }

    if map_path is not None:
        map_path = Path(map_path)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


__all__ = [
    "detect_active_segments",
    "detect_with_silero",
    "detect_with_energy",
]
