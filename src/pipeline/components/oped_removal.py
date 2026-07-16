"""Detect and remove opening/ending (OP/ED) segments from episode audio.

This module builds a small fingerprint library from all audio files in the
`oped_dir` and then searches for them at the beginning and end of each episode.
Detection is based on normalized cross-correlation of log-Mel spectrograms.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import correlate

from src.audio_utils import convert_to_wav, cut_segment, get_duration


SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4", ".mkv"}


def _load_for_fingerprint(path: str | Path, sr: int = 22050, mono: bool = True) -> np.ndarray:
    """Load audio into a standardized format for fingerprinting."""
    wav, _ = librosa.load(str(path), sr=sr, mono=mono)
    return wav


def _compute_fingerprint(
    wav: np.ndarray,
    sr: int = 22050,
    n_mels: int = 64,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a compact log-Mel spectrogram fingerprint."""
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize along time axis."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


def _norm_correlation(template: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute normalized cross-correlation of template along time axis of target."""
    t = _zscore(template)
    x = _zscore(target)
    # correlate along time axis (axis=1)
    corr = correlate(x, t, mode="valid", method="direct")
    # Normalize by template length
    n = template.shape[1]
    corr = corr / n
    return corr


def detect_oped_position(
    episode_path: str | Path,
    oped_path: str | Path,
    search_head_seconds: float = 240.0,
    search_tail_seconds: float = 240.0,
    sr: int = 22050,
    similarity_threshold: float = 0.55,
) -> Optional[Tuple[float, float]]:
    """Return (start_sec, end_sec) if the OP/ED is found in episode, else None.

    Only searches the first `search_head_seconds` and last `search_tail_seconds`
    of the episode to reduce false positives and computation.
    """
    episode_path = Path(episode_path)
    oped_path = Path(oped_path)

    episode_wav = _load_for_fingerprint(episode_path, sr=sr)
    oped_wav = _load_for_fingerprint(oped_path, sr=sr)

    if len(oped_wav) == 0 or len(episode_wav) == 0:
        return None

    ep_fp = _compute_fingerprint(episode_wav, sr=sr)
    op_fp = _compute_fingerprint(oped_wav, sr=sr)

    op_len = op_fp.shape[1]
    if op_len == 0:
        return None

    head_samples = min(int(search_head_seconds * sr), len(episode_wav))
    tail_samples = min(int(search_tail_seconds * sr), len(episode_wav))

    hop_length = 512
    head_frames = librosa.samples_to_frames(head_samples, hop_length=hop_length)
    tail_frames = librosa.samples_to_frames(tail_samples, hop_length=hop_length)

    candidates: List[Tuple[float, float, float]] = []

    # Search head
    if head_frames > op_len:
        head_fp = ep_fp[:, :head_frames]
        corr = _norm_correlation(op_fp, head_fp)
        peak = np.max(corr)
        peak_idx = int(np.argmax(corr))
        if peak >= similarity_threshold:
            start = librosa.frames_to_time(peak_idx, sr=sr, hop_length=hop_length)
            end = librosa.frames_to_time(peak_idx + op_len, sr=sr, hop_length=hop_length)
            candidates.append((start, end, float(peak)))

    # Search tail
    if tail_frames > op_len:
        tail_fp = ep_fp[:, -tail_frames:]
        corr = _norm_correlation(op_fp, tail_fp)
        peak = np.max(corr)
        peak_idx = int(np.argmax(corr))
        if peak >= similarity_threshold:
            offset_frames = ep_fp.shape[1] - tail_frames
            start = librosa.frames_to_time(offset_frames + peak_idx, sr=sr, hop_length=hop_length)
            end = librosa.frames_to_time(offset_frames + peak_idx + op_len, sr=sr, hop_length=hop_length)
            candidates.append((start, end, float(peak)))

    if not candidates:
        return None

    # Pick highest similarity candidate
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def collect_oped_files(oped_dir: str | Path) -> List[Path]:
    """Collect all OP/ED media files in oped_dir."""
    oped_dir = Path(oped_dir)
    if not oped_dir.exists():
        return []
    files = [p for p in oped_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def remove_oped_segments(
    episode_path: str | Path,
    oped_dir: str | Path,
    output_path: str | Path,
    search_head_seconds: float = 240.0,
    search_tail_seconds: float = 240.0,
    similarity_threshold: float = 0.55,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Remove all detected OP/ED segments from an episode and save result.

    Returns the output path and a list of removed segment metadata.
    """
    episode_path = Path(episode_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oped_files = collect_oped_files(oped_dir)
    if not oped_files:
        # Nothing to remove; copy as-is to maintain pipeline contract.
        shutil.copy(str(episode_path), str(output_path))
        return output_path, []

    detections: List[Dict[str, Any]] = []
    for oped in oped_files:
        pos = detect_oped_position(
            episode_path,
            oped,
            search_head_seconds=search_head_seconds,
            search_tail_seconds=search_tail_seconds,
            similarity_threshold=similarity_threshold,
        )
        if pos is not None:
            detections.append({
                "oped_file": str(oped),
                "start_sec": round(pos[0], 3),
                "end_sec": round(pos[1], 3),
                "duration_sec": round(pos[1] - pos[0], 3),
            })

    if not detections:
        shutil.copy(str(episode_path), str(output_path))
        return output_path, []

    # Sort detections and merge overlapping ones.
    detections.sort(key=lambda x: x["start_sec"])
    merged = [detections[0]]
    for d in detections[1:]:
        last = merged[-1]
        if d["start_sec"] <= last["end_sec"]:
            last["end_sec"] = max(last["end_sec"], d["end_sec"])
            last["duration_sec"] = round(last["end_sec"] - last["start_sec"], 3)
        else:
            merged.append(d)

    # Cut out merged segments by keeping the complement.
    total_duration = get_duration(episode_path)
    keep_segments: List[Tuple[float, float]] = []
    prev_end = 0.0
    for d in merged:
        if d["start_sec"] > prev_end:
            keep_segments.append((prev_end, d["start_sec"]))
        prev_end = max(prev_end, d["end_sec"])
    if prev_end < total_duration:
        keep_segments.append((prev_end, total_duration))

    if len(keep_segments) == 1:
        # Single keep segment: just cut it.
        cut_segment(episode_path, output_path, keep_segments[0][0], keep_segments[0][1])
    else:
        # Multiple keep segments: cut each and merge losslessly.
        temp_dir = output_path.parent / ".oped_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cuts: List[Path] = []
        for idx, (start, end) in enumerate(keep_segments):
            cut_path = temp_dir / f"keep_{idx:03d}.wav"
            cut_segment(episode_path, cut_path, start, end)
            cuts.append(cut_path)
        # Import here to avoid circular imports at module load.
        from src.audio_utils import merge_audio

        merge_audio(cuts, output_path, concat_with_copy=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path, merged


# Forward reference helper
from typing import Any

__all__ = [
    "collect_oped_files",
    "detect_oped_position",
    "remove_oped_segments",
]
