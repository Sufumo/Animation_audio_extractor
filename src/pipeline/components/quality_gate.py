"""Post-extraction quality gate for training clips.

Filters by duration and optional loudness; optionally loudness-normalizes
accepted clips with ffmpeg loudnorm.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.audio_utils import get_duration


def _measure_lufs(audio_path: Path) -> Optional[float]:
    """Measure integrated loudness (LUFS) via ffmpeg loudnorm print_format=json."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(audio_path),
        "-af", "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm stats are printed to stderr.
    text = result.stderr or ""
    # Find last JSON object in stderr.
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return float(data.get("input_i", "nan"))
    except Exception:
        return None


def _loudnorm_copy(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -23.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def apply_quality_gate(
    clip_paths: Sequence[str | Path],
    output_dir: str | Path,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 12.0,
    min_lufs: Optional[float] = -50.0,
    max_lufs: Optional[float] = -5.0,
    normalize_loudness: bool = True,
    target_lufs: float = -23.0,
    results_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Filter and optionally normalize clips for TTS training.

    Returns:
        Dict with accepted clip paths and rejection reasons.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for idx, clip in enumerate(clip_paths):
        clip = Path(clip)
        duration = get_duration(clip)
        meta: Dict[str, Any] = {
            "source": str(clip),
            "duration_sec": round(duration, 4),
        }

        if duration < min_duration_sec:
            meta["reason"] = f"too_short<{min_duration_sec}"
            rejected.append(meta)
            continue
        if duration > max_duration_sec:
            meta["reason"] = f"too_long>{max_duration_sec}"
            rejected.append(meta)
            continue

        lufs = _measure_lufs(clip)
        meta["lufs"] = lufs
        if lufs is not None:
            if min_lufs is not None and lufs < min_lufs:
                meta["reason"] = f"too_quiet<{min_lufs}"
                rejected.append(meta)
                continue
            if max_lufs is not None and lufs > max_lufs:
                meta["reason"] = f"too_loud>{max_lufs}"
                rejected.append(meta)
                continue

        out = output_dir / f"qc_{idx:04d}_{clip.stem}.wav"
        if normalize_loudness:
            try:
                _loudnorm_copy(clip, out, target_lufs=target_lufs)
            except Exception:
                shutil.copy(str(clip), str(out))
        else:
            shutil.copy(str(clip), str(out))

        meta["path"] = str(out)
        accepted.append(meta)

    result = {
        "min_duration_sec": min_duration_sec,
        "max_duration_sec": max_duration_sec,
        "normalize_loudness": normalize_loudness,
        "target_lufs": target_lufs,
        "num_input": len(clip_paths),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


__all__ = ["apply_quality_gate"]
