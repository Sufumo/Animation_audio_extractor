"""Identify the target ASR speaker via pyannote embedding similarity.

Compares each diarization speaker's sampled clips against a reference enrollment
audio, then returns the speaker id with the highest cosine similarity.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.audio_utils import cut_segment
from src.pipeline.components.srt_cleaning import parse_srt


DEFAULT_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def _load_embedding_inference(model_id: str = DEFAULT_EMBEDDING_MODEL):
    """Lazy-load pyannote embedding model (keeps torch version untouched)."""
    from pyannote.audio import Inference, Model

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    model = Model.from_pretrained(model_id, token=token)
    return Inference(model, window="whole")


def _embedding_vector(inference, audio_path: str | Path) -> np.ndarray:
    raw = inference(str(audio_path))
    if hasattr(raw, "data"):
        raw = raw.data
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def select_speaker_samples(
    entries: List[Dict[str, Any]],
    samples_per_speaker: int = 8,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 12.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pick timeline-spread samples per speaker within a duration band."""
    by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        speaker = str(entry.get("speaker", "unknown"))
        if speaker == "unknown":
            continue
        dur = (entry["end_ms"] - entry["begin_ms"]) / 1000.0
        if dur < min_duration_sec or dur > max_duration_sec:
            continue
        by_speaker[speaker].append(entry)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    for speaker, segs in by_speaker.items():
        segs = sorted(segs, key=lambda e: e["begin_ms"])
        if not segs:
            continue
        n = min(samples_per_speaker, len(segs))
        idxs = np.linspace(0, len(segs) - 1, num=n, dtype=int)
        picked = []
        seen = set()
        for idx in idxs:
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(segs[idx])
        selected[speaker] = picked
    return selected


def identify_target_speaker(
    srt_path: str | Path,
    audio_path: str | Path,
    reference_path: str | Path,
    output_scores_path: Optional[str | Path] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    samples_per_speaker: int = 8,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 12.0,
) -> Dict[str, Any]:
    """Return the diarization speaker most similar to the reference audio.

    Returns:
        Dict with keys:
        - target_speaker: str
        - speaker_scores: {speaker_id: mean_cosine}
        - centroid_scores: {speaker_id: centroid_cosine}
        - samples: detailed per-clip scores
    """
    srt_path = Path(srt_path)
    audio_path = Path(audio_path)
    reference_path = Path(reference_path)

    entries = parse_srt(srt_path.read_text(encoding="utf-8"))
    samples = select_speaker_samples(
        entries,
        samples_per_speaker=samples_per_speaker,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )
    if not samples:
        raise ValueError(
            f"No usable speaker segments found in {srt_path} "
            f"(need duration in [{min_duration_sec}, {max_duration_sec}]s)"
        )

    inference = _load_embedding_inference(embedding_model)
    ref_emb = _embedding_vector(inference, reference_path)

    tmp_dir = audio_path.parent / ".speaker_verify_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    detailed: List[Dict[str, Any]] = []
    mean_scores: Dict[str, float] = {}
    centroid_scores: Dict[str, float] = {}

    try:
        for speaker, segs in samples.items():
            emb_list: List[np.ndarray] = []
            clip_scores: List[float] = []
            for i, seg in enumerate(segs):
                start = seg["begin_ms"] / 1000.0
                end = seg["end_ms"] / 1000.0
                clip_path = tmp_dir / f"spk{speaker}_{i:02d}_{start:.3f}_{end:.3f}.wav"
                cut_segment(audio_path, clip_path, start, end)
                emb = _embedding_vector(inference, clip_path)
                score = _cosine(ref_emb, emb)
                emb_list.append(emb)
                clip_scores.append(score)
                detailed.append({
                    "speaker": speaker,
                    "begin_sec": round(start, 4),
                    "end_sec": round(end, 4),
                    "duration_sec": round(end - start, 4),
                    "text": seg.get("text", ""),
                    "cosine_sim": round(score, 4),
                })

            mean_scores[speaker] = float(np.mean(clip_scores)) if clip_scores else -1.0
            centroid = np.mean(np.stack(emb_list), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            centroid_scores[speaker] = _cosine(ref_emb, centroid)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Prefer centroid score (more stable); break ties with mean.
    target_speaker = max(
        centroid_scores.keys(),
        key=lambda spk: (centroid_scores[spk], mean_scores.get(spk, -1.0)),
    )

    result = {
        "target_speaker": target_speaker,
        "embedding_model": embedding_model,
        "speaker_scores": {k: round(v, 4) for k, v in mean_scores.items()},
        "centroid_scores": {k: round(v, 4) for k, v in centroid_scores.items()},
        "samples": detailed,
    }

    if output_scores_path is not None:
        output_scores_path = Path(output_scores_path)
        output_scores_path.parent.mkdir(parents=True, exist_ok=True)
        output_scores_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "select_speaker_samples",
    "identify_target_speaker",
]
