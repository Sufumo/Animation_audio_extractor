"""Speaker verification via embedding cosine similarity.

Primary backend: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb).
Fallback: lightweight MFCC mean/std embedding (no download, for smoke tests).

Reference enrollments can be one or many files; embeddings are averaged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def _normalize_device(device: str) -> str:
    """SpeechBrain expects 'cuda:0' / 'cpu', not bare 'cuda'."""
    d = (device or "cpu").strip().lower()
    if d == "cuda" or d in ("gpu", "cuda0"):
        return "cuda:0"
    return device


def _collect_audio_files(path: str | Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_ECAPA_MODEL = None
_ECAPA_DEVICE = None


def _get_ecapa_model(device: str = "cpu"):
    global _ECAPA_MODEL, _ECAPA_DEVICE
    device = _normalize_device(device)
    if _ECAPA_MODEL is None or _ECAPA_DEVICE != device:
        from speechbrain.inference.speaker import EncoderClassifier
        _ECAPA_MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ECAPA_DEVICE = device
    return _ECAPA_MODEL


def embed_ecapa(audio_path: str | Path, device: str = "cpu") -> np.ndarray:
    """Extract ECAPA-TDNN embedding for one audio file."""
    import torchaudio

    device = _normalize_device(device)
    model = _get_ecapa_model(device=device)
    signal, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    try:
        signal = signal.to(device)
    except Exception:
        pass
    emb = model.encode_batch(signal)
    return emb.squeeze().detach().cpu().numpy()

def embed_mfcc(audio_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Lightweight MFCC mean+std embedding (offline fallback)."""
    import librosa

    wav, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if len(wav) < sr * 0.2:
        # Pad very short clips.
        wav = np.pad(wav, (0, max(0, int(sr * 0.2) - len(wav))))
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def build_reference_embedding(
    reference_paths: Sequence[str | Path],
    backend: str = "auto",
    device: str = "cpu",
) -> Tuple[np.ndarray, str]:
    """Average embeddings from one or more reference files.

    Returns:
        (embedding, used_backend)
    """
    device = _normalize_device(device)
    paths = [Path(p) for p in reference_paths]
    if not paths:
        raise ValueError("No reference audio provided")

    used = backend
    embeddings: List[np.ndarray] = []

    if backend in ("ecapa", "auto"):
        try:
            for p in paths:
                embeddings.append(embed_ecapa(p, device=device))
            used = "ecapa"
        except Exception as e:
            if backend == "ecapa":
                raise
            print(f"[speaker_verify] ECAPA unavailable ({e}); falling back to MFCC.")
            used = "mfcc"
            embeddings = []

    if used == "mfcc" or backend == "mfcc":
        embeddings = [embed_mfcc(p) for p in paths]
        used = "mfcc"

    stacked = np.stack(embeddings, axis=0)
    ref = stacked.mean(axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    return ref, used


def score_clip(
    clip_path: str | Path,
    reference_embedding: np.ndarray,
    backend: str = "ecapa",
    device: str = "cpu",
) -> float:
    """Return cosine similarity between clip and reference embedding."""
    device = _normalize_device(device)
    if backend == "ecapa":
        emb = embed_ecapa(clip_path, device=device)
    else:
        emb = embed_mfcc(clip_path)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return _cosine_similarity(emb, reference_embedding)


def verify_clips(
    clip_paths: Sequence[str | Path],
    reference_dir: str | Path,
    threshold: float = 0.45,
    backend: str = "auto",
    device: str = "cpu",
    results_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Score each clip against reference and keep those above threshold.

    Returns:
        Dict with accepted/rejected lists and per-clip scores.
    """
    device = _normalize_device(device)
    ref_files = _collect_audio_files(reference_dir)
    if not ref_files:
        raise ValueError(f"No reference audio in {reference_dir}")

    ref_emb, used_backend = build_reference_embedding(
        ref_files, backend=backend, device=device,
    )

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for clip in clip_paths:
        clip = Path(clip)
        try:
            score = score_clip(clip, ref_emb, backend=used_backend, device=device)
        except Exception as e:
            rejected.append({
                "path": str(clip),
                "score": None,
                "reason": f"embed_error: {e}",
            })
            continue

        entry = {"path": str(clip), "score": round(score, 4)}
        if score >= threshold:
            accepted.append(entry)
        else:
            entry["reason"] = "below_threshold"
            rejected.append(entry)

    result = {
        "backend": used_backend,
        "threshold": threshold,
        "reference_files": [str(p) for p in ref_files],
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


__all__ = [
    "build_reference_embedding",
    "verify_clips",
    "score_clip",
    "embed_ecapa",
    "embed_mfcc",
]
