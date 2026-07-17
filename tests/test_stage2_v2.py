"""Unit tests for Stage 2 v2 components (offline, no cloud / heavy models)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.components.vad_detection import detect_with_energy, detect_active_segments
from src.pipeline.components.speaker_verification import (
    build_reference_embedding,
    verify_clips,
    _cosine_similarity,
)
from src.pipeline.components.quality_gate import apply_quality_gate
from src.pipeline.stage2_vad_verify import _split_long_intervals


def _write_sine_wav(path: Path, duration: float = 1.0, sr: int = 16000, freq: float = 440.0, amp: float = 0.3):
    import soundfile as sf
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wav, sr)


def _write_silence_wav(path: Path, duration: float = 1.0, sr: int = 16000):
    import soundfile as sf
    wav = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), wav, sr)


def test_split_long_intervals():
    intervals = [(0.0, 5.0), (10.0, 30.0)]
    out = _split_long_intervals(intervals, max_sec=10.0)
    assert out[0] == (0.0, 5.0)
    assert out[1] == (10.0, 20.0)
    assert out[2] == (20.0, 30.0)


def test_energy_vad_detects_speech(tmp_path: Path):
    # speech + silence + speech
    import soundfile as sf
    sr = 16000
    speech = (0.4 * np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, sr))).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    wav = np.concatenate([speech, silence, speech])
    path = tmp_path / "mix.wav"
    sf.write(str(path), wav, sr)

    intervals = detect_with_energy(wav, sr, top_db=30.0, abs_dbfs_floor=-40.0)
    assert len(intervals) >= 2
    assert intervals[0][1] - intervals[0][0] > 0.5


def test_detect_active_segments_energy(tmp_path: Path):
    path = tmp_path / "s.wav"
    _write_sine_wav(path, duration=2.0)
    result = detect_active_segments(path, method="energy", top_db=30.0)
    assert result["method"] == "energy"
    assert result["num_segments"] >= 1


def test_mfcc_speaker_verify(tmp_path: Path):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _write_sine_wav(ref_dir / "ref.wav", duration=2.0, freq=220.0)

    same = tmp_path / "same.wav"
    diff = tmp_path / "diff.wav"
    _write_sine_wav(same, duration=1.5, freq=220.0)
    _write_sine_wav(diff, duration=1.5, freq=880.0, amp=0.3)

    result = verify_clips(
        [same, diff],
        reference_dir=ref_dir,
        threshold=0.5,
        backend="mfcc",
    )
    assert result["backend"] == "mfcc"
    assert result["num_accepted"] + result["num_rejected"] == 2
    # Same-frequency clip should score higher than different-frequency.
    scores = {Path(a["path"]).name: a["score"] for a in result["accepted"]}
    scores.update({Path(r["path"]).name: r["score"] for r in result["rejected"] if r["score"] is not None})
    assert scores["same.wav"] > scores["diff.wav"]


def test_quality_gate_filters_short(tmp_path: Path):
    short = tmp_path / "short.wav"
    long = tmp_path / "ok.wav"
    _write_sine_wav(short, duration=0.4)
    _write_sine_wav(long, duration=2.5)

    out = tmp_path / "qc"
    result = apply_quality_gate(
        [short, long],
        output_dir=out,
        min_duration_sec=1.0,
        max_duration_sec=12.0,
        normalize_loudness=False,
        min_lufs=None,
        max_lufs=None,
    )
    assert result["num_accepted"] == 1
    assert result["num_rejected"] == 1


def test_cosine_similarity_unit():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6
    assert abs(_cosine_similarity(a, c)) < 1e-6
