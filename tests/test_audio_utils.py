"""Tests for src.audio_utils."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src import audio_utils


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_wav(tmp_dir):
    """Create a 10-second 44100Hz stereo PCM WAV."""
    path = tmp_dir / "input.wav"
    audio_utils._run_ffmpeg([
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=10",
        "-ar", "44100",
        "-ac", "2",
        "-acodec", "pcm_s16le",
        str(path),
    ])
    return path


def test_get_duration(sample_wav):
    duration = audio_utils.get_duration(sample_wav)
    assert 9.9 <= duration <= 10.1


def test_get_audio_sample_rate(sample_wav):
    assert audio_utils.get_audio_sample_rate(sample_wav) == 44100


def test_convert_to_wav_same_format_copies(sample_wav, tmp_dir):
    out = tmp_dir / "out.wav"
    audio_utils.convert_to_wav(sample_wav, out)
    assert out.exists()
    ok, reason = audio_utils.validate_pcm_wav(out)
    assert ok, reason


def test_convert_to_wav_resample_and_mono(sample_wav, tmp_dir):
    out = tmp_dir / "out.wav"
    audio_utils.convert_to_wav(sample_wav, out, sample_rate=16000, mono=True)
    assert audio_utils.get_audio_sample_rate(out) == 16000
    info = audio_utils.get_media_info(out)
    audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert audio_stream["channels"] == 1


def test_split_and_merge_audio(sample_wav, tmp_dir):
    segments = audio_utils.split_audio(sample_wav, tmp_dir / "seg", segment_seconds=3.0)
    assert len(segments) == 4  # 10s / 3s = ceil(10/3) = 4

    merged = tmp_dir / "merged.wav"
    audio_utils.merge_audio(segments, merged, concat_with_copy=True)
    ok, reason = audio_utils.validate_pcm_wav(merged)
    assert ok, reason
    # Copy concat preserves total duration.
    assert abs(audio_utils.get_duration(merged) - 10.0) < 0.1


def test_resample(sample_wav, tmp_dir):
    out = tmp_dir / "resampled.wav"
    audio_utils.resample(sample_wav, out, target_sr=16000)
    assert audio_utils.get_audio_sample_rate(out) == 16000


def test_cut_segment(sample_wav, tmp_dir):
    out = tmp_dir / "cut.wav"
    audio_utils.cut_segment(sample_wav, out, start=2.0, end=5.0)
    duration = audio_utils.get_duration(out)
    assert 2.9 <= duration <= 3.1
