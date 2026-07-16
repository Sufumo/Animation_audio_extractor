"""Tests for Stage 1 components."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.pipeline.components import mp4_to_wav, oped_removal
from src import audio_utils


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _make_media(path: Path, duration: float, freq: float = 1000, sr: int = 44100, channels: int = 2):
    """Generate a synthetic audio file (WAV/MP3/MP4) with ffmpeg."""
    ext = path.suffix.lower()
    if ext == ".mp4":
        cmd = [
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-f", "lavfi",
            "-i", f"color=c=black:s=64x64:d={duration}",
            "-shortest",
            "-ar", str(sr),
            "-ac", str(channels),
            "-acodec", "aac",
            "-vcodec", "libx264",
            str(path),
        ]
    elif ext == ".mp3":
        cmd = [
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-ar", str(sr),
            "-ac", str(channels),
            "-acodec", "libmp3lame",
            str(path),
        ]
    else:
        cmd = [
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-ar", str(sr),
            "-ac", str(channels),
            "-acodec", "pcm_s16le",
            str(path),
        ]
    audio_utils._run_ffmpeg(cmd)


def test_convert_sources(tmp_dir):
    src = tmp_dir / "source"
    src.mkdir()
    _make_media(src / "ep01.wav", duration=2.0)
    _make_media(src / "ep02.mp3", duration=2.0)
    out = tmp_dir / "wav"
    outputs = mp4_to_wav.convert_sources(src, out, sample_rate=44100)
    assert len(outputs) == 2
    assert all(p.suffix == ".wav" for p in outputs)


def test_oped_detection_and_removal(tmp_dir):
    # Create a fake episode: 20s sine, with a 2s OP clip (different freq) inserted at start.
    episode = tmp_dir / "episode.wav"
    oped = tmp_dir / "oped.wav"

    # OP clip: 2s, 2000Hz
    _make_media(oped, duration=2.0, freq=2000)

    # Episode: OP (2s) + main (18s, 1000Hz)
    main = tmp_dir / "main.wav"
    _make_media(main, duration=18.0, freq=1000)

    # Merge OP + main using ffmpeg concat with copy (same fmt)
    list_file = tmp_dir / "list.txt"
    list_file.write_text(
        f"file '{oped.resolve()}'\nfile '{main.resolve()}'\n", encoding="utf-8"
    )
    audio_utils._run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(episode)
    ])

    out = tmp_dir / "no_oped.wav"
    oped_dir = tmp_dir / "oped_dir"
    oped_dir.mkdir()
    shutil.copy(str(oped), str(oped_dir / "op.wav"))

    _, removed = oped_removal.remove_oped_segments(episode, oped_dir, out)
    assert len(removed) >= 1
    # Original 20s; after removing 2s OP, duration should be ~18s.
    duration = audio_utils.get_duration(out)
    assert 17.5 <= duration <= 18.5
