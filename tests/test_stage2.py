"""Tests for Stage 2 components (no external API calls)."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src import audio_utils
from src.pipeline.components import silence_removal, audio_mapping, srt_cleaning


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _make_wav(path: Path, duration: float, freq: float = 1000, sr: int = 16000):
    audio_utils._run_ffmpeg([
        "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={duration}",
        "-ar", str(sr),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(path),
    ])


def test_silence_removal(tmp_dir):
    # Create audio: 2s speech + 1s silence + 2s speech
    s1 = tmp_dir / "s1.wav"
    silence = tmp_dir / "silence.wav"
    s2 = tmp_dir / "s2.wav"
    _make_wav(s1, 2.0)
    _make_wav(silence, 1.0, freq=0)
    _make_wav(s2, 2.0)
    concat = tmp_dir / "concat.txt"
    concat.write_text(
        f"file '{s1.resolve()}'\nfile '{silence.resolve()}'\nfile '{s2.resolve()}'\n",
        encoding="utf-8",
    )
    input_wav = tmp_dir / "input.wav"
    audio_utils._run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(input_wav)])

    out = tmp_dir / "out.wav"
    map_path = tmp_dir / "map.json"
    _, map_data = silence_removal.remove_silence(input_wav, out, map_path, sr=16000)

    duration = audio_utils.get_duration(out)
    assert 3.8 <= duration <= 4.2
    assert len(map_data["segments"]) == 2


def test_srt_parsing_and_main_speaker(tmp_dir):
    srt_text = """1
00:00:01,000 --> 00:00:03,000
[spk1] Hello world

2
00:00:04,000 --> 00:00:05,000
[spk2] 啊啊啊

3
00:00:06,000 --> 00:00:08,000
[spk1] Another sentence
"""
    entries = srt_cleaning.parse_srt(srt_text)
    assert len(entries) == 3
    main = srt_cleaning.find_main_speaker(entries)
    assert main == "spk1"
    assert srt_cleaning.is_meaningless("啊啊啊")
    assert not srt_cleaning.is_meaningless("Hello world")


def test_clean_srt_respects_main_speaker_override(tmp_dir):
    # Speaker 0 talks longer, but override forces speaker 1.
    srt_text = """1
00:00:01,000 --> 00:00:05,000
[0] long unwanted speech here

2
00:00:06,000 --> 00:00:07,500
[1] target line
"""
    input_srt = tmp_dir / "input.srt"
    input_srt.write_text(srt_text, encoding="utf-8")
    output_srt = tmp_dir / "output.srt"
    cleaned = srt_cleaning.clean_srt(
        input_srt, output_srt, use_llm=False, main_speaker="1"
    )
    assert "target line" in cleaned
    assert "unwanted" not in cleaned


def test_select_speaker_samples_spreads_and_filters():
    from src.pipeline.components import speaker_verify

    entries = []
    for i in range(10):
        entries.append({
            "speaker": "0",
            "begin_ms": i * 10000,
            "end_ms": i * 10000 + 3000,
            "text": f"a{i}",
        })
    entries.append({
        "speaker": "1",
        "begin_ms": 0,
        "end_ms": 500,  # too short
        "text": "short",
    })
    entries.append({
        "speaker": "1",
        "begin_ms": 1000,
        "end_ms": 4000,
        "text": "ok",
    })
    selected = speaker_verify.select_speaker_samples(entries, samples_per_speaker=4)
    assert "0" in selected
    assert len(selected["0"]) == 4
    assert selected["1"] == [entries[-1]]


def test_map_srt_to_original(tmp_dir):
    # Silence map: 0-5s original became 0-5s output (silence removed 5-10s)
    silence_map = {
        "segments": [
            {"original_start_sec": 0.0, "original_end_sec": 5.0, "output_start_sec": 0.0, "output_end_sec": 5.0},
            {"original_start_sec": 10.0, "original_end_sec": 15.0, "output_start_sec": 5.0, "output_end_sec": 10.0},
        ]
    }
    timestamps = [(6.0, 8.0)]  # in silence-removed audio
    mapped = audio_mapping.map_srt_to_original(timestamps, silence_map)
    assert abs(mapped[0][0] - 11.0) < 0.01
    assert abs(mapped[0][1] - 13.0) < 0.01


def test_map_srt_to_original_splits_range_across_removed_silence():
    silence_map = {
        "segments": [
            {"original_start_sec": 0.0, "original_end_sec": 5.0, "output_start_sec": 0.0, "output_end_sec": 5.0},
            {"original_start_sec": 10.0, "original_end_sec": 15.0, "output_start_sec": 5.0, "output_end_sec": 10.0},
        ]
    }

    mapped = audio_mapping.map_srt_to_original([(4.0, 6.0)], silence_map)

    assert mapped == [(4.0, 5.0), (10.0, 11.0)]


def test_map_asr_to_original_splits_utterance_across_gaps():
    asr_map = {
        "segments": [
            {
                "original_start_sec": 10.0,
                "original_end_sec": 15.0,
                "asr_start_sec": 0.0,
                "asr_end_sec": 5.0,
            },
            {
                "original_start_sec": 30.0,
                "original_end_sec": 34.0,
                "asr_start_sec": 7.0,
                "asr_end_sec": 11.0,
            },
        ]
    }

    mapped = audio_mapping.map_asr_to_original([(3.0, 9.0)], asr_map)

    assert mapped == [(13.0, 15.0), (30.0, 32.0)]


def test_map_asr_to_original_drops_artificial_gap_only_range():
    asr_map = {
        "segments": [
            {
                "original_start_sec": 10.0,
                "original_end_sec": 15.0,
                "asr_start_sec": 0.0,
                "asr_end_sec": 5.0,
            },
            {
                "original_start_sec": 30.0,
                "original_end_sec": 34.0,
                "asr_start_sec": 7.0,
                "asr_end_sec": 11.0,
            },
        ]
    }

    assert audio_mapping.map_asr_to_original([(5.2, 6.8)], asr_map) == []
