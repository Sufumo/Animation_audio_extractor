"""Aliyun ASR component with speaker diarization.

Wraps DashScope file transcription and converts the result to SRT format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.aliyun.dashscope_client import run_filetrans


def _ms_to_srt_time(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sentences_to_srt(sentences: List[Dict[str, Any]]) -> str:
    """Convert transcription sentences to SRT format."""
    lines: List[str] = []
    idx = 1
    for s in sentences:
        text = s.get("text", "") or ""
        begin = int(s.get("begin_time") or 0)
        end = int(s.get("end_time") or begin)
        # Skip zero-duration entries (begin == end) which cause ffmpeg errors.
        if end <= begin:
            continue
        speaker = s.get("speaker_id", "unknown")
        lines.append(str(idx))
        lines.append(f"{_ms_to_srt_time(begin)} --> {_ms_to_srt_time(end)}")
        lines.append(f"[{speaker}] {text.strip()}")
        lines.append("")
        idx += 1
    return "\n".join(lines).strip()


def transcribe_with_diarization(
    audio_path: str | Path,
    output_srt_path: Optional[str | Path] = None,
    speaker_count: Optional[int] = None,
) -> Dict[str, any]:
    """Transcribe audio with speaker diarization and save as SRT.

    Args:
        audio_path: Local audio file.
        output_srt_path: Optional path to write SRT.
        speaker_count: Optional expected speaker count hint.

    Returns:
        Dict with keys: srt, sentences, plain_text, task_id.
    """
    audio_path = Path(audio_path)
    result = run_filetrans(
        audio_path,
        enable_itn=False,
        enable_words=True,
        diarization_enabled=True,
        speaker_count=speaker_count,
    )
    sentences = result["sentences"]
    srt_text = sentences_to_srt(sentences)

    if output_srt_path is not None:
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        output_srt_path.write_text(srt_text, encoding="utf-8")

    return {
        "srt": srt_text,
        "sentences": sentences,
        "plain_text": result["plain_text"],
        "task_id": result["task_id"],
    }


# Forward reference helper
from typing import Any

__all__ = ["transcribe_with_diarization", "sentences_to_srt"]
