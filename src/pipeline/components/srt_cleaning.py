"""Clean an SRT file using Qwen 3.6-max.

Removes:
- Meaningless utterances (e.g., "啊啊啊啊", "嗯嗯", "哦" only).
- All speakers except the main speaker (the speaker with the most total speech time).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.aliyun.dashscope_client import qwen_chat


DEFAULT_MODEL = "qwen3.6-max"


def _ms_to_srt_time(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """Parse SRT text into structured entries."""
    entries: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # First line is index, second is time, rest is text.
        try:
            idx = int(lines[0])
        except ValueError:
            continue
        time_line = lines[1]
        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            time_line,
        )
        if not match:
            continue
        text = " ".join(lines[2:])
        # Try to extract [speaker] prefix.
        speaker_match = re.match(r"^\[(.+?)\]\s*(.*)$", text)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2)
        else:
            speaker = "unknown"

        def _parse_time(t: str) -> int:
            h, m, s_ms = t.split(":")
            s, ms = s_ms.split(",")
            return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

        entries.append({
            "index": idx,
            "begin_ms": _parse_time(match.group(1)),
            "end_ms": _parse_time(match.group(2)),
            "speaker": speaker,
            "text": text.strip(),
        })
    return entries


def find_main_speaker(entries: List[Dict[str, Any]]) -> str:
    """Identify the speaker with the longest total speech time."""
    speaker_time: Dict[str, int] = defaultdict(int)
    for e in entries:
        speaker_time[e["speaker"]] += e["end_ms"] - e["begin_ms"]
    if not speaker_time:
        return "unknown"
    return max(speaker_time.items(), key=lambda x: x[1])[0]


def all_speakers_unknown(entries: List[Dict[str, Any]]) -> bool:
    """Check if all entries have unknown speaker (diarization failed)."""
    return all(e["speaker"] == "unknown" for e in entries)


def is_meaningless(text: str) -> bool:
    """Heuristic: detect pure filler utterances."""
    cleaned = re.sub(r"[^一-龥a-zA-Z0-9]", "", text)
    if not cleaned:
        return True
    # Pure repeated single characters / fillers.
    if re.fullmatch(r"[啊阿哦嗯咦哎喂呵哈嘻嘿]*", cleaned):
        return True
    return False


def clean_srt(
    input_srt_path: str | Path,
    output_srt_path: Optional[str | Path] = None,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
    main_speaker: Optional[str] = None,
) -> str:
    """Clean an SRT file and return the cleaned SRT text.

    When `use_llm` is True, the full SRT is sent to Qwen 3.6-max for semantic
    cleaning. The local heuristics are always applied afterwards as a safety net.

    Args:
        main_speaker: Optional override for the target speaker id. When omitted,
            falls back to the longest-duration speaker heuristic.
    """
    input_srt_path = Path(input_srt_path)
    srt_text = input_srt_path.read_text(encoding="utf-8")
    entries = parse_srt(srt_text)

    if not entries:
        cleaned_text = srt_text
    else:
        if main_speaker is None:
            main_speaker = find_main_speaker(entries)
        # When diarization fails (all unknown), skip speaker filtering and keep all entries.
        skip_speaker_filter = all_speakers_unknown(entries)

        if use_llm:
            system_prompt = (
                "You are an SRT subtitle cleaning assistant. Your task:\n"
                "1. Keep only subtitles spoken by the main speaker. Delete all other speakers.\n"
                "2. Delete meaningless utterances such as '啊啊啊', '嗯嗯', '哦', '啊', '咦', etc.\n"
                "3. Do NOT modify the text content of kept subtitles.\n"
                "4. Preserve SRT format exactly: index line, time line, text line, blank line.\n"
                "5. Re-number indices sequentially starting from 1.\n"
                "6. Do not add any explanations or markdown code blocks."
            )
            user_prompt = (
                f"Main speaker is '{main_speaker}'. Clean the following SRT:\n\n{srt_text}"
            )
            try:
                llm_output = qwen_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                    temperature=0.1,
                )
                cleaned_entries = parse_srt(llm_output)
            except Exception:
                cleaned_entries = []
        else:
            cleaned_entries = []

        # Fallback / safety net: apply local filters.
        if not cleaned_entries:
            if skip_speaker_filter:
                # Diarization failed: keep all non-meaningless entries.
                cleaned_entries = [e for e in entries if not is_meaningless(e["text"])]
            else:
                cleaned_entries = [
                    e for e in entries
                    if e["speaker"] == main_speaker and not is_meaningless(e["text"])
                ]
        else:
            cleaned_entries = [
                e for e in cleaned_entries
                if not is_meaningless(e["text"])
            ]

        # Rebuild SRT.
        lines: List[str] = []
        for i, e in enumerate(cleaned_entries, 1):
            lines.append(str(i))
            lines.append(f"{_ms_to_srt_time(e['begin_ms'])} --> {_ms_to_srt_time(e['end_ms'])}")
            lines.append(e["text"])
            lines.append("")
        cleaned_text = "\n".join(lines).strip()

    if output_srt_path is not None:
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        output_srt_path.write_text(cleaned_text, encoding="utf-8")

    return cleaned_text


# Forward reference helper
from typing import Any

__all__ = ["parse_srt", "find_main_speaker", "all_speakers_unknown", "is_meaningless", "clean_srt"]
