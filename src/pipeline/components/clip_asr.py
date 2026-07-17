"""Per-clip ASR without speaker diarization.

Each clip is transcribed independently. Timestamps stay on the clip's own
timeline (or absolute original times if provided in metadata). No audio
concatenation / asr_map is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.aliyun.dashscope_client import run_filetrans
from src.pipeline.components.srt_cleaning import is_meaningless


def _ms_to_srt_time(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_clips(
    clip_paths: Sequence[str | Path],
    output_json_path: Optional[str | Path] = None,
    output_srt_path: Optional[str | Path] = None,
    filter_meaningless: bool = True,
    clip_meta: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Transcribe each clip with Aliyun ASR (diarization OFF).

    Args:
        clip_paths: Accepted training clips.
        clip_meta: Optional parallel metadata with original_start_sec /
            original_end_sec for absolute SRT timestamps.

    Returns:
        Dict with per-clip transcriptions and aggregated SRT text.
    """
    results: List[Dict[str, Any]] = []
    srt_lines: List[str] = []
    srt_idx = 1

    for i, clip in enumerate(clip_paths):
        clip = Path(clip)
        meta = dict(clip_meta[i]) if clip_meta and i < len(clip_meta) else {}
        entry: Dict[str, Any] = {
            "index": i,
            "path": str(clip),
            "original_start_sec": meta.get("original_start_sec"),
            "original_end_sec": meta.get("original_end_sec"),
        }
        try:
            asr = run_filetrans(
                clip,
                enable_itn=False,
                enable_words=True,
                diarization_enabled=False,
                speaker_count=None,
            )
            text = (asr.get("plain_text") or "").strip()
            sentences = asr.get("sentences") or []
            entry["text"] = text
            entry["sentences"] = sentences
            entry["task_id"] = asr.get("task_id")
        except Exception as e:
            entry["text"] = ""
            entry["error"] = str(e)
            results.append(entry)
            continue

        if filter_meaningless and is_meaningless(text):
            entry["filtered"] = True
            entry["filter_reason"] = "meaningless"
            results.append(entry)
            continue

        entry["filtered"] = False
        results.append(entry)

        # Absolute timeline if original bounds known; else clip-relative.
        if entry.get("original_start_sec") is not None and entry.get("original_end_sec") is not None:
            begin_ms = int(float(entry["original_start_sec"]) * 1000)
            end_ms = int(float(entry["original_end_sec"]) * 1000)
        elif sentences:
            begin_ms = int(sentences[0].get("begin_time") or 0)
            end_ms = int(sentences[-1].get("end_time") or begin_ms)
        else:
            begin_ms, end_ms = 0, 1000

        if end_ms <= begin_ms:
            continue

        srt_lines.append(str(srt_idx))
        srt_lines.append(f"{_ms_to_srt_time(begin_ms)} --> {_ms_to_srt_time(end_ms)}")
        srt_lines.append(text)
        srt_lines.append("")
        srt_idx += 1

    srt_text = "\n".join(srt_lines).strip()
    kept = [r for r in results if not r.get("filtered") and r.get("text") and not r.get("error")]

    output = {
        "num_input": len(clip_paths),
        "num_kept": len(kept),
        "clips": results,
        "srt": srt_text,
    }

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    if output_srt_path is not None:
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        output_srt_path.write_text(srt_text, encoding="utf-8")

    return output


__all__ = ["transcribe_clips"]
