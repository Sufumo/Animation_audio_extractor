"""Stage 2 dispatcher: prefer v2 (VAD + speaker verification), keep Aliyun as fallback.

Set ``stage2.mode`` in config:
- ``v2`` / ``vad`` (default): UniSE → VAD → ECAPA verify → quality gate → optional clip ASR
- ``v1`` / ``aliyun`` / ``legacy``: UniSE → ASR diarization → (optional pyannote) → SRT clean → rebuild
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.task_state import TaskState


def run_stage2(
    task_state: TaskState,
    cleaned_wavs: List[str | Path],
    reference_dir: str | Path,
    output_dir: str | Path,
    components: Optional[List[str]] = None,
    mode: str = "v2",
    **kwargs: Any,
) -> List[Path]:
    """Dispatch to v2 or legacy Aliyun Stage 2 implementation."""
    normalized = (mode or "v2").strip().lower()
    if normalized in ("v1", "aliyun", "legacy"):
        from src.pipeline.stage2_aliyun import run_stage2_aliyun

        print(f"[stage2] mode={normalized} → Aliyun ASR diarization pipeline (legacy)")
        return run_stage2_aliyun(
            task_state=task_state,
            cleaned_wavs=cleaned_wavs,
            reference_dir=reference_dir,
            output_dir=output_dir,
            components=components,
            **kwargs,
        )

    from src.pipeline.stage2_vad_verify import run_stage2_v2

    print(f"[stage2] mode={normalized or 'v2'} → VAD + speaker-verification pipeline")
    return run_stage2_v2(
        task_state=task_state,
        cleaned_wavs=cleaned_wavs,
        reference_dir=reference_dir,
        output_dir=output_dir,
        components=components,
        **kwargs,
    )


__all__ = ["run_stage2"]
