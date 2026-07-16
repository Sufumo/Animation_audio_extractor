"""Standalone script: convert animation episodes to a high-quality voice dataset.

This is a convenience wrapper around the pipeline that does not require writing
a config file. It is equivalent to running Stage 1 + Stage 2 with sensible
defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task_state import TaskState
from src.pipeline.stage1_data_cleaning import run_stage1
from src.pipeline.stage2_speaker_extraction import run_stage2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert animation episodes to a target-speaker voice dataset"
    )
    parser.add_argument("--source-dir", required=True, help="Animation source directory")
    parser.add_argument("--oped-dir", help="OP/ED directory (optional)")
    parser.add_argument("--reference-dir", required=True, help="Target speaker reference directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--task-dir", help="Task checkpoint folder (optional)")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip speaker extraction")
    parser.add_argument("--run-unise-v2", action="store_true", help="Run second UniSE pass")
    parser.add_argument(
        "--melband-dir",
        default="/Users/AITraining/Documents/Personal/train_audio_extract/plugins/Mel-Band-Roformer-Vocal-Model",
        help="Mel-Band-Roformer project directory",
    )
    parser.add_argument(
        "--melband-model",
        default="/Users/AITraining/Documents/Personal/train_audio_extract/models/MelBandRoformer.ckpt",
        help="Mel-Band-Roformer checkpoint path",
    )
    parser.add_argument(
        "--unise-dir",
        default="/Users/AITraining/Documents/Personal/train_audio_extract/plugins/unified-audio/QuarkAudio-UniSE",
        help="UniSE project directory",
    )
    parser.add_argument(
        "--unise-ckpt",
        default="/Users/AITraining/Documents/Personal/train_audio_extract/plugins/unified-audio/QuarkAudio-UniSE/checkpoints/epoch=20-step=109367.ckpt",
        help="UniSE checkpoint path",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    task_dir = Path(args.task_dir) if args.task_dir else output_dir / ".task"

    for d in (source_dir, reference_dir):
        if not d.exists():
            print(f"Error: directory not found: {d}", file=sys.stderr)
            return 1

    task_state = TaskState(task_dir)
    task_state.set_inputs({
        "source_dir": str(source_dir),
        "oped_dir": str(args.oped_dir) if args.oped_dir else None,
        "reference_dir": str(reference_dir),
    })

    cleaned_wavs = []
    if not args.skip_stage1:
        if task_state.is_done("stage1"):
            print("Stage 1 already completed; skipping.")
            cleaned_wavs = [Path(p) for p in task_state.get_outputs("stage1").get("cleaned_wavs", [])]
        else:
            print("Running Stage 1: data cleaning...")
            cleaned_wavs = run_stage1(
                task_state=task_state,
                source_dir=source_dir,
                output_dir=output_dir / "stage1",
                oped_dir=args.oped_dir,
                reference_dir=reference_dir,
                melband_dir=args.melband_dir,
                melband_model_path=args.melband_model,
            )
            print(f"Stage 1 done: {cleaned_wavs}")

    if not args.skip_stage2:
        if not cleaned_wavs:
            cleaned_wavs = [Path(p) for p in task_state.get_outputs("stage1").get("cleaned_wavs", [])]
        if not cleaned_wavs:
            print("Error: no cleaned WAVs available for Stage 2", file=sys.stderr)
            return 1

        if task_state.is_done("stage2"):
            print("Stage 2 already completed; skipping.")
        else:
            print("Running Stage 2: speaker extraction...")
            final_clips = run_stage2(
                task_state=task_state,
                cleaned_wavs=cleaned_wavs,
                reference_dir=reference_dir,
                output_dir=output_dir / "stage2",
                unise_dir=args.unise_dir,
                unise_ckpt_path=args.unise_ckpt,
                run_unise_v2=args.run_unise_v2,
            )
            print(f"Stage 2 done: {final_clips}")

    print(f"Task state saved to: {task_dir / 'task_state.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
