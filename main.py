"""Main entry point for the anime voice training pipeline.

Supports:
- Config-driven, pluggable stage/component execution.
- Per-task YAML checkpointing for resume/retry.
- Local smoke test mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task_state import TaskState
from src.pipeline.stage1_data_cleaning import run_stage1
from src.pipeline.stage2_speaker_extraction import run_stage2


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_args_into_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with command-line arguments."""
    if args.source_dir:
        config["source_dir"] = args.source_dir
    if args.oped_dir:
        config["oped_dir"] = args.oped_dir
    if args.reference_dir:
        config["reference_dir"] = args.reference_dir
    if args.task_dir:
        config["task_dir"] = args.task_dir
    if args.stage is not None:
        config["stage1"]["enabled"] = args.stage in ("1", "all")
        config["stage2"]["enabled"] = args.stage in ("2", "all")
    if args.resume is not None:
        config["resume"] = args.resume
    return config


def _resolve_optional_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None


def _set_aliyun_key(config: Dict[str, Any]) -> None:
    key = config.get("aliyun", {}).get("dashscope_api_key")
    if key:
        os.environ["DASHSCOPE_API_KEY"] = key


def run_pipeline(config: Dict[str, Any]) -> None:
    """Execute the configured pipeline."""
    task_dir = Path(config["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)

    task_state = TaskState(task_dir, config_path=config.get("config_path"))
    task_state.set_config_path(config.get("config_path"))

    source_dir = config.get("source_dir")
    oped_dir = config.get("oped_dir")
    reference_dir = config.get("reference_dir")
    output_dir = task_dir / config.get("output_dir", "output")
    resume = config.get("resume", True)

    if not source_dir or not Path(source_dir).exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    _set_aliyun_key(config)

    # Stage 1
    if config.get("stage1", {}).get("enabled", True):
        if resume and task_state.is_done("stage1"):
            print("[main] Stage 1 already completed; skipping.")
            cleaned_wavs = [Path(p) for p in task_state.get_outputs("stage1").get("cleaned_wavs", [])]
        else:
            print("[main] Running Stage 1: data cleaning...")
            s1_cfg = config["stage1"]
            cleaned_wavs = run_stage1(
                task_state=task_state,
                source_dir=source_dir,
                output_dir=output_dir / "stage1",
                oped_dir=oped_dir,
                reference_dir=reference_dir,
                components=s1_cfg.get("components"),
                melband_dir=config.get("melband_roformer", {}).get("project_dir"),
                melband_model_path=config.get("melband_roformer", {}).get("model_path"),
                melband_config_path=config.get("melband_roformer", {}).get("config_path"),
                bgm_segment_seconds=s1_cfg.get("bgm_segment_seconds", 360.0),
                keep_instrumental=s1_cfg.get("keep_instrumental", False),
                sample_rate=s1_cfg.get("sample_rate", 44100),
                mono=s1_cfg.get("mono", False),
                bit_depth=s1_cfg.get("bit_depth", 16),
            )
            print(f"[main] Stage 1 completed. Cleaned files: {cleaned_wavs}")
    else:
        cleaned_wavs = []

    # Stage 2
    if config.get("stage2", {}).get("enabled", False):
        if not cleaned_wavs:
            cleaned_wavs = [Path(p) for p in task_state.get_outputs("stage1").get("cleaned_wavs", [])]
        if not cleaned_wavs:
            raise ValueError("Stage 2 enabled but no cleaned WAVs available")
        if not reference_dir or not Path(reference_dir).exists():
            raise FileNotFoundError(f"Stage 2 requires reference_dir: {reference_dir}")

        if resume and task_state.is_done("stage2"):
            print("[main] Stage 2 already completed; skipping.")
        else:
            print("[main] Running Stage 2: speaker extraction...")
            s2_cfg = config["stage2"]
            final_clips = run_stage2(
                task_state=task_state,
                cleaned_wavs=cleaned_wavs,
                reference_dir=reference_dir,
                output_dir=output_dir / "stage2",
                components=s2_cfg.get("components"),
                unise_dir=config.get("unise", {}).get("project_dir"),
                unise_ckpt_path=config.get("unise", {}).get("ckpt_path"),
                segment_seconds=s2_cfg.get("segment_seconds", 360.0),
                asr_speaker_count=s2_cfg.get("asr_speaker_count"),
                srt_model=s2_cfg.get("srt_model", "qwen3.6-max"),
                run_unise_v2=s2_cfg.get("run_unise_v2", False),
            )
            print(f"[main] Stage 2 completed. Final clips: {final_clips}")

    print(f"[main] Task state saved to: {task_dir / 'task_state.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anime voice training dataset pipeline"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--task-dir", type=str, help="Task folder for checkpointing")
    parser.add_argument("--source-dir", type=str, help="Animation source directory")
    parser.add_argument("--oped-dir", type=str, help="OP/ED directory")
    parser.add_argument("--reference-dir", type=str, help="Reference speaker directory")
    parser.add_argument("--stage", type=str, choices=["1", "2", "all"],
                        help="Run only stage 1, stage 2, or all")
    parser.add_argument("--resume", action="store_true", dest="resume",
                        help="Resume from existing task state")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Do not resume; re-run all enabled stages")
    parser.add_argument("--test-local", action="store_true",
                        help="Run local smoke test with configs/test_local.yaml")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test_local:
        args.config = "configs/test_local.yaml"

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    config["config_path"] = str(config_path)

    # For local smoke test, prepare a source folder containing only test.mp4.
    if args.test_local:
        test_source_dir = Path("/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_test_source")
        test_source_dir.mkdir(parents=True, exist_ok=True)
        test_mp4 = Path("/Users/AITraining/Documents/Personal/train_audio_extract/data/test.mp4")
        if not test_mp4.exists():
            print(f"[main] Local test file not found: {test_mp4}", file=sys.stderr)
            return 1
        target = test_source_dir / "test.mp4"
        if not target.exists():
            import shutil
            shutil.copy(str(test_mp4), str(target))
        config["source_dir"] = str(test_source_dir)
        if not config.get("task_dir"):
            config["task_dir"] = "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_test"

    config = merge_args_into_config(config, args)

    if not config.get("task_dir"):
        raise ValueError("task_dir must be specified in config or via --task-dir")

    try:
        run_pipeline(config)
        return 0
    except Exception as e:
        print(f"[main] Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
