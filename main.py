"""Main entry point for the anime voice training pipeline.

Supports:
- Config-driven stage/component execution.
- Stage2 mode: ``v2`` (default, VAD+ECAPA) or ``aliyun`` (legacy).
- Per-task YAML checkpointing / resume.
- Local smoke test and cache-reuse modes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task_state import TaskState
from src.pipeline.stage1_data_cleaning import run_stage1
from src.pipeline.stage2_speaker_extraction import run_stage2
from src.audio_utils import cut_segment


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_args_into_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.source_dir:
        config["source_dir"] = args.source_dir
    if args.oped_dir:
        config["oped_dir"] = args.oped_dir
    if args.reference_dir:
        config["reference_dir"] = args.reference_dir
    if args.task_dir:
        config["task_dir"] = args.task_dir
    if args.stage is not None:
        config.setdefault("stage1", {})["enabled"] = args.stage in ("1", "all")
        config.setdefault("stage2", {})["enabled"] = args.stage in ("2", "all")
    if getattr(args, "mode", None):
        config.setdefault("stage2", {})["mode"] = args.mode
    if args.resume is not None:
        config["resume"] = args.resume
    return config


def _set_aliyun_key(config: Dict[str, Any]) -> None:
    key = config.get("aliyun", {}).get("dashscope_api_key")
    if key:
        os.environ["DASHSCOPE_API_KEY"] = key


def run_pipeline(
    config: Dict[str, Any],
    cached_tse_paths: Optional[List[Path]] = None,
    preloaded_cleaned_wavs: Optional[List[Path]] = None,
) -> None:
    task_dir = Path(config["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)

    task_state = TaskState(task_dir, config_path=config.get("config_path"))
    task_state.set_config_path(config.get("config_path"))

    source_dir = config.get("source_dir")
    oped_dir = config.get("oped_dir")
    reference_dir = config.get("reference_dir")
    output_dir = task_dir / config.get("output_dir", "output")
    resume = config.get("resume", True)

    _set_aliyun_key(config)

    cleaned_wavs: List[Path] = list(preloaded_cleaned_wavs or [])

    if config.get("stage1", {}).get("enabled", True):
        if not source_dir or not Path(source_dir).exists():
            raise FileNotFoundError(f"source_dir not found: {source_dir}")
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
            s2_cfg = config["stage2"]
            mode = s2_cfg.get("mode", "v2")
            print(f"[main] Running Stage 2 (mode={mode})...")
            final_clips = run_stage2(
                task_state=task_state,
                cleaned_wavs=cleaned_wavs,
                reference_dir=reference_dir,
                output_dir=output_dir / "stage2",
                components=s2_cfg.get("components"),
                mode=mode,
                unise_dir=config.get("unise", {}).get("project_dir"),
                unise_ckpt_path=config.get("unise", {}).get("ckpt_path"),
                segment_seconds=s2_cfg.get("segment_seconds", 360.0),
                asr_speaker_count=s2_cfg.get("asr_speaker_count"),
                srt_model=s2_cfg.get("srt_model", "qwen3.6-max"),
                run_unise_v2=s2_cfg.get("run_unise_v2", False),
                asr_gap_sec=s2_cfg.get("asr_gap_sec", 2.0),
                clip_padding_sec=s2_cfg.get("clip_padding_sec", 0.1),
                min_clip_duration_sec=s2_cfg.get("min_clip_duration_sec", 1.5),
                merge_gap_sec=s2_cfg.get("merge_gap_sec", 2.0),
                speaker_embedding_model=s2_cfg.get(
                    "speaker_embedding_model",
                    "pyannote/wespeaker-voxceleb-resnet34-LM",
                ),
                speaker_verify_samples=s2_cfg.get("speaker_verify_samples", 8),
                max_clip_duration_sec=s2_cfg.get("max_clip_duration_sec", 12.0),
                vad_method=s2_cfg.get("vad_method", "auto"),
                vad_top_db=s2_cfg.get("vad_top_db", 35.0),
                vad_min_silence_sec=s2_cfg.get("vad_min_silence_sec", 0.3),
                speaker_threshold=s2_cfg.get("speaker_threshold", 0.45),
                speaker_backend=s2_cfg.get("speaker_backend", "ecapa"),
                speaker_device=s2_cfg.get("speaker_device", "cpu"),
                normalize_loudness=s2_cfg.get("normalize_loudness", True),
                target_lufs=s2_cfg.get("target_lufs", -23.0),
                skip_asr=s2_cfg.get("skip_asr", False),
                cached_tse_paths=cached_tse_paths,
            )
            print(f"[main] Stage 2 completed. Final clips: {len(final_clips)}")
            for p in final_clips[:10]:
                print(f"  - {p}")
            if len(final_clips) > 10:
                print(f"  ... and {len(final_clips) - 10} more")

    print(f"[main] Task state saved to: {task_dir / 'task_state.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime voice training dataset pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task-dir", type=str)
    parser.add_argument("--source-dir", type=str)
    parser.add_argument("--oped-dir", type=str)
    parser.add_argument("--reference-dir", type=str)
    parser.add_argument("--stage", type=str, choices=["1", "2", "all"])
    parser.add_argument(
        "--mode", type=str, choices=["v2", "v1", "aliyun", "legacy"],
        help="Stage2 mode override (default from config: v2)",
    )
    parser.add_argument("--resume", action="store_true", dest="resume", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=None)
    parser.add_argument("--test-local", action="store_true")
    parser.add_argument("--test-local-full", action="store_true")
    parser.add_argument(
        "--test-from-cache",
        action="store_true",
        help="v2 Stage2 using existing test/task TSE+cleaned (no UniSE recompute)",
    )
    parser.add_argument("--test-trim-seconds", type=float, default=60.0)
    return parser


def _prepare_local_test_source(trim_seconds: Optional[float] = None) -> Path:
    test_source_dir = Path(
        "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_test_source"
    )
    test_source_dir.mkdir(parents=True, exist_ok=True)
    test_mp4 = Path("/Users/AITraining/Documents/Personal/train_audio_extract/data/test.mp4")
    if not test_mp4.exists():
        raise FileNotFoundError(f"Local test file not found: {test_mp4}")
    target = test_source_dir / "test.mp4"
    if trim_seconds and trim_seconds > 0:
        cut_segment(test_mp4, target, 0.0, trim_seconds)
        print(f"[main] Trimmed test.mp4 to first {trim_seconds}s.")
    elif not target.exists():
        shutil.copy(str(test_mp4), str(target))
    return test_source_dir


def _prepare_local_test_reference() -> Path:
    ref_dir = Path(
        "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_test_reference"
    )
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_src = Path("/Users/AITraining/Documents/Personal/train_audio_extract/data/reference.wav")
    if not ref_src.exists():
        raise FileNotFoundError(f"Local reference file not found: {ref_src}")
    ref_dst = ref_dir / "reference.wav"
    if not ref_dst.exists():
        shutil.copy(str(ref_src), str(ref_dst))
    return ref_dir


def _prepare_cache_test(config: Dict[str, Any], trim_seconds: float):
    cache = config.get("cache") or {}
    cleaned_src = Path(cache.get(
        "cleaned_wav",
        "/Users/AITraining/Documents/Personal/train_audio_extract/test/task/stage1/03_cleaned/test_no_oped_cleaned.wav",
    ))
    tse_src = Path(cache.get(
        "tse_wav",
        "/Users/AITraining/Documents/Personal/train_audio_extract/test/task/stage2/ep000_tse_v1.wav",
    ))
    if not cleaned_src.exists():
        raise FileNotFoundError(f"Cached cleaned wav not found: {cleaned_src}")
    if not tse_src.exists():
        raise FileNotFoundError(f"Cached TSE wav not found: {tse_src}")

    work = Path(
        "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_v2_cache_source"
    )
    work.mkdir(parents=True, exist_ok=True)
    cleaned_dst = work / "cleaned_trim.wav"
    tse_dst = work / "tse_trim.wav"
    if trim_seconds and trim_seconds > 0:
        print(f"[main] Trimming cache audio to first {trim_seconds}s ...")
        cut_segment(cleaned_src, cleaned_dst, 0.0, trim_seconds)
        cut_segment(tse_src, tse_dst, 0.0, trim_seconds)
    else:
        shutil.copy(str(cleaned_src), str(cleaned_dst))
        shutil.copy(str(tse_src), str(tse_dst))

    try:
        config["reference_dir"] = str(_prepare_local_test_reference())
    except FileNotFoundError:
        pass
    config["source_dir"] = str(work)
    if not config.get("task_dir"):
        config["task_dir"] = (
            "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_v2_cache_test"
        )
    return [cleaned_dst], [tse_dst]


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test_from_cache:
        args.config = "configs/test_from_cache.yaml"
    elif args.test_local_full:
        args.config = "configs/test_local_full.yaml"
    elif args.test_local:
        args.config = "configs/test_local.yaml"

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    config["config_path"] = str(config_path)

    cached_tse_paths = None
    preloaded_cleaned = None

    if args.test_from_cache:
        try:
            preloaded_cleaned, cached_tse_paths = _prepare_cache_test(
                config, trim_seconds=args.test_trim_seconds,
            )
        except FileNotFoundError as e:
            print(f"[main] {e}", file=sys.stderr)
            return 1
    elif args.test_local or args.test_local_full:
        try:
            trim = args.test_trim_seconds if args.test_local_full else None
            config["source_dir"] = str(_prepare_local_test_source(trim_seconds=trim))
        except FileNotFoundError as e:
            print(f"[main] {e}", file=sys.stderr)
            return 1
        if args.test_local_full:
            try:
                config["reference_dir"] = str(_prepare_local_test_reference())
            except FileNotFoundError as e:
                print(f"[main] {e}", file=sys.stderr)
                return 1
        if not config.get("task_dir"):
            config["task_dir"] = (
                "/Users/AITraining/Documents/Personal/train_audio_extract/data/anime_pipeline_test"
            )

    config = merge_args_into_config(config, args)
    if not config.get("task_dir"):
        raise ValueError("task_dir must be specified in config or via --task-dir")

    try:
        run_pipeline(
            config,
            cached_tse_paths=cached_tse_paths,
            preloaded_cleaned_wavs=preloaded_cleaned,
        )
        return 0
    except Exception as e:
        print(f"[main] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
