"""Target Speaker Extraction (TSE) using QuarkAudio-UniSE.

Wraps the external unified-audio/QuarkAudio-UniSE project. Long audio is split
into chunks (default 360s) to avoid OOM, processed with the same reference
enrollment for each chunk, then merged back.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.audio_utils import convert_to_wav, get_duration, merge_audio, split_audio


DEFAULT_SEGMENT_SECONDS = 360.0
DEFAULT_ENROLL_DURATION = 5.0


def _get_compatible_test_script(unise_dir: Path, tmp_dir: Path) -> Path:
    """Return a PyTorch 2.6+ compatible copy of UniSE test.py placed in unise_dir.

    The copy is placed inside unise_dir so that relative imports (`from model ...`)
    still work. The original test.py is never modified. The caller should remove
    the patched file when done.
    """
    original = unise_dir / "test.py"
    # Place patched copy in unise_dir so imports resolve.
    patched = unise_dir / f"test_patched_{tmp_dir.name}.py"
    source = original.read_text(encoding="utf-8")

    # If already patched or original already passes weights_only=False, use original.
    if "weights_only=False" in source:
        return original

    # Match the trainer.test call and inject weights_only=False.
    pattern = r"(trainer\.test\([^)]+ckpt_path=config\['ckpt_path'\])\)"
    replacement = r"\1, weights_only=False)"
    new_source = re.sub(pattern, replacement, source)

    if "weights_only=False" not in new_source:
        # Fallback: replace the exact original call.
        new_source = source.replace(
            "trainer.test(model, data_module, ckpt_path=config['ckpt_path'])",
            "trainer.test(model, data_module, ckpt_path=config['ckpt_path'], weights_only=False)",
        )

    patched.write_text(new_source, encoding="utf-8")
    return patched


def _build_unise_config(
    unise_dir: Path,
    mix_dir: Path,
    enroll_dir: Path,
    tgt_dir: Path,
    ckpt_path: Path,
    accelerator: str = "auto",
    devices: int = 1,
    enroll_duration: float = DEFAULT_ENROLL_DURATION,
) -> Path:
    """Write a temporary UniSE config YAML for TSE inference."""
    base_config_path = unise_dir / "conf" / "config.yaml"
    if base_config_path.exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config.update({
        "accelerator": accelerator,
        "devices": devices,
        "ckpt_path": str(ckpt_path),
        "dataset_config": {
            "train_kwargs": {},
            "val_kwargs": {},
            "test_kwargs": {
                "batch_size": 1,
                "num_workers": 1,
                "prefetch": 1,
                "mode": "tse",
                "data_enroll_dir": str(enroll_dir),
                "enroll_duration": enroll_duration,
                "data_src_dir": str(mix_dir),
                "data_tgt_dir": str(tgt_dir),
            },
        },
    })

    config_path = mix_dir.parent / "unise_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


def run_unise_tse(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    unise_dir: str | Path,
    ckpt_path: str | Path,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    enroll_duration: float = DEFAULT_ENROLL_DURATION,
    accelerator: str = "auto",
    devices: int = 1,
) -> Path:
    """Run UniSE target speaker extraction on an audio file.

    Args:
        input_path: Mixed audio/video file.
        reference_path: Reference enrollment audio for the target speaker.
        output_path: Where to save the extracted vocals WAV.
        unise_dir: Path to unified-audio/QuarkAudio-UniSE project root.
        ckpt_path: Path to UniSE checkpoint.
        segment_seconds: Chunk length for long audio.
        enroll_duration: Reference clip length used by UniSE.
        accelerator: PyTorch Lightning accelerator.
        devices: Number of devices.

    Returns:
        Path to the extracted audio WAV.
    """
    input_path = Path(input_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    unise_dir = Path(unise_dir)
    ckpt_path = Path(ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_script = unise_dir / "test.py"
    if not test_script.exists():
        raise FileNotFoundError(f"UniSE test.py not found: {test_script}")

    with tempfile.TemporaryDirectory(prefix="unise_tse_") as tmp:
        tmp_dir = Path(tmp)
        input_wav = tmp_dir / "input.wav"
        convert_to_wav(input_path, input_wav, sample_rate=16000, mono=True, bit_depth=16)

        duration = get_duration(input_wav)
        if duration <= segment_seconds:
            chunks = [input_wav]
        else:
            chunk_dir = tmp_dir / "chunks"
            chunks = split_audio(
                input_wav,
                chunk_dir,
                segment_seconds=segment_seconds,
                prefix="mix",
                sample_rate=16000,
                mono=True,
                bit_depth=16,
            )

        # Prepare reference enrollment once.
        ref_wav = tmp_dir / "reference.wav"
        convert_to_wav(reference_path, ref_wav, sample_rate=16000, mono=True, bit_depth=16)

        mix_dir = tmp_dir / "mix"
        enroll_dir = tmp_dir / "enroll"
        tgt_dir = tmp_dir / "tgt"
        out_dir = tmp_dir / "output"
        for d in (mix_dir, enroll_dir, tgt_dir, out_dir):
            d.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            name = chunk.name
            shutil.copy(str(chunk), str(mix_dir / name))
            shutil.copy(str(ref_wav), str(enroll_dir / name))
            shutil.copy(str(chunk), str(tgt_dir / name))

        config_path = _build_unise_config(
            unise_dir=unise_dir,
            mix_dir=mix_dir,
            enroll_dir=enroll_dir,
            tgt_dir=tgt_dir,
            ckpt_path=ckpt_path,
            accelerator=accelerator,
            devices=devices,
            enroll_duration=enroll_duration,
        )

        patched_test_script = _get_compatible_test_script(unise_dir, tmp_dir)
        try:
            cmd = [
                sys.executable,
                str(patched_test_script),
                "--config", str(config_path),
                "--save_enhanced", str(out_dir),
            ]
            subprocess.run(cmd, cwd=str(unise_dir), check=True)
        finally:
            # Clean up the patched copy if we created one.
            if patched_test_script != test_script and patched_test_script.exists():
                patched_test_script.unlink()

        enhanced_files = sorted(out_dir.glob("*.wav"))
        if not enhanced_files:
            raise RuntimeError("UniSE did not produce any output WAV files")

        # UniSE may prefix output names; sort by filename and merge.
        merge_audio(enhanced_files, output_path, concat_with_copy=True)

    return output_path
