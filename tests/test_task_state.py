"""Tests for src.task_state."""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from src.task_state import TaskState


@pytest.fixture
def tmp_task_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def test_creates_initial_state(tmp_task_dir):
    ts = TaskState(tmp_task_dir, config_path="configs/default.yaml")
    assert (tmp_task_dir / "task_state.yaml").exists()
    assert ts.task_id.startswith("20")
    assert ts.get_inputs() == {}
    assert ts.get_stage_status("stage1") is None


def test_set_inputs_and_config(tmp_task_dir):
    ts = TaskState(tmp_task_dir)
    ts.set_inputs({"source_dir": "/data/source", "oped_dir": Path("/data/oped")})
    ts.set_config_path("configs/test.yaml")
    data = ts.data
    assert data["inputs"]["source_dir"] == "/data/source"
    assert data["inputs"]["oped_dir"] == "/data/oped"
    assert data["config_path"] == "configs/test.yaml"


def test_stage_lifecycle(tmp_task_dir):
    ts = TaskState(tmp_task_dir)
    assert not ts.is_done("stage1")

    ts.mark_started("stage1")
    assert ts.get_stage_status("stage1") == "running"

    ts.mark_done("stage1", outputs={"cleaned_wavs": ["a.wav"]})
    assert ts.is_done("stage1")
    assert ts.get_outputs("stage1")["cleaned_wavs"] == ["a.wav"]


def test_sub_step_lifecycle(tmp_task_dir):
    ts = TaskState(tmp_task_dir)
    ts.mark_started("stage1", step="mp4_to_wav")
    assert ts.get_stage_status("stage1") == "pending"  # parent stays pending

    ts.mark_done("stage1", step="mp4_to_wav", outputs={"wavs": ["a.wav"]})
    assert ts.is_done("stage1", step="mp4_to_wav")
    assert ts.get_outputs("stage1", step="mp4_to_wav")["wavs"] == ["a.wav"]


def test_failure_and_reset(tmp_task_dir):
    ts = TaskState(tmp_task_dir)
    ts.mark_started("stage2")
    ts.mark_failed("stage2", error="OOM")
    assert ts.get_stage_status("stage2") == "failed"
    assert "OOM" in ts.data["stages"]["stage2"]["error"]

    ts.reset("stage2")
    assert ts.get_stage_status("stage2") == "pending"


def test_persists_across_instances(tmp_task_dir):
    ts1 = TaskState(tmp_task_dir)
    ts1.mark_done("stage1", outputs={"x": 1})

    ts2 = TaskState(tmp_task_dir)
    assert ts2.is_done("stage1")
    assert ts2.get_outputs("stage1")["x"] == 1


def test_add_output_appends(tmp_task_dir):
    ts = TaskState(tmp_task_dir)
    ts.add_output("stage1", "files", "a.wav")
    ts.add_output("stage1", "files", Path("b.wav"))
    assert ts.get_outputs("stage1")["files"] == ["a.wav", "b.wav"]


def test_yaml_is_human_readable(tmp_task_dir):
    ts = TaskState(tmp_task_dir, config_path="configs/default.yaml")
    ts.set_inputs({"source_dir": "/src"})
    ts.mark_done("stage1", outputs={"wav": "out.wav"})

    raw = yaml.safe_load((tmp_task_dir / "task_state.yaml").read_text())
    assert raw["config_path"] == "configs/default.yaml"
    assert raw["inputs"]["source_dir"] == "/src"
    assert raw["stages"]["stage1"]["status"] == "completed"
