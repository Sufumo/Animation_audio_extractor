"""Persistent task state manager using YAML checkpoints.

Each task folder contains a `task_state.yaml` file that records which pipeline
steps are pending/running/completed/failed, their outputs, and any errors. This
enables resume/retry without re-running already completed work.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


TASK_STATE_FILE = "task_state.yaml"

# Valid status transitions are handled loosely: callers can reset to pending.
VALID_STATUSES = {"pending", "running", "completed", "failed", "skipped"}


class TaskState:
    """Manage per-task checkpoint state stored as YAML."""

    def __init__(self, task_dir: str | Path, config_path: Optional[str | Path] = None):
        self.task_dir = Path(task_dir).resolve()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.task_dir / TASK_STATE_FILE

        if self.state_path.exists():
            self._data = self._load()
        else:
            self._data = self._init_state(config_path)
            self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_state(self, config_path: Optional[str | Path]) -> Dict[str, Any]:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        return {
            "task_id": f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}",
            "created_at": now,
            "updated_at": now,
            "config_path": str(config_path) if config_path else None,
            "inputs": {},
            "stages": {},
        }

    def _load(self) -> Dict[str, Any]:
        with open(self.state_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save(self) -> None:
        self._data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.state_path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _ensure_stage(self, stage: str) -> Dict[str, Any]:
        if stage not in self._data["stages"]:
            self._data["stages"][stage] = {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "outputs": {},
            }
        return self._data["stages"][stage]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def task_id(self) -> str:
        return self._data["task_id"]

    @property
    def data(self) -> Dict[str, Any]:
        """Return a copy of the raw state dictionary."""
        return self._data.copy()

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """Record input paths or metadata for the task."""
        self._data["inputs"] = {k: str(v) if isinstance(v, Path) else v for k, v in inputs.items()}
        self._save()

    def get_inputs(self) -> Dict[str, Any]:
        return self._data.get("inputs", {}).copy()

    def set_config_path(self, config_path: str | Path) -> None:
        self._data["config_path"] = str(config_path)
        self._save()

    def is_done(self, stage: str, step: Optional[str] = None) -> bool:
        """Check whether a stage (or sub-step) is completed."""
        stage_state = self._data.get("stages", {}).get(stage)
        if stage_state is None:
            return False
        if step is None:
            return stage_state.get("status") == "completed"
        steps = stage_state.get("steps", {})
        return steps.get(step, {}).get("status") == "completed"

    def mark_started(self, stage: str, step: Optional[str] = None) -> None:
        """Mark stage/step as running and record start time."""
        stage_state = self._ensure_stage(stage)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if step is None:
            stage_state["status"] = "running"
            stage_state["started_at"] = now
            stage_state["error"] = None
        else:
            stage_state.setdefault("steps", {})[step] = {
                "status": "running",
                "started_at": now,
                "completed_at": None,
                "error": None,
            }
        self._save()

    def mark_done(
        self,
        stage: str,
        step: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark stage/step as completed and optionally record outputs."""
        stage_state = self._ensure_stage(stage)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if step is None:
            stage_state["status"] = "completed"
            stage_state["completed_at"] = now
            stage_state["error"] = None
            if outputs is not None:
                stage_state["outputs"].update(outputs)
        else:
            step_state = stage_state.setdefault("steps", {}).get(step)
            if step_state is None:
                step_state = {}
                stage_state["steps"][step] = step_state
            step_state["status"] = "completed"
            step_state["completed_at"] = now
            step_state["error"] = None
            if outputs is not None:
                step_state["outputs"] = outputs
        self._save()

    def mark_failed(self, stage: str, step: Optional[str] = None, error: Optional[str] = None) -> None:
        """Mark stage/step as failed and record error message."""
        stage_state = self._ensure_stage(stage)
        if step is None:
            stage_state["status"] = "failed"
            stage_state["error"] = str(error) if error else None
        else:
            step_state = stage_state.setdefault("steps", {}).get(step)
            if step_state is None:
                step_state = {}
                stage_state["steps"][step] = step_state
            step_state["status"] = "failed"
            step_state["error"] = str(error) if error else None
        self._save()

    def reset(self, stage: str, step: Optional[str] = None) -> None:
        """Reset stage/step to pending so it can be re-run."""
        stage_state = self._ensure_stage(stage)
        if step is None:
            stage_state["status"] = "pending"
            stage_state["started_at"] = None
            stage_state["completed_at"] = None
            stage_state["error"] = None
        else:
            if "steps" in stage_state and step in stage_state["steps"]:
                stage_state["steps"][step] = {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                }
        self._save()

    def get_stage_status(self, stage: str) -> Optional[str]:
        stage_state = self._data.get("stages", {}).get(stage)
        return stage_state.get("status") if stage_state else None

    def get_outputs(self, stage: str, step: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve recorded outputs for a stage or sub-step."""
        stage_state = self._data.get("stages", {}).get(stage, {})
        if step is None:
            return stage_state.get("outputs", {}).copy()
        return stage_state.get("steps", {}).get(step, {}).get("outputs", {}).copy()

    def add_output(self, stage: str, key: str, value: Any, step: Optional[str] = None) -> None:
        """Append a single output value to a list under `key`."""
        stage_state = self._ensure_stage(stage)
        target = stage_state["outputs"] if step is None else stage_state.setdefault("steps", {}).setdefault(step, {}).setdefault("outputs", {})
        if key not in target:
            target[key] = []
        if not isinstance(target[key], list):
            target[key] = [target[key]]
        target[key].append(str(value) if isinstance(value, Path) else value)
        self._save()

    def list_stages(self) -> List[str]:
        return list(self._data.get("stages", {}).keys())

    def __repr__(self) -> str:
        return f"TaskState(task_dir={self.task_dir}, task_id={self.task_id})"
