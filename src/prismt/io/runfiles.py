"""The run folder: what Python writes, and MATLAB reads, during and after a run.

``status.json`` is rewritten atomically as training progresses (MATLAB polls it),
``history.csv`` grows by one row per epoch, and the final results are written as JSON,
CSV and a ``results.mat`` that MATLAB's ``load`` opens directly. A ``STOP`` file created
in the folder asks the run to stop after the current step, keeping the best model so far.
"""

from __future__ import annotations

import csv
import io
import os
import platform
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from prismt.errors import ConfigError
from prismt.jsonutil import atomic_write_text, to_jsonable, write_json

STATUS_SCHEMA = "prismt.status/1"
STATES = ("queued", "preparing", "training", "evaluating", "finished", "failed", "cancelled")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:40]


def new_run_name(task: str, name: str | None = None) -> str:
    """``<task>-<YYYYmmdd-HHMMSS>-<random>[-name]``: never built from parameters, so never collides."""
    base = f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    return f"{base}-{_safe(name)}" if name else base


def create_run_dir(root: str | Path, task: str, name: str | None = None, *, overwrite: bool = False) -> Path:
    path = Path(root).expanduser() / new_run_name(task, name)
    return ensure_run_dir(path, overwrite=overwrite)


def ensure_run_dir(path: str | Path, *, overwrite: bool = False, allow_prepared: bool = True) -> Path:
    """Use ``path`` as the run folder. A folder that already holds results is never reused
    unless ``overwrite`` is set; a folder prepared by MATLAB (only run.json, no results) is fine."""
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        done = any((path / f).exists() for f in ("metrics.json", "model.pt", "history.csv"))
        if done or not allow_prepared:
            raise ConfigError("E_RUN_EXISTS", f"The run folder {path} already contains results.",
                              hint="Choose a new run name; runs are never overwritten unless 'Overwrite' is on.",
                              field="output.overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stop_requested(run_dir: str | Path | None) -> bool:
    return run_dir is not None and (Path(run_dir) / "STOP").exists()


class StatusWriter:
    """Keeps ``status.json`` current. Progress updates are throttled; state changes are immediate."""

    def __init__(self, path: str | Path, *, min_interval: float = 2.0, **base: Any) -> None:
        self.path = Path(path)
        self.min_interval = min_interval
        self.start = time.time()
        self._last = 0.0
        self.data: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "state": "queued",
            "message": "",
            "pid": os.getpid(),
            "host": platform.node(),
            "started": utc_now(),
            "error": None,
            **base,
        }

    def update(self, state: str | None = None, *, force: bool = False, **fields: Any) -> None:
        now = time.time()
        self.data.update(fields)
        if state is not None:
            if state not in STATES:
                raise ValueError(f"unknown run state {state}")
            changed = state != self.data.get("state")
            self.data["state"] = state
            force = force or changed
        if not force and now - self._last < self.min_interval:
            return
        self.data["updated"] = utc_now()
        self.data["elapsed_s"] = round(now - self.start, 1)
        write_json(self.path, self.data)
        self._last = now

    def fail(self, error: dict, state: str = "failed") -> None:
        self.update(state, force=True, error=error, message=error.get("title", "The run failed"), eta_s=None)


class HistoryWriter:
    """``history.csv``: one row per epoch (and fold). Rewritten atomically, header first."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append(dict(row))
        write_csv(self.path, self.rows)


def write_csv(path: str | Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: _csv_value(r.get(k)) for k in columns})
    atomic_write_text(path, buf.getvalue())


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)  # 1/0: MATLAB's readtable reads numbers, not "True"/"False" text
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


def write_results_mat(path: str | Path, data: dict) -> None:
    """Save ``data`` so that MATLAB's ``load`` gives numbers, logicals, cell arrays of text and structs."""
    from scipy.io import savemat

    tmp = Path(path).with_name(Path(path).name + ".tmp.mat")
    savemat(tmp, {_key(k): _matlab_value(v) for k, v in data.items()}, do_compression=True,
            oned_as="column", long_field_names=True)
    os.replace(tmp, path)


def _key(k: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", str(k))
    if not key or not key[0].isalpha():
        key = "x" + key
    return key[:63]


def _matlab_value(v: Any) -> Any:
    if v is None:
        return np.zeros((0, 0))
    if isinstance(v, dict):
        return {_key(k): _matlab_value(x) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return np.zeros((0, 1))
        if all(isinstance(x, str) for x in v):
            return np.array(list(v), dtype=object)
        if all(isinstance(x, (bool, np.bool_)) for x in v):
            return np.array(v, dtype=bool)
        if all(x is None or isinstance(x, (int, float, np.integer, np.floating)) for x in v):
            return np.array([np.nan if x is None else float(x) for x in v])
        out = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            out[i] = _matlab_value(x)
        return out
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            out = np.empty(v.shape, dtype=object)
            for i, x in np.ndenumerate(v):
                out[i] = "" if x is None else (x if isinstance(x, str) else _matlab_value(x))
            return out
        return v
    return _matlab_value(to_jsonable(v))
