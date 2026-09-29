"""Strict JSON for files that MATLAB reads.

MATLAB's ``jsondecode`` cannot read ``NaN`` or ``Infinity``, so every value is converted to
plain Python first and non-finite numbers become ``null``. Files are written atomically
(temporary file, then rename) so a reader never sees half a file.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert numpy values, paths, dataclasses, sets and tuples; NaN and Inf become None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()] if obj.ndim else to_jsonable(obj.item())
    if isinstance(obj, Path):
        return obj.as_posix()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    raise TypeError(f"Cannot convert {type(obj).__name__} to JSON")


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_jsonable(obj), allow_nan=False, indent=indent, ensure_ascii=False)


def atomic_write_text(path: str | os.PathLike, text: str, *, retries: int = 20) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same folder.

    ``os.replace`` can fail on Windows while another process (MATLAB polling the status
    file) has the target open, so it is retried for up to about a second.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, dumps(obj) + "\n")


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_list(value: Any) -> list:
    """MATLAB's ``jsonencode`` writes a one-element list as a scalar; undo that."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
