"""PRISMT: transformers for trial-structured neural and behavioral data.

The package is driven from MATLAB (``run_prismt_gui``) or from the command line
(``python -m prismt --help``). Importing it never imports torch, so the data and
configuration tools stay fast and ``prismt doctor`` can report a broken PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"

__version__ = (RESOURCES / "VERSION").read_text(encoding="utf-8").strip()

#: Version of every file format shared with MATLAB (see docs/dev/contracts.md).
FORMATS: dict[str, int] = json.loads((RESOURCES / "formats.json").read_text(encoding="utf-8"))

__all__ = ["FORMATS", "RESOURCES", "__version__"]
