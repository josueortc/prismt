"""Shared test fixtures. Tests never touch real data; they build synthetic datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    from prismt.data.synthetic import write_synthetic

    return write_synthetic(tmp_path / "tiny.mat", "tiny", "easy", seed=0)


@pytest.fixture
def fast_dataset(tmp_path: Path) -> Path:
    from prismt.data.synthetic import write_synthetic

    return write_synthetic(tmp_path / "fast.mat", "fast", "medium", seed=0)


def value_coded(N: int, R: int, T: int, M: int) -> np.ndarray:
    """X(n,r,t,m) = 1000n + 100r + 10t + m with 1-based indices, so any axis mix-up shows."""
    n, r, t, m = np.meshgrid(np.arange(1, N + 1), np.arange(1, R + 1), np.arange(1, T + 1),
                             np.arange(1, M + 1), indexing="ij")
    return (1000 * n + 100 * r + 10 * t + m).astype(np.float32)


def matlab_executable() -> str | None:
    exe = os.environ.get("PRISMT_MATLAB")
    return exe if exe and Path(exe).exists() else None
