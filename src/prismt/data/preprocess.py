"""Time window, time binning, baseline subtraction and scaling.

Scaling statistics are computed from training trials only and stored with the model, so
validation and test trials never influence preprocessing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from prismt.errors import ConfigError
from prismt.io.dataset import PrismtDataset

_TITLE = "The time settings do not work with this dataset"


@dataclass(frozen=True)
class TimeBins:
    centers_s: np.ndarray  # time of each bin (seconds)
    members: tuple[np.ndarray, ...]  # sample indices averaged into each bin
    width_s: float | None

    @property
    def n(self) -> int:
        return len(self.members)


def make_time_bins(times_s: np.ndarray, window: list[float] | None, bin_width: float | None) -> TimeBins:
    times = np.asarray(times_s, dtype=float)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    tol = 1e-6 * max(1.0, abs(step))
    if window is None:
        start, end = times[0], times[-1]
        inside = np.arange(len(times))
    else:
        start, end = window
        inside = np.flatnonzero((times >= start - tol) & (times <= end + tol))
        if inside.size == 0:
            raise ConfigError("E_CFG_WINDOW", f"No samples fall in the time window [{start}, {end}] s.",
                              hint=f"The trials span {times[0]:.3g} to {times[-1]:.3g} s.",
                              field="preprocess.time_window_s", title=_TITLE)
    if bin_width is None:
        return TimeBins(times[inside], tuple(np.array([i]) for i in inside), None)
    if bin_width < step - tol:
        raise ConfigError("E_CFG_BIN", f"The bin width ({bin_width} s) is smaller than the sampling interval "
                          f"({step:.3g} s).", hint="Use a bin width of at least one sample, or leave it empty.",
                          field="preprocess.bin_width_s", title=_TITLE)
    first = times[inside[0]]
    k = np.floor((times[inside] - first + tol) / bin_width).astype(int)
    members, centers = [], []
    for b in range(int(k.max()) + 1):
        idx = inside[k == b]
        if idx.size:
            members.append(idx)
            centers.append(float(times[idx].mean()))
    return TimeBins(np.asarray(centers), tuple(members), float(bin_width))


def extract(ds: PrismtDataset, trial_index: np.ndarray, channels: np.ndarray, modalities: np.ndarray,
            pairs: list[tuple[int, int]], bins: TimeBins, *, baseline_subtract: bool = False) -> np.ndarray:
    """Selected trials as float32 [n, channels, bins, modalities]; NaN outside modality channel sets."""
    X = np.take(np.take(np.take(ds.X, trial_index, axis=0), channels, axis=1), modalities, axis=3)
    exists = np.zeros((len(channels), len(modalities)), dtype=bool)
    for c, m in pairs:
        exists[c, m] = True
    X = np.where(exists[None, :, None, :], X, np.nan).astype(np.float32)
    if baseline_subtract:
        pre = ds.times_s < 0
        if not pre.any():
            raise ConfigError("E_CFG_BASELINE", "Baseline subtraction needs samples before time 0, but there are none.",
                              field="preprocess.baseline_subtract", title=_TITLE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            base = np.nanmean(X[:, :, pre, :], axis=2, keepdims=True)
        X = X - np.nan_to_num(base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of an all-missing bin is NaN, as wanted
        binned = np.stack([np.nanmean(X[:, :, idx, :], axis=2) for idx in bins.members], axis=2)
    return binned.astype(np.float32)


@dataclass
class Normalizer:
    method: str
    mean: np.ndarray  # [channels, modalities]
    std: np.ndarray
    clip_sd: float | None

    @classmethod
    def fit(cls, X: np.ndarray, train_rows: np.ndarray, method: str, clip_sd: float | None) -> "Normalizer":
        R, M = X.shape[1], X.shape[3]
        if method == "none":
            return cls(method, np.zeros((R, M), np.float32), np.ones((R, M), np.float32), clip_sd)
        train = X[train_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(train, axis=(0, 2))
            std = np.nanstd(train, axis=(0, 2))
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(method, mean.astype(np.float32), std.astype(np.float32), clip_sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean[None, :, None, :]) / self.std[None, :, None, :]
        if self.clip_sd:
            Z = np.clip(Z, -self.clip_sd, self.clip_sd)
        return Z.astype(np.float32)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return (Z * self.std[None, :, None, :] + self.mean[None, :, None, :]).astype(np.float32)

    def to_state(self) -> dict:
        return {"method": self.method, "mean": self.mean.tolist(), "std": self.std.tolist(), "clip_sd": self.clip_sd}

    @classmethod
    def from_state(cls, state: dict) -> "Normalizer":
        return cls(state["method"], np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32),
                   state.get("clip_sd"))
