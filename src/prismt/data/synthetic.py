"""Synthetic trial data with known structure. The same recipe is implemented in MATLAB.

The data imitate a two-colour widefield experiment: two modalities ("calcium" and "ach")
recorded on a small grid of cortical channels, trials aligned to a stimulus, several mice,
early and late learning sessions. What is planted, and therefore what a working model
should find:

* **Stimulus signal (calcium, "posterior" channels).** On CS+ trials (``stim == 1``) the two
  channels in the bottom-left corner of the grid show an evoked calcium response.
* **Learning signal (ACh, "frontal" channels).** In late sessions the two channels in the
  top-right corner show an evoked acetylcholine response on every trial. ``phase`` (early
  vs late) is constant within a session, so decoding it must generalize across mice.
* **Predictable structure for masked autoencoding.** Every channel mixes three smooth
  spatiotemporal latent factors (spatially smooth loadings on the grid), and ACh is a
  lagged mixture of the same factors, so hidden values are predictable from visible ones
  and ACh is predictable from calcium.
* **Nuisance.** Per-mouse gain, per-session offsets, a class-independent arousal ramp on
  ACh and white noise.
* **An unpredictable channel.** The top-left channel is pure noise in both modalities.
* **Missing data.** Two (mouse, channel) pairs are entirely NaN, and about 1% of
  (trial, channel, modality) traces are NaN.

Everything structural (grid, loadings, response kernel, pattern and NaN channels, time
axis) is deterministic and identical in Python and MATLAB; see :func:`structure`. Random
draws (latent amplitudes, noise, trial order) use each language's own generator, so the
two implementations agree in distribution, not value by value.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

RECIPE_VERSION = 1

PROFILES: dict[str, dict] = {
    "tiny": {"n_mice": 4, "n_sessions": 2, "n_trials": 16, "grid": (2, 3), "n_time": 6},
    "fast": {"n_mice": 8, "n_sessions": 2, "n_trials": 30, "grid": (3, 4), "n_time": 10},
    "tutorial": {"n_mice": 8, "n_sessions": 4, "n_trials": 40, "grid": (4, 6), "n_time": 12},
}

#: Evoked-response amplitude in units of the noise standard deviation.
DIFFICULTY = {"easy": 6.0, "medium": 3.0, "hard": 1.5}

FS_HZ = 10.0
NOISE_SD = 0.35
OFFSET_SD = 0.2  # per (mouse, session, channel, modality)
AROUSAL_SD = 0.3  # class-independent ramp on ACh
OMEGAS = (0.5, 1.0, 1.5)  # latent cycles per trial window
ACH_MIX = (1.0, 0.7, 0.5)  # how the latent factors drive ACh (with a 1-sample lag)
LATENT_CENTERS = ((0.15, 0.2), (0.85, 0.35), (0.45, 0.9))  # (x, y) as fractions of the grid
KERNEL_PEAK_S = 0.3
RANDOM_NAN_FRACTION = 0.01


def structure(profile: str = "fast") -> dict:
    """The deterministic part of the recipe (identical in Python and MATLAB).

    Channel numbers in the returned dict are 1-based, like MATLAB and the dataset file.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}'; choose from {', '.join(PROFILES)}")
    p = PROFILES[profile]
    nrow, ncol = p["grid"]
    R, T = nrow * ncol, p["n_time"]
    rows = np.arange(R) // ncol
    cols = np.arange(R) % ncol
    # Latent factors are centred at three spread-out points of the grid (fractions of its
    # width and height), so their spatial loadings are well separated.
    cx = (ncol - 1) * np.asarray(LATENT_CENTERS)[:, 0]
    cy = (nrow - 1) * np.asarray(LATENT_CENTERS)[:, 1]
    s = max(0.75, min(nrow, ncol) / 3.0)
    loadings = np.exp(-((cols[:, None] - cx[None, :]) ** 2 + (rows[:, None] - cy[None, :]) ** 2) / (2 * s**2))

    onset = T // 4  # 0-based index of the first post-stimulus sample
    times = (np.arange(T) - onset) / FS_HZ
    tau = (np.arange(T) - onset + 1) / FS_HZ
    kernel = np.where(tau > 0, (tau / KERNEL_PEAK_S) * np.exp(1 - tau / KERNEL_PEAK_S), 0.0)
    kernel = kernel / kernel.max()

    def at(row: int, col: int) -> int:
        return row * ncol + col

    pattern_a = [at(nrow - 1, 0), at(nrow - 1, 1)]
    pattern_b = [at(0, ncol - 1), at(0, ncol - 2)]
    noise_channel = at(0, 0)
    reserved = set(pattern_a) | set(pattern_b) | {noise_channel}
    free = [r for r in range(R - 1, -1, -1) if r not in reserved]
    c1 = free[0]
    c2 = free[1] if len(free) > 1 else free[0]
    nan_channels = [(2, c1 + 1), (p["n_mice"], c2 + 1)]  # (mouse number, channel number), 1-based

    return {
        "recipe": RECIPE_VERSION,
        "profile": profile,
        "n_mice": p["n_mice"],
        "n_sessions": p["n_sessions"],
        "n_trials_per_session": p["n_trials"],
        "grid": [nrow, ncol],
        "shape": [p["n_mice"] * p["n_sessions"] * p["n_trials"], R, T, 2],
        "channel_x": (cols + 1).tolist(),
        "channel_y": (rows + 1).tolist(),
        "loadings": np.round(loadings, 6).tolist(),
        "omegas": list(OMEGAS),
        "ach_mix": list(ACH_MIX),
        "onset_index": onset + 1,
        "times_s": np.round(times, 6).tolist(),
        "kernel": np.round(kernel, 6).tolist(),
        "pattern_a": [r + 1 for r in pattern_a],
        "pattern_b": [r + 1 for r in pattern_b],
        "noise_channel": noise_channel + 1,
        "nan_channels": [list(x) for x in nan_channels],
        "columns": ["mouse", "session", "phase", "stim", "response"],
        "modalities": ["calcium", "ach"],
    }


def make_synthetic(profile: str = "fast", difficulty: str = "medium", seed: int = 0) -> dict:
    """Generate the arrays and metadata; the keys match :func:`prismt.io.write_dataset`."""
    if difficulty not in DIFFICULTY:
        raise ValueError(f"unknown difficulty '{difficulty}'; choose from {', '.join(DIFFICULTY)}")
    st = structure(profile)
    rng = np.random.default_rng(seed)
    S, Q, n = st["n_mice"], st["n_sessions"], st["n_trials_per_session"]
    N, R, T, M = st["shape"]
    L = np.asarray(st["loadings"])
    K = L.shape[1]
    kernel = np.asarray(st["kernel"])
    delta = DIFFICULTY[difficulty] * NOISE_SD
    t = np.arange(T)

    mouse_idx = np.repeat(np.arange(S), Q * n)
    session_idx = np.tile(np.repeat(np.arange(Q), n), S)
    late = session_idx >= Q // 2
    stim = np.concatenate([rng.permutation(np.arange(n) % 2) for _ in range(S * Q)])
    p_correct = np.where(late, 0.85, 0.6)
    correct = rng.random(N) < p_correct
    response = np.where(stim == 1, np.where(correct, 1, 0), np.where(correct, 2, 3))

    amp = rng.standard_normal((N, K, 1))
    phase = rng.uniform(0, 2 * math.pi, (N, K, 1))
    f = amp * np.cos(2 * math.pi * np.asarray(OMEGAS)[None, :, None] * t[None, None, :] / T + phase)
    f_lag = np.concatenate([f[:, :, :1], f[:, :, :-1]], axis=2)
    calcium = np.einsum("rk,nkt->nrt", L, f)
    ach = np.einsum("rk,k,nkt->nrt", L, np.asarray(ACH_MIX), f_lag)
    ach = ach + rng.normal(0, AROUSAL_SD, (N, 1, 1)) * (t / max(T - 1, 1))[None, None, :]

    noise_ch = st["noise_channel"] - 1
    calcium[:, noise_ch, :] = 0.0
    ach[:, noise_ch, :] = 0.0

    evoked = kernel[None, :] * (1 + 0.3 * rng.standard_normal((N, 1)))
    for r in (c - 1 for c in st["pattern_a"]):
        calcium[:, r, :] += delta * evoked * (stim == 1)[:, None]
    evoked_b = kernel[None, :] * (1 + 0.3 * rng.standard_normal((N, 1)))
    for r in (c - 1 for c in st["pattern_b"]):
        ach[:, r, :] += delta * evoked_b * late[:, None]

    gain = 1 + 0.15 * rng.standard_normal(S)
    offsets = OFFSET_SD * rng.standard_normal((S, Q, R, M))
    X = np.stack([calcium, ach], axis=-1) * gain[mouse_idx][:, None, None, None]
    X = X + offsets[mouse_idx, session_idx][:, :, None, :]
    X = X + NOISE_SD * rng.standard_normal(X.shape)

    for mouse, channel in st["nan_channels"]:
        X[mouse_idx == mouse - 1, channel - 1, :, :] = np.nan
    reserved = set(st["pattern_a"]) | set(st["pattern_b"])
    eligible = np.array([r for r in range(R) if (r + 1) not in reserved])
    n_random = int(round(RANDOM_NAN_FRACTION * N * len(eligible) * M))
    for _ in range(n_random):
        X[rng.integers(N), rng.choice(eligible), :, rng.integers(M)] = np.nan

    mouse_ids = np.array([f"M{m + 1:02d}" for m in mouse_idx], dtype=object)
    phases = np.where(late, "late", "early").astype(object)
    provenance = {
        "synthetic": {
            "recipe": RECIPE_VERSION,
            "profile": profile,
            "difficulty": difficulty,
            "seed": int(seed),
            "pattern_a": st["pattern_a"],
            "pattern_b": st["pattern_b"],
            "noise_channel": st["noise_channel"],
            "nan_channels": st["nan_channels"],
            "onset_index": st["onset_index"],
            "description": "Stimulus signal: calcium in pattern_a on CS+ trials. Learning signal: "
            "ACh in pattern_b in late sessions. noise_channel is unpredictable.",
        }
    }
    return {
        "X": X.astype(np.float32),
        "trials": {
            "mouse": mouse_ids,
            "session": (session_idx + 1).astype(float),
            "phase": phases,
            "stim": stim.astype(float),
            "response": response.astype(float),
        },
        "channel_names": [f"ch{r + 1:02d}" for r in range(R)],
        "channel_x": [float(v) for v in st["channel_x"]],
        "channel_y": [float(v) for v in st["channel_y"]],
        "modality_names": ["calcium", "ach"],
        "modality_units": ["dF/F", "dF/F"],
        "modality_kinds": ["neural", "neural"],
        "times_s": st["times_s"],
        "event": "stimulus onset",
        "subject": "mouse",
        "session": "session",
        "value_labels": {
            "stim": {0.0: "CS-", 1.0: "CS+"},
            "response": {0.0: "miss", 1.0: "hit", 2.0: "CR", 3.0: "FA"},
        },
        "categories": {"phase": ["early", "late"]},
        "provenance": provenance,
    }


def write_synthetic(path: str | Path, profile: str = "fast", difficulty: str = "medium", seed: int = 0) -> Path:
    from prismt.io import write_dataset

    data = make_synthetic(profile, difficulty, seed)
    X = data.pop("X")
    trials = data.pop("trials")
    return write_dataset(path, X, trials, **data)
