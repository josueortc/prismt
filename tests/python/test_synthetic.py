"""The synthetic recipe: deterministic structure and learnability (checked with simple
estimators, before any transformer is trained)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score

from prismt.data.synthetic import PROFILES, make_synthetic, structure
from prismt.io import read_dataset


def test_structure_matches_the_golden_fixture(fixtures_dir):
    golden = json.loads((fixtures_dir / "synthetic_structure_v1.json").read_text())["profiles"]
    for profile in PROFILES:
        got = structure(profile)
        want = golden[profile]
        assert got.keys() == want.keys()
        for key in want:
            if isinstance(want[key], list) and want[key] and isinstance(want[key][0], (float, list)):
                np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(want[key], float), atol=1e-6)
            else:
                assert got[key] == want[key], key


def test_planted_channels_and_nans(tiny_dataset):
    ds = read_dataset(tiny_dataset)
    st = structure("tiny")
    mouse = ds.columns["mouse"].as_text()
    for m, ch in st["nan_channels"]:
        assert np.isnan(ds.X[mouse == f"M{m:02d}", ch - 1]).all()
    for ch in st["pattern_a"] + st["pattern_b"]:
        assert not np.isnan(ds.X[mouse == "M01", ch - 1]).any()
    assert ds.meta["provenance"]["synthetic"]["pattern_a"] == st["pattern_a"]


def test_stim_is_balanced_within_every_session():
    d = make_synthetic("fast", "medium", 3)
    keys = np.char.add(d["trials"]["mouse"].astype(str), d["trials"]["session"].astype(int).astype(str))
    for k in np.unique(keys):
        s = d["trials"]["stim"][keys == k]
        assert s.sum() == len(s) // 2


def _split(d):
    mouse = d["trials"]["mouse"]
    test = np.isin(mouse, sorted(set(mouse))[-2:])
    return ~test, test


def _decode(profile: str, difficulty: str, seed: int, label: str) -> float:
    d = make_synthetic(profile, difficulty, seed)
    st = structure(profile)
    X = d["X"].astype(float)
    on = st["onset_index"] - 1
    train, test = _split(d)
    Xb = X - np.nanmean(X[:, :, :on, :], axis=2, keepdims=True)
    F = np.nan_to_num(Xb[:, :, on:, :].reshape(len(X), -1))
    y = d["trials"]["stim"].astype(int) if label == "stim" else (d["trials"]["phase"] == "late").astype(int)
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(F[train], y[train])
    return balanced_accuracy_score(y[test], clf.predict(F[test]))


@pytest.mark.parametrize("label", ["stim", "phase"])
def test_labels_are_decodable_on_new_mice_at_medium_difficulty(label):
    accs = [_decode("fast", "medium", s, label) for s in range(3)]
    assert np.mean(accs) >= 0.85, accs


def test_difficulty_orders_decodability():
    easy = np.mean([_decode("fast", "easy", s, "stim") for s in range(3)])
    hard = np.mean([_decode("fast", "hard", s, "stim") for s in range(3)])
    assert easy > hard + 0.1


def test_hidden_channels_are_predictable_from_visible_ones():
    d = make_synthetic("fast", "medium", 0)
    st = structure("fast")
    X = d["X"].astype(float)
    Xd = X - np.nanmean(X, axis=2, keepdims=True)
    train, test = _split(d)
    N, R, T, M = X.shape
    rows_tr = np.repeat(train, T)
    ratios, noise_ratio = [], None
    for r in range(R):
        others = [q for q in range(R) if q != r]
        Xi = np.nan_to_num(Xd[:, others][..., 0].transpose(0, 2, 1).reshape(-1, R - 1))
        yi = Xd[:, r, :, 0].reshape(-1)
        ok = np.isfinite(yi)
        pred = Ridge(alpha=1.0).fit(Xi[rows_tr & ok], yi[rows_tr & ok]).predict(Xi[~rows_tr & ok])
        y = yi[~rows_tr & ok]
        ratio = np.mean((pred - y) ** 2) / np.var(y)
        if r + 1 == st["noise_channel"]:
            noise_ratio = ratio
        else:
            ratios.append(ratio)
    assert np.median(ratios) <= 0.6
    assert noise_ratio > 0.9  # the planted unpredictable channel really is unpredictable


def test_ach_is_predictable_from_calcium():
    d = make_synthetic("fast", "medium", 0)
    st = structure("fast")
    X = d["X"].astype(float)
    Xd = X - np.nanmean(X, axis=2, keepdims=True)
    train, _ = _split(d)
    N, R, T, M = X.shape
    rows_tr = np.repeat(train, T)
    Xc = np.nan_to_num(Xd[..., 0].transpose(0, 2, 1).reshape(-1, R))
    Ya = Xd[..., 1].transpose(0, 2, 1).reshape(-1, R)
    r2 = []
    for r in range(R):
        if r + 1 == st["noise_channel"]:
            continue
        y = Ya[:, r]
        ok = np.isfinite(y)
        pred = Ridge(alpha=1.0).fit(Xc[rows_tr & ok], y[rows_tr & ok]).predict(Xc[~rows_tr & ok])
        yt = y[~rows_tr & ok]
        r2.append(1 - np.mean((pred - yt) ** 2) / np.var(yt))
    assert np.median(r2) >= 0.15
