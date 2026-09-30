"""Simple reference models. A transformer result means little unless it beats them.

Classification: chance (1/C), always guessing the most common training class, and
logistic regression on the same preprocessed trials.

Masked reconstruction, each scored on exactly the model's hidden tokens:
* ``psth``: the average of that token (channel, modality, time) over training trials;
* ``visible_mean``: the average of the same channel's visible tokens in the same trial;
* ``hold_last``: the channel's last visible value earlier in the trial (causal);
* ``interp``: linear interpolation between the nearest visible values before and after.
Where a baseline has nothing to go on (e.g. a whole channel is hidden) it falls back to
the PSTH, so every baseline predicts every hidden token.
"""

from __future__ import annotations

import warnings

import numpy as np

MAE_BASELINES = ("psth", "visible_mean", "hold_last", "interp")


def majority(y_train: np.ndarray, y_test: np.ndarray, n_classes: int) -> dict:
    guess = int(np.bincount(y_train, minlength=n_classes).argmax())
    return {"accuracy": float((y_test == guess).mean()) if len(y_test) else None,
            "balanced_accuracy": 1.0 / n_classes, "class": guess}


def logistic(F_train, y_train, F_val, y_val, F_test, n_classes: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Balanced logistic regression; C chosen on validation balanced accuracy. Returns test probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    best, best_score, best_c = None, -1.0, None
    for c in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(F_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(F_val)) if len(y_val) else 0.0
        if score > best_score:
            best, best_score, best_c = clf, score, c
    prob = np.zeros((len(F_test), n_classes))
    prob[:, best.classes_] = best.predict_proba(F_test)
    return prob, {"C": best_c, "val_balanced_accuracy": float(best_score)}


def features(values: np.ndarray, valid: np.ndarray, max_features: int = 20000) -> tuple[np.ndarray, str]:
    """Flattened token values (missing = 0) for the linear baseline; time-averaged if too large."""
    V = np.where(valid[..., None], values, 0.0)
    n = V.shape[0]
    flat = V.reshape(n, -1)
    if flat.shape[1] <= max_features:
        return flat.astype(np.float32), "all token values"
    return V.mean(axis=2).astype(np.float32), "token averages"


def mae_predictions(x: np.ndarray, visible: np.ndarray, psth: np.ndarray) -> dict[str, np.ndarray]:
    """Baseline predictions for every token. x [b, patches, pairs, P]; visible [b, patches, pairs];
    psth [patches, pairs, P]. Returns name -> [b, patches, pairs, P]."""
    b, T, Q, P = x.shape
    fill = np.broadcast_to(psth[None], x.shape)
    out = {"psth": np.array(fill)}
    vis = visible[..., None]
    count = vis.sum(1, keepdims=True)
    mean = np.divide((x * vis).sum(1, keepdims=True), count, out=np.zeros((b, 1, Q, P)), where=count > 0)
    out["visible_mean"] = np.where(count > 0, np.broadcast_to(mean, x.shape), fill)
    last_val = np.zeros((b, Q, P))
    last_t = np.full((b, Q), -1)
    hold = np.array(fill)
    prev_val = np.zeros(x.shape)
    prev_t = np.full((b, T, Q), -1)
    for t in range(T):
        hold[:, t] = np.where((last_t >= 0)[..., None], last_val, fill[:, t])
        prev_val[:, t], prev_t[:, t] = last_val, last_t
        now = visible[:, t]
        last_val = np.where(now[..., None], x[:, t], last_val)
        last_t = np.where(now, t, last_t)
    out["hold_last"] = hold
    next_val = np.zeros((b, Q, P))
    next_t = np.full((b, Q), -1)
    interp = np.array(hold)
    for t in range(T - 1, -1, -1):
        pv, pt = prev_val[:, t], prev_t[:, t]
        have_both = (pt >= 0) & (next_t >= 0)
        w = np.divide(t - pt, next_t - pt, out=np.zeros(pt.shape, float), where=have_both)
        both = pv + w[..., None] * (next_val - pv)
        only_next = (pt < 0) & (next_t >= 0)
        interp[:, t] = np.where(have_both[..., None], both, np.where(only_next[..., None], next_val, hold[:, t]))
        now = visible[:, t]
        next_val = np.where(now[..., None], x[:, t], next_val)
        next_t = np.where(now, t, next_t)
    out["interp"] = interp
    return out
