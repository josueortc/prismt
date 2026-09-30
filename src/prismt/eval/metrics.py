"""Scores for classification and reconstruction.

Balanced accuracy (the mean recall over classes) is the headline for classification: it
is 1/C for any constant guess, whatever the class proportions, so "better than chance"
means the same thing on every dataset.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y: np.ndarray, prob: np.ndarray, n_classes: int) -> dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = prob.argmax(1)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(conf, (y, pred), 1)
    support = conf.sum(1)
    recall = np.divide(np.diag(conf), support, out=np.full(n_classes, np.nan), where=support > 0)
    predicted = conf.sum(0)
    precision = np.divide(np.diag(conf), predicted, out=np.full(n_classes, np.nan), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=np.nan_to_num(denom) > 0)
    present = support > 0
    eps = 1e-12
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else None,
        "macro_f1": float(np.mean(f1[present])) if present.any() else None,
        "log_loss": float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], eps, 1)))) if len(y) else None,
        "auroc": auroc(y, prob, n_classes),
        "confusion": conf.tolist(),
        "recall": [None if np.isnan(v) else float(v) for v in recall],
        "precision": [None if np.isnan(v) else float(v) for v in precision],
        "support": support.tolist(),
        "chance": 1.0 / n_classes,
    }
    return out


def auroc(y: np.ndarray, prob: np.ndarray, n_classes: int) -> float | None:
    """Binary AUROC, or the macro one-vs-rest average; None if a class is absent."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < n_classes or len(y) < 2:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        return float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(n_classes))))
    except ValueError:
        return None


def per_group(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int) -> list[dict]:
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        rec = [float((pred[m & (y == k)] == k).mean()) for k in range(n_classes) if (m & (y == k)).any()]
        rows.append({"group": str(g), "n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                     "balanced_accuracy": float(np.mean(rec)) if rec else None,
                     "classes_present": int(len(rec))})
    return rows


class R2Accumulator:
    """Sums for R² of masked reconstructions, pooled and per pair, per time patch and per trial.

    R² = 1 - SSE / SST where SST uses the mean of the scored targets themselves, so R² = 0
    means "no better than predicting the average of what was hidden".
    """

    def __init__(self, n_trials: int, n_pairs: int, n_patches: int) -> None:
        self.n = np.zeros((n_trials, n_pairs))
        self.s1 = np.zeros((n_trials, n_pairs))
        self.s2 = np.zeros((n_trials, n_pairs))
        self.sse = np.zeros((n_trials, n_pairs))
        self.patch_n = np.zeros(n_patches)
        self.patch_s1 = np.zeros(n_patches)
        self.patch_s2 = np.zeros(n_patches)
        self.patch_sse = np.zeros(n_patches)

    def add(self, rows: np.ndarray, scored: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
        """scored [b, patches, pairs]; target/pred [b, patches, pairs, patch_len] (normalized units)."""
        w = scored[..., None].astype(float)
        err = ((pred - target) ** 2) * w
        cnt = w * np.ones_like(target)
        self.n[rows] += cnt.sum((1, 3))
        self.s1[rows] += (target * w).sum((1, 3))
        self.s2[rows] += (target ** 2 * w).sum((1, 3))
        self.sse[rows] += err.sum((1, 3))
        self.patch_n += cnt.sum((0, 2, 3))
        self.patch_s1 += (target * w).sum((0, 2, 3))
        self.patch_s2 += (target ** 2 * w).sum((0, 2, 3))
        self.patch_sse += err.sum((0, 2, 3))

    @staticmethod
    def r2(n, s1, s2, sse):
        n = np.asarray(n, float)
        sst = np.asarray(s2, float) - np.divide(np.asarray(s1, float) ** 2, n, out=np.zeros_like(n), where=n > 0)
        return np.where((n > 1) & (sst > 1e-12), 1 - np.asarray(sse, float) / np.where(sst > 1e-12, sst, 1), np.nan)

    def summary(self) -> dict:
        tot = [a.sum() for a in (self.n, self.s1, self.s2, self.sse)]
        return {
            "r2": _num(self.r2(*tot)),
            "mse": _num(tot[3] / tot[0]) if tot[0] else None,
            "n_scored": int(tot[0]),
            "r2_by_pair": [_num(v) for v in self.r2(self.n.sum(0), self.s1.sum(0), self.s2.sum(0), self.sse.sum(0))],
            "r2_by_patch": [_num(v) for v in self.r2(self.patch_n, self.patch_s1, self.patch_s2, self.patch_sse)],
        }


def _num(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None
