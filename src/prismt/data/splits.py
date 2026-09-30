"""Train / validation / test splits that respect subjects and sessions.

Trials from one subject (an animal, a participant) or one session are not independent:
they share anatomy, recording conditions and behavioural state. If the label is constant within a group (for
example ``phase`` within a session, or genotype within an animal), putting trials of the
same group in training and testing lets a model recognize the group instead of learning
the label, and the score is inflated. So:

* ``test_on`` must be at least as coarse as the label's level (the coarsest grouping in
  which the label never changes); otherwise the split is refused (or flagged, with
  ``allow_leaky``);
* by default cross-validation is used, so every subject (or session) is tested once;
* groups are assigned by a stable hash within label strata, so the assignment does not
  depend on row order and every class appears in every part where possible;
* :func:`assert_no_leakage` turns all of this from a convention into a check.

Hash ranking and stratification follow MouseWFM's ``data/splits.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from prismt.errors import Issue, SplitError, raise_if_errors

LEVELS = ("trial", "session", "subject")  # from finest to coarsest
_TITLE = "The trials cannot be split safely"


class LeakageError(AssertionError):
    """A group appears on both sides of a split."""


@dataclass
class Fold:
    index: int
    train: np.ndarray  # positions in the selection (0-based)
    val: np.ndarray
    test: np.ndarray


@dataclass
class SplitPlan:
    folds: list[Fold]
    scheme: str  # single | kfold | loo
    test_on: str
    validate_on: str
    label_level: str
    n_groups: int
    flags: dict = field(default_factory=dict)
    warnings: list[Issue] = field(default_factory=list)

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=8)
        h.update(f"{self.scheme}|{self.test_on}|{self.validate_on}".encode())
        for f in self.folds:
            for part in (f.train, f.val, f.test):
                h.update(np.sort(part).astype(np.int64).tobytes())
                h.update(b"|")
        return h.hexdigest()

    def summary(self, y: np.ndarray | None, class_names: list[str], groups: dict[str, np.ndarray | None]) -> dict:
        folds = []
        for f in self.folds:
            parts = {}
            for name, idx in (("train", f.train), ("val", f.val), ("test", f.test)):
                entry = {"n_trials": int(len(idx))}
                if y is not None:
                    entry["class_counts"] = np.bincount(y[idx], minlength=len(class_names)).tolist()
                for level in ("subject", "session"):
                    g = groups.get(level)
                    if g is not None:
                        entry[f"n_{level}s"] = int(len(set(g[idx].tolist())))
                if groups.get("subject") is not None and self.test_on == "subject":
                    entry["subjects"] = sorted(set(groups["subject"][idx].tolist()))
                parts[name] = entry
            folds.append({"fold": f.index + 1, **parts})
        return {
            "scheme": self.scheme,
            "test_on": self.test_on,
            "validate_on": self.validate_on,
            "label_level": self.label_level,
            "n_groups": self.n_groups,
            "n_folds": len(self.folds),
            "folds": folds,
            "flags": self.flags,
            "warnings": [w.to_dict() for w in self.warnings],
            "fingerprint": self.fingerprint(),
        }


def stable_hash(value: str, salt: str) -> float:
    """Deterministic number in [0, 1), the same on every machine (unlike Python's hash)."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def plan_splits(
    y: np.ndarray | None,
    subject: np.ndarray | None,
    session: np.ndarray | None,
    cfg: dict,
    *,
    label_name: str = "label",
    class_names: list[str] | None = None,
) -> SplitPlan:
    """Plan the folds for ``n`` selected trials. ``cfg`` is the ``split`` section of the config."""
    from prismt.data.selection import label_level as _label_level

    n = len(y) if y is not None else len(subject if subject is not None else session)
    issues: list[Issue] = []
    warnings: list[Issue] = []
    flags: dict = {}
    groups = {"trial": np.array([str(i) for i in range(n)], dtype=object), "session": session, "subject": subject}
    level = _label_level(y, subject, session)

    test_on = cfg["test_on"]
    if test_on == "auto":
        if subject is not None and len(set(subject)) >= 3:
            test_on = "subject"
        elif session is not None and len(set(session)) >= 3:
            test_on = "session"
        else:
            test_on = "trial"
    elif groups[test_on] is None:
        raise SplitError("E_SPLIT_NO_ROLE", f"Testing on new {test_on}s needs a {test_on} column.",
                         hint=f"Mark the column that identifies each {test_on} in the dataset, or choose another "
                              "'Test on'.", field="split.test_on", title=_TITLE)
    if LEVELS.index(test_on) < LEVELS.index(level):
        message = (f"'{label_name}' is the same for every trial of a {level}, so testing on held-out "
                   f"{test_on}s would let the model recognize {level}s instead of learning '{label_name}'.")
        if cfg["allow_leaky"]:
            warnings.append(Issue("warning", "W_LEAKY_SPLIT", message + " Scores will be inflated."))
            flags["leaky_split"] = True
        else:
            raise SplitError("E_LEAKY_SPLIT", message,
                             hint=f"Test on new {level}s (or set 'Test on' to auto).", field="split.test_on",
                             title=_TITLE)
    if subject is None and y is not None:
        warnings.append(Issue("warning", "W_NO_SUBJECT", "No subject column: the score only shows that the model "
                              "works on held-out trials, not on new subjects."))

    g = groups[test_on]
    uniq = sorted(set(g.tolist()))
    n_groups = len(uniq)
    strata = _group_strata(g, y, uniq)
    if y is not None and LEVELS.index(level) >= LEVELS.index(test_on):
        names = class_names or [str(k) for k in range(int(y.max()) + 1)]
        for k, name in enumerate(names):
            n_with = sum(1 for s in strata.values() if s == k)
            if n_with < 2:
                issues.append(Issue("error", "E_SPLIT_INFEASIBLE",
                                    f"Only {n_with} {test_on} has class '{name}', so it cannot be used for both "
                                    "training and testing.",
                                    f"Each class needs at least two {test_on}s. Add data, merge classes, or test on "
                                    "a finer level if the label varies within it.", "split.test_on"))
    raise_if_errors(issues, SplitError, title=_TITLE)

    folds_cfg = cfg["folds"]
    if folds_cfg == "auto":
        # Cross-validation by default: every subject (session, trial) is tested exactly once,
        # so the score does not hinge on which few groups happened to land in one test set.
        if n_groups >= 6:
            scheme, k = "kfold", 5
        elif n_groups >= 3:
            scheme, k = "loo", n_groups
            warnings.append(Issue("info", "I_FEW_GROUPS",
                                  f"Only {n_groups} {test_on}s: every {test_on} is tested once "
                                  "(leave-one-out cross-validation). Report the pooled result and the spread."))
        else:
            raise SplitError("E_SPLIT_FEW_GROUPS", f"Only {n_groups} {test_on}(s): at least 3 are needed to train, "
                             "validate and test on different ones.", hint="Add data or test on a finer level.",
                             field="split.test_on", title=_TITLE)
    elif folds_cfg == "loo":
        scheme, k = "loo", n_groups
    elif folds_cfg == 1:
        scheme, k = "single", 1
    else:
        scheme, k = "kfold", int(folds_cfg)
        if k > n_groups:
            raise SplitError("E_SPLIT_FOLDS", f"{k} folds need at least {k} {test_on}s; there are {n_groups}.",
                             field="split.folds", title=_TITLE)

    salt = f"prismt-split-{cfg['seed']}"
    ranked = _rank_within_strata(uniq, strata, salt)
    if scheme == "single":
        test_groups = [_pick_fraction(ranked, cfg["test_fraction"])]
    elif scheme == "loo":
        test_groups = [[u] for u in sorted(uniq, key=lambda u: stable_hash(u, salt))]
    else:
        buckets: list[list[str]] = [[] for _ in range(k)]
        i = 0
        for members in ranked.values():
            for u in members:
                buckets[i % k].append(u)
                i += 1
        test_groups = buckets

    folds = []
    validate_levels = []
    for fi, tg in enumerate(test_groups):
        test_mask = np.isin(g, tg)
        rest = np.flatnonzero(~test_mask)
        val_rows, v_level, v_warn = _validation(rest, y, groups, test_on, level, cfg, salt + f"-val{fi}")
        warnings.extend(v_warn)
        validate_levels.append(v_level)
        train_rows = np.setdiff1d(rest, val_rows)
        folds.append(Fold(fi, train_rows, np.sort(val_rows), np.flatnonzero(test_mask)))

    plan = SplitPlan(folds, scheme, test_on, validate_levels[0] if validate_levels else test_on, level, n_groups,
                     flags, _dedupe(warnings))
    _check_classes(plan, y, class_names, issues, plan.warnings)
    raise_if_errors(issues, SplitError, title=_TITLE)
    assert_no_leakage(plan, groups)
    return plan


def _group_strata(g: np.ndarray, y: np.ndarray | None, uniq: list[str]) -> dict[str, int]:
    """Each group's label if it has one label, otherwise its most common label (-1 without labels)."""
    if y is None:
        return {u: -1 for u in uniq}
    out = {}
    for u in uniq:
        vals = y[g == u]
        out[u] = int(np.bincount(vals).argmax())
    return out


def _rank_within_strata(groups: list[str], strata: dict[str, int], salt: str) -> dict[int, list[str]]:
    by: dict[int, list[str]] = {}
    for u in groups:
        by.setdefault(strata[u], []).append(u)
    return {s: sorted(m, key=lambda u: stable_hash(u, salt)) for s, m in sorted(by.items())}


def _pick_fraction(ranked: dict[int, list[str]], fraction: float) -> list[str]:
    out = []
    for members in ranked.values():
        n = len(members)
        take = int(round(n * fraction))
        if n >= 2:
            take = min(max(take, 1), n - 1)
        else:
            take = 0
        out.extend(members[:take])
    return out


def _validation(rest: np.ndarray, y, groups, test_on: str, level: str, cfg: dict, salt: str):
    """Hold out validation groups from the training part of one fold."""
    warnings: list[Issue] = []
    wanted = cfg["validate_on"]
    floor = "trial" if cfg["allow_leaky"] else level
    order = [lv for lv in LEVELS[::-1] if LEVELS.index(lv) >= LEVELS.index(floor) and groups[lv] is not None]
    if wanted != "auto":
        if groups[wanted] is None:
            raise SplitError("E_SPLIT_NO_ROLE", f"Validating on {wanted}s needs a {wanted} column.",
                             field="split.validate_on", title=_TITLE)
        if LEVELS.index(wanted) < LEVELS.index(floor):
            raise SplitError("E_LEAKY_SPLIT", f"The label is constant within {level}s, so validation must hold out "
                             f"whole {level}s.", field="split.validate_on", title=_TITLE)
        order = [wanted]
    else:
        start = LEVELS.index(test_on)
        order = [lv for lv in order if LEVELS.index(lv) <= start]
    chosen = None
    for lv in order:
        g = groups[lv][rest]
        uniq = sorted(set(g.tolist()))
        if len(uniq) < 2:
            continue
        strata = _group_strata(g, None if y is None else y[rest], uniq)
        if wanted == "auto" and lv != "trial":
            if len(uniq) < 4:
                continue
            if y is not None and LEVELS.index(level) >= LEVELS.index(lv):
                counts = np.bincount(list(strata.values()), minlength=int(y.max()) + 1)
                if (counts[np.unique(y[rest])] < 2).any():
                    continue
        chosen = (lv, g, uniq, strata)
        break
    if chosen is None:
        raise SplitError("E_SPLIT_NO_VAL", "There is not enough data left to hold out validation trials.",
                         hint="Add data, use fewer folds, or test on a finer level.", field="split.validate_on",
                         title=_TITLE)
    lv, g, uniq, strata = chosen
    if lv != test_on:
        warnings.append(Issue("info", "I_VAL_FINER",
                              f"Validation holds out {lv}s of the training {test_on}s (there are too few {test_on}s "
                              "to spare). This only affects when training stops; the test score is unaffected."))
    ranked = _rank_within_strata(uniq, strata, salt)
    val_groups = _pick_fraction(ranked, cfg["val_fraction"])
    if not val_groups:
        val_groups = [sorted(uniq, key=lambda u: stable_hash(u, salt))[0]]
    return rest[np.isin(g, val_groups)], lv, warnings


def _check_classes(plan: SplitPlan, y, class_names, issues: list[Issue], warnings: list[Issue]) -> None:
    if y is None:
        return
    names = class_names or [str(k) for k in range(int(y.max()) + 1)]
    for f in plan.folds:
        for k, name in enumerate(names):
            if not (y[f.train] == k).any():
                issues.append(Issue("error", "E_SPLIT_CLASS_MISSING",
                                    f"Fold {f.index + 1}: no training trials of class '{name}'.",
                                    "Add data for this class or merge classes.", "labels.classes"))
            if len(f.val) and not (y[f.val] == k).any():
                warnings.append(Issue("warning", "W_VAL_CLASS_MISSING",
                                      f"Fold {f.index + 1}: the validation trials contain no '{name}'."))
            if plan.scheme == "single" and not (y[f.test] == k).any():
                issues.append(Issue("error", "E_SPLIT_CLASS_MISSING",
                                    f"The test trials contain no class '{name}'.",
                                    "Use cross-validation (Evaluation: auto or a number of folds).", "split.folds"))


def assert_no_leakage(plan: SplitPlan, groups: dict[str, np.ndarray | None]) -> None:
    """Raise LeakageError if a test-level (or label-level) group is on two sides of a fold."""
    levels = {plan.test_on}
    if plan.label_level != "trial":
        levels.add(plan.label_level)
    for f in plan.folds:
        for lv in levels:
            g = groups.get(lv)
            if g is None:
                continue
            test = set(g[f.test].tolist())
            for part, idx in (("training", f.train), ("validation", f.val)):
                both = test & set(g[idx].tolist())
                if both and not plan.flags.get("leaky_split"):
                    raise LeakageError(f"fold {f.index + 1}: {lv} {sorted(both)[:3]} in test and {part}")
        if plan.validate_on != "trial" and groups.get(plan.validate_on) is not None:
            gv = groups[plan.validate_on]
            both = set(gv[f.train].tolist()) & set(gv[f.val].tolist())
            if both:
                raise LeakageError(f"fold {f.index + 1}: {plan.validate_on} {sorted(both)[:3]} in training and validation")
        if len(np.intersect1d(f.train, f.test)) or len(np.intersect1d(f.val, f.test)) or len(np.intersect1d(f.train, f.val)):
            raise LeakageError(f"fold {f.index + 1}: a trial is in two parts")


def _dedupe(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for i in issues:
        if (i.code, i.message) not in seen:
            seen.add((i.code, i.message))
            out.append(i)
    return out
