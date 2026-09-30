"""Which trials, channels and modalities a run uses, and what the classes are.

Everything here works on dataset metadata only (no torch), so the MATLAB app can preview
it quickly through ``prismt check``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prismt.errors import ConfigError, Issue
from prismt.io.dataset import Column, PrismtDataset, _format_number

_TITLE = "The trial selection does not work with this dataset"


@dataclass
class Selection:
    trial_index: np.ndarray  # dataset trial numbers (0-based) in the order used
    y: np.ndarray | None  # class number per selected trial (classification only)
    class_names: list[str]
    class_members: list[list[str]]  # the column values merged into each class
    label_column: str | None
    channels: np.ndarray  # selected channels (0-based dataset indices)
    modalities: np.ndarray  # selected modalities (0-based dataset indices)
    pairs: list[tuple[int, int]]  # (position in channels, position in modalities)
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.trial_index))


def value_keys(col: Column, value) -> set[str]:
    """The display strings a filter or class value refers to (numbers, codes or labels)."""
    keys = {str(value)}
    if col.kind == "numeric":
        num = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            num = float(value)
        else:
            reverse = {lab: code for code, lab in col.labels.items()}
            if str(value) in reverse:
                num = reverse[str(value)]
            else:
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    num = None
        if num is not None:
            keys.add(col.labels.get(num, _format_number(num)))
            keys.add(_format_number(num))
    elif col.kind == "bool":
        keys = {"true"} if str(value).lower() in ("true", "1", "yes") else {"false"}
    return keys


def _match(col: Column, values: list, issues: list[Issue], where: str) -> np.ndarray:
    text = col.as_text()
    present = set(text[~col.missing()].tolist())
    wanted: set[str] = set()
    for v in values:
        keys = value_keys(col, v)
        if not keys & present:
            shown = ", ".join(sorted(present)[:12]) or "(none)"
            issues.append(Issue("error", "E_SEL_VALUE", f"Column '{col.name}' has no value '{v}'.",
                                f"Values present: {shown}.", where))
        wanted |= keys
    return np.isin(text, list(wanted))


def select(ds: PrismtDataset, cfg: dict) -> Selection:
    """Apply the trial filters, the channel/modality choice and the class definition."""
    issues: list[Issue] = []
    warnings: list[Issue] = []
    dropped: dict[str, int] = {}
    N = ds.n_trials
    keep = np.ones(N, dtype=bool)

    for i, f in enumerate(cfg["selection"]["filters"]):
        where = f"selection.filters[{i + 1}]"
        if f["column"] not in ds.columns:
            issues.append(Issue("error", "E_SEL_COLUMN", f"There is no trial column '{f['column']}' to filter on.",
                                f"Columns: {', '.join(ds.columns)}.", where))
            continue
        col = ds.columns[f["column"]]
        if f["op"] == "range":
            if col.kind != "numeric":
                issues.append(Issue("error", "E_SEL_RANGE", f"A range filter needs a numeric column; '{col.name}' is {col.kind}.",
                                    "Use 'in' with a list of values instead.", where))
                continue
            lo, hi = f["values"]
            m = np.ones(N, dtype=bool)
            if lo is not None:
                m &= col.values >= lo
            if hi is not None:
                m &= col.values <= hi
        else:
            m = _match(col, f["values"], issues, where)
            if f["op"] == "not_in":
                m = ~m & ~col.missing()
        before = int(keep.sum())
        keep &= m
        dropped[f"filter on {col.name}"] = before - int(keep.sum())

    channels = _pick(ds.channel_names, cfg["selection"]["channels"], "channel", "selection.channels", issues)
    wanted_groups = cfg["selection"].get("channel_groups")
    if wanted_groups:
        groups = ds.channel_groups or ("",) * len(ds.channel_names)
        present = sorted({g for g in groups if g})
        unknown = [g for g in wanted_groups if g not in present]
        if unknown:
            issues.append(Issue("error", "E_SEL_NAME", f"There is no channel group called {', '.join(map(repr, unknown))}.",
                                f"Groups in this dataset: {', '.join(present) or 'none (set ChannelGroups when making it)'}.",
                                "selection.channel_groups"))
        channels = np.asarray([c for c in channels if groups[int(c)] in wanted_groups], dtype=np.int64)
    modalities = _pick(ds.modality_names, cfg["selection"]["modalities"], "modality", "selection.modalities", issues)
    pairs = []
    chan_pos = {int(c): i for i, c in enumerate(channels)}
    for mi, m in enumerate(modalities):
        for c in ds.modalities[int(m)].channels:
            if int(c) in chan_pos:
                pairs.append((chan_pos[int(c)], mi))
    if not pairs and not issues:
        issues.append(Issue("error", "E_SEL_EMPTY", "No channel is recorded in the chosen modalities.", "", "selection.channels"))

    # Trials with too little data in the chosen channels.
    if pairs:
        sub = ds.X[:, channels][:, :, :, modalities]
        finite = np.isfinite(sub).mean(axis=(1, 2, 3)) * sub.shape[1] * sub.shape[3] / max(len(pairs), 1)
        enough = finite >= cfg["selection"]["min_valid_fraction"]
        dropped["too much missing data"] = int((keep & ~enough).sum())
        keep &= enough

    y = None
    class_names: list[str] = []
    members: list[list[str]] = []
    label_column = cfg["labels"]["column"] if cfg["task"] == "classify" else None
    if label_column is not None:
        if label_column not in ds.columns:
            issues.append(Issue("error", "E_SEL_COLUMN", f"There is no trial column '{label_column}' to classify.",
                                f"Columns: {', '.join(ds.columns)}.", "labels.column"))
        else:
            col = ds.columns[label_column]
            text = col.as_text()
            missing = col.missing()
            dropped["label missing"] = int((keep & missing).sum())
            keep &= ~missing
            classes = cfg["labels"]["classes"] or [{"name": v, "values": [v]} for v in _display_unique(col)]
            y = np.full(N, -1, dtype=np.int64)
            for k, c in enumerate(classes):
                m = _match(col, c["values"], issues, "labels.classes")
                y[m] = k
                class_names.append(c["name"])
                members.append(sorted(set(text[m].tolist())))
            dropped["label not in the chosen classes"] = int((keep & (y < 0)).sum())
            keep &= y >= 0
            counts = np.bincount(y[keep], minlength=len(classes)) if keep.any() else np.zeros(len(classes), int)
            if len(classes) < 2:
                issues.append(Issue("error", "E_SEL_ONE_CLASS", "Classification needs at least two classes.",
                                    "Choose at least two values of the label column.", "labels.classes"))
            for k, n in enumerate(counts):
                if n == 0 and len(classes) >= 2:
                    issues.append(Issue("error", "E_SEL_EMPTY_CLASS",
                                        f"Class '{classes[k]['name']}' has no trials after filtering.",
                                        "Check the trial filters and the class values.", "labels.classes"))

    max_per = cfg["selection"]["max_trials_per_session"]
    if max_per:
        keys = ds.session_keys()
        if keys is None:
            warnings.append(Issue("warning", "W_SEL_NO_SESSION",
                                  "'Trials per session' was ignored because the dataset has no session column."))
        else:
            rng = np.random.default_rng(cfg["split"]["seed"])
            limited = np.zeros(N, dtype=bool)
            for key in np.unique(keys[keep]):
                idx = np.flatnonzero(keep & (keys == key))
                limited[rng.permutation(idx)[:max_per]] = True
            dropped["trials per session limit"] = int((keep & ~limited).sum())
            keep &= limited

    if not issues and not keep.any():
        issues.append(Issue("error", "E_SEL_EMPTY", "No trials are left after filtering.",
                            "Loosen the trial filters.", "selection.filters"))
    if any(i.level == "error" for i in issues):
        from prismt.errors import raise_if_errors

        raise_if_errors(issues, ConfigError, title=_TITLE)
    idx = np.flatnonzero(keep)
    return Selection(
        trial_index=idx,
        y=None if y is None else y[idx],
        class_names=class_names,
        class_members=members,
        label_column=label_column,
        channels=channels,
        modalities=modalities,
        pairs=pairs,
        dropped={k: v for k, v in dropped.items() if v},
        warnings=warnings,
    )


def _display_unique(col: Column) -> list[str]:
    text = col.as_text()[~col.missing()]
    if col.kind == "categorical":
        return [c for c in col.unique()]
    if col.kind == "numeric":
        return [col.labels.get(v, _format_number(v)) for v in col.unique()]
    return sorted(set(text.tolist()))


def _pick(names: tuple[str, ...], wanted, what: str, where: str, issues: list[Issue]) -> np.ndarray:
    if not wanted:
        return np.arange(len(names))
    idx = []
    for w in wanted:
        if w in names:
            idx.append(names.index(w))
        else:
            issues.append(Issue("error", "E_SEL_NAME", f"There is no {what} called '{w}'.",
                                f"Available: {', '.join(names[:20])}{'…' if len(names) > 20 else ''}.", where))
    return np.asarray(sorted(set(idx)), dtype=np.int64)


def confound_warnings(ds: PrismtDataset, sel: Selection, max_levels: int = 20) -> list[Issue]:
    """Columns that almost perfectly predict the label, and missing data that differ by class."""
    out: list[Issue] = []
    if sel.y is None:
        return out
    y = sel.y
    for name, col in ds.columns.items():
        if name in (sel.label_column, ds.subject, ds.session) or col.kind != "categorical":
            continue
        values = col.as_text()[sel.trial_index]
        levels = np.unique(values)
        if not (2 <= len(levels) <= max_levels):
            continue
        v = _cramers_v(values, y)
        if v >= 0.9:
            out.append(Issue("warning", "W_CONFOUND",
                             f"'{name}' almost perfectly predicts the label (Cramér's V = {v:.2f}).",
                             f"The model may be learning '{name}' rather than '{sel.label_column}'. Consider "
                             "balancing them or testing within one level of it."))
    sub = ds.X[sel.trial_index][:, sel.channels][:, :, :, sel.modalities]
    miss = (~np.isfinite(sub)).mean(axis=(1, 2, 3))
    per_class = [float(miss[y == k].mean()) for k in range(len(sel.class_names)) if (y == k).any()]
    if per_class and max(per_class) - min(per_class) > 0.1:
        out.append(Issue("warning", "W_MISSING_BY_CLASS",
                         "The amount of missing data differs between classes "
                         f"({', '.join(f'{p:.0%}' for p in per_class)}).",
                         "A model could tell the classes apart by where data are missing."))
    return out


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)))
    np.add.at(table, (ia, ib), 1)
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def label_level(y: np.ndarray | None, subject: np.ndarray | None, session: np.ndarray | None) -> str:
    """The coarsest grouping within which the label never changes: subject, session or trial."""
    if y is None:
        return "trial"
    for level, groups in (("subject", subject), ("session", session)):
        if groups is not None and _constant_within(y, groups):
            return level
    return "trial"


def _constant_within(y: np.ndarray, groups: np.ndarray) -> bool:
    first: dict = {}
    for g, v in zip(groups, y):
        if first.setdefault(g, v) != v:
            return False
    return True
