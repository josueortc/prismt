"""The PRISMT dataset file (format 1): reading, validating, summarizing and writing.

A PRISMT dataset holds one array ``X`` of size trials x channels x time x modalities
(MATLAB order ``[N R T M]``) plus a JSON description of the channels, modalities, time axis
and per-trial metadata. MATLAB writes it with ``prismt.writeDataset``; this module reads it
with no guessing: the file declares its own shape and carries orientation probes that are
checked after the axes are restored. The full specification is ``docs/data-format.md``.

Values that are NaN are missing (a channel outside the cranial window, a dropped frame, or
padding for a modality with fewer channels than ``R``). They are never used as inputs or
scored as targets.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from prismt import FORMATS, __version__
from prismt.errors import DatasetError, Issue, raise_if_errors
from prismt.io import matv73
from prismt.jsonutil import as_list

FORMAT_TAG = "prismt.dataset"
FORMAT_VERSION = FORMATS["dataset"]
COLUMN_TYPES = ("numeric", "categorical", "bool")
MODALITY_KINDS = ("neural", "behavior", "other")
_READ_TITLE = "The dataset file could not be used"


# ---------------------------------------------------------------------------------------
# In-memory representation
# ---------------------------------------------------------------------------------------


def _format_number(v: float) -> str:
    """How a numeric metadata value is shown: 1 not 1.0, 0.25 as 0.25."""
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


@dataclass(frozen=True)
class Column:
    """One per-trial metadata column."""

    name: str
    kind: str  # numeric | categorical | bool
    values: np.ndarray  # float64 (NaN = missing) | object str/None | bool
    categories: tuple[str, ...] = ()
    labels: dict[float, str] = field(default_factory=dict)

    def missing(self) -> np.ndarray:
        if self.kind == "numeric":
            return np.isnan(self.values)
        if self.kind == "categorical":
            return np.array([v is None for v in self.values], dtype=bool)
        return np.zeros(self.values.shape, dtype=bool)

    def as_text(self) -> np.ndarray:
        """Values as display strings (value labels applied); missing values become ''."""
        out = np.empty(self.values.shape, dtype=object)
        for i, v in enumerate(self.values):
            if self.kind == "numeric":
                out[i] = "" if np.isnan(v) else self.labels.get(float(v), _format_number(v))
            elif self.kind == "bool":
                out[i] = "true" if v else "false"
            else:
                out[i] = "" if v is None else str(v)
        return out

    def unique(self) -> list:
        """Distinct non-missing values: category order for categoricals, sorted otherwise."""
        present = self.values[~self.missing()]
        if self.kind == "categorical":
            seen = set(present.tolist())
            ordered = [c for c in self.categories if c in seen]
            ordered += sorted(seen - set(ordered))
            return ordered
        if self.kind == "bool":
            return sorted({bool(v) for v in present})
        return sorted({float(v) for v in present})


@dataclass(frozen=True)
class Modality:
    name: str
    unit: str
    kind: str
    channels: np.ndarray  # 0-based channel indices that exist for this modality


@dataclass(frozen=True)
class PrismtDataset:
    X: np.ndarray  # float32, (N, R, T, M)
    channel_names: tuple[str, ...]
    modalities: tuple[Modality, ...]
    times_s: np.ndarray
    event: str
    columns: dict[str, Column]
    subject: str | None
    session: str | None
    trial_uid: np.ndarray | None
    channel_x: np.ndarray | None
    channel_y: np.ndarray | None
    hemisphere: tuple[str, ...] | None
    atlas: str | None
    meta: dict
    warnings: tuple[Issue, ...] = ()
    path: Path | None = None
    fingerprint: str = ""

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_time(self) -> int:
        return int(self.X.shape[2])

    @property
    def n_modalities(self) -> int:
        return int(self.X.shape[3])

    @property
    def modality_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modalities)

    def token_pairs(self) -> list[tuple[int, int]]:
        """(channel, modality) pairs that exist, modality by modality (0-based)."""
        return [(int(r), m) for m, mod in enumerate(self.modalities) for r in mod.channels]

    def column(self, name: str) -> Column:
        if name not in self.columns:
            raise DatasetError(
                "E_DATA_NO_COLUMN",
                f"The dataset has no trial column called '{name}'.",
                hint=f"Available columns: {', '.join(self.columns) or '(none)'}.",
                field="labels.column",
            )
        return self.columns[name]

    def subject_ids(self) -> np.ndarray | None:
        return None if self.subject is None else self.columns[self.subject].as_text()

    def session_keys(self) -> np.ndarray | None:
        """Session identity made unique per subject ("session 1" of two mice differs)."""
        if self.session is None:
            return None
        sess = self.columns[self.session].as_text()
        if self.subject is None:
            return sess
        subj = self.columns[self.subject].as_text()
        return np.array([f"{a}/{b}" for a, b in zip(subj, sess)], dtype=object)


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------


def read_dataset(path: str | os.PathLike, *, compute_fingerprint: bool = True) -> PrismtDataset:
    """Read and validate a PRISMT dataset. Raises :class:`DatasetError` with every problem."""
    path = Path(path).expanduser()
    X, meta = _read_raw(path)
    issues = validate_contents(X, meta)
    raise_if_errors(issues, DatasetError, title=_READ_TITLE)
    ds = _build(X, meta, path, tuple(i for i in issues if i.level != "error"))
    if compute_fingerprint:
        ds = _with_fingerprint(ds)
    return ds


def validate_file(path: str | os.PathLike) -> dict:
    """Validate without raising: ``{"ok", "errors", "warnings", "summary"}`` for the CLI."""
    path = Path(path).expanduser()
    try:
        X, meta = _read_raw(path)
    except DatasetError as err:
        return {"ok": False, "errors": [err.to_dict()], "warnings": [], "summary": None}
    issues = validate_contents(X, meta)
    errors = [i.to_dict() for i in issues if i.level == "error"]
    warnings = [i.to_dict() for i in issues if i.level != "error"]
    summary = None
    if not errors:
        summary = summarize_dataset(_with_fingerprint(_build(X, meta, path, ())))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}


def _read_raw(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise DatasetError(
            "E_DATA_MISSING",
            f"The dataset file does not exist: {path}",
            hint="Check the path. In MATLAB, datasets are created with prismt.writeDataset "
            "or the Data tab.",
            title=_READ_TITLE,
        )
    if path.suffix.lower() == ".npz":
        return _read_npz(path)
    if matv73.is_hdf5(path):
        return _read_hdf5(path)
    header = matv73.mat_header(path)
    if header.startswith("MATLAB 5.0 MAT-file"):
        raise DatasetError(
            "E_DATA_NOT_PRISMT",
            f"{path.name} is a MATLAB file in the older (v7) format, not a PRISMT dataset.",
            hint="Convert it in MATLAB with the Data tab (Import…) or prismt.importData, "
            "which writes a PRISMT dataset.",
            title=_READ_TITLE,
        )
    raise DatasetError(
        "E_DATA_UNREADABLE",
        f"{path.name} is not a readable PRISMT dataset (not an HDF5/.mat v7.3 or .npz file).",
        hint="If the file is still being copied, wait for the copy to finish and try again.",
        title=_READ_TITLE,
    )


def _read_hdf5(path: Path) -> tuple[np.ndarray, dict]:
    try:
        fh = h5py.File(path, "r", locking=False)
    except OSError as exc:
        raise DatasetError(
            "E_DATA_UNREADABLE",
            f"{path.name} could not be opened ({exc}).",
            hint="The file may be incomplete. If it was just copied, copy it again.",
            title=_READ_TITLE,
        ) from exc
    with fh:
        if "prismt_format" not in fh:
            contents = ", ".join(list(fh.keys())[:8]) or "nothing"
            raise DatasetError(
                "E_DATA_NOT_PRISMT",
                f"{path.name} is a MATLAB v7.3 file but not a PRISMT dataset "
                f"(it contains: {contents}).",
                hint="Convert it in MATLAB with the Data tab (Import…) or prismt.importData, "
                "which writes a PRISMT dataset.",
                title=_READ_TITLE,
            )
        _check_tag(matv73.read_text(fh["prismt_format"]), matv73.read_scalar(fh["prismt_version"]))
        meta = _parse_meta(matv73.read_text(fh["meta_json"]))
        shape = _declared_shape(meta)
        if "X" not in fh:
            raise DatasetError("E_DATA_NO_X", "The dataset has no data array X.", title=_READ_TITLE)
        try:
            X = matv73.matlab_array(fh["X"][()], shape)
        except ValueError as exc:
            raise DatasetError(
                "E_DATA_SHAPE",
                f"The data array X does not have the size the file declares ({exc}).",
                hint="Re-create the file with prismt.writeDataset.",
                title=_READ_TITLE,
            ) from exc
    return X, meta


def _read_npz(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as npz:
        for key in ("prismt_format", "prismt_version", "meta_json", "X"):
            if key not in npz:
                raise DatasetError(
                    "E_DATA_NOT_PRISMT",
                    f"{path.name} is not a PRISMT dataset (missing '{key}').",
                    hint="Write datasets with prismt.io.write_dataset or prismt.writeDataset.",
                    title=_READ_TITLE,
                )
        _check_tag(str(npz["prismt_format"]), float(npz["prismt_version"]))
        meta = _parse_meta(str(npz["meta_json"]))
        shape = _declared_shape(meta)
        X = np.asarray(npz["X"])
    if tuple(X.shape) != tuple(shape):
        raise DatasetError(
            "E_DATA_SHAPE",
            f"X has size {list(X.shape)} but the file declares {shape}.",
            title=_READ_TITLE,
        )
    return X, meta


def _check_tag(tag: str, version: float) -> None:
    if tag != FORMAT_TAG:
        raise DatasetError(
            "E_DATA_NOT_PRISMT",
            f"The file's format tag is '{tag}', not '{FORMAT_TAG}'.",
            title=_READ_TITLE,
        )
    if int(version) > FORMAT_VERSION:
        raise DatasetError(
            "E_DATA_NEWER_FORMAT",
            f"This dataset uses format version {int(version)}; this copy of PRISMT reads "
            f"up to version {FORMAT_VERSION}.",
            hint="Update PRISMT on this computer (for example with git pull).",
            title=_READ_TITLE,
        )
    if int(version) < 1:
        raise DatasetError("E_DATA_BAD_VERSION", f"Invalid format version {version}.", title=_READ_TITLE)


def _parse_meta(text: str) -> dict:
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            "E_DATA_META_JSON",
            f"The dataset description (meta_json) is not valid JSON ({exc}).",
            hint="Re-create the file with prismt.writeDataset.",
            title=_READ_TITLE,
        ) from exc
    if not isinstance(meta, dict):
        raise DatasetError("E_DATA_META_JSON", "meta_json must be a JSON object.", title=_READ_TITLE)
    return meta


def _declared_shape(meta: dict) -> list[int]:
    shape = as_list(meta.get("shape"))
    if len(shape) != 4 or not all(isinstance(s, (int, float)) and int(s) == s and s >= 1 for s in shape):
        raise DatasetError(
            "E_DATA_SHAPE",
            f"meta.shape must be four positive integers [trials channels time modalities], got {shape}.",
            field="shape",
            title=_READ_TITLE,
        )
    return [int(s) for s in shape]


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------


def validate_contents(X: np.ndarray, meta: dict) -> list[Issue]:
    """Every problem with ``X`` and ``meta``, as a list of issues (errors and warnings)."""
    issues: list[Issue] = []

    def error(code: str, message: str, hint: str = "", field: str = "") -> None:
        issues.append(Issue("error", code, message, hint, field))

    def warn(code: str, message: str, hint: str = "", field: str = "") -> None:
        issues.append(Issue("warning", code, message, hint, field))

    if X.ndim != 4:
        error("E_DATA_SHAPE", f"X must have 4 dimensions (trials x channels x time x modalities), found {X.ndim}.")
        return issues
    N, R, T, M = X.shape
    if X.dtype == np.float64:
        warn("W_DATA_FLOAT64", "X is stored as double; PRISMT uses single precision.",
             "Store X as single to halve the file size.")
    elif X.dtype != np.float32:
        error("E_DATA_DTYPE", f"X must be single (float32), found {X.dtype}.", field="X")
        return issues
    if N < 1:
        error("E_DATA_EMPTY", "The dataset has no trials.")

    inf_at = np.argwhere(np.isinf(X))
    if inf_at.size:
        n, r, t, m = (inf_at[0] + 1).tolist()
        error(
            "E_DATA_INF",
            f"X contains {len(inf_at)} infinite value(s); the first is X({n},{r},{t},{m}).",
            "Replace infinite values with NaN (missing) before writing the dataset.",
            "X",
        )

    channels = meta.get("channels") or {}
    names = as_list(channels.get("names"))
    if len(names) != R:
        error("E_DATA_CHANNELS", f"channels.names has {len(names)} names but X has {R} channels.",
              "Give one name per channel (the second dimension of X).", "channels.names")
    else:
        if any(not isinstance(n, str) or not n.strip() for n in names):
            error("E_DATA_CHANNELS", "Every channel name must be non-empty text.", field="channels.names")
        dup = _duplicates(names)
        if dup:
            error("E_DATA_CHANNELS", f"Channel names must be unique; repeated: {', '.join(dup[:5])}.",
                  field="channels.names")
    for key in ("x", "y"):
        vals = channels.get(key)
        if vals is not None and len(as_list(vals)) != R:
            error("E_DATA_CHANNELS", f"channels.{key} must have one value per channel ({R}).",
                  field=f"channels.{key}")
    hemi = channels.get("hemisphere")
    if hemi is not None and len(as_list(hemi)) != R:
        error("E_DATA_CHANNELS", f"channels.hemisphere must have one value per channel ({R}).",
              field="channels.hemisphere")

    mods = meta.get("modalities") or {}
    mnames = as_list(mods.get("names"))
    if len(mnames) != M:
        error("E_DATA_MODALITIES", f"modalities.names has {len(mnames)} names but X has {M} modalities.",
              "Give one name per modality (the fourth dimension of X), e.g. 'calcium'.",
              "modalities.names")
    elif _duplicates(mnames):
        error("E_DATA_MODALITIES", "Modality names must be unique.", field="modalities.names")
    for key in ("units", "kinds"):
        vals = mods.get(key)
        if vals is not None and len(as_list(vals)) != M:
            error("E_DATA_MODALITIES", f"modalities.{key} must have one entry per modality ({M}).",
                  field=f"modalities.{key}")
    for kind in as_list(mods.get("kinds")):
        if kind not in MODALITY_KINDS:
            error("E_DATA_MODALITIES", f"Unknown modality kind '{kind}'; use one of {', '.join(MODALITY_KINDS)}.",
                  field="modalities.kinds")
    chan_sets = mods.get("channels")
    if chan_sets is not None:
        chan_sets = as_list(chan_sets)
        if len(chan_sets) != M:
            error("E_DATA_MODALITIES", f"modalities.channels must list channels for each of the {M} modalities.",
                  field="modalities.channels")
        else:
            for m, cs in enumerate(chan_sets):
                cs = as_list(cs)
                bad = [c for c in cs if not (isinstance(c, (int, float)) and int(c) == c and 1 <= c <= R)]
                if not cs or bad:
                    error("E_DATA_MODALITIES",
                          f"modalities.channels for '{mnames[m] if m < len(mnames) else m + 1}' must be "
                          f"channel numbers between 1 and {R}.", field="modalities.channels")
                elif _duplicates(cs):
                    error("E_DATA_MODALITIES", "A channel is listed twice in modalities.channels.",
                          field="modalities.channels")
                elif len(cs) < R and not issues_have_errors(issues):
                    outside = np.setdiff1d(np.arange(R), np.asarray(cs, dtype=int) - 1)
                    if np.isfinite(X[:, outside, :, m]).any():
                        warn("W_DATA_OUTSIDE_CHANNEL_SET",
                             f"Modality '{mnames[m] if m < len(mnames) else m + 1}' has values in channels "
                             "outside its channel list; they will be ignored.",
                             field="modalities.channels")

    time_ = meta.get("time") or {}
    times = time_.get("times_s")
    fs = time_.get("fs_hz")
    if times is not None:
        times = as_list(times)
        if len(times) != T:
            error("E_DATA_TIME", f"time.times_s has {len(times)} values but X has {T} time points.",
                  field="time.times_s")
        elif any(v is None for v in times) or np.any(np.diff(np.asarray(times, dtype=float)) <= 0):
            error("E_DATA_TIME", "time.times_s must be strictly increasing numbers.", field="time.times_s")
    elif fs is not None:
        if not isinstance(fs, (int, float)) or fs <= 0:
            error("E_DATA_TIME", f"time.fs_hz must be a positive number, got {fs}.", field="time.fs_hz")
    else:
        warn("W_DATA_NO_TIME", "The time axis is not described; times are shown as sample numbers.",
             "Set the sampling rate (and the time of the first sample) when creating the dataset.")

    trials = meta.get("trials") or {}
    columns = as_list(trials.get("columns"))
    col_names = []
    for i, col in enumerate(columns):
        where = f"trials.columns[{i + 1}]"
        if not isinstance(col, dict) or not isinstance(col.get("name"), str) or not col["name"].strip():
            error("E_DATA_COLUMNS", f"Column {i + 1} has no name.", field=where)
            continue
        name = col["name"]
        col_names.append(name)
        kind = col.get("type")
        if kind not in COLUMN_TYPES:
            error("E_DATA_COLUMNS", f"Column '{name}' has unknown type '{kind}'; use numeric, categorical or bool.",
                  field=where)
            continue
        values = as_list(col.get("values"))
        if len(values) != N:
            error("E_DATA_COLUMNS", f"Column '{name}' has {len(values)} values but there are {N} trials.",
                  "Each trial column needs exactly one value per trial.", where)
            continue
        if kind == "numeric" and any(v is not None and not isinstance(v, (int, float)) for v in values):
            error("E_DATA_COLUMNS", f"Column '{name}' is numeric but contains non-numbers.", field=where)
        if kind == "bool" and any(not isinstance(v, bool) for v in values):
            error("E_DATA_COLUMNS", f"Column '{name}' is true/false but contains other values.", field=where)
        if kind == "categorical":
            if any(v is not None and not isinstance(v, str) for v in values):
                error("E_DATA_COLUMNS", f"Column '{name}' is categorical but contains non-text values.",
                      field=where)
            cats = col.get("categories")
            if cats is not None:
                missing = sorted({v for v in values if v is not None} - set(as_list(cats)))
                if missing:
                    error("E_DATA_COLUMNS",
                          f"Column '{name}' has values that are not in its categories: {', '.join(missing[:5])}.",
                          field=where)
        for lab in as_list(col.get("labels")):
            if not isinstance(lab, dict) or "value" not in lab or not isinstance(lab.get("label"), str):
                error("E_DATA_COLUMNS", f"Column '{name}' has a malformed value label.", field=where)
                break
    dup = _duplicates(col_names)
    if dup:
        error("E_DATA_COLUMNS", f"Trial column names must be unique; repeated: {', '.join(dup)}.",
              field="trials.columns")
    roles = trials.get("roles") or {}
    for role in ("subject", "session"):
        ref = roles.get(role)
        if ref is not None and ref not in col_names:
            error("E_DATA_ROLES", f"The {role} role refers to a column '{ref}' that does not exist.",
                  field=f"trials.roles.{role}")
    if roles.get("subject") is None:
        warn("W_DATA_NO_SUBJECT",
             "No column is marked as the subject (animal). PRISMT cannot keep animals separate "
             "between training and testing.",
             "Mark the column that identifies each animal as the subject.")
    uid = trials.get("uid")
    if uid is not None:
        uid = as_list(uid)
        if len(uid) != N:
            error("E_DATA_COLUMNS", f"trials.uid has {len(uid)} values but there are {N} trials.",
                  field="trials.uid")
        elif _duplicates(uid):
            error("E_DATA_COLUMNS", "trials.uid values must be unique.", field="trials.uid")

    if issues_have_errors(issues):
        return issues

    for probe in as_list(meta.get("orientation_probes")):
        idx = as_list(probe.get("index")) if isinstance(probe, dict) else []
        if len(idx) != 4 or not all(isinstance(i, (int, float)) for i in idx):
            error("E_DATA_ORIENTATION", "An orientation probe is malformed.", field="orientation_probes")
            break
        n, r, t, m = (int(i) for i in idx)
        if not (0 <= n < N and 0 <= r < R and 0 <= t < T and 0 <= m < M):
            error("E_DATA_ORIENTATION", "An orientation probe points outside X.", field="orientation_probes")
            break
        expected = probe.get("value")
        got = X[n, r, t, m]
        same = np.isnan(got) if expected is None else (np.float32(expected) == got)
        if not same:
            error(
                "E_DATA_ORIENTATION",
                "The data array does not match the orientation checks stored with it, so its "
                "dimensions would be misread.",
                "The file was changed after it was written, or not written by prismt.writeDataset. "
                "Re-create it with prismt.writeDataset.",
                "orientation_probes",
            )
            break

    if M == len(mnames):
        for m in range(M):
            if not np.isfinite(X[..., m]).any():
                error("E_DATA_EMPTY_MODALITY", f"Modality '{mnames[m]}' contains no values (all NaN).",
                      field="modalities.names")
    empty_trials = int((~np.isfinite(X).reshape(N, -1).any(axis=1)).sum())
    if empty_trials:
        warn("W_DATA_EMPTY_TRIALS", f"{empty_trials} trial(s) contain no values at all; they will be skipped.")
    return issues


def issues_have_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


def _duplicates(values: Sequence) -> list:
    seen, dup = set(), []
    for v in values:
        key = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
        if key in seen and v not in dup:
            dup.append(v)
        seen.add(key)
    return dup


# ---------------------------------------------------------------------------------------
# Building the in-memory dataset
# ---------------------------------------------------------------------------------------


def _build(X: np.ndarray, meta: dict, path: Path | None, warnings: tuple[Issue, ...]) -> PrismtDataset:
    X = np.ascontiguousarray(X, dtype=np.float32)
    N, R, T, M = X.shape
    channels = meta.get("channels") or {}
    mods = meta.get("modalities") or {}
    mnames = [str(n) for n in as_list(mods.get("names"))]
    units = [str(u) for u in as_list(mods.get("units"))] or [""] * M
    kinds = [str(k) for k in as_list(mods.get("kinds"))] or ["neural"] * M
    chan_sets = as_list(mods.get("channels")) if mods.get("channels") is not None else [list(range(1, R + 1))] * M
    modalities = tuple(
        Modality(mnames[m], units[m], kinds[m], np.asarray(as_list(chan_sets[m]), dtype=np.int64) - 1)
        for m in range(M)
    )
    time_ = meta.get("time") or {}
    if time_.get("times_s") is not None:
        times = np.asarray(as_list(time_["times_s"]), dtype=np.float64)
    else:
        fs = float(time_.get("fs_hz") or 1.0)
        t0 = float(time_.get("t0_s") or 0.0)
        times = t0 + np.arange(T) / fs
    trials = meta.get("trials") or {}
    columns: dict[str, Column] = {}
    for col in as_list(trials.get("columns")):
        columns[col["name"]] = _make_column(col, N)
    roles = trials.get("roles") or {}
    uid = trials.get("uid")

    def optional(key: str) -> np.ndarray | None:
        vals = channels.get(key)
        return None if vals is None else np.asarray([np.nan if v is None else v for v in as_list(vals)], float)

    hemi = channels.get("hemisphere")
    return PrismtDataset(
        X=X,
        channel_names=tuple(str(n) for n in as_list(channels.get("names"))),
        modalities=modalities,
        times_s=times,
        event=str(time_.get("event") or ""),
        columns=columns,
        subject=roles.get("subject"),
        session=roles.get("session"),
        trial_uid=None if uid is None else np.asarray(as_list(uid), dtype=object),
        channel_x=optional("x"),
        channel_y=optional("y"),
        hemisphere=None if hemi is None else tuple(str(h) for h in as_list(hemi)),
        atlas=channels.get("atlas"),
        meta=meta,
        warnings=warnings,
        path=path,
    )


def _make_column(col: dict, n: int) -> Column:
    kind = col["type"]
    values = as_list(col.get("values"))
    if kind == "numeric":
        arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    elif kind == "bool":
        arr = np.array([bool(v) for v in values], dtype=bool)
    else:
        arr = np.array([None if v is None else str(v) for v in values], dtype=object)
    labels = {float(lab["value"]): str(lab["label"]) for lab in as_list(col.get("labels"))}
    cats = tuple(str(c) for c in as_list(col.get("categories")))
    return Column(col["name"], kind, arr, cats, labels)


def _with_fingerprint(ds: PrismtDataset) -> PrismtDataset:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(ds.X).tobytes())
    canonical = {
        "channels": ds.channel_names,
        "modalities": [(m.name, m.channels.tolist()) for m in ds.modalities],
        "times": np.round(ds.times_s, 9).tolist(),
        "columns": [(c.name, c.kind, c.as_text().tolist()) for c in ds.columns.values()],
    }
    h.update(json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"))
    return _replace(ds, fingerprint=h.hexdigest())


def _replace(ds: PrismtDataset, **changes: Any) -> PrismtDataset:
    from dataclasses import replace

    return replace(ds, **changes)


# ---------------------------------------------------------------------------------------
# Summary (what MATLAB shows on the Data tab and `prismt validate --json` prints)
# ---------------------------------------------------------------------------------------


def summarize_dataset(ds: PrismtDataset, *, max_values: int = 50) -> dict:
    X = ds.X
    finite = np.isfinite(X)
    subj = ds.subject_ids()
    sess = ds.session_keys()
    cols = []
    for c in ds.columns.values():
        text = c.as_text()
        miss = c.missing()
        uniq, counts = np.unique(text[~miss].astype(str), return_counts=True) if (~miss).any() else ([], [])
        order = np.argsort(-np.asarray(counts), kind="stable") if len(counts) else []
        cols.append({
            "name": c.name,
            "type": c.kind,
            "n_unique": int(len(uniq)),
            "n_missing": int(miss.sum()),
            "values": [str(uniq[i]) for i in order[:max_values]],
            "counts": [int(counts[i]) for i in order[:max_values]],
            "constant_within_session": _constant_within(text, sess) if sess is not None else None,
            "constant_within_subject": _constant_within(text, subj) if subj is not None else None,
        })
    return {
        "path": None if ds.path is None else str(ds.path),
        "fingerprint": ds.fingerprint,
        "n_trials": ds.n_trials,
        "n_channels": ds.n_channels,
        "n_time": ds.n_time,
        "n_modalities": ds.n_modalities,
        "channel_names": list(ds.channel_names),
        "modalities": [
            {
                "name": m.name,
                "unit": m.unit,
                "kind": m.kind,
                "n_channels": int(len(m.channels)),
                "missing_fraction": float(1.0 - finite[:, m.channels, :, i].mean()),
            }
            for i, m in enumerate(ds.modalities)
        ],
        "time": {
            "first_s": float(ds.times_s[0]),
            "last_s": float(ds.times_s[-1]),
            "step_s": float(np.median(np.diff(ds.times_s))) if ds.n_time > 1 else None,
            "event": ds.event,
        },
        "roles": {"subject": ds.subject, "session": ds.session},
        "n_subjects": None if subj is None else int(len(set(subj))),
        "n_sessions": None if sess is None else int(len(set(sess))),
        "columns": cols,
        "n_tokens_full": len(ds.token_pairs()) * ds.n_time + 1,
        "warnings": [w.to_dict() for w in ds.warnings],
    }


def _constant_within(values: np.ndarray, groups: np.ndarray) -> bool:
    seen: dict[str, str] = {}
    for g, v in zip(groups, values):
        if seen.setdefault(g, v) != v:
            return False
    return True


# ---------------------------------------------------------------------------------------
# Writing (for Python users and tests; MATLAB users call prismt.writeDataset)
# ---------------------------------------------------------------------------------------


def write_dataset(
    path: str | os.PathLike,
    X: np.ndarray,
    trials: Mapping[str, Sequence[Any]] | None = None,
    *,
    channel_names: Sequence[str] | None = None,
    channel_x: Sequence[float] | None = None,
    channel_y: Sequence[float] | None = None,
    hemisphere: Sequence[str] | None = None,
    atlas: str | None = None,
    modality_names: Sequence[str] | None = None,
    modality_units: Sequence[str] | None = None,
    modality_kinds: Sequence[str] | None = None,
    modality_channels: Sequence[Sequence[int]] | None = None,
    times_s: Sequence[float] | None = None,
    fs_hz: float | None = None,
    t0_s: float | None = None,
    event: str = "",
    subject: str | None = None,
    session: str | None = None,
    trial_uid: Sequence[str] | None = None,
    value_labels: Mapping[str, Mapping[float, str]] | None = None,
    categories: Mapping[str, Sequence[str]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a PRISMT dataset. ``X`` is trials x channels x time (x modalities).

    ``modality_channels`` uses 1-based channel numbers, like MATLAB. ``.mat`` and ``.h5``
    files are MATLAB v7.3 files that MATLAB's ``load`` opens; ``.npz`` is also accepted.
    """
    path = Path(path).expanduser()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.ndim != 4:
        raise DatasetError("E_DATA_SHAPE", "X must be trials x channels x time (x modalities).")
    N, R, T, M = X.shape
    meta = build_meta(
        X,
        trials or {},
        channel_names=channel_names,
        channel_x=channel_x,
        channel_y=channel_y,
        hemisphere=hemisphere,
        atlas=atlas,
        modality_names=modality_names,
        modality_units=modality_units,
        modality_kinds=modality_kinds,
        modality_channels=modality_channels,
        times_s=times_s,
        fs_hz=fs_hz,
        t0_s=t0_s,
        event=event,
        subject=subject,
        session=session,
        trial_uid=trial_uid,
        value_labels=value_labels,
        categories=categories,
        provenance=provenance,
    )
    issues = validate_contents(X, meta)
    raise_if_errors(issues, DatasetError, title="The dataset could not be written")
    text = json.dumps(meta, allow_nan=False, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez(tmp, X=X, meta_json=np.array(text), prismt_format=np.array(FORMAT_TAG),
                 prismt_version=np.array(float(FORMAT_VERSION)))
        os.replace(tmp, path)
        return path
    matv73.write_mat73(path, {
        "prismt_format": matv73.text_to_uint8(FORMAT_TAG),
        "prismt_version": np.array(float(FORMAT_VERSION)),
        "X": X,
        "meta_json": matv73.text_to_uint8(text),
    })
    return path


def build_meta(X: np.ndarray, trials: Mapping[str, Sequence[Any]], **kw: Any) -> dict:
    N, R, T, M = X.shape
    value_labels = kw.get("value_labels") or {}
    categories = kw.get("categories") or {}
    columns = []
    for name, values in trials.items():
        columns.append(_column_meta(str(name), values, N, value_labels.get(name), categories.get(name)))
    times_s = kw.get("times_s")
    fs_hz = kw.get("fs_hz")
    time_meta: dict[str, Any] = {"event": kw.get("event") or ""}
    if times_s is not None:
        time_meta["times_s"] = [float(t) for t in times_s]
    if fs_hz is not None:
        time_meta["fs_hz"] = float(fs_hz)
        time_meta["t0_s"] = float(kw.get("t0_s") or 0.0)
    mchan = kw.get("modality_channels")

    def listify(v: Sequence | None, cast=str) -> list | None:
        return None if v is None else [cast(x) for x in v]

    return {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "created_by": f"prismt-python {__version__}",
        "shape": [N, R, T, M],
        "channels": {
            "names": listify(kw.get("channel_names")) or [f"ch{r + 1:02d}" for r in range(R)],
            "x": listify(kw.get("channel_x"), float),
            "y": listify(kw.get("channel_y"), float),
            "hemisphere": listify(kw.get("hemisphere")),
            "atlas": kw.get("atlas"),
        },
        "modalities": {
            "names": listify(kw.get("modality_names")) or [f"signal{m + 1}" for m in range(M)],
            "units": listify(kw.get("modality_units")) or [""] * M,
            "kinds": listify(kw.get("modality_kinds")) or ["neural"] * M,
            "channels": None if mchan is None else [[int(c) for c in cs] for cs in mchan],
        },
        "time": time_meta,
        "trials": {
            "columns": columns,
            "roles": {"subject": kw.get("subject"), "session": kw.get("session")},
            "uid": listify(kw.get("trial_uid")),
        },
        "orientation_probes": orientation_probes(X),
        "provenance": dict(kw.get("provenance") or {}),
    }


def _column_meta(name: str, values: Sequence[Any], n: int, labels: Mapping | None, cats: Sequence | None) -> dict:
    arr = np.asarray(values)
    if arr.shape[0] != n:
        raise DatasetError("E_DATA_COLUMNS", f"Column '{name}' has {arr.shape[0]} values but there are {n} trials.")
    if arr.dtype == np.bool_:
        return {"name": name, "type": "bool", "values": [bool(v) for v in arr]}
    if np.issubdtype(arr.dtype, np.number):
        vals = [None if not np.isfinite(v) else (int(v) if float(v).is_integer() else float(v)) for v in arr.astype(float)]
        out: dict[str, Any] = {"name": name, "type": "numeric", "values": vals}
        if labels:
            out["labels"] = [{"value": float(k), "label": str(v)} for k, v in labels.items()]
        return out
    vals = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in arr.tolist()]
    out = {"name": name, "type": "categorical", "values": vals}
    if cats is not None:
        out["categories"] = [str(c) for c in cats]
    return out


def orientation_probes(X: np.ndarray, k: int = 8) -> list[dict]:
    """Values at a few fixed positions, so a reader can prove it restored the axes."""
    flat = X.reshape(-1)
    positions = sorted(set(np.linspace(0, flat.size - 1, k).round().astype(int).tolist()))
    finite = np.flatnonzero(np.isfinite(flat))
    if finite.size:
        positions = sorted(set(positions) | {int(finite[0]), int(finite[-1]), int(finite[finite.size // 2])})
    probes = []
    for p in positions:
        idx = np.unravel_index(p, X.shape)
        v = flat[p]
        probes.append({"index": [int(i) for i in idx], "value": None if not np.isfinite(v) else float(v)})
    return probes
