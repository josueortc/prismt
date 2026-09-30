"""Run settings: defaults, presets and validation.

A run config is a plain nested dict with the same shape as ``config.json``. Defaults,
allowed values and the plain-language help for every field live in
``resources/config_schema.json``, which the MATLAB app reads too, so a default is never
written twice. :func:`load_config` merges, in order, the defaults, the chosen preset and
the user's values, then validates everything and reports every problem at once.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from prismt import FORMATS, RESOURCES
from prismt.errors import ConfigError, Issue, raise_if_errors
from prismt.jsonutil import as_list

SCHEMA: dict = json.loads((RESOURCES / "config_schema.json").read_text(encoding="utf-8"))
FIELDS: dict[str, dict] = {f["path"]: f for f in SCHEMA["fields"]}
PRESETS: dict[str, dict] = {p["name"]: p for p in SCHEMA["presets"]}
_SECTIONS = {p.split(".")[0] for p in FIELDS if "." in p}
_PATH_FIELDS = [p for p, f in FIELDS.items() if f["type"] == "path"]
_MISSING = object()
_TITLE = "The run settings are not valid"


def get(cfg: dict, path: str, default: Any = _MISSING) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(path)
            return default
        node = node[part]
    return node


def set_value(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def defaults() -> dict:
    cfg: dict = {}
    for path, spec in FIELDS.items():
        set_value(cfg, path, copy.deepcopy(spec.get("default")))
    return cfg


def preset_values(name: str) -> dict[str, Any]:
    if name == "custom":
        return {}
    if name not in PRESETS:
        raise ConfigError("E_CFG_PRESET", f"Unknown preset '{name}'.",
                          hint=f"Choose one of: {', '.join([*PRESETS, 'custom'])}.", field="preset", title=_TITLE)
    return {v["path"]: v["value"] for v in PRESETS[name]["values"]}


def load_config(source: str | Path | dict, *, base_dir: str | Path | None = None) -> dict:
    """Merge defaults < preset < user values, validate, and resolve relative paths.

    ``source`` is a config dict or a JSON file. Relative paths inside it are resolved
    against the file's folder (or ``base_dir``).
    """
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError("E_CFG_MISSING", f"The settings file does not exist: {path}", title=_TITLE) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError("E_CFG_JSON", f"The settings file is not valid JSON ({exc}).", title=_TITLE) from exc
        base_dir = base_dir or path.parent
    else:
        user = copy.deepcopy(source)
    if not isinstance(user, dict):
        raise ConfigError("E_CFG_JSON", "The settings must be a JSON object.", title=_TITLE)
    issues: list[Issue] = []
    flat = _flatten_user(user, issues)
    version = flat.pop("config_version", FORMATS["config"])
    if isinstance(version, (int, float)) and version > FORMATS["config"]:
        raise ConfigError("E_CFG_NEWER", f"These settings use format version {version}; this PRISMT reads up to "
                          f"{FORMATS['config']}.", hint="Update PRISMT.", title=_TITLE)
    cfg = defaults()
    preset = flat.get("preset", cfg["preset"])
    if isinstance(preset, str):
        try:
            for p, v in preset_values(preset).items():
                set_value(cfg, p, copy.deepcopy(v))
        except ConfigError as err:
            issues.append(Issue("error", err.code, err.message, err.hint, err.field))
    for p, v in flat.items():
        set_value(cfg, p, v)
    for p, spec in FIELDS.items():
        set_value(cfg, p, _check_field(p, spec, get(cfg, p), issues))
    _cross_checks(cfg, issues)
    raise_if_errors(issues, ConfigError, title=_TITLE)
    base = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
    for p in _PATH_FIELDS:
        value = get(cfg, p)
        if isinstance(value, str) and value:
            set_value(cfg, p, str((base / Path(value).expanduser()).resolve()))
    cfg["config_version"] = FORMATS["config"]
    return cfg


def _flatten_user(user: dict, issues: list[Issue], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in user.items():
        path = f"{prefix}{key}"
        if path == "config_version" or path in FIELDS:
            flat[path] = value
        elif isinstance(value, dict) and (path in _SECTIONS or any(f.startswith(path + ".") for f in FIELDS)):
            flat.update(_flatten_user(value, issues, path + "."))
        else:
            close = difflib.get_close_matches(path, [*FIELDS, *_SECTIONS], n=1, cutoff=0.6)
            hint = f"Did you mean '{close[0]}'?" if close else "Remove it, or check the spelling."
            issues.append(Issue("error", "E_CFG_UNKNOWN", f"Unknown setting '{path}'.", hint, path))
    return flat


def _check_field(path: str, spec: dict, value: Any, issues: list[Issue]) -> Any:
    def bad(message: str, hint: str = "") -> Any:
        issues.append(Issue("error", "E_CFG_VALUE", f"{spec.get('label', path)}: {message}", hint, path))
        return value

    kind = spec["type"]
    if value is None:
        if spec.get("required"):
            return bad("this setting is required.")
        if spec.get("nullable") or spec.get("default") is None:
            return None
        return bad("a value is required.")
    if kind == "choice":
        if value not in spec["choices"]:
            return bad(f"'{value}' is not one of {', '.join(map(str, spec['choices']))}.")
        return value
    if kind == "bool":
        return value if isinstance(value, bool) else bad("must be true or false.")
    if kind in ("text", "path"):
        return value if isinstance(value, str) else bad("must be text.")
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            return bad(f"must be a whole number, got {value!r}.")
        value = int(value)
        if "choices" in spec and value not in spec["choices"]:
            return bad(f"must be one of {', '.join(map(str, spec['choices']))}.")
        return _check_range(value, spec, bad)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return bad(f"must be a number, got {value!r}.")
        return _check_range(float(value), spec, bad)
    if kind == "list":
        return [str(v) for v in as_list(value)]
    if kind == "range":
        vals = as_list(value)
        if len(vals) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            return bad("must be two numbers [start, end].")
        if vals[0] >= vals[1]:
            return bad("the start must be before the end.")
        return [float(vals[0]), float(vals[1])]
    if kind == "folds":
        if value in ("auto", "loo"):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) == value and value >= 1:
            return int(value)
        return bad("must be auto, loo or a whole number of folds (1 = a single split).")
    if kind == "patch":
        if value == "all":
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) == value and value >= 1:
            return int(value)
        return bad("must be a whole number of time bins (1 or more) or all.")
    if kind == "filters":
        return _check_filters(as_list(value), bad)
    if kind == "classes":
        return _check_classes(as_list(value), bad)
    return bad(f"internal: unknown field type {kind}")


def _check_range(value: float, spec: dict, bad) -> Any:
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and (value < lo or (spec.get("exclusive_min") and value == lo)):
        return bad(f"must be {'greater than' if spec.get('exclusive_min') else 'at least'} {lo}, got {value}.")
    if hi is not None and value > hi:
        return bad(f"must be at most {hi}, got {value}.")
    return value


def _check_filters(filters: list, bad) -> list:
    out = []
    for i, f in enumerate(filters):
        if not isinstance(f, dict) or not isinstance(f.get("column"), str):
            bad(f"filter {i + 1} needs a column name.")
            continue
        op = f.get("op", "in")
        if op not in ("in", "not_in", "range"):
            bad(f"filter on '{f['column']}': unknown operation '{op}' (use in, not_in or range).")
            continue
        values = as_list(f.get("values"))
        if op == "range" and (len(values) != 2 or not all(v is None or isinstance(v, (int, float)) for v in values)):
            bad(f"filter on '{f['column']}': a range needs two numbers [low, high] (either may be empty).")
            continue
        if op != "range" and not values:
            bad(f"filter on '{f['column']}': choose at least one value.")
            continue
        out.append({"column": f["column"], "op": op, "values": values})
    return out


def _check_classes(classes: list, bad) -> list:
    out, names, seen = [], set(), {}
    for i, c in enumerate(classes):
        if not isinstance(c, dict):
            bad(f"class {i + 1} must have a name and values.")
            continue
        values = as_list(c.get("values"))
        name = c.get("name")
        if name is None and len(values) == 1:
            name = str(values[0])
        if not isinstance(name, str) or not name.strip():
            bad(f"class {i + 1} needs a name.")
            continue
        if not values:
            bad(f"class '{name}' has no values.")
            continue
        if name in names:
            bad(f"two classes are both called '{name}'.", "Give each class a different name; to merge values, list them in one class.")
            continue
        for v in values:
            key = str(v)
            if key in seen:
                bad(f"value '{key}' is in both '{seen[key]}' and '{name}'.")
            seen[key] = name
        names.add(name)
        out.append({"name": name, "values": values})
    return out


def _cross_checks(cfg: dict, issues: list[Issue]) -> None:
    def err(code: str, message: str, hint: str, field: str) -> None:
        issues.append(Issue("error", code, message, hint, field))

    model = cfg["model"]
    if isinstance(model["d_model"], int) and isinstance(model["n_heads"], int) and model["d_model"] % model["n_heads"]:
        divisors = [h for h in range(1, 17) if model["d_model"] % h == 0]
        err("E_CFG_HEADS", f"The model width ({model['d_model']}) must be divisible by the number of attention heads "
            f"({model['n_heads']}).", f"Use {', '.join(map(str, divisors))} heads.", "model.n_heads")
    if cfg["task"] == "classify" and not cfg["labels"]["column"]:
        err("E_CFG_NO_LABEL", "Classification needs a label column (what the classes are).",
            "Choose the trial column to classify, for example phase or stim.", "labels.column")
    if cfg["task"] == "mae" and model["init_from"]:
        err("E_CFG_INIT", "Starting from an autoencoder is only for classification runs.", "", "model.init_from")
    split = cfg["split"]
    if isinstance(split["test_fraction"], float) and isinstance(split["val_fraction"], float) \
            and split["test_fraction"] + split["val_fraction"] >= 0.8:
        err("E_CFG_SPLIT", "Test and validation fractions leave too little data for training.",
            "Keep their sum below 0.8.", "split.test_fraction")
    if cfg["task"] == "mae" and cfg["mae"]["mask"]["strategy"] == "forecast" and model["time_patch"] == "all":
        err("E_CFG_MASK", "Forecasting needs several time steps per channel, but 'Time bins per token' is all.",
            "Use a smaller number of time bins per token.", "model.time_patch")


def config_hash(cfg: dict) -> str:
    """Identity of everything that affects training (not names, output folders or tuning)."""
    relevant = {k: v for k, v in cfg.items() if k not in ("name", "output", "hpo", "config_version")}
    text = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def schema_for_matlab() -> dict:
    """What ``prismt defaults --json`` prints: the schema plus the resolved default config."""
    return {"schema": SCHEMA, "defaults": defaults(), "formats": FORMATS}
