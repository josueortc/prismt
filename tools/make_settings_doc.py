"""Write docs/hyperparameters.md from src/prismt/resources/config_schema.json.

The app's tooltips, the config validation and this page all come from the same file, so
they cannot disagree. Run after changing the schema (a test checks the page is current):

    python tools/make_settings_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "src" / "prismt" / "resources" / "config_schema.json"
OUT = ROOT / "docs" / "hyperparameters.md"

INTRO = """\
# Settings reference

Every setting PRISMT understands, with its default and what it does. The same text appears
as tooltips in the app (Model & training tab > Show all settings) and in
`prismt.defaultConfig`'s help. This page is generated from
`src/prismt/resources/config_schema.json` by `tools/make_settings_doc.py`; do not edit it by hand.

## How settings are combined

A run's settings are built in three layers, each overriding the one before:

1. the defaults below;
2. the chosen **preset** (Quick test, Standard or Paper), which sets model size and training
   budget;
3. what you set yourself, in the app or in the `cfg` struct of a script.

The resolved settings of every run are saved in its `config.json`, so a run can always be
repeated exactly (Results tab > Export as MATLAB script).

## What to change first

Most analyses only need the **Task** tab (what to predict, which trials, how to test) and a
preset. When a result looks wrong, the usual fixes are, in order:

- **The model does no better than the reference models.** Train longer (`train.epochs`,
  `train.min_epochs`), or try the Standard preset. Check the learning curves: a validation
  loss that is still falling at the last epoch means the budget was too small.
- **Validation loss rises early while training loss keeps falling (overfitting).** Use a
  smaller model (`model.d_model`, `model.n_layers`), more dropout (`model.dropout`), or more
  data (fewer trial filters, all sessions).
- **Training is too slow or runs out of memory.** Use fewer tokens: a shorter
  `preprocess.time_window_s`, wider `preprocess.bin_width_s`, or `model.time_patch` > 1
  (or `all`, one token per channel). The Check reports the number of tokens per trial; above
  about 1,100 a laptop becomes slow.
- **Loss becomes NaN.** Lower `train.lr` (for example 3e-4).
- **You want the best settings found for you.** Use automatic tuning (`hpo`), ideally on a
  cluster; it compares settings on validation trials only and tests the winner once.
"""

SECTION_NOTES = {
    "split": "How trials are divided into training, validation (choosing when to stop) and test "
             "(the reported score). See docs/results-and-pitfalls.md for why testing on new animals matters.",
    "mae": "Only used by masked-autoencoder runs.",
    "hpo": "Only used by tuning runs (Tune automatically in the app, or prismt.hpo).",
}


def fmt(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(fmt(v) for v in value) if value else "(empty)"
    return str(value)


def allowed(field: dict) -> str:
    if field.get("choices"):
        return " / ".join(f"`{c}`" for c in field["choices"])
    lo, hi = field.get("min"), field.get("max")
    kind = {"int": "whole number", "number": "number", "bool": "on / off", "range": "start, end (s)",
            "list": "list", "text": "text", "path": "file or folder", "patch": "number or `all`",
            "folds": "number or `auto`", "filters": "filters", "classes": "classes"}.get(field["type"], field["type"])
    if lo is not None and hi is not None:
        return f"{kind}, {fmt(lo)} to {fmt(hi)}"
    if lo is not None:
        return f"{kind}, at least {fmt(lo)}"
    return kind


def render(schema: dict) -> str:
    presets = schema["presets"]
    preset_values = {p["name"]: {v["path"]: v["value"] for v in p["values"]} for p in presets}
    lines = [INTRO, "## Presets", "", "| Preset | What it is for |", "|---|---|"]
    for p in presets:
        lines.append(f"| {p['label']} (`{p['name']}`) | {p['description']} |")
    lines += ["", "Settings that the presets change:", ""]
    preset_paths = []
    for p in presets:
        for v in p["values"]:
            if v["path"] not in preset_paths:
                preset_paths.append(v["path"])
    lines.append("| Setting | " + " | ".join(p["label"] for p in presets) + " |")
    lines.append("|---|" + "---|" * len(presets))
    for path in preset_paths:
        cells = [fmt(preset_values[p["name"]].get(path, "—")) for p in presets]
        lines.append(f"| `{path}` | " + " | ".join(cells) + " |")
    lines.append("")
    fields = schema["fields"]
    for section in schema["sections"]:
        key = section["key"]
        rows = [f for f in fields if f["path"] == key or f["path"].startswith(key + ".")]
        if not rows:
            continue
        lines += [f"## {section['label']} (`{key}`)", ""]
        if key in SECTION_NOTES:
            lines += [SECTION_NOTES[key], ""]
        lines += ["| Setting | Default | Allowed | What it does |", "|---|---|---|---|"]
        for f in rows:
            name = f"`{f['path']}`" + (" (advanced)" if f.get("advanced") else "")
            default = fmt(f.get("default"))
            if f["path"] in preset_paths:
                default += " (preset)"
            help_text = f"**{f['label']}.** {f['help']}".replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {default} | {allowed(f)} | {help_text} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    text = render(json.loads(SCHEMA.read_text(encoding="utf-8")))
    if "--check" in sys.argv:
        return 0 if OUT.read_text(encoding="utf-8") == text else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
