"""The documentation only mentions things that exist: links, MATLAB functions, commands, settings."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), *sorted((REPO / "docs" / "dev").glob("*.md"))]
# pages written before the rebuild, kept for reference (untracked or legacy)
DOCS = [d for d in DOCS if d.name not in ("MATLAB_GUI_Tutorial.md",)]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _anchors(path: Path) -> set[str]:
    out = set()
    for line in _text(path).splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            out.add(re.sub(r"[^\w\- ]", "", title).replace(" ", "-"))
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_links_resolve(doc):
    for target in re.findall(r"\]\(([^)\s]+)\)", _text(doc)):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file, _, anchor = target.partition("#")
        path = (doc.parent / file).resolve() if file else doc
        assert path.exists(), f"{doc.name}: broken link {target}"
        if anchor and path.suffix == ".md":
            assert anchor in _anchors(path), f"{doc.name}: no section #{anchor} in {path.name}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_matlab_functions_exist(doc):
    root = REPO / "matlab" / "+prismt"
    for name in set(re.findall(r"(?<![/\w])prismt\.((?:[a-z]+\.)*[a-zA-Z]+)\b", _text(doc))):
        parts = name.split(".")
        if parts[0] in ("dataset", "status", "config", "defaults", "sif", "git"):   # file names, schemas, URLs
            continue
        py = REPO / "src" / "prismt" / Path(*parts)
        if py.with_suffix(".py").exists() or (py / "__init__.py").exists():      # a Python module
            continue
        folder = root.joinpath(*("+" + p for p in parts[:-1]))
        candidates = [folder / f"{parts[-1]}.m", folder / f"+{parts[-1]}"]
        # methods of classes (e.g. prismt.run.LocalRun.attach) are not checked beyond the class
        if len(parts) >= 2:
            candidates.append(root.joinpath(*("+" + p for p in parts[:-2])) / f"{parts[-2]}.m")
        assert any(c.exists() for c in candidates), f"{doc.name}: prismt.{name} does not exist"


def test_cli_commands_exist():
    help_text = subprocess.run([sys.executable, "-m", "prismt", "--help"], capture_output=True, text=True).stdout
    for doc in DOCS:
        for cmd in set(re.findall(r"python -m prismt ([a-z]+)", _text(doc))):
            assert cmd in help_text, f"{doc.name}: python -m prismt {cmd} is not a command"


def test_settings_named_in_docs_exist():
    schema = json.loads((REPO / "src" / "prismt" / "resources" / "config_schema.json").read_text())
    paths = {f["path"] for f in schema["fields"]}
    sections = {s["key"] for s in schema["sections"]}
    for doc in DOCS:
        if doc.name == "hyperparameters.md":
            continue
        for name in set(re.findall(r"`((?:" + "|".join(sections) + r")\.[a-z_.]+)`", _text(doc))):
            if name.rsplit(".", 1)[-1] in ("pt", "json", "csv", "mat", "txt", "sh"):   # a file name
                continue
            assert name in paths, f"{doc.name}: setting {name} does not exist"


def test_settings_reference_is_current():
    out = subprocess.run([sys.executable, str(REPO / "tools" / "make_settings_doc.py"), "--check"])
    assert out.returncode == 0, "docs/hyperparameters.md is out of date: run python tools/make_settings_doc.py"
