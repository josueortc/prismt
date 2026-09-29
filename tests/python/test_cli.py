"""The command line as MATLAB uses it: one JSON object on stdout, meaningful exit codes."""

from __future__ import annotations

import json
import subprocess
import sys


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "prismt", *args], capture_output=True, text=True, timeout=300)


def test_version():
    out = run("--version")
    assert out.returncode == 0 and out.stdout.startswith("prismt ")


def test_no_command_prints_help_and_exits_2():
    out = run()
    assert out.returncode == 2 and "usage" in out.stderr


def test_synth_then_validate_json(tmp_path):
    path = tmp_path / "s.mat"
    out = run("synth", "--profile", "tiny", "--out", str(path), "--json")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["path"] == str(path)
    out = run("validate", str(path), "--json")
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["ok"] and report["summary"]["n_trials"] == 128


def test_validate_bad_file_exits_2_with_error_json(tmp_path):
    bad = tmp_path / "bad.mat"
    bad.write_bytes(b"not a mat file")
    out = run("validate", str(bad), "--json")
    assert out.returncode == 2
    report = json.loads(out.stdout)
    assert not report["ok"] and report["errors"][0]["code"] == "E_DATA_UNREADABLE"
    assert len(out.stdout.strip().splitlines()) == 1  # exactly one JSON object


def test_doctor_json_reports_checks():
    out = run("doctor", "--json")
    report = json.loads(out.stdout)
    assert {"ok", "checks", "device", "packages", "prismt"} <= report.keys()
    assert out.returncode in (0, 4)
    assert any(c["name"] == "torch" for c in report["checks"])
