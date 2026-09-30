"""The command line as MATLAB uses it: one JSON object on stdout, meaningful exit codes."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys

import pytest


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


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1 on this platform")
def test_sigusr1_stops_training_and_keeps_results(tmp_path, tiny_dataset):
    """What a cluster job receives 5 minutes before its time limit (submit.sh --signal B:USR1@300)."""
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"task": "mae", "dataset": {"path": str(tiny_dataset)},
                               "train": {"epochs": 1000, "min_epochs": 1000, "device": "cpu"},
                               "model": {"d_model": 16, "n_layers": 1}}))
    out = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    proc = subprocess.Popen([sys.executable, "-m", "prismt", "train", "--config", str(cfg), "--run-dir", str(out)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while time.time() - t0 < 120 and not (out / "history.csv").exists():
        time.sleep(0.5)
    time.sleep(1)
    proc.send_signal(signal.SIGUSR1)
    _, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err[-2000:]
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "finished"
    assert (out / "metrics.json").exists(), "the best model so far is evaluated"
    lines = json.loads((out / "metrics.json").read_text())["summary_lines"]
    assert any("stopped early" in line for line in lines), lines
