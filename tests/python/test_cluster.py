"""Cluster job folders, run end to end under a fake Slurm (tests/fake_slurm/sbatch)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prismt.cluster import SCRIPTS, write_job_folder

FAKE = Path(__file__).resolve().parents[1] / "fake_slurm"


def submit(job: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PRISMT_PYTHON=sys.executable, FAKE_SLURM_LOG=str(tmp_path / "slurm.log"),
               PATH=f"{FAKE}:{os.environ['PATH']}")
    env.pop("PYTHONPATH", None)
    return subprocess.run(["bash", str(job / "submit.sh")], env=env, capture_output=True, text=True, timeout=1200)


def test_job_folder_contents(tmp_path, tiny_dataset):
    job = write_job_folder({"task": "classify", "dataset": {"path": str(tiny_dataset)}}, tmp_path,
                           {"gpus": 0}, n_folds=1)
    for f in (*SCRIPTS, "job.env", "config.json", "README.txt", "prismt_src/src/prismt/__init__.py"):
        assert (job / f).exists(), f
    assert (job / "logs").is_dir()
    for script in SCRIPTS:
        text = (job / script).read_bytes()
        assert b"\r" not in text and b"CUDA_VISIBLE_DEVICES=" not in text
        assert subprocess.run(["bash", "-n", str(job / script)]).returncode == 0
        if script.endswith(".sbatch"):
            assert b'$(dirname "$0")' not in text and b"SLURM_SUBMIT_DIR" in text
    env = (job / "job.env").read_text()
    cfg = json.loads((job / "config.json").read_text())
    assert "PRISMT_GPUS=0" in env and cfg["output"]["root"] == "results"
    # without a path on the cluster, the dataset travels inside the (movable) job folder
    assert cfg["dataset"]["path"] == f"data/{tiny_dataset.name}" and (job / "data" / tiny_dataset.name).is_file()
    remote = write_job_folder({"task": "classify", "dataset": {"path": str(tiny_dataset)}}, tmp_path / "r",
                              {"gpus": 0}, dataset_remote="~/data/tiny.mat")
    assert json.loads((remote / "config.json").read_text())["dataset"]["path"] == "~/data/tiny.mat"
    assert not (remote / "data").exists()


@pytest.mark.slow
def test_cross_validation_array_then_combine(tmp_path, tiny_dataset):
    cfg = {"task": "classify", "dataset": {"path": str(tiny_dataset)}, "labels": {"column": "stim"},
           "train": {"epochs": 2, "min_epochs": 0, "device": "cpu"}, "model": {"d_model": 16, "n_layers": 1}}
    job = write_job_folder(cfg, tmp_path, {"gpus": 0}, n_folds=4)
    out = submit(job, tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    log = (tmp_path / "slurm.log").read_text()
    assert "failed" not in log, (job / "logs").iterdir()
    assert "--signal B:USR1@300" in log, "training jobs must be warned before the time limit"
    assert "summarize.sh" in (job / "README.txt").read_text()
    res = job / "results"
    assert sorted(p.name for p in res.glob("fold_*")) == ["fold_01", "fold_02", "fold_03", "fold_04"]
    assert not (res / "metrics.json").exists()
    env = dict(os.environ, PRISMT_PYTHON=sys.executable, PATH=f"{FAKE}:{os.environ['PATH']}")
    comb = subprocess.run(["bash", str(job / "summarize.sh")], env=env, capture_output=True, text=True, timeout=600)
    assert comb.returncode == 0, comb.stderr[-2000:]
    metrics = json.loads((res / "metrics.json").read_text())
    assert metrics["n_folds"] == 4 and metrics["test"]["n"] == 128


@pytest.mark.slow
def test_tuning_workers_then_final_retraining(tmp_path, tiny_dataset):
    cfg = {"task": "classify", "dataset": {"path": str(tiny_dataset)}, "labels": {"column": "stim"},
           "train": {"epochs": 2, "min_epochs": 0, "device": "cpu"}, "model": {"d_model": 16, "n_layers": 1},
           "split": {"folds": 1, "test_fraction": 0.25, "test_on": "trial"},
           "hpo": {"n_trials": 3, "final_seeds": 2, "space": "small"}}
    job = write_job_folder(cfg, tmp_path, {"gpus": 0}, mode="hpo", hpo_workers=2)
    out = submit(job, tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    log = (tmp_path / "slurm.log").read_text()
    assert "failed" not in log, log
    summary = json.loads((job / "results" / "hpo_summary.json").read_text())
    assert 3 <= summary["n_trials"] <= 4  # a late worker may start one extra trial
    assert len(summary["seeds"]) == 2
