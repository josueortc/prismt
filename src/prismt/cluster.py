"""Self-contained SLURM job folders (the MATLAB app writes the same layout).

A job folder holds everything a cluster needs: the run settings, a snapshot of the PRISMT
code (so the cluster runs exactly the version you used locally), static job scripts,
``job.env`` with the resources, and a README with copy-paste commands. Copy it to the
cluster, run ``bash setup_env.sh`` once, then ``bash submit.sh``.
"""

from __future__ import annotations

import json
import shlex
import shutil
from datetime import datetime
from pathlib import Path

from prismt import RESOURCES, __version__
from prismt.errors import ConfigError

TEMPLATES = RESOURCES / "cluster"
SCRIPTS = ("common.sh", "setup_env.sh", "submit.sh", "summarize.sh", "train.sbatch", "hpo_worker.sbatch",
           "hpo_final.sbatch")

DEFAULT_PROFILE = {
    "name": "generic",
    "host": "",
    "user": "",
    "remote_root": "~/prismt_jobs",
    "partition": "gpu",
    "account": "",
    "qos": "",
    "constraint": "",
    "gpus": 1,
    "cpus": 4,
    "mem": "32G",
    "time": "04:00:00",
    "modules": "miniconda",
    "conda_env": "prismt",
    "runtime": "conda",
    "image": "prismt.sif",
    "image_uri": "docker://ghcr.io/josueortc/prismt:cuda",
    "torch_version": "2.5.1",
    "cuda_tag": "cu121",
    "device": "auto",
}


def write_job_folder(config: dict, out_dir: str | Path, profile: dict | None = None, *, mode: str = "train",
                     n_folds: int = 1, hpo_workers: int = 4, dataset_remote: str | None = None,
                     name: str | None = None) -> Path:
    """Create ``out_dir/<name>_<timestamp>/``. ``config`` is the partial run config (run.json)."""
    prof = {**DEFAULT_PROFILE, **(profile or {})}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{name or config.get('task', 'run')}_{stamp}"
    job = Path(out_dir).expanduser() / job_name
    (job / "logs").mkdir(parents=True)
    (job / "results").mkdir()
    for script in SCRIPTS:
        text = (TEMPLATES / script).read_text(encoding="utf-8")
        (job / script).write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
        (job / script).chmod(0o755)
    cfg = json.loads(json.dumps(config))
    if dataset_remote:
        cfg.setdefault("dataset", {})["path"] = dataset_remote
    else:
        # No path on the cluster given: ship the dataset inside the job folder (paths in
        # config.json are relative to it, so the folder can be copied anywhere).
        local = Path(cfg.get("dataset", {}).get("path") or "").expanduser()
        if not local.is_file():
            raise ConfigError("E_JOB_DATASET", f"The dataset file {local} was not found.",
                              hint="Save the dataset first, or give its path on the cluster.", field="dataset.path")
        (job / "data").mkdir()
        shutil.copy2(local, job / "data" / local.name)
        cfg["dataset"]["path"] = f"data/{local.name}"
    init = (cfg.get("model") or {}).get("init_from")
    if init:
        # Fine-tuning starts from a local autoencoder run: ship it too.
        src_run = Path(init).expanduser()
        if not src_run.is_dir():
            raise ConfigError("E_JOB_INIT", f"The autoencoder run {src_run} was not found.",
                              hint="Choose a finished masked-autoencoder run.", field="model.init_from")
        shutil.copytree(src_run, job / "init_from" / src_run.name,
                        ignore=shutil.ignore_patterns("last.pt", "STOP", "*.tmp*"))
        cfg["model"]["init_from"] = f"init_from/{src_run.name}"
    cfg.setdefault("output", {})["root"] = "results"
    (job / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    src = Path(__file__).resolve().parent
    shutil.copytree(src, job / "prismt_src" / "src" / "prismt", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (job / "prismt_src" / "snapshot_info.json").write_text(json.dumps({"prismt_version": __version__,
                                                                        "created": stamp}), encoding="utf-8")
    values = {
        "JOB_NAME": job_name, "MODE": mode, "FOLDS": n_folds, "HPO_WORKERS": hpo_workers,
        "PARTITION": prof["partition"], "ACCOUNT": prof["account"], "QOS": prof["qos"],
        "CONSTRAINT": prof["constraint"], "GPUS": prof["gpus"], "CPUS": prof["cpus"], "MEM": prof["mem"],
        "TIME": prof["time"], "MODULES": prof["modules"], "CONDA_ENV": prof["conda_env"],
        "RUNTIME": prof["runtime"], "IMAGE": prof["image"], "IMAGE_URI": prof["image_uri"],
        "TORCH_VERSION": prof["torch_version"], "CUDA_TAG": prof["cuda_tag"], "DEVICE": prof["device"],
    }
    env = "".join(f"PRISMT_{k}={shlex.quote(str(v))}\n" for k, v in values.items())
    (job / "job.env").write_bytes(("# Resources for this job; edit if needed, then run: bash submit.sh\n" + env).encode())
    (job / "README.txt").write_text(readme(job_name, prof, dataset_remote, job, n_folds=n_folds, mode=mode),
                                     encoding="utf-8")
    return job


def readme(job_name: str, prof: dict, dataset_remote: str | None, job: Path | None = None, *,
           n_folds: int = 1, mode: str = "train") -> str:
    login = f"{prof['user']}@{prof['host']}" if prof["user"] and prof["host"] else "NETID@CLUSTER"
    root = prof["remote_root"]
    lines = [
        f"PRISMT cluster job: {job_name}",
        "",
        "1. Copy this folder to the cluster (on your computer; Windows without rsync: use scp -r):",
        f"     rsync -av \"{job or job_name}\" {login}:{root}/",
    ]
    if dataset_remote:
        lines += ["   and the dataset, if it is not there yet:",
                  f"     rsync -av --progress <your dataset file> {login}:{dataset_remote}"]
    lines += [
        "2. Log in:",
        f"     ssh {login}",
        "3. The first time on this cluster only (about 10 minutes, on the login node):",
        f"     bash {root}/{job_name}/setup_env.sh",
        "4. Submit:",
        f"     bash {root}/{job_name}/submit.sh",
        "5. Check progress:",
        f"     squeue --me      tail -f {root}/{job_name}/logs/*.out",
    ]
    if mode == "train" and n_folds > 1:
        lines += [f"   The {n_folds} cross-validation folds run as separate jobs. When all have finished, combine them:",
                  f"     bash {root}/{job_name}/summarize.sh"]
    lines += [
        "6. When finished, copy the results back (on your computer):",
        f"     rsync -av {login}:{root}/{job_name}/results/ \"{job or job_name}/results/\"",
        "   then open the job folder in the PRISMT app (Results tab).",
        "",
        "Stop a job with: scancel <job id>. Submitting again is safe; tuning continues where it stopped.",
    ]
    return "\n".join(lines) + "\n"
