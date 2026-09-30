"""Automatic tuning ("Tune automatically" in the app) with Optuna.

* Tried settings are scored on validation trials of the first fold only; test trials are
  used once, by the final retraining.
* The study lives in ``<run>/hpo/study.journal`` (Optuna's journal file storage), which is
  safe on network file systems, so several cluster jobs can search in parallel and a job
  that is re-submitted continues the same study. The budget counts finished and pruned
  trials across all workers.
* ``finalize`` retrains the best settings with several seeds on every fold and reports the
  test score as mean and spread; the model kept is the seed with the best validation score.
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np

from prismt.errors import ConfigError, EnvironmentProblem
from prismt.io.runfiles import StatusWriter, ensure_run_dir
from prismt.jsonutil import read_json, write_json

log = logging.getLogger("prismt")


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise EnvironmentProblem("E_ENV_NO_OPTUNA", "Automatic tuning needs the optuna package.",
                                 hint="Install it with: pip install optuna (or recreate the PRISMT environment).") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _storage(hpo_dir: Path):
    optuna = _optuna()
    from optuna.storages import JournalStorage

    path = str(hpo_dir / "study.journal")
    try:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    except ImportError:  # optuna 3.x
        from optuna.storages import JournalFileOpenLock, JournalFileStorage

        backend = JournalFileStorage(path, lock_obj=JournalFileOpenLock(path))
    return optuna, JournalStorage(backend)


def suggest(trial, cfg: dict) -> dict:
    """Sampled overrides as {dotted path: value}."""
    out = {"train.lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
           "model.dropout": trial.suggest_float("dropout", 0.0, 0.3)}
    if cfg["hpo"]["space"] == "default":
        out["train.weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        out["model.d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
        out["model.n_layers"] = trial.suggest_int("n_layers", 1, 4)
    return out


def apply(cfg: dict, overrides: dict) -> dict:
    from prismt.config import set_value

    new = copy.deepcopy(cfg)
    for path, value in overrides.items():
        set_value(new, path, value)
    if new["model"]["d_model"] % new["model"]["n_heads"]:
        new["model"]["n_heads"] = 4
    return new


def run_worker(cfg: dict, run_dir: Path, *, worker: int = 0) -> dict:
    """Try settings until the shared budget (or this worker's time budget) is used up."""
    import torch

    from prismt.env import select_device
    from prismt.run import _build, prepare
    from prismt.train.loop import fit

    hpo_dir = run_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    optuna, storage = _storage(hpo_dir)
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState

    sampler = optuna.samplers.TPESampler(seed=cfg["train"]["seed"] + 1000 * worker, constant_liar=True,
                                         multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(study_name="prismt", storage=storage, sampler=sampler, pruner=pruner,
                                direction="minimize", load_if_exists=True)
    prep = prepare(cfg)
    device = select_device(cfg["train"]["device"])
    fold = prep.plan.folds[0]
    status = StatusWriter(run_dir / "status.json", task=cfg["task"], mode="hpo", run_dir=str(run_dir))
    n_trials = cfg["hpo"]["n_trials"]
    epochs = cfg["hpo"]["epochs_per_trial"] or cfg["train"]["epochs"]

    def objective(trial):
        overrides = suggest(trial, cfg)
        tcfg = apply(cfg, overrides)
        tcfg["train"]["epochs"] = epochs
        prep.cfg = tcfg
        model = None
        try:
            model, task, tensors, _ = _build(prep, fold, device)

            def report(epoch, value):
                trial.report(value, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            result = fit(model, task, tensors, fold, tcfg["train"], monitor="val_loss", on_epoch_end=report)
            done = len([t for t in study.get_trials(deepcopy=False) if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
            status.update("training", force=True, message=f"Tried {done + 1} of {n_trials} settings",
                          trials_done=done + 1, n_trials=n_trials)
            if not np.isfinite(result.best_value):
                raise optuna.TrialPruned()
            return result.best_value
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log.warning("trial %d ran out of memory; pruned", trial.number)
                raise optuna.TrialPruned() from exc
            raise
        finally:
            prep.cfg = cfg
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    status.update("training", force=True, message="Tuning", n_trials=n_trials)
    timeout = 60 * cfg["hpo"]["timeout_minutes"] if cfg["hpo"]["timeout_minutes"] else None
    finished = [t for t in study.get_trials(deepcopy=False) if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    if len(finished) < n_trials:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True,
                       callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))],
                       catch=())
    return {"worker": worker, "trials": len(study.trials)}


def finalize(cfg: dict, run_dir: Path) -> dict:
    """Retrain the best settings with several seeds, evaluate each once on the test trials."""
    from prismt.run import run

    optuna, storage = _storage(run_dir / "hpo")
    from optuna.trial import TrialState

    study = optuna.load_study(study_name="prismt", storage=storage)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete:
        raise ConfigError("E_HPO_NO_TRIALS", "No tried setting finished, so there is nothing to retrain.",
                          hint="Check log.txt; give each worker more time or reduce the model size.")
    best = min(complete, key=lambda t: t.value)
    overrides = {k: best.params[p] for k, p in (("train.lr", "lr"), ("model.dropout", "dropout"),
                                                 ("train.weight_decay", "weight_decay"), ("model.d_model", "d_model"),
                                                 ("model.n_layers", "n_layers")) if p in best.params}
    best_cfg = apply(cfg, overrides)
    best_cfg["preset"] = "custom"
    write_json(run_dir / "best_config.json", {k: v for k, v in best_cfg.items() if k != "config_version"})
    rows = [{"number": t.number, "state": t.state.name, "value": t.value, **t.params} for t in study.trials]
    from prismt.io.runfiles import write_csv

    write_csv(run_dir / "trials.csv", rows)
    seeds = []
    for k in range(cfg["hpo"]["final_seeds"]):
        scfg = copy.deepcopy(best_cfg)
        scfg["train"]["seed"] = cfg["train"]["seed"] + k
        out = run(scfg, run_dir=run_dir / "final" / f"seed_{k + 1}")
        m = read_json(out / "metrics.json")
        seeds.append({"seed": scfg["train"]["seed"], "run_dir": str(out), "headline": m["headline"],
                      "val": m["val"].get("loss") if cfg["task"] == "mae" else -(m["val"].get("balanced_accuracy") or 0.0)})
    values = [s["headline"]["value"] for s in seeds if s["headline"]["value"] is not None]
    keep = min(seeds, key=lambda s: s["val"] if s["val"] is not None else np.inf)
    summary = {"best_trial": best.number, "best_value": best.value, "best_params": best.params,
               "n_trials": len(study.trials), "seeds": seeds, "kept": keep["run_dir"],
               "test": {"name": seeds[0]["headline"]["name"], "mean": float(np.mean(values)),
                        "sd": float(np.std(values)), "values": values},
               "summary_lines": [f"Best settings from {len(study.trials)} tried: " +
                                 ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
                                           for k, v in best.params.items()),
                                 f"Test {seeds[0]['headline']['name'].replace('_', ' ')} over "
                                 f"{len(values)} seeds: {np.mean(values):.2f} ± {np.std(values):.2f}."]}
    write_json(run_dir / "hpo_summary.json", summary)
    return summary


def run_hpo(cfg: dict, run_dir: str | Path | None, *, worker: int | None = None, do_finalize: bool | None = None,
            source: dict | None = None) -> Path:
    """Local use runs one worker then finalizes; cluster arrays run workers and a separate finalize job."""
    from prismt.io.runfiles import create_run_dir

    out = Path(run_dir) if run_dir else create_run_dir(cfg["output"]["root"], "hpo-" + cfg["task"], cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    if source is not None and not (out / "run.json").exists():
        write_json(out / "run.json", source)
    status = StatusWriter(out / "status.json", task=cfg["task"], mode="hpo", run_dir=str(out))
    try:
        if not do_finalize:
            run_worker(cfg, out, worker=worker or 0)
        if do_finalize or (worker is None and do_finalize is None):
            status.update("evaluating", force=True, message="Retraining the best settings")
            summary = finalize(cfg, out)
            status.update("finished", force=True, message=summary["summary_lines"][-1])
        else:
            status.update("finished", force=True, message=f"Worker {worker} finished")
    except Exception as exc:  # noqa: BLE001
        err = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_INTERNAL", "title": "Tuning failed",
                                                              "message": str(exc), "hint": "", "field": ""}
        status.fail(err)
        raise
    return out
