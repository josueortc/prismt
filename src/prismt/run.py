"""Running PRISMT: the preflight check, training every fold, and writing the run folder."""

from __future__ import annotations

import csv
import json
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prismt import FORMATS, __version__
from prismt.config import config_hash, load_config
from prismt.data.preprocess import Normalizer, TimeBins, extract, make_time_bins
from prismt.data.selection import Selection, confound_warnings, label_level, select
from prismt.data.splits import LEVELS, Fold, SplitPlan, _check_classes, assert_no_leakage, plan_splits
from prismt.data.tokens import TokenGrid
from prismt.errors import (CheckpointError, ConfigError, Issue, PrismtError, SplitError, raise_if_errors)
from prismt.io.dataset import PrismtDataset, read_dataset
from prismt.io.runfiles import (HistoryWriter, StatusWriter, ensure_run_dir, create_run_dir, stop_requested,
                                utc_now, write_csv, write_results_mat)
from prismt.jsonutil import read_json, to_jsonable, write_json

log = logging.getLogger("prismt")

TOKEN_WARN = 1100
TOKEN_LIMIT = 4096


@dataclass
class Prepared:
    cfg: dict
    ds: PrismtDataset
    sel: Selection
    bins: TimeBins
    X: np.ndarray  # [n selected, channels, bins, modalities], not yet scaled
    grid: TokenGrid
    subject: np.ndarray | None
    session: np.ndarray | None
    plan: SplitPlan
    warnings: list[Issue]
    pair_modality: np.ndarray
    init_from: dict | None = None

    @property
    def n_classes(self) -> int:
        return len(self.sel.class_names)


# ---------------------------------------------------------------------------------------
# Preparation (shared by check and run)
# ---------------------------------------------------------------------------------------


def prepare(cfg: dict) -> Prepared:
    ds = read_dataset(cfg["dataset"]["path"])
    sel = select(ds, cfg)
    pre = cfg["preprocess"]
    bins = make_time_bins(ds.times_s, pre["time_window_s"], pre["bin_width_s"])
    X = extract(ds, sel.trial_index, sel.channels, sel.modalities, sel.pairs, bins,
                baseline_subtract=pre["baseline_subtract"])
    grid = TokenGrid.make(sel.pairs, bins.n, cfg["model"]["time_patch"])
    subject = None if ds.subject is None else ds.subject_ids()[sel.trial_index]
    session = None if ds.session is None else ds.session_keys()[sel.trial_index]
    n_tokens = grid.n_tokens + 1
    if n_tokens > TOKEN_LIMIT:
        raise ConfigError("E_CFG_TOKENS", f"Each trial would become {n_tokens} tokens; the limit is {TOKEN_LIMIT}.",
                          hint="Use a shorter time window, wider time bins, more time bins per token, or fewer "
                               "channels.", field="preprocess.bin_width_s",
                          title="The model settings do not fit this dataset")
    if cfg["task"] == "mae" and cfg["mae"]["mask"]["strategy"] == "modality" and len(sel.modalities) < 2:
        raise ConfigError("E_CFG_MASK", "Hiding a modality needs at least two modalities.",
                          hint="Choose another masking strategy.", field="mae.mask.strategy")
    init = None
    if cfg["model"]["init_from"]:
        plan, init = inherit_plan(cfg, ds, sel, subject, session)
    else:
        plan = plan_splits(sel.y, subject, session, cfg["split"], label_name=sel.label_column or "label",
                           class_names=sel.class_names)
    warnings = list(ds.warnings) + sel.warnings + confound_warnings(ds, sel) + plan.warnings
    if n_tokens > TOKEN_WARN:
        warnings.append(Issue("warning", "W_MANY_TOKENS",
                              f"Each trial is {n_tokens} tokens; training will be slow on a laptop.",
                              "Wider time bins or more time bins per token make it much faster."))
    return Prepared(cfg, ds, sel, bins, X, grid, subject, session, plan, warnings,
                    np.array([m for _, m in grid.pairs]), init)


def inherit_plan(cfg, ds, sel, subject, session) -> tuple[SplitPlan, dict]:
    """Fine-tuning uses the autoencoder run's split, so its test trials were never pretrained on."""
    src = Path(cfg["model"]["init_from"]).expanduser()
    run_dir = src if src.is_dir() else src.parent
    if not (run_dir / "config.json").exists() or not (run_dir / "splits.csv").exists():
        raise CheckpointError("E_CKPT_MISSING", f"{run_dir} is not a finished PRISMT run.",
                              hint="Choose the folder of a finished autoencoder run.", field="model.init_from")
    mae_cfg = read_json(run_dir / "config.json")
    if mae_cfg.get("task") != "mae":
        raise CheckpointError("E_CKPT_NOT_MAE", "The run to start from is not a masked-autoencoder run.",
                              field="model.init_from")
    if mae_cfg.get("dataset_fingerprint") and mae_cfg["dataset_fingerprint"] != ds.fingerprint:
        raise CheckpointError("E_CKPT_DATASET", "The autoencoder was trained on a different dataset (or another "
                              "version of this file).", hint="Fine-tune on the dataset the autoencoder used.",
                              field="model.init_from")
    assign: dict[int, tuple[int, str]] = {}
    with open(run_dir / "splits.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            assign.setdefault(int(row["trial_index"]) - 1, []).append((int(row["fold"]), row["split"]))
    info = mae_cfg.get("split_plan", {})
    n_folds = int(info.get("n_folds", 1))
    folds = []
    for k in range(1, n_folds + 1):
        parts = {"train": [], "val": [], "test": []}
        for pos, t in enumerate(sel.trial_index):
            for fold, part in assign.get(int(t), []):
                if fold == k:
                    parts[part].append(pos)
        folds.append(Fold(k - 1, *(np.asarray(parts[p], dtype=np.int64) for p in ("train", "val", "test"))))
    level = label_level(sel.y, subject, session)
    test_on = info.get("test_on", "trial")
    flags = {"inherited_from": str(run_dir)}
    plan = SplitPlan(folds, info.get("scheme", "single"), test_on, info.get("validate_on", test_on), level,
                     int(info.get("n_groups", 0)), flags, [])
    if LEVELS.index(test_on) < LEVELS.index(level) and not cfg["split"]["allow_leaky"]:
        raise SplitError("E_LEAKY_SPLIT", f"The autoencoder was tested on held-out {test_on}s, but "
                         f"'{sel.label_column}' is constant within {level}s.",
                         hint=f"Re-run the autoencoder with 'Test on' set to {level}s or subjects.",
                         field="model.init_from")
    issues: list[Issue] = []
    _check_classes(plan, sel.y, sel.class_names, issues, plan.warnings)
    raise_if_errors(issues, SplitError)
    assert_no_leakage(plan, {"trial": np.array([str(i) for i in range(sel.n)], dtype=object),
                             "subject": subject, "session": session})
    fold_dirs = [str(run_dir if n_folds == 1 else run_dir / f"fold_{k:02d}") for k in range(1, n_folds + 1)]
    return plan, {"run_dir": str(run_dir), "fold_dirs": fold_dirs}


# ---------------------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------------------


def n_parameters(d: int, layers: int, ff: int, pairs: int, patches: int, patch: int, pos: str, head: int) -> int:
    tok = patch * d + d + d + (pairs * d + patches * d if pos == "channel_time" else pairs * patches * d)
    block = 4 * d + 3 * d * d + 3 * d + d * d + d + 2 * ff * d * d + ff * d + d
    return int(tok + d + layers * block + 2 * d + head * d + head)


def check(cfg: dict, *, timing: bool = False) -> dict:
    """What the MATLAB app shows before a run: selection, split plan, size and time estimates."""
    prep = prepare(cfg)
    sel, plan, grid, m = prep.sel, prep.plan, prep.grid, cfg["model"]
    head = prep.n_classes if cfg["task"] == "classify" else grid.patch
    params = n_parameters(m["d_model"], m["n_layers"], m["ff_mult"], grid.n_pairs, grid.n_patches, grid.patch,
                          m["position_embedding"], head)
    S = grid.n_tokens + 1
    B = cfg["train"]["batch_size"]
    attn_gb = 3 * m["n_layers"] * B * m["n_heads"] * S * S * 4 / 2**30
    out = {
        "ok": True,
        "task": cfg["task"],
        "dataset": {"path": cfg["dataset"]["path"], "n_trials": prep.ds.n_trials, "fingerprint": prep.ds.fingerprint},
        "selection": {
            "n_trials": sel.n,
            "dropped": sel.dropped,
            "label_column": sel.label_column,
            "class_names": sel.class_names,
            "class_members": sel.class_members,
            "class_counts": None if sel.y is None else np.bincount(sel.y, minlength=prep.n_classes).tolist(),
            "channels": [prep.ds.channel_names[i] for i in sel.channels],
            "modalities": [prep.ds.modality_names[i] for i in sel.modalities],
            "n_time_bins": prep.bins.n,
            "bin_centers_s": prep.bins.centers_s.tolist(),
        },
        "split": plan.summary(sel.y, sel.class_names, {"subject": prep.subject, "session": prep.session}),
        "model": {"tokens_per_trial": S, "parameters": params, "attention_memory_gb": round(attn_gb, 2)},
        "warnings": [w.to_dict() for w in prep.warnings],
        "estimate": None,
    }
    if timing:
        out["estimate"] = _time_steps(prep)
    return out


def _time_steps(prep: Prepared) -> dict:
    import torch

    from prismt.env import select_device

    cfg = prep.cfg
    device = select_device(cfg["train"]["device"])
    fold = prep.plan.folds[0]
    model, task, tensors, _ = _build(prep, fold, device)
    from prismt.train.optim import build_optimizer

    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])
    bs = cfg["train"]["batch_size"]
    rows = fold.train[:bs]
    gen = torch.Generator().manual_seed(0)
    times = []
    model.train()
    for i in range(4):
        t0 = time.time()
        x, v, y = tensors.batch(rows)
        loss = task.loss(model, x, v, y, generator=gen)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        if i:
            times.append(time.time() - t0)
    step = float(np.median(times))
    steps = math.ceil(len(fold.train) / bs)
    epoch = step * steps * 1.3  # plus evaluation
    total = epoch * cfg["train"]["epochs"] * len(prep.plan.folds)
    return {"device": str(device), "seconds_per_step": round(step, 4), "seconds_per_epoch": round(epoch, 1),
            "max_total_minutes": round(total / 60, 1)}


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------


def eval_specs(cfg: dict, prep: Prepared):
    from prismt.masking import MaskSpec

    mc = cfg["mae"]["mask"]
    names = [prep.ds.modality_names[i] for i in prep.sel.modalities]
    mod = None
    if mc["modality"]:
        if mc["modality"] not in names:
            raise ConfigError("E_CFG_MASK", f"There is no modality '{mc['modality']}' to hide.",
                              hint=f"Modalities: {', '.join(names)}.", field="mae.mask.modality")
        mod = names.index(mc["modality"])
    train = MaskSpec(mc["strategy"], mc["ratio"], mc["context_fraction"], mod,
                     label=f"modality_{names[mod]}" if mc["strategy"] == "modality" and mod is not None else None)
    specs = [train]
    extra = [MaskSpec("random", 0.5), MaskSpec("channel", 0.2)]
    if prep.grid.n_patches > 1:
        extra.append(MaskSpec("forecast", context_fraction=0.5))
    if len(names) > 1:
        extra += [MaskSpec("modality", modality=i, label=f"modality_{n}") for i, n in enumerate(names)]
    for s in extra:
        if s.name not in [x.name for x in specs]:
            specs.append(s)
    return train, specs


def _build(prep: Prepared, fold: Fold, device, ckpt=None):
    import torch

    from prismt.data.tensors import TrialTensors
    from prismt.model.checkpoint import extra
    from prismt.model.prismt_model import ModelSpec, PrismtModel
    from prismt.tasks.classify import ClassifyTask
    from prismt.tasks.mae import MAETask

    cfg = prep.cfg
    pre = cfg["preprocess"]
    if ckpt is not None:
        norm = Normalizer.from_state(extra(ckpt, "normalizer"))
    else:
        norm = Normalizer.fit(prep.X, fold.train, pre["normalize"], pre["clip_sd"])
    values, valid = prep.grid.to_tokens(norm.transform(prep.X))
    tensors = TrialTensors.make(values, valid, prep.sel.y, device)
    torch.manual_seed(cfg["train"]["seed"] + fold.index)
    spec = ModelSpec.from_config(cfg["model"], cfg["task"], prep.grid.n_pairs, prep.grid.n_patches,
                                 prep.grid.patch, prep.n_classes)
    model = PrismtModel(spec)
    if cfg["task"] == "classify":
        w = None
        if cfg["labels"]["class_weighting"] == "balanced":
            w = ClassifyTask.balanced_weights(prep.sel.y[fold.train], prep.n_classes)
        task = ClassifyTask(prep.n_classes, w)
    else:
        train_spec, specs = eval_specs(cfg, prep)
        task = MAETask(prep.grid, train_spec, specs, prep.pair_modality, cfg["train"]["seed"])
    return model.to(device), task, tensors, (norm, values, valid)


def data_signature(prep: Prepared) -> dict:
    ch = [prep.ds.channel_names[i] for i in prep.sel.channels]
    mo = [prep.ds.modality_names[i] for i in prep.sel.modalities]
    return {"pairs": [[ch[c], mo[m]] for c, m in prep.grid.pairs],
            "bin_centers_s": [round(float(t), 6) for t in prep.bins.centers_s], "patch": prep.grid.patch}


@dataclass
class FoldOutcome:
    fold: Fold
    fit: object
    val: dict
    test: dict
    extras: dict = field(default_factory=dict)


def run(cfg: dict, *, run_dir: str | Path | None = None, only_fold: int | None = None,
        source: dict | None = None) -> Path:
    """Train and evaluate every fold (or one, for cluster arrays); returns the run folder."""
    out = ensure_run_dir(run_dir, overwrite=cfg["output"]["overwrite"]) if run_dir else \
        create_run_dir(cfg["output"]["root"], cfg["task"], cfg["name"], overwrite=cfg["output"]["overwrite"])
    handler = logging.FileHandler(out / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger("prismt").addHandler(handler)
    logging.getLogger("prismt").setLevel(logging.INFO)
    status = StatusWriter(out / "status.json", task=cfg["task"], run_dir=str(out))
    status.update("preparing", force=True, message="Reading the dataset and planning the split")
    flag = {"reason": None}

    def on_signal(signum, _frame):
        flag["reason"] = "cancel" if signum == signal.SIGTERM else "stop_requested"

    previous = {}
    for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if sig is not None:
            try:
                previous[sig] = signal.signal(sig, on_signal)
            except ValueError:  # not in the main thread
                pass

    def should_stop():
        if flag["reason"]:
            return flag["reason"]
        if stop_requested(out):
            text = (out / "STOP").read_text(errors="ignore").strip().lower()
            return "cancel" if text == "cancel" else "stop_requested"
        return None

    try:
        prep = prepare(cfg)
        from prismt.env import environment_record, select_device

        device = select_device(cfg["train"]["device"])
        resolved = dict(cfg)
        resolved.update({"config_hash": config_hash(cfg), "dataset_fingerprint": prep.ds.fingerprint,
                         "split_plan": {k: v for k, v in prep.plan.summary(prep.sel.y, prep.sel.class_names, {
                             "subject": prep.subject, "session": prep.session}).items() if k != "folds"},
                         "prismt_version": __version__})
        write_json(out / "config.json", resolved)
        if source is not None and not (out / "run.json").exists():
            write_json(out / "run.json", source)
        write_json(out / "run_info.json", {**environment_record(device), "command": sys.argv, "started": utc_now()})
        _write_splits(out, prep)
        n_folds = len(prep.plan.folds)
        if only_fold is not None and not 1 <= only_fold <= n_folds:
            raise ConfigError("E_RUN_FOLD", f"There are {n_folds} folds; fold {only_fold} does not exist.")
        folds = prep.plan.folds if only_fold is None else [prep.plan.folds[only_fold - 1]]
        if only_fold is not None and n_folds > 1:
            # A fold running as its own cluster job must not share files with its siblings.
            fdir = out / f"fold_{only_fold:02d}"
            fdir.mkdir(exist_ok=True)
            status.path = fdir / "status.json"
            history = HistoryWriter(fdir / "history.csv")
        else:
            history = HistoryWriter(out / "history.csv")
        outcomes = []
        for fold in folds:
            fold_dir = out if n_folds == 1 else out / f"fold_{fold.index + 1:02d}"
            fold_dir.mkdir(exist_ok=True)
            outcomes.append(_run_fold(prep, fold, fold_dir, device, status, history, n_folds, should_stop))
            if flag["reason"] == "cancel":
                break
        status.update("evaluating", force=True, message="Writing results")
        if only_fold is None or n_folds == 1:
            _write_summary(out, prep, outcomes, device)
        status.update("finished", force=True, message="Finished", eta_s=0,
                      stopped=[str(o.fit.stopped) for o in outcomes])
        return out
    except PrismtError as err:
        status.fail(err.to_dict(), "cancelled" if err.exit_code == 3 else "failed")
        raise
    except Exception as exc:
        status.fail({"code": "E_INTERNAL", "title": "PRISMT hit an unexpected error", "message": str(exc),
                     "hint": "Please report this, with log.txt from the run folder.", "field": ""})
        raise
    finally:
        logging.getLogger("prismt").removeHandler(handler)
        handler.close()
        for sig, old in previous.items():
            try:
                signal.signal(sig, old)
            except (ValueError, TypeError):
                pass


def _run_fold(prep: Prepared, fold: Fold, fold_dir: Path, device, status, history, n_folds, should_stop) -> FoldOutcome:
    from prismt.model.checkpoint import load_checkpoint, extra, load_pretrained_encoder, save_checkpoint
    from prismt.train.loop import fit

    cfg = prep.cfg
    ckpt = None
    if prep.init_from:
        ckpt = load_checkpoint(prep.init_from["fold_dirs"][fold.index])
        if extra(ckpt, "data_signature") != data_signature(prep):
            raise CheckpointError("E_CKPT_SIGNATURE", "The autoencoder used different channels, modalities or time "
                                  "bins than this run.", hint="Use the same channels, modalities, time window, bin "
                                  "width and time bins per token as the autoencoder run.", field="model.init_from")
    model, task, tensors, (norm, values, valid) = _build(prep, fold, device, ckpt)
    if ckpt is not None:
        report = load_pretrained_encoder(model, ckpt, source=prep.init_from["fold_dirs"][fold.index])
        log.info("loaded %d/%d encoder tensors from %s", report.loaded, report.total, report.source)
        if cfg["model"]["freeze_encoder"]:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
    monitor = cfg["train"]["monitor"]
    if monitor == "auto":
        monitor = "val_loss"
    if monitor == "val_balanced_accuracy" and cfg["task"] != "classify":
        raise ConfigError("E_CFG_MONITOR", "Balanced accuracy can only be monitored when classifying.",
                          field="train.monitor")
    result = fit(model, task, tensors, fold, cfg["train"], monitor=monitor, status=status, history=history,
                 fold_label=fold.index + 1, n_folds=n_folds, should_stop=should_stop)
    status.update("evaluating", force=True, message=f"Evaluating fold {fold.index + 1}/{n_folds}")
    outcome = _evaluate_fold(prep, fold, model, task, tensors, values, valid, norm, result)
    save_checkpoint(fold_dir / "model.pt", model, extra={
        "normalizer": norm.to_state(), "data_signature": data_signature(prep),
        "class_names": prep.sel.class_names, "best_epoch": result.best_epoch,
        "best_value": result.best_value if np.isfinite(result.best_value) else None,
        "epochs_run": result.epochs_run, "stopped": result.stopped, "seconds": result.seconds,
        "dataset_fingerprint": prep.ds.fingerprint, "split_fingerprint": prep.plan.fingerprint(),
        "run_config": prep.cfg, "fold": fold.index + 1})
    if n_folds > 1:
        _write_fold(fold_dir, prep, [outcome])
    return outcome


def _evaluate_fold(prep: Prepared, fold: Fold, model, task, tensors, values, valid, norm, result) -> FoldOutcome:
    cfg = prep.cfg
    bs = max(cfg["train"]["batch_size"], 64)
    all_rows = np.arange(prep.sel.n)
    extras = {"normalizer": norm}
    if cfg["task"] == "classify":
        from prismt.eval.baselines import features, logistic, majority
        from prismt.eval.metrics import classification_metrics

        val = task.evaluate(model, tensors, fold.val, bs)
        test = task.evaluate(model, tensors, fold.test, bs)
        y = prep.sel.y
        base = {"majority": majority(y[fold.train], y[fold.test], prep.n_classes)}
        if cfg["baselines"]["logistic"]:
            F, desc = features(values, valid)
            prob, info = logistic(F[fold.train], y[fold.train], F[fold.val], y[fold.val], F[fold.test],
                                  prep.n_classes, cfg["train"]["seed"])
            base["logistic"] = {**classification_metrics(y[fold.test], prob, prep.n_classes), **info,
                                "features": desc}
            extras["logistic_prob"] = prob
            if cfg["baselines"]["permutations"]:
                from prismt.eval.baselines import permutation_predictions

                groups = {"session": prep.session, "subject": prep.subject}.get(prep.plan.label_level)
                extras["null_pred"] = permutation_predictions(F, y, fold, prep.n_classes,
                                                              int(cfg["baselines"]["permutations"]), groups,
                                                              cfg["train"]["seed"])
        extras["baselines"] = base
        if cfg["output"]["save_embeddings"]:
            extras["embedding"] = task.predict(model, tensors, fold.test, bs)["embedding"]
    else:
        from prismt.tasks.mae import MAETask

        psth = MAETask.psth(values, valid, fold.train, prep.grid)
        val = task.evaluate(model, tensors, fold.val, bs, specs=task.eval_specs[:1])
        test = task.evaluate(model, tensors, fold.test, bs, psth=psth, n_examples=cfg["mae"]["n_examples"],
                             with_baselines=True, need_embeddings=cfg["output"]["save_embeddings"])
        if cfg["output"]["save_embeddings"]:
            extras["embedding"] = test.pop("embedding")
    return FoldOutcome(fold, result, val, test, extras)


def combine(cfg: dict, run_dir: str | Path) -> Path:
    """Pool cross-validation folds that were trained as separate cluster jobs (no retraining)."""
    from types import SimpleNamespace

    from prismt.env import select_device
    from prismt.model.checkpoint import extra, load_checkpoint

    out = Path(run_dir)
    prep = prepare(cfg)
    device = select_device(cfg["train"]["device"])
    outcomes = []
    missing = [f.index + 1 for f in prep.plan.folds if not (out / f"fold_{f.index + 1:02d}" / "model.pt").exists()]
    if missing:
        raise ConfigError("E_RUN_FOLDS_MISSING", f"Folds {missing} have not finished yet.",
                          hint="Wait for every fold job to finish (squeue --me), then combine again.")
    for fold in prep.plan.folds:
        ckpt = load_checkpoint(out / f"fold_{fold.index + 1:02d}")
        model, task, tensors, (norm, values, valid) = _build(prep, fold, device, ckpt)
        model.load_state_dict(ckpt["state_dict"])
        result = SimpleNamespace(best_epoch=extra(ckpt, "best_epoch"), epochs_run=extra(ckpt, "epochs_run", 0),
                                 stopped=extra(ckpt, "stopped", ""), seconds=extra(ckpt, "seconds", 0.0),
                                 best_value=extra(ckpt, "best_value"))
        outcomes.append(_evaluate_fold(prep, fold, model, task, tensors, values, valid, norm, result))
    _write_summary(out, prep, outcomes, device)
    StatusWriter(out / "status.json", task=cfg["task"], run_dir=str(out)).update(
        "finished", force=True, message="Finished (folds combined)")
    return out


# ---------------------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------------------


def _groups_text(prep: Prepared):
    return (prep.subject if prep.subject is not None else np.array([""] * prep.sel.n, dtype=object),
            prep.session if prep.session is not None else np.array([""] * prep.sel.n, dtype=object))


def _write_splits(out: Path, prep: Prepared) -> None:
    subj, sess = _groups_text(prep)
    rows = []
    for f in prep.plan.folds:
        for part, idx in (("train", f.train), ("val", f.val), ("test", f.test)):
            for pos in idx:
                rows.append({"trial_index": int(prep.sel.trial_index[pos]) + 1, "fold": f.index + 1, "split": part,
                             "subject": subj[pos], "session": sess[pos],
                             "label": "" if prep.sel.y is None else prep.sel.class_names[prep.sel.y[pos]]})
    write_csv(out / "splits.csv", rows, ["trial_index", "fold", "split", "subject", "session", "label"])


def _write_fold(out: Path, prep: Prepared, outcomes: list[FoldOutcome]) -> None:
    metrics, mat = _collect(prep, outcomes)
    write_json(out / "metrics.json", metrics)
    write_results_mat(out / "results.mat", mat)


def _write_summary(out: Path, prep: Prepared, outcomes: list[FoldOutcome], device) -> None:
    metrics, mat = _collect(prep, outcomes)
    write_json(out / "metrics.json", metrics)
    write_results_mat(out / "results.mat", mat)
    if prep.cfg["task"] == "classify":
        subj, sess = _groups_text(prep)
        rows = []
        for o in outcomes:
            for i, pos in enumerate(o.fold.test):
                r = {"trial_index": int(prep.sel.trial_index[pos]) + 1, "fold": o.fold.index + 1,
                     "subject": subj[pos], "session": sess[pos],
                     "true_class": prep.sel.class_names[prep.sel.y[pos]],
                     "predicted_class": prep.sel.class_names[int(o.test["prob"][i].argmax())]}
                for k, name in enumerate(prep.sel.class_names):
                    r[f"p_{name}"] = float(o.test["prob"][i, k])
                rows.append(r)
        write_csv(out / "predictions.csv", rows)


def _collect(prep: Prepared, outcomes: list[FoldOutcome]) -> tuple[dict, dict]:
    cfg, sel, plan, grid = prep.cfg, prep.sel, prep.plan, prep.grid
    subj, sess = _groups_text(prep)
    ch = [prep.ds.channel_names[i] for i in sel.channels]
    mo = [prep.ds.modality_names[i] for i in sel.modalities]
    test_fold = np.zeros(sel.n)
    for o in outcomes:
        test_fold[o.fold.test] = o.fold.index + 1
    split = np.array([""] * sel.n, dtype=object)
    if len(plan.folds) == 1:
        f = plan.folds[0]
        split[f.train], split[f.val], split[f.test] = "train", "val", "test"
    tested = np.concatenate([o.fold.test for o in outcomes])
    training = [{"fold": o.fold.index + 1, "best_epoch": o.fit.best_epoch, "epochs_run": o.fit.epochs_run,
                 "stopped": o.fit.stopped, "seconds": round(o.fit.seconds, 1)} for o in outcomes]
    norm = outcomes[0].extras["normalizer"]
    mat = {
        "format": "prismt.results/1", "task": cfg["task"], "channel_names": ch,
        "channel_index": (sel.channels + 1).astype(float), "modality_names": mo,
        "pairs": np.array([[c + 1, m + 1] for c, m in grid.pairs], dtype=float),
        "bin_centers_s": prep.bins.centers_s, "patch": float(grid.patch),
        "trial_index": (sel.trial_index + 1).astype(float), "test_fold": test_fold, "split": split,
        "subject": subj.astype(object), "session": sess.astype(object),
        "norm_mean": norm.mean.astype(float), "norm_std": norm.std.astype(float),
    }
    emb = [o.extras.get("embedding") for o in outcomes]
    if all(e is not None for e in emb) and emb:
        E = np.concatenate(emb)
        E = E - E.mean(0)
        k = min(10, E.shape[1], max(E.shape[0] - 1, 1))
        u, s, vt = np.linalg.svd(E, full_matrices=False)
        mat["embedding_pcs"] = (u[:, :k] * s[:k]).astype(float)
        mat["embedding_explained"] = (s[:k] ** 2 / max((s ** 2).sum(), 1e-12)).astype(float)
        mat["embedding_trial_index"] = (sel.trial_index[tested] + 1).astype(float)
    metrics = {
        "schema": "prismt.metrics/1", "task": cfg["task"], "scheme": plan.scheme, "n_folds": len(plan.folds),
        "test_on": plan.test_on, "validate_on": plan.validate_on, "label_level": plan.label_level,
        "flags": plan.flags, "warnings": [w.to_dict() for w in prep.warnings],
        "selection": {"n_trials": sel.n, "dropped": sel.dropped, "class_names": sel.class_names,
                      "class_counts": None if sel.y is None else np.bincount(sel.y, minlength=prep.n_classes).tolist()},
        "training": training,
    }
    what = {"subject": "new animals", "session": "new sessions", "trial": "held-out trials"}[plan.test_on]
    how = "a single split" if plan.scheme == "single" else f"{len(plan.folds)}-fold cross-validation"
    if cfg["task"] == "classify":
        from prismt.eval.metrics import classification_metrics, per_group

        C = prep.n_classes
        y = np.concatenate([o.test["y"] for o in outcomes])
        prob = np.concatenate([o.test["prob"] for o in outcomes])
        test = classification_metrics(y, prob, C)
        val_y = np.concatenate([o.val["y"] for o in outcomes])
        val = classification_metrics(val_y, np.concatenate([o.val["prob"] for o in outcomes]), C)
        folds = [o.test["metrics"] for o in outcomes]
        pred = prob.argmax(1)
        by_group = {}
        if prep.subject is not None:
            by_group["subject"] = per_group(y, pred, prep.subject[tested], C)
        if prep.session is not None:
            by_group["session"] = per_group(y, pred, prep.session[tested], C)
        baselines = {"chance": {"balanced_accuracy": 1.0 / C},
                     "majority": {"balanced_accuracy": 1.0 / C,
                                  "accuracy": float(np.mean([o.extras["baselines"]["majority"]["accuracy"] or 0
                                                             for o in outcomes]))}}
        if all("logistic_prob" in o.extras for o in outcomes):
            lp = np.concatenate([o.extras["logistic_prob"] for o in outcomes])
            baselines["logistic"] = classification_metrics(y, lp, C)
            mat["logistic_prob"] = lp
        ba = test["balanced_accuracy"]
        if all("null_pred" in o.extras for o in outcomes) and "logistic" in baselines:
            from sklearn.metrics import balanced_accuracy_score

            pooled = np.concatenate([o.extras["null_pred"] for o in outcomes], axis=1)
            null = np.array([balanced_accuracy_score(y, row) for row in pooled])
            p_model = float((1 + np.sum(null >= ba)) / (1 + len(null)))
            p_log = float((1 + np.sum(null >= baselines["logistic"]["balanced_accuracy"])) / (1 + len(null)))
            baselines["shuffled_labels"] = {
                "balanced_accuracy": float(np.mean(null)), "p95": float(np.quantile(null, 0.95)),
                "n": int(len(null)), "shuffled_by": prep.plan.label_level, "p_value_model": p_model,
                "p_value_logistic": p_log, "description": "logistic regression refitted with shuffled training labels"}
            mat["shuffled_null"] = null
        lines = [f"Balanced accuracy on {what}: {ba:.2f} (chance {1 / C:.2f}), from {how}."]
        if "logistic" in baselines:
            lines.append(f"Logistic regression on the same trials: {baselines['logistic']['balanced_accuracy']:.2f}.")
        if "shuffled_labels" in baselines:
            sh = baselines["shuffled_labels"]
            lines.append(f"With shuffled labels ({sh['n']} repeats) logistic regression reaches {sh['balanced_accuracy']:.2f} "
                         f"on average (95th percentile {sh['p95']:.2f}); p = {sh['p_value_model']:.3g} for the transformer.")
        if len(folds) > 1:
            vals = [f["balanced_accuracy"] for f in folds if f.get("balanced_accuracy") is not None]
            lines.append(f"Across folds: {np.mean(vals):.2f} ± {np.std(vals):.2f} (mean ± sd).")
        metrics.update({"headline": {"name": "balanced_accuracy", "value": ba, "chance": 1 / C,
                                     "description": f"Balanced accuracy on {what}"},
                        "test": test, "val": val, "folds": folds, "baselines": baselines, "by_group": by_group,
                        "summary_lines": lines})
        mat.update({"class_names": sel.class_names, "y_true": (y + 1).astype(float), "y_pred": (pred + 1).astype(float),
                    "prob": prob, "test_trial_index": (sel.trial_index[tested] + 1).astype(float),
                    "test_trial_fold": np.concatenate([np.full(len(o.fold.test), o.fold.index + 1.0) for o in outcomes]),
                    "confusion_test": np.asarray(test["confusion"], dtype=float)})
    else:
        from prismt.eval.baselines import MAE_BASELINES
        from prismt.eval.metrics import R2Accumulator

        names = list(outcomes[0].test["by_mask"])
        S, Q, P = len(names), grid.n_pairs, grid.n_patches
        r2, r2_pair, r2_patch = np.full(S, np.nan), np.full((S, Q), np.nan), np.full((S, P), np.nan)
        B = len(MAE_BASELINES)
        b_r2, b_pair, skill = np.full((S, B), np.nan), np.full((S, B, Q), np.nan), np.full((S, B), np.nan)
        pt = {k: [] for k in ("n", "s1", "s2", "sse")}
        by_mask = {}
        for si, name in enumerate(names):
            accs = [o.test["by_mask"][name]["_acc"] for o in outcomes]
            tot = [sum(getattr(a, k).sum() for a in accs) for k in ("n", "s1", "s2", "sse")]
            r2[si] = R2Accumulator.r2(*tot)
            pair = [sum(getattr(a, k).sum(0) for a in accs) for k in ("n", "s1", "s2", "sse")]
            r2_pair[si] = R2Accumulator.r2(*pair)
            patch = [sum(getattr(a, "patch_" + k) for a in accs) for k in ("n", "s1", "s2", "sse")]
            r2_patch[si] = R2Accumulator.r2(*patch)
            for k in pt:
                pt[k].append(np.concatenate([getattr(a, k) for a in accs]))
            entry = {"r2": _f(r2[si]), "mse": _f(tot[3] / tot[0]) if tot[0] else None, "n_scored": int(tot[0]),
                     "baselines": {}}
            for bi, bname in enumerate(MAE_BASELINES):
                bacc = [o.test["by_mask"][name]["_base"][bname] for o in outcomes]
                bsse = sum(a.sse.sum() for a in bacc)
                b_r2[si, bi] = R2Accumulator.r2(tot[0], tot[1], tot[2], bsse)
                b_pair[si, bi] = R2Accumulator.r2(pair[0], pair[1], pair[2], sum(a.sse.sum(0) for a in bacc))
                skill[si, bi] = 1 - tot[3] / bsse if bsse > 0 else np.nan
                entry["baselines"][bname] = {"r2": _f(b_r2[si, bi]), "skill": _f(skill[si, bi])}
            by_mask[name] = entry
        train_name = names[0]
        best_b = int(np.nanargmax(b_r2[0])) if np.isfinite(b_r2[0]).any() else 0
        lines = [f"Hidden values ({train_name.replace('_', ' ')}) on {what}: R² = {r2[0]:.2f} "
                 f"(best simple baseline, {MAE_BASELINES[best_b].replace('_', ' ')}: {b_r2[0, best_b]:.2f}), "
                 f"from {how}."]
        metrics.update({"headline": {"name": "r2", "value": _f(r2[0]), "mask": train_name,
                                     "baseline": MAE_BASELINES[best_b], "baseline_value": _f(b_r2[0, best_b]),
                                     "description": f"R² of hidden values on {what}"},
                        "test": by_mask, "val": {"loss": float(np.mean([o.val["metrics"]["loss"] for o in outcomes]))},
                        "summary_lines": lines})
        mat.update({"mask_names": names, "r2": r2, "r2_by_pair": r2_pair, "r2_by_patch": r2_patch,
                    "baseline_names": list(MAE_BASELINES), "baseline_r2": b_r2, "baseline_r2_by_pair": b_pair,
                    "skill": skill, "pt_trial_index": (sel.trial_index[tested] + 1).astype(float),
                    "pt_n": np.stack(pt["n"]), "pt_sum_y": np.stack(pt["s1"]), "pt_sum_y2": np.stack(pt["s2"]),
                    "pt_sse": np.stack(pt["sse"])})
        ex = outcomes[0].test.get("examples")
        if ex:
            K = len(ex["rows"])
            shape = (K, grid.n_tokens, grid.patch)
            orig = grid.from_tokens(np.where(ex["valid"][..., None], ex["target"], np.nan).reshape(shape), len(ch), len(mo))
            rec = grid.from_tokens(ex["recon"].reshape(shape), len(ch), len(mo))
            hid = grid.from_tokens(np.repeat(ex["masked"].reshape(K, -1, 1), grid.patch, 2).astype(np.float32), len(ch), len(mo))
            mat.update({"example_trial_index": (sel.trial_index[ex["rows"]] + 1).astype(float),
                        "example_original": norm.inverse(orig), "example_reconstruction": norm.inverse(rec),
                        "example_hidden": np.nan_to_num(hid) > 0.5, "example_mask_name": train_name})
    metrics["run_summary"] = metrics["summary_lines"][0]
    early = [o for o in outcomes if o.fit.stopped in ("stop_requested", "time_limit")]
    if early:
        why = "the time limit was reached" if all(o.fit.stopped == "time_limit" for o in early) else \
            "it was asked to stop (Stop button or the cluster's time limit)"
        where = "" if len(outcomes) == 1 else f" in {len(early)} of {len(outcomes)} folds"
        metrics.setdefault("summary_lines", []).append(
            f"Training was stopped early{where} because {why}; the results use the best model up to then.")
    return to_jsonable(metrics), mat


def _f(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


def summarize(run_dir: str | Path) -> dict:
    """Short description of a finished (or running) run, for the command line and the app."""
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir)}
    for name in ("status.json", "metrics.json"):
        p = run_dir / name
        if p.exists():
            out[name.split(".")[0]] = read_json(p)
    return out
