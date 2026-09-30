"""The one training loop, used for classification, masked autoencoding and tuning.

One validation score drives everything: the best epoch is the one with the best score,
early stopping counts epochs since that best, and automatic tuning ranks settings by it.
The test trials are never given to this function; they are evaluated once, after the best
model has been restored.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from prismt.data.splits import Fold
from prismt.data.tensors import TrialTensors, batches
from prismt.errors import Cancelled, TrainingDiverged
from prismt.io.runfiles import HistoryWriter, StatusWriter
from prismt.train.optim import build_optimizer, build_scheduler

log = logging.getLogger("prismt")

HIGHER_IS_BETTER = {"val_balanced_accuracy": True, "val_loss": False}


@dataclass
class FitResult:
    best_epoch: int
    best_value: float
    epochs_run: int
    stopped: str  # patience | max_epochs | time_limit | stop_requested
    seconds: float
    history: list[dict] = field(default_factory=list)
    skipped_steps: int = 0


class StopTraining(Exception):
    """Raised by callbacks (a STOP file, a signal) to end training after the current step."""


def fit(
    model: torch.nn.Module,
    task,
    tensors: TrialTensors,
    fold: Fold,
    train_cfg: dict,
    *,
    monitor: str = "val_loss",
    status: StatusWriter | None = None,
    history: HistoryWriter | None = None,
    fold_label: int = 1,
    n_folds: int = 1,
    should_stop: Callable[[], str | None] | None = None,
    on_epoch_end: Callable[[int, float], None] | None = None,
    eval_batch_size: int | None = None,
) -> FitResult:
    """Train on ``fold.train``, select and stop on ``fold.val``; restores the best weights."""
    device = tensors.device
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    eval_bs = eval_batch_size or max(batch_size, 64)
    steps_per_epoch = max(1, math.ceil(len(fold.train) / batch_size))
    optimizer = build_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, epochs * steps_per_epoch, train_cfg["warmup_fraction"],
                                train_cfg["min_lr_ratio"])
    seed = int(train_cfg["seed"])
    rng = np.random.default_rng(seed + 7919 * fold_label)
    mask_gen = torch.Generator().manual_seed(seed + 104729 * fold_label)
    higher = HIGHER_IS_BETTER[monitor]
    best_value = -math.inf if higher else math.inf
    best_epoch, best_state = 0, copy.deepcopy(model.state_dict())
    patience = int(train_cfg["patience"])
    deadline = time.time() + 60 * train_cfg["max_minutes"] if train_cfg.get("max_minutes") else None
    start = time.time()
    epoch_times: list[float] = []
    stopped, skipped, nan_streak = "max_epochs", 0, 0
    rows: list[dict] = []
    epoch = 0
    model.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        reason = None
        for step, rows_b in enumerate(batches(fold.train, batch_size, shuffle=True, rng=rng), start=1):
            x, v, y = tensors.batch(rows_b)
            loss = task.loss(model, x, v, y, generator=mask_gen)
            if not torch.isfinite(loss):
                skipped += 1
                nan_streak += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_streak >= 5:
                    raise TrainingDiverged(
                        "E_TRAIN_DIVERGED",
                        "The training loss became invalid (NaN or infinite) five steps in a row.",
                        hint="Lower the learning rate (for example by a factor of 10), or check the data for "
                             "extreme values.", field="train.lr")
                continue
            nan_streak = 0
            loss.backward()
            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(rows_b)
            count += len(rows_b)
            if status is not None:
                status.update("training", fold=fold_label, n_folds=n_folds, epoch=epoch, epochs=epochs,
                              step=step, steps_per_epoch=steps_per_epoch)
            reason = _stop_reason(should_stop, deadline)
            if reason:
                break
        train_loss = total / count if count else float("nan")
        evaluation = task.evaluate(model, tensors, fold.val, eval_bs)
        metrics = evaluation["metrics"]
        value = metrics["loss"] if monitor == "val_loss" else metrics.get("balanced_accuracy")
        value = float("nan") if value is None else float(value)
        improved = np.isfinite(value) and ((value > best_value) if higher else (value < best_value))
        if improved:
            best_value, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_times.append(time.time() - t0)
        row = {"fold": fold_label, "epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_loss": train_loss,
               "val_loss": metrics.get("loss")}
        for key in ("balanced_accuracy", "accuracy", "macro_f1", "auroc", "r2"):
            if metrics.get(key) is not None:
                row[f"val_{key}"] = metrics[key]
        row.update({"seconds": round(epoch_times[-1], 2), "is_best": bool(improved)})
        rows.append(row)
        if history is not None:
            history.add(row)
        remaining = epochs - epoch
        if status is not None:
            status.update("training", force=True, fold=fold_label, n_folds=n_folds, epoch=epoch, epochs=epochs,
                          best_epoch=best_epoch, best_value=best_value if np.isfinite(best_value) else None,
                          monitor=monitor, latest={k: v for k, v in row.items() if k.startswith(("val_", "train_"))},
                          eta_s=round(remaining * float(np.mean(epoch_times[-5:])), 1),
                          message=f"Fold {fold_label}/{n_folds}, epoch {epoch}/{epochs}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, value)
        log.info("fold %d epoch %d/%d  train %.4f  val %s %.4f%s", fold_label, epoch, epochs, train_loss,
                 monitor, value, "  *" if improved else "")
        if reason:
            stopped = reason
            break
        if epoch >= int(train_cfg.get("min_epochs") or 0) and epoch - best_epoch >= patience:
            stopped = "patience"
            break
    model.load_state_dict(best_state)
    if best_epoch == 0:
        best_value = float("nan")
    return FitResult(best_epoch, best_value, epoch, stopped, time.time() - start, rows, skipped)


def _stop_reason(should_stop: Callable[[], str | None] | None, deadline: float | None) -> str | None:
    if deadline is not None and time.time() > deadline:
        return "time_limit"
    if should_stop is not None:
        reason = should_stop()
        if reason == "cancel":
            raise Cancelled("E_CANCELLED", "The run was stopped before training finished.",
                            hint="Start it again to continue from scratch.")
        return reason
    return None
