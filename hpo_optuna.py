#!/usr/bin/env python3
"""
PRISMT: Optuna-based Hyperparameter Optimization

Runs Optuna trials over the WidefieldTransformer, reusing PRISMT data loaders
and model infrastructure. Supports both phase and genotype classification.
"""

import os
import math
import json
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna

# Add project root
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from data.data_loader import WidefieldDataset, TaskDefinition, WidefieldTrialDataset
from models.transformer import WidefieldTransformer
from train import (
    detect_task_type,
    create_data_loaders_unified,
    create_train_val_split,
)
from utils.helpers import set_seed, get_device
from training.trainer import run_attention_and_diagnosis

logger = logging.getLogger(__name__)


@dataclass
class TrialConfig:
    """Config for a single Optuna trial."""
    lr: float
    scheduler: str
    warmup_ratio: float
    batch_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    dropout: float
    weight_decay: float
    max_epochs: int
    grad_clip: float
    seed: int


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        if len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits, y)

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()

        bs = y.shape[0]
        total_loss += loss.item() * bs
        n += bs

    return {
        "val_loss": total_loss / max(1, n),
        "val_acc": correct / max(1, n),
    }


def make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    steps_per_epoch: int,
    warmup_ratio: float,
):
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)

    if name == "none":
        return None

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, max_epochs // 3), gamma=0.2
        )

    if name == "plateau" or name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

    if name == "cosine_warmup" or name == "cosine":

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )

    raise ValueError(f"Unknown scheduler: {name}")


def objective(
    trial: optuna.Trial,
    args,
    dataset: WidefieldDataset,
    train_indices: List[int],
    val_indices: List[int],
    task_type: str,
    n_brain_areas: int,
    time_points: int,
    target_values=None,
    filters=None,
) -> float:
    cfg = TrialConfig(
        lr=trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        scheduler=trial.suggest_categorical(
            "scheduler", ["cosine_warmup", "step", "reduce_on_plateau", "cosine"]
        ),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
        num_layers=trial.suggest_int("num_layers", 1, 6),
        ff_dim=trial.suggest_categorical("ff_dim", [128, 256, 512]),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    # Transformer requires hidden_dim % num_heads == 0
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise optuna.TrialPruned()

    if cfg.batch_size % 2 != 0:
        cfg.batch_size = (cfg.batch_size // 2) * 2

    try:
        return _run_trial(trial, args, cfg, dataset, train_indices, val_indices,
                         task_type, n_brain_areas, time_points,
                         target_values=target_values, filters=filters)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise


def _run_trial(
    trial: optuna.Trial,
    args,
    cfg: TrialConfig,
    dataset: WidefieldDataset,
    train_indices: List[int],
    val_indices: List[int],
    task_type: str,
    n_brain_areas: int,
    time_points: int,
    target_values=None,
    filters=None,
) -> float:
    set_seed(cfg.seed + trial.number)
    device = get_device(args.device)
    if target_values is None and args.target_values:
        target_values = [v.strip() for v in args.target_values.split(",") if v.strip()]
    if filters is None and args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError:
            filters = None

    train_loader, val_loader, _ = create_data_loaders_unified(
        dataset,
        train_indices,
        val_indices,
        task_type,
        args.data_type,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        phase1=args.phase1,
        phase2=args.phase2,
        region_pool=args.region_pool,
        time_pool=args.time_pool,
        task_mode=args.task_mode,
        target_column=args.target_column,
        target_values=target_values,
        filters=filters,
    )

    model = WidefieldTransformer(
        n_brain_areas=n_brain_areas,
        time_points=time_points,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        num_classes=2,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    sched = make_scheduler(
        cfg.scheduler,
        optimizer,
        cfg.max_epochs,
        steps_per_epoch,
        cfg.warmup_ratio,
    )

    best_metric = -float("inf") if args.mode == "max" else float("inf")
    best_path = os.path.join(args.out_dir, f"trial_{trial.number}_best.pt")

    global_step = 0
    for epoch in range(cfg.max_epochs):
        _ = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, cfg.grad_clip
        )
        metrics = evaluate(model, val_loader, loss_fn, device)

        score = metrics[args.metric]

        if sched is not None:
            if cfg.scheduler == "plateau" or cfg.scheduler == "reduce_on_plateau":
                sched.step(score)
            elif cfg.scheduler in ("cosine_warmup", "cosine"):
                for _ in range(steps_per_epoch):
                    sched.step()
                    global_step += 1
            else:
                sched.step()

        trial.report(score, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        improved = (score > best_metric) if args.mode == "max" else (score < best_metric)
        if improved:
            best_metric = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "metrics": metrics,
                },
                best_path,
            )

    trial.set_user_attr("best_metric", best_metric)
    trial.set_user_attr("best_ckpt", best_path)
    trial.set_user_attr("cfg", asdict(cfg))
    return best_metric


def main():
    parser = argparse.ArgumentParser(description="PRISMT Optuna HPO")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="dff", choices=["dff", "zscore"])
    parser.add_argument("--task_type", type=str, default="auto", choices=["auto", "genotype", "phase"])
    parser.add_argument("--phase1", type=str, default=None)
    parser.add_argument("--phase2", type=str, default=None)
    parser.add_argument("--task_mode", type=str, default="classification", choices=["classification", "regression"])
    parser.add_argument("--target_column", type=str, default=None, help="Column to predict (phase, mouse, stim, response)")
    parser.add_argument("--target_values", type=str, default=None, help="Comma-separated values for classification")
    parser.add_argument("--filters", type=str, default=None, help="JSON filters, e.g. '{\"stim\":[1],\"response\":[0,1]}'")
    parser.add_argument("--region_pool", type=int, default=1, help="Pool factor for brain regions (1=none, 2=avg pairs)")
    parser.add_argument("--time_pool", type=int, default=1, help="Pool factor for timepoints (1=none, 2=avg pairs)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--metric", type=str, default="val_acc")
    parser.add_argument("--mode", type=str, choices=["max", "min"], default="max")
    parser.add_argument("--out_dir", type=str, default="hpo_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--storage", type=str, default="",
        help="SQLite DB path. Default: sqlite:///<out_dir>/study.db for cluster resume"
    )
    parser.add_argument(
        "--pruner", type=str, default="median",
        choices=["median", "hyperband"],
        help="MedianPruner (default) or HyperbandPruner"
    )
    parser.add_argument(
        "--study_name", type=str, default="prismt_hpo",
        help="Optuna study name (for SQLite storage)"
    )
    parser.add_argument(
        "--attention_samples", type=int, default=1000,
        help="Number of samples for attention extraction (post-HPO)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = get_device(args.device)

    logger.info("Loading dataset...")
    dataset = WidefieldDataset(args.data_path)

    if args.task_type == "auto":
        task_type = detect_task_type(dataset)
    else:
        task_type = args.task_type

    if args.target_column is None and task_type == "phase" and (args.phase1 is None or args.phase2 is None):
        raise ValueError("--phase1 and --phase2 required for phase task")

    target_values = None
    filters = None
    if args.target_values:
        target_values = [v.strip() for v in args.target_values.split(",") if v.strip()]
    if args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --filters JSON: {e}") from e

    split_task = "phase" if args.target_column else task_type
    train_indices, val_indices = create_train_val_split(
        dataset, split_task, args.val_split, args.seed
    )

    train_loader, _, _ = create_data_loaders_unified(
        dataset,
        train_indices,
        val_indices,
        task_type,
        args.data_type,
        batch_size=16,
        num_workers=0,
        phase1=args.phase1,
        phase2=args.phase2,
        region_pool=args.region_pool,
        time_pool=args.time_pool,
        task_mode=args.task_mode,
        target_column=args.target_column,
        target_values=target_values,
        filters=filters,
    )
    sample_batch, _ = next(iter(train_loader))
    sample_data = sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
    # Batch shape: (batch, time_points, n_brain_areas)
    time_points = sample_data.shape[1]
    n_brain_areas = sample_data.shape[2]

    os.makedirs(args.out_dir, exist_ok=True)

    # Cluster-friendly: SQLite on shared FS enables resume after preemption/timeout
    storage = args.storage if args.storage else f"sqlite:///{os.path.abspath(args.out_dir)}/study.db"
    pruner_cls = optuna.pruners.HyperbandPruner if args.pruner == "hyperband" else optuna.pruners.MedianPruner
    pruner = pruner_cls() if args.pruner == "hyperband" else optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=3
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize" if args.mode == "max" else "minimize",
        pruner=pruner,
    )

    study.optimize(
        lambda t: objective(
            t, args, dataset, train_indices, val_indices,
            task_type, n_brain_areas, time_points,
            target_values=target_values,
            filters=filters,
        ),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_ckpt": best.user_attrs.get("best_ckpt"),
        "best_cfg": best.user_attrs.get("cfg"),
    }
    with open(os.path.join(args.out_dir, "best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Best value: %s", best.value)
    logger.info("Best params: %s", best.params)
    logger.info("Best checkpoint: %s", summary["best_ckpt"])

    # Attention and diagnosis on best model
    best_ckpt = summary["best_ckpt"]
    if best_ckpt and os.path.exists(best_ckpt):
        logger.info("Extracting attention and running diagnosis on best model...")
        best_cfg = summary.get("best_cfg", {})
        ckpt = torch.load(best_ckpt, map_location=device)
        best_model = WidefieldTransformer(
            n_brain_areas=n_brain_areas,
            time_points=time_points,
            hidden_dim=best_cfg.get("hidden_dim", 128),
            num_heads=best_cfg.get("num_heads", 4),
            num_layers=best_cfg.get("num_layers", 3),
            ff_dim=best_cfg.get("ff_dim", 256),
            num_classes=2,
            dropout=best_cfg.get("dropout", 0.3),
        ).to(device)
        best_model.load_state_dict(ckpt["model_state"])
        _, val_loader_hpo, _ = create_data_loaders_unified(
            dataset, train_indices, val_indices, task_type, args.data_type,
            batch_size=best_cfg.get("batch_size", 32),
            num_workers=args.num_workers,
            phase1=args.phase1,
            phase2=args.phase2,
            region_pool=args.region_pool,
            time_pool=args.time_pool,
            task_mode=args.task_mode,
            target_column=args.target_column,
            target_values=target_values,
            filters=filters,
        )
        run_attention_and_diagnosis(
            model=best_model,
            data_loader=val_loader_hpo,
            device=device,
            save_dir=args.out_dir,
            num_samples=args.attention_samples,
            num_classes=2,
        )
    else:
        logger.warning("Best checkpoint not found; skipping attention and diagnosis.")


if __name__ == "__main__":
    main()
