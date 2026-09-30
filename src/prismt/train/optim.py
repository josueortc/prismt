"""AdamW with decoupled weight decay on linear weights only, and a warm-up + cosine schedule."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Decay only the weight matrices of linear layers; never biases, norms, embeddings, CLS or [MASK]."""
    decay, no_decay = [], []
    linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if id(p) in linear_weights else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    """Linear warm-up from lr/warmup to lr, then cosine decay to lr * min_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float, min_ratio: float):
    warmup = int(round(warmup_fraction * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total_steps, warmup, min_ratio))
