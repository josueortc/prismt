"""Which tokens a masked autoencoder hides.

Convention (as in MouseWFM's ``masking/apply.py``): ``masked`` is True where a token is
hidden from the encoder and scored by the loss. Tokens that are invalid to begin with
(missing values) are never masked, never visible and never scored; they are tracked in
``valid``. So ``scored = masked & valid`` and ``visible = ~masked & valid``.

Masks are drawn on the CPU from an explicit generator, so a given seed gives the same mask
on a laptop and on a cluster GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from prismt.data.tokens import TokenGrid

STRATEGIES = ("random", "channel", "forecast", "modality")


@dataclass(frozen=True)
class MaskSpec:
    strategy: str
    ratio: float = 0.9
    context_fraction: float = 0.5
    modality: int | None = None  # position of the modality to hide; None = random per trial
    label: str | None = None  # name used in results (e.g. "modality_ach")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        if self.strategy == "random":
            return f"random_{self.ratio:g}"
        if self.strategy == "channel":
            return f"channel_{self.ratio:g}"
        if self.strategy == "forecast":
            return f"forecast_{self.context_fraction:g}"
        return "modality" if self.modality is None else f"modality_{self.modality}"


def build_mask(spec: MaskSpec, valid: torch.Tensor, grid: TokenGrid, generator: torch.Generator,
               pair_modality: np.ndarray | None = None) -> torch.Tensor:
    """Boolean [batch, tokens] mask, True = hidden. Always leaves at least one valid token visible."""
    valid_cpu = valid.detach().to("cpu")
    B, L = valid_cpu.shape
    n_pairs, n_patches = grid.n_pairs, grid.n_patches
    if spec.strategy == "random":
        scores = torch.rand((B, L), generator=generator)
        scores[~valid_cpu] = 2.0  # invalid tokens sort last and are never chosen
        n_valid = valid_cpu.sum(1)
        k = torch.floor(spec.ratio * n_valid.float()).long().clamp(max=(n_valid - 1).clamp(min=0))
        rank = scores.argsort(1).argsort(1)
        masked = rank < k[:, None]
    elif spec.strategy == "channel":
        pv = valid_cpu.view(B, n_patches, n_pairs).any(1)  # [B, pairs]: pair has some data
        scores = torch.rand((B, n_pairs), generator=generator)
        scores[~pv] = 2.0
        n_valid = pv.sum(1)
        k = torch.floor(spec.ratio * n_valid.float()).long().clamp(min=1).clamp(max=(n_valid - 1).clamp(min=0))
        rank = scores.argsort(1).argsort(1)
        hide = rank < k[:, None]  # [B, pairs]
        masked = hide[:, None, :].expand(B, n_patches, n_pairs).reshape(B, L)
    elif spec.strategy == "forecast":
        first_hidden = min(max(1, math.ceil(spec.context_fraction * n_patches)), n_patches - 1) if n_patches > 1 else 1
        patch = torch.as_tensor(grid.token_patch)
        masked = (patch >= first_hidden)[None, :].expand(B, L).clone()
    elif spec.strategy == "modality":
        if pair_modality is None:
            raise ValueError("modality masking needs the modality of each pair")
        mods = np.unique(pair_modality)
        if len(mods) < 2:
            raise ValueError("modality masking needs at least two modalities")
        pm = torch.as_tensor(pair_modality)
        if spec.modality is None:
            choice = torch.as_tensor(mods)[torch.randint(len(mods), (B,), generator=generator)]
        else:
            choice = torch.full((B,), int(spec.modality), dtype=torch.long)
        hide = pm[None, :] == choice[:, None]  # [B, pairs]
        masked = hide[:, None, :].expand(B, n_patches, n_pairs).reshape(B, L)
    else:
        raise ValueError(f"unknown masking strategy {spec.strategy}")
    masked = masked & valid_cpu
    # Never hide every valid token of a trial: reveal one at random if that happened.
    none_visible = (valid_cpu & ~masked).sum(1) == 0
    if none_visible.any():
        for b in torch.nonzero(none_visible).flatten().tolist():
            cand = torch.nonzero(valid_cpu[b]).flatten()
            if len(cand):
                masked[b, cand[torch.randint(len(cand), (1,), generator=generator)]] = False
    return masked.to(valid.device)
