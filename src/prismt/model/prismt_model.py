"""The full PRISMT model: encoder plus a classification or reconstruction head."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from prismt.model.backbone import Encoder


@dataclass(frozen=True)
class ModelSpec:
    task: str  # classify | mae
    n_pairs: int
    n_patches: int
    patch: int
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int
    dropout: float
    attention: str
    position_embedding: str
    n_classes: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModelSpec":
        return cls(**json.loads(text))

    @classmethod
    def from_config(cls, model_cfg: dict, task: str, n_pairs: int, n_patches: int, patch: int, n_classes: int) -> "ModelSpec":
        return cls(task=task, n_pairs=n_pairs, n_patches=n_patches, patch=patch, d_model=model_cfg["d_model"],
                   n_layers=model_cfg["n_layers"], n_heads=model_cfg["n_heads"], ff_mult=model_cfg["ff_mult"],
                   dropout=model_cfg["dropout"], attention=model_cfg["attention"],
                   position_embedding=model_cfg["position_embedding"], n_classes=n_classes)


@dataclass
class ModelOutput:
    logits: torch.Tensor | None  # [batch, classes]
    reconstruction: torch.Tensor | None  # [batch, tokens, patch]
    cls: torch.Tensor
    internals: dict


class PrismtModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = Encoder(spec.n_pairs, spec.n_patches, spec.patch, spec.d_model, spec.n_layers, spec.n_heads,
                               spec.ff_mult, spec.dropout, spec.attention, spec.position_embedding)
        if spec.task == "classify":
            self.head = nn.Sequential(nn.Dropout(spec.dropout), nn.Linear(spec.d_model, spec.n_classes))
        elif spec.task == "mae":
            self.head = nn.Linear(spec.d_model, spec.patch)
        else:
            raise ValueError(f"unknown task {spec.task}")

    def forward(self, x: torch.Tensor, valid: torch.Tensor, masked: torch.Tensor | None = None,
                need_weights: bool = False) -> ModelOutput:
        out = self.encoder(x, valid, masked, need_weights)
        if self.spec.task == "classify":
            return ModelOutput(self.head(out.cls), None, out.cls, out.internals)
        return ModelOutput(None, self.head(out.tokens), out.cls, out.internals)

    def head_logit_weights(self) -> torch.Tensor:
        """[classes, d_model] weights of the classification layer (for later attribution)."""
        return self.head[-1].weight


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
