"""Saving and loading models safely.

Checkpoints hold only tensors, strings and numbers, so they load with
``torch.load(weights_only=True)``: no pickled code runs when a model is opened. The model
architecture is stored with the weights, and loading a pretrained encoder is strict: every
parameter must match in name and shape, or the load fails. A partially loaded model
trains silently from random weights and invalidates the result (knowledge-base pitfall
"silent-partial-checkpoint-loads").
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from prismt import __version__
from prismt.errors import CheckpointError
from prismt.model.prismt_model import ModelSpec, PrismtModel

FORMAT = "prismt.checkpoint"
VERSION = 1
_TITLE = "A saved model could not be used"


def save_checkpoint(path: str | Path, model: PrismtModel, *, extra: dict | None = None) -> Path:
    path = Path(path)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "prismt_version": __version__,
        "model_spec": model.spec.to_json(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    for key, value in (extra or {}).items():
        payload[key] = value if isinstance(value, (torch.Tensor, str, int, float, bool)) else json.dumps(value)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise CheckpointError("E_CKPT_MISSING", f"No saved model at {path}.",
                              hint="Give the folder of a finished run (it contains model.pt).", title=_TITLE)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError("E_CKPT_UNREADABLE", f"{path} could not be read ({exc}).", title=_TITLE) from exc
    if not isinstance(ckpt, dict) or ckpt.get("format") != FORMAT:
        raise CheckpointError("E_CKPT_FORMAT", f"{path} is not a PRISMT model.", title=_TITLE)
    if int(ckpt.get("version", 0)) > VERSION:
        raise CheckpointError("E_CKPT_NEWER", f"{path} was saved by a newer PRISMT.", hint="Update PRISMT.", title=_TITLE)
    return ckpt


def build_from_checkpoint(ckpt: dict) -> PrismtModel:
    model = PrismtModel(ModelSpec.from_json(ckpt["model_spec"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def extra(ckpt: dict, key: str, default=None):
    value = ckpt.get(key, default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass
class LoadReport:
    loaded: int
    total: int
    source: str

    @property
    def coverage(self) -> float:
        return self.loaded / self.total if self.total else 0.0


def load_pretrained_encoder(model: PrismtModel, ckpt: dict, *, source: str = "") -> LoadReport:
    """Copy an autoencoder's encoder into ``model``. Refuses anything short of a perfect match."""
    spec = ModelSpec.from_json(ckpt["model_spec"])
    mine = model.spec
    for name in ("n_pairs", "n_patches", "patch", "d_model", "n_layers", "n_heads", "ff_mult", "attention",
                 "position_embedding"):
        if getattr(spec, name) != getattr(mine, name):
            raise CheckpointError(
                "E_CKPT_MISMATCH",
                f"The autoencoder was built with {name} = {getattr(spec, name)}, but this run uses {getattr(mine, name)}.",
                hint="Use the same model and time settings (and the same channels and modalities) as the "
                     "autoencoder run.", field=f"model.{name}", title=_TITLE)
    encoder_state = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    target = model.encoder.state_dict()
    missing = sorted(set(target) - set(encoder_state))
    unexpected = sorted(set(encoder_state) - set(target))
    wrong = sorted(k for k in set(target) & set(encoder_state) if target[k].shape != encoder_state[k].shape)
    if missing or unexpected or wrong:
        raise CheckpointError("E_CKPT_PARTIAL",
                              f"The autoencoder's weights do not fully match this model ({len(missing)} missing, "
                              f"{len(unexpected)} unexpected, {len(wrong)} with another shape).",
                              hint="Re-run the autoencoder with this version of PRISMT.", title=_TITLE)
    model.encoder.load_state_dict(encoder_state, strict=True)
    report = LoadReport(len(encoder_state), len(target), source)
    if report.coverage != 1.0:
        raise CheckpointError("E_CKPT_PARTIAL", "Not every encoder weight was loaded.", title=_TITLE)
    return report
