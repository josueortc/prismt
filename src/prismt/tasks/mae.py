"""Masked autoencoding: hide tokens, predict their values, score only what was hidden.

The loss is the mean squared error over hidden, valid values. Evaluation uses a fixed set
of masks drawn from counter-based seeds (the same masks every epoch), so the validation
loss can be compared from epoch to epoch, and every baseline is scored on exactly the
same hidden values as the model.
"""

from __future__ import annotations

import numpy as np
import torch

from prismt.data.tensors import TrialTensors, batches
from prismt.data.tokens import TokenGrid
from prismt.eval.baselines import MAE_BASELINES, mae_predictions
from prismt.eval.metrics import R2Accumulator
from prismt.masking import MaskSpec, build_mask


def masked_mse(recon: torch.Tensor, target: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    err = ((recon - target) ** 2).mean(-1)
    w = scored.to(err.dtype)
    return (err * w).sum() / w.sum().clamp(min=1.0)


class MAETask:
    name = "mae"
    monitor_default = "val_loss"

    def __init__(self, grid: TokenGrid, train_spec: MaskSpec, eval_specs: list[MaskSpec],
                 pair_modality: np.ndarray, seed: int) -> None:
        self.grid = grid
        self.train_spec = train_spec
        self.eval_specs = eval_specs
        self.pair_modality = pair_modality
        self.seed = seed

    def loss(self, model, x, valid, y=None, *, generator: torch.Generator) -> torch.Tensor:
        masked = build_mask(self.train_spec, valid, self.grid, generator, self.pair_modality)
        out = model(x, valid, masked)
        return masked_mse(out.reconstruction, x, masked & valid)

    def _eval_generator(self, spec_index: int, batch_index: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed * 1_000_003 + spec_index * 10_007 + batch_index)

    @torch.no_grad()
    def evaluate(self, model, tensors: TrialTensors, rows: np.ndarray, batch_size: int, *,
                 psth: np.ndarray | None = None, specs: list[MaskSpec] | None = None,
                 n_examples: int = 0, with_baselines: bool = False, need_embeddings: bool = False) -> dict:
        """Loss and R² under each mask spec (the first is the training mask, which gives 'loss')."""
        model.eval()
        g = self.grid
        specs = specs or self.eval_specs
        results = {}
        examples = None
        embeddings = []
        for si, spec in enumerate(specs):
            acc = R2Accumulator(len(rows), g.n_pairs, g.n_patches)
            base_acc = {name: R2Accumulator(len(rows), g.n_pairs, g.n_patches) for name in MAE_BASELINES} \
                if with_baselines else {}
            sse, count = 0.0, 0.0
            offset = 0
            for bi, b in enumerate(batches(rows, batch_size, shuffle=False)):
                x, v, _ = tensors.batch(b)
                masked = build_mask(spec, v, g, self._eval_generator(si, bi), self.pair_modality)
                out = model(x, v, masked)
                scored = masked & v
                err = ((out.reconstruction - x) ** 2).mean(-1)
                sse += float((err * scored).sum())
                count += float(scored.sum())
                local = np.arange(offset, offset + len(b))
                offset += len(b)
                shape = (len(b), g.n_patches, g.n_pairs, g.patch)
                target = x.float().cpu().numpy().reshape(shape)
                pred = out.reconstruction.float().cpu().numpy().reshape(shape)
                sc = scored.cpu().numpy().reshape(shape[:3])
                acc.add(local, sc, target, pred)
                if with_baselines and psth is not None:
                    vis = (v & ~masked).cpu().numpy().reshape(shape[:3])
                    for name, bp in mae_predictions(target, vis, psth).items():
                        base_acc[name].add(local, sc, target, bp)
                if si == 0 and need_embeddings:
                    embeddings.append(model(x, v).cls.float().cpu().numpy())
                if si == 0 and n_examples and examples is None:
                    k = min(n_examples, len(b))
                    examples = {"rows": b[:k], "target": target[:k], "recon": pred[:k], "masked": sc[:k],
                                "valid": v.cpu().numpy().reshape(shape[:3])[:k]}
            summary = acc.summary()
            summary["loss"] = sse / count if count else float("nan")
            if with_baselines:
                summary["baselines"] = {}
                for name, a in base_acc.items():
                    bs = a.summary()
                    base_sse = a.sse.sum()
                    bs["skill"] = float(1 - acc.sse.sum() / base_sse) if base_sse > 0 else None
                    summary["baselines"][name] = bs
            summary["_acc"] = acc  # raw sums, used to pool folds; not written to JSON
            summary["_base"] = base_acc
            results[spec.name] = summary
        first = results[specs[0].name]
        out = {"metrics": {"loss": first["loss"], "r2": first["r2"]}, "by_mask": results, "examples": examples}
        if need_embeddings:
            out["embedding"] = np.concatenate(embeddings) if embeddings else np.zeros((0, model.spec.d_model))
        return out

    @staticmethod
    def psth(values: np.ndarray, valid: np.ndarray, rows: np.ndarray, grid: TokenGrid) -> np.ndarray:
        """Training-trial average of every token, [patches, pairs, patch_len] (0 where never present)."""
        V = values[rows].reshape(len(rows), grid.n_patches, grid.n_pairs, grid.patch)
        W = valid[rows].reshape(len(rows), grid.n_patches, grid.n_pairs)[..., None]
        s = np.where(W, V, 0.0).sum(0)
        c = W.sum(0)
        return np.divide(s, c, out=np.zeros_like(s), where=c > 0)
