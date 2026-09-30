"""Classification: predict each trial's class from the CLS token."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from prismt.data.tensors import TrialTensors, batches
from prismt.eval.metrics import classification_metrics


class ClassifyTask:
    name = "classify"
    monitor_default = "val_loss"

    def __init__(self, n_classes: int, class_weights: np.ndarray | None) -> None:
        self.n_classes = n_classes
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)

    @staticmethod
    def balanced_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
        counts = np.bincount(y_train, minlength=n_classes).astype(float)
        w = np.divide(len(y_train), n_classes * counts, out=np.zeros(n_classes), where=counts > 0)
        return w

    def loss(self, model, x, valid, y, *, generator=None) -> torch.Tensor:
        logits = model(x, valid).logits
        w = None if self.class_weights is None else self.class_weights.to(logits.device)
        return F.cross_entropy(logits, y, weight=w)

    @torch.no_grad()
    def predict(self, model, tensors: TrialTensors, rows: np.ndarray, batch_size: int) -> dict:
        model.eval()
        probs, embeddings, losses, counts = [], [], [], []
        w = None if self.class_weights is None else self.class_weights.to(tensors.device)
        for b in batches(rows, batch_size, shuffle=False):
            x, v, y = tensors.batch(b)
            out = model(x, v)
            probs.append(out.logits.float().softmax(-1).cpu().numpy())
            embeddings.append(out.cls.float().cpu().numpy())
            if y is not None:
                losses.append(float(F.cross_entropy(out.logits, y, weight=w, reduction="sum")))
                counts.append(float(w[y].sum()) if w is not None else len(b))
        prob = np.concatenate(probs) if probs else np.zeros((0, self.n_classes))
        emb = np.concatenate(embeddings) if embeddings else np.zeros((0, model.spec.d_model))
        loss = sum(losses) / max(sum(counts), 1e-12) if losses else float("nan")
        return {"prob": prob, "embedding": emb, "loss": loss}

    def evaluate(self, model, tensors: TrialTensors, rows: np.ndarray, batch_size: int) -> dict:
        out = self.predict(model, tensors, rows, batch_size)
        y = tensors.y[torch.as_tensor(rows, dtype=torch.long, device=tensors.y.device)].cpu().numpy()
        metrics = classification_metrics(y, out["prob"], self.n_classes) if len(rows) else {}
        metrics["loss"] = out["loss"]
        return {"metrics": metrics, "prob": out["prob"], "embedding": out["embedding"], "y": y}
