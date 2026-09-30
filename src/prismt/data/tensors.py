"""Token arrays as torch tensors, and mini-batch iteration.

Data are kept in memory (on the GPU when they fit), and batches are drawn with plain
index arrays: no DataLoader worker processes, which avoids duplicated memory and
orphaned workers on Windows and macOS. Copies to the device are blocking; non-blocking
copies have corrupted batches on Apple GPUs (MPS) in PyTorch 2.5.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

_ON_DEVICE_LIMIT = 1 << 30  # keep the whole dataset on the GPU below 1 GB


@dataclass
class TrialTensors:
    x: torch.Tensor  # [n, tokens, patch], missing values set to 0
    valid: torch.Tensor  # [n, tokens] bool
    y: torch.Tensor | None  # [n] long
    device: torch.device

    @classmethod
    def make(cls, values: np.ndarray, valid: np.ndarray, y: np.ndarray | None, device: torch.device) -> "TrialTensors":
        x = torch.from_numpy(np.nan_to_num(values, nan=0.0).astype(np.float32))
        v = torch.from_numpy(np.asarray(valid, dtype=bool))
        t = None if y is None else torch.from_numpy(np.asarray(y, dtype=np.int64))
        home = device if (device.type != "cpu" and x.element_size() * x.nelement() < _ON_DEVICE_LIMIT) else torch.device("cpu")
        x, v = x.to(home), v.to(home)
        t = None if t is None else t.to(home)
        return cls(x, v, t, device)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def batch(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        idx = torch.as_tensor(rows, dtype=torch.long, device=self.x.device)
        x = self.x.index_select(0, idx).to(self.device)
        v = self.valid.index_select(0, idx).to(self.device)
        y = None if self.y is None else self.y.index_select(0, idx).to(self.device)
        return x, v, y


def batches(rows: np.ndarray, batch_size: int, *, shuffle: bool, rng: np.random.Generator | None = None) -> Iterator[np.ndarray]:
    rows = np.asarray(rows)
    if shuffle:
        rows = rng.permutation(rows) if rng is not None else np.random.permutation(rows)
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]
