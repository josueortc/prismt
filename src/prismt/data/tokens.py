"""How trials become token sequences.

Each (channel, modality) pair that exists contributes one token per time patch; a patch is
``patch`` consecutive time bins (1 = the PRISMt paper's one-value tokens). Tokens are
ordered by time: every pair at patch 0, then every pair at patch 1, and so on. A token is
valid only if all of its values are present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prismt.errors import ConfigError


@dataclass(frozen=True)
class TokenGrid:
    pairs: tuple[tuple[int, int], ...]  # (channel position, modality position) in the extracted array
    n_bins: int
    patch: int

    @staticmethod
    def make(pairs: list[tuple[int, int]], n_bins: int, time_patch: int | str) -> "TokenGrid":
        patch = n_bins if time_patch == "all" else int(time_patch)
        if patch > n_bins or n_bins % patch:
            divisors = [d for d in range(1, n_bins + 1) if n_bins % d == 0]
            raise ConfigError(
                "E_CFG_PATCH",
                f"'Time bins per token' ({patch}) must divide the number of time bins ({n_bins}).",
                hint=f"Use one of {', '.join(map(str, divisors))}, or all.",
                field="model.time_patch",
                title="The model settings do not fit this dataset",
            )
        return TokenGrid(tuple((int(c), int(m)) for c, m in pairs), int(n_bins), patch)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_patches(self) -> int:
        return self.n_bins // self.patch

    @property
    def n_tokens(self) -> int:
        return self.n_pairs * self.n_patches

    @property
    def token_patch(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_patches), self.n_pairs)

    @property
    def token_pair(self) -> np.ndarray:
        return np.tile(np.arange(self.n_pairs), self.n_patches)

    def to_tokens(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X [n, channels, bins, modalities] -> values [n, tokens, patch] and valid [n, tokens]."""
        rs = np.array([c for c, _ in self.pairs])
        ms = np.array([m for _, m in self.pairs])
        # Two index arrays separated by a slice: numpy puts their dimension first -> (pairs, n, bins).
        V = X[:, rs, :, ms].transpose(1, 0, 2)
        n = V.shape[0]
        V = V.reshape(n, self.n_pairs, self.n_patches, self.patch).transpose(0, 2, 1, 3)
        V = np.ascontiguousarray(V.reshape(n, self.n_tokens, self.patch), dtype=np.float32)
        valid = np.isfinite(V).all(axis=2)
        return V, valid

    def from_tokens(self, V: np.ndarray, n_channels: int, n_modalities: int) -> np.ndarray:
        """Inverse of :meth:`to_tokens`: [n, tokens, patch] -> [n, channels, bins, modalities] (NaN elsewhere)."""
        n = V.shape[0]
        W = V.reshape(n, self.n_patches, self.n_pairs, self.patch).transpose(0, 2, 1, 3).reshape(n, self.n_pairs, self.n_bins)
        out = np.full((n, n_channels, self.n_bins, n_modalities), np.nan, dtype=np.float32)
        for q, (c, m) in enumerate(self.pairs):
            out[:, c, :, m] = W[:, q, :]
        return out

    def signature(self) -> dict:
        return {"pairs": [list(p) for p in self.pairs], "n_bins": self.n_bins, "patch": self.patch}
