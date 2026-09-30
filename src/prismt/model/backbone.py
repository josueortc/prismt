"""The PRISMT encoder: tokenizer, CLS token and pre-norm transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from prismt.model.attention import SelfAttention, attend_mask


class Tokenizer(nn.Module):
    """Values of a token (``patch`` time bins) -> vector, plus a learned position code.

    Hidden (masked) tokens get a learned [MASK] vector instead of their values, plus the
    same position code. Missing values arrive as 0 and are excluded by the attend mask.
    """

    def __init__(self, n_pairs: int, n_patches: int, patch: int, d_model: int, position_embedding: str) -> None:
        super().__init__()
        self.value_proj = nn.Linear(patch, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.position_embedding = position_embedding
        L = n_pairs * n_patches
        self.register_buffer("token_pair", torch.arange(n_pairs).repeat(n_patches), persistent=False)
        self.register_buffer("token_patch", torch.arange(n_patches).repeat_interleave(n_pairs), persistent=False)
        if position_embedding == "channel_time":
            self.pair_emb = nn.Embedding(n_pairs, d_model)
            self.patch_emb = nn.Embedding(n_patches, d_model)
            nn.init.normal_(self.pair_emb.weight, std=0.02)
            nn.init.normal_(self.patch_emb.weight, std=0.02)
        elif position_embedding == "per_token":
            self.token_emb = nn.Embedding(L, d_model)
            nn.init.normal_(self.token_emb.weight, std=0.02)
        else:
            raise ValueError(f"unknown position embedding {position_embedding}")

    def positions(self) -> torch.Tensor:
        if self.position_embedding == "channel_time":
            return self.pair_emb(self.token_pair) + self.patch_emb(self.token_patch)
        return self.token_emb.weight

    def forward(self, x: torch.Tensor, masked: torch.Tensor | None = None) -> torch.Tensor:
        h = self.value_proj(x)
        if masked is not None:
            h = torch.where(masked[..., None], self.mask_token.to(h.dtype), h)
        return h + self.positions()[None]


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, attend: torch.Tensor, need_weights: bool = False):
        a, weights, values = self.attn(self.norm1(h), attend, need_weights)
        h = h + a
        h = h + self.ff(self.norm2(h))
        return h, weights, values


@dataclass
class EncoderOutput:
    cls: torch.Tensor  # [batch, d_model]
    tokens: torch.Tensor  # [batch, tokens, d_model]
    internals: dict = field(default_factory=dict)


class Encoder(nn.Module):
    def __init__(self, n_pairs: int, n_patches: int, patch: int, d_model: int, n_layers: int, n_heads: int,
                 ff_mult: int, dropout: float, attention: str, position_embedding: str) -> None:
        super().__init__()
        self.attention = attention
        self.tokenizer = Tokenizer(n_pairs, n_patches, patch, d_model, position_embedding)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, masked: torch.Tensor | None = None,
                need_weights: bool = False) -> EncoderOutput:
        B = x.shape[0]
        tokens = self.tokenizer(x, masked)
        h = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        h = self.input_dropout(h)
        attend = attend_mask(self.tokenizer.token_patch, valid, self.attention)
        weights, values = [], []
        for block in self.blocks:
            h, w, v = block(h, attend, need_weights)
            if need_weights:
                weights.append(w)
                values.append(v)
        h = self.final_norm(h)
        internals = {"attention": weights, "values": values} if need_weights else {}
        return EncoderOutput(h[:, 0], h[:, 1:], internals)
