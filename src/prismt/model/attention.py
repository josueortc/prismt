"""Self-attention with PRISMT's attend mask.

Boolean masks here mean **True = may attend**, exactly what
``torch.nn.functional.scaled_dot_product_attention`` expects; they are passed straight
through and never negated at a call site (see the knowledge-base pitfall
"sdpa-boolean-mask-inverted").

The attend rule (position 0 is the CLS token, data tokens follow in time order)::

    attend[i, j] = (i == j)                                  # every row keeps its own key
                 | (i == CLS) & valid[j]                      # CLS reads every valid token
                 | data(i, j) & valid[j] & (full | t[j] <= t[i])
    # and no data token ever attends CLS

* The diagonal is always allowed, so no row is ever empty. A fully masked row gives zeros
  on CPU but NaN on Apple GPUs, and one NaN spreads to every token in the next layer.
* CLS is read-only. Under block-causal attention, a data token that could read CLS would
  see the future through it from the second layer on.
* Block-causal masking is applied before the softmax. Multiplying a 0/1 mask after the
  softmax (as the pre-rebuild code did) still lets future tokens change every weight
  through the normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def attend_mask(token_patch: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """[batch, 1, tokens + 1, tokens + 1] boolean mask; True = may attend."""
    B, L = valid.shape
    device = valid.device
    t = token_patch.to(device)
    if mode == "block_causal":
        order = t[None, :] <= t[:, None]  # [query, key]
    elif mode == "full":
        order = torch.ones(L, L, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"unknown attention mode {mode}")
    attend = torch.zeros(B, L + 1, L + 1, dtype=torch.bool, device=device)
    attend[:, 0, 1:] = valid
    attend[:, 1:, 1:] = order[None, :, :] & valid[:, None, :]
    idx = torch.arange(L + 1, device=device)
    attend[:, idx, idx] = True
    return attend[:, None]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, attend: torch.Tensor, need_weights: bool = False):
        B, S, _ = h.shape
        q, k, v = self.qkv(h).view(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~attend, float("-inf"))
            weights = scores.softmax(-1)
            ctx = F.dropout(weights, p, self.training) @ v
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attend, dropout_p=p)
            weights = None
        out = self.out(ctx.transpose(1, 2).reshape(B, S, self.d_model))
        return out, weights, (v if need_weights else None)

    def per_head_out_proj(self) -> torch.Tensor:
        """[heads, d_head, d_model]: how each head's value vector enters the residual stream.

        ``nn.Linear`` stores its weight as [out, in], so head h owns input columns
        h*d_head:(h+1)*d_head and its block is ``W[:, cols].T``.
        """
        W = self.out.weight
        return torch.stack([W[:, h * self.d_head:(h + 1) * self.d_head].T for h in range(self.n_heads)])
