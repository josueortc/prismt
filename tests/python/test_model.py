"""Invariants of tokens, attention, masking, losses and checkpoints (knowledge-base pitfalls)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prismt.data.tokens import TokenGrid
from prismt.errors import CheckpointError, ConfigError
from prismt.masking import MaskSpec, build_mask
from prismt.model.attention import attend_mask
from prismt.model.checkpoint import build_from_checkpoint, load_checkpoint, load_pretrained_encoder, save_checkpoint
from prismt.model.prismt_model import ModelSpec, PrismtModel
from prismt.tasks.mae import masked_mse

GRID = TokenGrid.make([(r, m) for m in range(2) for r in range(3)], 4, 1)  # 6 pairs x 4 patches = 24 tokens


def model(task="classify", attention="block_causal", **kw) -> PrismtModel:
    spec = ModelSpec(task, GRID.n_pairs, GRID.n_patches, GRID.patch, 16, 2, 2, 2, 0.0, attention, "channel_time",
                     2 if task == "classify" else 0, **kw)
    torch.manual_seed(0)
    return PrismtModel(spec).eval()


def batch(b=3):
    g = torch.Generator().manual_seed(1)
    return torch.randn(b, GRID.n_tokens, 1, generator=g), torch.ones(b, GRID.n_tokens, dtype=torch.bool)


# --- tokens ------------------------------------------------------------------------------

def test_tokens_are_time_major_and_round_trip():
    X = np.random.default_rng(0).standard_normal((2, 3, 4, 2)).astype(np.float32)
    X[0, 1, 2, 0] = np.nan
    V, valid = GRID.to_tokens(X)
    assert V.shape == (2, 24, 1)
    assert GRID.token_patch.tolist()[:7] == [0] * 6 + [1]
    np.testing.assert_array_equal(V[1, 6 + 4, 0], X[1, 1, 1, 1])  # patch 1, pair (1, 1)
    assert not valid[0, 2 * 6 + 1]
    np.testing.assert_array_equal(GRID.from_tokens(V, 3, 2), X)


def test_patch_must_divide_bins():
    with pytest.raises(ConfigError) as err:
        TokenGrid.make([(0, 0)], 10, 3)
    assert "1, 2, 5, 10" in err.value.hint


# --- attention ---------------------------------------------------------------------------

def test_sdpa_boolean_mask_means_attend():
    q = k = torch.randn(1, 1, 2, 4)
    v = torch.tensor([[[[1.0] * 4, [9.0] * 4]]])
    only_first = torch.tensor([[[[True, False], [True, False]]]])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=only_first)
    assert torch.allclose(out, torch.ones_like(out))


def test_attend_mask_structure():
    _, valid = batch(2)
    valid[1, 5] = False
    m = attend_mask(torch.as_tensor(GRID.token_patch), valid, "block_causal")[:, 0]
    assert m.any(-1).all(), "every row keeps a key"
    assert not m[:, 1:, 0].any(), "no data token reads CLS"
    assert torch.equal(m[:, 0, 1:], valid), "CLS reads every valid token"
    t = torch.as_tensor(GRID.token_patch)
    future = t[None, :] > t[:, None]
    assert not (m[0, 1:, 1:] & future).any(), "no future keys"
    readers = torch.nonzero(m[1, :, 6]).flatten().tolist()
    assert readers == [6], "an invalid key is read only by its own row"


def test_explicit_attention_equals_sdpa():
    x, v = batch()
    mdl = model()
    assert torch.allclose(mdl(x, v).logits, mdl(x, v, need_weights=True).logits, atol=1e-5)


def test_per_head_out_proj_reconstructs_the_output_layer():
    attn = model().encoder.blocks[0].attn
    ctx = torch.randn(5, attn.d_model)
    heads = attn.per_head_out_proj()
    parts = sum(ctx[:, h * attn.d_head:(h + 1) * attn.d_head] @ heads[h] for h in range(attn.n_heads))
    assert torch.allclose(parts + attn.out.bias, attn.out(ctx), atol=1e-5)


# --- perturbation tests ------------------------------------------------------------------

def test_future_token_does_not_change_earlier_outputs_under_block_causal():
    mdl = model("mae")
    x, v = batch(1)
    x2 = x.clone()
    x2[0, 3 * 6 + 2] += 5.0  # a token in the last patch
    a, b = mdl.encoder(x, v).tokens, mdl.encoder(x2, v).tokens
    early = torch.as_tensor(GRID.token_patch) < 3
    assert torch.equal(a[0, early], b[0, early])
    assert not torch.allclose(a[0, ~early], b[0, ~early])
    full = model("mae", attention="full")
    assert not torch.allclose(full.encoder(x, v).tokens[0, early], full.encoder(x2, v).tokens[0, early])


def test_every_visible_token_can_move_the_cls_output():
    mdl = model()
    x, v = batch(1)
    base = mdl(x, v).logits
    for i in (0, 11, 23):
        x2 = x.clone()
        x2[0, i] += 3.0
        assert not torch.allclose(mdl(x2, v).logits, base)


def test_invalid_and_masked_values_change_nothing():
    mdl = model("mae")
    x, v = batch(1)
    v[0, 4] = False
    masked = torch.zeros_like(v)
    masked[0, 7] = True
    a = mdl(x, v, masked).reconstruction
    x2 = x.clone()
    x2[0, 4] = 100.0  # invalid token's (filled) value
    x2[0, 7] = -100.0  # hidden token's value
    b = mdl(x2, v, masked).reconstruction
    others = torch.ones(24, dtype=torch.bool)
    others[4] = False  # an invalid token's own output is never used or scored
    assert torch.equal(a[0, others], b[0, others])
    assert torch.equal(mdl.encoder(x, v, masked).cls, mdl.encoder(x2, v, masked).cls)


def test_nan_inputs_and_empty_rows_stay_finite_on_every_device():
    from prismt.env import _mps_works

    devices = ["cpu"] + (["mps"] if _mps_works() else [])
    for dev in devices:
        mdl = model("mae").to(dev)
        x, v = batch(2)
        v[0] = False  # a trial with no valid token at all
        v[1, :6] = False  # the whole first time patch missing
        out = mdl(x.to(dev), v.to(dev)).reconstruction
        assert torch.isfinite(out).all(), dev


# --- masking and loss --------------------------------------------------------------------

@pytest.mark.parametrize("spec", [MaskSpec("random", 0.9), MaskSpec("channel", 0.3),
                                  MaskSpec("forecast", context_fraction=0.5), MaskSpec("modality")])
def test_masks_never_touch_invalid_tokens_and_leave_something_visible(spec):
    _, v = batch(4)
    v[2, ::3] = False
    m = build_mask(spec, v, GRID, torch.Generator().manual_seed(0), np.array([mm for _, mm in GRID.pairs]))
    assert not (m & ~v).any()
    assert ((v & ~m).sum(1) >= 1).all()
    assert m.any()


def test_random_mask_rate_and_determinism():
    _, v = batch(4)
    a = build_mask(MaskSpec("random", 0.75), v, GRID, torch.Generator().manual_seed(3))
    b = build_mask(MaskSpec("random", 0.75), v, GRID, torch.Generator().manual_seed(3))
    assert torch.equal(a, b) and (a.sum(1) == 18).all()


def test_loss_only_sees_scored_tokens():
    target = torch.randn(2, 24, 1)
    recon = torch.randn(2, 24, 1, requires_grad=True)
    scored = torch.zeros(2, 24, dtype=torch.bool)
    scored[:, :5] = True
    loss = masked_mse(recon, target, scored)
    assert torch.isclose(loss, ((recon[:, :5] - target[:, :5]) ** 2).mean())
    loss.backward()
    assert (recon.grad[:, 5:] == 0).all()
    assert masked_mse(recon, target, torch.zeros_like(scored)) == 0


# --- checkpoints -------------------------------------------------------------------------

def test_checkpoint_round_trip_with_weights_only(tmp_path):
    mdl = model()
    p = save_checkpoint(tmp_path / "model.pt", mdl, extra={"class_names": ["a", "b"]})
    ck = load_checkpoint(p)
    x, v = batch()
    assert torch.equal(build_from_checkpoint(ck).eval()(x, v).logits, mdl(x, v).logits)


def test_pretrained_encoder_loads_completely_or_not_at_all(tmp_path):
    mae = model("mae")
    ck = load_checkpoint(save_checkpoint(tmp_path / "m.pt", mae))
    clf = model()
    report = load_pretrained_encoder(clf, ck)
    assert report.coverage == 1.0
    for (k, a), b in zip(mae.encoder.state_dict().items(), clf.encoder.state_dict().values()):
        assert torch.equal(a, b), k
    other = PrismtModel(ModelSpec("classify", GRID.n_pairs, GRID.n_patches, 1, 32, 2, 2, 2, 0.0, "block_causal",
                                  "channel_time", 2))
    with pytest.raises(CheckpointError, match="d_model"):
        load_pretrained_encoder(other, ck)
    ck["state_dict"].pop(next(k for k in ck["state_dict"] if k.startswith("encoder.blocks")))
    with pytest.raises(CheckpointError) as err:
        load_pretrained_encoder(model(), ck)
    assert err.value.code == "E_CKPT_PARTIAL"
