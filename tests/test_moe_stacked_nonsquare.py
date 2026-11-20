"""Regression: ``moe_stacked.patch_model`` on a **non-square** MoE (dim != moe_inter_dim).

The aligned V4.1 config has dim=512 / moe_inter=256 (official ratio 0.45), unlike the older
D8/D12 configs where dim == moe_inter_dim.  ``patch_model`` used to read the two expert-weight
dimensions in the wrong order, which only crashes when they differ.  This test pins the correct
stacked layout [E, dim, inter] and checks forward parity vs the reference MoE (CPU path).
"""

from __future__ import annotations

import pytest
import torch

from nanov41.configs import TINY
from nanov41.model import NanoV41, init_random_weights
from nanov41.patched_env import import_patched


@pytest.fixture(autouse=True)
def _restore_head():
    """The identity-head trick below is a global module patch: restore it for other test files."""
    M = import_patched()
    prev = M.ParallelHead.forward
    yield
    M.ParallelHead.forward = prev


def _build(dim: int = 64, inter: int = 128):
    M = import_patched()
    cfg = dict(TINY, dim=dim, moe_inter_dim=inter, n_heads=4, head_dim=32, q_lora_rank=32,
               o_lora_rank=32, window_size=16, max_batch_size=4, max_seq_len=64,
               dtype="float32")
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, std=0.02, seed=7)
    # call the backbone for hidden states instead of vocab logits (cheap on CPU)
    M.ParallelHead.forward = lambda self, x, full_logits=False: x
    return M, model


def test_patch_model_non_square_layout_and_forward_parity() -> None:
    M, model = _build()
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    with torch.no_grad():
        out_ref = model(ids)
    n = __import__("nanov41.moe_stacked", fromlist=["patch_model"]).patch_model(model.inner, M)
    assert n == TINY["n_layers"], f"patched {n} MoE, expected {TINY['n_layers']}"
    for mod in model.inner.modules():
        if getattr(mod, "experts_stacked", None) is not None:
            assert tuple(mod.experts_stacked.w1.shape) == (mod.n_local_experts, 64, 128)
            assert tuple(mod.experts_stacked.w2.shape) == (mod.n_local_experts, 128, 64)
            assert tuple(mod.experts_stacked.w3.shape) == (mod.n_local_experts, 64, 128)
    with torch.no_grad():
        out = model(ids)
    assert torch.allclose(out, out_ref, atol=2e-5, rtol=2e-5), \
        f"stacked forward max|diff| = {(out - out_ref).abs().max().item():.3e}"

    # idempotent
    moe_stacked = __import__("nanov41.moe_stacked", fromlist=["patch_model"])
    assert moe_stacked.patch_model(model.inner, M) == 0


def test_patch_model_state_dict_roundtrip_non_square() -> None:
    M, model = _build()
    moe_stacked = __import__("nanov41.moe_stacked", fromlist=["patch_model"])
    moe_stacked.patch_model(model.inner, M)
    sd = model.inner.state_dict()
    # official per-expert key layout survives (used by the parity harness)
    assert "layers.0.ffn.experts.0.w1.weight" in sd
    assert "layers.0.ffn.experts_stacked.w1" not in sd
    assert tuple(sd["layers.0.ffn.experts.0.w1.weight"].shape) == (128, 64)  # [inter, dim]
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    with torch.no_grad():
        out = model(ids)
    M2, model2 = _build()
    moe_stacked.patch_model(model2.inner, M2)
    res = model2.inner.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    with torch.no_grad():
        out2 = model2(ids)
    assert torch.allclose(out, out2, atol=1e-6, rtol=1e-6)
