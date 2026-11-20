"""CPU tests for ``nanov41/head_fwdgrad.py`` (forward-precomputed head dX/dW).

No GPU needed: the bf16 variant uses plain torch ops, so the math (loss, dX, dW, ignore rows,
chunk accumulation, scalar-grad scaling, env gating) is exercised on CPU against the existing
``train_utils.fused_head_ce`` reference.  The fp8 variant is GPU-only and is covered by
``tools/r71_head_fwdgrad_validate.py``.
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch
import torch.nn.functional as F

from nanov41 import head_fwdgrad as hfg
from nanov41 import train_utils as tu

V, D, B, S = 96, 8, 2, 8
CHUNK = 8


def _make(seed=0, weight_dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, S, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    weight = (torch.randn(V, D, generator=g, dtype=torch.float32) * 0.05).to(weight_dtype)
    ids = torch.randint(0, V, (B, S), generator=g)
    return hidden, weight, ids


def _run(fn, hidden, weight, ids, **kw):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = fn(h, w, ids, **kw)
    loss.backward()
    return loss.detach(), h.grad.detach(), w.grad.detach()


def _rel(a, b, eps=1e-6):
    a, b = a.float(), b.float()
    return float(((a - b).abs() / b.abs().clamp_min(eps)).max())


def _relmax(a, b):
    """max abs diff normalized by the reference absmax (robust to near-zero bf16 entries)."""
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("NANOV41_HEAD_FWDGRAD", raising=False)
    monkeypatch.delenv("NANOV41_HEAD_FWDGRAD_DW", raising=False)
    monkeypatch.delenv("NANOV41_HEAD_FP8", raising=False)
    yield


def test_env_gate_default_off(monkeypatch):
    assert not hfg.enabled()
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "1")
    assert hfg.enabled()
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "off")
    assert not hfg.enabled()


def test_math_matches_reference_bf16_weight(monkeypatch):
    hidden, weight, ids = _make()
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "0")
    loss0, gx0, gw0 = _run(tu.fused_head_ce, hidden, weight, ids, chunk=CHUNK)
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "1")
    loss1, gx1, gw1 = _run(tu.fused_head_ce, hidden, weight, ids, chunk=CHUNK)
    assert torch.isfinite(loss1)
    assert _rel(loss1, loss0) < 5e-3
    assert _relmax(gx1, gx0) < 5e-2
    assert _relmax(gw1, gw0) < 5e-2


def test_weight_fp32_master_keeps_dtype():
    hidden, weight, ids = _make(weight_dtype=torch.float32)
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = hfg.fused_head_ce_fwdgrad(h, w, ids, chunk=CHUNK)
    loss.backward()
    assert w.grad.dtype == torch.float32
    assert h.grad.dtype == hidden.dtype
    assert torch.isfinite(w.grad).all()


def test_chunk_accumulation_equivalence():
    hidden, weight, ids = _make()
    _, gx4, gw4 = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=4)
    _, gx8, gw8 = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=8)
    _, gx16, gw16 = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=16)
    for gx, gw in ((gx8, gw8), (gx16, gw16)):
        assert _relmax(gx, gx4) < 5e-2
        assert _relmax(gw, gw4) < 5e-2


def test_scalar_grad_linearity():
    hidden, weight, ids = _make()
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = hfg.fused_head_ce_fwdgrad(h, w, ids, chunk=CHUNK)
    scaled = loss * 2.0
    gx2, gw2 = torch.autograd.grad(scaled, (h, w))
    # independent graph for the 1x reference (the graph is freed after autograd.grad)
    h1 = hidden.detach().clone().requires_grad_(True)
    w1 = weight.detach().clone().requires_grad_(True)
    loss1 = hfg.fused_head_ce_fwdgrad(h1, w1, ids, chunk=CHUNK)
    gx1, gw1 = torch.autograd.grad(loss1, (h1, w1))
    assert _relmax(gx2, gx1 * 2) < 1e-3
    assert _relmax(gw2, gw1 * 2) < 1e-3


def test_non_scalar_grad_raises():
    with pytest.raises(RuntimeError, match="scalar grad_out"):
        hfg._scalar_scale(torch.ones(2))
    # scalar / 1-element tensors are fine
    assert float(hfg._scalar_scale(torch.tensor(2.5))) == 2.5
    assert float(hfg._scalar_scale(torch.tensor([2.5]))) == 2.5


def test_ignore_index_rows_get_zero_input_grad():
    """Shift makes the last token of every sequence ignore_index -> its dX row must be 0."""
    hidden, weight, ids = _make()
    _, gx, _ = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=CHUNK)
    gx_rows = gx.reshape(B * S, D)
    last_rows = gx_rows.view(B, S, D)[:, S - 1, :]
    assert torch.count_nonzero(last_rows) == 0
    # and a normal row is non-zero
    assert torch.count_nonzero(gx_rows.view(B, S, D)[:, 0, :]) > 0


def test_z_loss_supported():
    """z-loss is implemented in the fast path (used by the P1 8B training run)."""
    hidden, weight, ids = _make()
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = hfg.fused_head_ce_fwdgrad(h, w, ids, chunk=CHUNK, z_loss=1e-4)
    assert torch.isfinite(loss)
    gx, gw = torch.autograd.grad(loss, (h, w))
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
    # z-loss must actually change the loss value relative to plain CE
    loss0 = hfg.fused_head_ce_fwdgrad(hidden.detach(), weight.detach(), ids, chunk=CHUNK)
    assert float(loss) > float(loss0)


def test_dw_bf16_option_is_close(monkeypatch):
    hidden, weight, ids = _make()
    _, gx32, gw32 = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=CHUNK)
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD_DW", "bf16")
    assert hfg._dw_bf16()
    _, gx16, gw16 = _run(hfg.fused_head_ce_fwdgrad, hidden, weight, ids, chunk=CHUNK)
    assert _relmax(gx16, gx32) < 1e-2
    # dW is stored in bf16 then cast up: expect bf16 rounding-level differences
    assert _relmax(gw16, gw32) < 5e-2


def test_dispatch_env_off_uses_reference(monkeypatch):
    hidden, weight, ids = _make()
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "0")
    loss0, gx0, gw0 = _run(tu.fused_head_ce, hidden, weight, ids, chunk=CHUNK)
    assert torch.isfinite(loss0) and gx0 is not None
    # head_fwdgrad itself still works when called directly (unit-test entry point)
    monkeypatch.setenv("NANOV41_HEAD_FWDGRAD", "1")
    loss1, _, _ = _run(tu.fused_head_ce, hidden, weight, ids, chunk=CHUNK)
    assert torch.isfinite(loss1)


def test_fp8_env_on_cpu_falls_back_to_bf16(monkeypatch):
    hidden, weight, ids = _make()
    monkeypatch.setenv("NANOV41_HEAD_FP8", "v2")   # fp8 backend unavailable on CPU
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = hfg.fused_head_ce_fwdgrad(h, w, ids, chunk=CHUNK)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(h.grad).all()
