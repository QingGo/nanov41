"""CPU parity + algorithm-reference tests for ``nanov41/fused_ce_triton.py`` (Plan-A CE).

The Triton kernels themselves need CUDA (``triton_ce.available()``); this file tests on CPU:
  1. ``reference_online_ce`` (pure-torch online-softmax algorithm) against ``F.cross_entropy``
     for loss and dlogits: ignore_index, all-ignored rows, chunk boundaries, V=129280 subset;
  2. ``fused_head_ce_v2`` autograd Function CPU fallback against an equivalent torch graph
     (loss + input/weight grads), chunk-aligned and not;
  3. env gate / install contract (default path unchanged on CPU);
  4. CUDA kernel parity tests are written but skipped when triton/CUDA is unavailable.
"""
from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

import nanov41.fused_ce_triton as fct
import nanov41.train_utils as tu


def _triton_ctx() -> bool:
    try:
        return bool(fct.available())
    except Exception:  # noqa: BLE001
        return False


requires_cuda_triton = pytest.mark.skipif(
    not _triton_ctx(), reason="needs CUDA + triton (fused CE kernels)")


# =========================================================================== reference algorithm
def _torch_ce_grad(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100):
    """Reference ``dloss/dlogits`` for the *mean* CE over valid positions."""
    x = logits.detach().clone().float().requires_grad_(True)
    loss = F.cross_entropy(x, target, ignore_index=ignore_index)
    loss.backward()
    return float(loss.detach()), x.grad


@pytest.mark.parametrize("n,V", [(7, 128), (48, 1024), (4, 129280)])
def test_reference_online_ce_matches_cross_entropy(n: int, V: int):
    torch.manual_seed(0)
    logits = torch.randn(n, V)
    target = torch.randint(0, V, (n,))
    ref = fct.reference_online_ce(logits, target)
    loss_ref, grad_ref = _torch_ce_grad(logits, target)

    loss = float(ref["loss_sum"] / ref["count"])
    assert loss == pytest.approx(loss_ref, rel=1e-5, abs=1e-6)
    scaled_dlogits = ref["dlogits"] / ref["count"]
    assert torch.allclose(scaled_dlogits, grad_ref, rtol=1e-4, atol=1e-6)


def test_reference_online_ce_ignore_index_rows():
    torch.manual_seed(1)
    n, V = 33, 512
    logits = torch.randn(n, V)
    target = torch.randint(0, V, (n,))
    target[::5] = -100                      # every 5th row ignored (shift's last positions)
    ref = fct.reference_online_ce(logits, target)
    loss_ref, grad_ref = _torch_ce_grad(logits, target)
    assert int(ref["count"]) == int((target != -100).sum())
    assert float(ref["loss_sum"] / ref["count"]) == pytest.approx(loss_ref, rel=1e-5)
    assert torch.allclose(ref["dlogits"] / ref["count"], grad_ref, rtol=1e-4, atol=1e-6)
    # ignored rows have exactly zero dlogits
    ignored = (target == -100)
    assert torch.equal(ref["dlogits"][ignored], torch.zeros_like(ref["dlogits"][ignored]))


def test_reference_online_ce_all_ignored_rows():
    torch.manual_seed(2)
    n, V = 9, 64
    logits = torch.randn(n, V)
    target = torch.full((n,), -100, dtype=torch.long)
    ref = fct.reference_online_ce(logits, target)
    assert int(ref["count"]) == 0
    assert float(ref["loss_sum"]) == 0.0
    assert torch.equal(ref["dlogits"], torch.zeros_like(ref["dlogits"]))
    # ignore_index with a positive value (not the default -100) must behave the same
    ref2 = fct.reference_online_ce(logits, torch.full((n,), 0, dtype=torch.long),
                                   ignore_index=0)
    assert int(ref2["count"]) == 0
    assert float(ref2["loss_sum"]) == 0.0


def test_reference_online_ce_zero_ignore_index_is_handled():
    torch.manual_seed(3)
    n, V = 16, 256
    logits = torch.randn(n, V)
    target = torch.randint(0, V, (n,))
    target[3:6] = 0                            # token id 0 == ignore_index in this call
    ref = fct.reference_online_ce(logits, target, ignore_index=0)
    loss_ref, grad_ref = _torch_ce_grad(logits, target, ignore_index=0)
    assert float(ref["loss_sum"] / ref["count"]) == pytest.approx(loss_ref, rel=1e-5)
    assert torch.allclose(ref["dlogits"] / ref["count"], grad_ref, rtol=1e-4, atol=1e-6)


def test_shift_targets():
    ids = torch.arange(24).reshape(2, 12)
    target = fct.shift_targets(ids)
    assert target.shape == (24,)
    assert torch.equal(target.view(2, 12)[:, :-1], ids[:, 1:])
    assert torch.equal(target.view(2, 12)[:, -1], torch.full((2,), -100))


# =========================================================================== v2 autograd function
def _torch_head_ce(hidden, weight, ids, ignore_index=-100):
    """Reference graph identical in math to ``_FusedHeadCE``: bf16 chunked head + mean CE."""
    target = fct.shift_targets(ids, ignore_index)
    logits = F.linear(hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16),
                      weight.to(torch.bfloat16))
    loss = F.cross_entropy(logits.float(), target, ignore_index=ignore_index)
    return loss, target


@pytest.mark.parametrize("chunk", [None, 4, 16])
def test_fused_head_ce_v2_cpu_fallback_loss_and_grads(chunk):
    torch.manual_seed(4)
    b, s, D, V = 2, 8, 16, 96
    hidden = torch.randn(b, s, D, dtype=torch.bfloat16)
    weight = (torch.randn(V, D) * 0.02).to(torch.bfloat16)
    ids = torch.randint(0, V, (b, s))

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    loss1 = fct.fused_head_ce_v2(h1, w1, ids, chunk=chunk)
    loss1.backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    loss2, _ = _torch_head_ce(h2, w2, ids)
    loss2.backward()

    assert float(loss1.detach()) == pytest.approx(float(loss2.detach()), rel=1e-4, abs=1e-4)
    # bf16 matmul with different K-blocking (chunk-sized) has bf16 rounding-level variance: compare
    # at bf16 resolution (abs tolerance scaled by the gradient magnitude).
    scale = float(h2.grad.abs().max())
    assert torch.allclose(h1.grad, h2.grad, rtol=2e-2, atol=1e-3 * scale)
    scale = float(w2.grad.abs().max())
    assert torch.allclose(w1.grad, w2.grad, rtol=3e-2, atol=5e-3 * scale)


def test_fused_head_ce_v2_chunk_boundaries_consistent():
    torch.manual_seed(5)
    b, s, D, V = 1, 16, 8, 64
    hidden = torch.randn(b, s, D, dtype=torch.bfloat16)
    weight = (torch.randn(V, D) * 0.02).to(torch.bfloat16)
    ids = torch.randint(0, V, (b, s))
    losses = []
    for chunk in (4, 8, 16, 64):
        hidden_ = hidden.clone().requires_grad_(True)
        weight_ = weight.clone().requires_grad_(True)
        loss = fct.fused_head_ce_v2(hidden_, weight_, ids, chunk=chunk)
        loss.backward()
        losses.append((float(loss), hidden_.grad.clone(), weight_.grad.clone()))
    scale_h = float(losses[0][1].abs().max())
    scale_w = float(losses[0][2].abs().max())
    for loss, gh, gw in losses[1:]:
        assert loss == pytest.approx(losses[0][0], rel=1e-5, abs=1e-6)
        assert torch.allclose(gh, losses[0][1], rtol=2e-2, atol=1e-3 * scale_h)
        # chunk size changes the bf16 matmul K-blocking -> bf16-rounding-level variance in dW
        assert torch.allclose(gw, losses[0][2], rtol=3e-2, atol=5e-3 * scale_w)


def test_z_loss_not_implemented_is_explicit():
    hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.bfloat16)
    ids = torch.randint(0, 16, (1, 4))
    with pytest.raises(NotImplementedError):
        fct.fused_head_ce_v2(hidden, weight, ids, z_loss=1e-4)


# =========================================================================== env gate / install
def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv(fct.ENV, raising=False)
    assert fct.enabled() is False
    # CPU has no usable Triton -> install() must be a no-op and leave the training path untouched
    orig = tu.fused_head_ce
    assert fct.maybe_install() is False
    assert tu.fused_head_ce is orig


def test_env_gate_on_but_cpu_still_noop(monkeypatch):
    monkeypatch.setenv(fct.ENV, "1")
    assert fct.enabled() is True
    if fct.available():  # GPU machine: install() is expected to patch
        assert fct.install() is True
        try:
            assert fct.enabled()
        finally:
            fct.uninstall()
    else:
        orig = tu.fused_head_ce
        assert fct.maybe_install() is False
        assert tu.fused_head_ce is orig


# =========================================================================== CUDA kernel parity
@requires_cuda_triton
def test_triton_kernel_ce_fwd_matches_torch():
    torch.manual_seed(6)
    n, V = 32, 1024
    logits = (torch.randn(n, V, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, V, (n,), device="cuda")
    target[::7] = -100
    rows, lse = fct._triton_ce.ce_fwd(logits, target, -100)
    loss_ref = F.cross_entropy(logits.float(), target, reduction="sum", ignore_index=-100)
    assert float(rows) == pytest.approx(float(loss_ref), rel=1e-4, abs=1e-3)
    ref = fct.reference_online_ce(logits, target)
    assert torch.allclose(lse, ref["lse"], rtol=1e-5, atol=1e-5)


@requires_cuda_triton
def test_triton_kernel_ce_bwd_matches_torch():
    torch.manual_seed(7)
    n, V = 16, 512
    logits = (torch.randn(n, V, device="cuda")).to(torch.bfloat16)
    target = torch.randint(0, V, (n,), device="cuda")
    target[0] = -100
    rows, lse = fct._triton_ce.ce_fwd(logits, target, -100)
    d = fct._triton_ce.ce_bwd(logits, target, lse, 1.0, -100)
    assert d.dtype == torch.bfloat16 and d.shape == logits.shape
    x = logits.float().requires_grad_(True)
    F.cross_entropy(x, target, reduction="sum", ignore_index=-100).backward()
    assert torch.allclose(d.float(), x.grad, rtol=3e-2, atol=3e-2)
    assert torch.equal(d[0].float(), torch.zeros(V, device="cuda"))


@requires_cuda_triton
def test_fused_head_ce_v2_gpu_matches_torch():
    torch.manual_seed(8)
    b, s, D, V = 1, 32, 16, 256
    hidden = torch.randn(b, s, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(V, D, device="cuda") * 0.02).to(torch.bfloat16).requires_grad_(True)
    ids = torch.randint(0, V, (b, s), device="cuda")
    loss = fct.fused_head_ce_v2(hidden, weight, ids, chunk=32)
    loss.backward()
    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = weight.detach().clone().requires_grad_(True)
    loss2, _ = _torch_head_ce(h2, w2, ids)
    loss2.backward()
    assert float(loss) == pytest.approx(float(loss2), rel=1e-4, abs=1e-3)
    assert math.isfinite(float(loss))
