# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""R65: `NANOV41_MOE_FUSED_DY` — NOPAD MoE bwd without materialising `dy_pad` (CUDA only).

The fused-dy path replaces

    dy_pad = zeros(P, D); dy_pad[row_of_flat[f]] = dout[f // k]   (memset + tail_bwd)
    dW2  = agg_dw(act, dy_pad, ...)
    dgate, dup, dact = swiglu_bwd_tb(dy_pad, ...)

with two kernels that read `dout[flat_of_row[p] // k]` directly. Both read exactly the same
values in the same order, so **outputs and gradients must be bitwise identical** to the old path
(this test asserts `torch.equal`, not a tolerance).
"""
from __future__ import annotations

import pytest
import torch

import nanov41.triton_grouped_moe as tgm

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and tgm.HAS_TRITON),
    reason="needs CUDA + Triton for the grouped MoE kernels",
)


def _run(fused_dy: bool, seed: int = 0, dws_inline: bool = False, fused_dx: bool = False):
    E, dim, inter, T, k = 16, 384, 384, 1024, 2
    dev = "cuda"
    torch.manual_seed(seed)
    idx = torch.randint(0, E, (T * k,), device=dev)
    ws = (torch.rand(T * k, device=dev) * 0.5 + 0.5).to(torch.bfloat16)
    x = (torch.randn(T, dim, device=dev) * 0.05).to(torch.bfloat16)
    w1 = (torch.randn(E, dim, inter, device=dev) * 0.05).to(torch.bfloat16)
    w3 = (torch.randn(E, dim, inter, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, inter, dim, device=dev) * 0.05).to(torch.bfloat16)
    g = torch.randn(T, dim, device=dev, dtype=torch.float32) * 0.001

    xs = [x.clone().requires_grad_(True), ws.clone().requires_grad_(True),
          w1.clone().requires_grad_(True), w3.clone().requires_grad_(True),
          w2.clone().requires_grad_(True)]
    old_nopad, old_tail, old_dws, old_dx = tgm.NOPAD, tgm.FUSED_TAIL, tgm.DWS_INLINE, tgm.FUSED_DX
    tgm.NOPAD, tgm.FUSED_TAIL = True, True
    tgm.FUSED_DY, tgm.DWS_INLINE, tgm.FUSED_DX = fused_dy, dws_inline, fused_dx
    try:
        # layout reads the module flags at construction time, so build it *after* the toggles
        layout = tgm.MoELayout(idx, E, k)
        out = tgm.grouped_moe_experts_v2(xs[0], xs[2], xs[4], xs[3], layout, xs[1], 10.0)
        (out.float() * g).sum().backward()
    finally:
        tgm.NOPAD, tgm.FUSED_TAIL, tgm.DWS_INLINE = old_nopad, old_tail, old_dws
        tgm.FUSED_DY, tgm.FUSED_DX = False, old_dx
    return out.detach(), [t.grad.detach().clone() for t in xs]


def test_fused_dy_is_bitwise_identical_to_dy_pad_path():
    o_ref, g_ref = _run(fused_dy=False)
    o_new, g_new = _run(fused_dy=True)
    assert torch.equal(o_ref, o_new), "forward changed"
    names = ["dx", "dws", "dW1", "dW3", "dW2"]
    for name, a, b in zip(names, g_ref, g_new, strict=True):
        assert a is not None and b is not None, name
        assert torch.equal(a, b), f"{name} differs (rel {float((a - b).norm() / (a.norm() + 1e-9))})"


def test_fused_dy_runs_with_k1_routing():
    o_ref, g_ref = _run(fused_dy=False, seed=3)
    o_new, g_new = _run(fused_dy=True, seed=3)
    assert torch.equal(o_ref, o_new)
    for a, b in zip(g_ref, g_new, strict=True):
        assert torch.equal(a, b)


def test_dws_inline_matches_reference_dws():
    """Inline-dws changes two things in the routing-weight gradient: the fp32 summation order of
    the per-tile partial sums, and the fact that it multiplies the *pre-bf16-rounding* dact
    (`acc`) whereas the reference path reads `DACT` after a bf16 round-trip. The second effect
    is bf16 quantisation noise (~4e-3 relative), so the gate is 1e-2; the other gradients must
    stay bitwise identical."""
    o_ref, g_ref = _run(fused_dy=False)
    o_new, g_new = _run(fused_dy=True, dws_inline=True)
    assert torch.equal(o_ref, o_new), "forward changed"
    names = ["dx", "dws", "dW1", "dW3", "dW2"]
    for name, a, b in zip(names, g_ref, g_new, strict=True):
        assert a is not None and b is not None, name
        rel = float((a - b).norm() / (a.norm() + 1e-30))
        if name == "dws":
            assert rel < 1e-2, f"dws rel {rel}"
        else:
            assert torch.equal(a, b), f"{name} should be bitwise identical (rel {rel})"


def test_fused_dx_single_kernel_close_to_two_gemm_path():
    """`agg_forward2_t` accumulates both dx GEMMs in one fp32 accumulator (bf16-rounded once)
    instead of two bf16 GEMMs plus an add; only `dx` may differ, and only by rounding."""
    o_ref, g_ref = _run(fused_dy=True)
    o_new, g_new = _run(fused_dy=True, fused_dx=True)
    assert torch.equal(o_ref, o_new), "forward changed"
    names = ["dx", "dws", "dW1", "dW3", "dW2"]
    for name, a, b in zip(names, g_ref, g_new, strict=True):
        rel = float((a - b).norm() / (a.norm() + 1e-30))
        if name == "dx":
            assert rel < 5e-3, f"dx rel {rel}"
        else:
            assert torch.equal(a, b), f"{name} should be bitwise identical (rel {rel})"
