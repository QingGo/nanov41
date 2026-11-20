# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""R66: `NANOV41_MOE_LAYOUT=fast` — launch-minimised MoE layout (CUDA only).

The fast layout replaces ~30 small aten ops per MoE layer (argsort aside: histogram, padded
offsets, `cum_excl`, `tile_e`, `row_of_flat`, `flat_of_row`, `ws_pad`) with four Triton kernels.
It must be **bitwise identical** to the `sort` layout: same stable argsort, same integer
arithmetic for `pad_pos`, same scatter targets. This test compares every layout tensor and then
the full MoE forward/backward (output + five gradients) with `torch.equal`.
"""
from __future__ import annotations

import pytest
import torch

import nanov41.triton_grouped_moe as tgm

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and tgm.HAS_TRITON),
    reason="needs CUDA + Triton for the grouped MoE kernels",
)


def _tensors(seed: int = 0):
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
    return E, dim, inter, k, idx, ws, x, w1, w3, w2, g


def _run(fast: bool, seed: int = 0):
    E, dim, inter, k, idx, ws, x, w1, w3, w2, g = _tensors(seed)
    old = (tgm._LAYOUT, tgm.FAST_LAYOUT, tgm.NOPAD, tgm.FUSED_TAIL, tgm.FUSED_DY,
           tgm.DWS_INLINE, tgm.FUSED_DX)
    tgm._LAYOUT = "fast" if fast else "sort"
    tgm.FAST_LAYOUT = fast
    tgm.NOPAD = tgm.FUSED_TAIL = True
    tgm.FUSED_DY = tgm.DWS_INLINE = tgm.FUSED_DX = False
    try:
        layout = tgm.MoELayout(idx, E, k)
        if fast:
            ws_pad = layout.map_flat(ws)
        else:
            ws_pad = layout.pad(ws[layout.order][:, None], 1).squeeze(1)
        xs = [x.clone().requires_grad_(True), ws.clone().requires_grad_(True),
              w1.clone().requires_grad_(True), w3.clone().requires_grad_(True),
              w2.clone().requires_grad_(True)]
        out = tgm.grouped_moe_experts_v2(xs[0], xs[2], xs[4], xs[3], layout, xs[1], 10.0)
        (out.float() * g).sum().backward()
        gt = {
            "order": layout.order,
            "seg_start": layout.seg_start,
            "seg_end": layout.seg_end,
            "tile_e": layout.tile_e,
            "count_i32": layout.count_i32,
            "row_of_flat": layout.row_of_flat,
            "flat_of_row": layout.flat_of_row,
            "ws_pad": ws_pad,
        }
    finally:
        (tgm._LAYOUT, tgm.FAST_LAYOUT, tgm.NOPAD, tgm.FUSED_TAIL, tgm.FUSED_DY,
         tgm.DWS_INLINE, tgm.FUSED_DX) = old
    return out.detach(), [t.grad.detach().clone() for t in xs], gt


def test_fast_layout_is_bitwise_identical_to_sort():
    o_ref, g_ref, gt_ref = _run(fast=False)
    o_new, g_new, gt_new = _run(fast=True)
    for name, a in gt_ref.items():
        assert torch.equal(a, gt_new[name]), f"layout tensor {name} differs"
    assert torch.equal(o_ref, o_new), "forward differs"
    names = ["dx", "dws", "dW1", "dW3", "dW2"]
    for name, a, b in zip(names, g_ref, g_new, strict=True):
        assert torch.equal(a, b), f"{name} differs"
