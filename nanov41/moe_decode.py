# SPDX-License-Identifier: MIT
"""Decode-oriented MoE forward for small M (native inference path).

The training MoE kernels are tuned for large prefill batches (M thousands, grouped GEMM with
per-expert padding).  At decode M is 1--8 (top-2 => 2--16 (token,slot) pairs), where the grouped
layout overhead dominates.  This module uses batched small GEMMs (``bmm``) instead:

    x_rows [N,1,D] @ W1[e] [N,D,I] -> gate [N,1,I]
    x_rows @ W3[e]                 -> up
    act = silu(clamp(gate)) * clamp(up) * weight
    act [N,1,I] @ W2[e] [N,I,D]    -> down [N,1,D]
    out.index_add_(token_row, down)

3 batched GEMM launches + 3 gathers per layer, versus one launch per expert per matrix for the
reference loop.  Weights are stacked once (`enable_moe_decode`) so the forward does only tensor
indexing.  This is a PyTorch-kernel path first; a dedicated Triton small-M kernel can replace it
later without changing the interface.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


def _stack_experts(moe, device, dtype):
    e0, e1 = moe.experts_start_idx, moe.experts_end_idx
    w1 = torch.stack([moe.experts[i].w1.weight.detach().t().to(device=device, dtype=dtype)
                      for i in range(e0, e1)])
    w2 = torch.stack([moe.experts[i].w2.weight.detach().t().to(device=device, dtype=dtype)
                      for i in range(e0, e1)])
    w3 = torch.stack([moe.experts[i].w3.weight.detach().t().to(device=device, dtype=dtype)
                      for i in range(e0, e1)])
    return w1, w2, w3


def moe_decode_small_m(x2: torch.Tensor, gate, shared_experts, w1, w2, w3, *,
                       swiglu_limit: float = 0.0, image_mask=None) -> torch.Tensor:
    """x2 [N,D] -> y [N,D] using top-k routed experts + shared expert (batched small GEMMs)."""
    n, dim = x2.shape
    weights, indices = gate(x2, None if image_mask is None else image_mask.flatten())
    k = indices.shape[1]
    flat_e = indices.reshape(-1)
    rows = torch.arange(n, device=x2.device).repeat_interleave(k)
    wts = weights.reshape(-1)
    wdtype = w1.dtype
    xr = x2[rows].to(wdtype)                                    # [N*k, D]
    gw1 = w1.index_select(0, flat_e)                            # [P, D, I]
    gw3 = w3.index_select(0, flat_e)
    gate_l = torch.bmm(xr.unsqueeze(1), gw1).squeeze(1)         # [P, I]
    up = torch.bmm(xr.unsqueeze(1), gw3).squeeze(1)
    lim = float(swiglu_limit or 0.0)
    if lim > 0:
        up = up.clamp(-lim, lim)
        gate_l = gate_l.clamp(max=lim)
    act = F.silu(gate_l) * up
    act = (act * wts[:, None].to(wdtype)).to(wdtype)
    gw2 = w2.index_select(0, flat_e)                            # [P, I, D]
    down = torch.bmm(act.unsqueeze(1), gw2).squeeze(1)          # [P, D]
    out = x2.new_zeros((n, dim))
    out.index_add_(0, rows, down.to(out.dtype))
    out = out + shared_experts(x2)
    return out


def enable_moe_decode(model, max_tokens: int = 8, dtype=torch.bfloat16) -> int:
    """Patch every MoE module with a small-M batched-GEMM forward (fallback to original above max)."""
    inner = getattr(model, "inner", model)
    n = 0
    for layer in getattr(inner, "layers", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is None or not hasattr(ffn, "experts"):
            continue
        dev = ffn.gate.weight.device
        w1, w2, w3 = _stack_experts(ffn, dev, dtype)
        ffn._decode_w1, ffn._decode_w2, ffn._decode_w3 = w1, w2, w3
        first = ffn.experts[ffn.experts_start_idx]
        ffn._decode_swiglu_limit = float(getattr(first, 'swiglu_limit', 0.0) or 0.0)
        ffn._decode_orig_forward = ffn.forward

        def _forward(self, x, image_mask=None):
            # token count, not batch size: prefill arrives as [1, seq, dim], so a batch-only
            # check would route a whole prefill chunk through the small-M path (huge gathers).
            n_tokens = x.numel() // self.dim
            if n_tokens > max_tokens:
                return self._decode_orig_forward(x, image_mask)
            return moe_decode_small_m(
                x.view(-1, self.dim), self.gate, self.shared_experts,
                self._decode_w1, self._decode_w2, self._decode_w3,
                swiglu_limit=self._decode_swiglu_limit, image_mask=image_mask,
            ).view_as(x)

        ffn.forward = _forward.__get__(ffn, type(ffn))
        n += 1
    return n


__all__ = ["moe_decode_small_m", "enable_moe_decode", "HAS_TRITON"]
