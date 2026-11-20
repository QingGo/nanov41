# SPDX-License-Identifier: MIT
"""Nano-adapted Engram gate kernels, vendored from DeepSeek TileKernels (MIT).

Upstream: `TileKernels/tile_kernels/engram/` (Copyright (c) 2026 DeepSeek, MIT License).
Local changes: the forward reduction tile gains a 128-element variant (threads=32 x vec=4) and the
backward kernel uses 128 threads (1 warp/head) when `hidden_size % 256 != 0`, so the nano dims
384/640 run natively without the 4x power-of-two padding needed by the unmodified upstream kernels.
The math is unchanged; only tile/thread constants and the corresponding divisibility assumptions.
"""
from __future__ import annotations

import torch

from ._config import get_num_sms
from .gate_bwd_cuda import get_engram_gate_bwd_kernel_cuda
from .gate_fwd_cuda import get_engram_gate_fwd_kernel_cuda

__all__ = ["engram_gate_fwd", "engram_gate_bwd"]


def engram_gate_fwd(hidden_states, kv, weight_fused, eps: float, clamp_value: float,
                    save_for_backward: bool = True, image_token_mask=None, inplace: bool = False):
    """Fused Engram gate forward (nano tile variant)."""
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size ** -0.5
    kernel = get_engram_gate_fwd_kernel_cuda(
        hidden_size, float(eps), scalar, get_num_sms(), float(clamp_value), hc_mult,
        save_for_backward, image_token_mask is not None, inplace, use_pdl=False)
    output = hidden_states if inplace else torch.empty_like(hidden_states)
    if save_for_backward:
        dot = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_x = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_k = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
    else:
        dot = gate_score = rstd_x = rstd_k = None
    if num_tokens > 0:
        kernel(hidden_states, kv, weight_fused, image_token_mask,
               None if inplace else output, dot, gate_score, rstd_x, rstd_k)
    return output, dot, gate_score, rstd_x, rstd_k


def engram_gate_bwd(grad_out, hidden_states, kv, weight_fused, dot, gate_score,
                    rstd_x, rstd_k, clamp_value: float, image_token_mask=None):
    """Fused Engram gate backward (nano tile variant); grad_w_partial needs a first-dim reduction."""
    _, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size ** -0.5
    num_partials = get_num_sms()
    kernel = get_engram_gate_bwd_kernel_cuda(
        hidden_size, scalar, num_partials, float(clamp_value), hc_mult,
        has_image_token_mask=image_token_mask is not None, use_pdl=False)
    grad_hidden_states = torch.empty_like(hidden_states)
    grad_kv = torch.empty_like(kv)
    grad_w_partial = torch.empty((num_partials, hc_mult, hidden_size),
                                 dtype=torch.float32, device=hidden_states.device)
    kernel(grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k,
           image_token_mask, grad_hidden_states, grad_kv, grad_w_partial)
    return grad_hidden_states, grad_kv, grad_w_partial
