# SPDX-License-Identifier: MIT
"""Fused Engram gate forward/backward (TileLang).

Two kernel sources are supported, selected per hidden size:

* ``nanov41.engram_kernels`` -- vendored from DeepSeek TileKernels (MIT) and adapted with a
  128-element reduction tile (and 1 warp/head in the backward), so the nano dims **384/640 run
  natively without padding**.  Used for d12/d20 (and as a fallback when TileKernels is absent).
* official ``TileKernels/tile_kernels/engram`` (``TILE_KERNELS_PATH``) -- tuned for hidden sizes
  divisible by 256; used natively for d16 (dim 512).  For other dims it needs zero-padding.

For the padded official path the hidden axis is grown by factors of 4 so ``c = sqrt(P/D) = 2^k`` is
exact in bf16 (``x'=2x``, ``key'=2key``, ``w'=w/2``); the unpadded reference is recovered with the
chain rule ``grad_x = c*grad_x'-(c-1)*grad_out``, ``grad_key = c*grad_key'``,
``grad_value = grad_value'``, ``grad_w = (1/c)*grad_w'``.  The native nano path has ``c=1`` and the
mapping is the identity.
"""
from __future__ import annotations

import math
import os
import sys

import torch

__all__ = ["tl_available", "tl_engram_forward", "EngramGateTL", "pad_hidden"]

_KERNELS = None
_TRIED = False


def _load_kernels():
    global _KERNELS, _TRIED
    if _TRIED:
        return _KERNELS
    _TRIED = True
    path = os.environ.get("TILE_KERNELS_PATH", "")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    sources: dict[str, dict] = {}
    try:
        from .engram_kernels import engram_gate_bwd as _nb, engram_gate_fwd as _nf
        sources["nano"] = {"fwd": _nf, "bwd": _nb}
    except Exception:  # noqa: BLE001 - optional dependency (tilelang)
        pass
    try:
        from tile_kernels.engram import engram_gate_bwd as _ob, engram_gate_fwd as _of
        sources["official"] = {"fwd": _of, "bwd": _ob}
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    _KERNELS = sources or False
    return _KERNELS


def tl_available() -> bool:
    if os.environ.get("NANOV41_ENGRAM_TL", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool(_load_kernels())


def pad_hidden(dim: int) -> int:
    """Smallest kernel-valid hidden size >= dim with an exact power-of-two scale (official path)."""
    p = int(dim)
    while p < 512 or p % 256 != 0:
        p *= 4
    return p


def _select(dim: int):
    """Return (kernels, pad, c, wscale) for a hidden size, preferring native (no-padding) paths."""
    ks = _load_kernels()
    if not ks:
        return None, dim, 1.0, 1.0
    if "nano" in ks and dim % 128 == 0 and dim >= 256 and dim % 256 != 0:
        return ks["nano"], dim, 1.0, 1.0
    if "official" in ks and dim % 256 == 0:
        return ks["official"], dim, 1.0, 1.0
    if "nano" in ks and dim % 128 == 0 and dim >= 256:
        return ks["nano"], dim, 1.0, 1.0
    if "official" in ks:
        pad = pad_hidden(dim)
        c = math.sqrt(pad / dim)
        return ks["official"], pad, c, 1.0 / c
    return None, dim, 1.0, 1.0


class EngramGateTL(torch.autograd.Function):
    """out = x + gate(x, kv, wh*we) * value, forward+backward fused with TileLang kernels."""

    @staticmethod
    def forward(ctx, x, kv, wh, we, eps, clamp_value, image_mask):
        dim = x.shape[-1]
        src, pad, c, wscale = _select(dim)
        if src is None:
            raise RuntimeError("no TileLang Engram gate kernels available")
        n_tok, hc, _ = x.shape
        if pad == dim:
            xp, kvp = x, kv
            wp = wh.float() * we.float()
        else:
            xp = torch.zeros((n_tok, hc, pad), dtype=torch.bfloat16, device=x.device)
            xp[..., :dim].copy_(x * c)
            kvp = torch.zeros((n_tok, hc + 1, pad), dtype=torch.bfloat16, device=x.device)
            kvp[:, :hc, :dim].copy_(kv[:, :hc, :dim] * c)
            kvp[:, hc:, :dim].copy_(kv[:, hc:, :dim])
            wp = torch.zeros((hc, pad), dtype=torch.float32, device=x.device)
            wp[:, :dim].copy_((wh.float() * we.float()) * wscale)
        kout, dot, gate, rstd_x, rstd_k = src["fwd"](
            xp, kvp, wp, float(eps), float(clamp_value),
            save_for_backward=True, image_token_mask=image_mask)
        # no padding: kernel output already is x + gate*value (skip the extra addcmul kernel)
        out = kout if pad == dim else torch.addcmul(
            x, gate.to(x.dtype).unsqueeze(-1), kv[:, hc:, :])
        ctx.save_for_backward(xp, kvp, wp, dot, gate, rstd_x, rstd_k, wh, we)
        ctx.dim, ctx.pad, ctx.c, ctx.wscale = dim, pad, c, wscale
        ctx.clamp_value, ctx.image_mask = float(clamp_value), image_mask
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xp, kvp, wp, dot, gate, rstd_x, rstd_k, wh, we = ctx.saved_tensors
        dim = ctx.dim
        src, _, _, _ = _select(dim)
        if ctx.pad == dim:
            gpad = grad_out
        else:
            gpad = torch.zeros_like(xp)
            gpad[..., :dim].copy_(grad_out)
        grad_xp, grad_kvp, grad_w_partial = src["bwd"](
            gpad, xp, kvp, wp, dot, gate, rstd_x, rstd_k, ctx.clamp_value,
            image_token_mask=ctx.image_mask)
        c = ctx.c
        if ctx.pad == dim:
            grad_x = grad_xp
            grad_kv = grad_kvp
            grad_w = grad_w_partial.sum(0)
        else:
            grad_x = (c * grad_xp[..., :dim].float() - (c - 1.0) * grad_out.float()).to(grad_out.dtype)
            grad_kv = torch.empty_like(kvp[:, :, :dim])
            grad_kv[:, :hc, :].copy_((c * grad_kvp[:, :hc, :dim].float()).to(grad_out.dtype))
            grad_kv[:, hc:, :].copy_(grad_kvp[:, hc:, :dim].to(grad_out.dtype))
            grad_w = grad_w_partial.sum(0)[:, :dim]
        grad_wh = (grad_w * we.float() * ctx.wscale).to(wh.dtype)
        grad_we = (grad_w * wh.float() * ctx.wscale).to(we.dtype)
        return grad_x, grad_kv, grad_wh, grad_we, None, None, None


def tl_engram_forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                      token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in `Engram.forward` using the fused TileLang gate kernels."""
    kv = self.wkv(self.embed(hash_ids).flatten(-2))
    b, seq, hc, dim = x.shape
    x2 = x.reshape(b * seq, hc, dim)
    kv2 = kv.reshape(b * seq, hc + 1, dim)
    mask = None
    if token_mask is not None:
        mask = (~token_mask).reshape(-1)   # official kernels: True == image token (gate zeroed)
    out = EngramGateTL.apply(x2, kv2, self.q_weight, self.k_weight,
                             self.eps, self.clamp_value, mask)
    return out.reshape(b, seq, hc, dim)
