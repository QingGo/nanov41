"""Drop-in replacement for the mHC Sinkhorn part using DeepSeek TileKernels (fwd+bwd).

The reference model computes `comb` as: comb_mix -> softmax(rows) -> column normalize ->
(repeat-1) x [row, col] normalize. TileKernels `mhc_sinkhorn_fwd/bwd` implements exactly the
same normalization (and its analytic backward) as a single TileLang kernel; empirically it is
~40x faster than the pure-torch fallback on a 4090 and also runs on sm_120 (5090).

Falls back to the torch implementation when tile_kernels/tilelang are unavailable (e.g. Mac).
"""
from __future__ import annotations

import os
import sys

import torch

_CANDIDATES = [os.environ.get("TILE_KERNELS_PATH", "")]
_AVAILABLE = False
for _p in _CANDIDATES:
    if _p and os.path.isdir(_p):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
try:  # pragma: no cover - depends on remote environment
    from tile_kernels.mhc import mhc_sinkhorn_fwd, mhc_sinkhorn_bwd
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

REPEAT = 20
EPS = 1e-6


class _SinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mix: torch.Tensor, repeat: int, eps: float):
        out = torch.empty_like(mix)
        mhc_sinkhorn_fwd(mix.contiguous(), out, repeat=repeat, eps=eps)
        ctx.save_for_backward(mix)
        ctx.repeat, ctx.eps = repeat, eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mix, = ctx.saved_tensors
        gi = torch.empty_like(mix)
        mhc_sinkhorn_bwd(grad_out.contiguous(), mix.contiguous(), gi, repeat=ctx.repeat, eps=ctx.eps)
        return gi, None, None


def sinkhorn_tl(mix: torch.Tensor, repeat: int = REPEAT, eps: float = EPS) -> torch.Tensor:
    """mix: [N, hc, hc] fp32 pre-softmax comb values -> doubly-stochastic-ish comb."""
    assert _AVAILABLE, "tile_kernels not importable"
    return _SinkhornFn.apply(mix.reshape(-1, mix.size(-2), mix.size(-1)).float(), repeat, eps).view_as(mix)


def hc_split_sinkhorn_tl(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Same signature/semantics as patched_env.hc_split_sinkhorn, but the Sinkhorn loop is a
    single TileLang kernel (with analytic backward) when available."""
    b, s, _ = mixes.size()
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb_mix = m[..., 2 * hc:].reshape(b, s, hc, hc) * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)
    if _AVAILABLE:
        comb = sinkhorn_tl(comb_mix, repeat=sinkhorn_iters, eps=eps)
    else:  # torch fallback identical to patched_env
        comb = comb_mix.softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
