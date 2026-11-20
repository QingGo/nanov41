# SPDX-License-Identifier: MIT
"""Fused small-M MoE decode kernels (M2d).

The reference small-M decode path (``moe_decode.py``) runs, per layer: a gate (bf16->fp32 casts,
an fp32 GEMV, sqrt-softplus, bias add, ``topk(16)``, a gather, top-k normalisation), three
``index_select`` gathers of the selected expert weights + three batched small GEMMs for the routed
experts, the same three GEMMs for the shared expert, clamp/silu/cast chains, and a bf16
``index_add_``.  At M=1 that is 40+ tiny kernels per layer and it dominates the graph step (~0.45 ms
of GPU time before this module; the top-2 experts are only 3 x 384 x 384 bf16 = 0.88 MB per token).

This module keeps the arithmetic of the small-M path but replaces it with **4 kernels per layer**:

``moe_gate_kernel``
    one program per token: 16 fp32 logits -> sqrt-softplus -> + bias -> top-2 (descending) ->
    normalise + route-scale.  Writes int32 indices and fp32 weights.
``moe_gu_kernel``
    one program per (expert slot x inter-dim chunk) with 3 slots (2 routed + 1 shared): gate/up
    GEMV with the expert offset resolved inside the kernel (no weight gather), then the small-M
    path's bf16 rounding chain: ``bf16(gate) -> clamp -> bf16(silu) * bf16(up) -> [* bf16(weight)]``.
``moe_down_partial_kernel``
    one program per (token x slot x K-split x D-chunk): down GEMV, fp32 partial sums.
``moe_down_combine_kernel``
    sums the K partials, rounds each slot to bf16, then reproduces
    ``out = bf16(bf16(d0 + d1) + shared)`` (the reference order: ``index_add_`` the routed downs,
    then ``+ shared_experts``).

Only enabled for bf16 CUDA tensors with <= ``max_tokens`` tokens (decode); everything else falls
back to the previously installed forward.  Weight layouts follow ``moe_decode.enable_moe_decode``:
routed ``w1/w3`` are ``[E, dim, inter]`` and ``w2`` is ``[E, inter, dim]``; the shared-expert
weights are transposed once at install time to the same convention.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


# --------------------------------------------------------------------------------- kernels
if HAS_TRITON:
    @triton.jit
    def moe_gate_kernel(X, WG, BIAS, IDX, WTS,
                        D: tl.constexpr, E: tl.constexpr, BLOCK_D: tl.constexpr,
                        TEMP: tl.constexpr, ROUTE_SCALE: tl.constexpr,
                        NORM_TOPK: tl.constexpr):
        """x [N, D] -> top-2 expert indices (int32) and routing weights (fp32), sqrt-softplus gate."""
        n = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + n * D + cols, mask=mask, other=0.0).to(tl.float32)
        b1sel = -3.0e38
        b2sel = -3.0e38
        b1sc = 0.0
        b2sc = 0.0
        b1i = 0
        b2i = 0
        for e in tl.static_range(0, E):
            we = tl.load(WG + e * D + cols, mask=mask, other=0.0).to(tl.float32)
            logit = tl.sum(we * x, axis=0) / TEMP
            sp = tl.where(logit > 20.0, logit, tl.log(1.0 + tl.exp(logit)))
            sc = tl.sqrt(sp)
            sel = sc + tl.load(BIAS + e).to(tl.float32)
            if sel > b1sel:
                b2sel, b2sc, b2i = b1sel, b1sc, b1i
                b1sel, b1sc, b1i = sel, sc, e
            elif sel > b2sel:
                b2sel, b2sc, b2i = sel, sc, e
        w1 = b1sc
        w2 = b2sc
        if NORM_TOPK:
            denom = w1 + w2 + 1e-20
            w1 = w1 / denom
            w2 = w2 / denom
        w1 = w1 * ROUTE_SCALE
        w2 = w2 * ROUTE_SCALE
        tl.store(IDX + n * 2 + 0, b1i)
        tl.store(IDX + n * 2 + 1, b2i)
        tl.store(WTS + n * 2 + 0, w1)
        tl.store(WTS + n * 2 + 1, w2)

    @triton.jit
    def moe_gu_kernel(X, W1, W3, W1S, W3S, IDX, WTS, ACT,
                      D: tl.constexpr, I: tl.constexpr, BLOCK_D: tl.constexpr,
                      BD: tl.constexpr, BI: tl.constexpr, LIM: tl.constexpr):
        """Expert gate/up GEMV + clamp/silu/weight, one program per (slot, inter chunk)."""
        p = tl.program_id(0)                      # token * 3 + slot
        ic = tl.program_id(1)                     # inter-dim chunk
        n = p // 3
        s = p % 3
        cols = tl.arange(0, BLOCK_D)
        dmask = cols < D
        x = tl.load(X + n * D + cols, mask=dmask, other=0.0).to(tl.float32)
        ii = ic * BI + tl.arange(0, BI)
        imask = ii < I
        if s < 2:
            e = tl.load(IDX + n * 2 + s).to(tl.int32)
            w1p = W1 + e.to(tl.int64) * (D * I)
            w3p = W3 + e.to(tl.int64) * (D * I)
        else:
            w1p = W1S
            w3p = W3S
        gacc = tl.zeros((BI,), tl.float32)
        uacc = tl.zeros((BI,), tl.float32)
        for d0 in tl.static_range(0, BLOCK_D, BD):
            dd = d0 + tl.arange(0, BD)
            dvalid = dd < D
            w1t = tl.load(w1p + dd[:, None] * I + ii[None, :],
                          mask=dvalid[:, None] & imask[None, :], other=0.0)
            w3t = tl.load(w3p + dd[:, None] * I + ii[None, :],
                          mask=dvalid[:, None] & imask[None, :], other=0.0)
            xd = tl.load(X + n * D + dd, mask=dvalid, other=0.0).to(tl.float32)
            gacc += tl.sum(w1t.to(tl.float32) * xd[:, None], axis=0)
            uacc += tl.sum(w3t.to(tl.float32) * xd[:, None], axis=0)
        # bf16 rounding chain of the small-M path
        g = gacc.to(tl.bfloat16).to(tl.float32)
        u = uacc.to(tl.bfloat16).to(tl.float32)
        if LIM > 0.0:
            u = tl.minimum(tl.maximum(u, -LIM), LIM)
            g = tl.minimum(g, LIM)
        a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u
        a = a.to(tl.bfloat16).to(tl.float32)
        if s < 2:
            w = tl.load(WTS + n * 2 + s).to(tl.bfloat16).to(tl.float32)
            a = a * w
        tl.store(ACT + p * I + ii, a.to(ACT.dtype.element_ty), mask=imask)

    @triton.jit
    def moe_down_partial_kernel(ACT, W2, W2S, IDX, PART,
                                D: tl.constexpr, I: tl.constexpr, KSPLIT: tl.constexpr,
                                CHUNK_I: tl.constexpr, BD: tl.constexpr, BI: tl.constexpr):
        """Down GEMV, K-split over the inter dim; one program per (token, slot, split, D chunk)."""
        q = tl.program_id(0)                      # (n * 3 + s) * KSPLIT + kc
        dc = tl.program_id(1)                     # D chunk
        kc = q % KSPLIT
        p = q // KSPLIT                           # token * 3 + slot
        n = p // 3
        s = p % 3
        dd = dc * BD + tl.arange(0, BD)
        dmask = dd < D
        acc = tl.zeros((BD,), tl.float32)
        if s < 2:
            e = tl.load(IDX + n * 2 + s).to(tl.int32)
            w2p = W2 + e.to(tl.int64) * (I * D)
        else:
            w2p = W2S
        for i0 in tl.static_range(0, CHUNK_I, BI):
            ii = kc * CHUNK_I + i0 + tl.arange(0, BI)
            imask = ii < I
            w2t = tl.load(w2p + ii[:, None] * D + dd[None, :],
                          mask=imask[:, None] & dmask[None, :], other=0.0)
            a = tl.load(ACT + p * I + ii, mask=imask, other=0.0).to(tl.float32)
            acc += tl.sum(w2t.to(tl.float32) * a[:, None], axis=0)
        tl.store(PART + (q * D) + dd, acc, mask=dmask)

    @triton.jit
    def moe_down_combine_kernel(PART, OUT,
                                D: tl.constexpr, KSPLIT: tl.constexpr, BD: tl.constexpr):
        """Sum the K partials per slot, then ``bf16(bf16(d0 + d1) + shared)`` like the reference."""
        n = tl.program_id(0)
        for d0 in tl.static_range(0, D, BD):
            dd = d0 + tl.arange(0, BD)
            dmask = dd < D
            acc0 = tl.zeros((BD,), tl.float32)
            acc1 = tl.zeros((BD,), tl.float32)
            acc2 = tl.zeros((BD,), tl.float32)
            for k in tl.static_range(0, KSPLIT):
                base0 = ((n * 3 + 0) * KSPLIT + k) * D
                base1 = ((n * 3 + 1) * KSPLIT + k) * D
                base2 = ((n * 3 + 2) * KSPLIT + k) * D
                acc0 += tl.load(PART + base0 + dd, mask=dmask, other=0.0)
                acc1 += tl.load(PART + base1 + dd, mask=dmask, other=0.0)
                acc2 += tl.load(PART + base2 + dd, mask=dmask, other=0.0)
            d0b = acc0.to(tl.bfloat16).to(tl.float32)
            d1b = acc1.to(tl.bfloat16).to(tl.float32)
            dsb = acc2.to(tl.bfloat16).to(tl.float32)
            t = (d0b + d1b).to(tl.bfloat16).to(tl.float32)
            y = (t + dsb).to(tl.bfloat16)
            tl.store(OUT + n * D + dd, y.to(OUT.dtype.element_ty), mask=dmask)


# --------------------------------------------------------------------------------- forwards
def _moe_fast_forward(x2: torch.Tensor, moe) -> torch.Tensor:
    """x2 [N, D] bf16 contiguous -> y [N, D] bf16 (written into the module's static out buffer)."""
    cfg = moe._mof_cfg
    n = int(x2.shape[0])
    d, i = cfg["D"], cfg["I"]
    idx = moe._mof_idx[:n]
    wts = moe._mof_wts[:n]
    act = moe._mof_act
    part = moe._mof_part[: n * 3 * cfg["KSPLIT"]]
    out = moe._mof_out[:n]

    moe_gate_kernel[(n,)](
        x2, cfg["gate_w"], cfg["gate_b"], idx, wts,
        D=d, E=cfg["E"], BLOCK_D=cfg["BLOCK_D"], TEMP=cfg["TEMP"],
        ROUTE_SCALE=cfg["ROUTE_SCALE"], NORM_TOPK=cfg["NORM_TOPK"], num_warps=4,
    )
    moe_gu_kernel[(n * 3, triton.cdiv(i, cfg["BI_GU"]))](
        x2, cfg["w1"], cfg["w3"], cfg["w1s"], cfg["w3s"], idx, wts, act,
        D=d, I=i, BLOCK_D=cfg["BLOCK_D"], BD=cfg["BD_GU"], BI=cfg["BI_GU"],
        LIM=cfg["LIM"], num_warps=4,
    )
    moe_down_partial_kernel[(n * 3 * cfg["KSPLIT"], triton.cdiv(d, cfg["BD_DN"]))](
        act, cfg["w2"], cfg["w2s"], idx, part,
        D=d, I=i, KSPLIT=cfg["KSPLIT"], CHUNK_I=cfg["CHUNK_I"],
        BD=cfg["BD_DN"], BI=cfg["BI_DN"], num_warps=4,
    )
    moe_down_combine_kernel[(n,)](
        part, out, D=d, KSPLIT=cfg["KSPLIT"], BD=cfg["BD_COMBINE"], num_warps=4,
    )
    return out


def enable_moe_decode_fast(model, *, max_tokens: int = 8, bi_gu: int = 64,
                           bd_gu: int = 64, ksplt: int = 4, bd_dn: int = 64,
                           bi_dn: int = 32, bd_combine: int = 128) -> int:
    """Patch every small-M MoE module with the fused 4-kernel decode forward (inference only).

    Requires ``moe_decode.enable_moe_decode`` to have run first (it stacks the expert weights as
    ``_decode_w1/_decode_w2/_decode_w3``).
    """
    if not HAS_TRITON or not torch.cuda.is_available():
        raise RuntimeError("fused MoE decode kernels need CUDA+Triton")
    inner = getattr(model, "inner", model)
    n_patched = 0
    for layer in getattr(inner, "layers", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is None or not hasattr(ffn, "_decode_w1") or getattr(ffn, "_mof_installed", False):
            continue
        gate = ffn.gate
        if str(getattr(gate, "score_func", "")) != "sqrtsoftplus":
            continue
        w1, w2, w3 = ffn._decode_w1, ffn._decode_w2, ffn._decode_w3
        shared = ffn.shared_experts
        w1s = shared.w1.weight.detach().t().contiguous().to(w1.dtype)
        w3s = shared.w3.weight.detach().t().contiguous().to(w1.dtype)
        w2s = shared.w2.weight.detach().t().contiguous().to(w1.dtype)
        dev = w1.device
        dim = int(ffn.dim)
        inter = int(w1.shape[2])
        e = int(w1.shape[0])
        bi_gu = min(bi_gu, _next_pow2(inter))
        cfg = {
            "D": dim, "I": inter, "E": e,
            "BLOCK_D": _next_pow2(dim),
            "TEMP": float(getattr(gate, "gate_temp", 1.0) or 1.0),
            "ROUTE_SCALE": float(getattr(gate, "route_scale", 1.0) or 1.0),
            "NORM_TOPK": bool(getattr(gate, "norm_topk_prob", False)),
            "LIM": float(getattr(ffn, "_decode_swiglu_limit", 0.0) or 0.0),
            "gate_w": gate.weight.detach(),
            "gate_b": gate.bias.detach(),
            "w1": w1, "w2": w2, "w3": w3, "w1s": w1s, "w2s": w2s, "w3s": w3s,
            "BI_GU": bi_gu, "BD_GU": bd_gu,
            "KSPLIT": ksplt, "CHUNK_I": (inter + ksplt - 1) // ksplt,
            "BD_DN": bd_dn, "BI_DN": bi_dn, "BD_COMBINE": bd_combine,
        }
        ffn._mof_cfg = cfg
        ffn._mof_idx = torch.zeros(max_tokens, 2, dtype=torch.int32, device=dev)
        ffn._mof_wts = torch.zeros(max_tokens, 2, dtype=torch.float32, device=dev)
        ffn._mof_act = torch.empty(3 * max_tokens, inter, dtype=torch.bfloat16, device=dev)
        ffn._mof_part = torch.empty(max_tokens * 3 * ksplt, dim, dtype=torch.float32,
                                    device=dev)
        ffn._mof_out = torch.empty(max_tokens, dim, dtype=torch.bfloat16, device=dev)
        ffn._mof_max_tokens = max_tokens
        ffn._mof_orig_forward = ffn.forward

        def _forward(self, x, image_mask=None):
            n = x.numel() // self.dim
            if (image_mask is not None or n > self._mof_max_tokens
                    or x.dtype != torch.bfloat16 or not x.is_cuda):
                return self._mof_orig_forward(x, image_mask)
            x2 = x.reshape(n, self.dim)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            return _moe_fast_forward(x2, self).view_as(x)

        ffn.forward = _forward.__get__(ffn, type(ffn))
        ffn._mof_installed = True
        n_patched += 1
    return n_patched


def disable_moe_decode_fast(model) -> int:
    inner = getattr(model, "inner", model)
    n = 0
    for layer in getattr(inner, "layers", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is None or not getattr(ffn, "_mof_installed", False):
            continue
        ffn.forward = ffn._mof_orig_forward
        for name in ("_mof_cfg", "_mof_idx", "_mof_wts", "_mof_act", "_mof_part", "_mof_out",
                     "_mof_orig_forward", "_mof_installed", "_mof_max_tokens"):
            delattr(ffn, name)
        n += 1
    return n


__all__ = ["enable_moe_decode_fast", "disable_moe_decode_fast", "HAS_TRITON"]
