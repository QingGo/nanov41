# SPDX-License-Identifier: MIT
"""Decode-specialised mHC (Hyper-Connections) kernels — M2c, batched in M3.

At T=1 the training-fused mHC kernels still tile 16 rows (``tl.dot`` needs BM>=16) and launch
separate kernels for mixes / sinkhorn / post: ~648 us of the 2.31 ms graph-replay step.  This
module replaces the whole per-block mHC chain for decode with three per-token programs:

  A. ``mixes + split-sinkhorn (attn cell) + hc_pre + attn RMSNorm``  -> attn input (li)
  B. ``hc_post(attn) + mixes + split-sinkhorn (ffn cell) + hc_pre + ffn RMSNorm`` -> y, ffn input
  C. ``hc_post(ffn)``                                                -> block output

They mirror ``fused_mhc_v3``'s arithmetic (bf16 dot for the ``fn`` projections, float32 statistics,
the same softmax + sinkhorn iteration order with the same epsilon placement), so the numerics stay
in the same class as the validated fused mHC (bf16-level; see ``tools/verify_mhc.py``).

M3: each kernel runs one program **per sequence row** (``grid=(B,)``) over per-row workspaces, so a
batched decode step of up to ``max_batch_size`` rows re-uses the same three kernels.  The two block
RMSNorms are fused into A/B (the reference chain is ~7 ATen kernels each, 24 calls per step), and
the per-row ``li`` pass is done in fp32 from the bf16 hc_pre output exactly like the reference norm.

Inference only: ``enable_mhc_decode_fast`` wraps the fused-v3 ``Block.forward`` and only takes the
fast path for single-token (T=1) calls; prefill keeps the validated fused path.
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

HC = 4
MIX = 24
MIXP = 32
BK = 128
MAX_BATCH = 16
LOG2E = 1.4426950408889634


if HAS_TRITON:
    @triton.jit
    def _take(acc, i: tl.constexpr, MIXP: tl.constexpr):
        """Scalar element ``i`` of the (padded) mixes vector."""
        return tl.sum(tl.where(tl.arange(0, MIXP) == i, acc, 0.0))

    @triton.jit
    def _rowvec(acc, base_off, j: tl.constexpr, s2, BASE, MIXP: tl.constexpr,
                HC: tl.constexpr):
        """comb row j (4 base values + 4 scaled mixes) as a [HC] float32 vector."""
        off = base_off + j * HC
        rr = tl.arange(0, HC)
        v = tl.load(BASE + off + rr)
        for k in tl.static_range(0, HC):
            v += tl.where(rr == k, _take(acc, off + k, MIXP), 0.0) * s2
        return v

    @triton.jit
    def _sink(r0, r1, r2, r3, ITERS: tl.constexpr, EPS: tl.constexpr):
        """softmax rows + `ITERS` alternating row/column normalisations (reference order)."""
        a0 = tl.exp2((r0 - tl.max(r0)) * 1.4426950408889634)
        a1 = tl.exp2((r1 - tl.max(r1)) * 1.4426950408889634)
        a2 = tl.exp2((r2 - tl.max(r2)) * 1.4426950408889634)
        a3 = tl.exp2((r3 - tl.max(r3)) * 1.4426950408889634)
        a0 = a0 / tl.sum(a0) + EPS
        a1 = a1 / tl.sum(a1) + EPS
        a2 = a2 / tl.sum(a2) + EPS
        a3 = a3 / tl.sum(a3) + EPS
        cs = a0 + a1 + a2 + a3 + EPS
        b0 = a0 / cs
        b1 = a1 / cs
        b2 = a2 / cs
        b3 = a3 / cs
        for _ in range(1, ITERS):
            rs0 = tl.sum(b0)
            rs1 = tl.sum(b1)
            rs2 = tl.sum(b2)
            rs3 = tl.sum(b3)
            c0 = b0 / (rs0 + EPS)
            c1 = b1 / (rs1 + EPS)
            c2 = b2 / (rs2 + EPS)
            c3 = b3 / (rs3 + EPS)
            ds = c0 + c1 + c2 + c3 + EPS
            b0 = c0 / ds
            b1 = c1 / ds
            b2 = c2 / ds
            b3 = c3 / ds
        return b0, b1, b2, b3

    @triton.jit
    def _rmsnorm_rows_inplace(LI, W, b, D: tl.constexpr, BK: tl.constexpr,
                              EPS: tl.constexpr):
        """RMSNorm of row ``b`` of ``LI`` (bf16) in place, fp32 statistics like the reference."""
        rk = tl.arange(0, BK)
        ss = 0.0
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            li = tl.load(LI + b * D + cols).to(tl.float32)
            ss += tl.sum(li * li)
        rstd = 1.0 / tl.sqrt(ss / D + EPS)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            li = tl.load(LI + b * D + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            tl.store(LI + b * D + cols, (li * rstd * w).to(LI.dtype.element_ty))

    @triton.jit
    def decode_mixes_pre_kernel(X, FN, PMIX, NORMW, LI, PRE, POST, COMB, SCALE, BASE,
                                D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr,
                                MIX: tl.constexpr, MIXP: tl.constexpr, BK: tl.constexpr,
                                ITERS: tl.constexpr, EPS: tl.constexpr,
                                NORM_EPS: tl.constexpr, ROW_EPS: tl.constexpr):
        """mixes + attn-cell sinkhorn + hc_pre + attn RMSNorm, one program per sequence row."""
        b = tl.program_id(0)
        xb = X + b * K
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        ss = 0.0
        acc = tl.zeros((MIXP,), tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            li = tl.zeros((BK,), tl.float32)
            for hc in tl.static_range(0, HC):
                kcols = hc * D + cols
                xf = tl.load(xb + kcols).to(tl.float32)
                f = tl.load(FN + rc[:, None] * K + kcols[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                fb = f.to(tl.bfloat16).to(tl.float32)
                acc += tl.sum(fb * xf[None, :], axis=1)
                ss += tl.sum(xf * xf)
                li += tl.load(PMIX + b * HC + hc) * xf
            tl.store(LI + b * D + cols, li.to(LI.dtype.element_ty))
        r = 1.0 / tl.sqrt(ss / K + NORM_EPS)

        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        for j in tl.static_range(0, HC):
            tl.store(PRE + b * HC + j, 1.0 / (1.0 + tl.exp(-(_take(acc, j, MIXP) * r * s0
                                                            + tl.load(BASE + j)))) + EPS)
            tl.store(POST + b * HC + j, 2.0 / (1.0 + tl.exp(-(_take(acc, HC + j, MIXP) * r * s1
                                                              + tl.load(BASE + HC + j)))))
        r0 = _rowvec(acc, 2 * HC, 0, s2 * r, BASE, MIXP, HC)
        r1 = _rowvec(acc, 2 * HC, 1, s2 * r, BASE, MIXP, HC)
        r2 = _rowvec(acc, 2 * HC, 2, s2 * r, BASE, MIXP, HC)
        r3 = _rowvec(acc, 2 * HC, 3, s2 * r, BASE, MIXP, HC)
        b0, b1, b2, b3 = _sink(r0, r1, r2, r3, ITERS, EPS)
        rr = tl.arange(0, HC)
        tl.store(COMB + b * HC * HC + 0 * HC + rr, b0)
        tl.store(COMB + b * HC * HC + 1 * HC + rr, b1)
        tl.store(COMB + b * HC * HC + 2 * HC + rr, b2)
        tl.store(COMB + b * HC * HC + 3 * HC + rr, b3)
        _rmsnorm_rows_inplace(LI, NORMW, b, D, BK, ROW_EPS)

    @triton.jit
    def decode_post_mixes_pre_kernel(Y, LI, FN, X1, RES, POST, COMB, PMIX, NORMW,
                                     PRE2, POST2, COMB2, SCALE, BASE,
                                     D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr,
                                     MIX: tl.constexpr, MIXP: tl.constexpr, BK: tl.constexpr,
                                     ITERS: tl.constexpr, EPS: tl.constexpr,
                                     NORM_EPS: tl.constexpr, ROW_EPS: tl.constexpr):
        """hc_post(attn) -> y; ffn mixes+sinkhorn; hc_pre + ffn RMSNorm -> li (one program/row)."""
        b = tl.program_id(0)
        rk = tl.arange(0, BK)
        rc = tl.arange(0, MIXP)
        post_b = POST + b * HC
        comb_b = COMB + b * HC * HC
        res_b = RES + b * K
        p0 = tl.load(post_b + 0)
        p1 = tl.load(post_b + 1)
        p2 = tl.load(post_b + 2)
        p3 = tl.load(post_b + 3)
        c0 = tl.load(comb_b + 0 * HC + 0)
        c1 = tl.load(comb_b + 1 * HC + 0)
        c2 = tl.load(comb_b + 2 * HC + 0)
        c3 = tl.load(comb_b + 3 * HC + 0)
        d0_ = tl.load(comb_b + 0 * HC + 1)
        d1_ = tl.load(comb_b + 1 * HC + 1)
        d2_ = tl.load(comb_b + 2 * HC + 1)
        d3_ = tl.load(comb_b + 3 * HC + 1)
        e0 = tl.load(comb_b + 0 * HC + 2)
        e1 = tl.load(comb_b + 1 * HC + 2)
        e2 = tl.load(comb_b + 2 * HC + 2)
        e3 = tl.load(comb_b + 3 * HC + 2)
        f0 = tl.load(comb_b + 0 * HC + 3)
        f1 = tl.load(comb_b + 1 * HC + 3)
        f2 = tl.load(comb_b + 2 * HC + 3)
        f3 = tl.load(comb_b + 3 * HC + 3)
        pm0 = tl.load(PMIX + b * HC + 0)
        pm1 = tl.load(PMIX + b * HC + 1)
        pm2 = tl.load(PMIX + b * HC + 2)
        pm3 = tl.load(PMIX + b * HC + 3)
        ss = 0.0
        acc = tl.zeros((MIXP,), tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + b * D + cols).to(tl.float32)
            r0 = tl.load(res_b + 0 * D + cols).to(tl.float32)
            r1 = tl.load(res_b + 1 * D + cols).to(tl.float32)
            r2 = tl.load(res_b + 2 * D + cols).to(tl.float32)
            r3 = tl.load(res_b + 3 * D + cols).to(tl.float32)
            o0 = p0 * x1 + c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3
            o1 = p1 * x1 + d0_ * r0 + d1_ * r1 + d2_ * r2 + d3_ * r3
            o2 = p2 * x1 + e0 * r0 + e1 * r1 + e2 * r2 + e3 * r3
            o3 = p3 * x1 + f0 * r0 + f1 * r1 + f2 * r2 + f3 * r3
            b0 = o0.to(tl.bfloat16).to(tl.float32)
            b1 = o1.to(tl.bfloat16).to(tl.float32)
            b2 = o2.to(tl.bfloat16).to(tl.float32)
            b3 = o3.to(tl.bfloat16).to(tl.float32)
            tl.store(Y + b * K + 0 * D + cols, b0.to(Y.dtype.element_ty))
            tl.store(Y + b * K + 1 * D + cols, b1.to(Y.dtype.element_ty))
            tl.store(Y + b * K + 2 * D + cols, b2.to(Y.dtype.element_ty))
            tl.store(Y + b * K + 3 * D + cols, b3.to(Y.dtype.element_ty))
            ss += tl.sum(b0 * b0) + tl.sum(b1 * b1) + tl.sum(b2 * b2) + tl.sum(b3 * b3)
            tl.store(LI + b * D + cols, (pm0 * b0 + pm1 * b1 + pm2 * b2 + pm3 * b3)
                     .to(LI.dtype.element_ty))
            for hc in tl.static_range(0, HC):
                kcols = hc * D + cols
                bf = b0 if hc == 0 else (b1 if hc == 1 else (b2 if hc == 2 else b3))
                f = tl.load(FN + rc[:, None] * K + kcols[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                fb = f.to(tl.bfloat16).to(tl.float32)
                acc += tl.sum(fb * bf[None, :], axis=1)
        r = 1.0 / tl.sqrt(ss / K + NORM_EPS)

        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        for j in tl.static_range(0, HC):
            tl.store(PRE2 + b * HC + j, 1.0 / (1.0 + tl.exp(-(_take(acc, j, MIXP) * r * s0
                                                             + tl.load(BASE + j)))) + EPS)
            tl.store(POST2 + b * HC + j, 2.0 / (1.0 + tl.exp(-(_take(acc, HC + j, MIXP) * r * s1
                                                               + tl.load(BASE + HC + j)))))
        r0 = _rowvec(acc, 2 * HC, 0, s2 * r, BASE, MIXP, HC)
        r1 = _rowvec(acc, 2 * HC, 1, s2 * r, BASE, MIXP, HC)
        r2 = _rowvec(acc, 2 * HC, 2, s2 * r, BASE, MIXP, HC)
        r3 = _rowvec(acc, 2 * HC, 3, s2 * r, BASE, MIXP, HC)
        b0s, b1s, b2s, b3s = _sink(r0, r1, r2, r3, ITERS, EPS)
        rr = tl.arange(0, HC)
        tl.store(COMB2 + b * HC * HC + 0 * HC + rr, b0s)
        tl.store(COMB2 + b * HC * HC + 1 * HC + rr, b1s)
        tl.store(COMB2 + b * HC * HC + 2 * HC + rr, b2s)
        tl.store(COMB2 + b * HC * HC + 3 * HC + rr, b3s)
        _rmsnorm_rows_inplace(LI, NORMW, b, D, BK, ROW_EPS)

    @triton.jit
    def decode_post_kernel(Y2, X1, RES, POST, COMB,
                           D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr,
                           BK: tl.constexpr):
        """hc_post(ffn): y2[i] = post[i]*x1 + sum_j comb[j,i]*res[j] (bf16 out), per row."""
        b = tl.program_id(0)
        rk = tl.arange(0, BK)
        post_b = POST + b * HC
        comb_b = COMB + b * HC * HC
        res_b = RES + b * K
        p0 = tl.load(post_b + 0)
        p1 = tl.load(post_b + 1)
        p2 = tl.load(post_b + 2)
        p3 = tl.load(post_b + 3)
        c0 = tl.load(comb_b + 0 * HC + 0)
        c1 = tl.load(comb_b + 1 * HC + 0)
        c2 = tl.load(comb_b + 2 * HC + 0)
        c3 = tl.load(comb_b + 3 * HC + 0)
        d0_ = tl.load(comb_b + 0 * HC + 1)
        d1_ = tl.load(comb_b + 1 * HC + 1)
        d2_ = tl.load(comb_b + 2 * HC + 1)
        d3_ = tl.load(comb_b + 3 * HC + 1)
        e0 = tl.load(comb_b + 0 * HC + 2)
        e1 = tl.load(comb_b + 1 * HC + 2)
        e2 = tl.load(comb_b + 2 * HC + 2)
        e3 = tl.load(comb_b + 3 * HC + 2)
        f0 = tl.load(comb_b + 0 * HC + 3)
        f1 = tl.load(comb_b + 1 * HC + 3)
        f2 = tl.load(comb_b + 2 * HC + 3)
        f3 = tl.load(comb_b + 3 * HC + 3)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + b * D + cols).to(tl.float32)
            r0 = tl.load(res_b + 0 * D + cols).to(tl.float32)
            r1 = tl.load(res_b + 1 * D + cols).to(tl.float32)
            r2 = tl.load(res_b + 2 * D + cols).to(tl.float32)
            r3 = tl.load(res_b + 3 * D + cols).to(tl.float32)
            tl.store(Y2 + b * K + 0 * D + cols,
                     (p0 * x1 + c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3).to(Y2.dtype.element_ty))
            tl.store(Y2 + b * K + 1 * D + cols,
                     (p1 * x1 + d0_ * r0 + d1_ * r1 + d2_ * r2 + d3_ * r3).to(Y2.dtype.element_ty))
            tl.store(Y2 + b * K + 2 * D + cols,
                     (p2 * x1 + e0 * r0 + e1 * r1 + e2 * r2 + e3 * r3).to(Y2.dtype.element_ty))
            tl.store(Y2 + b * K + 3 * D + cols,
                     (p3 * x1 + f0 * r0 + f1 * r1 + f2 * r2 + f3 * r3).to(Y2.dtype.element_ty))


def _workspace(block, device, dtype, d):
    """Per-block decode workspace, sized for MAX_BATCH rows (fixed addresses = graph safe)."""
    ws = getattr(block, "_mhc_fast_ws", None)
    if ws is not None:
        return ws
    hc = HC
    k = hc * d
    n = MAX_BATCH
    ws = {
        "li": torch.empty(n * d, dtype=dtype, device=device),
        "attn_pre": torch.empty(n * hc, dtype=torch.float32, device=device),
        "attn_post": torch.empty(n * hc, dtype=torch.float32, device=device),
        "attn_comb": torch.empty(n * hc * hc, dtype=torch.float32, device=device),
        "y": torch.empty(n * k, dtype=dtype, device=device),
        "li2": torch.empty(n * d, dtype=dtype, device=device),
        "ffn_pre": torch.empty(n * hc, dtype=torch.float32, device=device),
        "ffn_post": torch.empty(n * hc, dtype=torch.float32, device=device),
        "ffn_comb": torch.empty(n * hc * hc, dtype=torch.float32, device=device),
        "y2": torch.empty(n * k, dtype=dtype, device=device),
    }
    block._mhc_fast_ws = ws
    return ws


def block_forward_decode_fast(block, x, start_pos, pre_mix, image_mask, *attn_args):
    """One-token-per-row ``Block.forward`` (1 <= B <= MAX_BATCH) on the decode mHC kernels."""
    b, s, hc, d = x.shape
    assert s == 1 and hc == HC and 1 <= b <= MAX_BATCH
    k = hc * d
    xf = x.reshape(b * k)
    ws = _workspace(block, x.device, x.dtype, d)
    pm = pre_mix.reshape(b * HC).float().contiguous()
    li = ws["li"][: b * d]
    attn_pre = ws["attn_pre"][: b * HC]
    attn_post = ws["attn_post"][: b * HC]
    attn_comb = ws["attn_comb"][: b * HC * HC]
    y = ws["y"][: b * k]
    li2 = ws["li2"][: b * d]
    ffn_pre = ws["ffn_pre"][: b * HC]
    ffn_post = ws["ffn_post"][: b * HC]
    ffn_comb = ws["ffn_comb"][: b * HC * HC]
    y2 = ws["y2"][: b * k]
    decode_mixes_pre_kernel[(b,)](
        xf, block.hc_attn_fn, pm, block.attn_norm.weight, li,
        attn_pre, attn_post, attn_comb,
        block.hc_attn_scale, block.hc_attn_base,
        D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK,
        ITERS=block.hc_sinkhorn_iters, EPS=block.hc_eps, NORM_EPS=block.norm_eps,
        ROW_EPS=block.attn_norm.eps, num_warps=4,
    )
    h = block.attn(li.view(b, 1, d), start_pos, *attn_args)
    decode_post_mixes_pre_kernel[(b,)](
        y, li2, block.hc_ffn_fn, h.reshape(b * d), xf,
        attn_post, attn_comb, attn_pre,
        block.ffn_norm.weight, ffn_pre, ffn_post, ffn_comb,
        block.hc_ffn_scale, block.hc_ffn_base,
        D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK,
        ITERS=block.hc_sinkhorn_iters, EPS=block.hc_eps, NORM_EPS=block.norm_eps,
        ROW_EPS=block.ffn_norm.eps, num_warps=4,
    )
    h2 = block.ffn(li2.view(b, 1, d), image_mask)
    decode_post_kernel[(b,)](
        y2, h2.reshape(b * d), y, ffn_post, ffn_comb,
        D=d, K=k, HC=HC, BK=BK, num_warps=4,
    )
    return y2.view(b, 1, HC, d), ffn_pre.view(b, 1, HC)


def _block_forward_dispatch(self, x, start_pos, pre_mix, image_mask, *attn_args):
    if x.shape[1] == 1 and x.shape[0] <= MAX_BATCH:
        return block_forward_decode_fast(self, x, start_pos, pre_mix, image_mask, *attn_args)
    return type(self)._mhc_fast_orig(self, x, start_pos, pre_mix, image_mask, *attn_args)


def enable_mhc_decode_fast(model) -> int:
    """Patch ``Block.forward`` with a single-token/row fast path (inference only)."""
    if not HAS_TRITON or not torch.cuda.is_available():
        raise RuntimeError("decode mHC kernels need CUDA+Triton")
    from . import fused_mhc_v3

    fused_mhc_v3.install()  # the validated fused path stays the fallback for T>1
    inner = getattr(model, "inner", model)
    block_cls = type(inner.layers[0])
    if getattr(block_cls, "_mhc_fast_installed", False):
        return 0
    block_cls._mhc_fast_orig = block_cls.__dict__.get("forward")
    block_cls.forward = _block_forward_dispatch
    block_cls._mhc_fast_installed = True
    return len(inner.layers)


def disable_mhc_decode_fast(model) -> int:
    inner = getattr(model, "inner", model)
    block_cls = type(inner.layers[0])
    if not getattr(block_cls, "_mhc_fast_installed", False):
        return 0
    block_cls.forward = block_cls._mhc_fast_orig
    del block_cls._mhc_fast_orig
    del block_cls._mhc_fast_installed
    for layer in inner.layers:
        if hasattr(layer, "_mhc_fast_ws"):
            del layer._mhc_fast_ws
    return len(inner.layers)


__all__ = ["enable_mhc_decode_fast", "disable_mhc_decode_fast", "block_forward_decode_fast"]
