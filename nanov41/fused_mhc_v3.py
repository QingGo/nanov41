"""fused_mhc_v3 — mHC (Hyper-Connections) training path as fused Triton kernels.

Drop-in replacement for the three `Block` methods in `_ref_patched_model.py`
(`hc_mixes` / `hc_pre` / `hc_post`) **plus** a fused `Block.forward`, so that the
two mHC cells of a block need 3 kernels per cell instead of ~8-10 torch ops.

## What it replaces (per mHC cell, d12 b=16 s=1024)

The reference trains through 24 mHC cells/step (12 layers x 2 sub-layers).  Each cell:

    mixes [T,24] = (x_flat @ fn^T) * rsqrt(mean(x_flat^2) + eps)     # 50 MB read
    pre/post/comb = split_sinkhorn(mixes)                            # 20-iter Sinkhorn
    li = sum_hc pre_mix * x                                          # 50 MB read + 12.6 MB write
    y  = post_j * x1 + sum_i comb[i,j] * res_i                       # 50+12.6 read, 50 write

`hc_mixes` is immediately followed by `hc_pre` (same `x`), and `hc_post` is immediately
followed by the next cell's `(hc_mixes, hc_pre)`.  The vanilla path re-reads the 50 MB
residual 4-6x per cell in separate elementwise kernels.  This module:

  * `_mixes_pre_*`       one kernel: norm + fn GEMM + rsqrt + hc_pre collapse.
  * `_split_sinkhorn_*`  one kernel: sigmoid split + 20-iter Sinkhorn (fwd) and its
                         analytic VJP (Megatron `_triton_sinkhorn_bwd` walk, but the
                         forward denominators are saved instead of all 40 matrices,
                         and the matrices are rebuilt exactly during the reverse walk).
  * `_post_*`            one kernel: hc_post fwd; one kernel: all 4 grads.
  * `_pm_pre_*`          one kernel: hc_post + mixes + hc_pre (the cross-cell fusion).
  * `du^T @ y` for `d_fn` is the only cuBLAS call (same bf16 GEMM class as level3).

Everything is CUDA-graph safe: no host sync, no `.item()`/`.tolist()`, no boolean
indexing, no shape-dependent Python branches at run time, all scratch buffers are
`torch.empty` (graph-pool friendly), and the forward/backward have no autotuning.

Numerics: bf16 inputs, fp32 accumulate everywhere, fp32 `mixes`/`pre`/`post`/`comb`
(level3 semantics).  The fn projection is a bf16 `tl.dot` (fp32 accumulate), exactly
the class of `F.linear(x_flat, fn.to(bf16))` used by `fast_mhc` level3.

Requirements: CUDA, bf16 activations, hc_mult == 4 (the d12 / V4.1-Flash setting).
"""
from __future__ import annotations

import os

import torch

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = [
    "mhc_mixes", "mhc_pre", "mhc_post", "mhc_mixes_pre", "mhc_post_mixes_pre",
    "mhc_split_sinkhorn", "install", "uninstall", "restore", "get_stats", "available",
]

HC = 4
MIX = 24
MIXP = 32

# Tile sizes (env-overridable for a quick sweep on a new GPU).
BM = int(os.environ.get("NANOV41_V3_BM", "16"))          # rows per program (>=16 for tl.dot)
BK = int(os.environ.get("NANOV41_V3_BK", "128"))         # hidden tile for the small GEMM
BM_S = int(os.environ.get("NANOV41_V3_BM_S", "64"))      # rows per sinkhorn program
BKP = int(os.environ.get("NANOV41_V3_BKP", "64"))        # hidden tile for the fused bwd
PMPF_BK = int(os.environ.get("NANOV41_V3_PMPF_BK", "128"))  # hidden tile for the fused fwd
# R45: `_pm_pre` 前向 v2（RES 读放大 4x -> 1x；i 外层、4 个 j 累加器同存，BK=64 时寄存器够）
PMPF_V2 = os.environ.get("NANOV41_V3_PMPF_V2", "0").lower() not in ("0", "false", "no", "off")
PMPF_BK2 = int(os.environ.get("NANOV41_V3_PMPF_BK2", "64"))
# R46：`_pm_pre` fwd/bwd 独立 tile/BM/warps/stages（全局 BM 被 mixes/post 共用，不能一起扫）
PMPF_BM = int(os.environ.get("NANOV41_V3_PMPF_BM", "16"))
PMPF_S = int(os.environ.get("NANOV41_V3_PMPF_S", "2"))
PMPFB_BM = int(os.environ.get("NANOV41_V3_PMPFB_BM", "16"))
PMPFB_BK = int(os.environ.get("NANOV41_V3_PMPFB_BK", "64"))
PMPFB_S = int(os.environ.get("NANOV41_V3_PMPFB_S", "2"))


def _w(name, default):
    return int(os.environ.get(name, str(default)))


if HAS_TRITON:

    # =====================================================================
    # 1. mixes (+ optional hc_pre) forward / backward
    # =====================================================================
    @triton.jit
    def _mixes_pre_fwd_kernel(
        X, FN, PMIX, LI, MIXES, R, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr,
        BK: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr, HAS_PRE: tl.constexpr,
    ):
        """mixes = (x @ fn^T) * rsqrt(ss/K + eps);  li = sum_hc pmix*x  (if HAS_PRE)."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, MIXP), dtype=tl.float32)
        ss = tl.zeros((BM,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            li_tile = tl.zeros((BM, BK), dtype=tl.float32)
            for hc in tl.static_range(0, HC):
                kcols = hc * D + cols
                x = tl.load(X + rows[:, None] * K + kcols[None, :],
                            mask=rmask[:, None], other=0.0)
                f = tl.load(FN + rc[:, None] * K + kcols[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                acc = tl.dot(x, tl.trans(f).to(tl.bfloat16), acc)
                xf = x.to(tl.float32)
                ss += tl.sum(xf * xf, axis=1)
                if HAS_PRE:
                    p = tl.load(PMIX + rows * HC + hc, mask=rmask, other=0.0)
                    li_tile += p[:, None] * xf
            if HAS_PRE:
                tl.store(LI + rows[:, None] * D + cols[None, :],
                         li_tile.to(LI.dtype.element_ty), mask=rmask[:, None])
        r = tl.rsqrt(ss / K + EPS)
        tl.store(MIXES + rows[:, None] * MIX + rc[None, :], acc * r[:, None],
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        tl.store(R + rows, r, mask=rmask)

    @triton.jit
    def _mixes_pre_bwd_kernel(
        X, FN, PMIX, DLI, MIXES, R, DMIX, DX, DU, DPRE, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr,
        BK: tl.constexpr, BM: tl.constexpr, HAS_PRE: tl.constexpr,
    ):
        """dx = du @ fn + 2*dss*x (+ pmix*gli); dpre = sum_d gli*x; du stored for d_fn."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        r = tl.load(R + rows, mask=rmask, other=1.0)
        mm = tl.load(MIXES + rows[:, None] * MIX + rc[None, :],
                     mask=rmask[:, None] & (rc[None, :] < MIX), other=0.0)
        dm = tl.load(DMIX + rows[:, None] * MIX + rc[None, :],
                     mask=rmask[:, None] & (rc[None, :] < MIX), other=0.0)
        u = mm / r[:, None]
        dss = -0.5 * r * r * r / K * tl.sum(dm * u, axis=1)
        du = dm * r[:, None]
        tl.store(DU + rows[:, None] * MIX + rc[None, :], du.to(tl.bfloat16),
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        dpre = tl.zeros((BM, HC), dtype=tl.float32)
        rc4 = tl.arange(0, HC)
        dub = du.to(tl.bfloat16)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            if HAS_PRE:
                gl = tl.load(DLI + rows[:, None] * D + cols[None, :],
                             mask=rmask[:, None], other=0.0).to(tl.float32)
            for hc in tl.static_range(0, HC):
                kcols = hc * D + cols
                x = tl.load(X + rows[:, None] * K + kcols[None, :],
                            mask=rmask[:, None], other=0.0)
                f = tl.load(FN + rc[:, None] * K + kcols[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                xf = x.to(tl.float32)
                dx = tl.dot(dub, f.to(tl.bfloat16)) + 2.0 * dss[:, None] * xf
                if HAS_PRE:
                    p = tl.load(PMIX + rows * HC + hc, mask=rmask, other=0.0)
                    dx += p[:, None] * gl
                    dp = tl.sum(gl * xf, axis=1)
                    dpre += tl.where(rc4[None, :] == hc, dp[:, None], 0.0)
                tl.store(DX + rows[:, None] * K + kcols[None, :],
                         dx.to(DX.dtype.element_ty), mask=rmask[:, None])
        if HAS_PRE:
            tl.store(DPRE + rows[:, None] * HC + rc4[None, :], dpre, mask=rmask[:, None])

    # =====================================================================
    # 2. split + Sinkhorn (fwd / analytic bwd)
    # =====================================================================
    @triton.jit
    def _split_sinkhorn_fwd_kernel(
        MIXES, SCALE, BASE, PRE, POST, COMB, MINIT, WS, T,
        ITERS: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        BM: tl.constexpr, EPS: tl.constexpr,
    ):
        """split + softmax + sinkhorn, one [HC,HC] matrix per row kept as HC row vectors."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, HC)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        b0 = tl.load(BASE + rc)
        b1 = tl.load(BASE + HC + rc)
        m0 = tl.load(MIXES + rows[:, None] * MIX + rc[None, :], mask=rmask[:, None], other=0.0)
        pre = 1.0 / (1.0 + tl.exp(-(m0 * s0 + b0[None, :]))) + EPS
        tl.store(PRE + rows[:, None] * HC + rc[None, :], pre, mask=rmask[:, None])
        m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rc)[None, :], mask=rmask[:, None],
                     other=0.0)
        post = 2.0 / (1.0 + tl.exp(-(m1 * s1 + b1[None, :])))
        tl.store(POST + rows[:, None] * HC + rc[None, :], post, mask=rmask[:, None])
        # comb_mix split into HC row vectors
        r0 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 0 * HC + rc)[None, :],
                     mask=rmask[:, None], other=0.0) * s2 + tl.load(BASE + 2 * HC + 0 * HC + rc)[None, :]
        r1 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 1 * HC + rc)[None, :],
                     mask=rmask[:, None], other=0.0) * s2 + tl.load(BASE + 2 * HC + 1 * HC + rc)[None, :]
        r2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 2 * HC + rc)[None, :],
                     mask=rmask[:, None], other=0.0) * s2 + tl.load(BASE + 2 * HC + 2 * HC + rc)[None, :]
        r3 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 3 * HC + rc)[None, :],
                     mask=rmask[:, None], other=0.0) * s2 + tl.load(BASE + 2 * HC + 3 * HC + rc)[None, :]
        # row softmax
        t0 = tl.exp2((r0 - tl.max(r0, axis=1)[:, None]) * 1.4426950408889634)
        t1 = tl.exp2((r1 - tl.max(r1, axis=1)[:, None]) * 1.4426950408889634)
        t2 = tl.exp2((r2 - tl.max(r2, axis=1)[:, None]) * 1.4426950408889634)
        t3 = tl.exp2((r3 - tl.max(r3, axis=1)[:, None]) * 1.4426950408889634)
        q0 = t0 / tl.sum(t0, axis=1)[:, None]
        q1 = t1 / tl.sum(t1, axis=1)[:, None]
        q2 = t2 / tl.sum(t2, axis=1)[:, None]
        q3 = t3 / tl.sum(t3, axis=1)[:, None]
        mb = MINIT + rows[:, None] * (HC * HC) + rc[None, :]
        tl.store(mb + 0 * HC, q0, mask=rmask[:, None])
        tl.store(mb + 1 * HC, q1, mask=rmask[:, None])
        tl.store(mb + 2 * HC, q2, mask=rmask[:, None])
        tl.store(mb + 3 * HC, q3, mask=rmask[:, None])
        a0 = q0 + EPS
        a1 = q1 + EPS
        a2 = q2 + EPS
        a3 = q3 + EPS
        csum = a0 + a1 + a2 + a3
        wb = WS + rows[:, None] * HC + rc[None, :]
        tl.store(wb, csum, mask=rmask[:, None])
        cs = csum + EPS
        b0v = a0 / cs
        b1v = a1 / cs
        b2v = a2 / cs
        b3v = a3 / cs
        for t in range(1, ITERS):
            rs0 = tl.sum(b0v, axis=1)
            rs1 = tl.sum(b1v, axis=1)
            rs2 = tl.sum(b2v, axis=1)
            rs3 = tl.sum(b3v, axis=1)
            wr = WS + ITERS * T * HC + t * T * HC + rows * HC
            tl.store(wr + 0, rs0, mask=rmask)
            tl.store(wr + 1, rs1, mask=rmask)
            tl.store(wr + 2, rs2, mask=rmask)
            tl.store(wr + 3, rs3, mask=rmask)
            c0v = b0v / (rs0[:, None] + EPS)
            c1v = b1v / (rs1[:, None] + EPS)
            c2v = b2v / (rs2[:, None] + EPS)
            c3v = b3v / (rs3[:, None] + EPS)
            csum2 = c0v + c1v + c2v + c3v
            tl.store(WS + t * T * HC + rows[:, None] * HC + rc[None, :], csum2,
                     mask=rmask[:, None])
            ds = csum2 + EPS
            b0v = c0v / ds
            b1v = c1v / ds
            b2v = c2v / ds
            b3v = c3v / ds
        cb = COMB + rows[:, None] * (HC * HC) + rc[None, :]
        tl.store(cb + 0 * HC, b0v, mask=rmask[:, None])
        tl.store(cb + 1 * HC, b1v, mask=rmask[:, None])
        tl.store(cb + 2 * HC, b2v, mask=rmask[:, None])
        tl.store(cb + 3 * HC, b3v, mask=rmask[:, None])

    @triton.jit
    def _split_sinkhorn_bwd_kernel(
        MIXES, SCALE, PRE, POST, COMB, MINIT, WS, GPRE, GPOST, GCOMB,
        DMIX, DB_PART, DS_PART, T,
        ITERS: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        BM: tl.constexpr, EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, HC)
        gb = GCOMB + rows[:, None] * (HC * HC) + rc[None, :]
        G0 = tl.load(gb + 0 * HC, mask=rmask[:, None], other=0.0)
        G1 = tl.load(gb + 1 * HC, mask=rmask[:, None], other=0.0)
        G2 = tl.load(gb + 2 * HC, mask=rmask[:, None], other=0.0)
        G3 = tl.load(gb + 3 * HC, mask=rmask[:, None], other=0.0)
        cb = COMB + rows[:, None] * (HC * HC) + rc[None, :]
        M0 = tl.load(cb + 0 * HC, mask=rmask[:, None], other=0.0)
        M1 = tl.load(cb + 1 * HC, mask=rmask[:, None], other=0.0)
        M2 = tl.load(cb + 2 * HC, mask=rmask[:, None], other=0.0)
        M3 = tl.load(cb + 3 * HC, mask=rmask[:, None], other=0.0)
        for k in range(0, ITERS - 1):
            t = ITERS - 1 - k
            c = tl.load(WS + t * T * HC + rows[:, None] * HC + rc[None, :],
                        mask=rmask[:, None], other=1.0)
            ce = c + EPS
            G0 = G0 / ce
            G1 = G1 / ce
            G2 = G2 / ce
            G3 = G3 / ce
            # column normalization: the correction is per column (sum over matrix rows)
            corr = G0 * M0 + G1 * M1 + G2 * M2 + G3 * M3
            G0 = G0 - corr
            G1 = G1 - corr
            G2 = G2 - corr
            G3 = G3 - corr
            C0 = M0 * ce
            C1 = M1 * ce
            C2 = M2 * ce
            C3 = M3 * ce
            rb = WS + ITERS * T * HC + t * T * HC + rows * HC
            re0 = tl.load(rb + 0, mask=rmask, other=1.0) + EPS
            re1 = tl.load(rb + 1, mask=rmask, other=1.0) + EPS
            re2 = tl.load(rb + 2, mask=rmask, other=1.0) + EPS
            re3 = tl.load(rb + 3, mask=rmask, other=1.0) + EPS
            G0 = G0 / re0[:, None]
            G1 = G1 / re1[:, None]
            G2 = G2 / re2[:, None]
            G3 = G3 / re3[:, None]
            G0 = G0 - tl.sum(G0 * C0, axis=1)[:, None]
            G1 = G1 - tl.sum(G1 * C1, axis=1)[:, None]
            G2 = G2 - tl.sum(G2 * C2, axis=1)[:, None]
            G3 = G3 - tl.sum(G3 * C3, axis=1)[:, None]
            M0 = C0 * re0[:, None]
            M1 = C1 * re1[:, None]
            M2 = C2 * re2[:, None]
            M3 = C3 * re3[:, None]
        c0 = tl.load(WS + rows[:, None] * HC + rc[None, :],
                     mask=rmask[:, None], other=1.0)
        c0e = c0 + EPS
        G0 = G0 / c0e
        G1 = G1 / c0e
        G2 = G2 / c0e
        G3 = G3 / c0e
        corr = G0 * M0 + G1 * M1 + G2 * M2 + G3 * M3
        G0 = G0 - corr
        G1 = G1 - corr
        G2 = G2 - corr
        G3 = G3 - corr
        # softmax VJP
        mb = MINIT + rows[:, None] * (HC * HC) + rc[None, :]
        S0 = tl.load(mb + 0 * HC, mask=rmask[:, None], other=0.0)
        S1 = tl.load(mb + 1 * HC, mask=rmask[:, None], other=0.0)
        S2 = tl.load(mb + 2 * HC, mask=rmask[:, None], other=0.0)
        S3 = tl.load(mb + 3 * HC, mask=rmask[:, None], other=0.0)
        G0 = S0 * (G0 - tl.sum(G0 * S0, axis=1)[:, None])
        G1 = S1 * (G1 - tl.sum(G1 * S1, axis=1)[:, None])
        G2 = S2 * (G2 - tl.sum(G2 * S2, axis=1)[:, None])
        G3 = S3 * (G3 - tl.sum(G3 * S3, axis=1)[:, None])

        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        m0 = tl.load(MIXES + rows[:, None] * MIX + rc[None, :], mask=rmask[:, None], other=0.0)
        m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rc)[None, :], mask=rmask[:, None],
                     other=0.0)
        gpre = tl.load(GPRE + rows[:, None] * HC + rc[None, :], mask=rmask[:, None], other=0.0)
        gpost = tl.load(GPOST + rows[:, None] * HC + rc[None, :], mask=rmask[:, None], other=0.0)
        pre = tl.load(PRE + rows[:, None] * HC + rc[None, :], mask=rmask[:, None], other=0.0)
        post = tl.load(POST + rows[:, None] * HC + rc[None, :], mask=rmask[:, None], other=0.0)
        sig0 = pre - EPS
        dz0 = gpre * sig0 * (1.0 - sig0)
        sig1 = post * 0.5
        dz1 = gpost * 2.0 * sig1 * (1.0 - sig1)
        db_ = DMIX + rows[:, None] * MIX + rc[None, :]
        tl.store(db_ + 0 * HC, dz0 * s0, mask=rmask[:, None])
        tl.store(db_ + 1 * HC, dz1 * s1, mask=rmask[:, None])
        tl.store(db_ + 2 * HC, G0 * s2, mask=rmask[:, None])
        tl.store(db_ + 3 * HC, G1 * s2, mask=rmask[:, None])
        tl.store(db_ + 4 * HC, G2 * s2, mask=rmask[:, None])
        tl.store(db_ + 5 * HC, G3 * s2, mask=rmask[:, None])
        tl.store(DB_PART + pid * MIX + rc, tl.sum(dz0, axis=0))
        tl.store(DB_PART + pid * MIX + HC + rc, tl.sum(dz1, axis=0))
        tl.store(DB_PART + pid * MIX + 2 * HC + 0 * HC + rc, tl.sum(G0, axis=0))
        tl.store(DB_PART + pid * MIX + 2 * HC + 1 * HC + rc, tl.sum(G1, axis=0))
        tl.store(DB_PART + pid * MIX + 2 * HC + 2 * HC + rc, tl.sum(G2, axis=0))
        tl.store(DB_PART + pid * MIX + 2 * HC + 3 * HC + rc, tl.sum(G3, axis=0))
        m2_0 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 0 * HC + rc)[None, :],
                       mask=rmask[:, None], other=0.0)
        m2_1 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 1 * HC + rc)[None, :],
                       mask=rmask[:, None], other=0.0)
        m2_2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 2 * HC + rc)[None, :],
                       mask=rmask[:, None], other=0.0)
        m2_3 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 3 * HC + rc)[None, :],
                       mask=rmask[:, None], other=0.0)
        ds0 = tl.sum(dz0 * m0)
        ds1 = tl.sum(dz1 * m1)
        ds2 = tl.sum(G0 * m2_0) + tl.sum(G1 * m2_1) + tl.sum(G2 * m2_2) + tl.sum(G3 * m2_3)
        r4 = tl.arange(0, 4)
        ds = tl.where(r4 == 0, ds0, tl.where(r4 == 1, ds1, tl.where(r4 == 2, ds2, 0.0)))
        tl.store(DS_PART + pid * 3 + r4, ds, mask=r4 < 3)

    # =====================================================================
    # 3. hc_pre alone (model's final collapse) and hc_post
    # =====================================================================
    @triton.jit
    def _pre_fwd_kernel(X, PMIX, LI, T, D: tl.constexpr, K: tl.constexpr,
                        HC: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            li = tl.zeros((BM, BK), dtype=tl.float32)
            for hc in tl.static_range(0, HC):
                x = tl.load(X + rows[:, None] * K + (hc * D + cols)[None, :],
                            mask=rmask[:, None], other=0.0)
                p = tl.load(PMIX + rows * HC + hc, mask=rmask, other=0.0)
                li += p[:, None] * x.to(tl.float32)
            tl.store(LI + rows[:, None] * D + cols[None, :],
                     li.to(LI.dtype.element_ty), mask=rmask[:, None])

    @triton.jit
    def _pre_bwd_kernel(X, PMIX, DLI, DX, DPRE, T, D: tl.constexpr, K: tl.constexpr,
                        HC: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        rc4 = tl.arange(0, HC)
        dpre = tl.zeros((BM, HC), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            gl = tl.load(DLI + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            for hc in tl.static_range(0, HC):
                x = tl.load(X + rows[:, None] * K + (hc * D + cols)[None, :],
                            mask=rmask[:, None], other=0.0)
                p = tl.load(PMIX + rows * HC + hc, mask=rmask, other=0.0)
                tl.store(DX + rows[:, None] * K + (hc * D + cols)[None, :],
                         (p[:, None] * gl).to(DX.dtype.element_ty), mask=rmask[:, None])
                dp = tl.sum(gl * x.to(tl.float32), axis=1)
                dpre += tl.where(rc4[None, :] == hc, dp[:, None], 0.0)
        tl.store(DPRE + rows[:, None] * HC + rc4[None, :], dpre, mask=rmask[:, None])

    @triton.jit
    def _post_fwd_kernel(Y, X1, RES, POST, COMB, T,
                         D: tl.constexpr, HC: tl.constexpr, BK: tl.constexpr,
                         BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r0 = tl.load(RES + rbase + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            rs = (r0, r1, r2, r3)
            o0 = p0[:, None] * x1
            o1 = p1[:, None] * x1
            o2 = p2[:, None] * x1
            o3 = p3[:, None] * x1
            for i in tl.static_range(0, HC):
                ri = rs[i]
                o0 += tl.load(COMB + rows * (HC * HC) + i * 4 + 0, mask=rmask,
                              other=0.0)[:, None] * ri
                o1 += tl.load(COMB + rows * (HC * HC) + i * 4 + 1, mask=rmask,
                              other=0.0)[:, None] * ri
                o2 += tl.load(COMB + rows * (HC * HC) + i * 4 + 2, mask=rmask,
                              other=0.0)[:, None] * ri
                o3 += tl.load(COMB + rows * (HC * HC) + i * 4 + 3, mask=rmask,
                              other=0.0)[:, None] * ri
            ob = rows[:, None] * (HC * D) + cols[None, :]
            tl.store(Y + ob + 0 * D, o0.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 1 * D, o1.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 2 * D, o2.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 3 * D, o3.to(Y.dtype.element_ty), mask=rmask[:, None])

    @triton.jit
    def _post_bwd_kernel(DX1, DRES, DPOST, DCOMB, GY, X1, RES, POST, COMB, T,
                         D: tl.constexpr, HC: tl.constexpr, BK: tl.constexpr,
                         BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        rc4 = tl.arange(0, HC)
        rc16 = tl.arange(0, HC * HC)
        dpost = tl.zeros((BM, HC), dtype=tl.float32)
        dcomb = tl.zeros((BM, HC * HC), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            gb = rows[:, None] * (HC * D)
            g0 = tl.load(GY + gb + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g1 = tl.load(GY + gb + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g2 = tl.load(GY + gb + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g3 = tl.load(GY + gb + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            gs = (g0, g1, g2, g3)
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            dx1 = p0[:, None] * g0 + p1[:, None] * g1 + p2[:, None] * g2 + p3[:, None] * g3
            tl.store(DX1 + rows[:, None] * D + cols[None, :], dx1.to(DX1.dtype.element_ty),
                     mask=rmask[:, None])
            r0 = tl.load(RES + rbase + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            rs = (r0, r1, r2, r3)
            for i in tl.static_range(0, HC):
                dri = tl.zeros((BM, BK), dtype=tl.float32)
                for j in tl.static_range(0, HC):
                    dri += tl.load(COMB + rows * (HC * HC) + i * 4 + j, mask=rmask,
                                   other=0.0)[:, None] * gs[j]
                tl.store(DRES + rbase + (i * D + d0 + rk)[None, :],
                         dri.to(DRES.dtype.element_ty), mask=rmask[:, None])
                ri = rs[i]
                for j in tl.static_range(0, HC):
                    dc = tl.sum(gs[j] * ri, axis=1)
                    dcomb += tl.where(rc16[None, :] == i * 4 + j, dc[:, None], 0.0)
            for j in tl.static_range(0, HC):
                dp = tl.sum(gs[j] * x1, axis=1)
                dpost += tl.where(rc4[None, :] == j, dp[:, None], 0.0)
        tl.store(DPOST + rows[:, None] * HC + rc4[None, :], dpost, mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + rc16[None, :], dcomb,
                 mask=rmask[:, None])

    # =====================================================================
    # 4. hc_post + hc_mixes + hc_pre in ONE kernel (cross-cell fusion)
    # =====================================================================
    @triton.jit
    def _pm_pre_fwd_kernel(
        Y, LI, MIXES, R, X1, RES, POST, COMB, PMIX, FN, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
    ):
        # hc_post + hc_mixes + hc_pre.  One j at a time keeps the live tile count low
        # (res chunks are re-read from L1), which is what stops ptxas from spilling.
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        q0 = tl.load(PMIX + rows * HC + 0, mask=rmask, other=0.0)
        q1 = tl.load(PMIX + rows * HC + 1, mask=rmask, other=0.0)
        q2 = tl.load(PMIX + rows * HC + 2, mask=rmask, other=0.0)
        q3 = tl.load(PMIX + rows * HC + 3, mask=rmask, other=0.0)
        ps = (p0, p1, p2, p3)
        qs = (q0, q1, q2, q3)
        acc = tl.zeros((BM, MIXP), dtype=tl.float32)
        ss = tl.zeros((BM,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            li = tl.zeros((BM, BK), dtype=tl.float32)
            for j in tl.static_range(0, HC):
                o = ps[j][:, None] * x1
                for i in tl.static_range(0, HC):
                    cij = tl.load(COMB + rows * (HC * HC) + i * 4 + j, mask=rmask,
                                  other=0.0)
                    ri = tl.load(RES + rbase + (i * D + d0 + rk)[None, :],
                                 mask=rmask[:, None], other=0.0).to(tl.float32)
                    o += cij[:, None] * ri
                b = o.to(tl.bfloat16)
                bf = b.to(tl.float32)
                tl.store(Y + rows[:, None] * (HC * D) + (j * D + d0 + rk)[None, :], b,
                         mask=rmask[:, None])
                ss += tl.sum(bf * bf, axis=1)
                li += qs[j][:, None] * bf
                w = tl.load(FN + rc[:, None] * K + (j * D + d0 + rk)[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                acc = tl.dot(b, tl.trans(w).to(tl.bfloat16), acc)
            tl.store(LI + rows[:, None] * D + cols[None, :], li.to(LI.dtype.element_ty),
                     mask=rmask[:, None])
        r = tl.rsqrt(ss / K + EPS)
        tl.store(MIXES + rows[:, None] * MIX + rc[None, :], acc * r[:, None],
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        tl.store(R + rows, r, mask=rmask)

    @triton.jit
    def _pm_pre_fwd_v2_kernel(
        Y, LI, MIXES, R, X1, RES, POST, COMB, PMIX, FN, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
    ):
        # 与 `_pm_pre_fwd_kernel` 数学一致，但 RES 只读一遍：
        # 旧 kernel 为每个 j 重读全部 HC 个 res tile（读放大 4x，实测 ~207us/层）；
        # 这里把 i 放外层、4 个 j 的累加器同时存活（BK=64 时寄存器够用），RES 读放大降到 1x。
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        q0 = tl.load(PMIX + rows * HC + 0, mask=rmask, other=0.0)
        q1 = tl.load(PMIX + rows * HC + 1, mask=rmask, other=0.0)
        q2 = tl.load(PMIX + rows * HC + 2, mask=rmask, other=0.0)
        q3 = tl.load(PMIX + rows * HC + 3, mask=rmask, other=0.0)
        ps = (p0, p1, p2, p3)
        qs = (q0, q1, q2, q3)
        # comb 系数与 d0 无关：16 个 [BM] 向量一次读入
        cs = (
            tl.load(COMB + rows * (HC * HC) + 0, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 1, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 2, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 3, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 4, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 5, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 6, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 7, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 8, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 9, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 10, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 11, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 12, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 13, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 14, mask=rmask, other=0.0),
            tl.load(COMB + rows * (HC * HC) + 15, mask=rmask, other=0.0),
        )
        acc = tl.zeros((BM, MIXP), dtype=tl.float32)
        ss = tl.zeros((BM,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r0 = tl.load(RES + rbase + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            li = tl.zeros((BM, BK), dtype=tl.float32)
            for j in tl.static_range(0, HC):
                o = ps[j][:, None] * x1
                o += cs[0 * HC + j][:, None] * r0
                o += cs[1 * HC + j][:, None] * r1
                o += cs[2 * HC + j][:, None] * r2
                o += cs[3 * HC + j][:, None] * r3
                b = o.to(tl.bfloat16)
                bf = b.to(tl.float32)
                tl.store(Y + rows[:, None] * (HC * D) + (j * D + d0 + rk)[None, :], b,
                         mask=rmask[:, None])
                ss += tl.sum(bf * bf, axis=1)
                li += qs[j][:, None] * bf
                w = tl.load(FN + rc[:, None] * K + (j * D + d0 + rk)[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                acc = tl.dot(b, tl.trans(w).to(tl.bfloat16), acc)
            tl.store(LI + rows[:, None] * D + cols[None, :], li.to(LI.dtype.element_ty),
                     mask=rmask[:, None])
        r = tl.rsqrt(ss / K + EPS)
        tl.store(MIXES + rows[:, None] * MIX + rc[None, :], acc * r[:, None],
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        tl.store(R + rows, r, mask=rmask)

    @triton.jit
    def _pm_pre_bwd_kernel(
        DX1, DRES, DPOST, DCOMB, DPRE, DU, GY, GLI, Y, X1, RES, POST, COMB, PMIX,
        MIXES, R, DMIX, FN, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
    ):
        # `mixes` is computed from the *output* y inside the fused op, so the mixes-path
        # y-gradient must be added to the incoming gy before it is pushed through hc_post.
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        q0 = tl.load(PMIX + rows * HC + 0, mask=rmask, other=0.0)
        q1 = tl.load(PMIX + rows * HC + 1, mask=rmask, other=0.0)
        q2 = tl.load(PMIX + rows * HC + 2, mask=rmask, other=0.0)
        q3 = tl.load(PMIX + rows * HC + 3, mask=rmask, other=0.0)
        r = tl.load(R + rows, mask=rmask, other=1.0)
        mm = tl.load(MIXES + rows[:, None] * MIX + rc[None, :],
                     mask=rmask[:, None] & (rc[None, :] < MIX), other=0.0)
        dm = tl.load(DMIX + rows[:, None] * MIX + rc[None, :],
                     mask=rmask[:, None] & (rc[None, :] < MIX), other=0.0)
        u = mm / r[:, None]
        dss = -0.5 * r * r * r / K * tl.sum(dm * u, axis=1)
        dub = (dm * r[:, None]).to(tl.bfloat16)
        tl.store(DU + rows[:, None] * MIX + rc[None, :], dub,
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        rc4 = tl.arange(0, HC)
        rc16 = tl.arange(0, HC * HC)
        dpost = tl.zeros((BM, HC), dtype=tl.float32)
        dpre = tl.zeros((BM, HC), dtype=tl.float32)
        dcomb = tl.zeros((BM, HC * HC), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            gb = rows[:, None] * (HC * D)
            g0 = tl.load(GY + gb + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g1 = tl.load(GY + gb + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g2 = tl.load(GY + gb + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g3 = tl.load(GY + gb + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            gl = tl.load(GLI + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            y0 = tl.load(Y + rbase + (0 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            y1 = tl.load(Y + rbase + (1 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            y2 = tl.load(Y + rbase + (2 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            y3 = tl.load(Y + rbase + (3 * D + d0 + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            # mixes-path gradient wrt y (fc + the 2*dss*y norm term)
            w0 = tl.load(FN + rc[:, None] * K + (0 * D + d0 + rk)[None, :],
                         mask=rc[:, None] < MIX, other=0.0)
            g0 += tl.dot(dub, w0.to(tl.bfloat16)) + 2.0 * dss[:, None] * y0
            w1 = tl.load(FN + rc[:, None] * K + (1 * D + d0 + rk)[None, :],
                         mask=rc[:, None] < MIX, other=0.0)
            g1 += tl.dot(dub, w1.to(tl.bfloat16)) + 2.0 * dss[:, None] * y1
            w2 = tl.load(FN + rc[:, None] * K + (2 * D + d0 + rk)[None, :],
                         mask=rc[:, None] < MIX, other=0.0)
            g2 += tl.dot(dub, w2.to(tl.bfloat16)) + 2.0 * dss[:, None] * y2
            w3 = tl.load(FN + rc[:, None] * K + (3 * D + d0 + rk)[None, :],
                         mask=rc[:, None] < MIX, other=0.0)
            g3 += tl.dot(dub, w3.to(tl.bfloat16)) + 2.0 * dss[:, None] * y3
            # the fused hc_pre output `li` also depends on y internally
            g0 += q0[:, None] * gl
            g1 += q1[:, None] * gl
            g2 += q2[:, None] * gl
            g3 += q3[:, None] * gl
            gs = (g0, g1, g2, g3)
            dx1 = p0[:, None] * g0 + p1[:, None] * g1 + p2[:, None] * g2 + p3[:, None] * g3
            tl.store(DX1 + rows[:, None] * D + cols[None, :], dx1.to(DX1.dtype.element_ty),
                     mask=rmask[:, None])
            for i in tl.static_range(0, HC):
                dri = tl.zeros((BM, BK), dtype=tl.float32)
                for j in tl.static_range(0, HC):
                    dri += tl.load(COMB + rows * (HC * HC) + i * 4 + j, mask=rmask,
                                   other=0.0)[:, None] * gs[j]
                tl.store(DRES + rbase + (i * D + d0 + rk)[None, :],
                         dri.to(DRES.dtype.element_ty), mask=rmask[:, None])
                ri = tl.load(RES + rbase + (i * D + d0 + rk)[None, :], mask=rmask[:, None],
                             other=0.0).to(tl.float32)
                for j in tl.static_range(0, HC):
                    dc = tl.sum(gs[j] * ri, axis=1)
                    dcomb += tl.where(rc16[None, :] == i * 4 + j, dc[:, None], 0.0)
            for j in tl.static_range(0, HC):
                dp = tl.sum(gs[j] * x1, axis=1)
                dpost += tl.where(rc4[None, :] == j, dp[:, None], 0.0)
            yss = (y0, y1, y2, y3)
            for i in tl.static_range(0, HC):
                dpi = tl.sum(gl * yss[i], axis=1)
                dpre += tl.where(rc4[None, :] == i, dpi[:, None], 0.0)
        tl.store(DPOST + rows[:, None] * HC + rc4[None, :], dpost, mask=rmask[:, None])
        tl.store(DPRE + rows[:, None] * HC + rc4[None, :], dpre, mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + rc16[None, :], dcomb,
                 mask=rmask[:, None])


# =====================================================================
# launchers
# =====================================================================
def _cdiv(a, b):
    return (a + b - 1) // b


def _g_or_zeros(g, ref):
    if g is not None:
        return g
    return torch.zeros_like(ref)


def _require(x):
    if not HAS_TRITON:
        raise RuntimeError("fused_mhc_v3 requires Triton + CUDA")
    if not x.is_cuda:
        raise RuntimeError("fused_mhc_v3 requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"fused_mhc_v3 requires bf16 activations, got {x.dtype}")
    if x.shape[-2] != HC:
        raise RuntimeError(f"fused_mhc_v3 requires hc_mult={HC}, got {x.shape[-2]}")


def _launch_mixes_pre(x2, fn, pmix, want_pre, eps):
    """x2 [T,K] bf16, fn [MIX,K] fp32, pmix [T,HC] fp32|None -> mixes, li, r."""
    t, k = x2.shape
    d = k // HC
    mixes = torch.empty(t, MIX, dtype=torch.float32, device=x2.device)
    r = torch.empty(t, dtype=torch.float32, device=x2.device)
    li = torch.empty(t, d, dtype=x2.dtype, device=x2.device) if want_pre else None
    _mixes_pre_fwd_kernel[(_cdiv(t, BM),)](
        x2, fn, pmix if pmix is not None else x2, li if li is not None else x2,
        mixes, r, t, D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK, BM=BM,
        EPS=eps, HAS_PRE=want_pre, num_warps=_w("NANOV41_V3_W_MIX", 4), num_stages=2,
    )
    return mixes, li, r


def _launch_pre(x2, pmix):
    t, k = x2.shape
    d = k // HC
    li = torch.empty(t, d, dtype=x2.dtype, device=x2.device)
    _pre_fwd_kernel[(_cdiv(t, BM),)](x2, pmix, li, t, D=d, K=k, HC=HC, BK=BK, BM=BM,
                                     num_warps=_w("NANOV41_V3_W_PRE", 4), num_stages=2)
    return li


def _launch_split_sinkhorn(mixes, scale, base, iters, eps):
    t = mixes.shape[0]
    dev = mixes.device
    pre = torch.empty(t, HC, dtype=torch.float32, device=dev)
    post = torch.empty(t, HC, dtype=torch.float32, device=dev)
    comb = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
    minit = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
    ws = torch.empty(2, iters, t, HC, dtype=torch.float32, device=dev)
    _split_sinkhorn_fwd_kernel[(_cdiv(t, BM_S),)](
        mixes, scale, base, pre, post, comb, minit, ws, t,
        ITERS=iters, HC=HC, MIX=MIX, BM=BM_S, EPS=eps, num_warps=_w("NANOV41_V3_W_SS", 4),
    )
    return pre, post, comb, minit, ws


def _launch_post(x1, res, post, comb):
    t, hc, d = res.shape
    y = torch.empty_like(res)
    _post_fwd_kernel[(_cdiv(t, BM),)](y, x1, res, post, comb, t, D=d, HC=hc, BK=BK, BM=BM,
                                      num_warps=_w("NANOV41_V3_W_POST", 8), num_stages=2)
    return y


def _launch_pm_pre(x1, res, post, comb, pmix, fn, eps):
    t, hc, d = res.shape
    k = hc * d
    y = torch.empty_like(res)
    li = torch.empty(t, d, dtype=res.dtype, device=res.device)
    mixes = torch.empty(t, MIX, dtype=torch.float32, device=res.device)
    r = torch.empty(t, dtype=torch.float32, device=res.device)
    if PMPF_V2:
        _pm_pre_fwd_v2_kernel[(_cdiv(t, PMPF_BM),)](
            y, li, mixes, r, x1, res, post, comb, pmix, fn, t,
            D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=PMPF_BK2, BM=PMPF_BM, EPS=eps,
            num_warps=_w("NANOV41_V3_W_PMP", 4), num_stages=PMPF_S,
        )
    else:
        _pm_pre_fwd_kernel[(_cdiv(t, PMPF_BM),)](
            y, li, mixes, r, x1, res, post, comb, pmix, fn, t,
            D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=PMPF_BK, BM=PMPF_BM, EPS=eps,
            num_warps=_w("NANOV41_V3_W_PMP", 4), num_stages=PMPF_S,
        )
    return y, mixes, r, li


# =====================================================================
# autograd functions
# =====================================================================
class _MixesPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn, pmix, eps, want_pre):
        _require(x)
        t, hc, d = x.shape
        x2 = x.reshape(t, hc * d)
        mixes, li, r = _launch_mixes_pre(x2, fn, pmix, want_pre, eps)
        ctx.save_for_backward(x, fn, pmix, mixes, r)
        ctx.want_pre = want_pre
        return mixes, li

    @staticmethod
    def backward(ctx, dmixes, dli):
        x, fn, pmix, mixes, r = ctx.saved_tensors
        t, hc, d = x.shape
        k = hc * d
        x2 = x.reshape(t, k)
        dx = torch.empty_like(x)
        du = torch.empty(t, MIX, dtype=torch.bfloat16, device=x.device)
        dpre = torch.empty(t, HC, dtype=torch.float32, device=x.device) \
            if ctx.want_pre else None
        dmixes = _g_or_zeros(dmixes, mixes)
        _mixes_pre_bwd_kernel[(_cdiv(t, BM),)](
            x2, fn, pmix if pmix is not None else x2, dli if dli is not None else x2,
            mixes, r, dmixes, dx, du, dpre if dpre is not None else x2, t,
            D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK, BM=BM, HAS_PRE=ctx.want_pre,
            num_warps=_w("NANOV41_V3_W_MIX", 4), num_stages=2,
        )
        dfn = torch.mm(du.t(), x2).to(fn.dtype)
        return dx, dfn, dpre, None, None


class _SplitSinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, iters, eps):
        pre, post, comb, minit, ws = _launch_split_sinkhorn(mixes, scale, base, iters, eps)
        ctx.save_for_backward(mixes, scale, pre, post, comb, minit, ws)
        ctx.iters, ctx.eps = iters, eps
        return pre, post, comb

    @staticmethod
    def backward(ctx, gpre, gpost, gcomb):
        mixes, scale, pre, post, comb, minit, ws = ctx.saved_tensors
        t = mixes.shape[0]
        nprog = _cdiv(t, BM_S)
        dmix = torch.empty_like(mixes)
        db = torch.empty(nprog, MIX, dtype=torch.float32, device=mixes.device)
        ds = torch.empty(nprog, 3, dtype=torch.float32, device=mixes.device)
        gpre = _g_or_zeros(gpre, pre)
        gpost = _g_or_zeros(gpost, post)
        gcomb = _g_or_zeros(gcomb, comb)
        _split_sinkhorn_bwd_kernel[(nprog,)](
            mixes, scale, pre, post, comb, minit, ws,
            gpre.contiguous(), gpost.contiguous(), gcomb.contiguous(),
            dmix, db, ds, t, ITERS=ctx.iters, HC=HC, MIX=MIX, BM=BM_S, EPS=ctx.eps,
            num_warps=_w("NANOV41_V3_W_SS", 4),
        )
        return dmix, ds.sum(0), db.sum(0), None, None


class _PreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pmix):
        _require(x)
        t, hc, d = x.shape
        li = _launch_pre(x.reshape(t, hc * d), pmix)
        ctx.save_for_backward(x, pmix)
        return li

    @staticmethod
    def backward(ctx, dli):
        x, pmix = ctx.saved_tensors
        t, hc, d = x.shape
        dx = torch.empty_like(x)
        dpre = torch.empty(t, HC, dtype=torch.float32, device=x.device)
        _pre_bwd_kernel[(_cdiv(t, BM),)](
            x.reshape(t, hc * d), pmix, dli.reshape(t, d).contiguous(), dx, dpre, t,
            D=d, K=hc * d, HC=HC, BK=BK, BM=BM,
            num_warps=_w("NANOV41_V3_W_PRE", 4), num_stages=2,
        )
        return dx, dpre


class _PostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, res, post, comb):
        _require(res)
        t, hc, d = res.shape
        y = _launch_post(x1.reshape(t, d), res, post, comb)
        ctx.save_for_backward(x1, res, post, comb)
        return y

    @staticmethod
    def backward(ctx, gy):
        x1, res, post, comb = ctx.saved_tensors
        gy = _g_or_zeros(gy, res)
        t, hc, d = res.shape
        dx1 = torch.empty_like(x1)
        dres = torch.empty_like(res)
        dpost = torch.empty(t, HC, dtype=torch.float32, device=res.device)
        dcomb = torch.empty(t, HC * HC, dtype=torch.float32, device=res.device)
        _post_bwd_kernel[(_cdiv(t, BM),)](
            dx1, dres, dpost, dcomb, gy.contiguous(), x1.reshape(t, d), res, post, comb, t,
            D=d, HC=hc, BK=BK, BM=BM, num_warps=_w("NANOV41_V3_W_POST", 8), num_stages=2,
        )
        return dx1, dres, dpost, dcomb


class _PostMixesPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, res, post, comb, pmix, fn, eps):
        _require(res)
        t, hc, d = res.shape
        y, mixes, r, li = _launch_pm_pre(x1.reshape(t, d), res, post, comb, pmix, fn, eps)
        ctx.save_for_backward(x1, res, post, comb, pmix, fn, y, mixes, r)
        return y, mixes, li

    @staticmethod
    def backward(ctx, gy, dmixes, dli):
        x1, res, post, comb, pmix, fn, y, mixes, r = ctx.saved_tensors
        gy = _g_or_zeros(gy, res)
        dmixes = _g_or_zeros(dmixes, mixes)
        dli = _g_or_zeros(dli, y)
        t, hc, d = res.shape
        k = hc * d
        dx1 = torch.empty_like(x1)
        dres = torch.empty_like(res)
        du = torch.empty(t, MIX, dtype=torch.bfloat16, device=res.device)
        dpost = torch.empty(t, HC, dtype=torch.float32, device=res.device)
        dcomb = torch.empty(t, HC * HC, dtype=torch.float32, device=res.device)
        dpre = torch.empty(t, HC, dtype=torch.float32, device=res.device)
        _pm_pre_bwd_kernel[(_cdiv(t, PMPFB_BM),)](
            dx1.reshape(t, d), dres, dpost, dcomb, dpre, du,
            gy.contiguous(), dli.reshape(t, d).contiguous(), y, x1.reshape(t, d), res,
            post, comb, pmix, mixes, r, dmixes, fn, t,
            D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=PMPFB_BK, BM=PMPFB_BM,
            num_warps=_w("NANOV41_V3_W_PMPB", 4), num_stages=PMPFB_S,
        )
        dfn = torch.mm(du.t(), y.reshape(t, k)).to(fn.dtype)
        return dx1, dres, dpost, dcomb, dpre, dfn, None


# =====================================================================
# public ops (shapes as in the model: [b,s,...])
# =====================================================================
def mhc_mixes(x, fn, scale, base, iters=20, eps=1e-6, norm_eps=1e-20):
    """x [b,s,hc,d] -> mixes [b,s,24] fp32 (the norm + projection only)."""
    b, s, hc, d = x.shape
    mixes, _ = _MixesPreFn.apply(x.reshape(b * s, hc, d), fn, None, norm_eps, False)
    return mixes.view(b, s, MIX)


def mhc_mixes_pre(x, fn, pmix, eps=1e-20):
    """x [b,s,hc,d], pmix [b,s,hc] -> (mixes [b,s,24], li [b,s,d]) in one kernel."""
    b, s, hc, d = x.shape
    pm = pmix.reshape(b * s, hc) if pmix is not None else None
    mixes, li = _MixesPreFn.apply(x.reshape(b * s, hc, d), fn, pm, eps, True)
    return mixes.view(b, s, MIX), li.view(b, s, d)


def mhc_split_sinkhorn(mixes, scale, base, iters=20, eps=1e-6):
    """mixes [T,24] (or [b,s,24]) -> pre [..,hc], post [..,hc], comb [..,hc,hc]."""
    shape = mixes.shape
    m = mixes.reshape(-1, shape[-1]).contiguous()
    pre, post, comb = _SplitSinkhornFn.apply(m, scale, base, int(iters), float(eps))
    if len(shape) == 3:
        b, s, _ = shape
        return pre.view(b, s, HC), post.view(b, s, HC), comb.view(b, s, HC, HC)
    return pre, post, comb.view(-1, HC, HC)


def mhc_pre(x, pmix):
    """x [b,s,hc,d], pmix [b,s,hc] -> [b,s,d]."""
    b, s, hc, d = x.shape
    li = _PreFn.apply(x.reshape(b * s, hc, d), pmix.reshape(b * s, hc).contiguous())
    return li.view(b, s, d)


def mhc_post(x1, res, post, comb):
    """x1 [b,s,d], res [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc] -> [b,s,hc,d]."""
    b, s, d = x1.shape
    y = _PostFn.apply(x1.reshape(b * s, d), res.reshape(b * s, HC, d),
                      post.reshape(b * s, HC), comb.reshape(b * s, HC * HC).contiguous())
    return y.view(b, s, HC, d)


def mhc_post_mixes_pre(x1, res, post, comb, pmix, fn, eps=1e-20):
    """hc_post + hc_mixes + hc_pre in one kernel."""
    b, s, d = x1.shape
    t = b * s
    y, mixes, li = _PostMixesPreFn.apply(
        x1.reshape(t, d), res.reshape(t, HC, d), post.reshape(t, HC),
        comb.reshape(t, HC * HC).contiguous(), pmix.reshape(t, HC), fn, eps)
    return y.view(b, s, HC, d), mixes.view(b, s, MIX), li.view(b, s, d)


# =====================================================================
# Block patching
# =====================================================================
_MISSING = object()
_INSTALLED = None


def _hc_mixes(self, x, hc_fn, hc_scale, hc_base):
    mixes = mhc_mixes(x, hc_fn, hc_scale, hc_base, self.hc_sinkhorn_iters,
                      self.hc_eps, self.norm_eps)
    return mhc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_sinkhorn_iters,
                              self.hc_eps)


def _hc_pre(self, x, pre_mix):
    return mhc_pre(x, pre_mix)


def _hc_post(self, x, residual, post, comb):
    return mhc_post(x, residual, post, comb)


def _block_forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """Reference `Block.forward` with the mHC pairs fused (3 kernels per cell)."""
    # cell 1: (hc_mixes, hc_pre) share one read of x
    mixes, li = mhc_mixes_pre(x, self.hc_attn_fn, pre_mix, self.norm_eps)
    attn_pre, attn_post, attn_comb = mhc_split_sinkhorn(
        mixes, self.hc_attn_scale, self.hc_attn_base, self.hc_sinkhorn_iters, self.hc_eps)
    h = self.attn_norm(li)
    h = self.attn(h, start_pos, *attn_args)
    # cell 2: hc_post(attn) + hc_mixes(ffn) + hc_pre in one kernel
    y, mixes2, li2 = mhc_post_mixes_pre(h, x, attn_post, attn_comb, attn_pre,
                                        self.hc_ffn_fn, self.norm_eps)
    ffn_pre, ffn_post, ffn_comb = mhc_split_sinkhorn(
        mixes2, self.hc_ffn_scale, self.hc_ffn_base, self.hc_sinkhorn_iters, self.hc_eps)
    h2 = self.ffn_norm(li2)
    h2 = self.ffn(h2, image_mask)
    y2 = mhc_post(h2, y, ffn_post, ffn_comb)
    return y2, ffn_pre


_METHODS = {"hc_mixes": _hc_mixes, "hc_pre": _hc_pre, "hc_post": _hc_post}


def _resolve_block(target):
    if target is None:
        from .patched_env import import_patched
        target = import_patched()
    if isinstance(target, type):
        return target
    block = getattr(target, "Block", None)
    if block is None:
        raise TypeError("fused_mhc_v3.install() expects the patched model module or Block")
    return block


def install(target=None, *, fused_block: bool = True) -> dict:
    """Patch `Block.hc_mixes/hc_pre/hc_post` (+ `forward` when fused_block). Idempotent."""
    global _INSTALLED
    block = _resolve_block(target)
    if _INSTALLED is not None:
        if _INSTALLED["cls"] is not block:
            uninstall()
        else:
            if fused_block and not _INSTALLED["forward_patched"]:
                prev = _INSTALLED["prev"]
                prev["forward"] = block.__dict__.get("forward", _MISSING)
                block.forward = _block_forward
                _INSTALLED["forward_patched"] = True
            return dict(_INSTALLED["prev"])
    prev = {}
    for name, fn in _METHODS.items():
        prev[name] = block.__dict__.get(name, _MISSING)
        setattr(block, name, fn)
    forward_patched = False
    if fused_block:
        prev["forward"] = block.__dict__.get("forward", _MISSING)
        block.forward = _block_forward
        forward_patched = True
    _INSTALLED = {"cls": block, "prev": prev, "forward_patched": forward_patched}
    return dict(prev)


def uninstall(target=None) -> None:
    global _INSTALLED
    if _INSTALLED is None:
        return
    if target is not None and _resolve_block(target) is not _INSTALLED["cls"]:
        return
    block = _INSTALLED["cls"]
    for name, old in _INSTALLED["prev"].items():
        if old is _MISSING:
            try:
                delattr(block, name)
            except AttributeError:
                pass
        else:
            setattr(block, name, old)
    _INSTALLED = None


def restore(target, prev: dict) -> None:
    uninstall(target)


def get_stats() -> dict:
    return {"installed": _INSTALLED is not None,
            "block": None if _INSTALLED is None else _INSTALLED["cls"].__name__,
            "fused_block": None if _INSTALLED is None else _INSTALLED["forward_patched"]}


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
