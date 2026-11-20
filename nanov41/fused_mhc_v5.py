"""fused_mhc_v5 — one kernel per mHC cell boundary (hc_post + hc_mixes + hc_pre + Sinkhorn).

## Why (measured, 2026-11-10 attribution v4, production env, b1/s8192, fp8 head)

`runs/local-notes/trace_attrib_v4.json` gives the mHC cost per 8192-token replay:

    module (annotation)                  ms/8k     what it is
    mhc_mixes_pre_fwd                    0.567    _mixes_pre_fwd_kernel
    mhc_post_fwd                         0.769    _post_fwd_kernel
    mhc_mixes_pre_bwd                    1.634    _mixes_pre_bwd_kernel + dfn GEMM + .to copy
    mhc_post_bwd                         1.421    _post_bwd_kernel
    ------------------------------------------------------------------
    annotated mHC family                 4.391
    unannotated mHC boundary grad add    0.712    24x bf16 add, one per cell
    _split_sinkhorn_fwd + bwd            0.461    2 launch-bound kernels per cell
    ------------------------------------------------------------------
    real mHC cost                        5.56 ms/8k

Two structural wastes, both visible in that table:

1. **The residual stream is walked twice per boundary, and its gradient is
   materialised twice plus added.**  Per cell the four kernels move
   `3*K + 2*D` residual bytes where `K = hc*d` (forward reads `res` twice: once in
   `_mixes_pre`, once in `_post`), and the backward writes `dx` and `dres` into two
   different buffers that autograd then sums with a third full-size kernel
   (measured 0.712 ms/8k, 24 calls).  A cell only needs one read of `res` forward and
   one write of `dRes` backward.
2. **Sinkhorn's 20-iteration reverse walk lives in its own kernel** that moves 4 KB
   per call: 0.461 ms/8k is launch + ALU on an otherwise idle stream.

## What this module does

`fused_mhc_v5` treats a cell boundary as one autograd `Function`:

    inputs   x1 [T,D] bf16 (sublayer output), res [T,hc,d] bf16 (residual stream),
             post [T,hc] fp32, comb [T,hc*hc] fp32   (coefficients of the *previous* cell's
             Sinkhorn -- what hc_post consumes),
             pmix [T,hc] fp32 (hc_pre weights), fn [MIX,K] fp32 (next cell's mixes projection)
    outputs  y [T,hc,d] bf16  (= hc_post(x1, res)),  li [T,D] bf16 (hc_pre collapse),
             pre/post2/comb2   (Sinkhorn of mixes(y), consumed by the *next* hc_post)

Forward is one kernel (`_cell_fwd_kernel`): build y once, store it, and feed the *same*
bf16-rounded y into the norm statistic, the hc_pre collapse, the mixes projection and the
Sinkhorn -- no second read of the residual, no `mixes` round-trip.
Backward is one kernel (`_cell_bwd_kernel`): rebuild the Sinkhorn reverse walk in registers,
form `dY = gy + du@fn + 2*dss*y + pmix*dli` **in registers** (never stored), and push it
through hc_post in the same pass, writing `dx1`, `dres`, `dpost`, `dcomb`, `dpre`, `du`.

Autograd consequence: `y` has exactly one external consumer (the next hc_post), so the
24 full-size gradient accumulations per replay disappear.  Measured byte counts
(`fwd_bytes`/`bwd_bytes`, T=8192, D=512, hc=4, K=hc*d=2048):

    one cell                        forward    backward    total
    v3 split path (today)           125.2 MB   337.6 MB    462.8 MB  (+2 launch-bound reduces)
    v5 (y loaded)                    90.8 MB   200.4 MB    291.2 MB   -37%
    v5 (NANOV41_V5_BWD_RECOMPUTE=1)  90.8 MB   166.9 MB    257.7 MB   -44%

24 cells per replay, 21 fused (block 0's first cell, the two Engram-layer boundaries and the
final hc_post stay on the proven v3 kernels):

    21 x 257.7 MB + 3 x 462.8 MB = 6.80 GB/replay vs 24 x 462.8 MB = 11.11 GB   (-39%)

`tools/bench_mhc_v5.py` measures fwd/bwd per kernel in isolation (ms/8k + achieved GB/s against
the 3.35 TB/s HBM roofline), compares every output with the v3 composition, checks a CUDA-graph
capture/replay and validates the backward algebra on CPU (`--cpu-check`, PASS at 2e-7 rel).

## Numerics

* `y` uses the *same* fp32 expression, in the same order, as v3's `_post_fwd_kernel`
  (`post_j*x + c0*r0 + c1*r1 + c2*r2 + c3*r3`, one bf16 round), so it is expected to be
  **bitwise identical**; `tools/bench_mhc_v5.py --check` asserts that.
* `li`, `mixes`, `pre/post/comb` follow v3's arithmetic (the bf16-rounded `y` feeds the norm,
  the collapse and the projection) but the fp32 reduction order differs -> last-bit diffs.
* Backward folds two separate fp32 accumulations into one, so grads differ at the fp32
  re-association level (the v3-vs-v4 class): loss rel ~1e-7, worst-parameter grad ~1e-6..1e-3
  relative.  Run a 3-step parity against `NANOV41_V4_MODE=split` before flipping a default.

## Env knobs (all default OFF; opt-in `NANOV41_MHC_V5=1` or `NANOV41_V4_MODE=v5`)

    NANOV41_MHC_V5=1              install (driver patches Transformer.forward)
    NANOV41_V4_MODE=v5            same, through the existing v4 install hook
    NANOV41_V4_MODE=v5block       only Block.forward patched: the mid-block cell is fused,
                                  block-boundary cells stay split (no loop rewrite)
    NANOV41_V5_FWD_SINGLE=1       0 -> keep v3's two proven forward kernels inside the same
                                  Function (fuses the backward only; safest A/B step)
    NANOV41_V5_SINKHORN=1         0 -> keep v3's separate Sinkhorn fwd/bwd kernels
    NANOV41_V5_BWD_RECOMPUTE=0    1 -> rebuild y in the backward instead of loading it
                                  (saves one K*2 read per cell; needs bit-exact recompute)
    NANOV41_V5_BM / _BK / _W_FWD / _S_FWD / _BM_BWD / _BK_BWD / _W_BWD / _S_BWD
                                  tile / warps / stages (defaults 16 / 64 / 4 / 2 | 16 / 64 / 4 / 2)
    NANOV41_V5_DRIVER=1           0 -> do not patch Transformer.forward

## Guards / fallbacks

`install()` falls back to the original `Transformer.forward` for image/token_type inputs, VLM
weights and MTP/DSpark target layers (a pending hc_post cannot cross an `h.mean(dim=2)`
snapshot).  Engram layers are handled: the pending post is materialised before the engram write
(2 layers in aligned d12 -> 2 of 24 cells stay split).
"""
from __future__ import annotations

import os

import torch

from . import fused_mhc_v3 as v3

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = [
    "mhc_cell", "install", "uninstall", "available", "get_stats",
    "fwd_bytes", "bwd_bytes", "set_mode",
]

HC = 4
MIX = 24
MIXP = 32
SINK_ITERS = 20


def _flag(name, default="0"):
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


def _int_env(name, default):
    return int(os.environ.get(name, str(default)))


# Tile sizes.  The kernels are pure streaming (one read of `res`, one write of `dRes`), so the
# knobs that matter are rows/CTA (grid = T/BM), the hidden tile and the warp count.
BM = _int_env("NANOV41_V5_BM", 16)
BK = _int_env("NANOV41_V5_BK", 64)
W_FWD = _int_env("NANOV41_V5_W_FWD", 4)
S_FWD = _int_env("NANOV41_V5_S_FWD", 2)
BM_BWD = _int_env("NANOV41_V5_BM_BWD", BM)
BK_BWD = _int_env("NANOV41_V5_BK_BWD", BK)
W_BWD = _int_env("NANOV41_V5_W_BWD", 4)
S_BWD = _int_env("NANOV41_V5_S_BWD", 2)

FWD_SINGLE = _flag("NANOV41_V5_FWD_SINGLE", "1")
FUSE_SINKHORN = _flag("NANOV41_V5_SINKHORN", "1")
BWD_RECOMPUTE = _flag("NANOV41_V5_BWD_RECOMPUTE", "0")
# Cheap launch-contract validation (default ON): the kernels index raw pointers assuming exact
# contiguous shapes -- a violated contract is the difference between "works" and "illegal memory
# access".  The checks are a few Python attribute reads per cell (the fwd/bwd calls dominate).
VALIDATE = _flag("NANOV41_V5_VALIDATE", "1")


def _contract(what, x, shape, dtype):
    """Raise ValueError unless x is a contiguous tensor of the exact shape/dtype the kernel assumes."""
    if x is None:
        raise ValueError(f"fused_mhc_v5 contract: {what} is None")
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"fused_mhc_v5 contract: {what}.shape={tuple(x.shape)} != {tuple(shape)}")
    if x.dtype != dtype:
        raise ValueError(f"fused_mhc_v5 contract: {what}.dtype={x.dtype} != {dtype}")
    if not x.is_contiguous():
        raise ValueError(f"fused_mhc_v5 contract: {what} is not contiguous "
                         f"(strides={tuple(x.stride())}, shape={tuple(x.shape)})")


def set_mode(fwd_single=None, sinkhorn=None, recompute=None) -> dict:
    """Process-global variant switch (used by tools/bench_mhc_v5.py for the A/B sweep)."""
    global FWD_SINGLE, FUSE_SINKHORN, BWD_RECOMPUTE
    if fwd_single is not None:
        FWD_SINGLE = bool(fwd_single)
    if sinkhorn is not None:
        FUSE_SINKHORN = bool(sinkhorn)
    if recompute is not None:
        BWD_RECOMPUTE = bool(recompute)
    return {"fwd_single": FWD_SINGLE, "sinkhorn": FUSE_SINKHORN, "recompute": BWD_RECOMPUTE}


# --------------------------------------------------------------------------- byte accounting
def fwd_bytes(t, d, hc=HC, mix=MIX, iters=SINK_ITERS, single=True):
    """Bytes moved by one cell forward (single kernel or v3 post+mixes_pre+sinkhorn)."""
    k = hc * d
    if not single:
        post = t * k * 2 + t * d * 2 + t * k * 2                    # res + x1 + y
        mixes = t * k * 2 + t * mix * 4 + t * d * 2 + t * 4 + mix * k * 4
        sink = 2 * iters * t * hc * 4 + t * (mix + 3) * 4
        return post + mixes + sink
    return (t * k * 2 + t * d * 2 + t * k * 2 + t * d * 2 + t * mix * 4 + t * 4
            + 2 * iters * t * hc * 4 + t * (mix + 3) * 4 + mix * k * 4)


def bwd_bytes(t, d, hc=HC, mix=MIX, iters=SINK_ITERS, recompute=False):
    """Bytes moved by one cell backward (fused kernel + dfn GEMM)."""
    k = hc * d
    read = (t * k * 2 + t * k * 2 + t * d * 2 + t * d * 2 + t * mix * 4 + t * 4
            + 2 * iters * t * hc * 4 + mix * k * 4)
    if not recompute:
        read += t * k * 2                                          # y
    write = t * k * 2 + t * d * 2 + t * mix * 2
    dfn = t * mix * 2 + t * k * 2 + mix * k * 4                     # du + y + dfn out
    return read + write + dfn


if HAS_TRITON:

    # =====================================================================
    # 1. cell forward: hc_post + hc_mixes + hc_pre (+ Sinkhorn) in ONE kernel
    # =====================================================================
    @triton.jit
    def _cell_fwd_kernel(
        Y, LI, MIXES, R, PRE, POST, COMB, MINIT, WS,
        X1, RES, POSTIN, COMBIN, PMIX, FN, SCALE, BASE, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
        EPS: tl.constexpr, NORM_EPS: tl.constexpr, ITERS: tl.constexpr,
        SINK: tl.constexpr,
    ):
        """y_j = post_j*x1 + sum_i comb[i,j]*res_i ; (norm, li, mixes, sinkhorn) from bf16 y."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POSTIN + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POSTIN + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POSTIN + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POSTIN + rows * HC + 3, mask=rmask, other=0.0)
        q0 = tl.load(PMIX + rows * HC + 0, mask=rmask, other=0.0)
        q1 = tl.load(PMIX + rows * HC + 1, mask=rmask, other=0.0)
        q2 = tl.load(PMIX + rows * HC + 2, mask=rmask, other=0.0)
        q3 = tl.load(PMIX + rows * HC + 3, mask=rmask, other=0.0)
        ps = (p0, p1, p2, p3)
        qs = (q0, q1, q2, q3)
        cs = (
            tl.load(COMBIN + rows * (HC * HC) + 0, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 1, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 2, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 3, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 4, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 5, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 6, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 7, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 8, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 9, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 10, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 11, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 12, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 13, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 14, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 15, mask=rmask, other=0.0),
        )
        acc = tl.zeros((BM, MIXP), dtype=tl.float32)
        ss = tl.zeros((BM,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r0 = tl.load(RES + rbase + (0 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + (1 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + (2 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + (3 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            rs = (r0, r1, r2, r3)
            li = tl.zeros((BM, BK), dtype=tl.float32)
            for j in tl.static_range(0, HC):
                # identical expression order to v3 `_post_fwd_kernel` -> bitwise-identical y
                o = ps[j][:, None] * x1
                for i in tl.static_range(0, HC):
                    o += cs[i * HC + j][:, None] * rs[i]
                b = o.to(tl.bfloat16)
                bf = b.to(tl.float32)
                tl.store(Y + rbase + (j * D + cols)[None, :], b, mask=rmask[:, None])
                ss += tl.sum(bf * bf, axis=1)
                li += qs[j][:, None] * bf
                w = tl.load(FN + rc[:, None] * K + (j * D + cols)[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                acc = tl.dot(b, tl.trans(w).to(tl.bfloat16), acc)
            tl.store(LI + rows[:, None] * D + cols[None, :],
                     li.to(LI.dtype.element_ty), mask=rmask[:, None])
        r = tl.rsqrt(ss / K + NORM_EPS)
        mm = acc * r[:, None]
        tl.store(MIXES + rows[:, None] * MIX + rc[None, :], mm,
                 mask=rmask[:, None] & (rc[None, :] < MIX))
        tl.store(R + rows, r, mask=rmask)
        if SINK:
            # The Sinkhorn walk consumes the `mixes` we just wrote to MIXES.  Triton gives no
            # ordering between global stores and later global loads (the (BM,MIXP) store layout
            # and the (BM,HC) load layout assign different threads to the same element), so a
            # CTA-wide barrier is required before re-reading them.  Without it the Sinkhorn
            # inputs are stale/undefined.
            tl.debug_barrier()
            # ---- split + row softmax + Sinkhorn (same walk as v3 `_split_sinkhorn_fwd_kernel`)
            rh = tl.arange(0, HC)
            s0 = tl.load(SCALE + 0)
            s1 = tl.load(SCALE + 1)
            s2 = tl.load(SCALE + 2)
            b0 = tl.load(BASE + rh)
            b1 = tl.load(BASE + HC + rh)
            m0 = tl.load(MIXES + rows[:, None] * MIX + rh[None, :], mask=rmask[:, None],
                         other=0.0)
            pre = 1.0 / (1.0 + tl.exp(-(m0 * s0 + b0[None, :]))) + EPS
            tl.store(PRE + rows[:, None] * HC + rh[None, :], pre, mask=rmask[:, None])
            m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rh)[None, :],
                         mask=rmask[:, None], other=0.0)
            post = 2.0 / (1.0 + tl.exp(-(m1 * s1 + b1[None, :])))
            tl.store(POST + rows[:, None] * HC + rh[None, :], post, mask=rmask[:, None])
            m2_0 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 0 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_1 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 1 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 2 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_3 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 3 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            r0 = m2_0 * s2 + tl.load(BASE + 2 * HC + 0 * HC + rh)[None, :]
            r1 = m2_1 * s2 + tl.load(BASE + 2 * HC + 1 * HC + rh)[None, :]
            r2 = m2_2 * s2 + tl.load(BASE + 2 * HC + 2 * HC + rh)[None, :]
            r3 = m2_3 * s2 + tl.load(BASE + 2 * HC + 3 * HC + rh)[None, :]
            t0 = tl.exp2((r0 - tl.max(r0, axis=1)[:, None]) * 1.4426950408889634)
            t1 = tl.exp2((r1 - tl.max(r1, axis=1)[:, None]) * 1.4426950408889634)
            t2 = tl.exp2((r2 - tl.max(r2, axis=1)[:, None]) * 1.4426950408889634)
            t3 = tl.exp2((r3 - tl.max(r3, axis=1)[:, None]) * 1.4426950408889634)
            q0 = t0 / tl.sum(t0, axis=1)[:, None]
            q1 = t1 / tl.sum(t1, axis=1)[:, None]
            q2 = t2 / tl.sum(t2, axis=1)[:, None]
            q3 = t3 / tl.sum(t3, axis=1)[:, None]
            mb = MINIT + rows[:, None] * (HC * HC) + rh[None, :]
            tl.store(mb + 0 * HC, q0, mask=rmask[:, None])
            tl.store(mb + 1 * HC, q1, mask=rmask[:, None])
            tl.store(mb + 2 * HC, q2, mask=rmask[:, None])
            tl.store(mb + 3 * HC, q3, mask=rmask[:, None])
            a0 = q0 + EPS
            a1 = q1 + EPS
            a2 = q2 + EPS
            a3 = q3 + EPS
            csum = a0 + a1 + a2 + a3
            wb = WS + rows[:, None] * HC + rh[None, :]
            tl.store(wb, csum, mask=rmask[:, None])
            cse = csum + EPS
            b0v = a0 / cse
            b1v = a1 / cse
            b2v = a2 / cse
            b3v = a3 / cse
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
                tl.store(WS + t * T * HC + rows[:, None] * HC + rh[None, :], csum2,
                         mask=rmask[:, None])
                dse = csum2 + EPS
                b0v = c0v / dse
                b1v = c1v / dse
                b2v = c2v / dse
                b3v = c3v / dse
            cb = COMB + rows[:, None] * (HC * HC) + rh[None, :]
            tl.store(cb + 0 * HC, b0v, mask=rmask[:, None])
            tl.store(cb + 1 * HC, b1v, mask=rmask[:, None])
            tl.store(cb + 2 * HC, b2v, mask=rmask[:, None])
            tl.store(cb + 3 * HC, b3v, mask=rmask[:, None])

    # =====================================================================
    # 2. cell backward: hc_post + hc_mixes (+ Sinkhorn reverse walk) in ONE kernel
    # =====================================================================
    @triton.jit
    def _cell_bwd_kernel(
        DX1, DRES, DPOST, DCOMB, DPRE, DU, PART,
        GY, DLI, GPRE, GPOST, GCOMB,
        X1, RES, Y, MIXES, R, POSTIN, COMBIN, PMIX, FN,
        DMIX_IN, PRE, POST, COMB, MINIT, WS, SCALE, BASE, T,
        D: tl.constexpr, K: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
        MIXP: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
        EPS: tl.constexpr, ITERS: tl.constexpr, SINK: tl.constexpr,
        RECOMPUTE_Y: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rc = tl.arange(0, MIXP)
        rk = tl.arange(0, BK)
        rh = tl.arange(0, HC)
        rc16 = tl.arange(0, HC * HC)
        rbase = rows[:, None] * (HC * D)
        r = tl.load(R + rows, mask=rmask, other=1.0)
        if SINK:
            # ---- Sinkhorn analytic VJP (v3 `_split_sinkhorn_bwd_kernel`), kept in registers
            gb = GCOMB + rows[:, None] * (HC * HC) + rh[None, :]
            G0 = tl.load(gb + 0 * HC, mask=rmask[:, None], other=0.0)
            G1 = tl.load(gb + 1 * HC, mask=rmask[:, None], other=0.0)
            G2 = tl.load(gb + 2 * HC, mask=rmask[:, None], other=0.0)
            G3 = tl.load(gb + 3 * HC, mask=rmask[:, None], other=0.0)
            cb = COMB + rows[:, None] * (HC * HC) + rh[None, :]
            M0 = tl.load(cb + 0 * HC, mask=rmask[:, None], other=0.0)
            M1 = tl.load(cb + 1 * HC, mask=rmask[:, None], other=0.0)
            M2 = tl.load(cb + 2 * HC, mask=rmask[:, None], other=0.0)
            M3 = tl.load(cb + 3 * HC, mask=rmask[:, None], other=0.0)
            for k in range(0, ITERS - 1):
                t = ITERS - 1 - k
                c = tl.load(WS + t * T * HC + rows[:, None] * HC + rh[None, :],
                            mask=rmask[:, None], other=1.0)
                ce = c + EPS
                G0 = G0 / ce
                G1 = G1 / ce
                G2 = G2 / ce
                G3 = G3 / ce
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
            c0 = tl.load(WS + rows[:, None] * HC + rh[None, :], mask=rmask[:, None],
                         other=1.0)
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
            mb = MINIT + rows[:, None] * (HC * HC) + rh[None, :]
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
            m0 = tl.load(MIXES + rows[:, None] * MIX + rh[None, :], mask=rmask[:, None],
                         other=0.0)
            m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rh)[None, :],
                         mask=rmask[:, None], other=0.0)
            gpre = tl.load(GPRE + rows[:, None] * HC + rh[None, :], mask=rmask[:, None],
                           other=0.0)
            gpost = tl.load(GPOST + rows[:, None] * HC + rh[None, :], mask=rmask[:, None],
                            other=0.0)
            pre = tl.load(PRE + rows[:, None] * HC + rh[None, :], mask=rmask[:, None],
                          other=0.0)
            post = tl.load(POST + rows[:, None] * HC + rh[None, :], mask=rmask[:, None],
                           other=0.0)
            sig0 = pre - EPS
            dz0 = gpre * sig0 * (1.0 - sig0)
            sig1 = post * 0.5
            dz1 = gpost * 2.0 * sig1 * (1.0 - sig1)
            dm0 = dz0 * s0
            dm1 = dz1 * s1
            dm2_0 = G0 * s2
            dm2_1 = G1 * s2
            dm2_2 = G2 * s2
            dm2_3 = G3 * s2
            # partials for hc_*_base / hc_*_scale: ONE buffer, ONE reduction for the whole cell
            pbase = PART + pid * (MIX + 4) + rh
            tl.store(pbase + 0 * HC, tl.sum(dz0, axis=0))
            tl.store(pbase + 1 * HC, tl.sum(dz1, axis=0))
            tl.store(pbase + 2 * HC, tl.sum(G0, axis=0))
            tl.store(pbase + 3 * HC, tl.sum(G1, axis=0))
            tl.store(pbase + 4 * HC, tl.sum(G2, axis=0))
            tl.store(pbase + 5 * HC, tl.sum(G3, axis=0))
            dsp = PART + pid * (MIX + 4) + MIX + tl.arange(0, 4)
            m2_0 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 0 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_1 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 1 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 2 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            m2_3 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 3 * HC + rh)[None, :],
                           mask=rmask[:, None], other=0.0)
            ds0 = tl.sum(dz0 * m0)
            ds1 = tl.sum(dz1 * m1)
            ds2 = (tl.sum(G0 * m2_0) + tl.sum(G1 * m2_1)
                   + tl.sum(G2 * m2_2) + tl.sum(G3 * m2_3))
            ds = tl.where(tl.arange(0, 4) == 0, ds0,
                          tl.where(tl.arange(0, 4) == 1, ds1,
                                   tl.where(tl.arange(0, 4) == 2, ds2, 0.0)))
            tl.store(dsp, ds, mask=tl.arange(0, 4) < 3)
        else:
            dm0 = tl.load(DMIX_IN + rows[:, None] * MIX + rh[None, :], mask=rmask[:, None],
                          other=0.0)
            dm1 = tl.load(DMIX_IN + rows[:, None] * MIX + (HC + rh)[None, :],
                          mask=rmask[:, None], other=0.0)
            dm2_0 = tl.load(DMIX_IN + rows[:, None] * MIX + (2 * HC + 0 * HC + rh)[None, :],
                            mask=rmask[:, None], other=0.0)
            dm2_1 = tl.load(DMIX_IN + rows[:, None] * MIX + (2 * HC + 1 * HC + rh)[None, :],
                            mask=rmask[:, None], other=0.0)
            dm2_2 = tl.load(DMIX_IN + rows[:, None] * MIX + (2 * HC + 2 * HC + rh)[None, :],
                            mask=rmask[:, None], other=0.0)
            dm2_3 = tl.load(DMIX_IN + rows[:, None] * MIX + (2 * HC + 3 * HC + rh)[None, :],
                            mask=rmask[:, None], other=0.0)
        # ---- hc_mixes backward: u = mixes/r, dss, du  (v3 `_mixes_pre_bwd_kernel`)
        mm0 = tl.load(MIXES + rows[:, None] * MIX + rh[None, :], mask=rmask[:, None],
                      other=0.0)
        mm1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rh)[None, :], mask=rmask[:, None],
                      other=0.0)
        mm2_0 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 0 * HC + rh)[None, :],
                        mask=rmask[:, None], other=0.0)
        mm2_1 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 1 * HC + rh)[None, :],
                        mask=rmask[:, None], other=0.0)
        mm2_2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 2 * HC + rh)[None, :],
                        mask=rmask[:, None], other=0.0)
        mm2_3 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + 3 * HC + rh)[None, :],
                        mask=rmask[:, None], other=0.0)
        u0 = mm0 / r[:, None]
        u1 = mm1 / r[:, None]
        u2_0 = mm2_0 / r[:, None]
        u2_1 = mm2_1 / r[:, None]
        u2_2 = mm2_2 / r[:, None]
        u2_3 = mm2_3 / r[:, None]
        dot_sum = (tl.sum(dm0 * u0, axis=1) + tl.sum(dm1 * u1, axis=1)
                   + tl.sum(dm2_0 * u2_0, axis=1) + tl.sum(dm2_1 * u2_1, axis=1)
                   + tl.sum(dm2_2 * u2_2, axis=1) + tl.sum(dm2_3 * u2_3, axis=1))
        dss = -0.5 * r * r * r / K * dot_sum
        du0 = dm0 * r[:, None]
        du1 = dm1 * r[:, None]
        du2_0 = dm2_0 * r[:, None]
        du2_1 = dm2_1 * r[:, None]
        du2_2 = dm2_2 * r[:, None]
        du2_3 = dm2_3 * r[:, None]
        dub0 = DU + rows[:, None] * MIX + rh[None, :]
        tl.store(dub0 + 0 * HC, du0.to(DU.dtype.element_ty), mask=rmask[:, None])
        tl.store(dub0 + 1 * HC, du1.to(DU.dtype.element_ty), mask=rmask[:, None])
        tl.store(dub0 + 2 * HC + 0 * HC, du2_0.to(DU.dtype.element_ty), mask=rmask[:, None])
        tl.store(dub0 + 2 * HC + 1 * HC, du2_1.to(DU.dtype.element_ty), mask=rmask[:, None])
        tl.store(dub0 + 2 * HC + 2 * HC, du2_2.to(DU.dtype.element_ty), mask=rmask[:, None])
        tl.store(dub0 + 2 * HC + 3 * HC, du2_3.to(DU.dtype.element_ty), mask=rmask[:, None])
        # du was just written to DU; dY below re-reads it as the bf16 dot operand.  Same-CTA
        # global read-after-write needs a barrier (the (BM,HC)-shaped stores and the (BM,MIXP)
        # load do not share a thread mapping).
        tl.debug_barrier()
        dub = tl.load(DU + rows[:, None] * MIX + rc[None, :],
                      mask=rmask[:, None] & (rc[None, :] < MIX), other=0.0)
        # ---- hc_post backward + hc_mixes input gradient, one pass over D
        p0 = tl.load(POSTIN + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POSTIN + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POSTIN + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POSTIN + rows * HC + 3, mask=rmask, other=0.0)
        q0 = tl.load(PMIX + rows * HC + 0, mask=rmask, other=0.0)
        q1 = tl.load(PMIX + rows * HC + 1, mask=rmask, other=0.0)
        q2 = tl.load(PMIX + rows * HC + 2, mask=rmask, other=0.0)
        q3 = tl.load(PMIX + rows * HC + 3, mask=rmask, other=0.0)
        ps = (p0, p1, p2, p3)
        qs = (q0, q1, q2, q3)
        cs = (
            tl.load(COMBIN + rows * (HC * HC) + 0, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 1, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 2, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 3, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 4, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 5, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 6, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 7, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 8, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 9, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 10, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 11, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 12, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 13, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 14, mask=rmask, other=0.0),
            tl.load(COMBIN + rows * (HC * HC) + 15, mask=rmask, other=0.0),
        )
        dpost = tl.zeros((BM, HC), dtype=tl.float32)
        dpre = tl.zeros((BM, HC), dtype=tl.float32)
        dcomb = tl.zeros((BM, HC * HC), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            gl = tl.load(DLI + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            v0 = tl.load(RES + rbase + (0 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            v1 = tl.load(RES + rbase + (1 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            v2 = tl.load(RES + rbase + (2 * D + cols)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            v3t = tl.load(RES + rbase + (3 * D + cols)[None, :], mask=rmask[:, None],
                          other=0.0).to(tl.float32)
            vs = (v0, v1, v2, v3t)
            dx1 = tl.zeros((BM, BK), dtype=tl.float32)
            dr0 = tl.zeros((BM, BK), dtype=tl.float32)
            dr1 = tl.zeros((BM, BK), dtype=tl.float32)
            dr2 = tl.zeros((BM, BK), dtype=tl.float32)
            dr3 = tl.zeros((BM, BK), dtype=tl.float32)
            for j in tl.static_range(0, HC):
                if RECOMPUTE_Y:
                    o = ps[j][:, None] * x1
                    for i in tl.static_range(0, HC):
                        o += cs[i * HC + j][:, None] * vs[i]
                    yj = o.to(tl.bfloat16).to(tl.float32)
                else:
                    yj = tl.load(Y + rbase + (j * D + cols)[None, :], mask=rmask[:, None],
                                 other=0.0).to(tl.float32)
                g = tl.load(GY + rbase + (j * D + cols)[None, :], mask=rmask[:, None],
                            other=0.0).to(tl.float32)
                w = tl.load(FN + rc[:, None] * K + (j * D + cols)[None, :],
                            mask=rc[:, None] < MIX, other=0.0)
                dy = (g + tl.dot(dub, w.to(tl.bfloat16)) + 2.0 * dss[:, None] * yj
                      + qs[j][:, None] * gl)
                dx1 += ps[j][:, None] * dy
                for i in tl.static_range(0, HC):
                    dri = cs[i * HC + j][:, None] * dy
                    if i == 0:
                        dr0 += dri
                    elif i == 1:
                        dr1 += dri
                    elif i == 2:
                        dr2 += dri
                    else:
                        dr3 += dri
                    dcomb += tl.where(rc16[None, :] == i * HC + j,
                                      tl.sum(dy * vs[i], axis=1)[:, None], 0.0)
                dpost += tl.where(rh[None, :] == j, tl.sum(dy * x1, axis=1)[:, None], 0.0)
                dpre += tl.where(rh[None, :] == j, tl.sum(gl * yj, axis=1)[:, None], 0.0)
            tl.store(DX1 + rows[:, None] * D + cols[None, :], dx1.to(DX1.dtype.element_ty),
                     mask=rmask[:, None])
            tl.store(DRES + rbase + (0 * D + cols)[None, :], dr0.to(DRES.dtype.element_ty),
                     mask=rmask[:, None])
            tl.store(DRES + rbase + (1 * D + cols)[None, :], dr1.to(DRES.dtype.element_ty),
                     mask=rmask[:, None])
            tl.store(DRES + rbase + (2 * D + cols)[None, :], dr2.to(DRES.dtype.element_ty),
                     mask=rmask[:, None])
            tl.store(DRES + rbase + (3 * D + cols)[None, :], dr3.to(DRES.dtype.element_ty),
                     mask=rmask[:, None])
        tl.store(DPOST + rows[:, None] * HC + rh[None, :], dpost, mask=rmask[:, None])
        tl.store(DPRE + rows[:, None] * HC + rh[None, :], dpre, mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + rc16[None, :], dcomb, mask=rmask[:, None])


# =====================================================================
# launchers
# =====================================================================
def _cdiv(a, b):
    return (a + b - 1) // b


def _require(x):
    if not HAS_TRITON:
        raise RuntimeError("fused_mhc_v5 requires Triton + CUDA")
    if not x.is_cuda:
        raise RuntimeError("fused_mhc_v5 requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"fused_mhc_v5 requires bf16 activations, got {x.dtype}")
    if x.shape[-2] != HC:
        raise RuntimeError(f"fused_mhc_v5 requires hc_mult={HC}, got {x.shape[-2]}")


def _g_or_zeros(g, ref):
    return g if g is not None else torch.zeros_like(ref)


def _launch_fwd(x1, res, post, comb, pmix, fn, scale, base, iters, eps, norm_eps):
    """One cell forward -> y, li, mixes, r, pre, post2, comb2, minit, ws."""
    t, hc, d = res.shape
    k = hc * d
    dev = res.device
    if not FWD_SINGLE:
        # proven v3 kernels, but under this Function: the backward is still the fused one
        y = v3._launch_post(x1, res, post, comb)
        mixes, li, r = v3._launch_mixes_pre(y.reshape(t, k), fn, pmix, True, norm_eps)
        pre, post2, comb2, minit, ws = v3._launch_split_sinkhorn(
            mixes, scale, base, iters, eps)
        return y, li, mixes, r, pre, post2, comb2, minit, ws

    y = torch.empty_like(res)
    li = torch.empty(t, d, dtype=res.dtype, device=dev)
    mixes = torch.empty(t, MIX, dtype=torch.float32, device=dev)
    r = torch.empty(t, dtype=torch.float32, device=dev)
    if FUSE_SINKHORN:
        pre = torch.empty(t, HC, dtype=torch.float32, device=dev)
        post2 = torch.empty(t, HC, dtype=torch.float32, device=dev)
        comb2 = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
        minit = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
        ws = torch.empty(2, iters, t, HC, dtype=torch.float32, device=dev)
    else:
        # fwd single but Sinkhorn separate: run it after the cell kernel
        pre = post2 = comb2 = minit = ws = mixes
    _cell_fwd_kernel[(_cdiv(t, BM),)](
        y, li, mixes, r, pre, post2, comb2, minit, ws,
        x1, res, post, comb, pmix, fn, scale, base, t,
        D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK, BM=BM, EPS=eps,
        NORM_EPS=norm_eps, ITERS=iters, SINK=FUSE_SINKHORN,
        num_warps=W_FWD, num_stages=S_FWD,
    )
    if not FUSE_SINKHORN:
        pre, post2, comb2, minit, ws = v3._launch_split_sinkhorn(
            mixes, scale, base, iters, eps)
    return y, li, mixes, r, pre, post2, comb2, minit, ws


def _launch_bwd(y, res, x1, mixes, r, post, comb, pmix, fn,
                 pre, post2, comb2, minit, ws, scale, base,
                 gy, dli, gpre, gpost, gcomb, iters, eps):
    """One cell backward (fused kernel) + dfn GEMM -> all input grads."""
    t, hc, d = res.shape
    k = hc * d
    dev = res.device
    nprog = _cdiv(t, BM_BWD)
    dx1 = torch.empty(t, d, dtype=x1.dtype, device=dev)
    dres = torch.empty_like(res)
    dpost = torch.empty(t, HC, dtype=torch.float32, device=dev)
    dcomb = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
    dpre = torch.empty(t, HC, dtype=torch.float32, device=dev)
    du = torch.empty(t, MIX, dtype=torch.bfloat16, device=dev)
    part = torch.empty(nprog, MIX + 4, dtype=torch.float32, device=dev)
    if FUSE_SINKHORN:
        dmix = du                        # unused pointer in this mode (walk is in-kernel)
        db = ds = None
    else:
        # v3 Sinkhorn VJP kernel first (it owns the 20-iteration walk), then the fused cell bwd
        dmix = torch.empty(t, MIX, dtype=torch.float32, device=dev)
        db = torch.empty(nprog, MIX, dtype=torch.float32, device=dev)
        ds = torch.empty(nprog, 3, dtype=torch.float32, device=dev)
        # `_split_sinkhorn_bwd_kernel` is (MIXES, SCALE, PRE, POST, COMB, MINIT, WS, GPRE, GPOST,
        # GCOMB, DMIX, DB_PART, DS_PART, T) -- **no BASE** (R76 bench "ITERS bug": the extra
        # `base` shifted t onto the ITERS constexpr slot -> "got multiple values for 'ITERS'").
        v3._split_sinkhorn_bwd_kernel[(nprog,)](
            mixes, scale, pre, post2, comb2, minit, ws,
            gpre.contiguous(), gpost.contiguous(), gcomb.contiguous(),
            dmix, db, ds, t, ITERS=iters, HC=HC, MIX=MIX, BM=BM_BWD, EPS=eps,
            num_warps=v3._w("NANOV41_V3_W_SS", 4),
        )
    _cell_bwd_kernel[(nprog,)](
        dx1, dres, dpost, dcomb, dpre, du, part,
        gy, dli, gpre, gpost, gcomb,
        x1, res, y, mixes, r, post, comb, pmix, fn,
        dmix, pre, post2, comb2, minit, ws, scale, base, t,
        D=d, K=k, HC=HC, MIX=MIX, MIXP=MIXP, BK=BK_BWD, BM=BM_BWD,
        EPS=eps, ITERS=iters, SINK=FUSE_SINKHORN, RECOMPUTE_Y=BWD_RECOMPUTE,
        num_warps=W_BWD, num_stages=S_BWD,
    )
    if FUSE_SINKHORN:
        red = part.sum(0)                # ONE reduction per cell (v3 needed two)
        dbase = red[:MIX]
        dscale = red[MIX:MIX + 3]
    else:
        dbase = db.sum(0)
        dscale = ds.sum(0)
    return dx1, dres, dpost, dcomb, dpre, du, dbase, dscale


# =====================================================================
# autograd Function
# =====================================================================
class _CellFn(torch.autograd.Function):
    """hc_post + hc_mixes (+ hc_pre, Sinkhorn): one forward kernel, one backward kernel."""

    @staticmethod
    def forward(ctx, x1, res, post, comb, pmix, fn, scale, base, iters, eps, norm_eps):
        _require(res)
        t, hc, d = res.shape
        x1 = x1.reshape(t, d)
        post = post.reshape(t, HC)
        comb = comb.reshape(t, HC * HC).contiguous()
        pmix = pmix.reshape(t, HC)
        if VALIDATE:
            k = HC * d
            _contract("x1", x1, (t, d), torch.bfloat16)
            _contract("res", res, (t, HC, d), torch.bfloat16)
            _contract("post", post, (t, HC), torch.float32)
            _contract("comb", comb, (t, HC * HC), torch.float32)
            _contract("pmix", pmix, (t, HC), torch.float32)
            _contract("fn", fn, (MIX, k), torch.float32)
            _contract("scale", scale, (3,), torch.float32)
            _contract("base", base, (2 * HC + HC * HC,), torch.float32)
        out = _launch_fwd(x1, res, post, comb, pmix, fn, scale, base,
                          int(iters), float(eps), float(norm_eps))
        y, li, mixes, r, pre, post2, comb2, minit, ws = out
        ctx.save_for_backward(x1, res, post, comb, pmix, fn, y, li, mixes, r,
                              pre, post2, comb2, minit, ws, scale, base)
        ctx.iters, ctx.eps = int(iters), float(eps)
        return y, li, pre, post2, comb2

    @staticmethod
    def backward(ctx, gy, dli, gpre, gpost, gcomb):
        (x1, res, post, comb, pmix, fn, y, li, mixes, r,
         pre, post2, comb2, minit, ws, scale, base) = ctx.saved_tensors
        t, hc, d = res.shape
        k = hc * d
        # the kernels read dense pointers: autograd hands back views for reshaped grads
        gy = _g_or_zeros(gy, y).contiguous()
        dli = _g_or_zeros(dli, li).contiguous()
        gpre = _g_or_zeros(gpre, pre).contiguous()
        gpost = _g_or_zeros(gpost, post2).contiguous()
        gcomb = _g_or_zeros(gcomb, comb2).contiguous()
        if VALIDATE:
            _contract("gy", gy, (t, HC, d), torch.bfloat16)
            _contract("dli", dli, (t, d), torch.bfloat16)
            _contract("gpre", gpre, (t, HC), torch.float32)
            _contract("gpost", gpost, (t, HC), torch.float32)
            _contract("gcomb", gcomb, (t, HC * HC), torch.float32)
        dx1, dres, dpost, dcomb, dpre, du, dbase, dscale = _launch_bwd(
            y, res, x1, mixes, r, post, comb, pmix, fn,
            pre, post2, comb2, minit, ws, scale, base,
            gy, dli, gpre, gpost, gcomb, ctx.iters, ctx.eps)
        # dfn = du^T @ y  (one bf16 cuBLAS GEMM, fp32 accumulate)
        dfn = torch.mm(du.t(), y.reshape(t, k)).to(fn.dtype)
        return dx1, dres, dpost, dcomb, dpre, dfn, dscale, dbase, None, None, None


# =====================================================================
# public op (model shapes: [b, s, ...])
# =====================================================================
def mhc_cell(x1, res, post, comb, pmix, fn, scale, base, iters=20, eps=1e-6, norm_eps=1e-20):
    """hc_post(x1, res) followed by hc_mixes+hc_pre of the *result* -- one autograd Function.

    x1 [b,s,d] bf16, res [b,s,hc,d] bf16, post [b,s,hc] fp32, comb [b,s,hc,hc] fp32,
    pmix [b,s,hc] fp32, fn [MIX,hc*d] fp32 -> y [b,s,hc,d], li [b,s,d], pre/post/comb.
    """
    b, s, hc, d = res.shape
    t = b * s
    y, li, pre, post2, comb2 = _CellFn.apply(
        x1.reshape(t, d), res.reshape(t, hc, d), post.reshape(t, hc),
        comb.reshape(t, hc * hc), pmix.reshape(t, hc), fn, scale, base,
        int(iters), float(eps), float(norm_eps))
    return (y.view(b, s, hc, d), li.view(b, s, d), pre.view(b, s, hc),
            post2.view(b, s, hc), comb2.view(b, s, hc, hc))


# =====================================================================
# block / model patching
# =====================================================================
_ORIG_FORWARD = None
_CLS = None
_PREV_BLOCK_FORWARD = None
_STATE = {"installed": False, "driver": False, "forward_patched": False,
          "fwd_single": FWD_SINGLE, "sinkhorn": FUSE_SINKHORN,
          "calls": 0, "fallbacks": 0}


def _materialise(pend, h):
    """Flush a deferred hc_post (used before engram writes and after the last block)."""
    if pend is None:
        return h
    x1, res, post, comb = pend
    return v3.mhc_post(x1, res, post, comb)


def _cell(layer, x1, res, post, comb, pmix, fn, scale, base):
    return mhc_cell(x1, res, post, comb, pmix, fn, scale, base,
                    layer.hc_sinkhorn_iters, layer.hc_eps, layer.norm_eps)


def _block_start(layer, x, pre_mix):
    """mixes_pre for the attention cell; fused with the pending hc_post when there is one."""
    if isinstance(x, tuple):                      # deferred post from the previous block
        x1p, resp, postp, combp = x
        y, li, apre, apost, acomb = _cell(
            layer, x1p, resp, postp, combp, pre_mix,
            layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        return y, li, apre, apost, acomb
    mixes, li = v3.mhc_mixes_pre(x, layer.hc_attn_fn, pre_mix, layer.norm_eps)
    apre, apost, acomb = v3.mhc_split_sinkhorn(
        mixes, layer.hc_attn_scale, layer.hc_attn_base,
        layer.hc_sinkhorn_iters, layer.hc_eps)
    return x, li, apre, apost, acomb


def _block_body(layer, x, start_pos, pre_mix, image_mask, *attn_args):
    """One Block.forward body with the mid-block cell (attn post + ffn mixes) fused."""
    res, li, attn_pre, attn_post, attn_comb = _block_start(layer, x, pre_mix)
    h = layer.attn_norm(li)
    h = layer.attn(h, start_pos, *attn_args)
    y, li2, ffn_pre, ffn_post, ffn_comb = _cell(
        layer, h, res, attn_post, attn_comb, attn_pre,
        layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
    h2 = layer.ffn_norm(li2)
    h2 = layer.ffn(h2, image_mask)
    return y, ffn_pre, (h2, y, ffn_post, ffn_comb)


def _block_forward_v5(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """Block.forward with the mid-block cell fused (block-boundary cells stay split)."""
    _STATE["calls"] += 1
    y, ffn_pre, pend = _block_body(self, x, start_pos, pre_mix, image_mask, *attn_args)
    return _materialise(pend, y), ffn_pre


def _make_forward(M):
    def forward(self, input_ids, start_pos=0, images=None, token_types=None,
                full_logits=False):
        # ---- guards: only the plain text path with no MTP snapshots is reproduced ----
        if (images is not None or token_types is not None
                or getattr(self, "vision", None) is not None
                or getattr(self, "target_layer_ids", None)):
            _STATE["fallbacks"] += 1
            return _ORIG_FORWARD(self, input_ids, start_pos, images, token_types,
                                 full_logits)
        _STATE["calls"] += 1
        engram_hashes = (self.engram_hash(input_ids, start_pos, None)
                         if self.engram_hash is not None else None)
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, self.hc_mult)
        pend = None
        for layer in self.layers:
            if layer.engram is not None:
                h = _materialise(pend, h)
                pend = None
                h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :],
                                 None)
            # the residual entering this block: the deferred post, or the materialised h
            x = pend if pend is not None else h
            h, pre_mix, pend = _block_body(layer, x, start_pos, pre_mix, None)
        h = _materialise(pend, h)
        h = self.layers[-1].hc_pre(h, pre_mix)
        logits = self.head(self.norm(h), full_logits=full_logits)
        if full_logits:
            return logits
        return M.sample(logits, self.temperature), logits, None

    return forward


def install(M=None, *, driver: bool | None = None) -> dict:
    """Patch `Transformer.forward` (and `Block.forward`) with the fused cell chain.  Idempotent."""
    global _ORIG_FORWARD, _CLS
    if M is None:
        from .patched_env import import_patched
        M = import_patched()
    if not available():
        return {"installed": False, "reason": "triton/cuda unavailable"}
    if driver is None:
        driver = _flag("NANOV41_V5_DRIVER", "1")
    v3.install(M)                                  # hc_* ops used by the split cells
    cls = getattr(M, "Block", None)
    if cls is None:
        raise TypeError("fused_mhc_v5.install() expects the patched model module")
    tcls = getattr(M, "Transformer", None)
    if driver and tcls is None:
        raise TypeError("fused_mhc_v5 driver needs M.Transformer (or pass driver=False)")
    _STATE["fwd_single"] = FWD_SINGLE
    _STATE["sinkhorn"] = FUSE_SINKHORN
    global _PREV_BLOCK_FORWARD
    _PREV_BLOCK_FORWARD = cls.__dict__.get("forward")
    if driver:
        _CLS = tcls
        _ORIG_FORWARD = tcls.forward
        tcls.forward = _make_forward(M)
        _STATE["driver"] = True
    cls.forward = _block_forward_v5              # other entry points (decode/encode/spec)
    _STATE["forward_patched"] = True
    _STATE["installed"] = True
    return dict(_STATE)


def uninstall(M=None) -> None:
    global _ORIG_FORWARD, _CLS, _PREV_BLOCK_FORWARD
    if _CLS is not None and _ORIG_FORWARD is not None:
        _CLS.forward = _ORIG_FORWARD
    if M is None:
        try:
            from .patched_env import import_patched
            M = import_patched()
        except Exception:  # noqa: BLE001
            M = None
    if M is not None and _PREV_BLOCK_FORWARD is not None:
        cls = getattr(M, "Block", None)
        if cls is not None and getattr(cls, "forward", None) is _block_forward_v5:
            cls.forward = _PREV_BLOCK_FORWARD
    _ORIG_FORWARD = None
    _CLS = None
    _STATE["driver"] = False
    _STATE["forward_patched"] = False
    _STATE["installed"] = False


def available() -> bool:
    return v3.available()


def get_stats() -> dict:
    return dict(_STATE)
