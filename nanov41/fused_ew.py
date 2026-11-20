"""fused_ew — R79/T2-fusion: body elementwise + mHC add/copy folding (env-gated, default OFF).

Enabled by ``NANOV41_EW_FUSE=1`` through the hook in ``tools/bench_graph_train.py::build``.
Item selection: ``NANOV41_EW_ITEMS`` (comma list; default = all *validated* items).

Why this file: the post-R77 production mHC path runs through ``nanov41/fused_mhc_v3.py``
(``mhc_post`` / ``mhc_mixes_pre`` wrappers called from ``fused_mhc_v4``'s patched
``Transformer.forward``); ``nanov41/fast_mhc.py`` is no longer on that path.  This module keeps all
new Triton kernels here and swaps them into ``fused_mhc_v3`` module globals at install time, so
nothing outside the R79 scope needs editing.

Items
-----
``post``
    Rewritten mHC `hc_post` fwd/bwd kernels (``fused_mhc_v3._PostFn`` replacement).
    Changes vs v3: ``post``/``comb`` coefficients are loaded **once per row** instead of once per
    (d-chunk x i) and the backward accumulates ``dpost``/``dcomb`` in explicit registers instead of
    ``tl.where``-masked [BM,HC]/[BM,HC*HC] tiles.  The fp32 accumulation **order is unchanged**,
    so outputs/grads are expected to be bitwise identical (G1 gate).
    Tunables: ``NANOV41_EW_POST_BM`` (16), ``NANOV41_EW_POST_BK`` (128), ``NANOV41_EW_POST_WARPS`` (8),
    ``NANOV41_EW_POST_STAGES`` (2).

Default behaviour with ``NANOV41_EW_FUSE`` unset: this module is never imported; nothing changes.
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

__all__ = ["install", "uninstall", "available", "items", "get_state"]

_HC = 4
_V3: dict = {"m": None}
_STATE: dict = {"installed": False, "items": (), "patched": [], "v3": None, "_orig": {}}


def _flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def available() -> bool:
    return bool(HAS_TRITON)


def items() -> tuple[str, ...]:
    """Resolve the enabled item list (``NANOV41_EW_ITEMS``; default = all implemented items)."""
    raw = _flag("NANOV41_EW_ITEMS", "")
    _ALL = ("post", "postmix")
    _DEFAULT = ("postmix",)
    if not raw:
        return _DEFAULT
    want = tuple(x.strip() for x in raw.split(",") if x.strip())
    return tuple(x for x in want if x in _ALL)


# =========================================================================== mHC hc_post (post)
if HAS_TRITON:

    @triton.jit
    def _post_fwd_fused(
        Y, X1, RES, POST, COMB, T,
        D: tl.constexpr, HC: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
    ):
        """y[...,j,:] = post_j*x1 + sum_i comb[i,j]*res[...,i,:]; post/comb loaded once per row."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        cb = rows * (HC * HC)
        c00 = tl.load(COMB + cb + 0, mask=rmask, other=0.0)
        c01 = tl.load(COMB + cb + 1, mask=rmask, other=0.0)
        c02 = tl.load(COMB + cb + 2, mask=rmask, other=0.0)
        c03 = tl.load(COMB + cb + 3, mask=rmask, other=0.0)
        c10 = tl.load(COMB + cb + 4, mask=rmask, other=0.0)
        c11 = tl.load(COMB + cb + 5, mask=rmask, other=0.0)
        c12 = tl.load(COMB + cb + 6, mask=rmask, other=0.0)
        c13 = tl.load(COMB + cb + 7, mask=rmask, other=0.0)
        c20 = tl.load(COMB + cb + 8, mask=rmask, other=0.0)
        c21 = tl.load(COMB + cb + 9, mask=rmask, other=0.0)
        c22 = tl.load(COMB + cb + 10, mask=rmask, other=0.0)
        c23 = tl.load(COMB + cb + 11, mask=rmask, other=0.0)
        c30 = tl.load(COMB + cb + 12, mask=rmask, other=0.0)
        c31 = tl.load(COMB + cb + 13, mask=rmask, other=0.0)
        c32 = tl.load(COMB + cb + 14, mask=rmask, other=0.0)
        c33 = tl.load(COMB + cb + 15, mask=rmask, other=0.0)
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            o0 = p0[:, None] * x1
            o1 = p1[:, None] * x1
            o2 = p2[:, None] * x1
            o3 = p3[:, None] * x1
            # same accumulation order as fused_mhc_v3._post_fwd_kernel (i = 0..3, one load each)
            for i in tl.static_range(0, HC):
                r_i = tl.load(RES + rbase + ((i * D + d0) + rk)[None, :], mask=rmask[:, None],
                              other=0.0).to(tl.float32)
                c0 = (c00, c10, c20, c30)[i]
                c1 = (c01, c11, c21, c31)[i]
                c2 = (c02, c12, c22, c32)[i]
                c3 = (c03, c13, c23, c33)[i]
                o0 += c0[:, None] * r_i
                o1 += c1[:, None] * r_i
                o2 += c2[:, None] * r_i
                o3 += c3[:, None] * r_i
            ob = rows[:, None] * (HC * D) + cols[None, :]
            tl.store(Y + ob + 0 * D, o0.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 1 * D, o1.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 2 * D, o2.to(Y.dtype.element_ty), mask=rmask[:, None])
            tl.store(Y + ob + 3 * D, o3.to(Y.dtype.element_ty), mask=rmask[:, None])

    @triton.jit
    def _post_bwd_fused(
        DX1, DRES, DPOST, DCOMB, GY, X1, RES, POST, COMB, T,
        D: tl.constexpr, HC: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
    ):
        """dx1, dres, dpost, dcomb; same fp32 accumulation order as v3, no masked accumulators."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        rk = tl.arange(0, BK)
        rbase = rows[:, None] * (HC * D)
        p0 = tl.load(POST + rows * HC + 0, mask=rmask, other=0.0)
        p1 = tl.load(POST + rows * HC + 1, mask=rmask, other=0.0)
        p2 = tl.load(POST + rows * HC + 2, mask=rmask, other=0.0)
        p3 = tl.load(POST + rows * HC + 3, mask=rmask, other=0.0)
        cb = rows * (HC * HC)
        c00 = tl.load(COMB + cb + 0, mask=rmask, other=0.0)
        c01 = tl.load(COMB + cb + 1, mask=rmask, other=0.0)
        c02 = tl.load(COMB + cb + 2, mask=rmask, other=0.0)
        c03 = tl.load(COMB + cb + 3, mask=rmask, other=0.0)
        c10 = tl.load(COMB + cb + 4, mask=rmask, other=0.0)
        c11 = tl.load(COMB + cb + 5, mask=rmask, other=0.0)
        c12 = tl.load(COMB + cb + 6, mask=rmask, other=0.0)
        c13 = tl.load(COMB + cb + 7, mask=rmask, other=0.0)
        c20 = tl.load(COMB + cb + 8, mask=rmask, other=0.0)
        c21 = tl.load(COMB + cb + 9, mask=rmask, other=0.0)
        c22 = tl.load(COMB + cb + 10, mask=rmask, other=0.0)
        c23 = tl.load(COMB + cb + 11, mask=rmask, other=0.0)
        c30 = tl.load(COMB + cb + 12, mask=rmask, other=0.0)
        c31 = tl.load(COMB + cb + 13, mask=rmask, other=0.0)
        c32 = tl.load(COMB + cb + 14, mask=rmask, other=0.0)
        c33 = tl.load(COMB + cb + 15, mask=rmask, other=0.0)
        z = tl.zeros((BM,), dtype=tl.float32)
        dpost0 = z
        dpost1 = z
        dpost2 = z
        dpost3 = z
        dc00 = z
        dc01 = z
        dc02 = z
        dc03 = z
        dc10 = z
        dc11 = z
        dc12 = z
        dc13 = z
        dc20 = z
        dc21 = z
        dc22 = z
        dc23 = z
        dc30 = z
        dc31 = z
        dc32 = z
        dc33 = z
        for d0 in tl.static_range(0, D, BK):
            cols = d0 + rk
            gb = rows[:, None] * (HC * D)
            g0 = tl.load(GY + gb + ((0 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g1 = tl.load(GY + gb + ((1 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g2 = tl.load(GY + gb + ((2 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            g3 = tl.load(GY + gb + ((3 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            dx1 = p0[:, None] * g0 + p1[:, None] * g1 + p2[:, None] * g2 + p3[:, None] * g3
            tl.store(DX1 + rows[:, None] * D + cols[None, :],
                     dx1.to(DX1.dtype.element_ty), mask=rmask[:, None])
            r0 = tl.load(RES + rbase + ((0 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + ((1 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + ((2 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + ((3 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            rs = (r0, r1, r2, r3)
            # dres[i] = sum_j comb[i,j] * g_j   (same j order as v3)
            dr0 = c00[:, None] * g0
            dr0 += c01[:, None] * g1
            dr0 += c02[:, None] * g2
            dr0 += c03[:, None] * g3
            dr1 = c10[:, None] * g0
            dr1 += c11[:, None] * g1
            dr1 += c12[:, None] * g2
            dr1 += c13[:, None] * g3
            dr2 = c20[:, None] * g0
            dr2 += c21[:, None] * g1
            dr2 += c22[:, None] * g2
            dr2 += c23[:, None] * g3
            dr3 = c30[:, None] * g0
            dr3 += c31[:, None] * g1
            dr3 += c32[:, None] * g2
            dr3 += c33[:, None] * g3
            ob = rbase + cols[None, :]
            tl.store(DRES + ob + 0 * D, dr0.to(DRES.dtype.element_ty), mask=rmask[:, None])
            tl.store(DRES + ob + 1 * D, dr1.to(DRES.dtype.element_ty), mask=rmask[:, None])
            tl.store(DRES + ob + 2 * D, dr2.to(DRES.dtype.element_ty), mask=rmask[:, None])
            tl.store(DRES + ob + 3 * D, dr3.to(DRES.dtype.element_ty), mask=rmask[:, None])
            # dcomb[i,j] += sum_d g_j * res_i
            dc00 += tl.sum(g0 * r0, axis=1)
            dc01 += tl.sum(g1 * r0, axis=1)
            dc02 += tl.sum(g2 * r0, axis=1)
            dc03 += tl.sum(g3 * r0, axis=1)
            dc10 += tl.sum(g0 * r1, axis=1)
            dc11 += tl.sum(g1 * r1, axis=1)
            dc12 += tl.sum(g2 * r1, axis=1)
            dc13 += tl.sum(g3 * r1, axis=1)
            dc20 += tl.sum(g0 * r2, axis=1)
            dc21 += tl.sum(g1 * r2, axis=1)
            dc22 += tl.sum(g2 * r2, axis=1)
            dc23 += tl.sum(g3 * r2, axis=1)
            dc30 += tl.sum(g0 * r3, axis=1)
            dc31 += tl.sum(g1 * r3, axis=1)
            dc32 += tl.sum(g2 * r3, axis=1)
            dc33 += tl.sum(g3 * r3, axis=1)
            dpost0 += tl.sum(g0 * x1, axis=1)
            dpost1 += tl.sum(g1 * x1, axis=1)
            dpost2 += tl.sum(g2 * x1, axis=1)
            dpost3 += tl.sum(g3 * x1, axis=1)
        tl.store(DPOST + rows[:, None] * HC + 0, dpost0[:, None], mask=rmask[:, None])
        tl.store(DPOST + rows[:, None] * HC + 1, dpost1[:, None], mask=rmask[:, None])
        tl.store(DPOST + rows[:, None] * HC + 2, dpost2[:, None], mask=rmask[:, None])
        tl.store(DPOST + rows[:, None] * HC + 3, dpost3[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 0, dc00[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 1, dc01[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 2, dc02[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 3, dc03[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 4, dc10[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 5, dc11[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 6, dc12[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 7, dc13[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 8, dc20[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 9, dc21[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 10, dc22[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 11, dc23[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 12, dc30[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 13, dc31[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 14, dc32[:, None], mask=rmask[:, None])
        tl.store(DCOMB + rows[:, None] * (HC * HC) + 15, dc33[:, None], mask=rmask[:, None])


def _post_tile():
    def _i(name, default):
        try:
            return int(_flag(name, str(default)))
        except ValueError:
            return default
    return (_i("NANOV41_EW_POST_BM", 16), _i("NANOV41_EW_POST_BK", 128),
            _i("NANOV41_EW_POST_WARPS", 8), _i("NANOV41_EW_POST_STAGES", 2))


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _launch_post_fused(x1, res, post, comb):
    t, hc, d = res.shape
    bm, bk, warps, stages = _post_tile()
    y = torch.empty_like(res)
    _post_fwd_fused[(_cdiv(t, bm),)](y, x1, res, post, comb, t, D=d, HC=hc, BK=bk, BM=bm,
                                     num_warps=warps, num_stages=stages)
    return y


class _FusedPostFn(torch.autograd.Function):
    """Drop-in replacement for ``fused_mhc_v3._PostFn`` using the rewritten kernels."""

    @staticmethod
    def forward(ctx, x1, res, post, comb):
        t, hc, d = res.shape
        if not (res.is_cuda and res.dtype == torch.bfloat16 and hc == _HC):
            raise RuntimeError("fused_ew post needs CUDA bf16 [T,4,D] residual")
        y = _launch_post_fused(x1.reshape(t, d), res, post, comb)
        ctx.save_for_backward(x1, res, post, comb)
        return y

    @staticmethod
    def backward(ctx, gy):
        x1, res, post, comb = ctx.saved_tensors
        t, hc, d = res.shape
        gy = gy if gy is not None else torch.zeros_like(res)
        bm, bk, warps, stages = _post_tile()
        dx1 = torch.empty_like(x1)
        dres = torch.empty_like(res)
        dpost = torch.empty(t, hc, dtype=torch.float32, device=res.device)
        dcomb = torch.empty(t, hc * hc, dtype=torch.float32, device=res.device)
        _post_bwd_fused[(_cdiv(t, bm),)](
            dx1, dres, dpost, dcomb, gy.contiguous(), x1.reshape(t, d), res, post, comb, t,
            D=d, HC=hc, BK=bk, BM=bm, num_warps=warps, num_stages=stages)
        return dx1, dres, dpost, dcomb



# ======================================================= fused mhc_post+mixes_pre (item: postmix)
# Goal: keep the fast split FORWARD kernels (v3._launch_post + v3._launch_mixes_pre) but fuse the
# BACKWARD so that the split residual gradient is written in place instead of materialising a
# separate `dx` + letting autograd add `dres + dx` (24 x [T,4,D] = 4.8 GB/replay of extra traffic).
#
# Math (bitwise identical to the split path, by construction):
#     y            = post_fwd(x1, res, post, comb)
#     mixes, li, r = mixes_pre_fwd(y, fn, pmix)
# backward:
#     dmixes → du = dm*r ; dss ; dx_hc = dot(du^T,f_hc) + 2*dss*y_hc + pmix_hc*dli
#     dres_post, dx1, dpost, dcomb = post_bwd(gy)
#     dres = dres_post + round_bf16(dx)          <-- done IN PLACE by the kernel below
#     dfn  = du^T @ y   (cuBLAS, as in v3._MixesPreFn)
# The in-place add reproduces exactly what `dres_post + dx_bf16` would give, so the whole fused
# backward is bitwise identical to the current split path (same kernels, same fp32 order).
if HAS_TRITON:

    @triton.jit
    def _post_bwd_dual_kernel(
        DX1, DRES, DPOST, DCOMB, GY, DX, X1, RES, POST, COMB, T,
        D: tl.constexpr, HC: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
    ):
        """`fused_mhc_v3._post_bwd_kernel` with the input gradient given as the SUM of two buffers.

        The production split path feeds the post backward `gy + dx_mix` (the bf16 autograd add of the
        downstream grad and the mixes_pre grad wrt y).  This kernel forms that sum in registers as
        round_bf16(gy + dx) -- exactly the bf16 add autograd would have performed -- then runs the
        identical fp32 math/order of `_post_bwd_kernel`, so all four grads are bitwise identical.
        """
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
            # total input grad = round_bf16(gy + dx), then fp32 (same values autograd's add produced)
            g0 = (tl.load(GY + gb + ((0 * D + d0) + rk)[None, :], mask=rmask[:, None], other=0.0)
                  .to(tl.float32)
                  + tl.load(DX + gb + ((0 * D + d0) + rk)[None, :], mask=rmask[:, None],
                            other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            g1 = (tl.load(GY + gb + ((1 * D + d0) + rk)[None, :], mask=rmask[:, None], other=0.0)
                  .to(tl.float32)
                  + tl.load(DX + gb + ((1 * D + d0) + rk)[None, :], mask=rmask[:, None],
                            other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            g2 = (tl.load(GY + gb + ((2 * D + d0) + rk)[None, :], mask=rmask[:, None], other=0.0)
                  .to(tl.float32)
                  + tl.load(DX + gb + ((2 * D + d0) + rk)[None, :], mask=rmask[:, None],
                            other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            g3 = (tl.load(GY + gb + ((3 * D + d0) + rk)[None, :], mask=rmask[:, None], other=0.0)
                  .to(tl.float32)
                  + tl.load(DX + gb + ((3 * D + d0) + rk)[None, :], mask=rmask[:, None],
                            other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            gs = (g0, g1, g2, g3)
            x1 = tl.load(X1 + rows[:, None] * D + cols[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            dx1 = p0[:, None] * g0 + p1[:, None] * g1 + p2[:, None] * g2 + p3[:, None] * g3
            tl.store(DX1 + rows[:, None] * D + cols[None, :], dx1.to(DX1.dtype.element_ty),
                     mask=rmask[:, None])
            r0 = tl.load(RES + rbase + ((0 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r1 = tl.load(RES + rbase + ((1 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r2 = tl.load(RES + rbase + ((2 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            r3 = tl.load(RES + rbase + ((3 * D + d0) + rk)[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            rs = (r0, r1, r2, r3)
            for i in tl.static_range(0, HC):
                dri = tl.zeros((BM, BK), dtype=tl.float32)
                for j in tl.static_range(0, HC):
                    dri += tl.load(COMB + rows * (HC * HC) + i * 4 + j, mask=rmask,
                                   other=0.0)[:, None] * gs[j]
                tl.store(DRES + rbase + ((i * D + d0) + rk)[None, :],
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


def _v3():
    m = _V3.get("m")
    if m is None:
        import nanov41.fused_mhc_v3 as _m
        _V3["m"] = m = _m
    return m


class _PostMixesPreFusedFn(torch.autograd.Function):
    """Drop-in replacement for fused_mhc_v3._PostMixesPreFn's *split* composition.

    forward:  post_fwd (v3 kernel) + mixes_pre_fwd (v3 kernel) -> (y, mixes, li)
    backward: post_bwd(gy) then mixes_pre_bwd accumulated IN PLACE into dres (no separate dx /
              autograd add).  Bitwise identical to v3.mhc_post(...) followed by
              v3.mhc_mixes_pre(...) on the same inputs.
    """

    @staticmethod
    def forward(ctx, x1, res, post, comb, fn, pmix, eps):
        v3 = _v3()
        t, hc, d = res.shape
        y = v3._launch_post(x1, res, post, comb)                    # [T,HC,D] bf16
        y2 = y.reshape(t, hc * d)
        mixes, li, r = v3._launch_mixes_pre(y2, fn, pmix, True, eps)
        ctx.save_for_backward(x1, res, post, comb, fn, pmix, y, mixes, r)
        return y, mixes, li

    @staticmethod
    def backward(ctx, gy, dmixes, dli):
        v3 = _v3()
        x1, res, post, comb, fn, pmix, y, mixes, r = ctx.saved_tensors
        t, hc, d = res.shape
        k = hc * d
        if gy is None:
            gy = torch.zeros_like(y)
        if dmixes is None:
            dmixes = torch.zeros(t, v3.MIX, dtype=torch.float32, device=y.device)
        if dli is None:
            dli = torch.zeros(t, d, dtype=y.dtype, device=y.device)
        dli = dli.contiguous()
        dmixes = dmixes.contiguous()
        pmix = pmix.contiguous() if pmix is not None else None
        if pmix is None:
            raise RuntimeError("postmix fused backward requires pmix (want_pre=True)")
        y2 = y.reshape(t, k)
        # (1) mixes_pre backward (v3 kernel, unchanged): dx = grad wrt y, plus du / dpre
        gy = gy.contiguous()
        dx = torch.empty_like(res)
        du = torch.empty(t, v3.MIX, dtype=torch.bfloat16, device=y.device)
        dpre = torch.empty(t, hc, dtype=torch.float32, device=y.device)
        v3._mixes_pre_bwd_kernel[(v3._cdiv(t, v3.BM),)](
            y2, fn, pmix, dli, mixes, r, dmixes, dx, du, dpre, t,
            D=d, K=k, HC=hc, MIX=v3.MIX, MIXP=v3.MIXP, BK=v3.BK, BM=v3.BM, HAS_PRE=True,
            num_warps=v3._w("NANOV41_V3_W_MIX", 4), num_stages=2)
        # (2) post backward on round_bf16(gy + dx): the exact bf16 sum autograd's add would build
        dx1 = torch.empty_like(x1)
        dres = torch.empty_like(res)
        dpost = torch.empty(t, hc, dtype=torch.float32, device=res.device)
        dcomb = torch.empty(t, hc * hc, dtype=torch.float32, device=res.device)
        _post_bwd_dual_kernel[(v3._cdiv(t, v3.BM),)](
            dx1, dres, dpost, dcomb, gy, dx, x1.reshape(t, d), res, post, comb, t,
            D=d, HC=hc, BK=v3.BK, BM=v3.BM, num_warps=8, num_stages=2)
        dfn = torch.mm(du.t(), y2).to(fn.dtype)
        return dx1, dres, dpost, dcomb, dfn, dpre, None


def mhc_post_mixes_pre_fused(x1, res, post, comb, pmix, fn, eps=1e-20):
    """Same contract/shapes as fused_mhc_v3.mhc_post_mixes_pre (drop-in in the v4 driver)."""
    v3 = _v3()
    b, s, d = x1.shape
    t = b * s
    y, mixes, li = _PostMixesPreFusedFn.apply(
        x1.reshape(t, d), res.reshape(t, v3.HC, d), post.reshape(t, v3.HC),
        comb.reshape(t, v3.HC * v3.HC).contiguous(), fn,
        pmix.reshape(t, v3.HC) if pmix is not None else None, eps)
    return y.view(b, s, v3.HC, d), mixes.view(b, s, v3.MIX), li.view(b, s, d)


# --------------------------------------------------------------------------- driver (full fusion)
_ORIG_TFWD = {"fn": None}


def _make_forward_fused(M):
    """Training forward (NanoV41 calls it with full_logits=True) with all mHC boundary pairs fused.

    Faithful copy of the official Transformer.forward block loop (incl. Engram injection and MTP
    target capture), with `v3.mhc_post` followed by `v3.mhc_mixes_pre` replaced by ONE fused op.
    Falls back to the original forward for any path it does not cover (VL / decode / MTP).
    """
    def forward(self, input_ids, start_pos=0, images=None, token_types=None,
                full_logits=False):
        orig = _ORIG_TFWD.get("fn")
        if (orig is None or images is not None or token_types is not None
                or not full_logits or getattr(self, "vision", None) is not None
                or getattr(self, "target_layer_ids", None)):
            return orig(self, input_ids, start_pos, images, token_types, full_logits)
        v3 = _v3()
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        engram_hashes = (self.engram_hash(input_ids, start_pos, None)
                         if self.engram_hash is not None else None)
        pre_mix = M.make_identity_pre_mix(h, self.hc_mult)
        pend = None
        res = h
        for layer in self.layers:
            if layer.engram is not None:
                if pend is not None:                       # materialise before Engram writes
                    x1p, resp, postp, combp = pend
                    res = v3.mhc_post(x1p, resp, postp, combp)
                    pend = None
                res = layer.engram(res, engram_hashes[:, :, layer.engram.layer_hash_index, :],
                                   None)
            if pend is None:
                mixes, li = v3.mhc_mixes_pre(res, layer.hc_attn_fn, pre_mix, layer.norm_eps)
            else:
                x1p, resp, postp, combp = pend
                pend = None
                res, mixes, li = mhc_post_mixes_pre_fused(
                    x1p, resp, postp, combp, pre_mix, layer.hc_attn_fn, layer.norm_eps)
            attn_pre, attn_post, attn_comb = v3.mhc_split_sinkhorn(
                mixes, layer.hc_attn_scale, layer.hc_attn_base,
                layer.hc_sinkhorn_iters, layer.hc_eps)
            x = layer.attn_norm(li)
            x = layer.attn(x, start_pos)
            y, mixes2, li2 = mhc_post_mixes_pre_fused(
                x, res, attn_post, attn_comb, attn_pre, layer.hc_ffn_fn, layer.norm_eps)
            ffn_pre, ffn_post, ffn_comb = v3.mhc_split_sinkhorn(
                mixes2, layer.hc_ffn_scale, layer.hc_ffn_base,
                layer.hc_sinkhorn_iters, layer.hc_eps)
            x = layer.ffn_norm(li2)
            x = layer.ffn(x, image_mask=None)
            pend = (x, y, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        x1p, resp, postp, combp = pend
        res = v3.mhc_post(x1p, resp, postp, combp)
        h = self.layers[-1].hc_pre(res, pre_mix)
        return self.head(self.norm(h), full_logits=True)

    return forward


def _install_postmix(M) -> bool:
    """Patch M.Transformer.forward with the fully-fused driver (post-v4-install hook)."""
    if M is None:
        return False
    tcls = getattr(M, "Transformer", None)
    if tcls is None:
        return False
    if _ORIG_TFWD.get("fn") is None:
        _ORIG_TFWD["fn"] = tcls.forward
    tcls.forward = _make_forward_fused(M)
    _STATE.setdefault("orig_tfwd", _ORIG_TFWD["fn"])
    _STATE["patched"].append("Transformer.forward(postmix driver)")
    return True


def _hook_v4_install() -> bool:
    """monkeypatch fused_mhc_v4.install so the fused driver lands AFTER v4 patches Block.forward."""
    try:
        import nanov41.fused_mhc_v4 as v4
    except Exception as exc:  # noqa: BLE001
        _STATE["error"] = f"fused_mhc_v4 import failed: {exc!r}"
        return False
    orig = getattr(v4, "install", None)
    if orig is None or getattr(orig, "_ew_postmix_hooked", False):
        return True

    def _wrapped(*a, **kw):
        out = orig(*a, **kw)
        try:
            mm = a[0] if a else kw.get("M")
            if isinstance(out, dict) and out.get("mode") == "split":
                _install_postmix(mm)
                print(f"[fused_ew] postmix driver installed: {get_state()['patched']}",
                      flush=True)
            else:
                print(f"[fused_ew] postmix skipped (v4 mode={(out or {}).get('mode')!r})",
                      flush=True)
        except Exception as exc:  # noqa: BLE001 - never break the build
            _STATE["error"] = f"postmix install failed: {exc!r}"
            print(f"[fused_ew] postmix install failed: {exc!r}", flush=True)
        return out

    _wrapped._ew_postmix_hooked = True
    _wrapped._ew_orig_install = orig
    v4.install = _wrapped
    return True


# =========================================================================== install / uninstall
def install(M=None) -> dict:
    """Patch the enabled items into ``nanov41.fused_mhc_v3`` (idempotent)."""
    if _STATE["installed"]:
        return get_state()
    want = items()
    patched: list[str] = []
    state = {"installed": True, "items": want, "patched": patched, "v3": None, "_orig": {}}
    try:
        import nanov41.fused_mhc_v3 as v3
    except Exception as exc:  # noqa: BLE001 - optional; leave baseline untouched
        state["error"] = f"fused_mhc_v3 import failed: {exc!r}"
        _STATE.clear()
        _STATE.update(state)
        return get_state()
    state["v3"] = v3
    if "post" in want and available():
        orig = getattr(v3, "_PostFn", None)
        if orig is not None:
            state["_orig"]["_PostFn"] = orig
            v3._PostFn = _FusedPostFn
            patched.append("fused_mhc_v3._PostFn")
    if "postmix" in want and available():
        _ORIG_TFWD.setdefault("fn", None)
        if _hook_v4_install():
            patched.append("fused_mhc_v4.install(hook)")
    _STATE.clear()
    _STATE.update(state)
    return get_state()


def uninstall(M=None) -> dict:
    """Restore the original fused_mhc_v3 classes / v4.install / Transformer.forward."""
    v3 = _STATE.get("v3")
    if v3 is not None:
        for name, fn in (_STATE.get("_orig") or {}).items():
            setattr(v3, name, fn)
    try:  # restore fused_mhc_v4.install
        import nanov41.fused_mhc_v4 as v4
        cur = getattr(v4, "install", None)
        if getattr(cur, "_ew_postmix_hooked", False):
            v4.install = getattr(cur, "_ew_orig_install", None) or v4.install
    except Exception:  # noqa: BLE001
        pass
    orig_tfwd = _STATE.get("orig_tfwd")
    if orig_tfwd is not None and M is not None:
        try:
            M.Transformer.forward = orig_tfwd
        except Exception:  # noqa: BLE001
            pass
    _STATE.update({"installed": False, "items": (), "patched": [], "v3": None, "_orig": {}})
    return get_state()


def get_state() -> dict:
    return {"installed": _STATE["installed"], "items": list(_STATE["items"]),
            "patched": list(_STATE["patched"]),
            "error": _STATE.get("error", "")}
