# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Fused Triton kernel for the indexer head reduction (M4 step 2).

The reference indexer computes, per query chunk,

    score = einsum("chd,pd->chp")            # [C, H, n_pos]   (H = index_n_heads, HD = index dim)
    score = relu(score) * w                  # [C, H, n_pos]
    S     = score.sum(dim=1)                 # [C, n_pos]

which materialises `[C, H, n_pos]` in HBM and performs three separate elementwise passes. At 1M
context that is the dominant cost (GPU profile: relu+mul+sum ~45 s of 73 s). This kernel fuses the
4 small GEMMs with relu + per-head weight + head sum in fp32 and writes only `[C, n_pos]`.

One program handles `BLOCK_C` queries and one position segment (0..hi split into `segs` segments,
so a full-sequence prefill launches ~`C/BLOCK_C * segs` programs); it loops over `BLOCK_P` position
tiles, keeps the per-segment index-key tile in cache, and applies causal visibility in-kernel
(compressed position j visible to query i iff j < clens[i]).

CUDA-only and opt-in: `fused_indexer_score()` raises when Triton/CUDA is unavailable; the
chunked-indexer module falls back to the torch path on CPU or when the kernel is disabled.
"""
from __future__ import annotations

import torch

try:  # triton is optional: CPU installs and CI never need it
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    tl = None
    HAS_TRITON = False

import os

BLOCK_C = int(os.environ.get("NANOV41_IDX_FUSED_BC", "64"))
BLOCK_P = int(os.environ.get("NANOV41_IDX_FUSED_BP", "64"))


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _heads_score_kernel(
        Q, K, W, CLENS, S,
        C, HI, SEG_SPAN,
        BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr,
        NHEADS: tl.constexpr, HD: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_s = tl.program_id(1)
        c0 = pid_c * BLOCK_C
        n_c = C - c0  # rows beyond n_c are masked out
        rows_c = tl.arange(0, BLOCK_C)
        rows_h = tl.arange(0, NHEADS)
        cols = tl.arange(0, HD)
        row_ok = rows_c < n_c

        # index keys for this segment: [HI, HD], read as [HD, BP] per tile
        p_start = pid_s * SEG_SPAN
        p_stop = tl.minimum(HI, (pid_s + 1) * SEG_SPAN)
        clens = tl.load(CLENS + c0 + rows_c, mask=row_ok, other=0).to(tl.int32)  # [BC]
        w_mat = tl.load(W + (c0 + rows_c)[:, None] * NHEADS + rows_h[None, :],
                        mask=row_ok[:, None], other=0.0).to(tl.float32)          # [BC, H]
        Qbase = Q + (c0 + rows_c)[:, None] * (NHEADS * HD)                      # [BC, 1]
        n_tiles = tl.cdiv(p_stop - p_start, BLOCK_P)
        for t in tl.range(0, n_tiles):
            p0 = p_start + t * BLOCK_P
            p = p0 + tl.arange(0, BLOCK_P)
            p_ok = p < HI
            k = tl.load(K + p[None, :] * HD + cols[:, None],
                        mask=p_ok[None, :], other=0.0)                          # [HD, BP]
            acc = tl.zeros((BLOCK_C, BLOCK_P), dtype=tl.float32)
            for h in tl.static_range(0, NHEADS):
                qh = tl.load(Qbase + h * HD + cols[None, :],
                             mask=row_ok[:, None], other=0.0)                   # [BC, HD]
                s_h = tl.dot(qh, k)                                             # [BC, BP] fp32
                # emulate the reference dtype path exactly: einsum output is bf16, relu is applied
                # on the bf16 score, the bf16 weight multiply rounds back to bf16, and the head sum
                # accumulates in fp32 before the final bf16 store
                s_bf = s_h.to(tl.bfloat16)
                s_bf = tl.maximum(s_bf, 0.0).to(tl.bfloat16)
                w_h = tl.sum(tl.where(rows_h[None, :] == h, w_mat, 0.0), axis=1)  # [BC]
                s_bf = (s_bf.to(tl.float32) * w_h[:, None]).to(tl.bfloat16)
                acc += s_bf.to(tl.float32)
            vis = p[None, :] < clens[:, None]
            acc = tl.where(vis, acc, float("-inf"))
            tl.store(S + (c0 + rows_c)[:, None] * HI + p[None, :],
                     acc.to(tl.bfloat16), mask=row_ok[:, None] & p_ok[None, :])


def fused_indexer_score(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor,
                        clens: torch.Tensor, segs: int | None = None) -> torch.Tensor:
    """Return S [C, hi] = sum_h relu(q[c,h,:]·k[p,:]) * w[c,h], causal-masked, in bf16.

    q: [1, C, H, HD] bf16 CUDA, k: [1, hi, HD] bf16 CUDA, w: [1, C, H] bf16 CUDA,
    clens: [C] int32 CUDA = number of compressed positions visible to each query.
    """
    if not available():
        raise RuntimeError("fused indexer kernel needs Triton + CUDA")
    if q.dim() != 4 or q.size(0) != 1:
        raise ValueError(f"expected q [1, C, H, HD], got {tuple(q.shape)}")
    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    C, H, HD = q.size(1), q.size(2), q.size(3)
    hi = k.size(1)
    if hi == 0:
        return torch.empty((C, 0), dtype=torch.bfloat16, device=q.device)
    S = torch.empty((C, hi), dtype=torch.bfloat16, device=q.device)
    if segs is None:
        segs = int(os.environ.get("NANOV41_IDX_FUSED_SEGS", "0") or 0)
    if not segs:
        segs = max(1, min(64, triton.cdiv(hi, BLOCK_P * 4)))
    seg_span = triton.cdiv(hi, segs)
    grid = (triton.cdiv(C, BLOCK_C), segs)
    _heads_score_kernel[grid](
        q, k, w, clens, S,
        C, hi, seg_span,
        BLOCK_C=BLOCK_C, BLOCK_P=BLOCK_P, NHEADS=H, HD=HD,
    )
    return S
