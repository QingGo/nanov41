# SPDX-License-Identifier: MIT
"""bf16 output-head fast path (M2, M2d).

The fp32 ``ParallelHead.weight`` (129,280 x 384 fp32 = 198.5 MB) is the single largest streamed
tensor of a decode token: ~0.36 ms of the 3.1 ms graph-replay budget on a 4090.  Keeping an
inference-only bf16 copy halves that read (99 MB) and runs the GEMV in bf16; the logits are then
cast back to fp32.  This is a decode/inference-only numerics change (logits carry bf16 rounding,
~2e-3 relative), so it is validated against generation agreement + T0 BPB before being enabled by
default.

M2d adds ``fused_gemv=True``: a Triton GEMV that streams the bf16 head with enough CTAs to reach
memory bandwidth (cuBLAS is latency-bound at M=1: 99 MB at ~0.44 TB/s = 228 us; the kernel reaches
~0.7-0.9 TB/s).  It keeps ``F.linear``'s bf16 output rounding (fp32 accumulate -> round to bf16 ->
cast to fp32) and handles any number of rows <= ``max_rows``; larger calls (prefill) fall back to
the cuBLAS bf16 path.
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
    triton = None
    tl = None


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


if HAS_TRITON:
    @triton.jit
    def head_gemv_kernel(X, W, OUT,
                         V: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
                         BM: tl.constexpr, BK: tl.constexpr):
        """One program: BM vocab rows for one row of x; fp32 accumulate, bf16 round, fp32 store."""
        b = tl.program_id(0)
        pid = tl.program_id(1)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < V
        cols = tl.arange(0, BLOCK_K)
        kmask = cols < K
        x = tl.load(X + b * K + cols, mask=kmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM,), tl.float32)
        for k0 in tl.static_range(0, BLOCK_K, BK):
            kk = k0 + tl.arange(0, BK)
            kv = kk < K
            w = tl.load(W + rows[:, None] * K + kk[None, :],
                        mask=rmask[:, None] & kv[None, :], other=0.0)
            xk = tl.load(X + b * K + kk, mask=kv, other=0.0).to(tl.float32)
            acc += tl.sum(w.to(tl.float32) * xk[None, :], axis=1)
        logits = acc.to(tl.bfloat16).to(tl.float32)          # F.linear bf16 output rounding
        tl.store(OUT + b * V + rows, logits, mask=rmask)


def bf16_head_fused(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
                    *, bm: int = 32) -> torch.Tensor:
    """``F.linear(x, weight).float()`` via the fused bf16 GEMV/GEMM; returns the out buffer view."""
    k = int(weight.shape[1])
    v = int(weight.shape[0])
    rows = int(x.numel()) // k
    x2 = x.reshape(rows, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    head_gemv_kernel[(rows, triton.cdiv(v, bm))](
        x2, weight, out, V=v, K=k, BLOCK_K=_next_pow2(k),
        BM=bm, BK=128, num_warps=4,
    )
    return out[:rows].view(x.shape[:-1] + (v,))


def enable_bf16_head(model, fused_gemv: bool = False, *, max_rows: int = 16,
                     bm: int = 32) -> int:
    """Replace the head forward with a bf16-weight GEMV/GEMM (idempotent).

    ``fused_gemv=True`` uses the Triton GEMV for decode-sized inputs (<= ``max_rows`` rows);
    everything larger keeps the cuBLAS bf16 path.
    """
    inner = getattr(model, "inner", model)
    head = getattr(inner, "head", None)
    if head is None or hasattr(head, "_bf16_orig_forward"):
        return 0
    head._bf16_orig_forward = head.forward
    head._bf16_weight = head.weight.detach().to(torch.bfloat16).contiguous()
    head._bf16_fused = bool(fused_gemv and HAS_TRITON and torch.cuda.is_available())
    if head._bf16_fused:
        dev = head._bf16_weight.device
        head._bf16_out = torch.empty(max_rows, int(head._bf16_weight.shape[0]),
                                     dtype=torch.float32, device=dev)
        head._bf16_max_rows = max_rows
        head._bf16_bm = bm

    def _forward(self, x: torch.Tensor, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        if self._bf16_fused and x.is_cuda and x.shape[-1] == self._bf16_weight.shape[1]:
            rows = x.numel() // int(x.shape[-1])
            if rows <= self._bf16_max_rows:
                return bf16_head_fused(x, self._bf16_weight, self._bf16_out, bm=self._bf16_bm)
        return F.linear(x.to(torch.bfloat16), self._bf16_weight).float()

    head.forward = _forward.__get__(head, type(head))
    return 1


def disable_bf16_head(model) -> int:
    inner = getattr(model, "inner", model)
    head = getattr(inner, "head", None)
    if head is None or not hasattr(head, "_bf16_orig_forward"):
        return 0
    head.forward = head._bf16_orig_forward
    for name in ("_bf16_orig_forward", "_bf16_weight", "_bf16_fused", "_bf16_out",
                 "_bf16_max_rows", "_bf16_bm"):
        if hasattr(head, name):
            delattr(head, name)
    return 1


__all__ = ["enable_bf16_head", "disable_bf16_head", "bf16_head_fused", "head_gemv_kernel",
           "HAS_TRITON"]
