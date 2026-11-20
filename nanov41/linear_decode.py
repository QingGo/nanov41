# SPDX-License-Identifier: MIT
"""Batch-invariant decode projections (M3).

Why this exists: cuBLAS picks a GEMV kernel for M=1 and a GEMM kernel for M>=2, and the two
accumulate in a different order.  The difference is ~1 fp32 ulp before the bf16 rounding, i.e. one
bf16 ulp on a few elements -- but a decode step is 12 layers of bf16 rounding, so a batched sequence
gradually drifts away from the single-sequence runtime and can eventually flip a greedy token.

``linear_decode_kernel`` removes that batch-size dependence for the decode projections: it tiles
``BLOCK_M=16`` rows, loads the whole weight tile once and reduces over K in fixed ``BLOCK_K`` chunks,
so a row's result is the same whether it is the only row or one of 16.  It also replaces the
latency-bound cuBLAS GEMV at M=1 with a tensor-core tile (same or better GPU time).

Installed by ``enable_linear_decode`` on the patched reference module's ``linear()`` helper (which
every ``Linear``/``Linear``-subclass forward calls) and used directly by the Engram decode gate.
Prefill (rows > ``MAX_ROWS``) falls back to the reference cuBLAS path.
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

MAX_ROWS = 16
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64

_ORIG_LINEAR = None


if HAS_TRITON:
    @triton.jit
    def linear_decode_kernel(X, W, OUT, M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, IEEE: tl.constexpr):
        """OUT[M,N] = X[M,K] @ W[N,K]^T, fp32 accumulation, one rounding to OUT's dtype.

        The K reduction is a fixed sequence of BLOCK_K dot chunks, so each row's result does not
        depend on how many other rows are in the launch (batch-invariant by construction).
        """
        pid = tl.program_id(0)
        rm = tl.arange(0, BLOCK_M)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rm < M
        nmask = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            kmask = rk < K
            x = tl.load(X + rm[:, None] * K + rk[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            w = tl.load(W + rn[:, None] * K + rk[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0.0)
            if IEEE:
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")
            else:
                acc += tl.dot(x, tl.trans(w))
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty),
                 mask=rmask[:, None] & nmask[None, :])


def linear_decode(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    """Drop-in ``linear()`` for decode-sized inputs; falls back to the original for anything else."""
    if (bias is None and HAS_TRITON and x.is_cuda and weight.dim() == 2
            and x.dtype == weight.dtype
            and x.dtype in (torch.bfloat16, torch.float32)
            and x.shape[-1] == weight.shape[1]):
        n, k = int(weight.shape[0]), int(weight.shape[1])
        rows = int(x.numel()) // k
        if 0 < rows <= MAX_ROWS and n >= 16 and k >= 16:
            out = torch.empty(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
            x2 = x.reshape(rows, k)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            grid = (triton.cdiv(n, BLOCK_N),)
            linear_decode_kernel[grid](
                x2, weight, out, rows, n, k,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                IEEE=(x.dtype == torch.float32), num_warps=4,
            )
            return out
    if _ORIG_LINEAR is None:                      # pragma: no cover - defensive
        return torch.nn.functional.linear(x, weight)
    return _ORIG_LINEAR(x, weight, bias)


def enable_linear_decode(model) -> int:
    """Patch the reference module's ``linear()`` with the batch-invariant decode kernel."""
    if not HAS_TRITON or not torch.cuda.is_available():
        raise RuntimeError("batch-invariant decode linear needs CUDA+Triton")
    global _ORIG_LINEAR
    from .patched_env import import_patched

    mod = import_patched()
    if getattr(mod, "_nanov41_linear_decode_installed", False):
        return 0
    _ORIG_LINEAR = mod.linear
    mod.linear = linear_decode
    mod._nanov41_linear_decode_installed = True
    return 1


def disable_linear_decode(model) -> int:
    global _ORIG_LINEAR
    from .patched_env import import_patched

    mod = import_patched()
    if not getattr(mod, "_nanov41_linear_decode_installed", False):
        return 0
    mod.linear = _ORIG_LINEAR
    del mod._nanov41_linear_decode_installed
    _ORIG_LINEAR = None
    return 1


__all__ = ["linear_decode", "enable_linear_decode", "disable_linear_decode",
           "linear_decode_kernel", "MAX_ROWS", "HAS_TRITON"]
