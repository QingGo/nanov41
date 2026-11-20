# SPDX-License-Identifier: MIT
"""Fused decode RMSNorm (M2d).

The reference ``RMSNorm.forward`` is ``x.float() -> square -> mean -> +eps -> rsqrt -> mul x ->
mul weight -> cast``: ~7 tiny ATen kernels per call, and a decode step calls it 24 times for the
block input norms (twice per layer) plus the final norm and the two compressor norms.  On a 4090
graph replay each of those kernels is launch-latency bound (~0.7-1.5 us of GPU time for 384
elements), so the whole family is ~150-200 us/step.

``rmsnorm_decode_kernel`` computes the same fp32 statistics and one final rounding in a single
program per row (decode rows are 1, so one CTA).  Installed by ``enable_rmsnorm_decode`` only for
small, contiguous, CUDA tensors; prefill/large calls fall back to the reference forward.

Numerics: the fp32 reduction order differs from ATen's ``mean`` (Triton tree reduction), so values
can differ by a few fp32 ulps before the bf16 store; the store rounds exactly like the reference's
final ``.to(dtype)``.
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


if HAS_TRITON:
    @triton.jit
    def rmsnorm_decode_kernel(X, W, OUT, eps, N: tl.constexpr, BLOCK: tl.constexpr):
        """One program: one row of ``N`` elements -> RMSNorm (fp32 math, dtype cast on store)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        tl.store(OUT + row * N + cols, (x * rstd * w).to(OUT.dtype.element_ty), mask=mask)


def rmsnorm_decode(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``RMSNorm.forward`` for a small contiguous CUDA tensor (dtype values in/out)."""
    n = int(x.shape[-1])
    rows = int(x.numel()) // n
    out = torch.empty_like(x)
    rmsnorm_decode_kernel[(rows,)](
        x, weight, out, float(eps), N=n, BLOCK=_next_pow2(n),
        num_warps=4 if n <= 2048 else 8,
    )
    return out


def enable_rmsnorm_decode(model, *, max_rows: int = 4096, max_dim: int = 8192) -> int:
    """Patch the model's ``RMSNorm`` class with a fused single-kernel forward for decode shapes.

    Decode calls are ``[1, 1, dim]`` (and the compressor's ``[1, 1, dim]``); prefill calls have
    thousands of rows and take the reference path.  Idempotent.
    """
    if not HAS_TRITON or not torch.cuda.is_available():
        raise RuntimeError("fused decode RMSNorm needs CUDA+Triton")
    inner = getattr(model, "inner", model)
    ref = getattr(inner, "norm", None)
    if ref is None:
        return 0
    cls = type(ref)
    if getattr(cls, "_decode_rms_fast_installed", False):
        return 0
    cls._decode_rms_orig_forward = cls.forward

    def _forward(self, x: torch.Tensor):
        if (x.is_cuda and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16,
                                                            torch.float32)
                and 0 < x.shape[-1] <= max_dim
                and x.numel() // int(x.shape[-1]) <= max_rows):
            return rmsnorm_decode(x, self.weight, self.eps)
        return self._decode_rms_orig_forward(x)

    cls.forward = _forward
    cls._decode_rms_fast_installed = True
    return 1


def disable_rmsnorm_decode(model) -> int:
    inner = getattr(model, "inner", model)
    ref = getattr(inner, "norm", None)
    if ref is None:
        return 0
    cls = type(ref)
    if not getattr(cls, "_decode_rms_fast_installed", False):
        return 0
    cls.forward = cls._decode_rms_orig_forward
    del cls._decode_rms_orig_forward
    del cls._decode_rms_fast_installed
    return 1


__all__ = ["rmsnorm_decode", "enable_rmsnorm_decode", "disable_rmsnorm_decode",
           "rmsnorm_decode_kernel", "HAS_TRITON"]
