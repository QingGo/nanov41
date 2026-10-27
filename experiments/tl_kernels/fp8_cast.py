# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Fused TileLang FP8 cast (bf16 -> e4m3) with a scalar or per-row scale, single pass.

Used for the large dC tensor in FP4-forward / FP8-backward training: PyTorch
``(x * inv).to(fp8)`` on [8192, 129280] costs ~4.9 ms on an RTX 5090; this kernel reads
2.1 GB bf16 + writes 1.05 GB fp8 (memory-bound).

Measured on the internal 5090 (torch 2.12.1+cu130 / tilelang 0.1.15; not re-validated by
the public test suite here): ~2.12 ms vs 4.90 ms for the PyTorch cast (2.3x); see
``README.md``. Requires TileLang + CUDA (sm_120 was the tested target).
"""

from __future__ import annotations

import torch

try:  # TileLang is optional; this module is importable without it.
    import tilelang
    import tilelang.language as T

    _PASS = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    TILELANG_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    tilelang = None
    T = None
    _PASS = None
    TILELANG_AVAILABLE = False


if TILELANG_AVAILABLE:

    @tilelang.jit(pass_configs=_PASS)
    def _cast_fp8_kernel(M: int, N: int, blk_m: int = 32, blk_n: int = 128):
        assert M % blk_m == 0 and N % blk_n == 0, (M, N, blk_m, blk_n)

        @T.prim_func
        def main(
            X: T.Tensor((M, N), T.bfloat16),
            Y: T.Tensor((M, N), T.float8_e4m3),
            S: T.Tensor((1,), T.float32),
        ):
            with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), threads=256) as (bx, by):
                x_s = T.alloc_shared((blk_m, blk_n), T.bfloat16)
                y_s = T.alloc_shared((blk_m, blk_n), T.float8_e4m3)
                T.copy(X[bx * blk_m, by * blk_n], x_s)
                inv = 1.0 / S[0]
                for i, j in T.Parallel(blk_m, blk_n):
                    y_s[i, j] = T.Cast(T.float8_e4m3, T.Cast(T.float32, x_s[i, j]) * inv)
                T.copy(y_s, Y[bx * blk_m, by * blk_n])

        return main

    @tilelang.jit(pass_configs=_PASS)
    def _cast_fp8_row_kernel(M: int, N: int, blk_m: int = 32, blk_n: int = 128):
        """Same as ``_cast_fp8_kernel`` but with a per-row scale vector S[M]."""
        assert M % blk_m == 0 and N % blk_n == 0

        @T.prim_func
        def main(
            X: T.Tensor((M, N), T.bfloat16),
            Y: T.Tensor((M, N), T.float8_e4m3),
            S: T.Tensor((M,), T.float32),
        ):
            with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), threads=256) as (bx, by):
                x_s = T.alloc_shared((blk_m, blk_n), T.bfloat16)
                y_s = T.alloc_shared((blk_m, blk_n), T.float8_e4m3)
                T.copy(X[bx * blk_m, by * blk_n], x_s)
                for i, j in T.Parallel(blk_m, blk_n):
                    y_s[i, j] = T.Cast(
                        T.float8_e4m3, T.Cast(T.float32, x_s[i, j]) / S[bx * blk_m + i]
                    )
                T.copy(y_s, Y[bx * blk_m, by * blk_n])

        return main

else:  # pragma: no cover
    _cast_fp8_kernel = None
    _cast_fp8_row_kernel = None


def cast_fp8(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """x bf16 [M,N] (M%32==0, N%128==0) -> e4m3 with x/scale. Requires TileLang + CUDA."""
    if not TILELANG_AVAILABLE:
        raise RuntimeError("cast_fp8 requires tilelang (see experiments/tl_kernels/README.md)")
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor([float(scale)], device=x.device, dtype=torch.float32)
    scale = scale.reshape(1).float().contiguous()
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    kernel = _cast_fp8_kernel(M, N)
    kernel(x.contiguous(), y, scale)
    return y


def cast_fp8_row(x: torch.Tensor, scale_row: torch.Tensor) -> torch.Tensor:
    """x bf16 [M,N] -> e4m3 with per-row scale_row [M] (M%32==0, N%128==0). Requires TileLang + CUDA."""
    if not TILELANG_AVAILABLE:
        raise RuntimeError("cast_fp8_row requires tilelang (see experiments/tl_kernels/README.md)")
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    kernel = _cast_fp8_row_kernel(M, N)
    kernel(x.contiguous(), y, scale_row.float().contiguous())
    return y


def tensor_amax(x: torch.Tensor) -> torch.Tensor:
    """abs-max of a tensor, clamped to avoid division by zero."""
    return x.abs().max().float().clamp_min(1e-8)


__all__ = ["TILELANG_AVAILABLE", "cast_fp8", "cast_fp8_row", "tensor_amax"]
