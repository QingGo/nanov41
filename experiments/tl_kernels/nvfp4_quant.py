# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Fused TileLang NVFP4 quantizer (bf16 -> packed E2M1 + per-1x16 E4M3 scales).

Adapted from the DeepSeek-V4.1-Flash inference reference
``ref/kernel.py: fp4_quant_kernel`` (MIT, Copyright (c) 2023 DeepSeek; see ``NOTICE``);
only the group size (32 -> 16) and the output scale packing (E4M3 bytes -> uint32 words
-> ``blockscaled_chunk_kmajor`` swizzle) change.

Format note: like ``nvfp4_gemm``, this produces **NVFP4** (E4M3 scales per 1x16), not the
official **MXFP4** expert-weight format (E8M0 scales per 1x32); see ``README.md``.

Why: the PyTorch quantizer (``nvfp4_gemm.quantize_nvfp4``) launches ~8-10 separate CUDA
kernels and materialises several 100 MB temporaries; for a [129280, 512] weight it costs
~9.4 ms on an RTX 5090, which wipes out the FP4 GEMM win (kernel-only GEMM is 1.66 ms).
One fused kernel does the whole job (read bf16 -> per-16 absmax -> E4M3 scale ->
clamp/quantize -> pack 2 codes/byte -> write packed FP4 + scale bytes); it is
memory-bound (~170 MB traffic).

Measured on the internal 5090 (torch 2.12.1+cu130 / tilelang 0.1.15; not re-validated by
the public test suite here): ~0.54 ms for the [129280, 512] weight, 17.4x faster than the
PyTorch quantizer, with 100% identical scale words and ~94% identical packed bytes
(remaining bytes differ only at E2M1 rounding ties). See ``README.md``.
"""

import torch
import torch.nn.functional as F

from .nvfp4_gemm import swizzle_blockscaled_chunk_kmajor_scale_words

try:  # TileLang is optional; this module is importable without it.
    import tilelang
    import tilelang.language as T

    _PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    TILELANG_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    tilelang = None
    T = None
    _PASS_CONFIGS = None
    TILELANG_AVAILABLE = False

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
BF16 = "bfloat16"
FP32 = "float32"


if TILELANG_AVAILABLE:

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _nvfp4_quant_kernel(N: int, group_size: int = 16, blk_m: int = 32):
        """X: [M, N] bf16 -> Y: [M, N] fp4 (torch stores N//2 packed bytes), S: [M, N/group] e4m3."""
        M = T.symbolic("M")
        fp4_max = 6.0
        assert N % group_size == 0

        @T.prim_func
        def main(
            X: T.Tensor[(M, N), BF16],
            Y: T.Tensor[(M, N), FP4],
            S: T.Tensor[(M, T.ceildiv(N, group_size)), FP8],
        ):
            with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (pid_m, pid_n):
                x_shared = T.alloc_shared((blk_m, group_size), BF16)
                x_local = T.alloc_fragment((blk_m, group_size), BF16)
                amax_local = T.alloc_fragment((blk_m,), FP32)
                s_local = T.alloc_fragment((blk_m,), FP32)
                y_local = T.alloc_fragment((blk_m, group_size), FP4)
                y_shared = T.alloc_shared((blk_m, group_size), FP4)

                for _ in T.Pipelined(1, num_stages=2):
                    T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                    T.copy(x_shared, x_local)
                    T.reduce_absmax(x_local, amax_local, dim=1)
                    for i in T.Parallel(blk_m):
                        # E4M3 min subnormal: keep an all-zero group's scale nonzero
                        amax_local[i] = T.max(amax_local[i], 6 * (2.0 ** -9))
                        s_local[i] = T.Cast(FP32, T.Cast(FP8, amax_local[i] / fp4_max))
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], -fp4_max, fp4_max)
                    for i in T.Parallel(blk_m):
                        S[pid_m * blk_m + i, pid_n] = T.Cast(FP8, s_local[i])
                    T.copy(y_local, y_shared)
                    T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

        return main

else:  # pragma: no cover
    _nvfp4_quant_kernel = None


def quantize_nvfp4_tl(x: torch.Tensor, k_pad_to: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [rows, K] -> (packed FP4 int8-view [rows, Kpad//2], swizzled uint32 words).

    Requires TileLang + an sm_120 GPU (e.g. RTX 5090). K is right-padded to a multiple of
    ``k_pad_to`` (must be a multiple of 256).
    """
    if not TILELANG_AVAILABLE:
        raise RuntimeError("quantize_nvfp4_tl requires tilelang (see experiments/tl_kernels/README.md)")
    assert x.dim() == 2
    rows, k = x.shape
    kpad = (k + k_pad_to - 1) // k_pad_to * k_pad_to
    if kpad != k:
        x = F.pad(x, (0, kpad - k))
    x = x.contiguous()
    kernel = _nvfp4_quant_kernel(kpad, 16, 32)
    # TileLang declares Y as [M, N] fp4; the torch tensor is [M, N//2] packed bytes.
    y = torch.empty((rows, kpad // 2), device=x.device, dtype=torch.float4_e2m1fn_x2)
    s = torch.empty((rows, kpad // 16), device=x.device, dtype=torch.float8_e4m3fn)
    kernel(x, y, s)
    # pack 4 E4M3 scale bytes into one uint32 word, then swizzle to the GEMM layout
    sb = s.view(torch.uint8).to(torch.int64).reshape(rows, kpad // 64, 4)
    words = sb[:, :, 0] | (sb[:, :, 1] << 8) | (sb[:, :, 2] << 16) | (sb[:, :, 3] << 24)
    words = words.to(torch.uint32)
    sw = swizzle_blockscaled_chunk_kmajor_scale_words(words, 128, 4)
    return y.view(torch.int8), sw.reshape(-1, 4)


__all__ = ["TILELANG_AVAILABLE", "quantize_nvfp4_tl"]
