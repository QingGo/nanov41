# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Experimental sm120 NVFP4 / FP8 kernels (optional, requires TileLang + sm_120).

These modules are a curated subset of the internal W2 kernel experiments. They were
measured on an RTX 5090 (sm_120) with torch 2.12.1+cu130 / tilelang 0.1.15; see
``experiments/tl_kernels/README.md`` for numbers, constraints and caveats.

Everything here is optional: ``import nanov41`` and the CPU test suite do not depend on
this package, and importing it without TileLang installed is safe (the kernels are then
unavailable; the pure-torch host quantizer still works).

Submodules::

    experiments.tl_kernels.nvfp4_gemm   # host quantizer + sm120 NVFP4 GEMM + NVFP4Linear
    experiments.tl_kernels.nvfp4_quant  # fused TileLang NVFP4 quantizer
    experiments.tl_kernels.fp8_cast     # fused TileLang FP8 cast (scalar / per-row scale)

Note: the GEMM function is exported as ``gemm_nvfp4`` (not ``nvfp4_gemm``) so that the
submodule stays importable as ``experiments.tl_kernels.nvfp4_gemm``.
"""

from .nvfp4_gemm import (
    FP4_LUT,
    FP4_MAX,
    NVFP4Linear,
    TILELANG_AVAILABLE,
    dequantize_nvfp4,
    nvfp4_gemm as gemm_nvfp4,
    quantize_nvfp4,
    swizzle_blockscaled_chunk_kmajor_scale_words,
)
from .nvfp4_quant import quantize_nvfp4_tl
from .fp8_cast import cast_fp8, cast_fp8_row, tensor_amax

__all__ = [
    "FP4_LUT",
    "FP4_MAX",
    "NVFP4Linear",
    "TILELANG_AVAILABLE",
    "cast_fp8",
    "cast_fp8_row",
    "dequantize_nvfp4",
    "gemm_nvfp4",
    "quantize_nvfp4",
    "quantize_nvfp4_tl",
    "swizzle_blockscaled_chunk_kmajor_scale_words",
    "tensor_amax",
]
