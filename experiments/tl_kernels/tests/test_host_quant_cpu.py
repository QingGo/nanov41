# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU contract tests for the pure-torch NVFP4 host quantizer.

These run without TileLang, CUDA or an sm_120 GPU: ``nvfp4_gemm.quantize_nvfp4`` and
``dequantize_nvfp4`` are pure torch. They pin the quantizer/dequantizer contract and the
row-padding regression (scale words are padded to 128-row blocks, so dequantization must
trim the padding instead of asserting an exact row count).
"""

import pytest
import torch

from experiments.tl_kernels import fp8_cast
from experiments.tl_kernels import nvfp4_gemm as G


def test_tilelang_is_optional() -> None:
    """Importing the package without TileLang must be safe; kernels raise a clear error."""
    assert isinstance(G.TILELANG_AVAILABLE, bool)
    assert isinstance(fp8_cast.TILELANG_AVAILABLE, bool)
    if not G.TILELANG_AVAILABLE:
        a = torch.zeros(128, 256, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="tilelang"):
            G.nvfp4_gemm(a, a.clone())
    if not fp8_cast.TILELANG_AVAILABLE:
        with pytest.raises(RuntimeError, match="tilelang"):
            fp8_cast.cast_fp8(torch.zeros(32, 128, dtype=torch.bfloat16), 1.0)


@pytest.mark.parametrize("rows,k", [(128, 256), (128, 384), (128, 512), (64, 384)])
def test_roundtrip_relative_error(rows: int, k: int) -> None:
    """FP4 round-trip (per-16 E4M3 scale) stays within a sane relative error; rows=64 is the
    row-padding regression case (scale words get padded to a 128-row block)."""
    torch.manual_seed(0)
    x = torch.randn(rows, k, dtype=torch.bfloat16) * 0.5
    packed, words = G.quantize_nvfp4(x)
    kpad = (k + 255) // 256 * 256
    assert packed.shape == (rows, kpad // 2)
    deq = G.dequantize_nvfp4(packed, words, rows, kpad)
    assert deq.shape == (rows, kpad)
    ref = torch.nn.functional.pad(x.float(), (0, kpad - k))
    rel = (deq - ref).norm() / ref.norm().clamp_min(1e-8)
    assert rel.item() < 0.15, f"relative error {rel.item():.3f} too high"


def test_quantize_is_deterministic() -> None:
    torch.manual_seed(1)
    x = torch.randn(128, 384, dtype=torch.bfloat16)
    p1, w1 = G.quantize_nvfp4(x)
    p2, w2 = G.quantize_nvfp4(x)
    assert torch.equal(p1, p2)
    assert torch.equal(w1, w2)


def test_zero_group_roundtrip() -> None:
    """An all-zero group must not divide by zero and must dequantize back to zero."""
    x = torch.zeros(128, 256, dtype=torch.bfloat16)
    packed, words = G.quantize_nvfp4(x)
    deq = G.dequantize_nvfp4(packed, words, 128, 256)
    assert torch.isfinite(deq).all()
    assert deq.abs().max().item() == 0.0


def test_dequantize_rejects_inconsistent_word_count() -> None:
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    packed, words = G.quantize_nvfp4(x)
    with pytest.raises(ValueError):
        G.dequantize_nvfp4(packed, words[:-1], 128, 256)


def test_padding_uses_documented_multiple() -> None:
    """K padding is to a multiple of 256 (the kernel's scale-block width)."""
    x = torch.randn(128, 384, dtype=torch.bfloat16)
    packed, words = G.quantize_nvfp4(x)
    assert packed.shape[1] == 512 // 2
    assert words.shape[1] == 4
