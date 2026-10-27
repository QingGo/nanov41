# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""GPU tests for the fused TileLang NVFP4 quantizer and GEMM (sm_120, e.g. RTX 5090).

These are skipped automatically when TileLang or a suitable GPU is not available.
Reference numbers from the internal development run (torch 2.12.1+cu130 / tilelang
0.1.15, RTX 5090): scale words 100% identical to the torch quantizer, packed bytes ~94%
identical (E2M1 rounding ties), dequantized values agree within ~1e-2 absolute.
"""

import pytest
import torch

pytest.importorskip("tilelang")

from experiments.tl_kernels import nvfp4_gemm as G  # noqa: E402
from experiments.tl_kernels import quantize_nvfp4_tl  # noqa: E402


def _require_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (12, 0):
        pytest.skip(f"requires sm_120 (found sm_{major}{minor}); measured on RTX 5090")


def test_fused_quantizer_matches_torch_quantizer() -> None:
    _require_sm120()
    torch.manual_seed(0)
    rows, k = 8192, 384
    x = (torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    kpad = (k + 255) // 256 * 256

    q_ref, s_ref = G.quantize_nvfp4(x)
    q_tl, s_tl = quantize_nvfp4_tl(x)

    same_words = (s_ref.reshape(-1) == s_tl.reshape(-1)).float().mean().item()
    same_bytes = (q_ref.reshape(-1) == q_tl.reshape(-1)).float().mean().item()
    assert same_words == 1.0, f"scale words only {same_words:.4f} identical"
    assert same_bytes >= 0.9, f"packed bytes only {same_bytes:.4f} identical"

    d_ref = G.dequantize_nvfp4(q_ref, s_ref, rows, kpad)
    d_tl = G.dequantize_nvfp4(q_tl, s_tl, rows, kpad)
    assert (d_ref - d_tl).abs().max().item() < 5e-2


def test_nvfp4_gemm_end_to_end() -> None:
    _require_sm120()
    torch.manual_seed(0)
    m, n, k = 128, 256, 256
    a = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    out = G.nvfp4_gemm(a, b)
    ref = a.float() @ b.float().t()
    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-8)).item()
    assert rel < 0.2, f"FP4 GEMM relative error {rel:.3f} too high"
