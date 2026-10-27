# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""GPU tests for the fused TileLang FP8 cast kernels (CUDA; tested on sm_120 / RTX 5090).

Skipped automatically when TileLang or CUDA is not available. The kernels are expected to
produce the same e4m3 bytes as the equivalent PyTorch cast (scalar scale: ~100% equal;
per-row scale: same expectation).
"""

import pytest
import torch

pytest.importorskip("tilelang")

from experiments.tl_kernels import fp8_cast as F  # noqa: E402


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_cast_fp8_matches_torch_scalar_scale() -> None:
    _require_cuda()
    torch.manual_seed(0)
    m, n = 256, 512
    x = (torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    scale = F.tensor_amax(x) / 448.0
    out = F.cast_fp8(x, scale)
    ref = (x * (1.0 / scale)).to(torch.float8_e4m3fn)
    eq = (out.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
    assert eq == 1.0, f"only {eq:.4f} of bytes match the PyTorch cast"


def test_cast_fp8_row_matches_torch_row_scale() -> None:
    _require_cuda()
    torch.manual_seed(0)
    m, n = 256, 512
    x = (torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    scale_row = (x.abs().amax(dim=1) / 448.0).clamp_min(1e-8)
    out = F.cast_fp8_row(x, scale_row)
    ref = (x / scale_row[:, None]).to(torch.float8_e4m3fn)
    eq = (out.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
    assert eq == 1.0, f"only {eq:.4f} of bytes match the PyTorch row cast"
