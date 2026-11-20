# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""FP8 head+CE v2 vs the bf16 path: same loss, ~3% gradient noise (CUDA only)."""
from __future__ import annotations

import os

import pytest
import torch

from nanov41.train_utils import fused_head_ce

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for fp8 GEMM")


def test_fp8_head_ce_matches_bf16_loss_and_grads_one_step():
    torch.manual_seed(0)
    b, s, d, v = 1, 1024, 384, 4096
    h = torch.randn(b, s, d, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(v, d, device="cuda", dtype=torch.float32) * 0.02
    ids = torch.randint(0, v, (b, s), device="cuda")
    outs = {}
    for tag, env in (("bf16", "0"), ("fp8", "2")):
        os.environ["NANOV41_HEAD_FP8"] = env
        h1 = h.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        loss = fused_head_ce(h1, w1, ids)
        loss.backward()
        outs[tag] = (float(loss.detach()), h1.grad, w1.grad)
    os.environ.pop("NANOV41_HEAD_FP8", None)
    lb, gh_b, gw_b = outs["bf16"]
    lf, gh_f, gw_f = outs["fp8"]
    assert abs(lf - lb) / max(abs(lb), 1e-9) < 1e-4
    rel_h = float((gh_f - gh_b).norm() / gh_b.norm())
    rel_w = float((gw_f - gw_b).norm() / gw_b.norm())
    assert rel_h < 0.08 and rel_w < 0.08, (rel_h, rel_w)
