# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU tests for `nanov41/grad_ckpt.py` FP8 group checkpointing.

- group checkpoint without FP8 must reproduce the plain backward bit-exactly;
- with FP8 it must keep the forward/loss identical and stay close in gradient space;
- quantize/dequantize round-trip error stays within the fp8 e4m3 expectation.
"""
from __future__ import annotations

import torch

import nanov41.grad_ckpt as gck
from nanov41.configs import TINY
from nanov41.model import NanoV41, init_random_weights
from nanov41.patched_env import import_patched


def _tiny():
    cfg = dict(TINY)
    cfg["max_seq_len"] = 128
    cfg["max_batch_size"] = 1
    torch.manual_seed(0)
    m = NanoV41(cfg, param_dtype="float32")
    init_random_weights(m.inner, seed=0)
    m.eval()
    return m, cfg


def _step(m, ids, labels):
    m.zero_grad(set_to_none=True)
    loss = m(ids, labels)
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().clone()
                                  for n, p in m.named_parameters() if p.grad is not None}


def test_fp8_quant_dequant_roundtrip_cpu():
    torch.manual_seed(0)
    x = torch.randn(64, 4, 32) * 3.0
    q, s = gck._quant_rows(x)
    y = gck._dequant(q, s, torch.float32)
    rel = (x - y).abs() / x.abs().clamp_min(1e-6)
    # e4m3 has 3 mantissa bits -> ~6% max relative error
    assert rel.max().item() < 0.08, rel.max().item()
    assert torch.allclose(y, x, atol=0.2, rtol=0.1)


def test_fp8_save_hooks_loss_identical_grads_close_cpu():
    m, cfg = _tiny()
    ids = torch.randint(0, cfg["vocab_size"], (1, 64))
    labels = torch.randint(0, cfg["vocab_size"], (1, 64))
    l0, g0 = _step(m, ids, labels)
    M = import_patched()
    gck.install(M, fp8_save=True)
    try:
        l1, g1 = _step(m, ids, labels)
    finally:
        gck.uninstall()
    assert abs(l0 - l1) < 1e-5, "forward stays bf16/fp32: loss identical"
    rel = max(float((g0[n] - g1[n]).abs().max() / g0[n].abs().max().clamp_min(1e-12))
              for n in g0)
    assert rel < 0.25, rel
