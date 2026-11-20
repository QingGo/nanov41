# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU equivalence for `nanov41/grad_ckpt.py`: recomputation must produce identical gradients."""
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


def test_grad_ckpt_matches_gradients_cpu():
    m, cfg = _tiny()
    ids = torch.randint(0, cfg["vocab_size"], (1, 64))
    labels = torch.randint(0, cfg["vocab_size"], (1, 64))

    def grads_after_step():
        m.zero_grad(set_to_none=True)
        loss = m(ids, labels)
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None}, float(loss.detach())

    base, base_loss = grads_after_step()
    M = import_patched()
    gck.install(M)
    try:
        rec, rec_loss = grads_after_step()
    finally:
        gck.uninstall()
    assert abs(base_loss - rec_loss) < 1e-6
    for name in base:
        assert torch.allclose(base[name], rec[name], atol=1e-6, rtol=1e-4), name
