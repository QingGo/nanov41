# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Regression test for the checkpoint/buffer autograd retention (feasibility report §3m).

Activation-checkpoint replay writes grad-tracking activations into registered buffers (window /
compressed KV, indexer k-cache). Autograd attaches CopySlices nodes to those persistent buffers and
the step's graph stays alive across steps. `detach_leaked_buffers` must release it: the number of
live 4-stream activation tensors has to stop growing after each step.
"""
from __future__ import annotations

import gc

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


def _live_4d(prefix):
    gc.collect()
    return sum(1 for o in gc.get_objects()
               if torch.is_tensor(o) and getattr(o, "ndim", 0) == 4
               and tuple(o.shape)[:3] == prefix)


def test_detach_leaked_buffers_stops_cross_step_growth_cpu():
    m, cfg = _tiny()
    ids = torch.randint(0, cfg["vocab_size"], (1, 64))
    labels = torch.randint(0, cfg["vocab_size"], (1, 64))
    M = import_patched()
    gck.install(M)
    counts, detached = [], []
    try:
        for _ in range(3):
            m.zero_grad(set_to_none=True)
            loss = m(ids, labels)
            loss.backward()
            del loss
            counts.append(_live_4d((1, 64, 4)))
            detached.append(gck.detach_leaked_buffers(m, M))
            counts[-1] = _live_4d((1, 64, 4))
    finally:
        gck.uninstall()
    assert detached[0] > 0, "expected at least one buffer to carry an autograd graph"
    assert counts[-1] == counts[1], f"live tensors still growing: {counts}"
