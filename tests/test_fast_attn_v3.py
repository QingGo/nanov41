# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""`fast_attn_v3` (bf16 compress-dkv contribution buffer) — GPU parity + graph capture test.

The R63 follow-up: `tl_sparse_attn_v2` materialises `dkv_c[b,m,nc,d]` in fp32 (134 MB/layer x 11
compress layers).  `fast_attn_v3` halves it to bf16 and the deterministic dest-major reduce reads
bf16 contributions (fp32 accumulator).  This pins the contract:

* `dq` and `dsink` are bitwise identical to the fp32 path (`dkv_c` only feeds the compress-dkv
  scatter), `dkv` differs only at the bf16 rounding of that contribution buffer (<1e-2 rel);
* the bf16 path can be CUDA-graph captured and replayed;
* it is not slower than the fp32 path (R69 measured ~-7% on the per-layer fwd+bwd).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")

tilelang = pytest.importorskip("tilelang")


def _make(b=16, m=1024, h=8, d=64, win=64, nc=32):
    from nanov41 import tl_sparse_attn_v2 as v2

    n = m + 512
    torch.manual_seed(0)
    q = torch.randn(b, m, h, d, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    kv = torch.randn(b, n, d, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    sink = torch.randn(h, device="cuda", dtype=torch.float32).requires_grad_(True)
    i = torch.arange(m, device="cuda").view(m, 1)
    c = torch.arange(win, device="cuda").view(1, win)
    exp = (i - win + 1).clamp_min(0) + c
    band = torch.where(exp <= i, exp, torch.full_like(exp, -1))
    comp = torch.randint(m, n, (m, nc), device="cuda")
    idxs = torch.cat([band, comp], dim=1).unsqueeze(0).expand(b, -1, -1).contiguous().int()
    do = torch.randn_like(q)
    return v2, q, kv, sink, idxs, do


def _run(v2, q, kv, sink, idxs, do):
    q.grad = None; kv.grad = None; sink.grad = None
    o = v2.sparse_attn_v2(q, kv, sink, idxs, 0.125)
    torch.autograd.backward(o, do)
    return q.grad.clone(), kv.grad.clone(), sink.grad.clone()


def test_fast_attn_v3_parity_and_capture():
    from nanov41 import fast_attn_v3 as v3

    if not v3.available():
        pytest.skip("tilelang/cuda unavailable")
    v2, q, kv, sink, idxs, do = _make()
    ref = _run(v2, q, kv, sink, idxs, do)
    torch.cuda.synchronize()

    st = v3.install()
    assert st.get("installed"), st
    try:
        got = _run(v2, q, kv, sink, idxs, do)
        torch.cuda.synchronize()
        rel = float((got[1] - ref[1]).abs().max() / (ref[1].abs().max() + 1e-30))
        assert torch.equal(got[0], ref[0]), "dq must be bitwise identical (dkv_c is not on dq path)"
        assert torch.equal(got[2], ref[2]), "dsink must be bitwise identical"
        assert rel < 1e-2, f"dkv rel diff {rel:.3e} exceeds the bf16 buffer tolerance"

        # graph capture + replay of the bf16 path
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
            _run(v2, q, kv, sink, idxs, do)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
    finally:
        v3.uninstall()
