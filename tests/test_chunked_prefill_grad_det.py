# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Chunked-BPTT gradients must be bitwise reproducible across identical passes (CUDA).

Regression test for the nondeterminism found while prototyping the per-chunk CUDA graphs
(report §3u): for every `start_pos > 0` chunk the window KV layout was not recognized by
`fast_attn`, so `sparse_attn` fell back to the reference torch implementation, whose kv gradient
goes through `index_select` backward -> atomic `index_add` and is therefore not bit-reproducible
(max relative gradient diff ~0.2-0.3 on ~1e-6-magnitude params, i.e. last-bit noise amplified by
near-zero grads).  The fix gives `chunked_prefill` an absolute-order window layout that
`fast_attn`'s right-band plan handles (deterministic), and this test pins it:

1. two identical passes produce **bitwise identical** gradients for every parameter,
2. all `start_pos > 0` chunks really take the fast path (no reference fallback).

CUDA-only: the reference implementation is what is nondeterministic on CPU it is exact by
construction, so this is a GPU regression test.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only determinism test")


def _zero_caches(model) -> None:
    for layer in model.inner.layers:
        attn = layer.attn
        for holder, name in ((attn, "window_kv_cache"), (attn, "compress_kv_cache"),
                             (getattr(attn, "indexer", None), "k_cache")):
            if holder is not None and getattr(holder, name, None) is not None:
                getattr(holder, name).zero_()
        comp = getattr(attn, "compressor", None)
        if comp is not None:
            for name in ("kv_state", "score_state"):
                if getattr(comp, name, None) is not None:
                    getattr(comp, name).zero_()
        sa = getattr(attn, "shared_attn", None)
        if sa is not None:
            for name in ("compress_kv", "index_k", "topk_idxs", "window_kv"):
                t = getattr(sa, name, None)
                if torch.is_tensor(t):
                    t.zero_()


def test_chunked_prefill_gradients_bitwise_reproducible_on_cuda():
    from nanov41 import chunked_indexer as ci
    from nanov41 import chunked_prefill as cp
    from nanov41 import fast_attn
    from nanov41.configs import D12
    from nanov41.model import NanoV41, init_random_weights, inner_forward
    from nanov41.patched_env import import_patched

    # Production d12 shapes (dim 384, 8 heads x head_dim 64, window 64, 16 experts) with a tiny
    # vocab so the CE over full logits stays cheap; seq 16K / chunk 8K = two chunks, enough to
    # exercise the start_pos>0 window layout (TINY's 64-wide dims change so many kernel tile
    # shapes that it is not representative and has its own shape-dependent nondeterminism).
    seq, chunk = 16384, 8192
    cfg = dict(D12)
    cfg.update({"vocab_size": 512, "max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="bfloat16")
    init_random_weights(model.inner, seed=0)
    model = model.cuda().train()
    vocab = cfg["vocab_size"]
    ids = torch.randint(0, vocab, (1, seq), device="cuda")
    params = [p for p in model.parameters() if p.requires_grad]

    M = import_patched()
    prev_sparse = fast_attn.install(M)
    ci.install(M, chunk=64, mode="prefix-pad")
    cp.install(M)
    try:
        def one_pass():
            _zero_caches(model)
            gs = [p.grad for p in params if p.grad is not None]
            if gs:
                torch._foreach_zero_(gs)
            for c0 in range(0, seq, chunk):
                c1 = min(seq, c0 + chunk)
                h = inner_forward(model.inner, ids[:, c0:c1], c0, full_logits=True)
                logits = h[:, :-1].reshape(-1, vocab).float()
                tgt = ids[:, c0 + 1:c1].reshape(-1)
                loss = torch.nn.functional.cross_entropy(logits, tgt) * ((c1 - c0) / seq)
                loss.backward()
            return [p.grad.detach().clone() if p.grad is not None else None for p in params]

        fast_attn.reset_stats()
        one_pass()
        stats = fast_attn.get_stats()
        n_layers = len(model.inner.layers)
        assert stats["fallback_calls"] == 0, f"chunked window layout fell back: {stats}"
        assert stats["layout_fail"] == 0, f"chunked window layout not recognized: {stats}"
        assert stats["fast_calls"] >= n_layers * (seq // chunk - 1), stats

        g1 = one_pass()
        g2 = one_pass()
        bad = [i for i, (a, b) in enumerate(zip(g1, g2, strict=True))
               if (a is None) != (b is None) or (a is not None and not torch.equal(a, b))]
        assert not bad, f"{len(bad)}/{len(params)} parameter gradients are not bitwise reproducible"
    finally:
        cp.uninstall()
        ci.uninstall()
        M.sparse_attn = prev_sparse
