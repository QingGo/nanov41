# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU equivalence test for `nanov41/chunked_prefill.py` (M3).

A sequence processed as two chunks (start_pos=0 then start_pos>0 with cached KV) must produce the
same second-half logits as a single full prefill. The window ring, the per-group compression, the
compressed-KV cache and the indexer k-cache all have to line up across the chunk boundary.
"""
from __future__ import annotations

import torch
import nanov41.grad_ckpt as gck

import nanov41.chunked_indexer as ci
import nanov41.chunked_prefill as cp
from nanov41.configs import TINY
from nanov41.model import NanoV41, init_random_weights
from nanov41.patched_env import import_patched


def _tiny_model(seq_len: int) -> NanoV41:
    cfg = dict(TINY)
    cfg["max_seq_len"] = seq_len
    cfg["max_batch_size"] = 1
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    return model


def test_chunked_prefill_matches_full_prefill_cpu():
    seq, chunk = 128, 64
    model = _tiny_model(seq)
    ids = torch.randint(0, model.vocab_size, (1, seq))
    with torch.no_grad():
        full = model(ids)
    M = import_patched()
    ci.install(M, chunk=32, mode="prefix-pad")
    cp.install(M)
    try:
        with torch.no_grad():
            model.inner(ids[:, :chunk], 0, None, None, full_logits=True)  # chunk 1: prefill
            h2 = model.inner(ids[:, chunk:], chunk, None, None, full_logits=True)  # chunk 2: +cache
    finally:
        cp.uninstall()
        ci.uninstall()
    assert h2.shape == full[:, chunk:].shape
    assert torch.isfinite(h2).all()
    diff = (h2.float() - full[:, chunk:].float()).abs()
    # small numerical differences are expected: sparse_attn sees a differently assembled key array
    assert diff.max().item() < 5e-2, f"max abs diff {diff.max().item():.3e}"
    assert diff.mean().item() < 5e-3


def _capture_sparse_attn(model, ids, chunk, n_layers):
    """Run full or chunked prefill and record every sparse_attn (kv, topk_idxs) call, per layer."""
    from nanov41.patched_env import import_patched

    M = import_patched()
    calls = []
    orig = M.sparse_attn
    state = {"li": 0}

    def wrap(q, kv, sink, topk, scale):
        li = state["li"]
        state["li"] = (li + 1) % n_layers
        calls.append({"li": li, "q": q.detach().clone(), "kv": kv.detach().clone(),
                      "topk": topk.detach().clone(), "kvlen": kv.size(1)})
        return orig(q, kv, sink, topk, scale)

    M.sparse_attn = wrap
    try:
        with torch.no_grad():
            for layer in model.inner.layers:
                for holder, name in ((layer.attn, "window_kv_cache"),
                                     (layer.attn, "compress_kv_cache"),
                                     (getattr(layer.attn, "indexer", None), "k_cache")):
                    if holder is not None and getattr(holder, name, None) is not None:
                        getattr(holder, name).zero_()
            parts, start = [], 0
            while start < ids.size(1):
                n = min(chunk, ids.size(1) - start)
                parts.append(model.inner(ids[:, start:start + n], start, None, None, full_logits=True))
                start += n
    finally:
        M.sparse_attn = orig
    return calls


def test_chunked_prefill_window_only_is_bitwise_exact_cpu():
    seq, chunk = 512, 128
    cfg = dict(TINY)
    n = cfg["n_layers"]
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "compress_ratios": [0] * n,
                "kv_source_layers": [], "index_source_layers": []})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()
    with torch.no_grad():
        for layer in model.inner.layers:
            layer.attn.window_kv_cache.zero_()
        full = model.inner(ids, 0, None, None, full_logits=True)
    cp.install(M)
    try:
        with torch.no_grad():
            for layer in model.inner.layers:
                layer.attn.window_kv_cache.zero_()
            parts = [model.inner(ids[:, c0:c0 + chunk], c0, None, None, full_logits=True)
                     for c0 in range(0, seq, chunk)]
            chunked = torch.cat(parts, dim=1)
    finally:
        cp.uninstall()
    assert torch.equal(chunked[:, chunk:], full[:, chunk:]), "window-only chunked prefill must be bitwise exact"


def test_chunked_prefill_topk_semantics_cpu(monkeypatch):
    """The window/compressed top-k sets and the window KV rows must be identical to full prefill.

    The remaining difference (checked by the tolerance test) is floating-point: chunked batches have
    different GEMM tile shapes, and MoE routing can amplify that. Here we assert the *semantics* are
    exact: same visible positions, same key values.
    """
    from nanov41.configs import D12

    seq, chunk = 256, 128
    cfg = dict(D12)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "n_activated_experts": 16})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    n_layers = len(model.inner.layers)
    win = cfg["window_size"]
    M = import_patched()
    monkeypatch.setenv("NANOV41_IDX_TIEBREAK", "1")   # validation aid: see chunked_indexer docstring
    ci.install(M, chunk=32, mode="prefix-pad")
    cp.install(M)
    try:
        # both runs use the same (tie-break enabled) indexer, so the comparison isolates the
        # window / compressed-KV / cache assembly itself
        full = _capture_sparse_attn(model, ids, seq, n_layers)
        ch = _capture_sparse_attn(model, ids, chunk, n_layers)
    finally:
        cp.uninstall()
        ci.uninstall()
    # chunk 2 calls are the second block of n_layers calls; layer 1/2 are the first sources
    for li in (1, 2):
        f, c = full[li], ch[n_layers + li]
        assert f["topk"].shape[-1] == c["topk"].shape[-1]
        # window part: normalize both to absolute positions, compare as sets (order is ascending)
        # chunked layout: window KV = cat([last win-1 keys in absolute order, this chunk's keys]),
        # so topk window slot c for query i is row i+c and row r is absolute position
        # chunk - win + 1 + r.
        fwin = torch.where(f["topk"][..., :win] >= 0, f["topk"][..., :win], -1)[:, chunk:]
        cwin = chunk - win + 1 + c["topk"][..., :win]
        assert torch.equal(fwin.sort(-1).values, cwin.sort(-1).values), f"layer{li} window idx"
        # window KV rows must be bitwise equal for every shared absolute position
        def chunked_local(p):
            return p - (chunk - win + 1)

        for qi in range(0, chunk, 31):
            fa = {int(p): f["kv"][0, int(p)] for p in fwin[0, qi] if p >= 0}
            ca = {int(p): c["kv"][0, chunked_local(int(p))] for j, p in enumerate(cwin[0, qi]) if p >= 0}
            for pos in set(fa) & set(ca):
                assert torch.equal(fa[pos], ca[pos]), f"layer{li} q{qi} window kv row {pos}"
        # compressed part: same compressed positions after removing the window offset
        fcp = torch.where(f["topk"][..., win:] >= f["kvlen"], f["topk"][..., win:] - f["kvlen"], -1)[:, chunk:]
        ccp = torch.where(c["topk"][..., win:] >= c["kvlen"], c["topk"][..., win:] - c["kvlen"], -1)
        assert torch.equal(fcp.sort(-1).values, ccp.sort(-1).values), f"layer{li} compress idx"


def test_chunked_prefill_bitwise_with_deterministic_tiebreak_cpu(monkeypatch):
    """With a width-independent indexer tie-break, chunked prefill == full prefill bitwise (dense).

    The indexer applies ReLU, so a randomly initialised model has many exact-zero scores; plain
    `topk` breaks those ties differently for a width-64 and a width-128 score tensor. With the
    deterministic tie-break (`NANOV41_IDX_TIEBREAK=1`, validation aid) the remaining window /
    compressed-KV / cache assembly is exact: the two paths agree bit for bit on the dense config.
    """
    from nanov41.configs import D12

    seq, chunk = 256, 128
    cfg = dict(D12)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "n_activated_experts": 16})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    ids = torch.randint(0, model.vocab_size, (1, seq))

    def zero_caches():
        for layer in model.inner.layers:
            for holder, name in ((layer.attn, "window_kv_cache"),
                                 (layer.attn, "compress_kv_cache"),
                                 (getattr(layer.attn, "indexer", None), "k_cache")):
                if holder is not None and getattr(holder, name, None) is not None:
                    getattr(holder, name).zero_()
            comp = getattr(layer.attn, "compressor", None)
            if comp is not None:
                for name in ("kv_state", "score_state"):
                    if getattr(comp, name, None) is not None:
                        getattr(comp, name).zero_()

    M = import_patched()
    monkeypatch.setenv("NANOV41_IDX_TIEBREAK", "1")
    ci.install(M, chunk=32, mode="prefix-pad")
    cp.install(M)
    try:
        with torch.no_grad():
            zero_caches()
            full = model.inner(ids, 0, None, None, full_logits=True)
            zero_caches()
            parts = [model.inner(ids[:, c0:c0 + chunk], c0, None, None, full_logits=True)
                     for c0 in range(0, seq, chunk)]
            chunked = torch.cat(parts, dim=1)
    finally:
        cp.uninstall()
        ci.uninstall()
    assert torch.equal(chunked[:, chunk:], full[:, chunk:]), "chunked prefill must match full prefill bitwise"


def test_chunked_bptt_backward_detaches_caches_cpu():
    """Chunked BPTT: backward per chunk must not walk into a previous chunk's graph.

    Cross-chunk KV context is detached by design (truncated BPTT), so the persistent caches must
    not carry autograd graphs between chunks; otherwise the second chunk's `.backward()` fails with
    "Trying to backward through the graph a second time" (seen on GPU before the fix).
    """
    from nanov41.model import inner_forward

    seq, chunk = 128, 64
    cfg = dict(TINY)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.train()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()
    ci.install(M, chunk=32, mode="prefix-pad")
    cp.install(M)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for c0, c1 in ((0, chunk), (chunk, seq)):
            opt.zero_grad(set_to_none=True)
            h = inner_forward(model.inner, ids[:, c0:c1], c0, full_logits=True)
            loss = h.float().pow(2).mean()
            loss.backward()
            assert torch.isfinite(torch.tensor(float(loss.detach())))
            for layer in model.inner.layers:
                for name in ("window_kv_cache", "compress_kv_cache"):
                    buf = getattr(layer.attn, name, None)
                    assert buf is None or getattr(buf, "grad_fn", None) is None, f"{name} carries graph"
                indexer = getattr(layer.attn, "indexer", None)
                if indexer is not None:
                    assert getattr(indexer.k_cache, "grad_fn", None) is None
            opt.step()
    finally:
        cp.uninstall()
        ci.uninstall()


def test_ffn_checkpointing_is_gradient_exact_under_chunked_bptt_cpu():
    """FFN-only activation checkpointing must reproduce chunked-BPTT gradients exactly.

    The FFN half of a block is a pure function of (x, attn_pre); unlike a whole-block checkpoint it
    does not lose the cross-block gradient path through `shared_attn` (compressed KV / index keys),
    which a full-block checkpoint drops by ~46% relative to the cached caches in `start_pos > 0`
    chunks.
    """
    import os

    from nanov41.configs import D12
    from nanov41.model import inner_forward

    seq, chunk = 256, 128
    cfg = dict(D12)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "n_activated_experts": 16})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.train()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()

    def grads(ckpt_mode):
        model.zero_grad(set_to_none=True)
        ci.install(M, chunk=32, mode="prefix-pad")
        cp.install(M)
        if ckpt_mode:
            gck.install(M, mode=ckpt_mode)
        try:
            with torch.no_grad():
                inner_forward(model.inner, ids[:, :chunk], 0, full_logits=True)  # warm caches
            h = inner_forward(model.inner, ids[:, chunk:], chunk, full_logits=True)
            h.float().pow(2).mean().backward()
        finally:
            if ckpt_mode:
                gck.uninstall()
            cp.uninstall()
            ci.uninstall()
        return {n: (None if p.grad is None else p.grad.clone())
                for n, p in model.named_parameters()}

    os.environ["NANOV41_IDX_TIEBREAK"] = "1"
    try:
        base = grads(None)
        ffn = grads("ffn")
    finally:
        os.environ.pop("NANOV41_IDX_TIEBREAK", None)
    for name, g in base.items():
        if g is None:
            continue
        assert torch.equal(g, ffn[name]), f"{name}: FFN checkpointing changed the gradient"


def test_state_checkpoint_is_gradient_exact_under_chunked_bptt_cpu():
    """Whole-block checkpointing with the shared_attn state threaded through the Function is exact.

    Two separate holes had to be plugged for `start_pos > 0` chunks:
      1. tensors published into the module-global `shared_attn` by earlier blocks (compressed KV,
         index keys) must be explicit checkpoint inputs/outputs, otherwise the consumer's replay
         treats them as leaves and drops the cross-block gradient path (~46% relative error);
      2. the mutable per-block caches (sliding-window ring / compressed-KV cache / index k-cache)
         must be saved and restored per block, otherwise a replay sees post-forward contents
         (~5e-4 error even on a window-only model).
    With both, a D12-dense two-chunk truncated-BPTT backward is bitwise identical to no checkpointing.
    """
    import os

    from nanov41.configs import D12
    from nanov41.model import inner_forward

    seq, chunk = 256, 128
    cfg = dict(D12)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "n_activated_experts": 16})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.train()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()

    def grads(ckpt_mode):
        model.zero_grad(set_to_none=True)
        ci.install(M, chunk=32, mode="prefix-pad")
        cp.install(M)
        if ckpt_mode:
            gck.install(M, mode=ckpt_mode)
        try:
            for c0, c1 in ((0, chunk), (chunk, seq)):
                h = inner_forward(model.inner, ids[:, c0:c1], c0, full_logits=True)
                h.float().pow(2).mean().backward()
        finally:
            if ckpt_mode:
                gck.uninstall()
            cp.uninstall()
            ci.uninstall()
        return {n: (None if p.grad is None else p.grad.clone())
                for n, p in model.named_parameters()}

    os.environ["NANOV41_IDX_TIEBREAK"] = "1"
    try:
        base = grads(None)
        state = grads("state")
    finally:
        os.environ.pop("NANOV41_IDX_TIEBREAK", None)
    for name, g in base.items():
        if g is None:
            continue
        assert torch.equal(g, state[name]), f"{name}: state-threaded checkpointing changed the gradient"


def test_partial_block_checkpointing_is_gradient_exact_cpu(monkeypatch):
    """NANOV41_CKPT_EVERY=2 (checkpoint every other block) must reproduce the gradients exactly.

    Partial checkpointing trades activation memory for recompute: only saved blocks pay the extra
    forward, so it is the Pareto knob for the memory budget (report §3s).
    """
    import os

    from nanov41.configs import D12
    from nanov41.model import inner_forward

    seq, chunk = 128, 64
    cfg = dict(D12)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1, "n_activated_experts": 16})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.train()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()

    def grads(ckpt_mode, every=1):
        model.zero_grad(set_to_none=True)
        ci.install(M, chunk=32, mode="prefix-pad")
        cp.install(M)
        if ckpt_mode:
            monkeypatch.setenv("NANOV41_CKPT_EVERY", str(every))
            gck.install(M, mode=ckpt_mode)
        try:
            for c0, c1 in ((0, chunk), (chunk, seq)):
                h = inner_forward(model.inner, ids[:, c0:c1], c0, full_logits=True)
                h.float().pow(2).mean().backward()
        finally:
            if ckpt_mode:
                gck.uninstall()
            cp.uninstall()
            ci.uninstall()
        return {n: (None if p.grad is None else p.grad.clone())
                for n, p in model.named_parameters()}

    os.environ["NANOV41_IDX_TIEBREAK"] = "1"
    try:
        base = grads(None)
        got = grads("state", every=2)
    finally:
        os.environ.pop("NANOV41_IDX_TIEBREAK", None)
        monkeypatch.delenv("NANOV41_CKPT_EVERY", raising=False)
    for name, g in base.items():
        if g is None:
            continue
        assert torch.equal(g, got[name]), f"{name}: partial checkpointing changed the gradient"
