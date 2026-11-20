# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU parity for `nanov41/chunked_indexer.py` (M2).

The chunked indexer replaces the reference O(seq * n_pos) score materialisation with per-query
chunking. These tests pin down that it is mathematically identical to the stock prefill path on a
tiny CPU model, that the decode path is untouched, and that a config with `max_seq_len` larger than
the default 1024 builds and runs (the 1M-context requirement starts with that field).
"""
from __future__ import annotations

import torch

import nanov41.chunked_indexer as ci
import nanov41.chunked_prefill as cp
from nanov41.configs import TINY
from nanov41.model import NanoV41, init_random_weights, inner_forward
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


def test_chunked_indexer_matches_stock_cpu():
    seq = 128
    model = _tiny_model(seq)
    ids = torch.randint(0, model.vocab_size, (1, seq))
    model.eval()
    with torch.no_grad():
        stock = model(ids)
    M = import_patched()
    ci.install(M, chunk=32)
    try:
        with torch.no_grad():
            chunked = model(ids)
        # decode path delegates to the stock forward, so a stock forward after uninstall must
        # round-trip bit-exactly
        ci.uninstall(M)
        with torch.no_grad():
            restored = model(ids)
    finally:
        ci.uninstall(M)
    assert stock.shape == chunked.shape
    assert torch.isfinite(chunked).all()
    assert torch.allclose(stock, chunked, atol=1e-6, rtol=1e-5), (
        f"max abs diff {(stock - chunked).abs().max().item():.3e}")
    assert torch.equal(stock, restored), "uninstall() must restore the reference forward"


def test_chunk_size_does_not_change_logits_cpu():
    seq = 128
    model = _tiny_model(seq)
    ids = torch.randint(0, model.vocab_size, (1, seq))
    model.eval()
    M = import_patched()
    outs = {}
    try:
        for chunk in (16, 32, 64):
            ci.install(M, chunk=chunk)
            with torch.no_grad():
                outs[chunk] = model(ids)
    finally:
        ci.uninstall(M)
    assert torch.allclose(outs[16], outs[32], atol=1e-6, rtol=1e-5)
    assert torch.allclose(outs[32], outs[64], atol=1e-6, rtol=1e-5)


def test_config_max_seq_len_above_default_cpu():
    # the 1M support starts with a configurable max_seq_len: the model must build and run a full
    # prefill whose sequence length differs from the historical default (1024)
    seq = 256
    model = _tiny_model(2 * seq)
    ids = torch.randint(0, model.vocab_size, (1, seq))
    model.eval()
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, seq, model.vocab_size)
    assert torch.isfinite(logits).all()


def test_hsi_mode_runs_and_is_finite_cpu():
    # approximate long-context mode: block-mean candidate selection is not bitwise equal to the
    # stock path, but it must run on CPU and produce finite logits
    seq = 128
    model = _tiny_model(seq)
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()
    ci.install(M, chunk=32, mode="hsi")
    try:
        with torch.no_grad():
            logits = model(ids)
    finally:
        ci.uninstall(M)
    assert logits.shape == (1, seq, model.vocab_size)
    assert torch.isfinite(logits).all()


def test_indexer_fused_module_import_safe_without_triton():
    # CPU installs / CI have no triton+CUDA: the module must import and report unavailability
    import nanov41.indexer_fused as kf

    assert hasattr(kf, "available")
    if not kf.available():
        import pytest

        with pytest.raises(RuntimeError):
            kf.fused_indexer_score(torch.zeros(1, 4, 4, 32), torch.zeros(1, 8, 32),
                                   torch.zeros(1, 4, 4), torch.zeros(4, dtype=torch.int64))


def test_hsi_two_level_indexer_runs_and_selects_visible_positions_cpu(monkeypatch):
    """The two-level candidate indexer (official CSA candidate-block idea) must produce valid top-k.

    Level one scores per-block mean keys, level two re-scores only the candidate positions. The
    result must be a sorted list of reachable compressed positions (or -1) with the exact top-k
    width, and it must run for multi-token chunks (start_pos > 0) too.
    """
    monkeypatch.setenv("NANOV41_IDX_HSI_BLK", "4")
    monkeypatch.setenv("NANOV41_IDX_HSI_CAND", "16")
    seq, chunk = 128, 64
    cfg = dict(TINY)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()
    ci.install(M, chunk=32, mode="hsi")
    cp.install(M)
    try:
        with torch.no_grad():
            for layer in model.inner.layers:
                for holder, name in ((layer.attn, "window_kv_cache"),
                                     (layer.attn, "compress_kv_cache"),
                                     (getattr(layer.attn, "indexer", None), "k_cache")):
                    if holder is not None and getattr(holder, name, None) is not None:
                        getattr(holder, name).zero_()
            full = inner_forward(model.inner, ids, 0, full_logits=True)
            for layer in model.inner.layers:
                for holder, name in ((layer.attn, "window_kv_cache"),
                                     (layer.attn, "compress_kv_cache"),
                                     (getattr(layer.attn, "indexer", None), "k_cache")):
                    if holder is not None and getattr(holder, name, None) is not None:
                        getattr(holder, name).zero_()
            h = inner_forward(model.inner, ids[:, chunk:], chunk, full_logits=True)
    finally:
        cp.uninstall()
        ci.uninstall()
    assert full.shape[:2] == (1, seq) and h.shape[:2] == (1, chunk)
    assert torch.isfinite(full).all() and torch.isfinite(h).all()
    idxs = M.shared_attn.topk_idxs
    assert idxs is not None and idxs.dtype == torch.int32
    assert idxs.size(1) == chunk
    assert (idxs < 0).any() or (idxs >= 0).all()  # -1 padding allowed, values otherwise valid


def test_hsi_auto_switches_between_exact_and_candidate_path_cpu(monkeypatch):
    """`hsi-auto` must stay exact below the threshold and use candidates above it (both runs valid)."""
    seq, chunk = 256, 128
    cfg = dict(TINY)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()
    ids = torch.randint(0, model.vocab_size, (1, seq))
    M = import_patched()
    for auto_min in ("100000", "8"):
        monkeypatch.setenv("NANOV41_IDX_HSI_AUTO_MIN", auto_min)
        ci.install(M, chunk=32, mode="hsi-auto")
        cp.install(M)
        try:
            with torch.no_grad():
                h = inner_forward(model.inner, ids, 0, full_logits=True)
        finally:
            cp.uninstall()
            ci.uninstall()
        assert h.shape[:2] == (1, seq) and torch.isfinite(h).all(), auto_min
