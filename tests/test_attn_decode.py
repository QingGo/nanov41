# SPDX-License-Identifier: MIT
"""CPU parity tests for the decode attention fast path (``nanov41/attn_decode.py``).

The Triton kernels are exercised separately on CUDA; here the torch mirrors are compared against
the reference ops they replace, and a whole TINY model runs a short decode with/without
``enable_attn_decode`` to check the patched forward matches the reference forward.
"""
from __future__ import annotations

import torch

from nanov41.attn_decode import (enable_attn_decode, rmsnorm_torch, rope_torch,
                                 sparse_attn_decode_torch)
from nanov41.configs import CONFIGS
from nanov41.model import NanoV41, init_random_weights
from nanov41.patched_env import import_patched, sparse_attn


def test_rmsnorm_and_rope_torch_match_reference():
    M = import_patched()
    torch.manual_seed(0)
    x = torch.randn(3, 5, 32)
    w = torch.randn(32)
    ref_norm = M.RMSNorm(32, 1e-6)
    with torch.no_grad():
        ref_norm.weight.copy_(w)
        got = rmsnorm_torch(x, w, 1e-6)
        exp = ref_norm(x)
    assert torch.equal(got, exp)

    M = import_patched()
    freqs = torch.polar(torch.ones(1, 4), torch.randn(1, 4))
    a = torch.randn(1, 1, 3, 8)
    b = a.clone()
    rope_torch(a[..., -8:], freqs)
    M.apply_rotary_emb(b[..., -8:], freqs)
    assert torch.equal(a, b)
    rope_torch(a[..., -8:], freqs, inverse=True)
    M.apply_rotary_emb(b[..., -8:], freqs, inverse=True)
    assert torch.equal(a, b)


def test_sparse_attn_decode_torch_matches_reference():
    torch.manual_seed(0)
    b, h, d, win = 1, 2, 8, 6
    for start_pos in (1, 4, 5, 11, 17):
        n = 3
        q = torch.randn(b, 1, h, d)
        window_kv = torch.randn(b, win, d)
        # reference window indices for this decode position
        ref_window_idxs = M_get_window_topk_idxs(win, b, 1, start_pos)
        compress_kv = torch.randn(b, n, d)
        # reference compress indices carry the +win offset
        compress_idxs = torch.tensor([[[win + 0, win + 2, -1]]])
        ref_kv = torch.cat([window_kv, compress_kv], dim=1)
        ref_idxs = torch.cat([ref_window_idxs, compress_idxs], dim=-1)
        sink = torch.randn(h)
        scale = d ** -0.5
        exp = sparse_attn(q, ref_kv, sink, ref_idxs, scale)
        got = sparse_attn_decode_torch(q, window_kv, compress_kv, compress_idxs, sink, scale,
                                       start_pos, win)
        torch.testing.assert_close(got, exp, atol=1e-6, rtol=0)


def M_get_window_topk_idxs(*args):
    M = import_patched()
    return M.get_window_topk_idxs(*args)


@torch.no_grad()
def _decode_logits(model, ids: list[int], n_prefill: int = 4) -> torch.Tensor:
    from nanov41.generate import reset_caches
    from nanov41.model import inner_forward

    inner = model.inner
    reset_caches(inner)
    logits = inner_forward(inner, torch.tensor([ids[:n_prefill]]), start_pos=0, full_logits=True)
    out = [logits[0, -1].clone()]
    cur = n_prefill
    for token in ids[n_prefill:]:
        logits = inner_forward(inner, torch.tensor([[token]]), start_pos=cur, full_logits=True)
        out.append(logits[0, -1].clone())
        cur += 1
    return torch.stack(out)


def test_attn_decode_full_model_parity():
    torch.manual_seed(0)
    model = NanoV41(CONFIGS["tiny"], param_dtype="float32")
    init_random_weights(model, seed=0)
    model.eval()
    ids = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59]
    ref = _decode_logits(model, ids)
    assert enable_attn_decode(model) > 0
    got = _decode_logits(model, ids)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=0)
    assert torch.equal(got.argmax(-1), ref.argmax(-1))


def test_attn_decode_falls_back_for_prefill():
    """start_pos == 0 and multi-token calls must hit the reference forward unchanged."""
    model = NanoV41(CONFIGS["tiny"], param_dtype="float32")
    init_random_weights(model, seed=0)
    model.eval()
    assert enable_attn_decode(model) == len(model.inner.layers)
    from nanov41.model import inner_forward

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = inner_forward(model.inner, ids, start_pos=0, full_logits=True)
    assert logits.shape == (1, 5, CONFIGS["tiny"]["vocab_size"])
    assert torch.isfinite(logits).all()
