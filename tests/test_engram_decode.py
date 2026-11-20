from __future__ import annotations

import os

import pytest
import torch

from nanov41.engram_decode import (DEAD, EngramDecodeCache, decode_hash_ids,
                                   engram_gate_decode, fp8_scale_to_float,
                                   gather_dequant_torch, quantize_table_ue8m0)


def test_gather_dequant_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    q, s = quantize_table_ue8m0(w)
    ids = torch.randint(0, 64, (7,))
    got = gather_dequant_torch(q, s, ids)
    vals = q.float()[ids]
    sc = fp8_scale_to_float(s[ids]).repeat_interleave(32, dim=-1)
    ref = (vals * sc).to(torch.bfloat16)
    assert torch.equal(got, ref)
    # quantisation error is bounded by the expected e4m3 + power-of-two block scale
    rel = ((got.float() - w[ids].float()).abs() / w[ids].float().abs().clamp_min(1e-6)).mean()
    assert rel < 0.06


def test_decode_hash_matches_reference():
    tok_path = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
    if not tok_path:
        pytest.skip("tokenizer env not set")
    ckpt = "/tmp/p2b_default.pt"
    if not os.path.exists(ckpt):
        pytest.skip("P2b checkpoint not available")
    from nanov41.model import NanoV41
    from nanov41.nano_engram import install, load_tokenizer_adapter
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    install()
    tok = load_tokenizer_adapter(tok_path)
    model = NanoV41(cfg, tokenizer=tok, param_dtype="float32")
    state = model.inner.engram_hash
    torch.manual_seed(0)
    ids = torch.randint(0, cfg["vocab_size"], (2, 37))
    ref = state(ids, 0)  # [B,L,layers,cols]
    compressed = state.token_map[ids]
    cache = torch.full((2, cfg["max_seq_len"]), int(state.pad_id), dtype=torch.int64)
    got = decode_hash_ids(cache, 0, compressed, state.multipliers,
                          state.primes, state.offsets, int(state.pad_id))
    assert torch.equal(got, ref), (got.shape, ref.shape, (got != ref).sum().item())


def test_gate_decode_reference():
    torch.manual_seed(0)
    b, l, hc, dim, heads = 2, 3, 2, 32, 2
    rows = 41
    w = torch.randn(rows, dim, dtype=torch.bfloat16)
    wq, ws = quantize_table_ue8m0(w)
    ids = torch.randint(0, rows, (b, l, heads))
    x = torch.randn(b, l, hc, dim, dtype=torch.bfloat16)
    wkv = torch.randn((hc + 1) * dim, heads * dim, dtype=torch.bfloat16)
    qw = torch.randn(hc, dim, dtype=torch.bfloat16)
    kw = torch.randn(hc, dim, dtype=torch.bfloat16)
    got = engram_gate_decode(x, ids, wq, ws, wkv, qw, kw, hc, dim, 1e-6, 1.0)

    # reference math is the official Engram.forward: bf16 gather/wkv, float32 gate + residual
    embed = gather_dequant_torch(wq, ws, ids).to(wkv.dtype).flatten(-2)
    kv = torch.nn.functional.linear(embed, wkv)
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim)); value = value.float().unsqueeze(-2)
    gw = qw.float() * kw.float()
    h = x.float()
    h2 = h.square().mean(-1); k2 = key.square().mean(-1)
    rstd = torch.rsqrt(h2 + 1e-6) * torch.rsqrt(k2 + 1e-6)
    dot = (h * gw * key).sum(-1) * rstd * (dim ** -0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1.0).sqrt(), dot))
    ref = (h + gate.unsqueeze(-1) * value).to(x.dtype)
    assert torch.equal(got, ref)
