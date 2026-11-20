from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("NANOV41_SLOW_TESTS"),
    reason="heavy full-model parity tests; set NANOV41_SLOW_TESTS=1 to run",
)

CKPT = "/tmp/p2b_default.pt"


def _load_pair():
    if not os.environ.get("NANOV41_DS_TOKENIZER_JSON") or not os.path.exists(CKPT):
        pytest.skip("P2b checkpoint/tokenizer not available")
    from nanov41.model import NanoV41
    from nanov41.nano_engram import install, load_tokenizer_adapter

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    install()
    tok = load_tokenizer_adapter(os.environ["NANOV41_DS_TOKENIZER_JSON"])
    a = NanoV41(ck["cfg"], tokenizer=tok, param_dtype="float32")
    b = NanoV41(ck["cfg"], tokenizer=tok, param_dtype="float32")
    a.load_state_dict(ck["state_dict"], strict=True)
    b.load_state_dict(ck["state_dict"], strict=True)
    a.eval(); b.eval()
    return ck["cfg"], a, b


def test_moe_small_m_matches_reference():
    from nanov41.moe_decode import enable_moe_decode

    cfg, ref, mod = _load_pair()
    assert enable_moe_decode(mod, max_tokens=8, dtype=torch.float32) > 0
    ids = torch.randint(0, cfg["vocab_size"], (1, 4))
    with torch.no_grad():
        a = ref.logits_for(ids)
        b = mod.logits_for(ids)
    assert torch.allclose(a, b, atol=2e-4, rtol=0)


def test_engram_decode_matches_dequantized_reference():
    from nanov41.engram_decode import (enable_engram_decode, gather_dequant_torch,
                                       quantize_table_ue8m0)

    cfg, ref, mod = _load_pair()
    # reference: replace each Engram table by its FP8-dequantized bf16/float version
    for layer in ref.inner.layers:
        eng = getattr(layer, "engram", None)
        if eng is None:
            continue
        q, s = quantize_table_ue8m0(eng.embed.weight.detach())
        eng.embed.weight.data.copy_(gather_dequant_torch(q, s, torch.arange(q.shape[0])).float())
    assert enable_engram_decode(mod) == 2
    ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        a = ref.logits_for(ids)
        b = mod.logits_for(ids)
    assert torch.allclose(a, b, atol=1e-5, rtol=0)
