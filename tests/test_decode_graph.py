# SPDX-License-Identifier: MIT
"""GPU regression test for the graph-safe decode runtime (``nanov41/decode_graph.py``).

Gated: needs CUDA + the P2b checkpoint + the pinned tokenizer, and is slow (model loads + graph
capture).  Enable with ``NANOV41_SLOW_TESTS=1``.  It checks the two properties that make the graph
route trustworthy: the rewritten tensor-position forward is *bit-identical* to the eager native
stack, and the captured graph replays bit-identically to that forward.
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (os.environ.get("NANOV41_SLOW_TESTS") and os.environ.get("NANOV41_DS_TOKENIZER_JSON")
         and os.path.exists("/tmp/p2b_default.pt") and torch.cuda.is_available()),
    reason="needs CUDA + P2b checkpoint + tokenizer (NANOV41_SLOW_TESTS=1)",
)

CKPT = "/tmp/p2b_default.pt"
TOK = os.environ.get("NANOV41_DS_TOKENIZER_JSON", "")


def _providers():
    from nanov41.eval.provider import ModelProvider

    eager = ModelProvider(checkpoint=CKPT, dtype="bf16", device="cuda")
    from nanov41.native_decode import enable_native_decode

    enable_native_decode(eager.model, engram=True, moe=True, mhc=True, attn=True,
                         dtype=torch.bfloat16)
    # the graph runtime defaults also install the M2d/M3 decode kernels; the eager reference must
    # use the same code paths for the bit-identical assertions below
    from nanov41.linear_decode import enable_linear_decode
    from nanov41.mhc_decode_fast import enable_mhc_decode_fast
    from nanov41.moe_decode_fast import enable_moe_decode_fast
    from nanov41.rmsnorm_decode import enable_rmsnorm_decode

    enable_linear_decode(eager.model)
    enable_mhc_decode_fast(eager.model)
    enable_moe_decode_fast(eager.model)
    enable_rmsnorm_decode(eager.model)
    graph = ModelProvider(checkpoint=CKPT, dtype="bf16", device="cuda")
    from nanov41.decode_graph import enable_graph_decode

    decoder = enable_graph_decode(graph.model)
    return eager, graph, decoder


def test_graph_forward_matches_eager_and_graph_replay():
    from nanov41.decode_graph import graph_forward
    from nanov41.generate import reset_caches
    from nanov41.model import inner_forward

    eager, graph, decoder = _providers()
    toks = [260, 892, 294, 270]
    steps = 6
    from tokenizers import Tokenizer

    tokz = Tokenizer.from_file(TOK)
    ids = tokz.encode("The sea was calm that morning", add_special_tokens=False).ids
    prompt_ids = (ids * 6)[:32]
    ctx = len(prompt_ids)

    def eager_run():
        reset_caches(eager.model.inner)
        logits = inner_forward(eager.model.inner, torch.tensor([prompt_ids], device="cuda"),
                               start_pos=0, full_logits=True)
        caps = [logits[0, -1].detach().float().cpu()]
        cur = len(prompt_ids)
        for tok in toks[:steps]:
            logits = inner_forward(eager.model.inner, torch.tensor([[tok]], device="cuda"),
                                   start_pos=cur, full_logits=True)
            caps.append(logits[0, -1].detach().float().cpu())
            cur += 1
        return caps

    eager_caps = eager_run()

    # tensor-position forward, executed eagerly
    decoder.prefill(prompt_ids)
    graph_caps = []
    for i, tok in enumerate(toks[:steps]):
        decoder.token.fill_(tok)
        decoder.pos.fill_(ctx + i)
        graph_caps.append(graph_forward(graph.model, decoder.token, decoder.pos)[0, 0]
                           .detach().float().cpu())
    for a, b in zip(eager_caps[1:], graph_caps):
        assert torch.equal(a, b), (float((a - b).abs().max()),)

    # captured replay, same forced prefixes
    decoder.prepare(prompt_ids)
    replay_caps = []
    for i, tok in enumerate(toks[:steps]):
        decoder.set_step(tok, ctx + i)
        replay_caps.append(decoder.replay()[0, 0].detach().float().cpu())
    for a, b in zip(graph_caps, replay_caps):
        assert torch.equal(a, b), (float((a - b).abs().max()),)

    # greedy generation through the graph agrees with the eager stack
    decoder.prepare(prompt_ids)
    dec = decoder.generate(prompt_ids, steps)
    reset_caches(eager.model.inner)
    logits = inner_forward(eager.model.inner, torch.tensor([prompt_ids], device="cuda"),
                           start_pos=0, full_logits=True)
    nxt = logits[0, -1]
    eager_tokens = []
    cur = len(prompt_ids)
    for i in range(steps):
        tok = int(nxt.argmax())
        eager_tokens.append(tok)
        if i == steps - 1:
            break
        nxt = inner_forward(eager.model.inner, torch.tensor([[tok]], device="cuda"),
                            start_pos=cur, full_logits=True)[0, -1]
        cur += 1
    assert dec == eager_tokens
