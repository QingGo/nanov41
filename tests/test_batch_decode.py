# SPDX-License-Identifier: MIT
"""GPU regression test for continuous-batched graph decode (``nanov41/batch_decode.py``, M3).

Gated: CUDA + P2b checkpoint + pinned tokenizer + slow (model loads, one graph capture per batch
size).  Enable with ``NANOV41_SLOW_TESTS=1``.

Checks the property the serving runtime claims: a batched step is *bit-identical* per sequence to
the single-sequence graph runtime (M3 made the decode projections batch-invariant for exactly this
reason), including when several prompts share the batch.
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


def test_batched_decode_matches_single_sequence():
    from tokenizers import Tokenizer

    from nanov41.batch_decode import BatchDecoder
    from nanov41.decode_graph import enable_graph_decode
    from nanov41.eval.provider import ModelProvider

    tokz = Tokenizer.from_file(TOK)
    ids = tokz.encode("The sea was calm that morning, and the fishermen rowed out.",
                      add_special_tokens=False).ids
    prompts = [ids[:8], ids[3:11], ids[:8]]
    n_new = 4

    provider = ModelProvider(checkpoint=CKPT, dtype="bf16", device="cuda")
    dec = enable_graph_decode(provider.model, bf16_head=True, max_batch=3)
    dec.generate(prompts[0], 2)                 # JIT warmup
    torch.cuda.synchronize()

    ref = [dec.generate(p, n_new) for p in prompts]
    engine = BatchDecoder(dec, max_batch=3)
    engine.capture_all()
    res = engine.generate_batch_parallel(prompts, n_new)
    for i in range(len(prompts)):
        assert res["tokens"][i] == ref[i], (i, res["tokens"][i], ref[i])
    # rows with the same prompt must be bit-identical to each other
    assert res["tokens"][0] == res["tokens"][2]
