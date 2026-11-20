# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Contract test for the pinned DeepSeek-V4.1-Flash tokenizer (reused as-is, not vendored).

The tokenizer is *not* trained by this project: the 129,280-token DeepSeek V4.1-Flash BPE is
reused verbatim because (a) the model config vocab is 129,280, (b) Engram hashes over a
normalized/compressed token map whose size must be **99,092**, and (c) the future OPD teacher API
returns top-20 token ids in exactly this vocabulary.  Only the *data pipeline* (tokenize/dedupe/
pack) and the eval harness remain to be built.

To run:

    curl -L https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/tokenizer.json \\
        -o /tmp/dsv41_tokenizer.json
    NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json pytest tests/test_tokenizer_engram.py

Pinned artifact (2026-11-06): size 6,367,257 bytes,
sha256 c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b.
The older DeepSeek-V3 tokenizer does NOT satisfy this contract (vocab 128,815, compressed 98,627).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NANOV41_DS_TOKENIZER_JSON"),
    reason="set NANOV41_DS_TOKENIZER_JSON to the pinned DeepSeek-V4.1-Flash tokenizer.json",
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_engram_module():
    spec = importlib.util.spec_from_file_location("_ref_engram", ROOT / "ref" / "engram.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tokenizer_contract():
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(os.environ["NANOV41_DS_TOKENIZER_JSON"])

    # 1) the model/EMBEDDING/head and OPD id space
    assert tok.get_vocab_size() == 129_280, tok.get_vocab_size()
    # 2) engram_pad_id = 2 in ref/config.json is the DeepSeek pad token
    assert tok.id_to_token(2) == "<｜▁pad▁｜>"
    # 3) Engram compressed-token map size (NFKC/lowercase/whitespace-normalized)
    engram = _load_engram_module()

    class _TokenizerWrapper:  # engram.py expects a HF-style tokenizer
        backend_tokenizer = tok

    wrapper = _TokenizerWrapper()
    wrapper.__class__.__len__ = lambda self: tok.get_vocab_size()
    _, compressed = engram.build_compressed_token_map(wrapper)
    assert compressed == 99_092, compressed
