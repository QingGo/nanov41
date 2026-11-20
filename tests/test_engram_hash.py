# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""P0: Engram hash parity and layout invariants.

The reference V4.1 Engram layout draws one prime bucket per (layer, n-gram size, head).  The
official config's `engram_num_embeddings` must equal the exact sum of those primes, otherwise some
hash ids point at masked rows.  These tests pin:

* the official V4.1-Flash layout (2 layers `[1,14]`, rows 384,006,168 / 384,016,682),
* the nano geometry presets in `nanov41.nano_engram` (rows = sum of primes; distinct primes),
* the hash algorithm itself against an independent re-implementation, including the chunked-prefill
  cache continuity and the dead-token (image span) mask,
* that `install()` removes the inference-mode decorator so hash ids are normal tensors during
  training.

Token-dependent tests are skipped unless the pinned DeepSeek-V4.1-Flash tokenizer is available:

    NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json pytest tests/test_engram_hash.py -q
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch

from nanov41.configs import CONFIGS
from nanov41.patched_env import import_patched

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
requires_tokenizer = pytest.mark.skipif(not TOKENIZER, reason="set NANOV41_DS_TOKENIZER_JSON")


def _patched():
    return import_patched()


# --------------------------------------------------------------------------- layout invariants
def test_official_layout_matches_reference_config():
    """The prime layout must reproduce ref/config.json's rows exactly (no masked hash ids)."""
    M = _patched()
    cfg = json.load(open(ROOT / "ref" / "config.json"))
    args = M.ModelArgs(**cfg)
    layout = M.EngramLayout.from_args(args)
    assert layout is not None
    assert layout.layer_ids == (1, 14)
    assert layout.num_embeddings == (384_006_168, 384_016_682)
    assert layout.max_ngram_size == 4 and layout.n_heads == 8 and layout.head_dim == 256
    for layer_idx in range(2):
        assert sum(sum(heads) for heads in layout.primes[layer_idx]) == layout.num_embeddings[layer_idx]


def test_presets_reproduce_design_rows_and_distinct_primes():
    from nanov41.nano_engram import ENGRAM_PRESETS, apply_preset

    expected_rows = {
        "d12": [2_000_136, 2_000_370],
        "d16": [4_000_112, 4_000_418],
        "d20": [8_000_152, 8_000_396],
        "tiny": [521],
    }
    for name, rows in expected_rows.items():
        cfg = apply_preset(CONFIGS["tiny"] if name == "tiny" else CONFIGS[name], name)
        M = _patched()
        layout = M.EngramLayout.from_args(M.ModelArgs(**cfg))
        assert layout is not None, name
        assert list(layout.num_embeddings) == rows, name
        seen: set[int] = set()
        for layer_idx in range(len(layout.layer_ids)):
            assert sum(sum(heads) for heads in layout.primes[layer_idx]) == rows[layer_idx]
            for per_ngram in layout.primes[layer_idx]:
                for prime in per_ngram:
                    assert prime not in seen, "prime buckets must never be reused"
                    seen.add(prime)
        assert len(seen) == (layout.max_ngram_size - 1) * layout.n_heads * len(layout.layer_ids)


def test_layout_disabled_by_default_in_scratch_configs():
    M = _patched()
    for name in ("tiny", "d8", "d12", "d16", "d20"):
        assert M.EngramLayout.from_args(M.ModelArgs(**CONFIGS[name])) is None, name


def test_hash_multipliers_are_odd_and_deterministic():
    _patched()
    import engram  # ref/engram.py is importable once patched_env set REF_DIR

    m1 = engram.compute_hash_multipliers((1, 4), 3, 99_092)
    m2 = engram.compute_hash_multipliers((1, 4), 3, 99_092)
    assert torch.equal(m1, m2)
    assert m1.dtype == torch.int64 and m1.shape == (2, 3)
    assert torch.all(m1 % 2 == 1)  # odd multipliers


# --------------------------------------------------------------------------- hash algorithm
def _two_layer_hash_state(ngram: int = 3, heads: int = 2, B: int = 1000):
    """Build a small NgramHashState (needs the pinned tokenizer for the compressed map)."""
    from nanov41.nano_engram import install, load_tokenizer_adapter

    install()
    M = _patched()
    adapter = load_tokenizer_adapter(TOKENIZER)
    cfg = dict(
        CONFIGS["tiny"],
        engram_layer_ids=[1],
        engram_num_embeddings=[1],  # unused by the hash state; only the primes matter here
        engram_max_ngram_size=ngram,
        engram_vocab_size=B,
        engram_n_heads=heads,
        engram_head_dim=32,
        engram_compressed_vocab_size=99_092,
        engram_pad_id=2,
    )
    args = M.ModelArgs(**cfg)
    layout = M.EngramLayout.from_args(args)
    state = M.NgramHashState(args, layout, adapter)
    return M, layout, state


def _reference_hash_ids(state, input_ids, start_pos=0, token_mask=None):
    """Independent re-implementation of the n-gram rolling-XOR hash (same buffers, fresh code)."""
    layout = state.layout
    batch, seqlen = input_ids.shape
    compressed = state.token_map[input_ids]
    if token_mask is not None:
        compressed = torch.where(token_mask, torch.full_like(compressed, state.DEAD), compressed)
    state.cache[:batch, start_pos:start_pos + seqlen] = compressed
    positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(layout.max_ngram_size):
        source = state.cache[:batch].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == state.DEAD)
        tokens.append(torch.where(blocked, torch.full_like(source, state.pad_id), source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(2) * state.multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % state.primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + state.offsets


@requires_tokenizer
def test_hash_ids_match_independent_implementation():
    torch.manual_seed(0)
    M, layout, state = _two_layer_hash_state()
    ids = torch.randint(0, CONFIGS["tiny"]["vocab_size"], (2, 8))
    got = state(ids, 0)
    want = _reference_hash_ids(state, ids, 0)
    assert got.shape == want.shape
    assert torch.equal(got, want)
    assert got.is_inference() is False  # install() unwrapped @torch.inference_mode
    # determinism across calls
    assert torch.equal(state(ids, 0), got)


@requires_tokenizer
def test_hash_ids_stay_inside_prime_buckets():
    M, layout, state = _two_layer_hash_state()
    ids = torch.randint(0, CONFIGS["tiny"]["vocab_size"], (3, 12))
    hashes = state(ids, 0)
    flat_primes = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    for layer_idx in range(len(layout.layer_ids)):
        for col in range(hashes.shape[-1]):
            lo = int(state.offsets[layer_idx, col])
            hi = lo + int(flat_primes[layer_idx][col])
            col_vals = hashes[..., layer_idx, col]
            assert int(col_vals.min()) >= lo and int(col_vals.max()) < hi, (layer_idx, col)


@requires_tokenizer
def test_hash_cache_continuity_prefill_then_chunk():
    """Hash ids for a chunk starting at start_pos>0 must match the full-sequence hashes."""
    torch.manual_seed(1)
    M, layout, state_full = _two_layer_hash_state()
    ids = torch.randint(0, CONFIGS["tiny"]["vocab_size"], (2, 10))
    full = state_full(ids, 0)

    M2, layout2, state_chunk = _two_layer_hash_state()
    head = state_chunk(ids[:, :4], 0)
    tail = state_chunk(ids[:, 4:], 4)
    assert torch.equal(head, full[:, :4])
    assert torch.equal(tail, full[:, 4:])


@requires_tokenizer
def test_dead_token_mask_blocks_ngrams():
    torch.manual_seed(2)
    M, layout, state = _two_layer_hash_state(ngram=3, heads=2)
    ids = torch.randint(0, CONFIGS["tiny"]["vocab_size"], (1, 8))
    unmasked = state(ids, 0)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 3] = False  # like an image span: no n-gram may cross it
    masked = state(ids, 0, mask)
    assert torch.equal(unmasked[:, :3], masked[:, :3])
    assert not torch.equal(unmasked[:, 3:], masked[:, 3:])
    # ids after the dead token must differ for at least the first 2 following positions
    assert not torch.equal(unmasked[:, 3], masked[:, 3])
    assert not torch.equal(unmasked[:, 4], masked[:, 4])
