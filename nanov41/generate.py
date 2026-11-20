# SPDX-License-Identifier: MIT
"""Incremental (KV-cache) greedy/sampled generation for NanoV41.

The official reference forward is already incremental: pass the not-yet-processed token slice plus
``start_pos`` and the per-layer ring/compressed KV caches advance.  This module wraps that path for
the nano model, resets the caches between independent prompts, and offers a no-cache reference loop
for parity testing.  Batch size 1 only (the eval/generation probe use case).

The cache tensors are the ones the training/chunked-prefill stack already uses:
window_kv_cache / compress_kv_cache / indexer k_cache / compressor state.  A prompt must fit in
``cfg['max_seq_len']``.
"""
from __future__ import annotations

import torch

from .model import inner_forward

__all__ = ["reset_caches", "generate_kv", "generate_no_cache"]


@torch.no_grad()
def reset_caches(model_or_inner) -> None:
    """Zero every cross-token cache; needed when starting an unrelated prompt."""
    inner = getattr(model_or_inner, "inner", model_or_inner)
    for layer in getattr(inner, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        for holder, names in ((attn, ("window_kv_cache", "compress_kv_cache")),
                              (getattr(attn, "indexer", None), ("k_cache",)),
                              (getattr(attn, "compressor", None), ("kv_state", "score_state"))):
            if holder is None:
                continue
            for name in names:
                buf = getattr(holder, name, None)
                if isinstance(buf, torch.Tensor):
                    buf.zero_()
    shared = getattr(inner, "shared_attn", None)
    if shared is not None:
        for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
            if hasattr(shared, name):
                setattr(shared, name, None)


def _pick(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_p, dim=-1) - sorted_p <= top_p
        probs = torch.zeros_like(probs)
        probs[idx[keep]] = sorted_p[keep]
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate_kv(model, prompt_ids, max_new_tokens: int = 32, temperature: float = 0.0,
                top_p: float = 1.0, eos_token_id: int | None = None,
                reset: bool = True) -> list[int]:
    """Greedy/temperature decode with the official incremental KV-cache path."""
    inner = getattr(model, "inner", model)
    if reset:
        reset_caches(inner)
    ids = list(prompt_ids)
    if not ids:
        raise ValueError("empty prompt")
    device = next(inner.parameters()).device
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    logits = inner_forward(inner, prompt, start_pos=0, full_logits=True)
    next_logits = logits[0, -1]
    cur_pos = len(ids)
    for _ in range(max_new_tokens):
        nxt = _pick(next_logits, temperature, top_p)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        logits = inner_forward(inner, torch.tensor([[nxt]], dtype=torch.long, device=device),
                               start_pos=cur_pos, full_logits=True)
        next_logits = logits[0, -1]
        cur_pos += 1
    return ids


@torch.no_grad()
def generate_no_cache(model, prompt_ids, max_new_tokens: int = 32, temperature: float = 0.0,
                      top_p: float = 1.0, eos_token_id: int | None = None) -> list[int]:
    """Reference loop: recompute the full prefix every step (slow; used to check the cache path)."""
    inner = getattr(model, "inner", model)
    device = next(inner.parameters()).device
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = inner_forward(inner, torch.tensor([ids], dtype=torch.long, device=device),
                               start_pos=0, full_logits=True)
        nxt = _pick(logits[0, -1], temperature, top_p)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return ids
