# SPDX-License-Identifier: MIT
"""Graph-safe decode runtime (M1), batched in M3.

The eager decode step is CPU launch-bound (`docs/decode-200tps-plan.md`): ~49 ms of Python/ATen
dispatch around ~3.1 ms of GPU work.  A CUDA graph removes the dispatch, but the reference decode
forward is not capturable: it derives Python ints from ``start_pos`` (ring-buffer slots, RoPE rows,
compressor group index, indexer ``topk`` count, hash-cache positions) and slices caches to
position-dependent lengths.

This module re-implements the *decode* forward with

* a static ``pos`` device tensor instead of the Python int (one row per sequence),
* fully static shapes (compressed top-k padded to ``index_topk`` with -1, indexer scores over the
  whole cache masked by position, fixed ``topk``/``sort``),
* persistent cross-layer buffers (the group's compressed cache / top-k buffer are written in
  place, never rebound),
* unconditional compressor writes (safe: an in-progress group index is never read before it is
  overwritten when the group completes),
* re-used Triton kernels from ``attn_decode`` (RMSNorm / RoPE / fused sparse attention, the last
  one now reading ``pos`` from a tensor so graph replay sees the updated position).

M3: every step function handles ``B`` sequence rows (``token`` [B,1], ``pos`` [B]) and indexes the
per-sequence state rows of the batched caches, so one captured graph serves a batch of B
independent requests.  ``GraphDecoder`` keeps the original single-sequence API on top of the same
code (B=1).

It keeps the native inference patches (FP8 Engram tables, small-M MoE, fused mHC) and all
``Linear`` projections exactly as the eager native path; ``GraphDecoder`` prefills eagerly, captures
the step once, re-prefills, then serves ``set_step()/replay()`` (or ``generate()``).
"""
from __future__ import annotations

import torch

from .attn_decode import (_rmsnorm, blockdiag_wo_a, rope_row, rope_rows_batch,
                          sparse_attn_decode_batch, sparse_attn_decode)


# --------------------------------------------------------------------------------- hash ids
def graph_hash_ids(state, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Tensor-position ``NgramHashState.forward`` for decode rows.

    ``token``: int64 [B, 1]; ``pos``: int64 [B].  Returns [B, 1, n_engram_layers, n_hash_cols].
    Same formula as ``ref/engram.py``: compressed-token map, look-back cache, XOR multipliers,
    prime buckets and per-layer offsets; the cache keeps its state across steps.
    """
    b = int(token.shape[0])
    max_ngram = state.layout.max_ngram_size
    dead = state.DEAD
    pad = state.pad_id
    compressed = state.token_map[token]                       # [B, 1] int64
    state.cache[:b].scatter_(1, pos.reshape(b, 1), compressed)
    positions = pos.reshape(b, 1)                             # [B, 1]
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(max_ngram):
        source = state.cache[:b].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == dead)
        tokens.append(torch.where(blocked, pad, source))
    tokens = torch.stack(tokens, dim=-1)                      # [B, 1, max_ngram]
    products = tokens.unsqueeze(2) * state.multipliers        # [B,1,L,max_ngram]
    rolling, hashes = products[..., 0], []
    for i in range(1, max_ngram):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % state.primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + state.offsets


# --------------------------------------------------------------------------------- engram
def graph_engram(eng, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
    """One Engram layer on the FP8/E8M0 decode path (same math as the eager decode patch)."""
    from .engram_decode import engram_gate_decode

    return engram_gate_decode(
        x, hash_ids, eng._decode_fp8, eng._decode_scales, eng.wkv.weight,
        eng.q_weight, eng.k_weight, eng.hc_mult, eng.dim, eng.eps, eng.clamp_value,
    )


# --------------------------------------------------------------------------------- attention
def _scatter_rows(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    """``dst[i, index[i,0], :] = src[i,0,:]`` for one row per batch element (graph safe)."""
    dst.scatter_(1, index, src)


def _batch_index(slot: torch.Tensor, b: int, dim: int) -> torch.Tensor:
    """[B] int64 slot -> [B, 1, dim] index for a ``scatter_`` along dim 1."""
    return slot.reshape(b, 1, 1).expand(b, 1, dim)


def _graph_compress(attn, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Compressor for the current token rows; writes state unconditionally, returns the latent."""
    comp = attn.compressor
    ratio = attn.compress_ratio
    b = int(x.shape[0])
    if ratio == 1:
        return comp.norm(comp.wkv(x))
    xf = x.float()
    kv = comp.wkv(xf)                                          # [B,1,D] fp32
    score = comp.wgate(xf)                                     # [B,1,ratio] fp32
    slot = pos % ratio
    _scatter_rows(comp.kv_state[:b], _batch_index(slot, b, kv.shape[-1]), kv)
    _scatter_rows(comp.score_state[:b], _batch_index(slot, b, score.shape[-1]), score)
    pooled = (comp.kv_state[:b] * comp.score_state[:b].softmax(dim=1)).sum(dim=1, keepdim=True)
    return comp.norm(pooled.to(x.dtype))


def _graph_indexer_write(attn, x, qr, latent_pre_rope, pos, freqs_row, group) -> None:
    """Index-source layer rows: append the index key and write the group's top-k idx buffer."""
    idxr = attn.indexer
    b = int(x.shape[0])
    rd = idxr.rope_head_dim
    ratio = idxr.compress_ratio
    gidx = pos // ratio
    k = idxr.k_norm(idxr.wk(latent_pre_rope))                  # [B,1,index_head_dim]
    k_freq = idxr.freqs_cis[pos + 1 - ratio]                   # [B, rd//2]
    rope_rows_batch(k, idxr.index_head_dim - rd, 1, rd, k_freq)
    _scatter_rows(idxr.k_cache[:b], _batch_index(gidx, b, idxr.index_head_dim), k)

    q_idx = idxr.wq_b(qr).unflatten(-1, (idxr.n_local_heads, idxr.index_head_dim))
    rope_rows_batch(q_idx, idxr.index_head_dim - rd, idxr.n_local_heads, rd, freqs_row)
    weights = idxr.weights_proj(x) * (idxr.softmax_scale * idxr.n_heads ** -0.5)

    index_k = idxr.k_cache[:b]                                 # [B, max_comp, index_head_dim]
    score = torch.einsum("bshd,btd->bsht", q_idx, index_k)     # [B,1,H,max_comp]
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)  # [B,1,max_comp]
    compress_len = (pos + 1) // ratio                          # [B]
    score = score.masked_fill(idxr._graph_positions >= compress_len.reshape(b, 1, 1),
                              float("-inf"))
    topk = idxr.index_topk
    idx = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    valid = idx < compress_len.reshape(b, 1, 1)
    out = torch.where(valid, idx + attn.window_size, torch.full_like(idx, -1))
    group["topk"][:b].copy_(out)


def graph_attn_forward(attn, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``Attention.forward`` for decode rows with a tensor position (graph-safe)."""
    b = int(x.shape[0])
    rd = attn.rope_head_dim
    d = attn.head_dim
    win = attn.window_size
    freqs_row = attn.freqs_cis[pos]                            # [B, rd//2] complex
    group = getattr(attn, "_graph_group", None)

    qr = _rmsnorm(attn.wq_a(x), attn.q_norm.weight, attn.q_norm.eps)
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, d))
    rope_rows_batch(q, d - rd, attn.n_local_heads, rd, freqs_row)

    kv = _rmsnorm(attn.wkv(x), attn.kv_norm.weight, attn.kv_norm.eps)
    rope_rows_batch(kv, d - rd, 1, rd, freqs_row)
    _scatter_rows(attn.window_kv_cache[:b], _batch_index(pos % win, b, d), kv)
    window_kv = attn.window_kv_cache[:b]

    if group is not None:
        if attn.is_kv_source:
            latent = _graph_compress(attn, x, pos)
            if attn.is_index_source:
                _graph_indexer_write(attn, x, qr, latent, pos, freqs_row, group)
            # the cache stores the RoPE'd latent of a completed group; writing unconditionally is
            # safe because the in-progress index is not read before it is overwritten
            rope_rows_batch(latent, d - rd, 1, rd,
                            attn.freqs_cis[pos + 1 - attn.compress_ratio])
            _scatter_rows(group["cache"][:b], _batch_index(pos // attn.compress_ratio, b, d),
                          latent)
        compress_kv = group["cache"][:b]                       # [B, max_comp, d]
        topk_idxs = group["topk"][:b]
    else:
        compress_kv, topk_idxs = None, None

    o = sparse_attn_decode_batch(q, window_kv, compress_kv, topk_idxs, attn.attn_sink,
                                 attn.softmax_scale, pos, win)
    rope_rows_batch(o, d - rd, attn.n_local_heads, rd, freqs_row, inverse=True)

    # batch-size-invariant block-diagonal wo_a (the b=1 einsum and the b>1 einsum differ ~1 ulp)
    o = blockdiag_wo_a(o, attn.wo_a.weight, attn.n_local_groups, attn.o_lora_rank)
    return attn.wo_b(o.flatten(2))


def _graph_attn_dispatch(attn, x: torch.Tensor, start_pos):
    """Decode with a pos tensor -> graph fast path; prefill / int positions -> eager forward."""
    if torch.is_tensor(start_pos):
        return graph_attn_forward(attn, x, start_pos)
    return attn._graph_orig_forward(x, start_pos)


# --------------------------------------------------------------------------------- whole step
def graph_forward(model, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """One decode step with static shapes; returns logits [B, 1, vocab] (fp32).

    ``token``: int64 [B, 1]; ``pos``: int64 [B] (the position of the token being fed).
    """
    inner = getattr(model, "inner", model)
    b = int(token.shape[0])
    h = inner.embed(token)                                     # [B,1,dim]
    h = h.unsqueeze(2).repeat(1, 1, inner.hc_mult, 1)          # [B,1,hc,dim]
    hashes = (graph_hash_ids(inner.engram_hash, token, pos)
              if inner.engram_hash is not None else None)
    pre_mix = inner._graph_identity_pre_mix[:b]                # [B,1,hc] fp32, constant
    last = None
    for layer in inner.layers:
        if layer.engram is not None:
            h = graph_engram(layer.engram, h,
                             hashes[:, :, layer.engram.layer_hash_index, :])
        h, pre_mix = layer(h, pos, pre_mix, None)
        last = layer
    h = last.hc_pre(h, pre_mix)
    return inner.head(inner.norm(h), full_logits=True)


# --------------------------------------------------------------------------------- enabling
def enable_graph_decode(model, *, bf16_head: bool = False,
                        mhc_fast: bool = True, moe_fast: bool = True,
                        rmsnorm_fast: bool = True, linear_fast: bool = True,
                        max_batch: int | None = None) -> "GraphDecoder":
    """Install the native inference patches + the graph-safe attention dispatch, return a decoder.

    ``bf16_head=True`` also swaps the fp32 output head for an inference bf16 copy (M2: halves the
    largest streamed tensor; generation agreement 100 %, T0 BPB delta 0 on the P2b checkpoint).
    ``mhc_fast=True`` (default) uses the decode-specialised mHC kernels for T=1 blocks (prefill
    keeps the validated fused path).
    ``moe_fast=True`` (default) replaces the small-M torch MoE with the fused 4-kernel decode MoE
    (M2d); ``rmsnorm_fast=True`` (default) replaces the torch RMSNorm chain with one kernel.
    ``max_batch`` sizes the batched decode step buffers (default: the model's ``max_batch_size``).
    """
    from .native_decode import enable_native_decode

    counts = enable_native_decode(model, engram=True, moe=True, mhc=True, attn=True)
    if linear_fast:
        # must be installed before any capture; it also serves the eager prefill (fallback >16 rows)
        from .linear_decode import enable_linear_decode

        counts["linear_decode"] = enable_linear_decode(model)
    if bf16_head:
        from .head_decode import enable_bf16_head

        counts["bf16_head"] = enable_bf16_head(model, fused_gemv=True, max_rows=max_batch or 16)
    if mhc_fast:
        from .mhc_decode_fast import enable_mhc_decode_fast

        counts["mhc_fast"] = enable_mhc_decode_fast(model)
    if moe_fast:
        from .moe_decode_fast import enable_moe_decode_fast

        counts["moe_fast"] = enable_moe_decode_fast(model, max_tokens=max_batch or 8)
    if rmsnorm_fast:
        from .rmsnorm_decode import enable_rmsnorm_decode

        counts["rmsnorm_fast"] = enable_rmsnorm_decode(model)
    inner = getattr(model, "inner", model)
    device = next(inner.parameters()).device
    dtype = inner.embed.weight.dtype
    nb = int(max_batch or getattr(inner, "max_batch_size", None)
             or getattr(getattr(inner, "args", None), "max_batch_size", None) or 8)

    group = None
    dummy_kv = torch.zeros(1, inner.layers[0].attn.head_dim, dtype=dtype, device=device)
    for layer in inner.layers:
        attn = layer.attn
        attn._graph_group = None
        attn._graph_dummy_kv = dummy_kv
        if getattr(attn, "compress_ratio", 0):
            if attn.is_kv_source:
                if not attn.is_index_source:
                    raise NotImplementedError("graph decode assumes every KV source is an index source")
                group = {
                    "topk": torch.zeros(nb, 1, attn.indexer.index_topk, dtype=torch.int32,
                                        device=device),
                    "cache": attn.compress_kv_cache,
                }
            if group is None:
                raise NotImplementedError("compressed layer without a source layer")
            attn._graph_group = group
        if attn.indexer is not None:
            if attn.indexer.freqs_cis is None:
                # the reference Indexer lazily aliases the owner's freqs table inside its forward;
                # the graph path indexes it directly, so make the alias explicit at install time
                attn.indexer.freqs_cis = attn.freqs_cis
            max_comp = attn.indexer.k_cache.shape[1]
            attn.indexer._graph_positions = torch.arange(max_comp, device=device)
        # keep the eager forward for prefill / int positions, then patch the dispatcher
        attn._graph_orig_forward = attn.forward
        attn.forward = _graph_attn_dispatch.__get__(attn, type(attn))

    pm = torch.zeros(nb, 1, inner.hc_mult, dtype=torch.float32, device=device)
    pm[..., 0] = 1.0
    inner._graph_identity_pre_mix = pm
    return GraphDecoder(model, patches=counts, max_batch=nb)


class GraphDecoder:
    """Prefill eagerly, capture one decode-step graph, then serve single-token replays."""

    def __init__(self, model, patches: dict | None = None, max_batch: int = 8):
        self.model = model
        self.inner = getattr(model, "inner", model)
        self.device = next(self.inner.parameters()).device
        self.patches = patches or {}
        self.max_batch = int(max_batch)
        self.token = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._logits: torch.Tensor | None = None
        self.ctx_len = 0

    # -- eager pieces -----------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        from .generate import reset_caches
        from .model import inner_forward

        reset_caches(self.inner)
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        return inner_forward(self.inner, ids, start_pos=0, full_logits=True)

    @torch.no_grad()
    def step_eager(self, token: int, pos: int) -> torch.Tensor:
        """Eager execution of the graph forward (used for parity checks / warmup)."""
        self.token.fill_(token)
        self.pos.fill_(pos)
        return graph_forward(self.model, self.token, self.pos)

    # -- capture ----------------------------------------------------------------------
    @torch.no_grad()
    def capture(self) -> None:
        """Capture the step graph; caches are mutated, so the caller must prefill again after."""
        torch.cuda.synchronize()
        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(3):
                graph_forward(self.model, self.token, self.pos)
        current.wait_stream(side)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._logits = graph_forward(self.model, self.token, self.pos)
        torch.cuda.synchronize()

    @torch.no_grad()
    def prepare(self, prompt_ids: list[int]) -> torch.Tensor:
        """Prefill -> capture -> prefill again, so a replay loop starts from a clean post-prefill state."""
        prompt_ids = list(prompt_ids)
        if len(prompt_ids) + 1 >= int(self.inner.max_seq_len):
            raise ValueError(f"prompt of {len(prompt_ids)} tokens leaves no decode room inside "
                             f"max_seq_len {int(self.inner.max_seq_len)}")
        logits = self.prefill(prompt_ids)
        self.ctx_len = len(prompt_ids)
        self.token.fill_(0)
        self.pos.fill_(self.ctx_len)
        self.capture()
        logits = self.prefill(prompt_ids)   # restore caches (capture/warmup wrote into them)
        return logits

    # -- serving ----------------------------------------------------------------------
    def set_step(self, token: int, pos: int) -> None:
        self.token.fill_(int(token))
        self.pos.fill_(int(pos))

    @torch.no_grad()
    def replay(self) -> torch.Tensor:
        if self._graph is None:
            raise RuntimeError("call prepare()/capture() first")
        self._graph.replay()
        return self._logits

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], n_new: int) -> list[int]:
        """Greedy decode; returns the new tokens (graph path).

        Single-sequence runtime: generation stays inside ``max_seq_len``; requests that would run
        past it are truncated.  Batched/continuous serving lives in ``nanov41.batch_decode``.
        """
        prompt_ids = list(prompt_ids)
        if len(prompt_ids) >= int(self.inner.max_seq_len):
            raise ValueError(f"prompt of {len(prompt_ids)} tokens exceeds max_seq_len "
                             f"{int(self.inner.max_seq_len)} (no paged KV yet)")
        n_new = min(int(n_new), int(self.inner.max_seq_len) - len(prompt_ids))
        logits = self.prepare(prompt_ids)
        next_logits = logits[0, -1]
        out: list[int] = []
        for i in range(n_new):
            tok = int(next_logits.argmax())
            out.append(tok)
            if i == n_new - 1:
                break
            self.set_step(tok, self.ctx_len + i)
            self.replay()
            next_logits = self._logits[0, 0]
        return out


__all__ = ["enable_graph_decode", "GraphDecoder", "graph_forward", "graph_hash_ids",
           "graph_engram", "graph_attn_forward"]
