# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Chunked prefill: process a long sequence in aligned chunks with an incremental KV cache.

Why: the reference incremental path only supports *decode* (seqlen == 1): `Compressor.forward`
with `start_pos > 0` fills a single slot per step and `Attention._window_kv` writes
`window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`, so a multi-token chunk with
`start_pos > 0` crashes. Both the 1M *inference* path (prefill + cache) and long-context *training*
via chunked BPTT need multi-token chunks *with* context, which this module provides.

`install(M)` patches, for `start_pos > 0` and `seqlen > 1` only (full prefill / decode are untouched):

* `Compressor.forward` — pools each chunk's complete compression groups independently (chunks must
  be aligned: `start_pos % ratio == 0` and `seqlen % ratio == 0`, which the caller enforces by using
  a chunk size that is a multiple of every compress ratio). No cross-chunk partial group is needed.
* `Attention._window_kv` — builds the sliding-window keys as the ring buffer contents (last
  `window_size` tokens before the chunk) followed by the chunk itself, with per-query window
  indices into that concatenation; the ring is then updated with the chunk's last
  `window_size` tokens.

It also requires the chunked indexer (`nanov41.chunked_indexer`) to be installed: it handles
`start_pos > 0` chunks by scoring against the full index-k cache and applying the per-query causal
visibility mask.

Window layout (2026-11, `NANOV41_CHUNK_ATTN`, default `fast`): for `start_pos > 0` chunks the
window KV is returned in **absolute position order** -- `cat([last win-1 keys, this chunk's keys])`
-- so query `i` attends to rows `i .. i+win-1` and the window part of `topk_idxs` is simply `i+c`.
That layout is recognized by `fast_attn`'s right-band plan, whose kv backward is deterministic.
Before this, the rotated-ring layout was not recognized, `sparse_attn` fell back to the reference
torch implementation and its `index_select -> index_add` kv gradient made the chunked-BPTT
gradients non-reproducible (22/284 parameters bitwise stable, max relative diff ~0.29; now
284/284 bitwise, and per-chunk CUDA-graph replay is bitwise equal to eager).  It is also faster:
262K step -8.1%, 1M step -10.5%.  `NANOV41_CHUNK_ATTN=ref` keeps the reference-fallback attention path for A/B / safety
(the absolute-order layout is still used; the reference implementation consumes it directly).

Correctness note (2026-11, see the feasibility report): the window ring, the per-group compression,
the compressed-KV cache and the index-k cache were each verified bitwise against a full prefill.
What is *not* bitwise is the indexer's top-k when scores tie: the indexer applies ReLU, so a
randomly-initialised model has many exact-zero scores and `torch.topk` then resolves those ties
differently for the full-prefill score width (seq/ratio) and the chunked width (chunk_end/ratio).
Set `NANOV41_IDX_TIEBREAK=1` to use a width-independent tie-break; with it, dense configs compare
bitwise and MoE configs differ only by floating-point noise (see tests/test_chunked_prefill.py).
Chunked BPTT training does not need the tie-break: ties are semantically arbitrary, and a fixed
chunk schedule stays consistent.

Opt-in only; on CPU it works for the tiny config and is covered by `tests/test_chunked_prefill.py`.
"""
from __future__ import annotations

import torch

_STATE = {"installed": False, "orig_comp": None, "orig_win": None, "orig_ckv": None,
          "orig_sparse": None, "win_hint": None, "M": None}


def _chunk_compressor_forward(M, orig):
    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        if ratio == 1 or start_pos == 0 or seqlen == 1:
            return orig(self, x, start_pos)
        if start_pos % ratio or seqlen % ratio:
            raise ValueError(f"chunked prefill needs aligned chunks: start_pos={start_pos}, "
                             f"seqlen={seqlen}, compress_ratio={ratio}")
        dtype = x.dtype
        xf = x.float()
        kv, score = self.wkv(xf), self.wgate(xf)
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(dtype))

    return forward


def _chunk_window_kv_forward(M, orig):
    def _window_kv(self, x, freqs_cis, start_pos):
        bsz, seqlen, _ = x.size()
        if start_pos == 0 or seqlen == 1:
            if start_pos == 0:
                _STATE["win_hint"] = None
            out = orig(self, x, freqs_cis, start_pos)
            if start_pos == 0 and getattr(self.window_kv_cache, "grad_fn", None) is not None:
                # the ring is cross-chunk context: keep it out of the autograd graph (see §3m)
                self.window_kv_cache = self.window_kv_cache.detach()
            return out
        win = self.window_size
        end = start_pos + seqlen
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(x))
        M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
        ring = self.window_kv_cache[:bsz].view(bsz, win, -1)    # slots == abs_pos % win
        if start_pos >= win - 1:
            # Absolute-order window layout, understood by fast_attn's right-band plan and its
            # deterministic kv backward: de-rotate the ring so kv row r is absolute position
            # (start_pos - win + 1 + r); query i then attends to rows i .. i+win-1 of
            # cat([history(win-1), chunk(seqlen)]) and the window part of topk_idxs is i+c.
            # The reference sparse_attn (used when this layout is not recognized) takes the kv
            # gradient through index_select -> atomic index_add, which is NOT bit-reproducible.
            keys = (start_pos - (win - 1) + torch.arange(win - 1, device=x.device)) % win
            hist = ring.index_select(1, keys)                   # [b, win-1, hd]
            kv_all = torch.cat([hist, kv], dim=1)               # [b, win-1+seqlen, hd]
            idxs = (torch.arange(seqlen, device=x.device)[:, None]
                    + torch.arange(win, device=x.device)[None, :]).int()
            idxs = idxs.unsqueeze(0).expand(bsz, -1, -1).contiguous()
            _STATE["win_hint"] = win
        else:
            # Too little history for the absolute-order layout (only a start_pos>0 chunk shorter
            # than the window can hit this; every supported chunk size is >= window_size).  Keep
            # the rotated-ring layout: the reference sparse_attn handles it, at the cost of the
            # nondeterministic kv gradient.
            kv_all = torch.cat([ring, kv], dim=1)               # [b, win+seqlen, hd]
            qpos = start_pos + torch.arange(seqlen, device=x.device)[:, None]        # [C, 1]
            pos = qpos - (win - 1) + torch.arange(win, device=x.device)[None, :]     # [C, win]
            in_chunk = pos >= start_pos
            idx = torch.where(in_chunk, win + (pos - start_pos), pos % win)
            valid = pos >= 0
            idxs = torch.where(valid, idx, -1).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
            _STATE["win_hint"] = None
        # update the ring with the last min(win, seqlen) keys so decode / the next chunk see them.
        # no_grad: these caches are cross-chunk *context* (truncated BPTT), and letting autograd
        # attach CopySlices nodes to the persistent buffer breaks backward (and leaks, report §3m).
        n_upd = min(win, seqlen)
        src = kv[:, seqlen - n_upd:].detach()
        slots = (end - n_upd + torch.arange(n_upd, device=x.device)) % win
        with torch.no_grad():
            ring.index_copy_(1, slots, src)
        return kv_all, idxs

    return _window_kv


def _chunk_compress_kv_forward(M, orig):
    def _compress_kv(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        if start_pos == 0 or seqlen == 1:
            out = orig(self, x, qr, start_pos, offset)
            if start_pos == 0:
                # same-chunk consumers already read the local latent (shared_attn.compress_kv), so the
                # persistent cache only needs to carry values, not graph, into the next chunk
                for name in ("compress_kv_cache",):
                    buf = getattr(self, name, None)
                    if buf is not None and getattr(buf, "grad_fn", None) is not None:
                        setattr(self, name, buf.detach())
            return out
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            # The cache holds the history for future chunks (detached context, truncated BPTT);
            # this chunk's consumers must see the grad-connected latent, so publish
            # cat(history, latent) instead of the raw buffer. History is sliced read-only from the
            # cache, whose values were written by previous chunks.
            hist = self.compress_kv_cache[:bsz, : start_pos // ratio]
            shared = torch.cat([hist, latent], dim=1)
            M.shared_attn.compress_kv = shared
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            freqs = self.freqs_cis[start_pos: start_pos + latent.size(1) * ratio: ratio]
            M.apply_rotary_emb(latent[..., -self.rope_head_dim:], freqs)
            M.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            # refresh the detached view after the in-place rope/quant, then persist it detached
            M.shared_attn.compress_kv = torch.cat(
                [M.shared_attn.compress_kv[:, : start_pos // ratio], latent], dim=1)
            with torch.no_grad():
                self.compress_kv_cache[:bsz, start_pos // ratio:
                                       start_pos // ratio + latent.size(1)] = latent.detach()
        return M.shared_attn.compress_kv[:bsz, :compress_len], idxs

    return _compress_kv


def install(M=None) -> dict:
    global _STATE
    if M is None:
        from .patched_env import import_patched

        M = import_patched()
    if _STATE["installed"] and _STATE["M"] is M:
        return dict(_STATE)
    if _STATE["installed"]:
        uninstall()
    prev_sparse = getattr(M, "sparse_attn", None)
    _STATE.update({
        "installed": True, "M": M,
        "orig_comp": M.Compressor.forward,
        "orig_win": M.Attention._window_kv,
        "orig_ckv": M.Attention._compress_kv,
        "orig_sparse": prev_sparse,
        "win_hint": None,
    })
    M.Compressor.forward = _chunk_compressor_forward(M, M.Compressor.forward)
    M.Attention._window_kv = _chunk_window_kv_forward(M, M.Attention._window_kv)
    M.Attention._compress_kv = _chunk_compress_kv_forward(M, M.Attention._compress_kv)

    import os as _os
    _STATE["attn_mode"] = _os.environ.get("NANOV41_CHUNK_ATTN", "fast").strip().lower() or "fast"
    if prev_sparse is not None and _STATE["attn_mode"] != "ref":
        def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
            # `_window_kv` publishes the window width it used for the absolute-order layout; any
            # layout it did not publish falls through to whatever was installed before us (v2
            # TileLang kernels, fast_attn, or the reference implementation).
            hint = _STATE.get("win_hint")
            if hint is not None and q.is_cuda:
                from .fast_attn import sparse_attn_fast
                return sparse_attn_fast(q, kv, attn_sink, topk_idxs, softmax_scale,
                                        win_hint=hint, right_band=True)
            return prev_sparse(q, kv, attn_sink, topk_idxs, softmax_scale)

        M.sparse_attn = _sparse_attn
    return {"installed": True}


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    M = _STATE["M"]
    if _STATE["orig_comp"] is not None:
        M.Compressor.forward = _STATE["orig_comp"]
    if _STATE["orig_win"] is not None:
        M.Attention._window_kv = _STATE["orig_win"]
    if _STATE["orig_ckv"] is not None:
        M.Attention._compress_kv = _STATE["orig_ckv"]
    if _STATE.get("orig_sparse") is not None:
        M.sparse_attn = _STATE["orig_sparse"]
    _STATE.update({"installed": False, "orig_comp": None, "orig_win": None,
                   "orig_ckv": None, "orig_sparse": None, "win_hint": None, "M": None})


def installed() -> bool:
    return bool(_STATE["installed"])
