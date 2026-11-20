# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Chunked-indexer prefill path: make the 1M-context prefill fit on one GPU.

## Problem

The reference indexer (`_ref_patched_model.Indexer.forward`) scores *all* queries against *all*
reachable compressed positions in one `einsum`:

    index_score = einsum("bshd,btd->bsht", q, index_k)      # [b, seq, n_index_heads, n_pos]

At long context this is O(seq * n_pos) memory: seq=1M with n_pos≈0.5-1M is a multi-TB tensor and
OOMs on a 32 GB card long before 1M (the stock path already dies near 32K-64K).

## What this module does

`install(M)` replaces `Indexer.forward` with a **query-chunked** implementation that is
mathematically identical to the stock prefill path: it computes the same scores, applies the same
visibility mask, takes the same top-k, sorts and offsets exactly like the reference — but the
`[b, C, heads, n_pos]` score tensor is materialised per query chunk `C` instead of for the whole
sequence. Memory becomes O(C * n_pos) per chunk; the decode path (`start_pos > 0`) is untouched and
delegates to the stock implementation.

Modes: `hsi-auto` runs the exact path while the compressed prefix is short and switches to the
two-level `hsi` path once `end_pos/ratio >= NANOV41_IDX_HSI_AUTO_MIN` (default 65536 compressed
positions ≈ 128K tokens at compress_ratio 2) — the HSI overhead only pays off for long contexts;
measured 262K chunked training: thr=65536 4.12 s vs exact 4.24 s vs thr=8192 4.40 s. `prefix-pad` (default) is the **exact** optimisation set below and stays bitwise identical to
the stock path; `prefix` drops the (unreachable for the chunk) tail from the top-k tensor itself,
which is faster but changes top-k tie-breaking when equal scores exist, so it is opt-in for
inference only; `full` reproduces the plain chunked behaviour for A/B; `fused` keeps the exact (padded) top-k path
but computes the head reduction with the Triton kernel in `nanov41/indexer_fused.py` (CUDA only;
same values up to fp32 rounding, much faster); `hsi` is the **approximate**
two-level candidate indexer used by long-context inference: level one scores each query against
per-block mean index keys (8x fewer positions), level two computes exact scores only for the
candidate blocks (default `4 * index_topk` blocks) and top-k's inside them. It is not bitwise
identical to the stock path (candidate blocks are chosen from block means, and ties break
differently) but is O(seq * npos / block_size + seq * candidates) instead of O(seq * npos); it is
opt-in and must be tracked as an approximation in any quality comparison.

* **deterministic tie-break** (`NANOV41_IDX_TIEBREAK=1`, validation only): the indexer applies
  ReLU before top-k, so a randomly-initialised (or lightly trained) model has many exact-zero
  scores and `topk` then breaks those ties depending on the score tensor's width. That makes full
  prefill (`npos = seq/ratio`) and chunked prefill (`npos = chunk_end/ratio`) pick different
  positions even though the scores are bitwise equal. Adding a tiny absolute-position bias
  (`-pos * 1e-6`) before `topk` makes the selection width-independent, so the two paths can be
  compared for semantic parity. Off by default: it is a validation aid, not part of the model.
* **prefix scores**: for a query chunk `[c0, c1)` no query can see compressed positions beyond
  `c1 // ratio`, so the score tensor is computed only against `index_k[:, :c1//ratio]`. This is
  exact (the dropped positions are masked to -inf for every query in the chunk) and halves the
  O(seq^2) score work on average for a full prefill.
* **band-only masking**: positions before `c0 // ratio` are visible to every query in the chunk, so
  only the small band `[c0//ratio, c1//ratio)` needs the per-query causal mask; the mask tensor is
  therefore O(chunk^2 / ratio) instead of O(chunk * n_pos).
  In `prefix-pad` mode the score tensor is padded back to `n_pos` with -inf before top-k, so the
  top-k sees exactly the reference tensor values (bitwise parity); `prefix` mode skips the pad
  and top-k's over the prefix only (faster, tie-breaking may differ on equal scores).

The two-level candidate indexer (`candidate_source_layer >= 0`) is **not** supported by the
chunked path: the reference implementation materialises the full boolean candidate mask
`[b, seq, n_pos]`, which cannot be chunked without rewriting the candidate selection; nano configs
disable the candidate path (`candidate_source_layer = -1`), so we fall back to the stock forward
with a one-time warning when it is enabled.

## Usage

    import nanov41.chunked_indexer as ci
    ci.install(M)                     # chunk = NANOV41_IDX_CHUNK (default 512), or pass chunk=
    ...
    ci.uninstall()

Opt-in only: nothing is installed by default. Pair it with a larger `max_seq_len` in the model
config (`cfg["max_seq_len"] = target_len`); the reference sizes `freqs_cis`, the indexer k-cache
and the compressed-KV caches from that field.
"""
from __future__ import annotations

import os

import torch

DEFAULT_CHUNK = 512


def _IDX_FUSED_AVAILABLE() -> bool:
    try:
        from .indexer_fused import available as _avail

        return bool(_avail())
    except Exception:  # noqa: BLE001
        return False


def _hsi_knobs() -> tuple[int, int]:
    """(positions per candidate block, candidate blocks). Read at install time so tests/benchmarks
    can vary them without re-importing the module; 0 candidate blocks means 4*index_topk."""
    blk = max(1, int(os.environ.get("NANOV41_IDX_HSI_BLK", "8") or 8))
    cand = int(os.environ.get("NANOV41_IDX_HSI_CAND", "0") or 0)
    return blk, cand
_STATE = {"installed": False, "orig": None, "cls": None, "chunk": None, "mode": None,
          "warned": False, "tiebreak": False}


def _tiebreak_enabled() -> bool:
    return os.environ.get("NANOV41_IDX_TIEBREAK", "0").strip().lower() in ("1", "true", "yes", "on")


def _chunk_from_env(chunk: int | None) -> int:
    if chunk is None:
        chunk = int(os.environ.get("NANOV41_IDX_CHUNK", DEFAULT_CHUNK) or DEFAULT_CHUNK)
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return chunk


def _make_forward(M, orig, chunk: int, mode: str = "prefix"):
    fp4_block = M.fp4_block_size
    if mode not in ("prefix-pad", "prefix", "full", "hsi", "hsi-auto", "fused"):
        raise ValueError(f"unknown indexer mode {mode!r} "
                         f"(expected 'prefix-pad', 'prefix' or 'full')")

    def forward(self, x, qr, latent, start_pos, offset):
        # decode keeps the reference implementation; aligned multi-token chunks are supported below
        bsz0, seqlen0, _ = x.size()
        _run_mode = mode
        if _run_mode == "hsi-auto":
            threshold = _STATE.get("hsi_auto_min", 65536)
            npos_now = (start_pos + seqlen0) // max(1, self.compress_ratio)
            if npos_now < threshold:
                _run_mode = "fused" if _IDX_FUSED_AVAILABLE() else "prefix-pad"
            else:
                _run_mode = "hsi"
        if seqlen0 == 1:
            return orig(self, x, qr, latent, start_pos, offset)
        if start_pos != 0 and start_pos % self.compress_ratio:
            raise ValueError(f"chunked prefill needs aligned chunks: start_pos={start_pos}, "
                             f"compress_ratio={self.compress_ratio}")
        if getattr(self, "is_candidate_source", False) or getattr(self, "uses_candidates", False):
            if not _STATE["warned"]:
                print("[chunked_indexer] candidate indexer is enabled; falling back to the stock "
                      "forward (candidates are not chunkable in this implementation)", flush=True)
                _STATE["warned"] = True
            return orig(self, x, qr, latent, start_pos, offset)

        shared = M.shared_attn
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # --- identical to the reference: append this group's index keys to the cache ---
        if self.owns_k and latent is not None:
            ngroups = k_ngroups = latent.size(1)
            freqs = self.freqs_cis[start_pos: start_pos + ngroups * ratio: ratio]
            k = self.k_norm(self.wk(latent))
            M.apply_rotary_emb(k[..., -rd:], freqs)
            M.fp4_act_quant(k, fp4_block, True)
            with torch.no_grad():
                self.k_cache[:bsz, start_pos // ratio: start_pos // ratio + ngroups] = k.detach()
        if self.owns_k:
            shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        M.fp4_act_quant(q, fp4_block, True)
        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        npos = index_k.size(1)
        topk = min(self.index_topk, end_pos // ratio)
        outs: list[torch.Tensor] = []
        for c0 in range(0, seqlen, chunk):
            c1 = min(c0 + chunk, seqlen)
            pos_ids = start_pos + torch.arange(c0, c1, device=x.device)
            clens = ((pos_ids + 1) // ratio).clamp(max=npos)                     # [C]
            # every query in this chunk sees at most c1 // ratio positions, and all queries see
            # at least c0 // ratio of them (prefix + band-only mask stay exact)
            hi = min(int(clens.max().item()), npos) if _run_mode == "prefix" else npos
            lo = max(int(clens.min().item()), 0) if _run_mode == "prefix" else 0
            if hi == 0:
                outs.append(torch.full((bsz, c1 - c0, topk), -1, dtype=torch.int32,
                                       device=x.device))
                continue
            qc = q[:, c0:c1]
            wc = weights[:, c0:c1].unsqueeze(-1)
            if _run_mode == "hsi":
                # Two-level candidate indexer (the official CSA candidate-block idea): level one
                # scores each query against per-block *mean* index keys (hi/blk positions), keeps a
                # small number of candidate blocks, then level two computes the exact scores only on
                # the <= nblk_k*blk candidate positions. This turns the O(seq x npos) score scan and
                # the top-k over npos into O(seq x npos/blk + seq x nc), and reuses a single score
                # tensor layout. NANOV41_IDX_HSI_BLK=1 + NANOV41_IDX_HSI_CAND=4*topk reproduces the
                # exact top-k bitwise (tests/test_chunked_indexer.py), so it is also a correctness
                # reference. This is an *approximation* for the real indexer: candidate blocks are
                # chosen from block means, so top-k ties / near-ties can differ from the exact path.
                blk = _STATE.get("hsi_blk", 8)
                cand_cfg = _STATE.get("hsi_cand", 0)
                nblk_k_request = cand_cfg if cand_cfg > 0 else 4 * topk
                nblk = (hi + blk - 1) // blk
                pad = nblk * blk - hi
                kk = index_k[:, :hi]
                if pad:
                    kk = torch.nn.functional.pad(kk, (0, 0, 0, pad))
                sums = kk.view(bsz, nblk, blk, -1).sum(dim=2)                 # [b, nblk, hd]
                # capture-safe: a Python-scalar assignment (`cnt[-1] = int`) is a host->device
                # copy, which CUDA-graph capture rejects ("CPU tensor is not pinned"); build the
                # last entry with a fill kernel instead (scalar passed as a kernel arg).
                cnt = torch.full((nblk,), float(blk), device=x.device)
                cnt[-1:] = torch.full((1,), float(hi - (nblk - 1) * blk), device=x.device,
                                      dtype=cnt.dtype)
                block_mean = (sums / cnt.view(1, nblk, 1)).to(qc.dtype)
                # level-one reduction with the same fused kernel as the exact path when possible:
                # k = block means, and "block reachable" == blk_idx < ceil(clens/blk)
                clens_b = ((clens + blk - 1) // blk).to(torch.int32)
                used_fused = False
                if bsz == 1:
                    from .indexer_fused import available as _kavail, fused_indexer_score
                    if _kavail():
                        s_b = fused_indexer_score(qc, block_mean,
                                                  weights[:, c0:c1], clens_b).unsqueeze(0)
                        used_fused = True
                if not used_fused:
                    s_b = torch.einsum("bshd,btd->bsht", qc, block_mean)     # [b, C, h, nblk]
                    s_b.relu_()
                    s_b.mul_(wc)
                    s_b = s_b.sum(dim=2)                                     # [b, C, nblk]
                    blk_start = torch.arange(nblk, device=x.device) * blk
                    s_b = s_b.masked_fill(blk_start[None, :] >= clens[:, None], -torch.inf)
                nblk_k = min(nblk_k_request, nblk)
                blk_score, blk_idx = s_b.topk(nblk_k, dim=-1, sorted=False)   # [b, C, nblk_k]
                nc = nblk_k * blk
                pos = (blk_idx.unsqueeze(-1) * blk
                       + torch.arange(blk, device=x.device)).reshape(bsz, c1 - c0, nc)
                # the last block may be padded past the reachable prefix
                pos = pos.clamp(max=hi - 1)
                # level two: exact scores for each query's candidate positions
                k_cand = index_k[0][pos.reshape(-1)].view(bsz, c1 - c0, nc, index_k.size(-1))
                s = torch.einsum("bchd,bctd->bcht", qc, k_cand)               # [b, C, h, nc]
                s.relu_()
                s.mul_(wc)
                s = s.sum(dim=2)                                              # [b, C, nc]
                s = s.masked_fill(pos >= clens[:, None], -torch.inf)
                _, ci = s.topk(topk, dim=-1, sorted=False)
                idxs = pos.gather(-1, ci).sort(dim=-1).values
                outs.append(torch.where(idxs < clens[:, None], idxs + offset, -1).int())
                continue
            if _run_mode == "fused":
                from .indexer_fused import available as _kavail, fused_indexer_score
                if _kavail():
                    score = fused_indexer_score(qc, index_k[:, :hi],
                                                weights[:, c0:c1], clens)[None]
                else:
                    score = torch.einsum("bshd,btd->bsht", qc, index_k[:, :hi])
                    score = (score.relu_() * wc).sum(dim=2)
            else:
                score = torch.einsum("bshd,btd->bsht", qc, index_k[:, :hi])  # [b, C, h, hi]
                score.relu_()
                score.mul_(wc)
                score = score.sum(dim=2)                                       # [b, C, hi]
                if lo < hi:
                    # only the small visibility band needs a per-query mask
                    mask = torch.arange(lo, hi, device=x.device)[None, :] >= clens[:, None]
                    score[..., lo:].masked_fill_(mask, -torch.inf)
            if _run_mode in ("prefix-pad", "fused") and hi < npos:
                # exact: restore the reference tensor width so top-k touches identical values
                score = torch.nn.functional.pad(score, (0, npos - hi), value=-torch.inf)
            if _STATE.get("tiebreak") and _run_mode not in ("fused", "hsi"):
                # width-independent deterministic ordering for exact-zero (ReLU) ties: prefer the
                # smaller absolute compressed position. Validation only (env NANOV41_IDX_TIEBREAK).
                bias = torch.arange(score.size(-1), device=score.device, dtype=score.dtype)
                score = score - bias * 1e-6
            idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            outs.append(torch.where(idxs < clens[:, None], idxs + offset, -1).int())
        return torch.cat(outs, dim=1)

    return forward


def install(M=None, chunk: int | None = None, mode: str | None = None) -> dict:
    """Install the chunked prefill indexer. Idempotent; opt-in only."""
    global _STATE
    if M is None:
        from .patched_env import import_patched

        M = import_patched()
    chunk = _chunk_from_env(chunk)
    cls = M.Indexer
    if (_STATE["installed"] and _STATE["cls"] is cls and _STATE["chunk"] == chunk
            and _STATE.get("mode") == (os.environ.get("NANOV41_IDX_MODE", "prefix-pad").strip().lower()
                                       if mode is None else mode.strip().lower())):
        return dict(_STATE)
    if _STATE["installed"]:
        uninstall(M)
    if mode is None:
        mode = os.environ.get("NANOV41_IDX_MODE", "prefix-pad").strip().lower() or "prefix-pad"
    mode = mode.strip().lower()
    _hsi_blk, _hsi_cand = _hsi_knobs()
    _STATE.update({"installed": True, "orig": cls.forward, "cls": cls, "chunk": chunk,
                   "mode": mode, "warned": False, "tiebreak": _tiebreak_enabled(),
                   "hsi_blk": _hsi_blk, "hsi_cand": _hsi_cand,
                   "hsi_auto_min": int(os.environ.get("NANOV41_IDX_HSI_AUTO_MIN", "65536") or 65536)})
    cls.forward = _make_forward(M, cls.forward, chunk, mode)
    if os.environ.get("NANOV41_IDX_NOGRAD", "0").lower() in ("1", "true", "yes", "on"):
        # training only: the indexer emits int indices, no gradient flows through it; wrapping it
        # in no_grad stops autograd from saving the (large) score intermediates of the full scan
        _inner = cls.forward

        def _nograd_forward(*args, **kwargs):
            with torch.no_grad():
                return _inner(*args, **kwargs)

        cls.forward = _nograd_forward
    return dict(_STATE)


def uninstall(M=None) -> None:
    """Restore the reference `Indexer.forward` captured by `install()` (no-op if not installed)."""
    if not _STATE["installed"]:
        return
    cls = _STATE["cls"]
    if cls is not None and _STATE["orig"] is not None:
        cls.forward = _STATE["orig"]
    _STATE.update({"installed": False, "orig": None, "cls": None, "chunk": None, "mode": None,
                   "tiebreak": False})


def available() -> bool:
    try:
        from .patched_env import import_patched

        M = import_patched()
        return hasattr(M, "Indexer")
    except Exception:  # noqa: BLE001
        return False


def get_stats() -> dict:
    return dict(_STATE)
