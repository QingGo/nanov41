# SPDX-License-Identifier: MIT
"""Continuous-batching decode engine with per-step CUDA graphs (M3).

Why this shape: the model context is 1024 tokens and every attention/compressor cache is already
allocated ``[max_batch_size, max_seq_len, ...]`` (the checkpoint's own batch dimension), so there is
nothing to evict and no page table to walk -- a request can own one *slot* (one batch row) for its
whole life.  Paging only pays when the cache is larger than the working set or sequences must share
prefixes; neither holds here, so the runtime uses **slot rows** with the graph replayed per bucket
size (``B = 1..max_batch``) on static buffers.

The engine:

* captures one CUDA graph per batch size at startup (``graph_forward`` with token [B,1] / pos [B]),
  so a step is one ``cudaGraphLaunch`` for the whole batch;
* prefills a new request through the reference eager forward on a **scratch cache row** (batch=1)
  and then copies that row into its slot -- the reference prefill always writes batch row 0, so the
  live cache rows are swapped out for the duration of the prefill;
* advances the batch with GPU-only ops when the request lengths are known in advance
  (``generate_batch_parallel``: replay -> argmax -> copy back, no host sync in the loop), and with a
  host sync per step for mixed-length/continuous serving;
* compacts slots when a request finishes (copy the last live row into the freed row).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .decode_graph import graph_forward


def log(msg: str) -> None:
    print(f"[batch] {msg}", flush=True)


# ----------------------------------------------------------------------------- cache plumbing
def iter_cache_buffers(inner):
    """Yield ``(holder, name, tensor)`` for every per-sequence cache tensor (batch dim 0)."""
    for layer in getattr(inner, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        if isinstance(getattr(attn, "window_kv_cache", None), torch.Tensor):
            yield attn, "window_kv_cache", attn.window_kv_cache
        if isinstance(getattr(attn, "compress_kv_cache", None), torch.Tensor):
            yield attn, "compress_kv_cache", attn.compress_kv_cache
        idxr = getattr(attn, "indexer", None)
        if idxr is not None and isinstance(getattr(idxr, "k_cache", None), torch.Tensor):
            yield idxr, "k_cache", idxr.k_cache
        comp = getattr(attn, "compressor", None)
        if comp is not None:
            for name in ("kv_state", "score_state"):
                buf = getattr(comp, name, None)
                if isinstance(buf, torch.Tensor):
                    yield comp, name, buf
    hash_state = getattr(inner, "engram_hash", None)
    if hash_state is not None and isinstance(getattr(hash_state, "cache", None), torch.Tensor):
        yield hash_state, "cache", hash_state.cache


def copy_cache_row(inner, src: int, dst: int) -> None:
    """Copy sequence row ``src`` -> ``dst`` in every cache tensor (slot compaction)."""
    if src == dst:
        return
    for _holder, _name, buf in iter_cache_buffers(inner):
        buf[dst].copy_(buf[src])


class ScratchPrefill:
    """Swap the live cache tensors for batch-1 scratch rows while the eager prefill runs."""

    def __init__(self, inner):
        self.inner = inner
        self._specs = [(holder, name, buf) for holder, name, buf in iter_cache_buffers(inner)]
        self._scratch: list[torch.Tensor] = []

    def __enter__(self):
        self._scratch = [torch.zeros_like(buf[:1]) for _h, _n, buf in self._specs]
        for (holder, name, _buf), scratch in zip(self._specs, self._scratch):
            setattr(holder, name, scratch)
        return self

    def prefilled(self) -> list[torch.Tensor]:
        """The scratch rows after a batch-1 prefill (call inside the ``with`` block)."""
        return self._scratch

    def copy_into(self, row: int) -> None:
        for (holder, name, buf), scratch in zip(self._specs, self._scratch):
            buf[row].copy_(scratch[0])

    def __exit__(self, *exc):
        for (holder, name, _buf) in self._specs:
            setattr(holder, name, _buf)
        self._scratch = []
        return False


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_new: int
    out: list[int] = field(default_factory=list)
    token: int = 0
    pos: int = 0

    @property
    def done(self) -> bool:
        return len(self.out) >= self.max_new


class BatchDecoder:
    """Continuous batching over slot rows, one captured graph per batch size."""

    def __init__(self, decoder, max_batch: int | None = None):
        self.decoder = decoder
        self.model = decoder.model
        self.inner = decoder.inner
        self.device = decoder.device
        self.nb = int(max_batch or decoder.max_batch or 8)
        self.token = torch.zeros(self.nb, 1, dtype=torch.int64, device=self.device)
        self.pos = torch.zeros(self.nb, dtype=torch.int64, device=self.device)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._logits: dict[int, torch.Tensor] = {}
        self.active: list[Request] = []
        self._finished: list[Request] = []
        self._next_rid = 0
        self._captured = False
        self._prefill_scratch = ScratchPrefill(self.inner)

    # -- graphs -----------------------------------------------------------------------
    @torch.no_grad()
    def capture_all(self, *, warmup: int = 3) -> None:
        """Capture one graph per batch size (1..max_batch) on static token/pos buffers."""
        if self._captured:
            return
        torch.cuda.synchronize()
        for b in range(1, self.nb + 1):
            tok = self.token[:b]
            pos = self.pos[:b]
            tok.zero_()
            pos.zero_()
            current = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    graph_forward(self.model, tok, pos)
            current.wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = graph_forward(self.model, tok, pos)
            self._graphs[b] = graph
            self._logits[b] = out
            torch.cuda.synchronize()
        self._captured = True
        from .generate import reset_caches

        reset_caches(self.inner)          # capture warmups advanced the caches
        log(f"captured {len(self._graphs)} step graphs (batch 1..{self.nb})")

    # -- request lifecycle -------------------------------------------------------------
    @torch.no_grad()
    def add_request(self, prompt_ids: list[int], max_new: int) -> Request | None:
        """Prefill ``prompt_ids`` on the scratch row, copy it into a free slot, return the request."""
        if len(self.active) >= self.nb:
            return None
        req = Request(rid=self._next_rid, prompt=list(prompt_ids), max_new=int(max_new))
        self._next_rid += 1
        from .generate import reset_caches
        from .model import inner_forward

        with self._prefill_scratch as scratch:
            reset_caches(self.inner)
            ids = torch.tensor([req.prompt], dtype=torch.long, device=self.device)
            logits = inner_forward(self.inner, ids, start_pos=0, full_logits=True)
            req.token = int(logits[0, -1].argmax())
            req.pos = len(req.prompt)
            req.out.append(req.token)
            scratch.copy_into(len(self.active))
        self.active.append(req)
        return req

    def _remove(self, idx: int) -> None:
        req = self.active[idx]
        last = len(self.active) - 1
        if idx != last:
            copy_cache_row(self.inner, src=last, dst=idx)
            self.active[idx] = self.active[last]
        self.active.pop()
        self._finished.append(req)

    # -- stepping ----------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> list[list[int]]:
        """Advance every active request by one token (host sync at the end)."""
        b = len(self.active)
        if b == 0:
            return []
        while self.active and self.active[0].done:      # finished during prefill (max_new <= 1)
            self._remove(0)
        b = len(self.active)
        if b == 0:
            return []
        tok_host = torch.tensor([[r.token] for r in self.active], dtype=torch.int64)
        pos_host = torch.tensor([r.pos for r in self.active], dtype=torch.int64)
        self.token[:b].copy_(tok_host.to(self.device))
        self.pos[:b].copy_(pos_host.to(self.device))
        self._graphs[b].replay()
        logits = self._logits[b]
        new_tokens = logits.argmax(dim=-1).reshape(b).tolist()
        finished = []
        for i, (req, tok) in enumerate(zip(self.active, new_tokens)):
            req.out.append(tok)
            req.token = tok
            req.pos += 1
            if req.done or req.pos >= int(self.inner.max_seq_len):
                finished.append(i)
        for i in reversed(finished):
            self._remove(i)
        return [r.out for r in self.active]

    @torch.no_grad()
    def step_gpu(self, n_active: int) -> None:
        """One batch step entirely on-device (no host sync); assumes equal request lengths."""
        self._graphs[n_active].replay()
        new_tokens = self._logits[n_active].argmax(dim=-1).reshape(n_active)
        self.token[:n_active].copy_(new_tokens.unsqueeze(1))
        self.pos[:n_active].add_(1)

    # -- generation --------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch_parallel(self, prompts: list[list[int]], n_new: int,
                                *, record_tokens: bool = True) -> dict:
        """All prompts prefilled, then ``n_new`` graph steps with no host sync in the loop.

        Fair for throughput: every sequence has the same output length and nothing is decoded
        between replays except the on-device argmax/copy.  Returns the per-request token lists
        (collected on device and read back once) plus timing.
        """
        import time

        self.capture_all()
        n = len(prompts)
        assert n <= self.nb, f"batch of {n} exceeds max_batch {self.nb}"
        self.active = []
        self._finished = []
        for prompt in prompts:
            req = self.add_request(prompt, n_new)
            assert req is not None
        # seed the static step buffers with each request's first generated token / position
        self.token[:n].copy_(torch.tensor([[r.token] for r in self.active], dtype=torch.int64,
                                          device=self.device))
        self.pos[:n].copy_(torch.tensor([r.pos for r in self.active], dtype=torch.int64,
                                        device=self.device))
        torch.cuda.synchronize()

        out_dev = torch.zeros(n_new, n, dtype=torch.int64, device=self.device)
        # first generated token was produced by the prefill
        out_dev[0, :] = self.token[:n, 0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(1, n_new):
            self.step_gpu(n)
            if record_tokens:
                out_dev[i, :] = self.token[:n, 0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = out_dev.tolist()
        per_req = [[int(tokens[i][j]) for i in range(n_new)] for j in range(n)]
        n_steps = max(n_new - 1, 1)
        return {
            "tokens": per_req,
            "n": n,
            "n_new": n_new,
            "elapsed_s": elapsed,
            "aggregate_tok_s": (n * n_new) / elapsed,
            "per_seq_tok_s": n_new / elapsed,
            "ms_per_step": elapsed / n_steps * 1e3,
        }

    @torch.no_grad()
    def generate_batch_continuous(self, prompts: list[list[int]], n_new: int) -> dict:
        """Same as above, but steps through the host-synchronising path (request order preserved)."""
        import time

        self.capture_all()
        self._finished = []
        for prompt in prompts:
            self.add_request(prompt, n_new)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = 0
        while self.active:
            self.step()
            steps += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        done = sorted(self._finished, key=lambda r: r.rid)
        return {"tokens": [r.out for r in done], "n": len(done), "n_new": n_new,
                "elapsed_s": elapsed, "steps": steps,
                "aggregate_tok_s": sum(len(r.out) for r in done) / elapsed,
                "ms_per_step": elapsed / max(steps, 1) * 1e3}


__all__ = ["BatchDecoder", "Request", "iter_cache_buffers", "copy_cache_row", "ScratchPrefill"]
