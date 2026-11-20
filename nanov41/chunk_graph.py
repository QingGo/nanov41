"""Per-chunk CUDA-graph training step for chunked BPTT (opt-A, productionized).

Why one graph per chunk: chunked BPTT is already a sequence of independent (chunk fwd + loss +
backward) units -- the cross-chunk KV caches are detached, so a graph with `start_pos` baked in is
enough per chunk.  The eager step is launch/CPU-bound (measured 38% gap at 16K full-seq; the 1M
chunked step moves ~13.8k kernels/step), and replaying one graph per chunk removes almost all of
that: measured on a 5090, 262K step -27.7% and 1M step -20.4%, *bitwise identical* gradients and,
after the determinism fix (report §3v), bitwise identical weights over multi-step training.

What stays outside the graphs: the optimizer step, `detach_leaked_buffers`, loss readout, cache
zeroing and the data copy into the static input buffers -- all of them are cheap and/or need the
fresh gradients.  What is captured: for every chunk, `chunk_loss(...) * (chunk_len / total_len)`
+ backward, plus a few tiny writes that publish the per-chunk and total loss into static buffers
(so logging needs no per-chunk host sync).

Requirements on the loss function (all graph-capture rules apply):
  * it must be a pure function of the static buffers + parameters (no `.item()`, `.cpu()`, `float()`,
    `if tensor:`... -- otherwise capture raises).  In particular the FP8 CE head
    (`NANOV41_HEAD_FP8=2` / `v2`) is capture-safe, while `train_utils`' bf16 head backward syncs
    with the host and cannot be captured -- use the FP8 head for production,
  * every input tensor that changes per step must be one of `buffers`; the caller updates it with
    `copy_` **before** the replay (`set_inputs` / `step(ids=...)` does that),
  * the model must keep its parameter storages stable after capture (optimizers update in place;
    `load_state_dict` copies in place, but re-building the model requires re-capturing),
  * the model stack must be capture-safe: the production fused stack (grouped Triton MoE, fast
    attention, FP8 CE head) is; the reference MoE loop (`bincount().tolist()`) and the bf16 CE
    head's backward sync with the host and make capture fail with
    "operation not permitted when stream is capturing".

Usage (chunked 1M training):

    trainer = ChunkedGraphTrainer(model, opt, chunks=[(0, C), (C, 2C), ...],
                                  loss_fn=loss_fn,          # (c0, c1, buffers) -> chunk-mean loss
                                  buffers={"ids": ids_buf}, # [b, total] int64, allocated once
                                  total_len=1_048_576, capture=True)
    trainer.capture()                                     # one-time; see capture_seconds/peak_mem_gib
    for batch in loader:
        loss = trainer.step(ids=batch)                     # replay + opt.step + detach; float loss

    # when capture is impossible (CPU, unsupported op, or for debugging):
    trainer = ChunkedGraphTrainer(..., capture=False)     # step() runs the eager pass

Determinism: two identical steps give bitwise identical gradients (report §3v), and graph replay
matches eager bitwise (tools/g46_chunk_train.py re-verifies multi-step parity on every run).
"""
from __future__ import annotations

import gc
import time

import torch

__all__ = ["ChunkedGraphTrainer"]


def _default_zero_caches(model) -> None:
    """Zero the chunked-prefill cross-chunk caches (no-op for models without them).

    Strictly, a full chunk schedule overwrites every cache slot before it is read, so a training
    loop does not need this; keeping it makes the eager and graph paths bit-identical to the
    audit / parity protocol (which zeroes caches between passes).
    """
    for layer in getattr(model, "inner", model).layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        for holder, name in ((attn, "window_kv_cache"), (attn, "compress_kv_cache"),
                             (getattr(attn, "indexer", None), "k_cache")):
            buf = getattr(holder, name, None) if holder is not None else None
            if buf is not None:
                buf.zero_()
        comp = getattr(attn, "compressor", None)
        if comp is not None:
            for name in ("kv_state", "score_state"):
                buf = getattr(comp, name, None)
                if buf is not None:
                    buf.zero_()


class ChunkedGraphTrainer:
    """Capture `n_chunks` graphs (one per chunk, shared memory pool) and replay them per step."""

    def __init__(self, model, optimizer, *, chunks, loss_fn, buffers, total_len=None,
                 capture: bool = True, warmup_passes: int = 2, zero_caches: bool = False,
                 detach_leaked: bool = True, optimizer_in_step: bool = True,
                 loss_scale: bool = True, device=None, engram_optimizer=None):
        """
        model:        the training model; parameters must stay in place after capture.
        optimizer:    optional; stepped outside the graphs (in `step()` unless disabled).
        chunks:       [(c0, c1), ...] covering the sequence in order.
        loss_fn:      callable(c0, c1, buffers) -> chunk loss (mean over the chunk's tokens).
                      It is called inside capture, so it must not sync or read Python-side state
                      that changes between steps.
        buffers:      dict[str, Tensor] of *static* input tensors (e.g. {"ids": ids_buf}).
        total_len:    denominator for the loss weighting; defaults to sum of chunk lengths.
        capture:      False -> eager-only (CPU/debug).
        warmup_passes: side-stream eager passes before capture (compiles kernels + allocates the
                      graph pool state); 2 is what the prototypes validated.
        zero_caches:  zero the chunked-prefill caches before each pass.  Off by default: a full
                      chunk schedule overwrites every cache slot before it is read, so this is
                      only needed to reproduce the audit/parity protocol bit-for-bit.
        detach_leaked: call `grad_ckpt.detach_leaked_buffers(model)` after each pass (frees the
                      autograd graphs that checkpoints attach to module buffers).
        optimizer_in_step: step `optimizer` inside `step()` (still outside the CUDA graphs).
        loss_scale:   weight each chunk loss by `(c1-c0)/total_len` before backward.
        """
        self.model = model
        self.optimizer = optimizer
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("chunks must not be empty")
        self.loss_fn = loss_fn
        self.buffers = dict(buffers)
        self.total_len = int(total_len if total_len is not None else
                             sum(c1 - c0 for c0, c1 in self.chunks))
        self.warmup_passes = max(1, int(warmup_passes))
        self.zero_caches = bool(zero_caches)
        self.detach_leaked = bool(detach_leaked)
        self.optimizer_in_step = bool(optimizer_in_step)
        self.loss_scale = bool(loss_scale)
        self.device = device or next(model.parameters()).device
        self.params = [p for p in model.parameters() if p.requires_grad]
        # optional Engram: sparse table optimizer + the per-chunk record bookkeeping for graphs
        self.engram_optimizer = engram_optimizer
        self.engram_tables = []
        if engram_optimizer is not None:
            from .nano_engram import engram_tables as _engram_tables
            self.engram_tables = _engram_tables(model)
        self.captured = False
        self.capture_error: str | None = None
        self.capture_seconds: float | None = None
        self.peak_mem_gib: float | None = None
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self._grad_ptrs: list[tuple[torch.nn.Parameter, int]] = []
        # static output buffers (written by the captured graphs, read without syncing)
        self._loss_chunks = torch.zeros(len(self.chunks), dtype=torch.float32, device=self.device)
        self._loss_total = torch.zeros((), dtype=torch.float32, device=self.device)
        self._chunk_index = {ch: i for i, ch in enumerate(self.chunks)}
        self._steps = 0
        self._want_capture = bool(capture)

    # ------------------------------------------------------------------ helpers
    def _zero_grads(self) -> None:
        grads = [p.grad for p in self.params if p.grad is not None]
        if grads:
            torch._foreach_zero_(grads)
        for table in self.engram_tables:
            table.zero_leaf_grads()

    def _reset_buffers(self) -> None:
        self._loss_chunks.zero_()
        self._loss_total.zero_()

    def _zero_caches(self) -> None:
        if not self.zero_caches:
            return
        fn = getattr(self, "_zero_caches_fn", None)
        if fn is None:
            fn = _default_zero_caches
        fn(self.model)

    def _detach(self) -> None:
        if not self.detach_leaked:
            return
        try:
            from .grad_ckpt import detach_leaked_buffers
        except ImportError:  # pragma: no cover - standalone import
            from nanov41.grad_ckpt import detach_leaked_buffers
        detach_leaked_buffers(self.model)

    def _chunk_body(self, c0: int, c1: int) -> None:
        """One chunk: loss -> (scaled) loss buffers -> backward.  Runs eagerly or during capture."""
        loss = self.loss_fn(c0, c1, self.buffers)
        scaled = loss * ((c1 - c0) / self.total_len) if self.loss_scale else loss
        # publish losses into static buffers (graph-captured tiny writes; no host sync needed):
        # per-chunk mean for logging, length-weighted mean for the step loss
        self._loss_chunks[self._chunk_index[(c0, c1)]].copy_(loss.detach())
        if self.loss_scale:
            self._loss_total.add_(scaled.detach())
        else:
            self._loss_total.add_(loss.detach())
        scaled.backward()

    def _pass(self) -> None:
        """One full chunked pass (fwd + loss + bwd for every chunk), eager."""
        for table in self.engram_tables:
            table.clear_records()
        self._zero_grads()
        self._reset_buffers()
        self._zero_caches()
        for c0, c1 in self.chunks:
            self._chunk_body(c0, c1)

    def _after_pass(self) -> None:
        self._detach()

    # ------------------------------------------------------------------ capture
    def capture(self) -> "ChunkedGraphTrainer":
        """Warm up on a side stream, then record one graph per chunk into a shared memory pool."""
        if not self._want_capture:
            return self
        if not torch.cuda.is_available() or str(self.device).startswith("cpu"):
            raise RuntimeError("CUDA graph capture requires a CUDA model")
        t0 = time.perf_counter()
        # 0) Engram: warmup/capture must start from empty record lists; after capture the
        #    per-chunk records are replayed by the graphs (grads written in place, zeroed per step).
        for table in self.engram_tables:
            table.clear_records()
            table.set_static_records(False)
        # 1) main-stream warmup: allocate p.grad and compile kernels for every chunk (the chunk
        #    schedule, HSI threshold transitions and per-chunk start_pos are all exercised here).
        self._pass()
        self._after_pass()
        torch.cuda.synchronize()
        # 2) side-stream warmup: PyTorch's recipe so that cuBLAS workspaces / allocator state are
        #    created on a non-default stream, then hand over to the current stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_passes):
                self._pass()
                self._after_pass()
        torch.cuda.current_stream().wait_stream(side)
        for table in self.engram_tables:
            table.clear_records()
        gc.collect()
        torch.cuda.synchronize()
        # 3) capture: one graph per chunk, one shared pool, always replayed in chunk order.
        pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        self.capture_error = None
        try:
            for c0, c1 in self.chunks:
                self._zero_grads()
                self._reset_buffers()
                self._zero_caches()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self._chunk_body(c0, c1)
                self.graphs.append(g)
            torch.cuda.synchronize()
            for table in self.engram_tables:
                table.set_static_records(True)  # keep this run's per-chunk records for replay
        except Exception as exc:  # noqa: BLE001 - caller may fall back to eager
            self.capture_error = f"{type(exc).__name__}: {exc}"
            self.graphs = []
            self.captured = False
            raise
        finally:
            # capture-time autograd graph references (checkpoint replay etc.) are not needed anymore
            self._after_pass()
            gc.collect()
        # The captured graphs write gradients into the *current* p.grad storages.  A later
        # optimizer.zero_grad(set_to_none=True) (or p.grad = ...) would silently disconnect them,
        # so remember the pointers and verify them every step.
        self._grad_ptrs = [(p, p.grad.data_ptr()) for p in self.params if p.grad is not None]
        self.captured = True
        self.capture_seconds = round(time.perf_counter() - t0, 1)
        self.peak_mem_gib = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return self

    # ------------------------------------------------------------------ stepping
    def set_inputs(self, **tensors) -> None:
        """Copy new per-step data into the static input buffers (outside the graphs)."""
        for name, src in tensors.items():
            if name not in self.buffers:
                raise KeyError(f"unknown static buffer {name!r}; have {sorted(self.buffers)}")
            buf = self.buffers[name]
            src_t = src if torch.is_tensor(src) else torch.as_tensor(src)
            if src_t.shape != buf.shape:
                src_t = src_t.reshape(buf.shape)
            buf.copy_(src_t, non_blocking=True)

    def step(self, ids: torch.Tensor | None = None, return_loss: bool = True):
        """Run one training step: [data copy] + replay/eager pass + optimizer + detach.

        The graphs only cover fwd+loss+backward; the optimizer step, detach and the loss readout
        happen here, outside the graphs.  Returns the mean per-token loss as a float when
        `return_loss` (one device->host sync); pass False and call `loss_value()` later to avoid
        the sync.
        """
        if ids is not None:
            self.set_inputs(ids=ids)
        if self.captured:
            for p, ptr in self._grad_ptrs:
                if p.grad is None or p.grad.data_ptr() != ptr:
                    raise RuntimeError(
                        "gradient tensor storage changed after CUDA-graph capture; the captured "
                        "backward writes into the old buffers. Do not use "
                        "optimizer.zero_grad(set_to_none=True) or replace p.grad -- the trainer's "
                        "step() zeroes the existing gradient tensors in place.")
            self._zero_grads()
            self._reset_buffers()
            self._zero_caches()
            for g in self.graphs:
                g.replay()
        else:
            self._pass()
        self._after_pass()
        if self.optimizer_in_step and self.optimizer is not None:
            self.optimizer.step()
        if self.engram_optimizer is not None:
            self.engram_optimizer.step()  # collects replayed row grads, sparse Adam, outside graphs
        self._steps += 1
        if return_loss:
            return self.loss_value()
        return None

    # ------------------------------------------------------------------ readouts
    def loss_value(self) -> float:
        """Mean per-token loss of the last step (one host sync; logging only)."""
        return float(self._loss_total)

    def chunk_losses(self) -> torch.Tensor:
        """Per-chunk losses of the last step (device tensor, no sync)."""
        return self._loss_chunks

    def eager_pass(self) -> None:
        """One eager pass on the current buffers (no optimizer), for parity checks/debug."""
        self._pass()
        self._after_pass()

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        self.graphs = []
        self.captured = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
