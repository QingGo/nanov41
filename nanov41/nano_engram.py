"""Training-capable nano-Engram: bf16 table, deterministic sparse gradients, sparse Adam.

The reference Engram (`ref/engram.py`) is an inference implementation: its table lives in fp8 with
block scales and the 2..max_ngram hash ids are computed under `torch.inference_mode()`.  Training
needs a different path, added here without touching the vendored reference:

1. **bf16 (or fp32) table** — a plain `nn.Buffer` of rows, no fp8 scales.  The buffer follows the
   model dtype, so the existing CPU float32 parity build keeps working.
2. **Deterministic sparse gradients** — the table is detached in the forward pass and re-attached as
   a leaf (`requires_grad_(True)`); after `loss.backward()` the leaf gradient is reduced to rows by a
   *sorted segmented sum* (`segment_sum_rows`), never by atomic `index_add`.  This is what lets the
   repo keep its bitwise eager-vs-graph parity gate.  (CUDA graph capture of the hash computation is
   still TODO: hash ids must move to static graph inputs on GPU; see docs/engram-design.md.)
3. **Sparse Adam** — only touched rows get moments.  Moments are fp32 in a packed growing buffer
   (`_PackedMoments`), so a 128 M-param table does not allocate dense optimizer state.
4. **Install** — `install()` swaps `ParallelEngramEmbedding` in the patched reference module and
   unwraps the `inference_mode` decorator on `NgramHashState.forward`, so hash ids stay normal
   tensors and can index the training table.

`ENGRAM_PRESETS` holds the geometry from `docs/engram-design.md` for d12/d16/d20 (and a `tiny`
preset used by tests).  The current scratch configs keep Engram disabled until P1 lands; `train.py
--engram d12 --tokenizer ...` enables it.

Usage::

    from nanov41.nano_engram import install, apply_preset, engram_tables, SparseRowAdam
    install()
    cfg = apply_preset(CONFIGS["d12"], "d12")
    model = NanoV41(cfg, tokenizer=tokenizer_adapter(tokenizer_path))
    ...
    loss.backward()
    sparse_opt.step()          # after the dense optimizer step, collects row grads internally
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterable

import torch
import torch.nn as nn

# Geometry from docs/engram-design.md.  `engram_num_embeddings` are the exact sums of the prime
# buckets drawn by EngramLayout (verified against ref/config.json: official rows 384,006,168 /
# 384,016,682), so every hash id is a valid row and no id is masked off.
ENGRAM_PRESETS: dict[str, dict] = {
    "d12": {
        "engram_layer_ids": [1, 4],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_vocab_size": 500_000,
        "engram_num_embeddings": [2_000_136, 2_000_370],
    },
    # P2 capacity sweep variants of the d12 preset (same layers/heads/ngram, table rows x0.5/x2).
    "d12_thin": {
        "engram_layer_ids": [1, 4],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_vocab_size": 250_000,
        "engram_num_embeddings": [1_000_078, 1_000_180],
    },
    "d12_thick": {
        "engram_layer_ids": [1, 4],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_vocab_size": 1_000_000,
        "engram_num_embeddings": [4_000_112, 4_000_418],
    },
    "d16": {
        "engram_layer_ids": [1, 6],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_vocab_size": 1_000_000,
        "engram_num_embeddings": [4_000_112, 4_000_418],
    },
    "d20": {
        "engram_layer_ids": [1, 7],
        "engram_max_ngram_size": 3,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_vocab_size": 2_000_000,
        "engram_num_embeddings": [8_000_152, 8_000_396],
    },
    # Tiny smoke geometry (used by CPU tests / train.py --engram tiny).
    "tiny": {
        "engram_layer_ids": [1],
        "engram_max_ngram_size": 2,
        "engram_n_heads": 1,
        "engram_head_dim": 32,
        "engram_vocab_size": 512,
        "engram_num_embeddings": [521],
    },
}

# Hash multipliers derive from this; pinned by docs/tokenizer-contract.md (V4.1-Flash tokenizer).
DEFAULT_COMPRESSED_VOCAB_SIZE = 99_092
DEFAULT_PAD_ID = 2


def apply_preset(cfg: dict, name: str) -> dict:
    """Return a copy of ``cfg`` with the engineering preset's engram_* fields applied."""
    if name not in ENGRAM_PRESETS:
        raise ValueError(f"unknown Engram preset {name!r}; known: {sorted(ENGRAM_PRESETS)}")
    out = dict(cfg)
    out.update(ENGRAM_PRESETS[name])
    out["engram_compressed_vocab_size"] = DEFAULT_COMPRESSED_VOCAB_SIZE
    out["engram_pad_id"] = DEFAULT_PAD_ID
    return out


def engram_param_count(model: nn.Module) -> int:
    """Number of table parameters (excluded from model.num_params by design)."""
    return sum(t.weight.numel() for t in engram_tables(model))


def engram_tables(model: nn.Module) -> list["NanoEngramTable"]:
    return [m for m in model.modules() if isinstance(m, NanoEngramTable)]


# --------------------------------------------------------------------------- deterministic seg-sum
def segment_sum_rows(ids: torch.Tensor, grads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic row-wise sum: ids [N] -> (unique sorted rows [R], summed grads [R, H]).

    Uses a stable argsort + fp32 prefix sum over the sorted rows, so duplicated hash ids (collisions
    and repeated n-grams in a batch) are summed in a fixed order on CPU and CUDA.  Never atomics.
    """
    if ids.ndim != 1:
        ids = ids.reshape(-1)
    n = ids.shape[0]
    if n == 0:
        return ids.to(torch.long), grads.new_zeros((0, grads.shape[-1]))
    # Engram hash ids are non-negative and < 2^31 for every preset, so sort in int32 (smaller/faster
    # keys).  On CUDA use torch.segment_reduce: it is a fixed-tree segmented sum (deterministic for a
    # given input) and avoids the explicit cumsum gather.  CPU keeps the cumsum implementation so the
    # exact CPU unit-test reference is unchanged.
    ids32 = ids.to(torch.int32) if ids.numel() < 2**31 else ids
    order = torch.argsort(ids32, stable=True)
    sorted_ids = ids[order]
    sorted_grad = grads[order].to(torch.float32)          # [N, H]
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    if grads.is_cuda and hasattr(torch, "segment_reduce"):
        sums = torch.segment_reduce(sorted_grad, "sum", lengths=counts, axis=0)
        return uniq.to(torch.long), sums.contiguous()
    # Scan along a contiguous last dimension: cumsum over [N, H] dim0 is ~130x slower on CUDA
    # (few independent columns), while [H, N] dim1 scans are fast and deterministic.
    csum = torch.cumsum(sorted_grad.t().contiguous(), dim=1)  # [H, N]
    ends = torch.cumsum(counts, dim=0) - 1                # [R]
    starts = ends - counts + 1
    prev = torch.where((starts > 0).unsqueeze(0), csum[:, (starts - 1).clamp(min=0)],
                       torch.zeros_like(csum[:, ends]))
    sums = (csum[:, ends] - prev).t().contiguous()        # [R, H]
    return uniq.to(torch.long), sums


# --------------------------------------------------------------------------- table
class NanoEngramTable(nn.Module):
    """Training table: rows x head_dim buffer, sparse row gradients, no fp8 scales."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        dtype = torch.get_default_dtype()
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = torch.bfloat16
        self.register_buffer("weight", torch.empty(self.num_embeddings, self.dim, dtype=dtype),
                             persistent=True)
        self.reset_parameters()
        # (ids, values-leaf) pairs from forwards not yet reduced by collect_row_grads()
        self._records: list[tuple[torch.Tensor, torch.Tensor]] = []
        # per-forward (unique rows, fp32 summed grads); combined vectorised in merged_row_grads()
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        # CUDA-graph mode: capture records once; replay rewrites the leaf grads in place, so
        # collect_row_grads() must keep the records and zero (not replace) the grad buffers.
        self._static_records = False
        self._static_zero_grads = True

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02, chunk_rows: int = 1 << 22) -> None:
        # Small random rows; chunked fp32 generation because CPU bf16 normal_ is not portable.
        for start in range(0, self.num_embeddings, chunk_rows):
            end = min(start + chunk_rows, self.num_embeddings)
            rows = torch.randn(end - start, self.dim, dtype=torch.float32) * std
            self.weight[start:end].copy_(rows.to(self.weight.dtype))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        out_of_range = (flat < 0) | (flat >= self.num_embeddings)
        safe = flat.clamp(0, self.num_embeddings - 1)
        values = self.weight.index_select(0, safe).view(*indices.shape, self.dim)
        # Always mask (no `.any()` host sync): CUDA-graph capture forbids value-dependent branches.
        values = values.masked_fill(out_of_range.view(*indices.shape, 1), 0)
        if torch.is_grad_enabled():
            # detach -> leaf; backward fills leaf.grad, then a deterministic row reduction happens
            # in collect_row_grads() (called by the sparse optimizer, not by autograd).
            values = values.detach().requires_grad_(True)
            self._records.append((indices.detach(), values))
        return values

    def collect_row_grads(self) -> None:
        """Reduce recorded leaf gradients to per-forward (sorted rows, fp32 sums) - vectorised."""
        for ids, values in self._records:
            grad = values.grad
            if grad is None:
                continue
            rows, sums = segment_sum_rows(ids.reshape(-1), grad.reshape(-1, self.dim))
            if rows.numel():
                self._pending.append((rows, sums))
            if self._static_records:
                if self._static_zero_grads:
                    values.grad.zero_()  # captured tensors: keep the buffer for the next replay
            else:
                values.grad = None
        if not self._static_records:
            self._records.clear()

    def merged_row_grads(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorted unique rows and fp32 summed grads, deterministically combined across forwards."""
        device = self.weight.device
        if not self._pending:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, self.dim, dtype=torch.float32, device=device))
        if len(self._pending) == 1:
            rows, sums = self._pending[0]
            return rows.to(device), sums.to(device)
        rows = torch.cat([r for r, _ in self._pending])
        sums = torch.cat([g for _, g in self._pending])
        rows, sums = segment_sum_rows(rows, sums)  # stable sorted order across forwards
        return rows.to(device), sums.to(device)

    def zero_row_grad(self) -> None:
        self._pending.clear()

    # ------------------------------------------------------------ CUDA-graph support
    def clear_records(self) -> None:
        """Drop capture/eager gradient records (before warmup passes or before re-capture)."""
        self._records.clear()
        self._pending.clear()

    def set_static_records(self, flag: bool = True, zero_grads: bool = True) -> None:
        """Capture-time records are replayed by CUDA graphs; keep them and zero grads in place.

        ``zero_grads=False`` is valid for assignment-based captures (R69's
        ``torch.autograd.grad`` recipe): replay overwrites the whole leaf-grad buffer, so the
        pre-replay zeroing is unnecessary.  Backward-accumulation captures (R68 chunked BPTT)
        must keep it True.
        """
        self._static_records = bool(flag)
        self._static_zero_grads = bool(zero_grads)
        if not flag:
            self.clear_records()

    def leaf_grads(self) -> list[torch.Tensor]:
        """Grad buffers written by the captured backward, one per recorded forward."""
        return [values.grad for _, values in self._records if values.grad is not None]

    def single_record(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(flat ids, flat leaf grad) when exactly one captured forward is pending."""
        if len(self._records) != 1:
            return None
        ids, values = self._records[0]
        if values.grad is None:
            return None
        return ids.reshape(-1), values.grad.reshape(-1, self.dim)

    def zero_leaf_grads(self) -> None:
        """Zero captured leaf-grad buffers in place (must not replace them between replays)."""
        if self._static_records and not self._static_zero_grads:
            return  # assignment-captured graphs overwrite every element on replay
        for values in (v for _, v in self._records):
            if values.grad is not None:
                values.grad.zero_()

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, dim={self.dim}, dtype={self.weight.dtype}"


# --------------------------------------------------------------------------- sparse Adam
class _PackedMoments:
    """Growing fp32 (m, v) buffers with a device-side row -> slot map (sparse Adam state).

    The row->slot map is one int64 tensor over the table's row space (8 B/row), so assigning slots
    for a step is a handful of GPU ops and needs no `rows.tolist()` sync (the first version used a
    Python dict and cost ~40 ms/step at 32 k touched rows).
    """

    def __init__(self, dim: int, num_rows: int, device=None):
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.device = device
        self.row_to_slot = torch.full((self.num_rows,), -1, dtype=torch.long, device=device)
        self.size = 0
        self.m = torch.zeros(0, dim, dtype=torch.float32, device=device)
        self.v = torch.zeros(0, dim, dtype=torch.float32, device=device)

    def slot_ids(self, rows: torch.Tensor) -> torch.Tensor:
        """rows: sorted unique longs on device -> slot ids (assigning new slots as needed)."""
        slots = self.row_to_slot[rows]
        new_mask = slots < 0
        n_new = int(new_mask.sum())   # one host sync (the old version did any() + sum(), i.e. two)
        if n_new:
            self._grow(self.size + n_new)
            new_rows = rows[new_mask]
            new_slots = torch.arange(self.size, self.size + n_new, dtype=torch.long,
                                     device=rows.device)
            self.row_to_slot[new_rows] = new_slots
            self.size += n_new
            slots = self.row_to_slot[rows]
        return slots

    def _grow(self, capacity: int) -> None:
        cur = self.m.shape[0]
        if capacity <= cur:
            return
        new_cap = max(64, 1 << (max(capacity, 1) - 1).bit_length())
        m = torch.zeros(new_cap, self.dim, dtype=torch.float32, device=self.device)
        v = torch.zeros(new_cap, self.dim, dtype=torch.float32, device=self.device)
        if cur:
            m[:cur] = self.m
            v[:cur] = self.v
        self.m, self.v = m, v

    def rows_present(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(rows, slots) for every tracked row, ordered by slot."""
        rows = torch.nonzero(self.row_to_slot >= 0, as_tuple=False).flatten()
        slots = self.row_to_slot[rows]
        order = torch.argsort(slots)
        return rows[order], slots[order]

    def state_dict(self) -> dict:
        rows, slots = self.rows_present()
        return {"rows": rows.cpu(), "m": self.m[slots].cpu(), "v": self.v[slots].cpu()}

    def load_state_dict(self, state: dict) -> None:
        rows = state["rows"].to(torch.long)
        n = int(rows.numel())
        self._grow(n)
        self.row_to_slot.zero_().fill_(-1)
        self.row_to_slot[rows.to(self.device)] = torch.arange(n, dtype=torch.long,
                                                             device=self.device)
        self.size = n
        self.m = state["m"].to(torch.float32).to(self.device)
        self.v = state["v"].to(torch.float32).to(self.device)


class SparseRowAdam:
    """Adam over only the hash-touched rows of one or more NanoEngramTable modules.

    The dense optimizer (AdamW) must NOT receive the table buffers; they are not Parameters.  Call
    `step()` after `loss.backward()` (it collects row grads internally), then `zero_grad()`.
    """

    def __init__(self, tables: Iterable[NanoEngramTable], *, lr: float = 1e-2,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.tables = list(tables)
        self.lr = float(lr)
        self.b0, self.b1 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self._moments = {
            id(t): _PackedMoments(t.dim, t.num_embeddings, t.weight.device) for t in self.tables
        }

    @torch.no_grad()
    def step(self, harvest: bool = True) -> None:
        """Apply one sparse-Adam step.

        ``harvest=True`` (default) first reduces the leaf gradients of the pending forward(s).
        Gradient-accumulated training harvests after *every* micro-batch (calling
        ``table.collect_row_grads()`` from the trainer) and then calls ``step(harvest=False)`` so
        the last micro-batch is not counted twice.
        """
        self.step_count += 1
        bias0 = 1.0 - self.b0 ** self.step_count
        bias1 = 1.0 - self.b1 ** self.step_count
        for table in self.tables:
            # Fast path for graph capture: one forward pending, so the deterministic segment sum and
            # the sparse Adam row update can be fused into one Triton kernel (no intermediate
            # [rows, dim] tensors).  Captures that accumulate multiple forwards/chunks keep the
            # general two-stage path below.
            if harvest and table._static_records and table.weight.is_cuda:
                rec = table.single_record()
                if rec is not None:
                    try:
                        from .sparse_adam_triton import segment_adam_update
                        ids, leaf_grad = rec
                        ids32 = ids.to(torch.int32)
                        order = torch.argsort(ids32, stable=True)
                        sorted_ids = ids[order]
                        uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
                        if int(counts.max()) <= 8:
                            starts = torch.cumsum(counts, dim=0) - counts
                            sorted_grad = leaf_grad[order]  # bf16, reordered
                            moments = self._moments[id(table)]
                            moments.slot_ids(uniq)
                            segment_adam_update(
                                table.weight, uniq, starts, counts, sorted_grad,
                                moments.row_to_slot, moments.m, moments.v,
                                lr=self.lr, b0=self.b0, b1=self.b1, bias0=bias0,
                                bias1=bias1, eps=self.eps)
                            table.zero_row_grad()
                            continue
                    except Exception:  # noqa: BLE001 - fall back to the general path
                        pass
            if harvest:
                table.collect_row_grads()
            rows, grads = table.merged_row_grads()
            if rows.numel() == 0:
                continue
            moments = self._moments[id(table)]
            slots = moments.slot_ids(rows)
            if table.weight.is_cuda:
                try:
                    from .sparse_adam_triton import sparse_adam_update
                    sparse_adam_update(
                        table.weight, rows, grads, moments.row_to_slot, moments.m, moments.v,
                        lr=self.lr, b0=self.b0, b1=self.b1, bias0=bias0, bias1=bias1,
                        eps=self.eps)
                    continue
                except Exception:  # noqa: BLE001 - fall back to the torch path
                    pass
            param = table.weight.index_select(0, rows).to(torch.float32)
            grad = grads
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            m = moments.m[slots]
            v = moments.v[slots]
            m = self.b0 * m + (1.0 - self.b0) * grad
            v = self.b1 * v + (1.0 - self.b1) * grad.square()
            moments.m[slots] = m
            moments.v[slots] = v
            m_hat = m / bias0
            v_hat = v / bias1
            new = param - self.lr * m_hat / (v_hat.sqrt() + self.eps)
            table.weight.index_copy_(0, rows, new.to(table.weight.dtype))
        self.zero_grad()

    def zero_grad(self) -> None:
        for table in self.tables:
            table.zero_row_grad()

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "lr": self.lr,
                "betas": [self.b0, self.b1],
                "tables": [moments.state_dict() for moments in self._moments.values()]}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state.get("step_count", 0))
        for table, saved in zip(self.tables, state.get("tables", [])):
            self._moments[id(table)].load_state_dict(saved)


# --------------------------------------------------------------------------- fast training forward
def fast_engram_forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                        token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Training forward with bf16 (dtype of x) gate math instead of fp32 copies.

    The reference inference forward casts every [B, L, hc, dim] operand to fp32, which at 64 k
    tokens means ~400 MB per temporary and dominates the graph step.  This variant keeps the big
    tensors in the model dtype, does only the tiny [B, L, hc, 1] gate math in fp32, and uses
    `addcmul` for the residual write.  Numerics differ from `ref/engram.py` in the gate's rounding
    (training path only; documented in docs/engram-design.md).
    """
    kv = self.wkv(self.embed(hash_ids).flatten(-2))
    key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
    key = key.unflatten(-1, (self.hc_mult, self.dim))
    value = value.unsqueeze(-2)
    weight = (self.q_weight * self.k_weight).to(key.dtype)  # [hc, dim]
    h2 = x.square().mean(-1, keepdim=True)
    k2 = key.square().mean(-1, keepdim=True)
    rstd = torch.rsqrt(h2 + self.eps) * torch.rsqrt(k2 + self.eps)
    dot = (x * weight * key).sum(-1, keepdim=True) * rstd * (self.dim ** -0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return torch.addcmul(x, gate.to(x.dtype), value)  # gate [B,L,hc,1] * value [B,L,1,dim]


# --------------------------------------------------------------------------- installation
_installed_modules: list = []


def install(module=None):
    """Patch the reference module for training: bf16 table + non-inference hash state."""
    if module is None:
        from .patched_env import import_patched

        module = import_patched()
    if module in _installed_modules:
        return module
    module.ParallelEngramEmbedding = NanoEngramTable
    if os.environ.get("NANOV41_ENGRAM_FAST", "1").strip().lower() not in (
            "0", "false", "no", "off"):
        tl_fwd = None
        if os.environ.get("NANOV41_ENGRAM_TL", "1").strip().lower() not in (
                "0", "false", "no", "off"):
            try:
                from .engram_tl import tl_available, tl_engram_forward
                if tl_available():
                    tl_fwd = tl_engram_forward
            except Exception:  # noqa: BLE001 - optional dependency
                tl_fwd = None
        compiled = None
        if os.environ.get("NANOV41_ENGRAM_COMPILE", "0").strip().lower() not in (
                "0", "false", "no", "off"):
            # Inductor (Triton) fuses the gate elementwise chain into a few kernels; compile during
            # warmup, then the captured graph replays the generated kernels.
            compiled = torch.compile(fast_engram_forward, dynamic=False, fullgraph=False)

        def _engram_forward(self, x, hash_ids, token_mask=None):
            # TileLang gate kernels: native official (hidden%256==0, e.g. d16) or nano-adapted
            # 128-tile kernels (d12/d20) with no padding; plain/compiled torch otherwise.
            if tl_fwd is not None and x.shape[-1] % 128 == 0:
                return tl_fwd(self, x, hash_ids, token_mask)
            if compiled is not None:
                return compiled(self, x, hash_ids, token_mask)
            return fast_engram_forward(self, x, hash_ids, token_mask)

        module.Engram.forward = _engram_forward
    fwd = module.NgramHashState.forward
    if hasattr(fwd, "__wrapped__"):  # @torch.inference_mode() from ref/engram.py
        module.NgramHashState.forward = fwd.__wrapped__
    _installed_modules.append(module)
    return module


# --------------------------------------------------------------------------- tokenizer adapter
class TokenizerAdapter:
    """Minimal HF-style tokenizer wrapper that ref/engram.build_compressed_token_map expects."""

    def __init__(self, tokenizer_path: str | pathlib.Path):
        try:
            import tokenizers  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Engram needs `tokenizers` (pip install tokenizers)") from exc
        self.backend_tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._n = self.backend_tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self._n


def load_tokenizer_adapter(path: str | pathlib.Path) -> TokenizerAdapter:
    return TokenizerAdapter(path)


__all__ = [
    "ENGRAM_PRESETS",
    "DEFAULT_COMPRESSED_VOCAB_SIZE",
    "DEFAULT_PAD_ID",
    "apply_preset",
    "engram_param_count",
    "engram_tables",
    "segment_sum_rows",
    "fast_engram_forward",
    "NanoEngramTable",
    "SparseRowAdam",
    "install",
    "TokenizerAdapter",
    "load_tokenizer_adapter",
]
