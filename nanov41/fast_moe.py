"""Fast, CUDA-graph-safe MoE forward for the d12 model (drop-in for `MoE.forward`).

`fast_moe_forward` is mathematically the same as `_ref_patched_model.MoE.forward`
(sqrtsoftplus gate, `route_scale`, `norm_topk_prob`, both SwiGLU clamps, shared expert,
`y.type_as(x).view(shape)`), but it removes everything that is hostile to a CUDA graph and
to a GPU's launch budget:

  ref (per layer, T=8192, k=2, E=16)              fast (grouped path)
  -------------------------------------------------------------------------------------
  `bincount(...).tolist()`     host sync          `scatter_add_` histogram + `cumsum` on GPU
  `torch.where(indices == i)`  dynamic shapes     one stable `argsort` (+ fixed-size gather)
  (= aten.nonzero, one per expert)
  `if counts[i] == 0`          host branch        `offs` (int32, group ends) is a tensor
  `y[idx] += expert(...)`      dynamic scatter    one fixed-shape index_add / index_copy
  3 small GEMMs/expert/launch                    2 grouped GEMMs for all 16 experts
  3 GEMMs for the shared expert                  2 GEMMs (w1/w3 fused), see FUSE_W13

Nothing on the **grouped path** or on the **static-routing loop** calls `.item()`, `.tolist()`,
`.cpu()`, `bool(tensor)`, `nonzero` or `masked_select`; every shape they allocate is a static
function of (T, k, dim, inter, E).  The only deliberate host sync is `record_group_sizes()`
(eager, before capture) and the plain loop fallback's `counts.tolist()`.

!!! CUDA-graph status of `torch._grouped_mm` (measured by the team on torch 2.12.1) !!!
The op itself is **not capturable**: it copies `offs` to the host internally to derive the
per-group tile counts, so `torch.cuda.graph()` around it fails with a "CPU<->CUDA copy during
capture" error (sm89 and sm120 alike; see nanov41/graph_moe.py, which works around it by baking
the per-expert token counts before capture).  Consequently, on today's torch:
  * eager speed          -> grouped_mm path (`fast_moe_forward` default on CUDA bf16)
  * CUDA-graph training  -> **static-routing fused loop** (`record_group_sizes` +
                            `install_static_routing` below): group sizes are recorded once in
                            an eager warmup, the in-graph forward only recomputes argsort /
                            gather / GEMMs / scatter (no `.tolist()`, no dynamic shapes).
                            graph_moe.py does the same with the unfused loop; this variant
                            keeps the w1/w3 fusion and fp32 combine.
  * caveat               -> static sizes are valid only while routing is unchanged (fixed
                            ids/weights), exactly like graph_moe.py.  Re-record if data or
                            weights change; do not put the optimizer inside the graph.
`graph_safe(moe, x)` is a design-level check; use `graph_safe(moe, x, verify=True)` or
`probe_grouped_mm_capture()` to test the actual torch, and `assert_no_sync()` to audit our own
fwd+bwd for host syncs.  `python3 tools/test_fast_moe.py --bench --capture` runs both.

CUDA-graph recipe (torch <= 2.12):
    sizes = fm.record_group_sizes(moe, layer_input)   # eager, outside capture
    fm.install_static_routing(moe, sizes)             # this module now uses the sync-free loop
    ... capture the step with GraphStep ...
    fm.uninstall_static_routing(moe)                  # back to auto (grouped on CUDA)
For the whole model, get per-layer sizes from graph_moe.record_static_routing(model, loss_fn)
and call install_static_routing on each MoE module.

Why `scatter_add_` instead of `torch.bincount`: `bincount`'s output length is data dependent
(the CUDA implementation derives the output length from the input's max index and the
op is tagged `dynamic_output_shape`), so it is a host sync even when `minlength=E`.  A zero-filled
`[E]` int64 tensor plus `scatter_add_(0, flat_idx, ones)` has a static shape, is exact
(integer adds) and costs ~one tiny kernel.

--------------------------------------------------------------------------------------------
Tensor-core utilisation: 16 experts x 3 small GEMMs (T=8192, topk=2, inter=dim=384)
--------------------------------------------------------------------------------------------
Per layer, routed fwd FLOPs = 2(topk) * 3(w1,w3,w2) * 2*dim*inter per token
                            = 14.50 GFLOP; shared expert = 7.25 GFLOP  -> 21.74 GFLOP.
fwd+bwd is ~3x that (one dA and one dW GEMM per forward GEMM) = 65.2 GFLOP/layer,
783 GFLOP for the 12-layer step.  At the measured 61.1 ms/step that is **12.8 TFLOP/s,
~3% of a 5090's ~419 TFLOP/s bf16 dense peak** - so the problem is not FLOPs, it is shape.

* With a 128x128 CTA tile an expert GEMM (M=1024, N=384) has only 8*3 = 24 CTAs; a 5090 has
  170 SMs, so a single-expert launch occupies ~14% of the device (observed kernels use
  128x128 or 256x128 tiles, i.e. 24 or 12 CTAs).  K=384 is only 3 k-iterations of 128, so
  prologue/epilogue/launch latency is amortised over 0.30 GFLOP - the kernel is latency- and
  wave-quantisation-bound, not tensor-core-bound.
* 48 routed + 3 shared fwd GEMMs/layer, plus ~2x as many in bwd, plus one one-arg `torch.where`
  (= `aten.nonzero`) + one indexed write per expert: ~200 launches/layer.  That is where the 61 ms goes.
* Per-expert groups are variable length (top-2, E=16: mean T*k/E = 1024, sd ~31), so the
  Python loop cannot overlap experts; each expert's 24-CTA GEMM runs alone.

`torch._grouped_mm` (CUDA, bf16, `offs` = per-group end offsets) collapses the routed part
into one launch per matrix.  With w1/w3 concatenated: M=16384, N=768 (fused) / 384 (w2),
K=384; that is 768 / 384 CTA tiles (4.5 / 2.3 waves at 128x128), and A [16384,384] is read
once.  A pure streaming roofline (A 12.6 MB once + B 9.4 MB + C 25 MB, L2-resident A, 5090
HBM ~1.8 TB/s) is ~26-60 us per grouped GEMM; the FLOP roofline at 100-200 TFLOP/s is
48-97 us.  So a well-scheduled grouped kernel should land at **~100-200 TFLOP/s for these
shapes** (24-48% MFU) and make routed fwd ~0.15-0.25 ms/layer.  Caveat: the measured
`runs/5090_b8_s1024_fused_{loop,gmm}.json` show the *current* torch 2.12/CUTLASS grouped
path only improving fwd by ~6% and regressing bwd by ~9% at this exact shape (its segmented
wgrad is the weak part).  Hence this module keeps a **cuBLAS loop fallback with the same
w1/w3 fusion** (9 -> 6 GEMMs/expert) so the Lead can A/B both on the real machine:
`python3 tools/test_fast_moe.py --bench`.

Knobs (env or `configure()`):
  NANOV41_MOE_FORCE_LOOP=1     always use the cuBLAS loop (also useful on old CUDA)
  NANOV41_MOE_SCATTER=auto|index_add|scatter_add|permute
                               auto = index_add on CUDA, permute on CPU
  NANOV41_MOE_FUSE_W13=0       do not fuse w1/w3 into one GEMM
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

_GROUPED_MM = getattr(torch, "_grouped_mm", None)

_VALID_SCATTER = ("auto", "index_add", "scatter_add", "permute")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "off", "no")


FORCE_LOOP = _env_flag("NANOV41_MOE_FORCE_LOOP", False)
FUSE_W13 = _env_flag("NANOV41_MOE_FUSE_W13", True)
SCATTER_MODE = (os.environ.get("NANOV41_MOE_SCATTER", "auto") or "auto").strip().lower()
if SCATTER_MODE not in _VALID_SCATTER:  # fail loudly at import instead of per step
    raise ValueError(f"NANOV41_MOE_SCATTER must be one of {_VALID_SCATTER}, got {SCATTER_MODE!r}")


def configure(*, force_loop: bool | None = None, scatter: str | None = None,
              fuse_w13: bool | None = None) -> None:
    """Change the knobs in-process (used by the A/B bench; env vars cover the normal case)."""
    global FORCE_LOOP, SCATTER_MODE, FUSE_W13
    if force_loop is not None:
        FORCE_LOOP = bool(force_loop)
    if scatter is not None:
        scatter = scatter.strip().lower()
        if scatter not in _VALID_SCATTER:
            raise ValueError(f"scatter must be one of {_VALID_SCATTER}, got {scatter!r}")
        SCATTER_MODE = scatter
    if fuse_w13 is not None:
        FUSE_W13 = bool(fuse_w13)


# --------------------------------------------------------------------------------------- utils
def _world_size(moe) -> int:
    """nano configs are single-rank; keep the ref's sharded semantics working if they are not."""
    return moe.n_routed_experts // moe.n_local_experts


def _expert_weight_is_bf16(expert) -> bool:
    for lin in (expert.w1, expert.w2, expert.w3):
        if getattr(lin, "scale", None) is not None or lin.weight.dtype != torch.bfloat16:
            return False
    return True


def _grouped_mm_ready(moe, x: torch.Tensor) -> bool:
    """True iff the grouped_mm path is safe *and* available for this module/input."""
    if FORCE_LOOP or _GROUPED_MM is None:
        return False
    if _static_sizes(moe) is not None:
        return False                        # static routing chooses the sync-free loop
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if moe.n_local_experts != moe.n_routed_experts or moe.experts_start_idx != 0:
        return False                        # grouped path assumes all experts live on this rank
    expert = moe.experts[moe.experts_start_idx]
    if expert is None or not _expert_weight_is_bf16(expert):
        return False
    # grouped_mm requires 16-byte aligned last-dim strides (8 bf16 elements)
    return x.size(-1) % 8 == 0 and expert.w1.weight.shape[0] % 8 == 0


def path_for(moe, x: torch.Tensor) -> str:
    """Which implementation `fast_moe_forward` would pick for (moe, x): 'grouped_mm'|'loop'."""
    return "grouped_mm" if _grouped_mm_ready(moe, x) else "loop"


_GMM_CAPTURE_STATUS: dict[int, tuple[bool, str]] = {}


def probe_grouped_mm_capture(device=None) -> tuple[bool, str]:
    """Capture a tiny `torch._grouped_mm` in a CUDA graph; cached per device.  (ok, error).

    This is the ground truth for `graph_safe(..., verify=True)`: torch 2.12's grouped_mm
    host-copies `offs` internally, so this returns False ("... CPU<->CUDA copy ...") on both
    sm89 and sm120.  On a torch that fixed it, this returns True and the grouped path can be
    captured directly.  CPU / missing op returns (False, reason) without touching CUDA.
    """
    if _GROUPED_MM is None:
        return False, "torch._grouped_mm not available"
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key in _GMM_CAPTURE_STATUS:
        return _GMM_CAPTURE_STATUS[key]
    try:
        with torch.cuda.device(dev):
            a = torch.randn(64, 384, dtype=torch.bfloat16, device=dev)
            b = torch.randn(4, 384, 384, dtype=torch.bfloat16, device=dev)
            offs = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device=dev)
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    _GROUPED_MM(a, b, offs=offs)
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _GROUPED_MM(a, b, offs=offs)
            graph.replay()
            torch.cuda.synchronize(dev)
        status = (True, "")
    except Exception as exc:  # noqa: BLE001 - probe: any failure is the answer
        status = (False, f"{type(exc).__name__}: {exc}")
    _GMM_CAPTURE_STATUS[key] = status
    return status


def graph_safe(moe=None, x: torch.Tensor | None = None, verify: bool = False) -> bool:
    """Whether the selected path is intended to be CUDA-graph safe.

    - loop -> False unless `install_static_routing()` baked the group sizes first (then True)
    - grouped_mm -> True at the *design* level (no host syncs in our code), but torch 2.12's
      `_grouped_mm` host-copies `offs` internally and is not capturable.  Pass `verify=True`
      to run (and cache) the tiny capture probe and get the real answer for your torch.
    """
    if moe is not None and x is not None:
        if path_for(moe, x) != "grouped_mm":
            return _static_sizes(moe) is not None   # fused loop + baked sizes: sync-free
        return probe_grouped_mm_capture(x.device)[0] if verify else True
    ok = (not FORCE_LOOP) and _GROUPED_MM is not None and torch.cuda.is_available()
    if ok and verify:
        ok = probe_grouped_mm_capture()[0]
    return ok


def _scatter_mode(device: torch.device) -> str:
    if SCATTER_MODE != "auto":
        return SCATTER_MODE
    return "index_add" if device.type == "cuda" else "permute"


_STATIC_ATTR = "_fast_moe_static_sizes"


def _static_sizes(moe):
    return getattr(moe, _STATIC_ATTR, None)


def record_group_sizes(moe, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> list[int]:
    """One eager gate call -> per-expert token counts (host list) for static routing.

    This is the *only* host sync in the static-routing workflow and it happens before capture.
    Call it with the exact input that will be replayed (routing must stay unchanged).
    """
    xf = x.view(-1, moe.dim)
    mask = None if image_mask is None else image_mask.flatten()
    with torch.no_grad():
        _, indices = moe.gate(xf, mask)
    counts = _count_experts(indices.reshape(-1), moe.n_routed_experts).tolist()
    if sum(counts) != indices.numel():
        raise RuntimeError(f"group sizes {sum(counts)} != rows {indices.numel()}")
    return counts


def install_static_routing(moe, sizes) -> tuple[int, ...]:
    """Bake per-expert token counts into `moe`; `fast_moe_forward` then uses the fused loop
    (no `.tolist()`, no dynamic shapes) even on CUDA.  Returns the validated tuple.

    World-size 1 only (nano configs): with expert sharding the sorted rows of the local
    experts do not start at position 0, so the host-side sizes are not enough to slice them.
    """
    if moe.n_local_experts != moe.n_routed_experts:
        raise ValueError("static routing supports world_size == 1 only")
    sizes = tuple(int(c) for c in sizes)
    if len(sizes) != moe.n_local_experts:
        raise ValueError(f"expected {moe.n_local_experts} group sizes, got {len(sizes)}")
    if any(c < 0 for c in sizes):
        raise ValueError(f"negative group size in {sizes}")
    setattr(moe, _STATIC_ATTR, sizes)
    return sizes


def uninstall_static_routing(moe):
    """Remove baked sizes; returns them (or None).  Restores the auto path selection."""
    sizes = _static_sizes(moe)
    if sizes is not None:
        delattr(moe, _STATIC_ATTR)
    return sizes


def _count_experts(flat: torch.Tensor, n_experts: int) -> torch.Tensor:
    """bincount-free per-expert histogram: static [E] shape, integer adds, no host round-trip."""
    counts = torch.zeros(n_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    return counts


def _sort_tokens(indices: torch.Tensor, n_experts: int):
    """Stable expert-major ordering of the (token, topk-slot) rows, without bincount.

    Returns (order [M], tok [M], counts [E]).  `order[r]` is the flat row that goes to sorted
    position r; `tok[r] = order[r] // k` is its token; `counts[e]` is the group size.
    All shapes are static: M = T*k, E = n_experts.
    """
    _, k = indices.shape
    flat = indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.div(order, k, rounding_mode="floor")
    counts = _count_experts(flat, n_experts)
    return order, tok, counts


def _scatter_topk(outs, tok, order, T: int, k: int, dim: int, mode: str | None = None):
    """Sum the k per-slot expert outputs back into fp32 [T, dim] (fixed shapes only).

    The fp32 accumulator is not cosmetic: the reference keeps `y` in fp32, adds the shared
    expert there and rounds to bf16 only once at the end (`y.type_as(x)`).  Rounding the
    routed sum to bf16 first and then adding the shared expert rounds twice, which costs
    ~4e-3 relative output error and ~1.3e-2 gradient error (measured).  Casting `outs` to
    fp32 here (one 12.6 MB pass at d12 shapes) reproduces the ref bit-for-bit on CPU.

    Three equivalent strategies (Lead benchmarks them on the GPU; CPU numbers are in
    tools/test_fast_moe.py):
      permute      inverse-permute then `view(T,k,dim).sum(1)` - deterministic
      index_add    atomic add into zeros - least traffic, non-deterministic order
      scatter_add  same as index_add via `scatter_add_`
    """
    mode = mode or _scatter_mode(outs.device)
    outs = outs.float()                     # no-op if outs is already fp32
    if mode == "index_add":
        y = torch.zeros((T, dim), dtype=torch.float32, device=outs.device)
        y.index_add_(0, tok, outs)
        return y
    if mode == "scatter_add":
        y = torch.zeros((T, dim), dtype=torch.float32, device=outs.device)
        y.scatter_add_(0, tok.unsqueeze(1).expand(-1, dim), outs)
        return y
    flat = torch.empty(T * k, dim, dtype=torch.float32, device=outs.device)
    flat.index_copy_(0, order, outs)        # flat[order[r]] = outs[r], exactly once
    return flat.view(T, k, dim).sum(1)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float,
            weights: torch.Tensor | None = None) -> torch.Tensor:
    """Reference Expert.forward: up clamped both sides, gate only from above; fp32 math."""
    if limit is not None and limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h                     # [N,1] * [N,inter], same order as the ref
    return h


def _can_fuse_w13(expert) -> bool:
    w1, w3 = getattr(expert, "w1", None), getattr(expert, "w3", None)
    if w1 is None or w3 is None:
        return False
    if getattr(w1, "scale", None) is not None or getattr(w3, "scale", None) is not None:
        return False                        # fp8/fp4 weights carry a scale; keep the ref path
    return w1.weight.dtype == w3.weight.dtype and w1.weight.dtype in (
        torch.float16, torch.bfloat16, torch.float32)


def _expert_ffn(expert, x: torch.Tensor, weights: torch.Tensor | None = None,
                fuse: bool | None = None) -> torch.Tensor:
    """One expert (or the shared expert).  With FUSE_W13, w1/w3 become one N=2*inter GEMM."""
    fuse = FUSE_W13 if fuse is None else fuse
    if not fuse or not _can_fuse_w13(expert):
        return expert(x, weights)
    dtype = x.dtype
    inter = expert.w1.weight.shape[0]
    w13 = torch.cat((expert.w1.weight, expert.w3.weight), dim=0)   # [2*inter, dim]
    gate, up = F.linear(x, w13).split(inter, dim=-1)
    h = _swiglu(gate.float(), up.float(), getattr(expert, "swiglu_limit", 0.0), weights)
    return F.linear(h.to(dtype), expert.w2.weight)


def _stacked_weights(moe):
    """[E, K, N] views of the local experts' weights for `torch._grouped_mm` (no copies except
    the stack/cat themselves).  Transposed strides are explicitly supported by grouped_mm."""
    experts = [moe.experts[i] for i in range(moe.experts_start_idx, moe.experts_end_idx)]
    w1 = torch.stack([e.w1.weight for e in experts])       # [E, inter, dim]
    w3 = torch.stack([e.w3.weight for e in experts])       # [E, inter, dim]
    w2 = torch.stack([e.w2.weight for e in experts])       # [E, dim, inter]
    if FUSE_W13:
        return torch.cat((w1, w3), dim=1).transpose(1, 2), w2.transpose(1, 2)
    return w1.transpose(1, 2), w3.transpose(1, 2), w2.transpose(1, 2)


def _swiglu_limit(moe) -> float:
    return getattr(moe.experts[moe.experts_start_idx], "swiglu_limit", 0.0)


# ----------------------------------------------------------------------------------- forward
def _moe_grouped(moe, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
    """Routed experts via `torch._grouped_mm`; offs computed on-device (no .tolist())."""
    T, k = indices.shape
    order, tok, counts = _sort_tokens(indices, moe.n_routed_experts)
    offs = torch.cumsum(counts, 0).to(torch.int32)          # per-group END offsets, int32
    xs = x.index_select(0, tok)                             # [T*k, dim], expert-major
    ws = weights.reshape(-1).index_select(0, order)         # [T*k] fp32
    inter = moe.experts[moe.experts_start_idx].w1.weight.shape[0]
    limit = _swiglu_limit(moe)

    if FUSE_W13:
        B13, B2 = _stacked_weights(moe)                     # [E, dim, 2*inter], [E, inter, dim]
        gu = _GROUPED_MM(xs, B13, offs=offs)
        gate, up = gu.split(inter, dim=1)
        h = _swiglu(gate.float(), up.float(), limit, ws[:, None])
        outs = _GROUPED_MM(h.to(x.dtype), B2, offs=offs)
    else:
        B1, B3, B2 = _stacked_weights(moe)
        gate = _GROUPED_MM(xs, B1, offs=offs).float()
        up = _GROUPED_MM(xs, B3, offs=offs).float()
        h = _swiglu(gate, up, limit, ws[:, None])
        outs = _GROUPED_MM(h.to(x.dtype), B2, offs=offs)
    return _scatter_topk(outs, tok, order, T, k, moe.dim)


def _moe_loop(moe, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
    """cuBLAS fallback: stable sort + one contiguous GEMM per (expert, matrix), no where().

    Not graph-safe on CUDA *unless* `install_static_routing()` baked the group sizes: then the
    only host sync (`counts.tolist()`) is skipped and fwd+bwd are capture-safe.
    """
    T, k = indices.shape
    order, tok, counts = _sort_tokens(indices, moe.n_routed_experts)
    static = _static_sizes(moe)
    if static is not None:
        counts_list = list(static)          # baked before capture: no host sync here
    else:
        counts_list = counts.tolist()
    xs = x.index_select(0, tok)
    ws = weights.reshape(-1).index_select(0, order)
    start, end = moe.experts_start_idx, moe.experts_end_idx
    pos = sum(counts_list[:start])
    local_start = pos
    outs = []
    for e in range(start, end):
        c = counts_list[e]
        if c:
            outs.append(_expert_ffn(moe.experts[e], xs[pos:pos + c], ws[pos:pos + c, None]))
        pos += c
    y_sorted = torch.cat(outs, 0) if outs else None
    if start == 0 and pos == T * k and y_sorted is not None:
        return _scatter_topk(y_sorted, tok, order, T, k, moe.dim)
    # sharded rank: rows of other ranks contribute zero here and are summed by all_reduce
    flat = torch.zeros(T * k, moe.dim, dtype=torch.float32, device=x.device)
    if y_sorted is not None:
        flat.index_copy_(0, order[local_start:pos], y_sorted.float())
    return flat.view(T, k, moe.dim).sum(1)


def fast_moe_forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for `_ref_patched_model.MoE.forward`.

    x: [..., dim]; image_mask: [..., ] bool or None (vision routing bias; all-False == None
    when `bias_vl` is present).  Returns `y.type_as(x).view(x.shape)`.
    """
    shape = x.size()
    xf = x.view(-1, self.dim)
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = self.gate(xf, mask)                  # ref Gate: fp32 scores, fp32 weights
    if _grouped_mm_ready(self, xf):
        y = _moe_grouped(self, xf, weights, indices)
    else:
        y = _moe_loop(self, xf, weights, indices)
    shared = _expert_ffn(self.shared_experts, xf)
    if _world_size(self) > 1:
        y = y.float()
        torch.distributed.all_reduce(y)                     # ref accumulates in fp32
        y = y + shared.float()
    else:
        y = y + shared
    return y.type_as(x).view(shape)


# ------------------------------------------------------------------------------------ install
def install(M):
    """`M.MoE.forward = fast_moe_forward`; returns the previous forward for restoration.

    `M` is the patched-ref module (bench_mfu/GraphStep style: `M.MoE` is the class).  The
    assignment is direct, so calling install() twice is idempotent (the second call returns
    `fast_moe_forward` itself).
    """
    target = M.MoE
    previous = target.forward
    target.forward = fast_moe_forward
    return previous


def uninstall(M, previous) -> None:
    """Restore the forward returned by `install()` (e.g. the reference `MoE.forward`)."""
    M.MoE.forward = previous


# ------------------------------------------------------------------------------- graph audit
# Ops that read a device value back to the host, or that allocate a data-dependent shape.
# `_local_scalar_dense` covers .item()/float()/int()/bool(tensor); `.tolist()` on CUDA goes
# through either it or a device->host `to`; `nonzero`/`masked_select`/`unique` and `bincount`
# have data-dependent output shapes (bincount also reads the max index back to the host).
_SYNC_OPS = (
    "aten._local_scalar_dense", "aten.item", "aten.numpy",
    "aten.nonzero", "aten.masked_select", "aten.unique", "aten._unique2",
    "aten.unique_consecutive", "aten.bincount",
)


def _is_host_transfer(op: str, args, kwargs) -> bool:
    if not args or not isinstance(args[0], torch.Tensor):
        return False
    if op.startswith("aten.copy_"):
        src = args[1] if len(args) > 1 else None
        return args[0].device.type == "cpu" and isinstance(src, torch.Tensor) and src.is_cuda
    if not args[0].is_cuda:
        return False
    target = kwargs.get("device")
    if target is None and len(args) > 1:
        candidate = args[1]
        if isinstance(candidate, torch.Tensor):
            target = candidate.device
        elif isinstance(candidate, (torch.device, str)):
            target = candidate
        elif isinstance(candidate, int):
            return False                       # cuda:<int>, not a host transfer
    if target is None:
        return False
    target = torch.device(target) if isinstance(target, str) else target
    return target.type == "cpu"


class _NoSyncMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        for bad in _SYNC_OPS:
            if name.startswith(bad):
                raise RuntimeError(
                    f"CUDA-graph-unsafe op {name}: reads a device value on the host "
                    f"or has a data-dependent output shape")
        if name.startswith(("aten.to.", "aten._to_copy", "aten.copy_")) and _is_host_transfer(
                name, args, kwargs):
            raise RuntimeError(f"CUDA-graph-unsafe host transfer {name}")
        return func(*args, **kwargs)


_AUDIT_LOCK = threading.Lock()
_AUDIT_METHODS = ("tolist", "item", "cpu", "numpy")


@contextmanager
def assert_no_sync():
    """Audit a code block for graph-hostile operations (dispatch-level + Python methods).

    Raises RuntimeError on `Tensor.tolist/item/cpu/numpy`, on device->host copies, on
    `<op>.item()` inside C++ (`_local_scalar_dense`), and on dynamic-shape ops.  The Python
    method patches are process-wide for the duration of the block (not thread-safe, audit
    tool only) - always use it on the calling thread and outside `torch.cuda.graph()`.
    """
    if not _AUDIT_LOCK.acquire(blocking=False):
        raise RuntimeError("assert_no_sync() is already active on another thread")
    saved = {}

    def _blocked(method_name):
        def _raise(self, *a, **k):  # noqa: ANN001
            raise RuntimeError(
                f"CUDA-graph-unsafe host sync: Tensor.{method_name}() was called")
        return _raise

    try:
        for method_name in _AUDIT_METHODS:
            saved[method_name] = getattr(torch.Tensor, method_name)
            setattr(torch.Tensor, method_name, _blocked(method_name))
        with _NoSyncMode():
            yield
    finally:
        for method_name, fn in saved.items():
            setattr(torch.Tensor, method_name, fn)
        _AUDIT_LOCK.release()


def audit_no_sync(moe, x: torch.Tensor, image_mask: torch.Tensor | None = None,
                  backward: bool = True) -> str:
    """Run fwd (+bwd) under `assert_no_sync()`; returns the path used.  Mutates `.grad`."""
    path = path_for(moe, x)
    with assert_no_sync():
        y = fast_moe_forward(moe, x, image_mask)
        if backward:
            if not x.requires_grad:
                raise ValueError("audit_no_sync(backward=True) needs x.requires_grad")
            y.float().square().mean().backward()
    return path
