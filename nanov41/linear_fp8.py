"""Body linear layers in FP8 — O1 of ``docs/official-kernel-alignment.md`` (design note).

## What this is / why

The official DeepSeek-V4.1-Flash **training** recipe runs its linear layers in FP8 (report §2.4:
"linear 层 FP8"; the released inference kernels ``ref/kernel.py::act_quant`` +
``fp8_gemm`` are the block-scaled reference: ``block_size=128``, fp32 scales, optional
power-of-2 / UE8M0 scales).  Our training body is still bf16 — the head already runs fp8
(``fp8_head_v2`` / ``head_fwdgrad``).  This module adds the body path, **env-gated and
default-OFF** so the unset path is byte-identical (``install()`` is then a no-op and
``M.linear`` is never touched).

Offline census (aligned d12, b2×seq8192 = 16,384 tokens/replay; fwd FLOPs/replay):

    shape (M,K,N)      count  fwd GFLOP   who
    16384,1024,512       12     206.2     attention wo_b
    16384,768,2560        2     128.9     engram wkv (Linear 768→2560)
    16384,512,256        24     103.1     shared expert w1/w3
    16384,256,512        12      51.5     shared expert w2
    16384,512,128        12      25.8     attention wq_a
    16384,128,512        12      25.8     attention wq_b
    16384,512,64         18      19.3     attention wkv (12) + compressor ratio-1 (6)
     16384,128,64          2       0.3     indexer wq_b
    ----------------------------------------------
    body linear fwd            ~561 GFLOP  (≈ 1.7 TFLOP fwd+dgrad+wgrad / replay)

At the R77 production rate the whole ``gemm_other`` bucket is 6.78 ms/8k = 13.6 ms/replay, which
matches 1.7 TFLOP at ~125-150 TF/s — i.e. the census above *is* the 13.6 ms.  An effective 2×
on fwd+dgrad+wgrad would be −6.8 ms/replay ≈ −218 ms/step (2786 ms replay/step) ≈ **+7.8 %**.
That is the +6~8 % hypothesis of O1; the shape microbench decides per shape (R50/R51 prior:
nano-shape GEMMs are launch/tile bound and do NOT win; large-M shapes are where fp8 pays).

## Production model (why the weight cache)

One optimizer step = 32 CUDA-graph replays of the *same* weights (b2/seq8192/accum32).  Weight
fp8 quantisation is therefore done **once per optimizer step**, in eager code **outside the
captured graph**, into **static preallocated buffers** refilled in place with ``copy_`` (data
pointers never change, so the graph keeps reading them).  This is exactly the proven head-weight
cache pattern (``nanov41/fp8_head_v2.py::_WeightQuantCache`` + the
``tools/p2_train.py::_hc_maybe_refresh`` hook after ``bundle.step()``).  Activations are
quantised per call (they change every replay) — but see ``_QSHARE``: the two calls that share
one input tensor object (``wq_a``/``wkv``, ``w1``/``w3``) reuse one quantisation.

Layouts cached per weight (built eagerly, pointers stable):

* fwd   : ``wq  [N,K]`` fp8 + ``ws [ceil(N/bs), K/bs]`` scales — the official ``fp8_gemm``
          B layout (one scale per (bs,bs) block, applied to a whole bs-row block of N);
* dgrad : ``wtq [K,N]`` fp8 + ``wts [ceil(K/bs), N/bs]`` scales = transpose of ``(wq, ws)``.
          fp8 values and block amax are transpose-invariant, so ``wq.t().contiguous()`` +
          ``ws.t().contiguous()`` is an exact, cheap (eager, out-of-graph) transpose.

## Backends (both measured by ``tools/bench_linear_fp8.py``)

* ``tl`` — the **official** ``ref/kernel.py`` TileLang kernels (``act_quant`` block-128 +
  block-scaled ``fp8_gemm``), loaded by path (the training process' ``import kernel`` is a torch
  stub, see ``patched_env``).  fwd + dgrad; wgrad stays bf16 (the official kernel is A@Bᵀ only).
  This is the **alignment reference** (exact official quantisation semantics).
* ``sm`` — ``torch._scaled_mm`` (per-row activation scale / per-tensor weight scale — the recipe
  the fp8 head already uses, ~75 % of fp8 peak for the head).  fwd + dgrad + optional fp8 wgrad.
  Documented precision deviation: per-tensor weight scale instead of per-128-block.

Numerics: one e4m3 GEMM is ~3-4 % max-rel (``fp8_head_v2`` docstring); a chain of them is
"bf16-grade" but **not** bitwise bf16 — G1 records max_rel + a 20-step loss curve, never a
bitwise parity claim.

## Env flags (all default OFF / unchanged)

    NANOV41_LINEAR_FP8=1            master switch; unset/0 => install() no-op, unset path identical
    NANOV41_LINEAR_FP8_BACKEND      tl (default) | sm
    NANOV41_LINEAR_FP8_BLOCK        128 (default) | 32       (tl block size; also the weight quant block)
    NANOV41_LINEAR_FP8_ROUND_SCALE  0 (default) | 1          (power-of-2 / UE8M0 scales)
    NANOV41_LINEAR_FP8_DGRAD        1 (default) | 0          (fp8 dgrad; else bf16 cuBLAS)
    NANOV41_LINEAR_FP8_WGRAD        0 (default) | 1          (fp8 wgrad; sm backend only)
    NANOV41_LINEAR_FP8_PER_ROW_A    1 (default) | 0          (sm: per-row vs per-tensor A scale)
    NANOV41_LINEAR_FP8_QSHARE       1 (default) | 0          (reuse A quant between calls sharing x)
    NANOV41_LINEAR_FP8_SCOPE        comma list of parameter-name substrings (default ``all``);
                                    e.g. ``wo_b,experts,engram`` to roll out biggest shapes only
    NANOV41_LINEAR_FP8_MIN_FLOPS    skip weights whose per-token fwd GEMM work 2*N*K is below this
                                    (default 0; e.g. 1e6 drops indexer/weights_proj-sized GEMMs)

## Wire plan (who calls what; Lead owns the training/loop files)

1. ``tools/bench_graph_train.py::build()`` (after ``moe_stacked``/``fast_proj_fuse``, before
   ``return model, cfg`` — file currently ends that block at ``return model, cfg``):
       import nanov41.linear_fp8 as linear_fp8
       print(f"[bench] linear_fp8: {linear_fp8.install(M, model)}", flush=True)
   ``install`` is a no-op unless ``NANOV41_LINEAR_FP8=1``; it registers every eligible
   ``Linear``/``ColumnParallelLinear``/``RowParallelLinear`` parameter and patches
   ``M.linear`` (the module-global dispatcher used by ``Linear.forward`` in
   ``nanov41/_ref_patched_model.py:256/270/285``).  First fp8 forward (eager warm-up, before
   capture) allocates the static buffers, exactly like the head WCACHE.
2. ``tools/p2_train.py`` — next to the existing head-cache hook:
       `_hc_maybe_refresh()` is defined at line ~374 and called after the optimizer step at
       line ~720 (captured branch) and line ~912 (eager branch).  Add one line right after each:
           _lfp8_maybe_refresh()
       with a tiny helper next to ``_hc_maybe_refresh``:
           def _lfp8_maybe_refresh():
               try:
                   import nanov41.linear_fp8 as _lfp8
                   _lfp8.maybe_refresh_linear_weight_cache()
               except Exception as _exc:  # noqa: BLE001
                   print(f"[linear-fp8] refresh failed: {type(_exc).__name__}: {_exc}", flush=True)
       (``maybe_refresh_linear_weight_cache`` is a cheap no-op when the flag is off.)
   No other training-file edit is needed: all fwd linear calls live in the reference model and
   go through ``M.linear``.  Calls that must NOT be converted are already excluded by identity/
   dtype (``Gate.forward`` passes ``weight.float()``; ``wo_a`` is an ``einsum`` on the raw
   weight, not a ``Linear``; mHC/head use ``F.linear`` directly).
3. Gates: G1 20-step loss + max_rel (bench), G2 ``tools/r77_profile.sh`` in-graph replay delta,
   G3 ``tools/r77_final_ab.sh`` 50 steps.  Roll out with ``NANOV41_LINEAR_FP8_SCOPE`` on the
   shapes the bench marks positive, biggest first.

## API

    enabled()                              -> master flag
    install(M, model, verbose=True)        -> patch M.linear + register weights (no-op if off)
    maybe_refresh_linear_weight_cache()    -> per-optimizer-step refill (no-op if off)
    refresh_linear_weight_cache()          -> unconditional refill of ready slots
    register_weight(name, weight)          -> slot key (bench/tests; quantises immediately)
    linear_fp8(x, weight)                  -> fwd only (no autograd)
    linear_fp8_dgrad(dy, weight)           -> dX only (no autograd)
    linear_fp8_wgrad(dy, x, weight)        -> dW only (no autograd; sm backend)
    quant_act_blockwise / quant_weight_blockwise / quant_act_per_row / quant_act_per_tensor /
    quant_weight_per_tensor                -> the reusable quantisation primitives
    wcache_state()                         -> diagnostics dict
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
E4M3_MAX = 448.0

_TRUE = ("1", "true", "yes", "on")

# module globals resolved lazily (keeps `import nanov41.linear_fp8` cheap and Mac-safe)
_BACKEND: str | None = None            # "tl" | "sm" | "bf16"
_REFK = None                           # real ref/kernel.py module, None=unavailable, False=tried
_CASTK = None                          # fused cast kernels (tl_kernels/cast_fp8_tl.py) or None
_CASTK_TRIED = False
_GEMM_KERNELS: dict = {}
_INSTALLED = False
_ORIG_LINEAR = None
_INSTALL_STATE: dict = {}
_WARNED: set[str] = set()


# --------------------------------------------------------------------------- flags
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def enabled() -> bool:
    """Master switch ``NANOV41_LINEAR_FP8`` (default OFF)."""
    return _flag("NANOV41_LINEAR_FP8", "0")


def block_size() -> int:
    try:
        bs = int(os.environ.get("NANOV41_LINEAR_FP8_BLOCK", "128"))
    except ValueError:
        bs = 128
    return bs if bs in (32, 128) else 128


def round_scale() -> bool:
    return _flag("NANOV41_LINEAR_FP8_ROUND_SCALE", "0")


def dgrad_enabled() -> bool:
    return _flag("NANOV41_LINEAR_FP8_DGRAD", "1")


def wgrad_enabled() -> bool:
    return _flag("NANOV41_LINEAR_FP8_WGRAD", "0")


def per_row_a() -> bool:
    """sm backend: per-row activation scale (default) vs one per-tensor scalar."""
    return _flag("NANOV41_LINEAR_FP8_PER_ROW_A", "1")


def qshare_enabled() -> bool:
    return _flag("NANOV41_LINEAR_FP8_QSHARE", "1")


def scope() -> tuple:
    raw = os.environ.get("NANOV41_LINEAR_FP8_SCOPE", "all").strip()
    items = tuple(s.strip() for s in raw.split(",") if s.strip())
    return items or ("all",)


def min_flops() -> float:
    try:
        return float(os.environ.get("NANOV41_LINEAR_FP8_MIN_FLOPS", "0") or 0)
    except ValueError:
        return 0.0


def set_backend_override(name: str | None) -> None:
    """Bench/tests only: force ``tl`` / ``sm`` / ``bf16`` (None clears)."""
    global _BACKEND, _FORCE
    _FORCE = name
    _BACKEND = None


_FORCE: str | None = None


def backend(resolve: bool = True) -> str:
    """Resolved backend: ``tl`` (official kernels) | ``sm`` (torch._scaled_mm) | ``bf16`` (off)."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    name = _FORCE or os.environ.get("NANOV41_LINEAR_FP8_BACKEND", "tl").strip().lower()
    if name not in ("tl", "sm"):
        name = "tl"
    if not torch.cuda.is_available():
        _BACKEND = "bf16"
        return _BACKEND
    if name == "tl" and _ref_kernel() is None:
        _warn_once("tl", "ref/kernel.py unavailable (tilelang?) -> falling back to backend=sm")
        name = "sm"
    if name == "sm" and not hasattr(torch, "_scaled_mm"):
        _warn_once("sm", "torch._scaled_mm unavailable -> linear_fp8 disabled (bf16 path)")
        name = "bf16"
    if resolve:
        _BACKEND = name
    return name


def _warn_once(tag: str, msg: str) -> None:
    if tag not in _WARNED:
        _WARNED.add(tag)
        print(f"[linear-fp8] WARNING {msg}", flush=True)


# --------------------------------------------------------------------------- kernel loading
def _load_module_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ref_kernel():
    """The real ``ref/kernel.py`` (official TileLang kernels), loaded by path.

    ``patched_env.install()`` installs a no-op torch stub under the name ``kernel``, so the
    training process must NOT ``import kernel`` for the real thing — we load the file directly.
    """
    global _REFK
    if _REFK is not None:
        return _REFK or None
    try:
        _REFK = _load_module_by_path("_nanov41_ref_kernel", ROOT / "ref" / "kernel.py")
        if _REFK is not None:
            _REFK.fp8_gemm_kernel  # noqa: B018 - attribute probe
    except Exception as exc:  # noqa: BLE001 - tilelang missing / import error
        _REFK = False
        _warn_once("refk", f"could not load ref/kernel.py ({type(exc).__name__}: {str(exc)[:120]})")
    return _REFK or None


def _cast_kernels():
    """Optional fused bf16->fp8 cast kernels from ``tl_kernels/cast_fp8_tl.py`` (sm backend)."""
    global _CASTK, _CASTK_TRIED
    if _CASTK_TRIED:
        return _CASTK
    _CASTK_TRIED = True
    try:
        _CASTK = _load_module_by_path("_nanov41_cast_fp8_tl", ROOT / "tl_kernels" / "cast_fp8_tl.py")
    except Exception:  # noqa: BLE001
        _CASTK = None
    return _CASTK


# --------------------------------------------------------------------------- quantisation primitives
def _round_scale_pow2(amax: torch.Tensor) -> torch.Tensor:
    """``2**ceil(log2(amax/448))`` — matches ref ``fast_round_scale`` (UE8M0 / MXFP scales)."""
    return torch.pow(2.0, torch.ceil(torch.log2((amax / E4M3_MAX).clamp_min(1e-30)))).clamp_min(1e-30)


def _to_scale_dtype(s: torch.Tensor, rs: bool) -> torch.Tensor:
    return s.to(torch.float8_e8m0fnu) if rs else s


def quant_act_blockwise(x: torch.Tensor, block: int = 128, rs: bool = False,
                        use_kernel: bool = True):
    """Block-wise activation FP8 quantisation, official ``ref::act_quant`` semantics.

    x: [..., K] bf16/fp32 -> (xq [..., K] e4m3, scales [..., K // block]) with
    ``scale = amax(block) / 448`` per row and per ``block``-wide group of K (fp32 scales by
    default, power-of-2 when ``rs``).  Uses the official TileLang kernel when available
    (single fused pass), else a torch fallback with identical math.
    """
    K = x.size(-1)
    assert K % block == 0, f"quant_act_blockwise: K={K} not divisible by block={block}"
    x2 = x.reshape(-1, K)
    if use_kernel:
        refk = _ref_kernel()
        if refk is not None:
            scale_fmt = "ue8m0" if rs else None
            scale_dtype = torch.float8_e8m0fnu if rs else torch.float32
            xq, s = refk.act_quant(x2.contiguous(), block, scale_fmt, scale_dtype)
            return xq.reshape(*x.shape), s.reshape(*x.shape[:-1], K // block)
    g = x2.reshape(-1, K // block, block).float()
    amax = g.abs().amax(dim=-1).clamp_min(1e-4)
    s = _round_scale_pow2(amax) if rs else amax / E4M3_MAX
    q = (g / s.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), _to_scale_dtype(s, rs).reshape(*x.shape[:-1], K // block)


def quant_weight_blockwise(w: torch.Tensor, block: int = 128, rs: bool = False):
    """Weight FP8 quantisation in the official ``fp8_gemm`` B-scale layout.

    w: [N, K] bf16/fp32 -> (wq [N, K] e4m3, ws [ceil(N/block), K//block]).
    One scale per (block, block) block of W = exactly what ``ref::fp8_gemm`` indexes as
    ``scales_b[(bx*block_N + j) // group_size, k]`` (i.e. shared by a whole block-row of N).
    """
    N, K = w.shape
    assert K % block == 0, f"quant_weight_blockwise: K={K} not divisible by block={block}"
    nb = (N + block - 1) // block
    if N % block:
        w = torch.nn.functional.pad(w, (0, 0, 0, nb * block - N))
    wb = w.detach().reshape(nb, block, K // block, block).float()
    amax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-4)          # [nb, K//block]
    s = _round_scale_pow2(amax) if rs else amax / E4M3_MAX
    q = (wb / s[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX)
    q = q.reshape(nb * block, K)[:N]
    return q.to(torch.float8_e4m3fn), _to_scale_dtype(s, rs)


def quant_act_per_row(x: torch.Tensor):
    """Per-row activation quantisation (sm backend): -> (xq [.., K] e4m3, scale [.., 1] fp32)."""
    x2 = x.reshape(-1, x.size(-1))
    amax = x2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12).float()
    s = amax / E4M3_MAX
    cast = _cast_kernels()
    if cast is not None and x2.size(0) % 32 == 0 and x2.size(1) % 128 == 0:
        xq = cast.cast_fp8_row(x2.contiguous(), s.squeeze(-1))
    else:
        xq = (x2.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.reshape(*x.shape), s.reshape(*x.shape[:-1], 1)


def quant_act_per_tensor(x: torch.Tensor):
    """Per-tensor activation quantisation (sm backend wgrad): -> (xq, scale [1])."""
    x2 = x.reshape(-1, x.size(-1))
    s = x2.abs().amax().clamp_min(1e-12).float().reshape(1) / E4M3_MAX
    cast = _cast_kernels()
    if cast is not None and x2.size(0) % 32 == 0 and x2.size(1) % 128 == 0:
        xq = cast.cast_fp8(x2.contiguous(), s)
    else:
        xq = (x2.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.reshape(*x.shape), s


def quant_weight_per_tensor(w: torch.Tensor):
    """Per-tensor weight quantisation (sm backend): -> (wq [N,K] e4m3, scale [1] fp32)."""
    s = w.detach().abs().amax().clamp_min(1e-12).float().reshape(1) / E4M3_MAX
    wq = (w.detach().float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return wq, s


# --------------------------------------------------------------------------- official (tl) GEMM
def _tl_scale_dtype(rs: bool):
    return torch.float8_e8m0fnu if rs else torch.float32


def tl_fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
                block: int = 128, rs: bool = False, out_dtype=torch.bfloat16):
    """``C[M,N] = A[M,K] @ B[N,K]^T`` with per-block scales — thin wrapper over the official
    ``ref::fp8_gemm`` kernel with an explicit bf16 output (``ref::fp8_gemm`` allocates its output
    in ``torch.get_default_dtype()``, which is fp32 in a training process)."""
    refk = _ref_kernel()
    assert refk is not None, "tl backend requested but ref/kernel.py is unavailable"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block == 0, f"tl_fp8_gemm: K={K} not divisible by block={block}"
    assert N % block == 0, f"tl_fp8_gemm: N={N} not divisible by block={block} (official tile)"
    key = (N, K, block, rs)
    kern = _GEMM_KERNELS.get(key)
    if kern is None:
        tl_dtype = refk.FE8M0 if rs else refk.FP32
        kern = refk.fp8_gemm_kernel(N, K, block_size=block, scale_dtype=tl_dtype)
        _GEMM_KERNELS[key] = kern
    c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    kern(a.reshape(M, K), b, c, a_s.reshape(M, -1), b_s)
    return c


# --------------------------------------------------------------------------- activation quant share
class _QuantShare:
    """Reuse one activation quantisation for call sites that share the same input tensor.

    In the reference model ``Attention.forward`` calls ``wq_a(x)`` and ``wkv(x)`` on the *same*
    ``x``, and the shared expert calls ``w1(x)``/``w3(x)`` on the same tensor — quantising twice
    would be pure repeated bandwidth (~0.16 ms/replay ≈ +1.8 % of the step if left in).  Keyed by
    tensor identity + in-place version counter; the strong reference held here prevents id reuse
    while the entry lives.  Only the Python dispatch is cached: a captured CUDA graph records the
    single fused quant kernel and both GEMMs consume its buffer.
    """

    __slots__ = ("_d", "_max")

    def __init__(self, max_entries: int = 4):
        self._d: dict = {}
        self._max = max_entries

    @staticmethod
    def _key(x: torch.Tensor, backend: str):
        # data_ptr + shape + in-place version; the strong reference stored below keeps the
        # storage alive, so a data_ptr cannot be recycled while the entry lives.
        return (x.data_ptr(), tuple(x.shape), backend)

    def get(self, x: torch.Tensor, backend: str):
        if not qshare_enabled():
            return None
        k = self._key(x, backend)
        ent = self._d.get(k)
        if ent is not None and ent[0] is x and ent[1] == x._version:
            self._d.pop(k, None)          # LRU touch
            self._d[k] = ent
            return ent[3], ent[4]
        return None

    def put(self, x: torch.Tensor, backend: str, xq: torch.Tensor, s: torch.Tensor) -> None:
        if not qshare_enabled():
            return
        self._d[self._key(x, backend)] = (x, x._version, backend, xq, s)
        while len(self._d) > self._max:
            self._d.pop(next(iter(self._d)))

    def clear(self) -> None:
        self._d.clear()


_QSHARE = _QuantShare()


# --------------------------------------------------------------------------- weight cache
class _WeightSlot:
    """Static fp8 copies of one 2-D weight + their scales (pointers never change)."""

    __slots__ = ("key", "name", "shape", "weight", "backend", "block", "rs", "ready",
                 "refreshes", "wq", "ws", "wtq", "wts", "sw", "swt", "nbytes")

    def __init__(self, key: int, name: str, weight: torch.Tensor, backend: str,
                 block: int, rs: bool):
        self.key = key
        self.name = name
        self.shape = tuple(weight.shape)
        self.weight = weight
        self.backend = backend
        self.block = block
        self.rs = rs
        self.ready = False
        self.refreshes = 0
        self.wq = None
        self.ws = None
        self.wtq = None
        self.wts = None
        self.sw = None       # sm: per-tensor W scale as [1, N] (colwise B for the fwd GEMM)
        self.swt = None      # sm: same scalar as [1, K] (colwise B for the dgrad GEMM)
        self.nbytes = 0

    # -- quantise current weight values into the static buffers (eager, out of graph) --------
    def refresh(self, weight: torch.Tensor | None = None) -> None:
        w = weight if weight is not None else self.weight
        if w is None:
            return
        self.weight = w
        with torch.no_grad():
            if self.backend == "tl":
                wq, ws = quant_weight_blockwise(w, self.block, self.rs)
                wtq = wq.t().contiguous()
                wts = ws.t().contiguous()
                if not self.ready:
                    self.wq = torch.empty_like(wq)
                    self.ws = torch.empty_like(ws)
                    self.wtq = torch.empty_like(wtq)
                    self.wts = torch.empty_like(wts)
                self.wq.copy_(wq)
                self.ws.copy_(ws)
                self.wtq.copy_(wtq)
                self.wts.copy_(wts)
                self.nbytes = (wq.numel() + wtq.numel() + ws.numel() * ws.element_size()
                               + wts.numel() * wts.element_size())
            else:
                wq, sw = quant_weight_per_tensor(w)
                wtq = wq.t().contiguous()
                n_out, k_in = wq.shape
                swf = sw.reshape(1).expand(1, n_out).contiguous()   # [1,N]
                swt = sw.reshape(1).expand(1, k_in).contiguous()    # [1,K]
                if not self.ready:
                    self.wq = torch.empty_like(wq)
                    self.wtq = torch.empty_like(wtq)
                    self.sw = torch.empty_like(swf)
                    self.swt = torch.empty_like(swt)
                self.wq.copy_(wq)
                self.wtq.copy_(wtq)
                self.sw.copy_(swf)
                self.swt.copy_(swt)
                self.nbytes = (wq.numel() + wtq.numel()
                               + swf.numel() * swf.element_size() + swt.numel() * swt.element_size())
        self.ready = True
        self.refreshes += 1

    def state(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "backend": self.backend,
                "block": self.block, "round_scale": self.rs, "ready": self.ready,
                "refreshes": self.refreshes, "bytes": self.nbytes}


class _WeightCache:
    def __init__(self):
        self.slots: dict = {}
        self.by_id: dict = {}
        self._next = 0

    def register(self, name: str, weight: torch.Tensor) -> _WeightSlot:
        key = self.by_id.get(id(weight))
        if key is not None:
            return self.slots[key]
        k = self._next
        self._next += 1
        slot = _WeightSlot(k, name, weight, backend(), block_size(), round_scale())
        self.slots[k] = slot
        self.by_id[id(weight)] = k
        return slot

    def get(self, weight: torch.Tensor):
        k = self.by_id.get(id(weight))
        return self.slots.get(k) if k is not None else None

    def refresh_all(self) -> int:
        n = 0
        for slot in self.slots.values():
            if slot.ready:
                slot.refresh(slot.weight)
                n += 1
        return n

    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.slots.values())


_CACHE = _WeightCache()


# --------------------------------------------------------------------------- low-level ops
def _check_input(x: torch.Tensor, slot: _WeightSlot) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.size(-1) == slot.shape[1]
            and x.numel() > 0)


def _act_quant(x: torch.Tensor, be: str):
    """Quantise activations for the active backend -> (xq, scales); x is [..., K]."""
    if be == "tl":
        return quant_act_blockwise(x, block_size(), round_scale())
    if per_row_a():
        return quant_act_per_row(x)
    return quant_act_per_tensor(x)


def _act_quant_shared(x: torch.Tensor, be: str):
    """``_act_quant`` with the identity-share cache (avoids duplicate quantisation of one input)."""
    hit = _QSHARE.get(x, be)
    if hit is not None:
        return hit
    xq, s = _act_quant(x, be)
    _QSHARE.put(x, be, xq, s)
    return xq, s


def _row_scale(s: torch.Tensor, m: int) -> torch.Tensor:
    """Normalise an A-scale to the [m, 1] rowwise form ``torch._scaled_mm`` expects."""
    if s.dim() == 2 and s.size(0) == m and s.size(1) == 1:
        return s
    return s.reshape(1).expand(m, 1).contiguous()


def _col_scale(s: torch.Tensor, n: int) -> torch.Tensor:
    """Normalise a B-scale to the [1, n] colwise form ``torch._scaled_mm`` expects."""
    if s.dim() == 2 and s.size(0) == 1 and s.size(1) == n:
        return s
    return s.reshape(1).expand(1, n).contiguous()


def _sm_scaled_mm(a8: torch.Tensor, sa: torch.Tensor, b_phys: torch.Tensor,
                  sb: torch.Tensor) -> torch.Tensor:
    """``torch._scaled_mm`` with explicitly shaped rowwise/colwise fp32 scales.

    ``b_phys`` is the physical [N_out, Kdim] layout; ``b_phys.t()`` is the column-major
    [Kdim, N_out] operand the op wants.  Scales are materialised only when a scalar/tensor form
    mismatch would otherwise be rejected (head-proven rowwise×colwise path).
    """
    m, n = a8.size(0), b_phys.size(0)
    return torch._scaled_mm(a8, b_phys.t(), scale_a=_row_scale(sa, m),
                            scale_b=_col_scale(sb, n), out_dtype=torch.bfloat16)


def _slot_forward(slot: _WeightSlot, x: torch.Tensor) -> torch.Tensor:
    """y = x @ W^T in fp8 (no autograd); x [..., K] bf16 -> [..., N] bf16."""
    if not slot.ready:
        slot.refresh(slot.weight)          # first forward: eager, before graph capture
    out_shape = (*x.shape[:-1], slot.shape[0])
    be = slot.backend
    xq, xs = _act_quant_shared(x, be)
    if xq.dim() > 2:
        xq = xq.reshape(-1, xq.size(-1))
    if xs.dim() > 2:
        xs = xs.reshape(-1, xs.size(-1))
    if be == "tl":
        y = tl_fp8_gemm(xq, xs, slot.wq, slot.ws, slot.block, slot.rs)
    else:
        y = _sm_scaled_mm(xq, xs, slot.wq, slot.sw)
    return y.reshape(out_shape)


def _slot_dgrad(slot: _WeightSlot, dy: torch.Tensor) -> torch.Tensor:
    """dX = dY @ W in fp8 (no autograd); dy [..., N] bf16 -> [..., K] bf16."""
    out_shape = (*dy.shape[:-1], slot.shape[1])
    dy2 = dy.reshape(-1, dy.size(-1)).contiguous()
    be = slot.backend
    dq, ds = _act_quant(dy2, be)
    if be == "tl":
        dx = tl_fp8_gemm(dq, ds, slot.wtq, slot.wts, slot.block, slot.rs)
    else:
        dx = _sm_scaled_mm(dq, ds, slot.wtq, slot.swt)
    return dx.reshape(out_shape)


def _slot_wgrad(slot: _WeightSlot, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dW = dY^T @ X in fp8 (no autograd; sm backend). dy [..., N], x [..., K] -> [N, K]."""
    assert slot.backend == "sm", "fp8 wgrad is only implemented for the sm backend"
    N, K = slot.shape
    dy2 = dy.reshape(-1, N).contiguous()
    x2 = x.reshape(-1, K).contiguous()
    dq, ds = quant_act_per_tensor(dy2)                 # per-tensor: dC^T rows are N
    xq, sx = quant_act_per_tensor(x2)
    dc8t = dq.t().contiguous()                         # [N, M] row-major operand
    xt8 = xq.t().contiguous()                          # [K, M] physical; .t() -> col-major [M, K]
    return _sm_scaled_mm(dc8t, ds, xt8, sx)


# --------------------------------------------------------------------------- autograd wrapper
class _LinearFP8Fn(torch.autograd.Function):
    """y = x @ W^T with the fp8 fwd and (optionally) fp8 dgrad/wgrad; W comes from the step cache.

    Grading policy mirrors the head: fwd stays fp8; the bwd GEMMs are fp8 when the bench shows
    they pay and bf16 cuBLAS otherwise (per-shape, env-gated).  ``wgrad`` bf16 is exactly the
    arithmetic ``F.linear`` does internally (cublas bf16 in / fp32 accumulate / bf16 out).
    """

    @staticmethod
    def forward(ctx, x, weight, slot_key):
        slot = _CACHE.slots[slot_key]
        y = _slot_forward(slot, x)
        ctx.slot_key = slot_key
        ctx.shape = tuple(x.shape)
        ctx.dgrad = dgrad_enabled() and slot.backend in ("tl", "sm")
        ctx.wgrad = wgrad_enabled() and slot.backend == "sm"
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        slot = _CACHE.slots[ctx.slot_key]
        N = slot.shape[0]
        dy2 = dy.reshape(-1, N).contiguous()
        if ctx.dgrad:
            dx = _slot_dgrad(slot, dy2)
        else:
            dx = torch.mm(dy2, slot.weight)            # cublas bf16, opB=T
        if ctx.wgrad:
            dw = _slot_wgrad(slot, dy2, x)
        else:
            dw = torch.mm(dy2.t(), x.reshape(-1, slot.shape[1]))
        return dx.reshape(ctx.shape), dw, None


# --------------------------------------------------------------------------- dispatch + install
def _linear_dispatch(x, weight, bias=None):
    """Drop-in for the reference model's module-global ``linear()`` (see ``install``)."""
    if _ORIG_LINEAR is None:                       # uninstalled/reset but the patch is still live
        return torch.nn.functional.linear(x, weight, bias)
    if bias is not None:
        return _ORIG_LINEAR(x, weight, bias)
    slot = _CACHE.get(weight)
    if slot is None or _BACKEND not in ("tl", "sm"):
        return _ORIG_LINEAR(x, weight, bias)
    if not _check_input(x, slot):
        return _ORIG_LINEAR(x, weight, bias)
    return _LinearFP8Fn.apply(x, weight, slot.key)


def _eligible(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim != 2 or p.dtype != torch.bfloat16 or not p.requires_grad:
        return False
    N, K = p.shape
    if _FORCE == "sm" or backend(resolve=False) == "sm":
        if K % 128 or N % 16:                          # _scaled_mm fp8: K multiple of 128 (safe)
            return False
    else:
        if K % block_size() or N % block_size():
            return False
    if min_flops() > 0 and 2.0 * N * K < min_flops():
        return False
    sc = scope()
    if sc != ("all",) and not any(s in name for s in sc):
        return False
    # never touch these by construction (kept for safety if a future refactor routes them here)
    for bad in ("head", "embed", "attn_sink", "hc_", "gate"):
        if bad in name:
            return False
    return True


def install(M=None, model=None, verbose: bool = True):
    """Patch ``M.linear`` with the fp8 dispatcher and register the model's eligible weights.

    No-op (returns ``{"enabled": False}``) unless ``NANOV41_LINEAR_FP8=1``; idempotent.  Must be
    called after the model (and any parameter-layout patch, e.g. ``moe_stacked``) is built and
    before the CUDA graph is captured.  Static fp8 weight buffers are allocated on the first fp8
    forward (eager warm-up) and refilled per optimizer step by
    :func:`maybe_refresh_linear_weight_cache`.
    """
    global _INSTALLED, _ORIG_LINEAR, _INSTALL_STATE
    if not enabled():
        return {"enabled": False}
    if _INSTALLED:
        return dict(_INSTALL_STATE)
    be = backend()                                     # resolves + may fall back
    if be == "bf16":
        _INSTALL_STATE = {"enabled": True, "backend": "bf16", "patched": False,
                          "note": "no fp8 backend available"}
        return dict(_INSTALL_STATE)
    if M is None:
        M = sys.modules.get("_ref_patched_model")
    if M is None:
        raise RuntimeError("linear_fp8.install: cannot find the patched reference model module "
                           "(pass M=import_patched())")
    if model is None:
        raise RuntimeError("linear_fp8.install: model= is required to register the trainable "
                           "weights (and pre-allocate their fp8 layouts)")
    names, nbytes, skipped = [], 0, 0
    for name, p in model.named_parameters():
        if not _eligible(name, p):
            skipped += 1
            continue
        slot = _CACHE.register(name, p)
        names.append(name)
        nbytes += slot.shape[0] * slot.shape[1]
    _ORIG_LINEAR = M.linear
    M.linear = _linear_dispatch
    _INSTALLED = True
    _INSTALL_STATE = {
        "enabled": True, "backend": be, "patched": True, "n_weights": len(names),
        "fp8_weight_mb": round(nbytes / 1e6, 2),
        "dgrad": dgrad_enabled(), "wgrad": wgrad_enabled(),
        "block": block_size(), "round_scale": round_scale(),
        "scope": list(scope()), "min_flops": min_flops(),
        "names": names,
    }
    if verbose:
        print(f"[linear-fp8] installed backend={be} weights={len(names)} "
              f"fp8={nbytes / 1e6:.2f}MB dgrad={dgrad_enabled()} wgrad={wgrad_enabled()} "
              f"scope={list(scope())}", flush=True)
    return dict(_INSTALL_STATE)


def uninstall(M=None) -> None:
    """Restore the original ``M.linear`` (tests / rollback)."""
    global _INSTALLED, _ORIG_LINEAR
    if M is None:
        M = sys.modules.get("_ref_patched_model")
    if M is not None and _ORIG_LINEAR is not None:
        M.linear = _ORIG_LINEAR
    _INSTALLED = False
    _ORIG_LINEAR = None
    _QSHARE.clear()


# --------------------------------------------------------------------------- refresh hooks
def refresh_linear_weight_cache(weight=None, verbose: bool = False) -> int:
    """Refill all ready fp8 weight buffers from the current weights.  Eager; outside the graph.

    ``weight`` (optional) refreshes only that tensor's slot.  Returns the number of slots refreshed.
    """
    if not enabled():
        return 0
    if weight is not None:
        slot = _CACHE.get(weight)
        if slot is None:
            return 0
        slot.refresh(weight)
        return 1
    n = _CACHE.refresh_all()
    if verbose:
        print(f"[linear-fp8] refreshed {n} weight slots", flush=True)
    return n


def maybe_refresh_linear_weight_cache(weight=None) -> bool:
    """Cheap no-op unless ``NANOV41_LINEAR_FP8=1`` — the one-line training-loop wiring."""
    if not enabled():
        return False
    if not _CACHE.slots:
        return False
    return refresh_linear_weight_cache(weight) > 0


def wcache_state() -> dict:
    return {
        "enabled": enabled(),
        "backend": _BACKEND,
        "installed": _INSTALLED,
        "n_slots": len(_CACHE.slots),
        "n_ready": sum(1 for s in _CACHE.slots.values() if s.ready),
        "fp8_bytes": _CACHE.nbytes(),
        "refreshes": sum(s.refreshes for s in _CACHE.slots.values()),
        "qshare_entries": len(_QSHARE._d),
        "slots": [s.state() for s in _CACHE.slots.values()],
    }


# --------------------------------------------------------------------------- public low-level API
def register_weight(name: str, weight: torch.Tensor):
    """Register (and immediately quantise) one weight into the step cache; returns its slot key."""
    slot = _CACHE.register(name, weight)      # keyed by the tensor object itself (identity lookup)
    slot.refresh(weight)
    return slot.key


def linear_fp8(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward only: ``y = x @ weight^T`` through the fp8 path (no autograd)."""
    slot = _CACHE.get(weight)
    if slot is None:
        raise RuntimeError("linear_fp8: weight not registered (call register_weight first)")
    return _slot_forward(slot, x)


def linear_fp8_dgrad(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Backward w.r.t. x only: ``dX = dY @ weight`` through the fp8 path (no autograd)."""
    slot = _CACHE.get(weight)
    if slot is None:
        raise RuntimeError("linear_fp8_dgrad: weight not registered")
    return _slot_dgrad(slot, dy)


def linear_fp8_wgrad(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Backward w.r.t. weight only: ``dW = dY^T @ x`` through the fp8 path (sm backend)."""
    slot = _CACHE.get(weight)
    if slot is None:
        raise RuntimeError("linear_fp8_wgrad: weight not registered")
    return _slot_wgrad(slot, dy, x)


def reset_for_tests() -> None:
    """Drop all registered weights / patch state (bench + unit tests); restores ``M.linear``."""
    global _CACHE, _INSTALLED, _ORIG_LINEAR, _BACKEND, _GEMM_KERNELS
    M = sys.modules.get("_ref_patched_model")
    if M is not None and _ORIG_LINEAR is not None and getattr(M, "linear", None) is _linear_dispatch:
        M.linear = _ORIG_LINEAR
    _QSHARE.clear()
    _CACHE = _WeightCache()
    _INSTALLED = False
    _ORIG_LINEAR = None
    _BACKEND = None
    _GEMM_KERNELS = {}


if __name__ == "__main__":  # tiny GPU smoke test: python -m nanov41.linear_fp8
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[linear-fp8] flags enabled={enabled()} backend={backend()} block={block_size()} "
          f"dgrad={dgrad_enabled()} wgrad={wgrad_enabled()} scope={scope()}")
    if torch.cuda.is_available():
        for (M_, K_, N_) in ((4096, 512, 128), (4096, 1024, 512), (4096, 768, 2560)):
            w = torch.randn(N_, K_, device=dev, dtype=torch.bfloat16) * 0.02
            x = torch.randn(M_, K_, device=dev, dtype=torch.bfloat16)
            dy = torch.randn(M_, N_, device=dev, dtype=torch.bfloat16)
            key = register_weight(f"smoke{M_}_{K_}_{N_}", w)
            y = linear_fp8(x, w)
            yref = x @ w.t()
            dx = linear_fp8_dgrad(dy, w)
            dxref = dy @ w
            rel = lambda a, b: ((a.float() - b.float()).abs().max()
                                / b.float().abs().max().clamp_min(1e-6)).item()
            print(f"  ({M_},{K_},{N_}) key={key} fwd max_rel_vs_scale={rel(y, yref):.3e} "
                  f"dgrad max_rel_vs_scale={rel(dx, dxref):.3e}", flush=True)
        print("[linear-fp8] state:", {k: v for k, v in wcache_state().items() if k != "slots"})
