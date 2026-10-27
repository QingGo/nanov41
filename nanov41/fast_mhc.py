"""Fast, autograd-safe mHC (Hyper-Connections) ops for the V4.1 reference `Block`.

Drop-in replacements for `Block.hc_mixes` / `Block.hc_pre` / `Block.hc_post` in
`_ref_patched_model.py`, plus `install(M)` / `set_mode(exact)`.

Why this module exists
----------------------
The reference ops are mathematically tiny but materialize full-size fp32 temporaries.  For
d12 at b=8/seq=1024 the hc residual is 25 MB bf16; per training step the blocks call
`hc_mixes` 24x, `hc_pre` 24x and `hc_post` 24x:

  * `hc_mixes`: `x.flatten(2).float()` (25 -> 50 MB) **and** `x.square()` (another 50 MB)
    before a [8192, 1536] x [1536, 24] GEMM.  ~280 MB of traffic per call.
  * `hc_pre`: an explicit `.float()` copy, then a broadcast fp32 product, then a reduction.
    ~258 MB per call.
  * `hc_post`: `comb[..., None] * residual[..., None, :]` builds a
    `[b, s, hc, hc, d]` fp32 tensor -- 8192*4*4*384*4 B = **201 MB for one intermediate**,
    and the whole op is ~762 MB of traffic.  x24 calls is ~18 GB, i.e. most of the ~20 ms
    per-step non-module forward time on a quiet 5090.

Replacement strategy (same signatures, same math, autograd untouched)
--------------------------------------------------------------------
  * `hc_pre`: no `.float()` copy.  In exact mode the promotion happens inside the multiply
    and the fp32 reduction (`sum(p32 * x_bf16, dim=2, dtype=fp32)`, bitwise identical to
    the ref); in fast mode it is one bmm `p @ x` with K=hc.
  * `hc_post`: one fused bmm over the concatenated `[residual; x]` with `W = [comb^T | post]`
    and K=hc+1.  The `[b,s,hc,hc,d]` tensor never exists.
    NB: the ref broadcast reduces over comb's **first** hc axis (dim=2 of the
    `[b,s,hc,hc,d]` product), i.e. it is
        y[..., j, :] = post[..., j] * x + sum_i comb[..., i, j] * residual[..., i, :]
    so the GEMM weight is `comb^T`, not `comb`.  (`Block.hc_post`'s docstring writes it as
    sum_h' comb[h,h'] * residual[h']; the code is the transpose of that.)
  * `hc_mixes`: one flattened view, one fp32 upcast reused by the norm and the GEMM, and
    the norm is a single `vector_norm` reduction -- no materialized fp32 `square()`.  Fast
    mode uses a bf16 GEMM (fp32 accumulate) plus a bf16 `vector_norm` (fp32 accumulator),
    then promotes `mixes` to fp32 for Sinkhorn.

Sinkhorn is deliberately **not** reimplemented.  `hc_mixes` calls the model module's
`hc_split_sinkhorn` (so patching `M.hc_split_sinkhorn = hc_split_sinkhorn_tl` is honoured)
and only decides the dtype of `mixes` (fp32, which is what the kernel expects).

Modes
-----
`set_mode(True)` (default, exact):
    fp32 intermediates, closest to the ref.  On fp32 activations this is numerically the
    ref (`hc_pre` bitwise; `hc_mixes`/`hc_post` differ only by fp32 reduction order).
`set_mode(False)` (fast):
    activation-dtype intermediates for bf16.  ~3x less traffic than exact mode for
    ~2-6e-3 relative error in the mHC outputs (measured in `tools/test_fast_mhc.py`); on
    fp32 activations it is the same code path as exact mode.

Estimated HBM traffic per forward call at d12 b=8/seq=1024 (residual 25 MB bf16):

    op         ref      exact    fast
    hc_mixes   279 MB   179 MB    53 MB
    hc_pre     258 MB   157 MB    32 MB
    hc_post    762 MB   348 MB   122 MB

Per step (24 calls each): ~31 GB (ref) -> ~16 GB (exact) -> ~5 GB (fast).  At the 5090's
effective ~1.2-1.5 TB/s for these elementwise-bound kernels that is the difference between
~20 ms and ~4-6 ms in the forward non-module budget.  The bmms are tiny (M=hc or hc+1,
N=384) and bandwidth-bound, so they should win; their actual timing -- especially the fp32
batched GEMM with K=5 plus the cat -- needs a 5090 measurement:
`python3 tools/test_fast_mhc.py --bench`.  CPU here only verifies correctness (on CPU the
fp32 bmm is correctness-only and slower than the broadcast).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

__all__ = [
    "hc_mixes",
    "hc_pre",
    "hc_post",
    "install",
    "uninstall",
    "restore",
    "set_mode",
    "get_mode",
    "set_sinkhorn",
]

_MISSING = object()          # class attribute did not exist before install
_EXACT = True                # conservative default: remove redundancy, keep fp32 semantics
_POST_BF16 = False           # 混合档 1：只有 hc_post 的 bmm 用 bf16
_CORE_TRITON = False         # mHC mixes 核心（norm+bf16 GEMM+rsqrt 乘）用 Triton 融合 kernel
_SPLIT_TRITON = False        # mHC split 段用 Triton 融合 kernel（sinkhorn 仍用官方 TileLang）
_POST_TRITON = False         # hc_post 用 Triton 融合 kernel（一个 kernel 出 out，一个出 4 个梯度）
_MIXES_BF16 = False          # 混合档 2：hc_post + hc_mixes 的投影都走 bf16（fp32 累加）
                             # op profile 里 exact hc_mixes 的 fp32 GEMM/gemv 在 5090 上占 ~3-8 ms
_SINKHORN = None             # explicit override (callable)
_SINKHORN_OWNER = None       # model module whose hc_split_sinkhorn is used (unless overridden)
_INSTALLED = None            # {"cls": Block, "prev": {...}, "owner": module|None}
_WARNED_TF32 = False


# --------------------------------------------------------------------------- mode / sinkhorn
def set_post_bf16(enabled: bool = True) -> bool:
    """混合档开关：hc_post 用 bf16 bmm（hc_mixes/hc_pre 仍按 exact）。"""
    global _POST_BF16
    _POST_BF16 = bool(enabled)
    return _POST_BF16


def get_post_bf16() -> bool:
    return _POST_BF16


def set_mixes_bf16(enabled: bool = True) -> bool:
    """混合档 2：hc_mixes 的投影用 bf16（norm 统计与 sieve 仍按原语义）。"""
    global _MIXES_BF16
    _MIXES_BF16 = bool(enabled)
    return _MIXES_BF16


def set_core_triton(enabled: bool = True) -> bool:
    """mHC mixes 的 norm+GEMM+rsqrt 段换 Triton 融合 kernel（fwd/bwd 都解析）。"""
    global _CORE_TRITON
    _CORE_TRITON = bool(enabled)
    return _CORE_TRITON


def set_split_triton(enabled: bool = True) -> bool:
    """mHC split（sigmoid/softmax/affine）换 Triton 融合 kernel；sinkhorn 仍走官方 TileLang。"""
    global _SPLIT_TRITON
    _SPLIT_TRITON = bool(enabled)
    return _SPLIT_TRITON


def set_post_triton(enabled: bool = True) -> bool:
    """hc_post 换 Triton 融合 kernel（5090 graph 步里 hc_post 占 14.8 ms，这一项能砍到 ~1.5 ms）。"""
    global _POST_TRITON
    _POST_TRITON = bool(enabled)
    return _POST_TRITON


def set_mixed_level(level: int) -> int:
    """0=exact  1=hc_post bf16  2=+hc_mixes bf16  3=全 bf16(nx)。返回生效的 level。"""
    global _EXACT, _POST_BF16, _MIXES_BF16
    level = int(level)
    _EXACT = level < 3
    _POST_BF16 = level >= 1
    _MIXES_BF16 = level >= 2
    return level


def set_mode(exact: bool) -> bool:
    """Select exact=True (fp32 intermediates) or exact=False (activation-dtype intermediates).

    Returns the previous mode.  Global (one training process uses one mode).
    """
    global _EXACT
    prev = _EXACT
    _EXACT = bool(exact)
    if _EXACT:
        _warn_if_tf32()
    return prev


def get_mode() -> bool:
    """True when exact (fp32) intermediates are enabled."""
    return _EXACT


def set_sinkhorn(fn=None) -> None:
    """Override the `hc_split_sinkhorn` callable used by `hc_mixes`.

    `fn=None` reverts to the model module's `hc_split_sinkhorn` (or `patched_env`'s when
    installed on a bare Block class).  The function must have the reference semantics:
    `fn(mixes, hc_scale, hc_base, hc_mult, iters, eps) -> (pre, post, comb)`.
    """
    global _SINKHORN
    _SINKHORN = fn


def _split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    if _SPLIT_TRITON:
        try:
            from . import triton_mhc_split as tms
            if tms.available():
                return tms.hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, hc, iters, eps)
        except Exception:  # noqa: BLE001
            pass
    fn = _SINKHORN
    if fn is None and _SINKHORN_OWNER is not None:
        fn = getattr(_SINKHORN_OWNER, "hc_split_sinkhorn", None)
    if fn is None:  # bare-class install, no model module to borrow from
        from .patched_env import hc_split_sinkhorn as fn
    return fn(mixes, hc_scale, hc_base, hc, iters, eps)


def _warn_if_tf32() -> None:
    """Exact `hc_post` uses an fp32 bmm; TF32 would silently break the fp32 promise."""
    global _WARNED_TF32
    if _WARNED_TF32 or not torch.cuda.is_available():
        return
    try:
        prec = torch.get_float32_matmul_precision()
    except Exception:  # pragma: no cover - very old torch
        return
    if prec != "highest":
        _WARNED_TF32 = True
        warnings.warn(
            "fast_mhc exact mode uses an fp32 matmul in hc_post, but torch float32 matmul "
            f"precision is {prec!r} (TF32). Call torch.set_float32_matmul_precision('highest') "
            "or use set_mode(False) if you accept the bf16 path.",
            RuntimeWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- the three ops
def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc].

    Same math as the ref: one normalization statistic over the flattened hc*d stream, one
    projection, then Sinkhorn.  Unlike the ref we do not materialize the fp32 square.
    """
    xf = x.flatten(2)
    n = xf.size(-1)
    if _CORE_TRITON and xf.dtype != torch.float32:
        try:
            from . import triton_mhc_core as tmc
            if tmc.available():
                mixes = tmc.hc_mixes_core(xf, hc_fn, self.norm_eps).float()
                return _split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult,
                                       self.hc_sinkhorn_iters, self.hc_eps)
        except Exception:  # noqa: BLE001
            pass
    if (_EXACT and not _MIXES_BF16) or xf.dtype == torch.float32:
        xf32 = xf if xf.dtype == torch.float32 else xf.float()
        # sum(x^2) without a [b,s,hc*d] fp32 square buffer; fp32 accumulator.
        nrm = torch.linalg.vector_norm(xf32, dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(nrm.square() / n + self.norm_eps)
        mixes = F.linear(xf32, hc_fn) * rsqrt
    else:
        # bf16/activation-dtype path: norm reduction still accumulates in fp32, GEMM uses
        # tensor cores with fp32 accumulate; `mixes * rsqrt` promotes the small bf16 GEMM
        # output to fp32 for Sinkhorn (one cast fewer than an explicit `.float()`).
        nrm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True).float()
        rsqrt = torch.rsqrt(nrm.square() / n + self.norm_eps)
        mixes = F.linear(xf, hc_fn.to(xf.dtype)) * rsqrt
    return _split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)


def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
    """Collapse the hc copies into one.  [b,s,hc,d] x [b,s,hc] -> [b,s,d].

    exact: `pre_mix` is promoted, x is *not* copied to fp32; `sum(p * x, dtype=fp32)` is
    bitwise identical to the ref's `sum(p * x.float())`.
    fast:  one bmm `pre_mix.bf16 @ x` (K=hc, fp32 accumulate, bf16 output).
    """
    if _EXACT or x.dtype == torch.float32:
        p = pre_mix if pre_mix.dtype == torch.float32 else pre_mix.float()
        y = torch.sum(p.unsqueeze(-1) * x, dim=2, dtype=torch.float32)
        return y.to(x.dtype)
    b, s, hc, d = x.shape
    p = pre_mix.to(x.dtype).reshape(b * s, 1, hc)
    y = torch.bmm(p, x.reshape(b * s, hc, d))
    return y.reshape(b, s, d)


def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    """Expand the sublayer output back to hc copies.  [b,s,d] x [b,s,hc,d] -> [b,s,hc,d].

    Fused as one bmm: with W = [comb^T | post] ([b,s,hc,hc+1]) and X = [residual; x]
    ([b,s,hc+1,d]) we get exactly the ref's
        y[..., j, :] = post[..., j] * x + sum_i comb[..., i, j] * residual[..., i, :]
    (see the module docstring for the transpose), without ever building the
    [b,s,hc,hc,d] fp32 product.
    """
    if _POST_TRITON and x.dtype == torch.bfloat16 and x.is_cuda and residual.shape[2] == 4:
        from . import triton_mhc
        if triton_mhc.available():
            b, s, hc, d = residual.shape
            n = b * s
            out = triton_mhc.hc_post_triton(
                x.reshape(n, d), residual.reshape(n, hc, d),
                post.reshape(n, hc), comb.reshape(n, hc, hc))
            return out.view(b, s, hc, d)
    w = torch.cat([comb.transpose(-1, -2), post.unsqueeze(-1)], dim=-1)
    b, s, hc, _ = residual.shape
    exact_here = _EXACT and not (_POST_BF16 and x.dtype != torch.float32)
    if exact_here or x.dtype == torch.float32:
        xt = torch.cat([residual, x.unsqueeze(-2)], dim=-2)
        y = torch.bmm(w.reshape(b * s, hc, hc + 1), xt.float().reshape(b * s, hc + 1, x.size(-1)))
        return y.reshape(b, s, hc, x.size(-1)).to(x.dtype)
    if residual.dtype != x.dtype:
        residual = residual.to(x.dtype)
    xt = torch.cat([residual, x.unsqueeze(-2)], dim=-2)
    y = torch.bmm(w.to(x.dtype).reshape(b * s, hc, hc + 1), xt.reshape(b * s, hc + 1, x.size(-1)))
    return y.reshape(b, s, hc, x.size(-1))


_PATCHES = {"hc_mixes": hc_mixes, "hc_pre": hc_pre, "hc_post": hc_post}


# --------------------------------------------------------------------------- install / restore
def _resolve_block(target):
    if target is None:
        from .patched_env import import_patched

        target = import_patched()
    if isinstance(target, type):
        return target, None
    block = getattr(target, "Block", None)
    if block is None:
        raise TypeError(
            "fast_mhc.install() expects the patched model module (with .Block) or the Block "
            f"class itself, got {type(target)!r}"
        )
    return block, target


def install(target=None, *, exact: bool | None = None, sinkhorn=None) -> dict:
    """Patch `Block.hc_mixes` / `hc_pre` / `hc_post` in place.  Idempotent.

    target: module returned by `patched_env.import_patched()` (patches `target.Block`) or a
        Block class.  None imports the patched model.
    exact:  if not None, `set_mode(exact)` (module default is exact=True).
    sinkhorn: optional callable override; otherwise hc_mixes calls
        `target.hc_split_sinkhorn` at run time, so setting
        `M.hc_split_sinkhorn = hc_split_sinkhorn_tl` before or after install both work.

    Returns `{name: previous attribute}` for `restore(target, prev)`; attributes that did not
    exist on the class are represented by a private sentinel.  Calling install again is a
    no-op (it returns the same previous mapping and does not stack wrappers).
    """
    global _INSTALLED, _SINKHORN_OWNER

    block, owner = _resolve_block(target)
    if _INSTALLED is not None and _INSTALLED["cls"] is not block:
        _restore_current()
    if exact is not None:
        set_mode(exact)
    sink_before = _SINKHORN
    if sinkhorn is not None:
        set_sinkhorn(sinkhorn)
    if owner is not None:
        _SINKHORN_OWNER = owner

    if _INSTALLED is not None:  # same class: idempotent, but honour the new arguments
        if sinkhorn is not None and not _INSTALLED["sink_changed"]:
            _INSTALLED["sink_before"] = sink_before
            _INSTALLED["sink_changed"] = True
        return dict(_INSTALLED["prev"])

    prev = {}
    for name, fn in _PATCHES.items():
        prev[name] = block.__dict__.get(name, _MISSING)
        setattr(block, name, fn)
    _INSTALLED = {"cls": block, "prev": prev, "owner": owner,
                  "sink_before": sink_before, "sink_changed": sinkhorn is not None}
    return dict(prev)


def _restore_current() -> None:
    global _INSTALLED, _SINKHORN
    if _INSTALLED is None:
        return
    st = _INSTALLED
    for name, old in st["prev"].items():
        if old is _MISSING:
            try:
                delattr(st["cls"], name)
            except AttributeError:
                pass
        else:
            setattr(st["cls"], name, old)
    if st["sink_changed"]:  # only undo an override that install() itself applied
        _SINKHORN = st["sink_before"]
    _INSTALLED = None


def restore(target, prev: dict) -> None:
    """Undo a previous `install()` using the dict it returned."""
    global _INSTALLED, _SINKHORN

    block, _ = _resolve_block(target)
    for name, old in prev.items():
        if old is _MISSING:
            try:
                delattr(block, name)
            except AttributeError:
                pass
        else:
            setattr(block, name, old)
    if _INSTALLED is not None and _INSTALLED["cls"] is block:
        if _INSTALLED["sink_changed"]:
            _SINKHORN = _INSTALLED["sink_before"]
        _INSTALLED = None


def uninstall(target=None) -> None:
    """Restore the methods captured by the last `install()` on `target` (no-op if not installed)."""
    if _INSTALLED is None:
        return
    if target is not None:
        block, _ = _resolve_block(target)
        if block is not _INSTALLED["cls"]:
            return
    _restore_current()
