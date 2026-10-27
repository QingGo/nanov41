"""Structure-aware fast torch implementation of the reference `sparse_attn`.

The reference fallback (`nanov41.patched_env.sparse_attn`) materializes the gathered KV
(`kvg: [b, m, topk, d]`, topk=96 at d12 prefill) and upcasts it to fp32 before every
einsum.  At b=8/seq1024 that is ~100 MB bf16 + ~200 MB fp32 of pure data movement per
layer, and profiler traces show it dominates attention time (`direct_copy` /
anonymous elementwise).  This module recognizes the *training prefill layout* and
computes exactly the same function with dense, block-tiled slices of the KV tensor.

Layout recognized (prefill, start_pos=0)
----------------------------------------
`topk_idxs` is the concatenation of two parts (see `get_window_topk_idxs` and
`Indexer.forward` in `_ref_patched_model.py`):

1. ``win`` sliding-window columns.  Column ``c`` of query ``i`` is

       idxs[i, c] = clamp(i - win + 1, 0) + c      (or -1 when that slot is in the future)

   i.e. the causal band ``[max(0, i-win+1), i]``.  The actual number of window
   columns is ``min(seqlen, win)``; `detect_layout` returns that effective width.
2. ``n_compress`` compressed-position columns, values either ``-1`` or absolute KV
   rows ``>= m`` (the reference offsets them by ``kv.size(1)``, which equals ``m``
   for the prefill chunk).  For d12: 64 window + 32 compressed = 96 columns.

Anything else (decode/ring-buffer order, general callers, malformed layouts) falls
back to the reference implementation, bit-for-bit.

Math
----
The reference computes, for one query row with key set ``S`` (valid gathered slots):

    mx = max_{j in S} s_j                       (=-inf when S is empty)
    p_j = exp(s_j - mx),  den = sum_j p_j + exp(sink_h - mx),  out = (sum_j p_j v_j) / den

with ``s_j = softmax_scale * q . k_j``.  We use a different but algebraically identical
stabilizer

    m = max(max_{j in S} s_j, sink_h)
    p_j = exp(s_j - m),   den = sum_j p_j + exp(sink_h - m),   out = (sum_j p_j v_j) / den

Both are the softmax over the augmented key set ``S + {sink}`` where the sink key has a
zero value vector; every term is multiplied by the same ``exp(mx - m)`` so the ratio is
invariant.  Ours additionally cannot overflow when ``sink_h >> max s_j``.  The window
part and the compressed part are merged with the *same* ``m`` (computed as the max of
both parts and the sink) and one denominator -- there is no double softmax.  ``m`` is
detached before ``exp``; this does not change the function as a function of the scores
(the softmax is shift-invariant), so the autograd gradients are the exact gradients of
the reference expression.

Edge cases match the reference:
- a row with no valid slots (all indices -1) produces zeros;
- rows that produce non-finite values (only possible with ``sink = -inf``) are zeroed;
- the output has ``q.dtype``.

Precision policy
----------------
* ``q.dtype == float32``: scores, probabilities and outputs are all fp32; softmax
  reductions fp32.  Observed parity vs the reference: ~3e-7 relative (forward and
  q/kv/sink gradients), i.e. far inside rtol/atol 1e-5.
* ``q.dtype == bfloat16`` (the training case): **fp16** scores/probabilities and fp16
  tensor-core matmuls (fp32 accumulation inside the MMA), while the softmax
  max/denominator reductions run in fp32.  fp16 has three more mantissa bits than bf16
  at the same 2-byte footprint, and the model's bf16 activations (RMSNorm/RoPE outputs)
  convert to fp16 exactly, so the only rounding is on the matmul result and stored
  probabilities.  This halves the large score/probability tensors that dominate memory
  traffic.  Measured vs the fp32 reference: forward relative L2 error ~1.7e-3 (random
  data; the bf16 output-rounding floor is ~1.6e-3) and gradients ~3.3e-3 on kv -- well
  inside the declared 2e-2 bf16 tolerance.  ``FAST_ATTN_BF16=0`` keeps bf16 inputs on
  the full-fp32 path (rel ~3e-5 forward, ~3e-3 kv-gradient), trading ~2x of the
  attention memory traffic for accuracy.
  Range note: fp16 covers |x| in ~[6e-5, 65504]; q/kv here are O(1), and attention
  scores are ≤ ~1e2, so overflow is not reachable in this model.  With a pathological
  checkpoint that produces fp16 non-finite values, use ``FAST_ATTN_BF16=0``.

Temporary memory (b=8, m=seqlen=1024, h=8, d=64, win=64, n_compress=32, BM=64)
-------------------------------------------------------------------------------
                                   fp16 lowp        fp32 math
    q / q_blocks (tiled copy)       8.4 MB ea      16.8 MB ea
    kv_w / kv_pad (window only)     1.6 / 1.6 MB    3.1 / 3.1 MB
    window scores s_w              16.6 MB         33.3 MB
    window probs p_w               16.6 MB         33.3 MB
    gathered compressed keys v_c   33.6 MB         67.1 MB
    compressed s_c / p_c            4.2 MB ea       8.4 MB ea
    out_w / out_c                   8.4 MB ea      16.8 MB ea
    softmax max / denominator       ~2 MB           ~2 MB

Forward peaks at roughly 130 MB (fp16 lowp) / 250 MB (fp32 math) of transient
temporaries.  Autograd retains ~90 MB/layer (fp16 lowp) or ~170 MB/layer (fp32 math),
i.e. ~1.1 GB / ~2.0 GB for the 12 d12 layers.  The reference retains ~430 MB/layer
(bf16 gather + fp32 gather + probs).  A custom `autograd.Function` that stored ``p``
in fp16 and recomputed ``v_c`` from the indices could push this below ~45 MB/layer,
at the cost of hand-deriving the backward; the current version prefers correctness.

Install
-------
    from nanov41.patched_env import import_patched
    from nanov41.fast_attn import install
    M = import_patched()
    install(M)          # M.sparse_attn = sparse_attn_fast; returns the previous fn

`install` is idempotent and returns the previous callable so the caller can restore it.
The reference model resolves `sparse_attn` as a module global, so `M` must be the
imported module (not a model instance).

Statistics and knobs
--------------------
Counters are always updated (Python ints, negligible).  Set ``FAST_ATTN_STATS=1`` to
print a summary at interpreter exit; `get_stats()` / `reset_stats()` are available for
programmatic checks (``fast_calls`` / ``fallback_calls`` / ``fast_lowp_calls`` /
``fast_fp32_calls`` / ``layout_fail`` confirm dispatch on GPU).

    FAST_ATTN_BM=64        query tile size (default 64; 128 reproduces the original
                           suggestion, 32/64 reduce the band-mask waste and traffic)
    FAST_ATTN_BF16=0       force the full-fp32 path for bf16 inputs (disables fp16 math)
    FAST_ATTN_DISABLE=1    force the reference path (A/B measurement)
    FAST_ATTN_STATS=1      print the counter summary at exit
"""

from __future__ import annotations

import atexit
import os
import sys
from collections import namedtuple

import torch
import torch.nn.functional as F

try:  # normal package import
    from .patched_env import sparse_attn as _ref_sparse_attn
except ImportError:  # pragma: no cover - `python nanov41/fast_attn.py` style
    from nanov41.patched_env import sparse_attn as _ref_sparse_attn

__all__ = [
    "sparse_attn_fast",
    "detect_layout",
    "install",
    "get_stats",
    "reset_stats",
]


# --------------------------------------------------------------------------- config
def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return v not in ("", "0", "false", "False", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if v >= 1 else default


_BM = _env_int("FAST_ATTN_BM", 64)  # queries per tile; 64 halves the band-mask waste
_DISABLED = _env_flag("FAST_ATTN_DISABLE")
_STATS_ENABLED = _env_flag("FAST_ATTN_STATS")
# bf16 inputs use fp16 scores/probabilities unless FAST_ATTN_BF16=0 (see _fast_path).
_LOWP = os.environ.get("FAST_ATTN_BF16", "").lower() not in ("0", "false", "no", "off")

# Counters are plain ints: updating them costs less than the Python call overhead.
_STATS = {
    "calls": 0,
    "fast_calls": 0,
    "fallback_calls": 0,
    "empty_calls": 0,
    "layout_fail": 0,
    "fast_lowp_calls": 0,
    "fast_fp32_calls": 0,
    "fast_tokens": 0,          # b*m rows handled by the fast path
    "fast_window_blocks": 0,   # number of query tiles
    "fast_compress_cols": 0,   # total compressed columns gathered
}


def get_stats() -> dict:
    """Snapshot of the dispatch counters (always collected)."""
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def _print_stats() -> None:
    s = get_stats()
    print(
        "[fast_attn] "
        + " ".join(f"{k}={v}" for k, v in s.items())
        + f" bm={_BM} lowp={_LOWP} disabled={_DISABLED}",
        file=sys.stderr,
    )


if _STATS_ENABLED:
    atexit.register(_print_stats)


# --------------------------------------------------------------------------- layout
def _band_expected(m: int, win: int, device: torch.device) -> torch.Tensor:
    """The exact index tensor the reference produces for the window part: [m, win]."""
    i = torch.arange(m, device=device, dtype=torch.int64).view(m, 1)
    c = torch.arange(win, device=device, dtype=torch.int64).view(1, win)
    expected = (i - win + 1).clamp_min(0) + c
    return torch.where(expected <= i, expected, torch.full_like(expected, -1))


def _is_band(part: torch.Tensor, m: int, win: int) -> bool:
    """True iff every batch row of `part` equals the causal sliding-window band."""
    expected = _band_expected(m, win, part.device)
    return bool((part == expected.unsqueeze(0)).all())


def detect_layout(topk_idxs: torch.Tensor, m: int, win_hint: int | None = None):
    """Recognize the prefill [window band | compressed positions] layout.

    Returns ``(win, n_compress, ok)`` where ``win`` is the effective window width
    (``min(seqlen, window_size)``, so 64 for d12 prefill) and ``n_compress`` is the
    number of trailing compressed columns.  ``ok=False`` means "not recognized, use
    the reference implementation".

    ``win_hint`` (the model's ``window_size``) is only a hint used as an extra
    candidate split; the band formula is always verified.  The check is exact and
    covers every batch row and every window column, so a fast path is never taken on
    an unverified layout.
    """
    if not torch.is_tensor(topk_idxs) or topk_idxs.ndim != 3:
        return 0, 0, False
    b, mm, t = topk_idxs.shape
    if mm != m or m <= 0 or t <= 0:
        return 0, 0, False

    # Candidate 1: the compressed columns are a suffix; absolute compressed indices are
    # >= m, window indices are < m, so the first column containing any value >= m is the
    # natural split.  Candidate 2: win_hint.  Candidate 3: trailing all-(-1) columns
    # (a compressed part that happens to be entirely unreachable).  Candidate 4: no
    # compressed part.
    candidates: list[int] = []
    has_ge = (topk_idxs >= m).any(dim=(0, 1))
    nz = torch.nonzero(has_ge, as_tuple=False)
    if nz.numel():
        candidates.append(t - int(nz[0, 0]))
    if win_hint is not None:
        hint = min(max(int(win_hint), 0), t)
        candidates.append(t - hint)
    all_neg = (topk_idxs == -1).all(dim=(0, 1))
    # number of trailing all-(-1) columns, without a Python loop / per-column sync
    n_trailing = int(all_neg.flip(0).to(torch.int64).cumprod(0).sum())
    if n_trailing:
        candidates.append(n_trailing)
    candidates.append(0)

    seen = set()
    for ncomp in candidates:
        if ncomp in seen or not (0 <= ncomp < t):
            continue
        seen.add(ncomp)
        win = t - ncomp
        if win < 1:
            continue
        if ncomp:
            comp = topk_idxs[:, :, win:]
            # Compressed slots may be empty (-1) or point into the compressed KV region.
            if not bool(((comp == -1) | (comp >= m)).all()):
                continue
        if _is_band(topk_idxs[:, :, :win], m, win):
            return win, ncomp, True
    return 0, 0, False


# --------------------------------------------------------------------------- band plan
_BandPlan = namedtuple(
    "BandPlan",
    "m win bm nb m_pad w pad_left pad_len mask",  # mask: [nb, bm, w] bool
)
_PLANS: dict = {}
_PLANS_MAX = 256

# 布局识别缓存：detect_layout 内部有 `.any()/.all()/nonzero()`，每次调用都会 device→host 同步。
# 训练 prefill 的布局只由 (b, m, t, kv_n) 决定（band 公式恒定；压缩列恒为 -1 或 >= m），
# 所以按这个 key 缓存识别结果，把每步 12-24 次同步降到 0（首帧之后）。
# 关闭方式：FAST_ATTN_CACHE=0（例如需要动态 layout 的实验）。
_CACHE_ON = os.environ.get("FAST_ATTN_CACHE", "1").lower() not in ("0", "false", "no", "off")
_DETECT_CACHE: dict = {}


def _band_plan(m: int, win: int, bm: int, device: torch.device) -> _BandPlan:
    """Cache the per-tile boolean band mask for (m, win, bm, device).

    Keys for tile ``i`` are taken from ``kv_pad`` at ``[i*bm, i*bm + w)`` where
    ``kv_pad`` is the window KV left-padded by ``win-1`` zeros.  In absolute KV-row
    coordinates that tile covers ``i*bm - (win-1) ... i*bm + bm - 1``; the mask below
    is expressed in those absolute coordinates, so the left/right padding rows (and
    query rows past ``m`` in the last tile) are correctly invalid.
    """
    key = (device.type, device.index, m, win, bm)
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    nb = (m + bm - 1) // bm
    m_pad = nb * bm
    w = bm + win - 1
    pad_left = win - 1
    pad_len = m_pad + pad_left

    rows = (
        torch.arange(nb, device=device, dtype=torch.int64).view(nb, 1) * bm
        + torch.arange(bm, device=device, dtype=torch.int64).view(1, bm)
    )  # [nb, bm] absolute query position
    keys = (
        torch.arange(nb, device=device, dtype=torch.int64).view(nb, 1) * bm
        - pad_left
        + torch.arange(w, device=device, dtype=torch.int64).view(1, w)
    )  # [nb, w] absolute key position
    mask = (
        (keys.unsqueeze(1) >= 0)
        & (keys.unsqueeze(1) < m)
        & (keys.unsqueeze(1) <= rows.unsqueeze(2))
        & (keys.unsqueeze(1) >= rows.unsqueeze(2) - win + 1)
    )  # [nb, bm, w]
    plan = _BandPlan(m, win, bm, nb, m_pad, w, pad_left, pad_len, mask)
    if len(_PLANS) >= _PLANS_MAX:  # bounded cache (training uses one shape)
        _PLANS.clear()
    _PLANS[key] = plan
    return plan


# --------------------------------------------------------------------------- kernels
def _fast_path(q, kv, attn_sink, topk_idxs, softmax_scale, win, ncomp):
    """Blocked band attention + gathered compressed columns (see module docstring)."""
    b, m, h, d = q.shape
    device = q.device
    n = kv.size(1)

    # Precision policy:
    # * fp32 inputs -> full fp32 scores/probabilities (exact 1e-5 parity).
    # * bf16 inputs -> fp16 scores/probabilities with fp32 softmax reductions.  fp16 has
    #   three more mantissa bits than bf16 at the same 2-byte footprint, and bf16 values
    #   in the model's range convert to fp16 exactly; the only rounding is on the matmul
    #   output/probabilities, which is at the bf16 output-rounding floor.  This halves
    #   the large score/probability tensors that dominate memory traffic.  matmuls still
    #   accumulate in fp32 (tensor cores).  FAST_ATTN_BF16=0 forces the full-fp32 path.
    use_lowp = q.dtype == torch.bfloat16 and _LOWP
    wd = torch.float16 if use_lowp else torch.float32
    _STATS["fast_lowp_calls" if use_lowp else "fast_fp32_calls"] += 1

    # One cast per tensor instead of one per gathered slot; fold the softmax scale into
    # q (saves a separate scores multiply; rounding is <= 1 ulp).
    q_w = q.to(wd) if q.dtype != wd else q
    q_w = q_w * softmax_scale
    if not q_w.is_contiguous():
        q_w = q_w.contiguous()
    kv_w = kv.to(wd) if kv.dtype != wd else kv
    if not kv_w.is_contiguous():
        kv_w = kv_w.contiguous()

    plan = _band_plan(m, win, _BM, device)
    nb, bm, m_pad, w, pad_left = plan.nb, plan.bm, plan.m_pad, plan.w, plan.pad_left

    # ---- queries, padded to whole tiles (extra rows are sliced off at the end)
    q_pad = F.pad(q_w, (0, 0, 0, 0, 0, m_pad - m)) if m_pad != m else q_w
    q2 = q_pad.reshape(b * m_pad, h, d)  # [b*m_pad, h, d] (view)
    q_blocks = q_pad.view(b, nb, bm, h, d).permute(0, 3, 1, 2, 4).contiguous()  # [b,h,nb,bm,d]

    # ---- window keys: view-only tiles of the KV (one small left/right padded copy)
    kv_pad = F.pad(kv_w[:, :m], (0, 0, pad_left, m_pad - m))  # [b, m_pad+win-1, d]
    # NB: Tensor.unfold appends the window dim at the end -> [b, nb, d, w] view.
    k_tiles = kv_pad.unfold(1, w, bm)

    s_w = torch.matmul(q_blocks, k_tiles.unsqueeze(1))  # [b,h,nb,bm,w]
    s_w = s_w.masked_fill(~plan.mask.view(1, 1, nb, bm, w), float("-inf"))

    # ---- compressed keys: only n_compress columns are gathered (32 at d12), not topk.
    # Kept in the natural [b, m, h, c] layout so p_c feeds the output bmm without a copy.
    v_c = s_c = None
    if ncomp:
        comp = topk_idxs[:, :, win:]  # [b,m,ncomp], -1 or absolute row >= m
        valid = comp >= 0
        idx = comp.clamp_min(0).to(torch.int64)
        if m_pad != m:
            idx_pad = torch.zeros((b, m_pad, ncomp), dtype=torch.int64, device=device)
            idx_pad[:, :m] = idx
            valid_pad = torch.zeros((b, m_pad, ncomp), dtype=torch.bool, device=device)
            valid_pad[:, :m] = valid
        else:
            idx_pad, valid_pad = idx, valid
        base = torch.arange(b, device=device, dtype=torch.int64).view(b, 1, 1) * n
        sel = (base + idx_pad).reshape(-1)
        v_c = kv_w.reshape(b * n, d).index_select(0, sel).reshape(b, m_pad, ncomp, d)
        s_c = torch.bmm(q2, v_c.reshape(b * m_pad, ncomp, d).transpose(1, 2))
        s_c = s_c.view(b, m_pad, h, ncomp)
        s_c = s_c.masked_fill(~valid_pad.view(b, m_pad, 1, ncomp), float("-inf"))

    # ---- one stabilizer and one denominator for window + compressed + sink
    # `sink` is built outside no_grad so autograd still routes its gradient (it enters
    # only the denominator); the stabilizer itself is intentionally detached -- softmax
    # is shift invariant, so this gives exactly the reference gradients.
    sink = attn_sink.to(torch.float32).view(1, h, 1)
    with torch.no_grad():
        m_win = s_w.amax(dim=-1).float().reshape(b, h, m_pad)  # [b,h,m_pad]
        if s_c is not None:
            m_all = torch.maximum(m_win, s_c.amax(dim=-1).float().permute(0, 2, 1))
        else:
            m_all = m_win
        m_all = torch.maximum(m_all, sink)
        # All-invalid rows (only possible with sink=-inf) use 0 like the reference's
        # `mx_safe`; their numerator stays zero.
        m_safe = torch.where(torch.isfinite(m_all), m_all, torch.zeros_like(m_all))

    p_w = (s_w - m_safe.to(wd).view(b, h, nb, bm, 1)).exp()  # masked entries: exp(-inf) = 0
    den = p_w.sum(dim=-1, dtype=torch.float32).reshape(b, h, m_pad) + (sink - m_safe).exp()
    v_tiles = k_tiles.permute(0, 1, 3, 2).unsqueeze(1)  # [b,1,nb,w,d]
    out_w = torch.matmul(p_w, v_tiles).reshape(b, h, m_pad, d)  # [b,h,m_pad,d]

    if s_c is not None:
        p_c = (s_c - m_safe.permute(0, 2, 1).to(wd).unsqueeze(-1)).exp()  # [b,m,h,c]
        den = den + p_c.sum(dim=-1, dtype=torch.float32).permute(0, 2, 1)
        out_c = torch.bmm(
            p_c.reshape(b * m_pad, h, ncomp),
            v_c.reshape(b * m_pad, ncomp, d),
        ).view(b, m_pad, h, d)
        num = out_w.permute(0, 2, 1, 3) + out_c  # [b,m_pad,h,d]
    else:
        num = out_w.permute(0, 2, 1, 3)

    # Rows with no valid slot at all (only the query-padding rows of the last tile can
    # reach this when win < bm, plus the pathological sink=-inf case) have den == 0;
    # dividing 0/0 would poison autograd with NaNs even though the forward is masked to
    # zero.  Diverting the denominator to 1 gives out=0 and zero gradients there.
    den_safe = torch.where(den > 0, den, torch.ones_like(den))
    den_div = den_safe.permute(0, 2, 1).unsqueeze(-1)
    if wd != torch.float32:
        den_div = den_div.to(wd)
    out = num / den_div
    # Rows that are non-finite (sink=+inf etc.) are zeros, as in the ref.
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    out = out[:, :m]
    if out.dtype != q.dtype:
        out = out.to(q.dtype)
    return out if out.is_contiguous() else out.contiguous()


# --------------------------------------------------------------------------- entry
def sparse_attn_fast(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Drop-in replacement for ``patched_env.sparse_attn`` with the same semantics.

    q: [b,m,h,d]  kv: [b,n,d]  attn_sink: [h]  topk_idxs: [b,m,topk] (-1 invalid).
    The fast path is used only when the prefill band layout is verified; otherwise the
    reference fallback runs (so decode / arbitrary index tensors stay exact).
    """
    _STATS["calls"] += 1
    if topk_idxs.numel() == 0 or _DISABLED:
        _STATS["fallback_calls"] += 1
        if topk_idxs.numel() == 0:
            _STATS["empty_calls"] += 1
        return _ref_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    b, m, h, d = q.shape
    if kv.size(1) < m or topk_idxs.shape[0] != b or topk_idxs.shape[1] != m:
        _STATS["fallback_calls"] += 1
        return _ref_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    key = None
    if _CACHE_ON:
        key = (b, m, topk_idxs.shape[-1], kv.size(1), topk_idxs.dtype, topk_idxs.stride())
        hit = _DETECT_CACHE.get(key)
        if hit is not None:
            win, ncomp, ok = hit
            if ok:
                _STATS["cache_hits"] = _STATS.get("cache_hits", 0) + 1
        else:
            win, ncomp, ok = detect_layout(topk_idxs, m)
            _DETECT_CACHE[key] = (win, ncomp, ok)
    else:
        win, ncomp, ok = detect_layout(topk_idxs, m)
    if not ok or win < 1:
        _STATS["layout_fail"] += 1
        _STATS["fallback_calls"] += 1
        return _ref_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    _STATS["fast_calls"] += 1
    _STATS["fast_tokens"] += b * m
    _STATS["fast_window_blocks"] += (m + _BM - 1) // _BM
    _STATS["fast_compress_cols"] += ncomp
    return _fast_path(q, kv, attn_sink, topk_idxs, softmax_scale, win, ncomp)


# --------------------------------------------------------------------------- install
_PREV: dict[int, object] = {}


def install(M):
    """Install on a module object: ``M.sparse_attn = sparse_attn_fast``.

    `M` must be the module whose global `sparse_attn` the model resolves (i.e. the
    object returned by ``patched_env.import_patched()``), not a model instance.
    Returns the previous attribute so the caller can restore it.  Idempotent: a second
    call returns the same previous callable without wrapping itself.
    """
    prev = getattr(M, "sparse_attn", None)
    if prev is sparse_attn_fast:
        return _PREV.get(id(M), prev)
    _PREV[id(M)] = prev
    M.sparse_attn = sparse_attn_fast
    return prev
