"""`fast_attn` 的 lse+softmax 变体：把 attention 合并段的 ~9 个小 kernel 压到 ~5 个。

## 背景与做法（见 S16 §34/§36）
原实现先求全局稳定化因子 m，再对 `[b,h,m,w]` 的 p 做 mask→amax→maximum→sub→exp→sum→cast→matmul。
本变体对每个 part（窗口 / 压缩 / sink）分别做 **softmax + logsumexp**（都是融合 kernel），再按 lse 合并：
    out = Σ_part softmax(s_p)@v_p · exp(lse_p − m) / (Σ_part exp(lse_p − m) + exp(sink − m))
数学与旧实现完全等价（同一个"有效 key + sink"的 softmax，平移不变）。

## 用法
    import nanov41.fast_attn_lse as lse
    lse.install(M)      # M.sparse_attn = sparse_attn_lse（否则原路 fast_attn 不变）
对拍：`tools/test_fast_attn_lse.py`；CPU 逻辑测试仍走 `tools/test_fast_attn.py`（原实现）。
"""
from __future__ import annotations

import torch

from .fast_attn import (  # noqa: F401 - 复用 band plan / 布局识别 / 统计
    _BM,
    _CACHE_ON,
    _DETECT_CACHE,
    _LOWP,
    _STATS,
    _band_plan,
    _ref_sparse_attn,
    detect_layout,
)

try:
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    F = None


def _fast_path_lse(q, kv, attn_sink, topk_idxs, softmax_scale, win, ncomp):
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

    # ---- 每个 part 各自 softmax + logsumexp，再按 lse 加权合并（kernel 更少）
    #   out_part = softmax(s_part) @ v_part,  lse_part = logsumexp(s_part)
    #   m = max(lse_w, lse_c, sink);  den = Σ exp(lse_part − m) + exp(sink − m)
    #   num = Σ out_part · exp(lse_part − m);  out = num / den
    # 与旧实现（先算全局 m 再 exp）数学等价：都是同一个"有效 key + sink"上的 softmax。
    sink = attn_sink.to(torch.float32).view(1, h, 1)
    v_tiles = k_tiles.permute(0, 1, 3, 2).unsqueeze(1)  # [b,1,nb,w,d]
    p_w = torch.softmax(s_w, dim=-1).to(wd)             # 融合 max/exp/sum/div
    out_w = torch.matmul(p_w, v_tiles).reshape(b, h, m_pad, d)
    lse_w = torch.logsumexp(s_w.float(), dim=-1).reshape(b, h, m_pad)
    if s_c is not None:
        p_c = torch.softmax(s_c, dim=-1).to(wd)
        out_c = torch.bmm(p_c.reshape(b * m_pad, h, ncomp),
                          v_c.reshape(b * m_pad, ncomp, d)).view(b, h, m_pad, d)
        lse_c = torch.logsumexp(s_c.float(), dim=-1).permute(0, 2, 1)  # [b,h,m_pad]
    with torch.no_grad():
        m_all = lse_w if s_c is None else torch.maximum(lse_w, lse_c)
        m_all = torch.maximum(m_all, sink)
        m_safe = torch.where(torch.isfinite(m_all), m_all, torch.zeros_like(m_all))
    w_w = (lse_w - m_safe).exp()
    den = w_w + (sink - m_safe).exp()
    num_h = out_w * w_w.unsqueeze(-1)                    # [b,h,m_pad,d]
    if s_c is not None:
        w_c = (lse_c - m_safe).exp()
        den = den + w_c
        num_h = num_h + out_c * w_c.unsqueeze(-1)
    num = num_h.permute(0, 2, 1, 3)                      # 下游按 [b,m,h,d] 使用
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
def sparse_attn_lse(q, kv, attn_sink, topk_idxs, softmax_scale):
    """与 fast_attn.sparse_attn_fast 同签名/同语义，只是合并段用 lse+softmax。"""
    _STATS["calls"] += 1
    if topk_idxs.numel() == 0 or _DISABLED:
        _STATS["fallback_calls"] += 1
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
    return _fast_path_lse(q, kv, attn_sink, topk_idxs, softmax_scale, win, ncomp)


_DISABLED = False
_PREV = {}


def install(M):
    prev = getattr(M, "sparse_attn", None)
    _PREV[id(M)] = prev
    M.sparse_attn = sparse_attn_lse
    return prev


def uninstall(M):
    prev = _PREV.pop(id(M), None)
    if prev is not None:
        M.sparse_attn = prev
