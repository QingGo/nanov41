"""Triton 融合 CE：在线 logsumexp 算 loss（不物化 log_softmax），反向直接出 dlogits（不物化 softmax）。

## 为什么

5090 graph 步组件消融：CE+head **25.7 ms** 是最大单项（head GEMM 本身 ~11.6 ms ≈ 峰值，剩下 ~14 ms 全是
CE 的 elementwise 搬运）。当前 `_FusedHeadCE` 每个 chunk 调 `F.cross_entropy` / `torch.softmax`：
  fwd：log_softmax 写 [chunk,V] bf16 (530 MB) + nll 读 530 MB
  bwd：softmax 写 530 MB + 后续 scatter/gather 再读
融合后：
  fwd：一遍读 logits（在线 max/sum）→ 只写 lse/target（每行 4 B）
  bwd：再读一遍 logits + 写 d（省掉 softmax 的那次写）
≈ 每 chunk 省 ~1 GB 流量、少若干 kernel。

接口（在 `_FusedHeadCE` 里用，logits 是分块算出来的）：
  `ce_fwd(logits [n,V] bf16, target [n] int64, ignore_index)` → (loss_sum fp32 标量, lse [n] fp32)
  `ce_bwd(logits, target, lse, scale, ignore_index)` → dlogits [n,V] bf16
语义与 `F.cross_entropy`（含 ignore_index）+ `softmax - onehot` 一致。
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

import os

BLOCK = int(os.environ.get("NANOV41_CE_BLOCK", "4096"))
WARPS = int(os.environ.get("NANOV41_CE_WARPS", "8"))


if HAS_TRITON:

    @triton.jit
    def _ce_fwd_kernel(LOGITS, TGT, LOSS, LSE, AMAX, V, IGNORE: tl.constexpr,
                       NEED_AMAX: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tgt = tl.load(TGT + row)
        m = -float("inf")
        s = 0.0
        tgt_val = 0.0
        p_amax = 0.0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            l = tl.load(LOGITS + row * V + cols, mask=mask, other=-float("inf")).to(tl.float32)
            blk_max = tl.max(l)
            new_m = tl.maximum(m, blk_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(l - new_m))
            m = new_m
            hit = cols == tgt
            tgt_val += tl.sum(tl.where(hit, l, 0.0))
        lse = m + tl.log(s)
        # 闭式：p_t = exp(l_t - lse)，p_max = exp(m - lse) = 1/s ⇒ max|d| = max(1 - p_t, 1/s)
        # 不需要第二次扫描（这是 fp8 dlogits 的行缩放因子）
        p_t = tl.exp(tgt_val - lse)
        d_max = tl.maximum(1.0 - p_t, 1.0 / s)
        tl.store(LSE + row, lse)
        valid = tgt != IGNORE
        tl.store(LOSS + row, tl.where(valid, lse - tgt_val, 0.0))
        # |d| = |p - onehot| 的最大值（闭式，见上）
        if NEED_AMAX:
            tl.store(AMAX + row, d_max)

    @triton.jit
    def _ce_bwd_kernel(DL, LOGITS, TGT, LSE, V, SCALE, IGNORE: tl.constexpr,
                       BLOCK: tl.constexpr):
        """写 (softmax - onehot)*scale；若 DL 是 fp8 则由调用方先按行缩放（scale=1/(row_amax/448)）。"""
        row = tl.program_id(0)
        tgt = tl.load(TGT + row)
        lse = tl.load(LSE + row)
        sc = tl.load(SCALE)            # 标量指针 → 必须先 load
        valid = tgt != IGNORE
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            l = tl.load(LOGITS + row * V + cols, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(l - lse)
            p = p - tl.where(cols == tgt, 1.0, 0.0)
            p = p * sc
            p = tl.where(valid, p, 0.0)
            tl.store(DL + row * V + cols, p.to(DL.dtype.element_ty), mask=mask)

    def ce_fwd(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
               need_amax: bool = False):
        """logits [n,V] bf16, target [n] → (loss_sum, lse[, amax])；need_amax 时多返回 max|d| [n]。"""
        n, V = logits.shape
        loss_rows = torch.empty(n, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n, dtype=torch.float32, device=logits.device)
        amax = torch.empty(n, dtype=torch.float32, device=logits.device)
        _ce_fwd_kernel[(n,)](logits, target, loss_rows, lse, amax, V, int(ignore_index),
                             NEED_AMAX=need_amax, BLOCK=BLOCK, num_warps=WARPS, num_stages=2)
        if need_amax:
            return loss_rows.sum(), lse, amax
        return loss_rows.sum(), lse

    @triton.jit
    def _ce_bwd_fp8_kernel(DL, LOGITS, TGT, LSE, AMAX, RSCALE, V, IGNORE: tl.constexpr,
                           E4M3: tl.constexpr, BLOCK: tl.constexpr):
        """融合量化：直接写 fp8 的 dlogits（按行独立缩放到 e4m3），并输出每行 scale [n] fp32。

        d = (softmax - onehot)；**单 pass**：行缩放来自 fwd 存下的 max|d|（AMAX）。
        """
        row = tl.program_id(0)
        tgt = tl.load(TGT + row)
        lse = tl.load(LSE + row)
        valid = tgt != IGNORE
        amax = tl.load(AMAX + row)
        scale = tl.where(valid, tl.maximum(amax, 1e-12) / E4M3, 1.0)
        tl.store(RSCALE + row, scale)
        inv = 1.0 / scale
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            l = tl.load(LOGITS + row * V + cols, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(l - lse)
            p = p - tl.where(cols == tgt, 1.0, 0.0)
            q = tl.where(valid, p * inv, 0.0)
            q = tl.maximum(tl.minimum(q, E4M3), -E4M3)
            tl.store(DL + row * V + cols, q.to(DL.dtype.element_ty), mask=mask)

    def ce_bwd_fp8(logits: torch.Tensor, target: torch.Tensor, lse: torch.Tensor,
                   amax: torch.Tensor, ignore_index: int = -100, e4m3_max: float = 448.0):
        """→ (d_fp8 [n,V] e4m3, row_scale [n] fp32)；amax 来自 `ce_fwd(..., need_amax=True)`。"""
        n, V = logits.shape
        dl = torch.empty(n, V, dtype=torch.float8_e4m3fn, device=logits.device)
        rscale = torch.empty(n, dtype=torch.float32, device=logits.device)
        _ce_bwd_fp8_kernel[(n,)](dl, logits, target, lse, amax, rscale, V, int(ignore_index),
                                 E4M3=e4m3_max, BLOCK=BLOCK, num_warps=WARPS, num_stages=2)
        return dl, rscale

    def ce_bwd(logits: torch.Tensor, target: torch.Tensor, lse: torch.Tensor,
               scale: torch.Tensor | float, ignore_index: int = -100):
        """→ dlogits [n,V] bf16 = (softmax - onehot) * scale（ignore 行全 0）。"""
        n, V = logits.shape
        dl = torch.empty_like(logits)
        if not torch.is_tensor(scale):
            scale = torch.tensor(float(scale), dtype=torch.float32, device=logits.device)
        _ce_bwd_kernel[(n,)](dl, logits, target, lse, V, scale, int(ignore_index),
                             BLOCK=BLOCK, num_warps=WARPS, num_stages=2)
        return dl
else:  # pragma: no cover
    def ce_fwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def ce_bwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
