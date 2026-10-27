"""FP8 head + CE（v2）：单 pass 量化、无 bf16 d 往返。

## 为什么 v2（v1 失败原因，见 S16 §21）
v1 把 head 三笔 GEMM 换 `_scaled_mm` fp8，但端到端只有 **0.41×**：`dl` 的 per-row 量化（读 530 MB + 写 265 MB）、
dW 的第二套量化（再读 530 MB）、`gw` 的 fp32 累加（198 MB×2）、每步权重两次量化……每步多 1.5–2 GB 搬运，
而 GEMM 只省 ~4 ms。

v2 的两个关键改动：
1. **CE 反向直接写 fp8 dlogits**（`triton_ce.ce_bwd_fp8`，单 pass；行缩放来自 `ce_fwd` 的**闭式** `max|d|`）
   ⇒ 不再有"写 bf16 d 再量化"的往返；
2. **dW 用 fp8 但把行缩放折进 x**：`dW = (d_fp8·s_d)ᵀ @ x = d_fp8ᵀ @ (s_d⊙x)`
   ⇒ a/b 都是 fp8、只乘一个 TensorWise 标量，无需对 d 再做第二次量化（唯一成本是 `d_fp8ᵀ` 的转置拷贝）。

## 数值口径
e4m3 单笔 GEMM 约 3–4% max-rel；**不是逐位对齐 bf16 ref**。训练可用，记录里必须标注档位。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import triton_ce as _triton_ce

E4M3 = 448.0


def _q_row(x):
    """[m, K] → fp8 + scale [m,1]（每行独立）。"""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12).float()
    s = amax / E4M3
    return (x / s).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn), s


def _q_col(x):
    """[m, K] → fp8 + scale [1,K]（每列独立）。"""
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-12).float()
    s = amax / E4M3
    return (x / s).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn), s


def _ones(dev):
    return torch.ones(1, 1, dtype=torch.float32, device=dev)


def _q_per_tensor(x):
    """[m,K] → fp8 + 单标量 scale [1]（TensorWise 用）。"""
    amax = x.abs().amax().clamp_min(1e-12).float()
    s = (amax / E4M3).reshape(1)
    return (x / s).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn), s


class _FP8HeadCEv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk, ignore_index, z_loss):
        b, s, d = hidden.shape
        n = b * s
        v = weight.shape[0]
        ctx.hidden_shape = hidden.shape
        hidden = hidden.reshape(n, d)
        assert z_loss == 0, "z-loss 暂不支持 fp8"
        target = torch.full((n,), ignore_index, dtype=torch.long, device=hidden.device)
        target.view(b, s)[:, :-1] = ids[:, 1:]
        wb = weight.to(torch.bfloat16)
        # fwd: b = W^T，RowWise 要求 scale_b [1,V] ⇒ 按 W 的行（输出 token）缩放
        wq_row, sw_row = _q_row(wb)                     # [V,D] fp8
        sw_f = sw_row.reshape(1, -1).contiguous()       # [1,V]
        # dX: b = W [V,D]（物理 [D,V]），scale_b [1,D] ⇒ 按 W 的列（输入维）缩放
        wq_col, sw_col = _q_col(wb)
        wphys = wq_col.t().contiguous()                 # [D,V]
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        count = torch.zeros((), dtype=torch.float32, device=hidden.device)
        lse_all = torch.empty(n, dtype=torch.float32, device=hidden.device)
        amax_all = torch.empty(n, dtype=torch.float32, device=hidden.device)
        hq = hidden.to(torch.bfloat16)
        for i in range(0, n, chunk):
            xq, sx = _q_row(hq[i:i + chunk])
            lc = torch._scaled_mm(xq, wq_row.t(), scale_a=sx, scale_b=sw_f,
                                  out_dtype=torch.bfloat16)
            tc = target[i:i + chunk]
            s_rows, lse, amax = _triton_ce.ce_fwd(lc, tc, ignore_index, need_amax=True)
            total = total + s_rows
            lse_all[i:i + chunk] = lse
            amax_all[i:i + chunk] = amax
            count = count + (tc != ignore_index).sum()
        ctx.save_for_backward(hidden, weight, target, wq_row, sw_f, wq_col, sw_col,
                              wphys, lse_all, amax_all)
        ctx.chunk, ctx.ignore_index, ctx.count = chunk, ignore_index, count
        return total / count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_out):
        (hidden, weight, target, wq_row, sw_f, wq_col, sw_col, wphys,
         lse_all, amax_all) = ctx.saved_tensors
        n, d = hidden.shape
        v = weight.shape[0]
        scale = (grad_out.float() / ctx.count.clamp_min(1)).detach()
        gh = torch.empty(n, d, dtype=torch.bfloat16, device=hidden.device)
        gwT = torch.zeros(d, v, dtype=torch.float32, device=hidden.device)  # dWᵀ [D,V]
        hq = hidden.to(torch.bfloat16)
        wq_dx = wphys.t()                                # [V,D] 视图，stride(0)==1
        for i in range(0, n, ctx.chunk):
            xc = hq[i:i + ctx.chunk]
            xq, sx = _q_row(xc)
            lc = torch._scaled_mm(xq, wq_row.t(), scale_a=sx, scale_b=sw_f,
                                  out_dtype=torch.bfloat16)
            tc = target[i:i + ctx.chunk]
            dfp8, sd = _triton_ce.ce_bwd_fp8(lc, tc, lse_all[i:i + ctx.chunk],
                                            amax_all[i:i + ctx.chunk], ctx.ignore_index)
            # --- dX = (d·sd) @ W ；A=dfp8 [m,V] 行缩放 sd，B=W [V,D]（scale_b [1,D]）
            dx = torch._scaled_mm(dfp8, wq_dx, scale_a=sd.reshape(-1, 1).contiguous(),
                                  scale_b=sw_col, out_dtype=torch.bfloat16)
            gh[i:i + ctx.chunk] = dx
            # --- dW = dᵀ @ x = dfp8ᵀ @ (sd⊙x)（TensorWise：sa=1, sb=列缩放）
            # dWᵀ = x'ᵀ @ d_fp8：**只转置 x'（12.6 MB）**，避免 fp8 d 的转置（torch 对 fp8 转置极慢）
            xp = (xc.float() * sd.reshape(-1, 1))          # x' = s_d ⊙ x（把 d 的行缩放折进来）
            xq_t, sx_t = _q_per_tensor(xp.to(torch.bfloat16))   # [m,D] fp8 + 单标量 scale
            a = xq_t.t().contiguous()                     # [D,m] fp8（12.6 MB，便宜）
            part = torch._scaled_mm(a, dfp8, scale_a=sx_t, scale_b=_ones(hidden.device),
                                    out_dtype=torch.bfloat16)   # [D,V] = dWᵀ
            gwT += part.float()
        gw = (gwT.t() * scale).to(weight.dtype)          # 99 MB 转置一次，~0.2 ms
        gh = (gh.float() * scale).to(hidden.dtype).reshape(ctx.hidden_shape)
        return gh, gw, None, None, None, None


def fused_head_ce_fp8(hidden, weight, ids, chunk=2048, ignore_index=-100, z_loss=0.0):
    return _FP8HeadCEv2.apply(hidden, weight, ids, chunk, ignore_index, z_loss)


def available() -> bool:
    return hasattr(torch, "_scaled_mm") and torch.cuda.is_available()
