"""FP8 head GEMM + CE：把 LM head 的 fwd/dX/dW 三笔 GEMM 换成 fp8（e4m3）`torch._scaled_mm`。

## 为什么

S16 §20 的天花板算术：head 占 fwd FLOPs 的 70.3%，head 的 3 笔 GEMM（fwd + dX + dW）共 2.44 TFLOP，
bf16 下即使 100% 峰值也要 11.6 ms；而 5090 的 fp8 峰值是 bf16 的 2×。
实测（`tools/probe_fp8_gemm.py`，同机）：head fwd 2048×384×129280 从 **0.995 → 0.493 ms（2.02×）**，
dW 形状 **2.73×**，4096³ 从 214.9 → 527.1 TF。⇒ 这是当前单项最大杠杆。

## 数值口径

fp8 e4m3 per-tensor scaling：单笔 GEMM 的 max-rel 误差 ~4%（随机数据；统计意义上 ~1%）。
训练里这是标准做法（DeepSeek V3/V4 的 FP8 训练即如此），但它**不是逐位对齐 bf16 ref**：
本项目口径是"默认 bf16 保证对齐，冲 MFU / 长跑时开 FP8，并在记录里注明"。

## 用法

`NANOV41_HEAD_FP8=1` 时，`train_utils.fused_head_ce` 自动走本模块；否则完全走原 bf16 实现。
CE 部分复用 `triton_ce`（在线 logsumexp + 直接 dlogits），与 bf16 路径共用同一套语义（含 ignore_index）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import triton_ce as _triton_ce

E4M3_MAX = 448.0
_ROWWISE = True     # 先试 per-row scaling（激活有 per-token 离群值），不支持则自动退回 per-tensor


# --------------------------------------------------------------------------- quantization
def _quant_per_tensor(x: torch.Tensor):
    amax = x.abs().amax().clamp_min(1e-12)
    scale = (amax / E4M3_MAX).to(torch.float32).reshape(1)
    xq = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale


def _quant_per_row(x: torch.Tensor):
    """[M, K] → fp8 + scale [M,1]（每行独立缩放）。"""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12).to(torch.float32)
    scale = amax / E4M3_MAX
    xq = (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale


def _quant_per_col(x: torch.Tensor):
    """[M, K] → fp8 + scale [1,K]（每列独立缩放；用于 B 操作数的 RowWise 配置）。"""
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-12).to(torch.float32)
    scale = amax / E4M3_MAX
    xq = (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale


def _quant(x: torch.Tensor, per_row: bool = True):
    """per_row=True → scale [M,1]（A 操作数）；False → 每列缩放 scale [1,N]（B 操作数）。

    torch._scaled_mm 的 RowWise 配置要求：a [M,K] fp8 + scale_a [M,1]，
    b [K,N] fp8 + scale_b [1,N]，且都 contiguous（见其报错信息的规格表）。
    """
    return _quant_per_row(x) if per_row else _quant_per_col(x)


def _scaled_mm(a, b, sa, sb):
    return torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


# --------------------------------------------------------------------------- autograd
class _FP8HeadCE(torch.autograd.Function):
    """RowWise scaling: A 侧 per-row scale [M,1]，B 侧 per-column scale [1,N]。

    torch._scaled_mm 的 RowWise 还要求 b 满足 stride(0)==1（即物理上是"转置着存"的），
    所以 b 用 [D,V]/[D,m] 的 contiguous 物理缓冲的 .t() 视图（零拷贝）。
    """

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk, ignore_index, z_loss):
        # 与 train_utils.fused_head_ce 完全一致的接口：输入是 ids（未经 shift），
        # CE 的 target = ids 左移一位、每条序列末位为 ignore_index。
        b, s, d_in = hidden.shape
        n = b * s
        target = torch.full((n,), ignore_index, dtype=torch.long, device=hidden.device)
        target.view(b, s)[:, :-1] = ids[:, 1:]
        ctx.hidden_shape = hidden.shape
        hidden = hidden.reshape(n, d_in)
        n, d = hidden.shape[0], hidden.shape[1]
        v = weight.shape[0]
        assert z_loss == 0, "z-loss 路径暂不支持 fp8"
        hq = hidden.to(torch.bfloat16)
        wb = weight.to(torch.bfloat16)
        # fwd: b = W^T [D,V]（物理 [V,D] contiguous 的转置视图），per-token scale [1,V]
        wq_row, sw_row = _quant_per_row(wb)                   # [V,D] fp8, scale [V,1]
        wq_row_t = wq_row.t()                                 # [D,V] 视图，stride(0)==1
        sw_f = sw_row.reshape(1, -1).contiguous()             # [1,V]
        # dX: b = W [V,D]（物理 [D,V] contiguous），per-column scale [1,D]
        # （RowWise 要求 scale_b 是 [1,N]；per-tensor [1] 会触发 "Invalid scaling configuration"）
        wq_col, sw_col = _quant_per_col(wb)                   # [V,D] fp8, scale [1,D]
        wphys = wq_col.t().contiguous()                       # [D,V] contiguous
        wq_dx = wphys.t()                                     # [V,D] 视图，stride(0)==1
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        count = torch.zeros((), dtype=torch.float32, device=hidden.device)
        lse_all = torch.empty(n, dtype=torch.float32, device=hidden.device)
        for i in range(0, n, chunk):
            xq, sx = _quant_per_row(hq[i:i + chunk])          # [m,D] fp8, scale [m,1]
            lc = _scaled_mm(xq, wq_row_t, sx, sw_f)           # [m,V] bf16
            tc = target[i:i + chunk]
            s_rows, lse = _triton_ce.ce_fwd(lc, tc, ignore_index)
            total = total + s_rows
            lse_all[i:i + chunk] = lse
            count = count + (tc != ignore_index).sum()
        ctx.save_for_backward(hidden, weight, target, wq_row_t, sw_f, wq_dx, sw_col,
                              lse_all)
        ctx.chunk, ctx.ignore_index, ctx.count = chunk, ignore_index, count
        return total / count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_out):
        (hidden, weight, target, wq_row_t, sw_f, wq_dx, sw_col,
         lse_all) = ctx.saved_tensors
        n, d = hidden.shape[0], hidden.shape[1]
        scale = (grad_out.float() / ctx.count.clamp_min(1)).detach().reshape(1)
        gh = torch.empty(n, d, dtype=torch.bfloat16, device=hidden.device)
        gw = torch.zeros_like(weight, dtype=torch.float32)
        hq = hidden.to(torch.bfloat16)
        E4 = E4M3_MAX
        for i in range(0, n, ctx.chunk):
            xc = hq[i:i + ctx.chunk]
            xq_r, sx_r = _quant_per_row(xc)
            lc = _scaled_mm(xq_r, wq_row_t, sx_r, sw_f)       # 重算 logits
            tc = target[i:i + ctx.chunk]
            dl = _triton_ce.ce_bwd(lc, tc, lse_all[i:i + ctx.chunk], 1.0, ctx.ignore_index)
            # --- dX = dl @ W
            dq, sd = _quant_per_row(dl)                       # [m,V] fp8, scale [m,1]
            gh[i:i + ctx.chunk] = _scaled_mm(dq, wq_dx, sd, sw_col)
            # --- dW = dl^T @ x（TensorWise：A 是 dq 的转置，逐行 scale 无法从 row-quant 复用）
            s_dl = dl.abs().amax().clamp_min(1e-12).float() / E4
            dlq = (dl / s_dl).clamp(-E4, E4).to(torch.float8_e4m3fn)
            dq_t = dlq.t().contiguous()                       # [V,m] contiguous
            xq_t, s_x = _quant_per_tensor(xc)                 # [m,D] fp8 + [1]
            xb = xq_t.t().contiguous().t()                    # [m,D] 视图，stride(0)==1
            part = _scaled_mm(dq_t, xb, s_dl.reshape(1), s_x)  # [V,D] bf16
            gw += part.float()
        gw = (gw * scale).to(weight.dtype)
        gh = (gh.float() * scale).to(hidden.dtype).reshape(ctx.hidden_shape)
        return gh, gw, None, None, None, None


def fused_head_ce_fp8(hidden, weight, ids, chunk=2048, ignore_index=-100, z_loss=0.0):
    """hidden [b,s,D] → loss（语义与 train_utils.fused_head_ce 一致：ids 内部左移）。"""
    return _FP8HeadCE.apply(hidden, weight, ids, chunk, ignore_index, z_loss)


def available() -> bool:
    return hasattr(torch, "_scaled_mm") and torch.cuda.is_available()
