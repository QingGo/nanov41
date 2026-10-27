"""共享专家（SwiGLU FFN）的小优化：w1/w3 权重拼成一个 GEMM + 官方/Triton 风格的激活。

## 背景（S16 §34/§35）
b=16 时 MoE 占 30.4 ms/步，其中 shared expert（每个 token 都过）是 ~1 次 fwd + 1 次 bwd 的 3 个 GEMM。
我们的 routed experts 已用融合 kernel（gate/up 一次 GEMM + epilogue），但 shared expert 仍走 ref 的
`Expert.forward`：`w1(x)`、`w3(x)` 两个独立 GEMM + float/clamp/silu/mul/cast 若干小 kernel，再 `w2`。

## 做法
把 `w1`、`w3` 的权重沿输出维拼接（[2*inter, dim]，每层每次 0.6 MB，可忽略），
一次 `F.linear` 出 `[.., 2*inter]`，再切片做 clamp+SwiGLU。减少 1 个 fwd GEMM 和 1 个 bwd GEMM，
并让 autograd 只走一条 GEMM 路径。数值与 ref 等价（拼接后同一行不变）。

## 用法
    import nanov41.fast_shared_expert as fse
    fse.install(M)      # 只替换 Expert.forward；routed experts 由 moe_triton 的 fused 路径处理，不受影响
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _fused_expert_forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    dtype = x.dtype
    w1, w3, w2 = self.w1.weight, self.w3.weight, self.w2.weight
    inter = w1.size(0)
    w13 = torch.cat([w1, w3], 0)                     # [2*inter, dim]
    gu = F.linear(x, w13).float()                    # 一次 GEMM
    gate, up = gu[..., :inter], gu[..., inter:]
    lim = self.swiglu_limit
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    act = F.silu(gate) * up
    if weights is not None:
        act = weights * act
    return self.w2(act.to(dtype))


_PREV = {}


def install(M):
    prev = getattr(M.Expert, "forward", None)
    _PREV[id(M)] = prev
    M.Expert.forward = _fused_expert_forward
    return prev


def uninstall(M):
    prev = _PREV.pop(id(M), None)
    if prev is not None:
        M.Expert.forward = prev
