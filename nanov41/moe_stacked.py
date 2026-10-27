"""MoE 专家参数「堆叠存储」：消灭每步的函数式 layout 税（R42）。

## 背景（S17 §2.5 / S16 §0.2）
现状 `moe_triton.forward_triton` 每层每步做：
    w1 = stack([experts[i].w1.weight]).transpose(1,2).contiguous()   # 12 层 × 3 权重
其中 72 次 `copy_ [16,384,384]`（1.71 ms/step），并且 576 个 per-expert `nn.Parameter`
在反向后产生 612 次 `add_ [384,384]`（2.07 ms/step）的梯度累积 —— 合称 **参数布局税 ≈3.8 ms**。

本模块把每个 MoE 的 routed experts 改成 3 个堆叠 `nn.Parameter`（**直接按 grouped kernel 需要的
layout 存**，forward 零拷贝）：
    w1 [E, dim, inter]   （kernel 里 w1 就是这个方向）
    w2 [E, inter, dim]
    w3 [E, dim, inter]
并在 `state_dict` 上做兼容映射（保存/加载都还原成官方的 `experts.<i>.w<k>.weight`），
因此 `parity.py` 的 T2（把训练模型 state_dict 灌进官方 pristine 模型）不需要改。

## 用法（必须在模型构建 + 初始化之后、优化器创建之前调用）
    import nanov41.moe_stacked as moe_stacked
    moe_stacked.patch_model(model, M)     # model = NanoV41 实例；M = patched 模块
    # 之后照常 moe_triton.install(M)（patch_model 已重写 MoE.forward，二者不冲突）

## 数值
- GPU：forward 走 `triton_grouped_moe.grouped_moe_experts_v2`（与原路径同一 kernel，逐位一致）；
- CPU/无 Triton：走 `_torch_forward`（与官方 Expert.forward 同数学，用于 CPU 对拍）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import triton_grouped_moe as tgm


# --------------------------------------------------------------------------- 堆叠参数
class _StackedWeights(nn.Module):
    """三个 [E, ...] 参数，layout 与 grouped kernel 一致。state_dict 映射到官方 per-expert key。"""

    def __init__(self, n_experts: int, dim: int, inter: int):
        super().__init__()
        self.n_experts = n_experts
        self.w1 = nn.Parameter(torch.zeros(n_experts, dim, inter))
        self.w2 = nn.Parameter(torch.zeros(n_experts, inter, dim))
        self.w3 = nn.Parameter(torch.zeros(n_experts, dim, inter))

    def _moe_prefix(self, prefix: str) -> str:
        suffix = "experts_stacked."
        assert prefix.endswith(suffix), prefix
        return prefix[: -len(suffix)]

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        moe_prefix = self._moe_prefix(prefix)
        for i in range(self.n_experts):
            for name, param in (("w1", self.w1), ("w2", self.w2), ("w3", self.w3)):
                v = param[i].t()
                destination[moe_prefix + f"experts.{i}.{name}.weight"] = (
                    v.detach() if not keep_vars else v)


def _copy_from_experts(stack: _StackedWeights, experts, start: int, end: int) -> None:
    with torch.no_grad():
        for i in range(start, end):
            e = experts[i]
            if e is None:
                continue
            stack.w1[i - start].copy_(e.w1.weight.t())
            stack.w2[i - start].copy_(e.w2.weight.t())
            stack.w3[i - start].copy_(e.w3.weight.t())



# --------------------------------------------------------------------------- forward
def _torch_forward(self, x, image_mask=None):
    """CPU/无 Triton 的等价 forward（与官方 Expert.forward 同数学，用堆叠参数）。"""
    shape = x.size()
    x2 = x.view(-1, self.dim)
    weights, indices = self.gate(x2, None if image_mask is None else image_mask.flatten())
    s = self.experts_stacked
    lim = getattr(self, "swiglu_limit", 0.0) or 0.0
    y = torch.zeros_like(x2, dtype=torch.float32)
    for i in range(self.experts_start_idx, self.experts_end_idx):
        rows, top = torch.where(indices == i)
        if rows.numel() == 0:
            continue
        xi = x2[rows]
        gate = (xi @ s.w1[i]).float()
        up = (xi @ s.w3[i]).float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        act = F.silu(gate) * up * weights[rows, top, None]
        y[rows] += (act.to(x2.dtype) @ s.w2[i]).float()
    y = y + self.shared_experts(x2)
    return y.type_as(x2).view(shape)


def _triton_forward(self, x, image_mask=None):
    """GPU forward：直接吃堆叠参数，零拷贝。"""
    shape = x.size()
    x2 = x.view(-1, self.dim)
    weights, indices = self.gate(x2, None if image_mask is None else image_mask.flatten())
    T, k = indices.shape
    layout = tgm.MoELayout(indices.reshape(-1), self.n_routed_experts, k)
    s = self.experts_stacked
    lim = getattr(self, "swiglu_limit", 0.0) or 0.0
    y = tgm.grouped_moe_experts_v2(x2, s.w1, s.w2, s.w3, layout, weights.reshape(-1), lim)
    y = y + self.shared_experts(x2)
    return y.type_as(x2).view(shape)


def _forward(self, x, image_mask=None):
    if x.is_cuda and tgm.HAS_TRITON:
        return _triton_forward(self, x, image_mask)
    return _torch_forward(self, x, image_mask)


# --------------------------------------------------------------------------- patch
def patch_model(model, M) -> int:
    """把 model 里每个 `M.MoE` 改成堆叠参数（幂等）。返回 patch 的 MoE 个数。

    必须在模型初始化之后、优化器创建之前调用；要求 world_size == 1（nano 训练恒成立）。
    """
    n = 0
    for mod in model.modules():
        if not isinstance(mod, M.MoE):
            continue
        if getattr(mod, "experts_stacked", None) is not None:
            continue
        experts = mod.experts
        if experts is None:
            continue
        e0 = experts[mod.experts_start_idx]
        dim = e0.w1.weight.shape[0]
        inter = e0.w1.weight.shape[1]
        stack = _StackedWeights(mod.n_local_experts, dim, inter).to(
            device=e0.w1.weight.device, dtype=e0.w1.weight.dtype)
        _copy_from_experts(stack, experts, mod.experts_start_idx, mod.experts_end_idx)
        mod.add_module("experts_stacked", stack)
        mod.swiglu_limit = float(getattr(e0, "swiglu_limit", 0.0) or 0.0)
        mod.experts = None

        def _pre_hook(state_dict, prefix, local_metadata, strict, missing_keys,
                      unexpected_keys, error_msgs, _stack=stack, _E=mod.n_local_experts):
            consumed = False
            for i in range(_E):
                for name, param in (("w1", _stack.w1), ("w2", _stack.w2), ("w3", _stack.w3)):
                    key = prefix + f"experts.{i}.{name}.weight"
                    if key in state_dict:
                        v = state_dict.pop(key)
                        with torch.no_grad():
                            param[i].copy_(v.t())
                        consumed = True
            if consumed:  # 让子模块 experts_stacked 能找到自己的 key（避免 missing）
                for name, param in (("w1", _stack.w1), ("w2", _stack.w2), ("w3", _stack.w3)):
                    state_dict[prefix + f"experts_stacked.{name}"] = param.detach()

        mod._register_load_state_dict_pre_hook(_pre_hook)
        mod.forward = _forward.__get__(mod, type(mod))
        n += 1
    return n
