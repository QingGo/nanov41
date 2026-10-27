"""把 Triton grouped MoE 接到 ref 的 MoE.forward 上（graph-safe，动态路由）。

与 `fast_moe` 的关系：`fast_moe` 里 grouped_mm 版本**不可进 CUDA graph**（内部 offs 回 host），
static-routing 版本只是"权重不变时的等价替代"。这里用 `triton_grouped_moe` 的 device-offsets
grouped GEMM，路由每步动态计算，整个 forward 无 host 同步 → 可以进 CUDA graph，也用于真实训练。

用法：
    import nanov41.moe_triton as moe_triton
    moe_triton.install(M)      # M = import_patched() 返回的模块（改 M.MoE.forward）
    M.MoE.forward = ... restore
"""
from __future__ import annotations

import os

import torch

from . import triton_grouped_moe as tgm

# v1 = 分开 GEMM + elementwise；v2 = 融合 SwiGLU/dW13（默认）。A/B 用 NANOV41_MOE_V=v1
_MOE_V = os.environ.get("NANOV41_MOE_V", "v2").lower()


def forward_triton(self, x, image_mask=None):
    """与 ref `MoE.forward` 数学一致：只把 routed experts 换成 grouped GEMM。"""
    shape = x.size()
    x2 = x.view(-1, self.dim)
    w, idx = self.gate(x2, None if image_mask is None else image_mask.flatten())  # [T,k]
    T, k = idx.shape
    layout = tgm.MoELayout(idx.reshape(-1), self.n_routed_experts, k)
    experts = [self.experts[i] for i in range(self.experts_start_idx, self.experts_end_idx)]
    w1 = tgm.stack_expert_weights(experts, "w1")     # [E, dim, inter]
    w3 = tgm.stack_expert_weights(experts, "w3")
    w2 = tgm.stack_expert_weights(experts, "w2")     # [E, inter, dim]
    lim = getattr(experts[0], "swiglu_limit", 0.0) or 0.0
    fn = tgm.grouped_moe_experts if _MOE_V == "v1" else tgm.grouped_moe_experts_v2
    y = fn(x2, w1, w2, w3, layout, w.reshape(-1), lim)
    y = y + self.shared_experts(x2)
    return y.type_as(x2).view(shape)


def install(M, tl_gate=None):
    """安装 Triton grouped MoE；可选同时安装官方 TileLang gate（tl_gate None = 自动）。"""
    prev = getattr(M.MoE, "forward", None)
    M.MoE.forward = forward_triton
    import os
    use_tl = (os.environ.get("NANOV41_TL_GATE", "1").lower() not in ("0", "false", "no", "off")
              if tl_gate is None else bool(tl_gate))
    if use_tl:
        try:
            from . import tl_gate as _tg
            if _tg.available():
                _tg.install(M)
        except Exception:  # noqa: BLE001
            pass
    return prev


def uninstall(M, prev):
    if prev is not None:
        M.MoE.forward = prev


def set_version(v: str) -> None:
    """运行时切 v1/v2（A/B 用）。"""
    global _MOE_V
    assert v in ("v1", "v2")
    _MOE_V = v
