"""官方 TileLang MoE gate（`TileKernels.moe.moe_topk_gate_forward`）的接入层。

## 为什么（按"官方优先"原则）
ref `Gate.forward` 的数学是：
    scores = softplus(linear(x.float(), W.float()) / gate_temp).sqrt()      # sqrtsoftplus
    idx    = topk(scores + bias)                                             # bias 只影响选择
    w      = scores.gather(idx); w /= w.sum(-1); w *= route_scale
官方 kernel 正好覆盖 scoring + topk + 归一化 + scaling（**对拍：idx 完全一致、weights 逐位 0 误差**），
实测 scoring+topk 段 **0.0325 ms vs torch 0.1199 ms（3.68×）**（T=8192/E=16/k=2，5090）。

我们只替换"scoring+topk+归一化"这一段，仍保留 ref 的 fp32 `linear`（那部分不在 gate kernel 里，
且 fp32 是为了路由稳定，不能随便降精度）。

用法：`install(M)` 后 `M.Gate.forward` 走官方 kernel；不满足条件（image_mask/bias_vl 等）时自动回退 ref。
"""
from __future__ import annotations

import os

import torch

_AVAILABLE = False
try:  # pragma: no cover - 只在装了 TileKernels 的机器上可用
    import sys

    _TK_PATH = os.environ.get("TILE_KERNELS_PATH", "")
    if os.path.isdir(_TK_PATH) and _TK_PATH not in sys.path:
        sys.path.insert(0, _TK_PATH)
    import tile_kernels  # noqa: F401
    _AVAILABLE = True
except Exception:  # noqa: BLE001
    tile_kernels = None

_ENABLED = os.environ.get("NANOV41_TL_GATE", "1").lower() not in ("0", "false", "no", "off")


def gate_forward_tl(self, x, image_mask=None):
    """替换 `Gate.forward`：官方 kernel 做 scoring+topk+归一化，linear 保持 fp32（与 ref 一致）。"""
    if (not _AVAILABLE or not _ENABLED or image_mask is not None
            or getattr(self, "bias_vl", None) is not None
            or self.score_func != "sqrtsoftplus"):
        return _ref_forward(self, x, image_mask)
    logits = torch.nn.functional.linear(x.float(), self.weight.float()) / self.gate_temp
    idx, weights = tile_kernels.moe.moe_topk_gate_forward(
        logits, self.topk, False, 1, self.route_scale, 0, "sqrtsoftplus", bias=self.bias)
    return weights, idx


def _ref_forward(self, x, image_mask=None):
    """原 ref 实现（保留为 fallback，逐位一致）。"""
    import torch.nn.functional as F

    scores = torch.nn.functional.linear(x.float(), self.weight.float()) / self.gate_temp
    if self.score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif self.score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    bias = self.bias
    if image_mask is not None and getattr(self, "bias_vl", None) is not None:
        bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
    indices = (scores + bias).topk(self.topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if self.norm_topk_prob and self.topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * self.route_scale
    return weights, indices


_PREV = {}


def install(M):
    """M = patched_env.import_patched() 返回的模块（含 Gate 类）。"""
    prev = getattr(M.Gate, "forward", None)
    _PREV[id(M)] = prev
    M.Gate.forward = gate_forward_tl
    return prev


def uninstall(M):
    prev = _PREV.pop(id(M), None)
    if prev is not None:
        M.Gate.forward = prev


def available() -> bool:
    return _AVAILABLE and _ENABLED
