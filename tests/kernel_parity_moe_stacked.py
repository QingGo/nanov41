# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""MoE 堆叠参数的 CPU 对拍（R42）：数学等价 + 梯度映射 + state_dict 兼容。

跑法：python3 tools/test_moe_stacked.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanov41.configs import D12                              # noqa: E402
from nanov41.patched_env import import_patched                # noqa: E402
from nanov41.model import NanoV41                             # noqa: E402
import nanov41.moe_stacked as moe_stacked                     # noqa: E402

M = import_patched()


class _Wrap(nn.Module):
    def __init__(self, ffn):
        super().__init__()
        self.ffn = ffn


def maxrel(a, b):
    return (a.float() - b.float()).abs().max().item() / (a.float().abs().max().item() + 1e-9)


def main():
    torch.manual_seed(0)
    print("building NanoV41(D12) on CPU ...", flush=True)
    model = NanoV41(D12, param_dtype="bf16")
    base = copy.deepcopy(model.inner.layers[0].ffn).float()
    for p in base.parameters():
        if p.is_floating_point():
            p.data.normal_(0.0, 0.02)
    stk = copy.deepcopy(base)
    n = moe_stacked.patch_model(_Wrap(stk), M)
    assert n == 1, n
    print(f"[0] patched {n} MoE; routed tensors {16 * 3} -> 3")

    T, dim = 257, D12["dim"]
    x = torch.randn(T, dim) * 0.1
    dy = torch.randn(T, dim) * 0.1

    # ---- 1) forward 等价 ----
    y1 = base(x.clone())
    y2 = stk(x.clone())
    print(f"[1] forward y max-rel = {maxrel(y1, y2):.3e}")
    assert maxrel(y1, y2) < 5e-6, "forward mismatch"

    # ---- 2) state_dict key 与数值兼容 ----
    k1, k2 = set(base.state_dict().keys()), set(stk.state_dict().keys())
    assert k1 == k2, (sorted(k1 ^ k2)[:8])
    print(f"[2] state_dict keys identical: {len(k1)} keys")

    # ---- 3) 双向 load_state_dict ----
    stk2 = copy.deepcopy(stk)
    stk2.load_state_dict(base.state_dict())
    print(f"[3a] stacked.load(base.state_dict) y max-rel = {maxrel(base(x.clone()), stk2(x.clone())):.3e}")
    base2 = copy.deepcopy(base)
    base2.load_state_dict(stk.state_dict())
    print(f"[3b] base.load(stack.state_dict)  y max-rel = {maxrel(base(x.clone()), base2(x.clone())):.3e}")

    # ---- 4) 梯度映射 ----
    base.zero_grad(set_to_none=True)
    stk.zero_grad(set_to_none=True)
    (base(x.clone()) * dy).sum().backward()
    (stk(x.clone()) * dy).sum().backward()
    sw = stk.experts_stacked
    worst = 0.0
    used = 0
    for i in range(stk.n_local_experts):
        for name in ("w1", "w2", "w3"):
            g_old = getattr(base.experts[i], name).weight.grad
            if g_old is None:                      # 该 expert 没被路由到
                continue
            g_new = getattr(sw, name).grad[i].t()
            used += 1
            worst = max(worst, maxrel(g_old, g_new))
    # shared expert grads 未变，也应一致
    for name in ("w1", "w2", "w3"):
        g_old = getattr(base.shared_experts, name).weight.grad
        g_new = getattr(stk.shared_experts, name).weight.grad
        worst = max(worst, maxrel(g_old, g_new))
    print(f"[4] grad map max-rel = {worst:.3e}  ({used} expert tensors routed)")
    assert worst < 5e-6, "grad mismatch"

    print(f"[5] trainable tensors: base {sum(1 for p in base.parameters() if p.requires_grad)}"
          f" -> stacked {sum(1 for p in stk.parameters() if p.requires_grad)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
