# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""整模型级 MoE 堆叠参数集成测试（CPU）：构建/前向/反向/优化器参数数/存档往返。

跑法：python3 tools/test_moe_stacked_model.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanov41.configs import D8                                # noqa: E402
from nanov41.patched_env import import_patched                # noqa: E402
from nanov41.model import NanoV41, init_random_weights         # noqa: E402
import nanov41.moe_stacked as moe_stacked                     # noqa: E402

M = import_patched()


def build():
    torch.manual_seed(0)
    model = NanoV41(dict(D8, max_batch_size=1), param_dtype="bf16")
    init_random_weights(model, seed=0)
    return model


def loss_fn(model, ids, head_w):
    x = model(ids)
    return torch.nn.functional.cross_entropy(
        x[:, :-1].reshape(-1, x.size(-1)).float(), ids[:, 1:].reshape(-1))


def main():
    torch.manual_seed(1)
    ids = torch.randint(0, D8["vocab_size"], (1, 64))
    base = build()
    stk = copy.deepcopy(base)
    n = moe_stacked.patch_model(stk, M)
    print(f"[0] patched {n} MoE")

    head_w = base.inner.head.weight.detach().clone()
    lb = loss_fn(base, ids, head_w)
    ls = loss_fn(stk, ids, head_w)
    print(f"[1] full-model loss base={float(lb):.6f} stacked={float(ls):.6f}"
          f" rel={abs(float(lb) - float(ls)) / abs(float(lb)):.2e}")
    # bf16 GEMM 的乘法顺序不同（x@Wᵀ 的两种 layout）会带来 ~1e-3 级差异；GPU 上两者走同一 Triton kernel，应逐位一致
    assert abs(float(lb) - float(ls)) / abs(float(lb)) < 5e-3

    base.zero_grad(set_to_none=True)
    stk.zero_grad(set_to_none=True)
    lb.backward()
    ls.backward()

    nz_b = sum(p.grad is not None for p in base.parameters())
    nz_s = sum(p.grad is not None for p in stk.parameters())
    print(f"[2] params with grad: base {nz_b} / stacked {nz_s}")

    ffn = stk.inner.layers[0].ffn
    sw = ffn.experts_stacked
    gnorm = sum((getattr(sw, n_).grad ** 2).sum() for n_ in ("w1", "w2", "w3")
                if getattr(sw, n_).grad is not None)
    gf = base.inner.layers[0].ffn
    gnorm_ref = sum((getattr(gf.experts[i], n_).weight.grad ** 2).sum()
                    for i in range(gf.n_local_experts) for n_ in ("w1", "w2", "w3")
                    if getattr(gf.experts[i], n_).weight.grad is not None)
    rel = abs(float(gnorm_ref) - float(gnorm)) / float(gnorm_ref)
    print(f"[3] layer0 expert grad ||g||^2: base {float(gnorm_ref):.4e} / stacked {float(gnorm):.4e} rel={rel:.2e}")
    assert rel < 5e-3

    n_base = sum(1 for p in base.parameters() if p.requires_grad)
    n_stk = sum(1 for p in stk.parameters() if p.requires_grad)
    print(f"[4] trainable tensors: {n_base} -> {n_stk}")

    # ---- 存档往返：把 stk 的 state_dict 灌回 base（官方 key 格式），loss 应回到 ls ----
    base.load_state_dict(stk.state_dict())
    l2 = loss_fn(base, ids, head_w)
    print(f"[5] base.load(stk.state_dict) loss={float(l2):.6f} (stk {float(ls):.6f})")
    assert abs(float(l2) - float(ls)) / abs(float(ls)) < 5e-3

    print("ALL PASS")


if __name__ == "__main__":
    main()
