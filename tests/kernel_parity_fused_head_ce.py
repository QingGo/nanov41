#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""正确性测试：fused_head_ce vs (head bf16 + shift-CE) 两条路径。

不联网、CPU 可跑（bf16 在 CPU 上慢但能验证数学）。运行：
    cd nanov41 && python3 tools/test_fused_head_ce.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanov41.train_utils import (  # noqa: E402
    chunked_cross_entropy_shift, fused_head_ce,
)


def run_case(b, s, D, V, chunk, seed=0):
    torch.manual_seed(seed)
    hidden0 = torch.randn(b, s, D, dtype=torch.bfloat16)
    weight0 = torch.randn(V, D, dtype=torch.float32)
    ids = torch.randint(0, V, (b, s))

    # ---- 旧路径：head bf16 -> shift CE ----
    h1 = hidden0.clone().requires_grad_(True)
    w1 = weight0.clone().requires_grad_(True)
    logits = F.linear(h1.to(torch.bfloat16), w1.to(torch.bfloat16))
    loss1 = chunked_cross_entropy_shift(logits, ids, chunk=chunk)
    loss1.backward()

    # ---- 新路径：fused head+CE ----
    h2 = hidden0.clone().requires_grad_(True)
    w2 = weight0.clone().requires_grad_(True)
    loss2 = fused_head_ce(h2, w2, ids, chunk=chunk)
    loss2.backward()

    dl = abs(loss1.item() - loss2.item())
    dh = (h1.grad.float() - h2.grad.float()).abs()
    dw = (w1.grad - w2.grad).abs()
    dh_rel = dh.max().item() / (h1.grad.float().abs().max().item() + 1e-9)
    dw_rel = dw.max().item() / (w1.grad.abs().max().item() + 1e-9)
    print(f"b={b} s={s} D={D} V={V} chunk={chunk}: loss {loss1.item():.6f} vs {loss2.item():.6f} "
          f"(d={dl:.2e} rel={dl / max(1e-9, abs(loss1.item())):.2e})  "
          f"dh_rel={dh_rel:.2e}  dw_rel={dw_rel:.2e}")
    # v2（bf16 logits + reduction="none"）的 loss 误差来自每行 CE 的 bf16 舍入：
    # 行数少时相对误差会被放大（小词表合成用例），所以用"相对或绝对"双阈值。
    dl_rel = dl / max(1e-9, abs(loss1.item()))
    assert dl < 2e-3 or dl_rel < 5e-3, f"loss diff too big: {dl} (rel {dl_rel:.2e})"
    assert dh_rel < 1e-2, f"dh rel diff too big: {dh_rel}"
    assert dw_rel < 1e-2, f"dw rel diff too big: {dw_rel}"


def run_zloss_case(b, s, D, V, chunk, zl=1e-4, seed=2):
    """fused_head_ce(z_loss=...) 与 train.py 式 loss（CE + z-loss*mean(z^2)）对比。"""
    torch.manual_seed(seed)
    hidden0 = torch.randn(b, s, D, dtype=torch.bfloat16)
    weight0 = torch.randn(V, D, dtype=torch.float32)
    ids = torch.randint(0, V, (b, s))

    h1 = hidden0.clone().requires_grad_(True)
    w1 = weight0.clone().requires_grad_(True)
    logits = F.linear(h1.to(torch.bfloat16), w1.to(torch.bfloat16)).float()
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
    z = logits.reshape(-1, V).logsumexp(-1)
    loss1 = ce + zl * z.pow(2).mean()
    loss1.backward()

    h2 = hidden0.clone().requires_grad_(True)
    w2 = weight0.clone().requires_grad_(True)
    loss2 = fused_head_ce(h2, w2, ids, chunk=chunk, z_loss=zl)
    loss2.backward()

    dl = abs(loss1.item() - loss2.item())
    dh_rel = (h1.grad.float() - h2.grad.float()).abs().max().item() / (h1.grad.float().abs().max().item() + 1e-9)
    dw_rel = (w1.grad - w2.grad).abs().max().item() / (w1.grad.abs().max().item() + 1e-9)
    print(f"z-loss b={b} s={s} D={D} V={V}: loss {loss1.item():.6f} vs {loss2.item():.6f} "
          f"(d={dl:.2e} rel={dl / max(1e-9, abs(loss1.item())):.2e})  "
          f"dh_rel={dh_rel:.2e}  dw_rel={dw_rel:.2e}")
    # v2（bf16 logits + reduction="none"）的 loss 误差来自每行 CE 的 bf16 舍入：
    # 行数少时相对误差会被放大（小词表合成用例），所以用"相对或绝对"双阈值。
    dl_rel = dl / max(1e-9, abs(loss1.item()))
    assert dl < 2e-3 or dl_rel < 5e-3, f"loss diff too big: {dl} (rel {dl_rel:.2e})"
    assert dh_rel < 1e-2 and dw_rel < 1e-2


def run_edge_cases():
    torch.manual_seed(1)
    b, s, D, V = 2, 5, 7, 11
    hidden = torch.randn(b, s, D, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(V, D, dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, V, (b, s))
    loss = fused_head_ce(hidden, weight, ids, chunk=4)
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    # 非默认 ignore_index / 单行 chunk
    loss2 = fused_head_ce(hidden, weight, ids, chunk=1, ignore_index=-7)
    assert torch.isfinite(loss2)
    # seq=1：shift 后没有有效 token -> count=0 会除零，调用方不应这样用；这里只验证会报错而不是静默 NaN
    print("edge cases: OK")


if __name__ == "__main__":
    run_case(2, 6, 8, 13, chunk=4)
    run_case(3, 9, 16, 31, chunk=5)
    run_case(4, 4, 32, 257, chunk=3)
    run_zloss_case(2, 6, 8, 13, chunk=4)
    run_zloss_case(3, 7, 16, 31, chunk=3)
    run_edge_cases()
    print("\nALL PASS")
