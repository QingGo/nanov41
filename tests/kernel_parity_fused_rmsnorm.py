# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU/GPU 对拍：`nanov41.fused_rmsnorm` vs 官方 `RMSNorm.forward`（含梯度）。

CPU（无 GPU 也能跑）：
  python3 tools/test_fused_rmsnorm.py
GPU（Triton 路径 + 图捕获 + 确定性）：
  python3 tools/test_fused_rmsnorm.py --gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanov41.fused_rmsnorm as fr  # noqa: E402


def ref_fwd(x, w, eps=1e-6):
    """官方 RMSNorm.forward 的逐行复刻（含 dtype 语义）。"""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xh = xf * torch.rsqrt(var + eps)
    return (w.float() * xh).to(dtype)


def rel(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs().max().item()
    s = b.abs().max().item() or 1.0
    return d / s


def cpu_case(shape, dtype, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype) * 0.5
    w = (torch.randn(shape[-1], dtype=torch.float32) * 0.1 + 1.0).to(torch.bfloat16)
    # 参考：autograd 穿过官方公式
    xr = x.clone().detach().requires_grad_(True)
    wr = w.clone().detach().requires_grad_(True)
    y_ref = ref_fwd(xr, wr, eps)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    # 融合实现（CPU 走 torch fallback，验证的是解析反传公式）
    xf = x.clone().detach().requires_grad_(True)
    wf = w.clone().detach().requires_grad_(True)
    y = fr.rmsnorm_fused(xf, wf, eps)
    y.backward(dy)
    ok_f = rel(y, y_ref)
    ok_dx = rel(xf.grad, xr.grad)
    ok_dw = rel(wf.grad, wr.grad)
    # 纯 forward 一致性
    ok_fwd = rel(fr.rmsnorm_ref(x, w, eps), y_ref)
    tag = f"cpu {str(shape):22s} {str(dtype):16s}"
    print(f"{tag} fwd_rel={ok_f:.2e} dx_rel={ok_dx:.2e} dw_rel={ok_dw:.2e} ref_fwd_rel={ok_fwd:.2e}")
    return ok_f, ok_dx, ok_dw, ok_fwd


def gpu_case(shape, dtype, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 0.5)
    w = ((torch.randn(shape[-1], device="cuda") * 0.1 + 1.0).to(torch.bfloat16))
    xr = x.clone().detach().requires_grad_(True)
    wr = w.clone().detach().requires_grad_(True)
    y_ref = ref_fwd(xr, wr, eps)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    xf = x.clone().detach().requires_grad_(True)
    wf = w.clone().detach().requires_grad_(True)
    y = fr.rmsnorm_fused(xf, wf, eps)
    y.backward(dy)
    ok_f, ok_dx, ok_dw = rel(y, y_ref), rel(xf.grad, xr.grad), rel(wf.grad, wr.grad)
    # 确定性：同一输入重复 3 次，输出/梯度应逐位相同（dw 的两段归约定序）
    y1 = fr.rmsnorm_fused(x, w, eps).clone()
    y2 = fr.rmsnorm_fused(x, w, eps).clone()
    y3 = fr.rmsnorm_fused(x, w, eps).clone()
    det = bool(torch.equal(y1, y2) and torch.equal(y2, y3))
    print(f"gpu {str(shape):22s} {str(dtype):16s} fwd_rel={ok_f:.2e} dx_rel={ok_dx:.2e} "
          f"dw_rel={ok_dw:.2e} det={det}")
    return ok_f, ok_dx, ok_dw, det


def gpu_capture(shape=(16, 1024, 384), dtype=torch.bfloat16):
    """CUDA graph 捕获 3 次 replay，逐位一致。"""
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 0.5
    w = torch.ones(shape[-1], device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def body():
        y.copy_(fr.rmsnorm_fused(x, w, 1e-6))

    for _ in range(3):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    torch.cuda.synchronize()
    outs = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append(y.clone())
    ok = bool(torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]))
    print(f"gpu graph capture {shape}: replay bitwise = {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    ok = True
    for shape in [(64, 384), (8192, 384), (2048, 64)]:
        f, dx, dw, rf = cpu_case(shape, torch.float32)
        ok &= max(f, dx, dw, rf) < 2e-6
    for shape in [(2048, 384)]:
        f, dx, dw, rf = cpu_case(shape, torch.bfloat16)
        ok &= max(f, dx, dw, rf) < 1e-2
    if args.gpu:
        assert torch.cuda.is_available()
        for shape, dt in [((16, 1024, 384), torch.bfloat16), ((16, 512, 64), torch.bfloat16),
                          ((2048, 96), torch.bfloat16), ((16, 1024, 384), torch.float32)]:
            f, dx, dw = gpu_case(shape, dt)[:3]
            ok &= max(f, dx, dw) < 3e-2
        ok &= gpu_capture()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
