# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""对拍：`nanov41.fused_swiglu` vs 官方 `Expert.forward` 的 SwiGLU 段（含梯度）。

CPU: python3 tools/test_fused_swiglu.py
GPU:  python3 tools/test_fused_swiglu.py --gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanov41.fused_swiglu as fs  # noqa: E402

LIM = 10.0


def ref_pair(gate, up):
    """官方 Expert.forward 的 SwiGLU 段（逐行复刻）。"""
    dt = gate.dtype
    g = gate.float()
    u = up.float()
    u = torch.clamp(u, min=-LIM, max=LIM)
    g = torch.clamp(g, max=LIM)
    return (torch.nn.functional.silu(g) * u).to(dt)


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()


def case(shape, dtype, gpu=False, scale=4.0):
    dev = "cuda" if gpu else "cpu"
    torch.manual_seed(0)
    # 故意少量元素超过 ±10，覆盖 clamp 饱和 mask。
    # 注意：clamp 反传在边界 x==±limit 上的语义随 torch 版本不同
    # （2.9 含等号 / 2.14 严格），本地 CPU 对拍用 scale=2 规避恰好落边界的元素。
    g0 = (torch.randn(*shape, device=dev) * scale).to(dtype)
    u0 = (torch.randn(*shape, device=dev) * scale).to(dtype)
    gr = g0.clone().requires_grad_(True)
    ur = u0.clone().requires_grad_(True)
    yr = ref_pair(gr, ur)
    dy = torch.randn_like(yr)
    yr.backward(dy)
    gf = g0.clone().requires_grad_(True)
    uf = u0.clone().requires_grad_(True)
    yf = fs.swiglu_fused(gf, uf, LIM)
    yf.backward(dy)
    rf, rg, ru = rel(yf, yr), rel(gf.grad, gr.grad), rel(uf.grad, ur.grad)
    det = True
    if gpu:
        a = fs.swiglu_fused(g0, u0, LIM).clone()
        b = fs.swiglu_fused(g0, u0, LIM).clone()
        c = fs.swiglu_fused(g0, u0, LIM).clone()
        det = bool(torch.equal(a, b) and torch.equal(b, c))
    ok = rf < (2e-2 if dtype == torch.bfloat16 else 2e-6) and \
        rg < (5e-2 if dtype == torch.bfloat16 else 2e-5) and \
        ru < (5e-2 if dtype == torch.bfloat16 else 2e-5) and det
    print(f"{'gpu' if gpu else 'cpu'} {str(shape):22s} {str(dtype):16s} "
          f"fwd={rf:.2e} dgate={rg:.2e} dup={ru:.2e} det={det} {'OK' if ok else 'FAIL'}")
    return ok


def gpu_capture():
    g = torch.randn(16384, 384, device="cuda", dtype=torch.bfloat16) * 2
    u = torch.randn(16384, 384, device="cuda", dtype=torch.bfloat16) * 2
    y = torch.empty_like(g)

    def body():
        yy = fs.swiglu_fused(g, u, LIM)
        y.copy_(yy)

    for _ in range(3):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    outs = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        outs.append(y.clone())
    ok = bool(torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]))
    print(f"gpu graph capture: replay bitwise = {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    ok = True
    ok &= case((64, 128), torch.float32)
    ok &= case((1024, 384), torch.float32)
    ok &= case((1024, 384), torch.bfloat16, scale=2.0)
    ok &= case((512, 384), torch.float32, scale=8.0)  # 覆盖 clamp 饱和（fp32 不会精确落边界）
    if args.gpu:
        assert torch.cuda.is_available()
        ok &= case((16384, 384), torch.bfloat16, gpu=True)
        ok &= case((16384, 384), torch.float32, gpu=True)
        ok &= case((1000, 385), torch.bfloat16, gpu=True)  # 非 2 的幂 / 非整除
        ok &= gpu_capture()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
