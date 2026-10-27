# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""对拍：`nanov41.fused_rope`（Triton 单 kernel） vs 官方 `apply_rotary_emb`。

CPU（不需要 GPU）：用 as_strided 复刻 kernel 的索引公式，验证 stride/shape 推导与原地语义。
GPU：`python3 tools/test_fused_rope.py --gpu`，测 3D/4D、inverse、非连续 freqs、
bf16/fp32、图捕获与确定性。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanov41.fused_rope as fr  # noqa: E402


def make_freqs(n: int, rd: int, dtype=torch.float32):
    ang = torch.randn(n, rd // 2, dtype=dtype) * 0.5
    return torch.polar(torch.ones_like(ang), ang)


def rope_sim(x_slice: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """纯 torch 复刻 kernel 的索引数学（验证 stride 推导；无 GPU 也能跑）。"""
    rd = x_slice.shape[-1]
    if x_slice.dim() == 3:
        b, s, _ = x_slice.shape
        h = 1
        sb, st, pr = x_slice.stride(0), x_slice.stride(1), x_slice.stride(1)
    else:
        b, s, h, _ = x_slice.shape
        sb, st, pr = x_slice.stride(0), x_slice.stride(1), x_slice.stride(2)
    f, fs = fr._freqs_2d(freqs_cis)
    # 用 as_strided 把切片按 (b, t, h, rd) 读出（与 kernel 的 base 公式一致）
    flat = x_slice.as_strided((b, s, h, rd), (sb, st, pr, 1),
                              storage_offset=x_slice.storage_offset())
    tail = flat.float().clone()
    fre = f[:, 0::2].view(1, s, 1, rd // 2).expand(b, s, h, rd // 2).float()
    fim = f[:, 1::2].view(1, s, 1, rd // 2).expand(b, s, h, rd // 2).float()
    if inverse:
        fim = -fim
    re, im = tail[..., 0::2], tail[..., 1::2]
    out = torch.empty_like(tail)
    out[..., 0::2] = re * fre - im * fim
    out[..., 1::2] = re * fim + im * fre
    flat.copy_(out.to(flat.dtype))
    return x_slice


def cpu_case(parent_shape, slice_start, rd, dtype, inverse=False, ratio=1):
    torch.manual_seed(0)
    parent = torch.randn(*parent_shape, dtype=dtype)
    s = parent_shape[1]
    freqs = make_freqs(s, rd)
    if ratio > 1:
        # 该情形对应 latent rope：freqs 非连续（按 ratio 取行），但行数仍等于 x 的 token 数
        freqs = make_freqs(s * ratio, rd)[::ratio]
    # 参考：官方实现
    xr = parent.clone()
    slice_r = xr[..., -rd:]
    fr.apply_rotary_emb_ref(slice_r, freqs, inverse)
    # 复刻 kernel：从同一个 parent 拷一份，按 slice 偏移取 view
    xk = parent.clone()
    slice_k = xk[..., -rd:]
    rope_sim(slice_k, freqs, inverse)
    d = (xk.float() - xr.float()).abs().max().item()
    ok = d < 1e-6 if dtype == torch.float32 else d < 2e-2
    print(f"cpu {str(parent_shape):22s} rd={rd} inv={int(inverse)} ratio={ratio} "
          f"{str(dtype):16s} maxdiff={d:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def gpu_case(parent_shape, rd, dtype, inverse=False, ratio=1):
    torch.manual_seed(0)
    parent = (torch.randn(*parent_shape, device="cuda", dtype=dtype) * 0.5)
    s = parent_shape[1]
    freqs = make_freqs(s, rd).cuda()
    if ratio > 1:
        freqs = make_freqs(s * ratio, rd).cuda()[::ratio]
    xr = parent.clone()
    xk = parent.clone()
    fr.apply_rotary_emb_ref(xr[..., -rd:], freqs, inverse)
    out = fr.apply_rotary_emb_fused(xk[..., -rd:], freqs, inverse)
    assert out.data_ptr() == xk[..., -rd:].data_ptr()
    d = (xk.float() - xr.float()).abs().max().item()
    ok = d < (1e-6 if dtype == torch.float32 else 2e-2)
    print(f"gpu {str(parent_shape):22s} rd={rd} inv={int(inverse)} ratio={ratio} "
          f"{str(dtype):16s} maxdiff={d:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def gpu_grad_case(parent_shape=(2, 64, 8, 64), rd=16, inverse=False):
    """梯度对拍：融合路径（Function + 可追踪 copy_）vs 官方路径。"""
    torch.manual_seed(0)
    p0 = torch.randn(*parent_shape, device="cuda", dtype=torch.bfloat16) * 0.5
    freqs = make_freqs(parent_shape[1], rd).cuda()

    def run(fn):
        p = p0.clone().requires_grad_(True)
        sl = p[..., -rd:]
        fn(sl, freqs, inverse)
        out = (p.float() ** 2).sum()
        out.backward()
        return p.grad

    ga = run(fr.apply_rotary_emb_ref)
    gb = run(fr.apply_rotary_emb_fused)
    d = (ga.float() - gb.float()).abs().max().item()
    ok = d < 5e-2
    print(f"gpu-grad {str(parent_shape):20s} inv={int(inverse)} maxdiff={d:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def gpu_capture():
    parent = torch.randn(16, 1024, 8, 64, device="cuda", dtype=torch.bfloat16)
    freqs = make_freqs(1024, 16).cuda()
    x = parent.clone()
    x0 = parent.clone()
    sl = x[..., -16:]
    sl0 = x0[..., -16:]

    def body():
        sl.copy_(sl0)   # 每 replay 复位，否则原地旋转会累积（不是确定性问题）
        fr.apply_rotary_emb_fused(sl, freqs, False)

    for _ in range(3):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    outs = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append(x.clone())
    ok = bool(torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]))
    print(f"gpu graph capture: replay bitwise = {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    ok = True
    # 4D：q/o（head_dim 64，旋转尾 16）
    ok &= cpu_case((4, 128, 8, 64), 0, 16, torch.float32)
    ok &= cpu_case((4, 128, 8, 64), 0, 16, torch.bfloat16, inverse=True)
    # 3D：window kv / latent（每行 64，旋转尾 16）
    ok &= cpu_case((2, 256, 64), 0, 16, torch.float32)
    ok &= cpu_case((2, 256, 32), 0, 16, torch.bfloat16)
    # 非连续 freqs（[::2]）+ 3D
    ok &= cpu_case((2, 256, 64), 0, 16, torch.float32, ratio=2)
    if args.gpu:
        assert torch.cuda.is_available()
        ok &= gpu_case((16, 1024, 8, 64), 16, torch.bfloat16)
        ok &= gpu_case((16, 1024, 8, 64), 16, torch.bfloat16, inverse=True)
        ok &= gpu_case((16, 1024, 8, 64), 16, torch.float32)
        ok &= gpu_case((16, 1024, 64), 16, torch.bfloat16)
        ok &= gpu_case((16, 512, 32), 16, torch.bfloat16)
        ok &= gpu_case((16, 1024, 64), 16, torch.bfloat16, ratio=2)
        ok &= gpu_grad_case()
        ok &= gpu_grad_case(inverse=True)
        ok &= gpu_capture()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
