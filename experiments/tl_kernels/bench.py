# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""sm_120 NVFP4 / FP8 micro-benchmarks + correctness checks (RTX 5090).

This is the script used to produce the numbers in ``README.md`` / ``docs/benchmarks.md``.
It needs TileLang and an sm_120 GPU (RTX 5090 was the tested target); it is not part of CI.

Usage:
    python experiments/tl_kernels/bench.py                 # default shapes
    python experiments/tl_kernels/bench.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

# Make the repository root importable when run as `python experiments/tl_kernels/bench.py`.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tl_kernels import (  # noqa: E402
    TILELANG_AVAILABLE,
    cast_fp8,
    cast_fp8_row,
    dequantize_nvfp4,
    gemm_nvfp4,
    quantize_nvfp4,
    quantize_nvfp4_tl,
    tensor_amax,
)

HEAD = (8192, 129280, 384)  # d12 head: [tokens, vocab, dim]; K padded to 512
EXPERT = (1024, 384, 384)  # one expert shape


def timed(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def check_correctness(rows: int, k: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    kpad = (k + 255) // 256 * 256
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.02

    q_ref, s_ref = quantize_nvfp4(x)
    q_tl, s_tl = quantize_nvfp4_tl(x)
    same_words = (s_ref.reshape(-1) == s_tl.reshape(-1)).float().mean().item()
    same_bytes = (q_ref.reshape(-1) == q_tl.reshape(-1)).float().mean().item()
    d_ref = dequantize_nvfp4(q_ref, s_ref, rows, kpad)
    d_tl = dequantize_nvfp4(q_tl, s_tl, rows, kpad)

    m, n = 256, 256
    a = torch.randn(m, kpad, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(n, kpad, device="cuda", dtype=torch.bfloat16) * 0.02
    aq, sfa = quantize_nvfp4_tl(a)
    bq, sfb = quantize_nvfp4_tl(b)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    gemm_nvfp4(a, b)  # build/warm the kernel
    from experiments.tl_kernels.nvfp4_gemm import _make_nvfp4_gemm_kernel

    kernel = _make_nvfp4_gemm_kernel(m, n, kpad)
    kernel(aq, bq, sfa, sfb, out)
    ref = (dequantize_nvfp4(aq, sfa, m, kpad) @ dequantize_nvfp4(bq, sfb, n, kpad).t()).to(
        torch.bfloat16
    )
    gemm_max = (out.float() - ref.float()).abs().max().item()
    gemm_rel = gemm_max / ref.float().abs().max().clamp_min(1e-8).item()

    fp8_x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    scale = tensor_amax(fp8_x) / 448.0
    fp8_out = cast_fp8(fp8_x, scale)
    fp8_ref = (fp8_x.float() * (1.0 / scale)).to(torch.float8_e4m3fn)
    fp8_eq = (fp8_out.view(torch.uint8) == fp8_ref.view(torch.uint8)).float().mean().item()
    row_scale = (fp8_x.float().abs().amax(dim=1) / 448.0).clamp_min(1e-8)
    row_out = cast_fp8_row(fp8_x, row_scale)
    row_ref = (fp8_x.float() / row_scale[:, None]).to(torch.float8_e4m3fn)
    row_eq = (row_out.view(torch.uint8) == row_ref.view(torch.uint8)).float().mean().item()

    return {
        "shape": [rows, k],
        "quant_scale_words_identical_pct": round(100 * same_words, 3),
        "quant_packed_bytes_identical_pct": round(100 * same_bytes, 3),
        "dequant_max_abs_diff": float((d_ref - d_tl).abs().max().item()),
        "gemm_max_abs_diff": gemm_max,
        "gemm_rel_diff": gemm_rel,
        "fp8_cast_bytes_identical_pct": round(100 * fp8_eq, 3),
        "fp8_row_cast_bytes_identical_pct": round(100 * row_eq, 3),
    }


def bench_head() -> dict:
    m, n, k = HEAD
    kpad = (k + 255) // 256 * 256
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02

    aq, sfa = quantize_nvfp4(a)
    bq_ref, sfb_ref = quantize_nvfp4(b)
    bq_tl, sfb_tl = quantize_nvfp4_tl(b)
    from experiments.tl_kernels.nvfp4_gemm import _make_nvfp4_gemm_kernel

    kernel = _make_nvfp4_gemm_kernel(m, n, kpad)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def gemm_cached() -> None:
        kernel(aq, bq_tl, sfa, sfb_tl, c)

    def gemm_with_quant_a() -> None:
        aq2, sfa2 = quantize_nvfp4_tl(a)
        kernel(aq2, bq_tl, sfa2, sfb_tl, c)

    t_bf16 = timed(lambda: a @ b.t())
    t_gemm = timed(gemm_cached)
    t_fwd = timed(gemm_with_quant_a)
    t_q_torch = timed(lambda: quantize_nvfp4(b, k_pad_to=512), iters=10)
    t_q_tl = timed(lambda: quantize_nvfp4_tl(b), iters=10)

    fp8_x = torch.randn(8192, 129280, device="cuda", dtype=torch.bfloat16) * 0.5
    fp8_scale = tensor_amax(fp8_x) / 448.0
    t_fp8_tl = timed(lambda: cast_fp8(fp8_x, fp8_scale), iters=10)
    # baseline A: the training-loss path (bf16 math; PyTorch keeps bf16 for a 0-dim scalar)
    t_fp8_pt_bf16 = timed(lambda: (fp8_x * (1.0 / fp8_scale)).to(torch.float8_e4m3fn), iters=10)
    # baseline B: strict float32 math (what the kernel actually computes)
    t_fp8_pt_fp32 = timed(
        lambda: (fp8_x.float() * (1.0 / fp8_scale)).to(torch.float8_e4m3fn), iters=10
    )
    del fp8_x

    return {
        "head_shape": [m, n, k, kpad],
        "bf16_ms": round(1e3 * t_bf16, 3),
        "nvfp4_gemm_kernel_only_ms": round(1e3 * t_gemm, 3),
        "nvfp4_fwd_with_quant_a_ms": round(1e3 * t_fwd, 3),
        "head_gemm_tflops": round(2 * m * n * kpad / t_gemm / 1e12, 1),
        "gemm_speedup_vs_bf16": round(t_bf16 / t_gemm, 3),
        "fwd_with_quant_speedup_vs_bf16": round(t_bf16 / t_fwd, 3),
        "torch_quantizer_ms": round(1e3 * t_q_torch, 3),
        "fused_quantizer_ms": round(1e3 * t_q_tl, 3),
        "quantizer_speedup": round(t_q_torch / t_q_tl, 2),
        "fp8_cast_ms": round(1e3 * t_fp8_tl, 3),
        "fp8_cast_torch_bf16_ms": round(1e3 * t_fp8_pt_bf16, 3),
        "fp8_cast_torch_fp32_ms": round(1e3 * t_fp8_pt_fp32, 3),
        "fp8_cast_speedup_vs_bf16_path": round(t_fp8_pt_bf16 / t_fp8_tl, 2),
        "fp8_cast_speedup_vs_fp32_path": round(t_fp8_pt_fp32 / t_fp8_tl, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="write results as JSON to this path")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("bench.py requires CUDA (tested on an sm_120 RTX 5090); aborting", flush=True)
        return 2
    print(
        f"GPU={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)} "
        f"torch={torch.__version__} cuda={torch.version.cuda} tilelang_available={TILELANG_AVAILABLE}",
        flush=True,
    )
    if not TILELANG_AVAILABLE:
        print("tilelang is not importable; nothing to benchmark", flush=True)
        return 2
    import tilelang

    print(f"tilelang={tilelang.__version__}", flush=True)

    result: dict = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tilelang": tilelang.__version__,
        "correctness": check_correctness(HEAD[0], HEAD[2]),
        "bench": bench_head(),
    }
    for key, val in result.items():
        if isinstance(val, dict):
            print(f"\n[{key}]", flush=True)
            for k2, v2 in val.items():
                print(f"  {k2}: {v2}", flush=True)
        else:
            print(f"{key}: {val}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\nwritten {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
