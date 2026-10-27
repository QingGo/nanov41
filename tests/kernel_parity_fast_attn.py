#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU parity tests for `nanov41.fast_attn.sparse_attn_fast`.

Compares the fast, layout-aware implementation against the reference
`nanov41.patched_env.sparse_attn` on forward values, output dtype and gradients
(q / kv / sink), plus the layout detector and the dispatch counters.

Run:
    python3 tools/test_fast_attn.py

`--bench` (CUDA only) times reference vs fast forward / forward+backward with cuda
events on a d12-shaped prefill (b=8, seq=1024, 64 window + 32 compressed columns).
On a machine without CUDA it prints a skip line and exits 0.

Tolerances
----------
* fp32 forward/backward: rtol=1e-5, atol=1e-5 (observed ~1e-6).
* bf16: output/grad relative L2 error < 2e-2 (observed ~1.7e-3 forward / ~3.3e-3
  gradients with the default fp16-math path -- at the bf16 output-rounding floor;
  ~1e-6 forward / ~3e-3 kv-grad with FAST_ATTN_BF16=0, where the kv grad error is
  just the bf16 rounding of the returned gradient).

The bf16 cases are run twice: once with the default fp16 scores/probabilities and
once with the strict full-fp32 path, so both dispatch modes are covered.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.fast_attn import (  # noqa: E402
    detect_layout,
    get_stats,
    install,
    reset_stats,
    sparse_attn_fast,
)
import nanov41.fast_attn as fast_attn_mod  # noqa: E402
from nanov41.patched_env import sparse_attn as ref_sparse_attn  # noqa: E402

FP32_RTOL = 1e-5
FP32_ATOL = 1e-5
BF16_REL = 2e-2

_RESULTS: list[tuple[bool, str, str]] = []


def report(ok: bool, name: str, detail: str) -> None:
    _RESULTS.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _fmt(x: float) -> str:
    return f"{x:.2e}"


def _math_mode(dtype: torch.dtype, bf16_math: bool | None) -> str:
    if dtype != torch.bfloat16:
        return "fp32"
    use = fast_attn_mod._LOWP if bf16_math is None else bf16_math
    return "fp16" if use else "fp32"


# --------------------------------------------------------------------------- layouts
def make_window_part(m: int, win: int) -> torch.Tensor:
    """Reference prefill window indices: [m, min(m, win)], -1 in the future slots."""
    win_cols = min(m, win)
    i = torch.arange(m).view(m, 1)
    c = torch.arange(win_cols).view(1, win_cols)
    idxs = (i - win + 1).clamp_min(0) + c
    return torch.where(idxs > i, torch.full_like(idxs, -1), idxs)


def make_compress_part(m: int, ncomp: int, compress_len: int, seed: int = 0) -> torch.Tensor:
    """Sorted reachable compressed positions per row, -1 padded.

    Row r can reach the first ``(r+1)*compress_len//m`` compressed positions; positions
    are sorted and valid entries come first (matching the reference's sort+where).
    """
    out = torch.full((m, ncomp), -1, dtype=torch.int64)
    if ncomp == 0 or compress_len == 0:
        return out
    g = torch.Generator().manual_seed(seed)
    for r in range(m):
        avail = max(1, (r + 1) * compress_len // m)
        k = min(ncomp, avail)
        pos = torch.randperm(avail, generator=g)[:k].sort().values
        out[r, :k] = pos
    return out


def make_band_idxs(
    b: int,
    m: int,
    win: int,
    ncomp: int = 0,
    compress_len: int = 0,
    seed: int = 0,
) -> tuple[torch.Tensor, int]:
    """A realistic prefill `topk_idxs` plus the matching KV length n = m + compress_len."""
    parts = [make_window_part(m, win)]
    if ncomp:
        comp = make_compress_part(m, ncomp, compress_len, seed)
        comp = torch.where(comp >= 0, comp + m, torch.full_like(comp, -1))  # ref offset = m
        parts.append(comp)
    idxs = torch.cat(parts, dim=1).unsqueeze(0).expand(b, -1, -1).contiguous().int()
    return idxs, m + compress_len


def make_general_idxs(b: int, m: int, topk: int, n: int, seed: int = 0) -> torch.Tensor:
    """Arbitrary (non-band) indices, including all--1 rows; must hit the fallback."""
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randint(-1, n, (b, m, topk), generator=g, dtype=torch.int64)
    idxs[0, : min(4, m), :] = -1  # at least one fully masked row
    return idxs.int()


# --------------------------------------------------------------------------- compare
def forward_errors(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, bool]:
    o = out.float()
    r = ref.float()
    diff = (o - r).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    rel_l2 = float(diff.norm() / r.norm().clamp_min(1e-12)) if r.numel() else 0.0
    if out.dtype == torch.float32:
        ok = bool(torch.allclose(o, r, rtol=FP32_RTOL, atol=FP32_ATOL))
    else:
        ok = bool(torch.isfinite(o).all() and rel_l2 < BF16_REL)
    return max_abs, rel_l2, ok


def grad_errors(grads_fast, grads_ref) -> tuple[list[float], list[float], bool]:
    rels, absmax = [], []
    ok = True
    for gf, gr in zip(grads_fast, grads_ref):
        gf = gf.float()
        gr = gr.float()
        finite = bool(torch.isfinite(gf).all() and torch.isfinite(gr).all())
        d = (gf - gr).abs()
        absmax.append(float(d.max()) if d.numel() else 0.0)
        rels.append(float(d.norm() / gr.norm().clamp_min(1e-12)) if gr.numel() else 0.0)
        ok = ok and finite
    return rels, absmax, ok


def run_forward_case(
    name: str,
    b: int,
    m: int,
    h: int,
    d: int,
    win: int,
    ncomp: int,
    compress_len: int,
    dtype: torch.dtype,
    *,
    expect_fast: bool = True,
    sink_neg_inf: bool = False,
    seed: int = 0,
    bf16_math: bool | None = None,
) -> None:
    idxs, n = make_band_idxs(b, m, win, ncomp, compress_len, seed)
    q = torch.randn(b, m, h, d, dtype=dtype)
    kv = torch.randn(b, n, d, dtype=dtype)
    sink = torch.randn(h)
    if sink_neg_inf:
        sink = torch.full_like(sink, float("-inf"))
    scale = d**-0.5

    old_math = fast_attn_mod._LOWP
    if bf16_math is not None:
        fast_attn_mod._LOWP = bf16_math
    try:
        reset_stats()
        ref = ref_sparse_attn(q, kv, sink, idxs, scale)
        out = sparse_attn_fast(q, kv, sink, idxs, scale)
        stats = get_stats()
        mode = _math_mode(dtype, bf16_math)
    finally:
        fast_attn_mod._LOWP = old_math

    max_abs, rel_l2, ok = forward_errors(out, ref)
    ok = ok and out.dtype == q.dtype and out.shape == ref.shape
    if expect_fast:
        ok = ok and stats["fast_calls"] == 1 and stats["fallback_calls"] == 0
        if q.dtype == torch.bfloat16:
            key = "fast_lowp_calls" if mode == "fp16" else "fast_fp32_calls"
            ok = ok and stats[key] == 1
        path = "fast"
    else:
        ok = ok and stats["fast_calls"] == 0 and stats["fallback_calls"] == 1
        path = "fallback"
    detail = (
        f"max_abs={_fmt(max_abs)} rel_l2={_fmt(rel_l2)} dtype={out.dtype} "
        f"path={path}/{mode}"
    )
    report(ok, name, detail)


def run_backward_case(
    name: str,
    b: int,
    m: int,
    h: int,
    d: int,
    win: int,
    ncomp: int,
    compress_len: int,
    dtype: torch.dtype,
    *,
    general: bool = False,
    sink_neg_inf: bool = False,
    seed: int = 0,
    bf16_math: bool | None = None,
) -> None:
    if general:
        topk = win + ncomp if ncomp else win
        n = m + compress_len
        idxs = make_general_idxs(b, m, topk, n, seed)
    else:
        idxs, n = make_band_idxs(b, m, win, ncomp, compress_len, seed)
    q = torch.randn(b, m, h, d, dtype=dtype)
    kv = torch.randn(b, n, d, dtype=dtype)
    sink = torch.randn(h)
    if sink_neg_inf:
        sink = torch.full_like(sink, float("-inf"))
    scale = d**-0.5
    upstream = torch.randn(b, m, h, d)

    def grads(fn):
        qq = q.clone().requires_grad_()
        kk = kv.clone().requires_grad_()
        ss = sink.clone().requires_grad_()
        out = fn(qq, kk, ss, idxs, scale)
        loss = (out.float() * upstream).sum()
        gq, gk, gs = torch.autograd.grad(loss, [qq, kk, ss])
        return out.detach(), (gq, gk, gs)

    ref, gref = grads(ref_sparse_attn)
    old_math = fast_attn_mod._LOWP
    if bf16_math is not None:
        fast_attn_mod._LOWP = bf16_math
    try:
        out, gfast = grads(sparse_attn_fast)
        mode = _math_mode(dtype, bf16_math)
    finally:
        fast_attn_mod._LOWP = old_math

    rels, absmax, grads_finite = grad_errors(gfast, gref)
    if dtype == torch.float32:
        ok = all(
            bool(torch.allclose(a.float(), b.float(), rtol=FP32_RTOL, atol=FP32_ATOL))
            for a, b in zip(gfast, gref)
        )
    else:
        ok = all(r < BF16_REL for r in rels)
    ok = ok and grads_finite
    fwd_abs, fwd_rel, fwd_ok = forward_errors(out, ref)
    ok = ok and fwd_ok
    worst_abs = max(absmax) if all(math.isfinite(a) for a in absmax) else float("nan")
    detail = (
        f"dq_rel={_fmt(rels[0])} dkv_rel={_fmt(rels[1])} dsink_rel={_fmt(rels[2])} "
        f"max_abs={_fmt(worst_abs)} fwd_rel={_fmt(fwd_rel)} finite={grads_finite} "
        f"math={mode}"
    )
    report(ok, name, detail)


# --------------------------------------------------------------------------- cases
def run_forward_tests() -> None:
    f32, bf16 = torch.float32, torch.bfloat16
    cases = [
        # name, b, m, h, d, win, ncomp, compress_len, dtype
        ("f32 m<win b1 h2 d32", 1, 3, 2, 32, 8, 0, 0, f32),
        ("f32 m==win b3 h4 d64", 3, 8, 4, 64, 8, 0, 0, f32),
        ("f32 m>win b1 h8 d64", 1, 9, 8, 64, 8, 0, 0, f32),
        ("f32 m>win+ncomp b3 h8 d64", 3, 9, 8, 64, 8, 5, 16, f32),
        ("f32 m=1 b1 h2 d32", 1, 1, 2, 32, 4, 0, 0, f32),
        ("f32 m<win=64 b2 h4 d32", 2, 16, 4, 32, 64, 0, 0, f32),
        ("f32 partial tile m=16+ncomp b3 h8 d64", 3, 16, 8, 64, 16, 8, 64, f32),
        ("f32 d12 layer0 m=64 b2 h8 d64", 2, 64, 8, 64, 64, 0, 0, f32),
        ("f32 d12 compressed m=64 b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, f32),
        ("f32 one full tile m=128 b1 h4 d64", 1, 128, 4, 64, 64, 32, 512, f32),
        ("f32 last partial tile m=130 b3 h4 d64", 3, 130, 4, 64, 64, 32, 512, f32),
        ("f32 multi-tile m=257 b2 h8 d32", 2, 257, 8, 32, 64, 32, 512, f32),
        ("f32 compressed all -1 (clen=0)", 2, 20, 4, 64, 8, 4, 0, f32),
        ("f32 sink=-inf fast layout", 2, 16, 4, 64, 8, 4, 16, f32, dict(sink_neg_inf=True)),
        ("bf16 m<win b3 h8 d64", 3, 9, 8, 64, 8, 0, 0, bf16),
        ("bf16 m>win+ncomp b3 h8 d64", 3, 9, 8, 64, 8, 5, 16, bf16),
        ("bf16 d12 compressed m=64 b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, bf16),
        ("bf16 one full tile m=128 b1 h4 d32", 1, 128, 4, 32, 64, 32, 512, bf16),
        ("bf16 last partial tile m=130 b3 h4 d64", 3, 130, 4, 64, 64, 32, 512, bf16),
        # bf16 inputs forced through the strict fp32 math path (FAST_ATTN_BF16=0)
        ("bf16-fp32math m<win b3 h8 d64", 3, 9, 8, 64, 8, 0, 0, bf16,
         dict(bf16_math=False)),
        ("bf16-fp32math d12 compressed b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, bf16,
         dict(bf16_math=False)),
    ]
    for case in cases:
        name, b, m, h, d, win, ncomp, clen, dtype = case[:9]
        kw = case[9] if len(case) > 9 else {}
        run_forward_case(name, b, m, h, d, win, ncomp, clen, dtype, **kw)

    # empty index tensor: reference returns zeros without touching KV
    q = torch.randn(2, 5, 4, 32)
    kv = torch.randn(2, 5, 32)
    sink = torch.randn(4)
    empty = torch.empty(2, 5, 0, dtype=torch.int32)
    ref = ref_sparse_attn(q, kv, sink, empty, 0.1)
    out = sparse_attn_fast(q, kv, sink, empty, 0.1)
    report(
        bool(torch.equal(out, ref) and out.dtype == q.dtype and not out.abs().any()),
        "empty topk_idxs -> zeros",
        f"shape={tuple(out.shape)} dtype={out.dtype}",
    )

    # General (non-band) layout must go through the reference fallback.
    b, m, h, d, n = 2, 20, 4, 64, 64
    idxs = make_general_idxs(b, m, 12, n, seed=3)
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, n, d)
    sink = torch.randn(h)
    reset_stats()
    ref = ref_sparse_attn(q, kv, sink, idxs, d**-0.5)
    out = sparse_attn_fast(q, kv, sink, idxs, d**-0.5)
    st = get_stats()
    max_abs, rel, ok = forward_errors(out, ref)
    ok = ok and st["fast_calls"] == 0 and st["fallback_calls"] == 1
    # rows whose indices are all -1 must be exactly zero (reference behavior)
    zero_rows_ok = not bool(out[0, :4].abs().any())
    ok = ok and zero_rows_ok
    report(
        ok,
        "general path (all--1 rows) -> fallback",
        f"max_abs={_fmt(max_abs)} rel_l2={_fmt(rel)} fast={st['fast_calls']} "
        f"fallback={st['fallback_calls']} zero_rows={zero_rows_ok}",
    )


def run_detect_tests() -> None:
    m, win, ncomp, clen = 40, 16, 6, 32
    good, n = make_band_idxs(2, m, win, ncomp, clen)
    got = detect_layout(good, m)
    report(got == (win, ncomp, True), "detect real layout", f"got={got}")

    # win_hint must not change the verified answer
    report(
        detect_layout(good, m, win_hint=win) == (win, ncomp, True),
        "detect with win_hint",
        f"got={detect_layout(good, m, win_hint=win)}",
    )

    bad = good.clone()
    bad[0, 5, 2] = 3  # corrupt one window cell
    report(detect_layout(bad, m)[2] is False, "detect rejects corrupted band", "ok=False")

    bad2 = good.clone()
    bad2[1, -1, win] = m - 1  # compressed column pointing into the window region
    report(detect_layout(bad2, m)[2] is False, "detect rejects mixed compressed idx", "ok=False")

    general = make_general_idxs(2, m, 12, n, seed=5)
    report(detect_layout(general, m)[2] is False, "detect rejects general idxs", "ok=False")

    empty = torch.empty(2, m, 0, dtype=torch.int32)
    report(detect_layout(empty, m)[2] is False, "detect rejects empty", "ok=False")


def run_backward_tests() -> None:
    f32, bf16 = torch.float32, torch.bfloat16
    run_backward_case("bwd f32 m<win b1 h2 d32", 1, 3, 2, 32, 8, 0, 0, f32, seed=11)
    run_backward_case("bwd f32 m>win+ncomp b3 h8 d64", 3, 9, 8, 64, 8, 5, 16, f32, seed=12)
    run_backward_case(
        "bwd f32 d12 compressed b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, f32, seed=13
    )
    run_backward_case(
        "bwd f32 last partial tile b2 h4 d64", 3, 130, 4, 64, 64, 32, 512, f32, seed=14
    )
    run_backward_case(
        "bwd f32 general fallback b2 h4 d64", 2, 20, 4, 64, 8, 4, 32, f32,
        general=True, seed=15,
    )
    run_backward_case(
        "bwd f32 sink=-inf b2 h4 d64", 2, 16, 4, 64, 8, 4, 16, f32,
        sink_neg_inf=True, seed=16,
    )
    run_backward_case("bwd bf16 m>win+ncomp b3 h8 d64", 3, 9, 8, 64, 8, 5, 16, bf16, seed=17)
    run_backward_case(
        "bwd bf16 d12 compressed b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, bf16, seed=18
    )
    run_backward_case(
        "bwd bf16-fp32math d12 compressed b2 h8 d64", 2, 64, 8, 64, 64, 32, 512, bf16,
        seed=19, bf16_math=False,
    )


def run_install_tests() -> None:
    class FakeModule:
        pass

    M = FakeModule()
    M.sparse_attn = ref_sparse_attn
    prev1 = install(M)
    prev2 = install(M)
    ok = prev1 is ref_sparse_attn and prev2 is ref_sparse_attn and M.sparse_attn is sparse_attn_fast
    report(ok, "install idempotent + returns previous", f"prev_is_ref={prev1 is ref_sparse_attn}")


# --------------------------------------------------------------------------- bench
def bench(bm: int | None, iters: int) -> int:
    if not torch.cuda.is_available():
        print("[bench] skip: CUDA not available")
        return 0

    if bm is not None:
        fast_attn_mod._BM = bm
    dev = torch.device("cuda")
    b, m, h, d = 8, 1024, 8, 64
    win, ncomp, clen = 64, 32, 512
    idxs, n = make_band_idxs(b, m, win, ncomp, clen, seed=0)
    idxs = idxs.to(dev)
    scale = d**-0.5
    q = torch.randn(b, m, h, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, device=dev, dtype=torch.bfloat16)
    sink = torch.randn(h, device=dev)
    upstream = torch.randn(b, m, h, d, device=dev)
    prop = torch.cuda.get_device_properties(0)
    print(
        f"[bench] {prop.name} b={b} seq={m} h={h} d={d} win={win} ncomp={ncomp} "
        f"BM={fast_attn_mod._BM}"
    )

    def timing(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def make_fwd(impl, qq, kk, ss):
        return lambda: impl(qq, kk, ss, idxs, scale)

    def make_step(impl, qq, kk, ss):
        def step():
            qq.grad = None
            kk.grad = None
            ss.grad = None
            loss = (impl(qq, kk, ss, idxs, scale).float() * upstream).sum()
            loss.backward()

        return step

    def measure(label, impl, bf16_math):
        old = fast_attn_mod._LOWP
        fast_attn_mod._LOWP = bf16_math
        try:
            qq = q.clone().requires_grad_()
            kk = kv.clone().requires_grad_()
            ss = sink.clone().requires_grad_()
            with torch.no_grad():
                ref = ref_sparse_attn(q, kv, sink, idxs, scale)
                out = impl(q, kv, sink, idxs, scale)
            err = float((out.float() - ref.float()).norm() / ref.float().norm())
            fwd = timing(make_fwd(impl, qq, kk, ss))
            step = timing(make_step(impl, qq, kk, ss))
            print(
                f"[bench] {label:16s} fwd={fwd:.2f} ms  bwd={step - fwd:.2f} ms  "
                f"fwd+bwd={step:.2f} ms  rel_l2={err:.2e}"
            )
            return fwd, step
        finally:
            fast_attn_mod._LOWP = old

    ref_fwd, ref_step = measure("reference", ref_sparse_attn, True)
    bf_fwd, bf_step = measure("fast fp16 math", sparse_attn_fast, True)
    fp_fwd, fp_step = measure("fast fp32 math", sparse_attn_fast, False)
    print(
        f"[bench] speedup  fp16-math fwd={ref_fwd / max(bf_fwd, 1e-9):.2f}x "
        f"fwd+bwd={ref_step / max(bf_step, 1e-9):.2f}x | "
        f"fp32-math fwd={ref_fwd / max(fp_fwd, 1e-9):.2f}x "
        f"fwd+bwd={ref_step / max(fp_step, 1e-9):.2f}x"
    )
    return 0


# --------------------------------------------------------------------------- main
def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", action="store_true", help="CUDA cuda-event benchmark (skips tests)")
    ap.add_argument("--bm", type=int, default=None, help="override FAST_ATTN_BM for --bench")
    ap.add_argument("--iters", type=int, default=20, help="benchmark iterations (after 5 warmup)")
    args = ap.parse_args()

    if args.bench:
        return bench(args.bm, max(1, args.iters))

    # Tests are deterministic regardless of the launch environment.
    fast_attn_mod._LOWP = True

    torch.manual_seed(0)
    print(f"torch {torch.__version__} | thread={torch.get_num_threads()} | device=cpu")
    print("=" * 78)
    run_forward_tests()
    print("-" * 78)
    run_detect_tests()
    print("-" * 78)
    run_backward_tests()
    print("-" * 78)
    run_install_tests()
    print("=" * 78)

    npass = sum(1 for ok, _, _ in _RESULTS if ok)
    nfail = len(_RESULTS) - npass
    if nfail:
        print(f"FAILED {nfail} / {len(_RESULTS)}")
        for ok, name, detail in _RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print(f"ALL {npass} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
