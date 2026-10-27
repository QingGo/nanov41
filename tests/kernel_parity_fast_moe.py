#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Correctness + graph-safety tests for `nanov41.fast_moe` (CPU is enough; --bench is CUDA).

Run:
    python3 tools/test_fast_moe.py
    python3 tools/test_fast_moe.py --bench --b 8 --seq 1024        # on the 5090
    python3 tools/test_fast_moe.py --bench --capture               # + CUDA graph capture

What is checked
  T1 parity vs the reference `MoE.forward`, on CPU:
       tiny/fp32 (b=2,s=8) and d12/bf16 (b=1,s=16), for the cuBLAS loop fallback *and* for
       the grouped_mm path.  grouped_mm is CUDA-only, so on CPU it is emulated by a
       mathematically identical per-group `mm` loop (the same semantics as torch's own
       `_grouped_mm` fallback); that exercises the real `_moe_grouped` code (offsets,
       stacking/transposes, fused w1/w3, scatter) including backward.
       Compared: output, loss, dx, and every parameter gradient (`gate.weight`, `gate.bias`
       (None by design: the bias only changes topk indices), every expert w1/w2/w3, shared).
  T2 image_mask: None == all-False (vision off), and with vision on the `bias_vl` branch is
       actually exercised; parity holds for None / all-False / mixed masks.
  T2b static-routing loop: record per-expert token counts once, then fwd+bwd runs under
       `assert_no_sync()` (no .tolist()/item/cpu/dynamic shape) and is bit-equal to the ref.
       This is the graph-safe fast path: torch._grouped_mm itself is not capturable on
       torch 2.12 (it host-copies offs; see nanov41/graph_moe.py), so graphs use this loop.
  T3 install()/uninstall() idempotence and end-to-end use.
  T4 aten op-count table via TorchDispatchMode (ref vs fast loop vs fast gmm-mock).
  T5 the three output-scatter strategies agree in value and gradient; CPU micro-bench.
  T6 assert_no_sync() catches tolist/item/cpu/numpy and passes normal tensor math.

Exit code 0 iff every check passes; prints max errors per dtype.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import statistics
import sys
import time
import types
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.configs import D12, TINY                       # noqa: E402
from nanov41.model import init_random_weights               # noqa: E402
from nanov41.patched_env import import_patched              # noqa: E402
import nanov41.fast_moe as fm                               # noqa: E402

M = import_patched()
TOL = {torch.float32: 1e-5, torch.bfloat16: 2e-2}

_RESULTS: list[tuple[str, bool, str]] = []
_MAXERR = {torch.float32: 0.0, torch.bfloat16: 0.0}


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    return ok


def rel_err(a, b) -> float:
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    scale = b.abs().max().item()
    diff = (a - b).abs().max().item()
    return diff / scale if scale > 0 else diff


# ------------------------------------------------------------------------------ test fixtures
def build_moe(cfg: dict, dtype: torch.dtype, seed: int = 0, std: float = 0.3):
    """MoE with the same construction path as the real model (`default_dtype` -> Linear dtype)."""
    old = M.default_dtype
    M.default_dtype = dtype
    try:
        moe = M.MoE(0, M.ModelArgs(**cfg))
    finally:
        M.default_dtype = old
    init_random_weights(moe, std=std, seed=seed)
    return moe


def run(forward, moe, x: torch.Tensor, mask: torch.Tensor | None = None):
    """Forward + scalar loss + backward; returns (y, loss, dx, {param: grad})."""
    for p in moe.parameters():
        p.grad = None
    xin = x.detach().clone().requires_grad_(True)
    y = forward(moe, xin, mask)
    loss = y.float().square().mean()
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in moe.named_parameters()}
    return y.detach(), loss.detach(), xin.grad.detach(), grads


def compare(tag: str, dtype: torch.dtype, ref, fast) -> bool:
    """Compare two `run()` tuples; returns True iff every checked quantity is within TOL.

    `None` vs all-zero gradient is accepted for an *unused* expert: the grouped_mm path
    stacks all E weight matrices, so an expert with no tokens gets a zero grad where the
    reference leaves `grad is None` (at T=8192 with E=16 every expert is used, so this only
    shows up in tiny tests / pathological routing).
    """
    tol = TOL[dtype]
    y_ref, loss_ref, dx_ref, g_ref = ref
    y_fast, loss_fast, dx_fast, g_fast = fast
    e_y = rel_err(y_fast, y_ref)
    e_loss = rel_err(loss_fast, loss_ref)
    e_dx = rel_err(dx_fast, dx_ref)
    worst_name, worst = "", 0.0
    n_grads = zero_only = 0
    for name in g_ref:
        if g_ref[name] is None:
            if g_fast[name] is None:
                continue
            if g_fast[name].abs().max().item() == 0.0:
                zero_only += 1                     # unused expert: zero == no contribution
                continue
            return check(f"{tag}: grad presence {name}", False,
                         "ref=None but fast has a non-zero grad")
        if g_fast[name] is None:
            return check(f"{tag}: grad presence {name}", False,
                         "ref has a grad but fast=None")
        n_grads += 1
        e = rel_err(g_fast[name], g_ref[name])
        if e > worst:
            worst_name, worst = name, e
    _MAXERR[dtype] = max(_MAXERR[dtype], e_y, e_loss, e_dx, worst)
    ok = e_y <= tol and e_loss <= tol and e_dx <= tol and worst <= tol
    detail = (f"y={e_y:.2e} loss={e_loss:.2e} dx={e_dx:.2e} "
              f"worst_grad={worst:.2e}@{worst_name or '-'} "
              f"({n_grads} grads" + (f", {zero_only} unused->zero" if zero_only else "") +
              f", tol={tol:g})")
    return check(tag, ok, detail)


@contextlib.contextmanager
def mock_grouped_mm():
    """Emulate torch._grouped_mm on CPU exactly like torch's own C++ fallback: per-group mm.

    This is how the CUDA-only `_moe_grouped` code path gets real coverage (fwd + bwd) on CPU.
    """
    orig_mm, orig_ready = fm._GROUPED_MM, fm._grouped_mm_ready
    calls = {"n": 0}

    def fake(a, b, offs=None, bias=None, out_dtype=None):
        calls["n"] += 1
        assert a.dim() == 2 and b.dim() == 3 and offs is not None, "grouped_mm contract changed"
        ends = [int(offs[e]) for e in range(offs.numel())]
        outs, start = [], 0
        for e, end in enumerate(ends):
            if end > start:
                outs.append(a[start:end] @ b[e])
            start = end
        if not outs:
            return a.new_zeros((0, b.size(1)))
        return torch.cat(outs, 0)

    fm._GROUPED_MM = fake
    fm._grouped_mm_ready = lambda moe, x: True
    try:
        yield calls
    finally:
        fm._GROUPED_MM = orig_mm
        fm._grouped_mm_ready = orig_ready


# ------------------------------------------------------------------------------------ T1 parity
def test_parity() -> None:
    print("\n== T1 parity vs reference MoE.forward (CPU) ==", flush=True)
    cases = [
        ("tiny/fp32", TINY, torch.float32, 2, 8, 0.5),
        ("d12/bf16", D12, torch.bfloat16, 1, 16, 0.1),
    ]
    for tag, cfg, dtype, b, s, std in cases:
        ref = build_moe(cfg, dtype, std=std)
        fast = copy.deepcopy(ref)
        x = torch.randn(b, s, cfg["dim"], dtype=dtype)
        expected = run(M.MoE.forward, ref, x)
        check(f"{tag}: gate.bias grad is None (bias only picks topk indices)",
              expected[3]["gate.bias"] is None and
              run(fm.fast_moe_forward, fast, x)[3]["gate.bias"] is None)

        for fuse in (True, False):
            fm.configure(fuse_w13=fuse)
            got = run(fm.fast_moe_forward, fast, x)
            compare(f"{tag} loop fuse_w13={fuse}", dtype, expected, got)

        # the reference also accepts a flattened [T, dim] input
        x2d = x.view(-1, cfg["dim"])
        compare(f"{tag} loop 2D input [T,dim]", dtype,
                run(M.MoE.forward, ref, x2d), run(fm.fast_moe_forward, fast, x2d))

        with mock_grouped_mm() as calls:
            for fuse in (True, False):
                fm.configure(fuse_w13=fuse)
                got = run(fm.fast_moe_forward, fast, x)
                ok = compare(f"{tag} grouped_mm(mock) fuse_w13={fuse}", dtype, expected, got)
                if ok:
                    expect_calls = 2 if fuse else 3   # routed only; shared uses F.linear
                    check(f"{tag} grouped_mm(mock) call count", calls["n"] == expect_calls * 1,
                          f"grouped_mm calls={calls['n']} (expect {expect_calls})")
                calls["n"] = 0
        fm.configure(fuse_w13=True)


# ------------------------------------------------------------------------------- T2 image_mask
def test_image_mask() -> None:
    print("\n== T2 image_mask semantics ==", flush=True)
    # vision enabled: bias_vl exists and is deliberately made dominant for expert 0
    cfg = dict(TINY, vision_n_layers=1)
    ref = build_moe(cfg, torch.float32, std=0.3)
    with torch.no_grad():
        ref.gate.bias_vl.zero_()
        ref.gate.bias_vl[0] = 50.0
    fast = copy.deepcopy(ref)
    x = torch.randn(2, 6, cfg["dim"])
    none = None
    zeros = torch.zeros(2, 6, dtype=torch.bool)
    ones = torch.ones(2, 6, dtype=torch.bool)
    mixed = torch.zeros(2, 6, dtype=torch.bool)
    mixed[:, ::2] = True

    y_none = M.MoE.forward(ref, x, none)
    y_zero = M.MoE.forward(ref, x, zeros)
    y_one = M.MoE.forward(ref, x, ones)
    check("vision on: ref(None) == ref(all-False)", torch.equal(y_none, y_zero),
          f"maxdiff={(y_none - y_zero).abs().max().item():.1e}")
    check("vision on: bias_vl is actually exercised (all-True != None)",
          not torch.equal(y_none, y_one), f"maxdiff={(y_none - y_one).abs().max().item():.1e}")

    fused_expected = run(M.MoE.forward, ref, x, mixed)
    got = run(fm.fast_moe_forward, fast, x, mixed)
    compare("vision on: fast == ref (mixed mask)", torch.float32, fused_expected, got)
    for mask, name in ((none, "None"), (zeros, "all-False"), (ones, "all-True")):
        got = run(fm.fast_moe_forward, fast, x, mask)
        compare(f"vision on: fast == ref (mask={name})", torch.float32,
                run(M.MoE.forward, ref, x, mask), got)

    # vision disabled (d12 as trained): ref passes None; an all-False mask must behave the same
    ref2 = build_moe(D12, torch.bfloat16, std=0.1)
    fast2 = copy.deepcopy(ref2)
    x2 = torch.randn(1, 16, 384, dtype=torch.bfloat16)
    z2 = torch.zeros(1, 16, dtype=torch.bool)
    e_none = run(M.MoE.forward, ref2, x2, None)
    e_zero = run(M.MoE.forward, ref2, x2, z2)
    compare("vision off: ref(None) == ref(all-False)", torch.bfloat16, e_none, e_zero)
    compare("vision off: fast(None) == ref(None)", torch.bfloat16, e_none,
            run(fm.fast_moe_forward, fast2, x2, None))
    compare("vision off: fast(all-False) == ref(None)", torch.bfloat16, e_none,
            run(fm.fast_moe_forward, fast2, x2, z2))


# ------------------------------------------------------------------------------------ T3 install
def test_static_routing() -> None:
    print("\n== T2b static-routing fused loop (graph-safe path) ==", flush=True)
    cases = [
        ("tiny/fp32", TINY, torch.float32, 2, 8, 0.5),
        ("d12/bf16", D12, torch.bfloat16, 1, 16, 0.1),
    ]
    for tag, cfg, dtype, b, s, std in cases:
        ref = build_moe(cfg, dtype, std=std)
        fast = copy.deepcopy(ref)
        x = torch.randn(b, s, cfg["dim"], dtype=dtype)
        expected = run(M.MoE.forward, ref, x)
        sizes = fm.record_group_sizes(ref, x)
        check(f"{tag}: recorded sizes cover all (token,slot) rows",
              sum(sizes) == b * s * cfg["n_activated_experts"],
              f"sizes sum={sum(sizes)} (min {min(sizes)}, max {max(sizes)})")
        fm.install_static_routing(fast, sizes)
        compare(f"{tag} static-routing loop == ref", dtype, expected,
                run(fm.fast_moe_forward, fast, x))
        try:
            with fm.assert_no_sync():
                run(fm.fast_moe_forward, fast, x)
            sync_free, detail = True, ""
        except RuntimeError as exc:
            sync_free, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"{tag}: static loop fwd+bwd is host-sync free", sync_free, detail)
        check(f"{tag}: graph_safe(static loop) == True", fm.graph_safe(fast, x))
        fm.uninstall_static_routing(fast)
        check(f"{tag}: plain loop is not graph-safe", not fm.graph_safe(fast, x))
        check(f"{tag}: uninstall restores auto path", not hasattr(fast, "_fast_moe_static_sizes"))


def test_install() -> None:
    print("\n== T3 install()/uninstall() ==", flush=True)
    ref = build_moe(TINY, torch.float32, std=0.3)
    fast = copy.deepcopy(ref)
    x = torch.randn(2, 6, TINY["dim"])
    holder = types.SimpleNamespace(MoE=M.MoE)
    original = M.MoE.forward
    prev = fm.install(holder)
    try:
        check("install returns previous forward", prev is original)
        check("install replaced MoE.forward", M.MoE.forward is fm.fast_moe_forward)
        again = fm.install(holder)
        check("install is idempotent", again is fm.fast_moe_forward and M.MoE.forward is fm.fast_moe_forward)
        y_ref = M.MoE.forward(ref, x)
        y_fast = fast(x)                                     # class-level install path
        check("installed class forward == direct call", torch.equal(y_ref, y_fast),
              f"maxdiff={(y_ref - y_fast).abs().max().item():.1e}")
    finally:
        fm.uninstall(holder, prev)
    check("uninstall restores the reference", M.MoE.forward is original)


# --------------------------------------------------------------------------------- T4 op counts
class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts: Counter = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def _count(forward, moe, x, backward: bool) -> Counter:
    for p in moe.parameters():
        p.grad = None
    counter = OpCounter()
    xin = x.detach().clone().requires_grad_(backward)
    with counter:
        y = forward(moe, xin)
        if backward:
            y.float().square().mean().backward()
    return counter.counts


def _bucket(counts: Counter):
    def n(*prefixes):
        return sum(v for op, v in counts.items() if op.startswith(prefixes))
    total = sum(counts.values())
    gemm = n("aten.mm", "aten.addmm", "aten.bmm", "aten.linear", "aten._grouped_mm")
    indexing = n("aten.index", "aten.gather", "aten.scatter", "aten.take")
    sort = n("aten.sort", "aten.argsort", "aten.topk")
    dynamic = n("aten.nonzero", "aten.bincount", "aten.masked_select", "aten.unique")
    where = n("aten.where")
    move = n("aten.to", "aten._to_copy", "aten.cat", "aten.stack", "aten.copy_")
    return total, gemm, indexing, sort, dynamic, where, move


def test_op_counts() -> None:
    print("\n== T4 aten op counts (d12 b=1 s=16 bf16, CPU dispatch) ==", flush=True)
    ref = build_moe(D12, torch.bfloat16, std=0.1)
    fast = copy.deepcopy(ref)
    x = torch.randn(1, 16, 384, dtype=torch.bfloat16)

    rows = []
    for name, fwd in (("ref (where/nonzero loop)", M.MoE.forward), ("fast (loop, w13 fused)", fm.fast_moe_forward)):
        f_counts = _count(fwd, ref if fwd is M.MoE.forward else fast, x, backward=False)
        fb_counts = _count(fwd, ref if fwd is M.MoE.forward else fast, x, backward=True)
        rows.append((name, f_counts, fb_counts))

    with mock_grouped_mm() as calls:
        g_counts = _count(fm.fast_moe_forward, fast, x, backward=False)
        gb_counts = _count(fm.fast_moe_forward, fast, x, backward=True)
    rows.append((f"fast (gmm mock, {calls['n'] // 2} grouped calls/fwd)", g_counts, gb_counts))

    head = (f"{'path':32s} {'fwd':>6s} {'fwd+bwd':>8s} | {'mm':>4s} {'index':>6s} "
            f"{'sort':>5s} {'dyn':>5s} {'where':>6s} {'move':>5s}")
    print(head)
    print("-" * len(head))
    for name, f_counts, fb_counts in rows:
        ft, fg, fi, fs, fd, fw, fmv = _bucket(f_counts)
        bt, bg, bi, bs, bd, bw, bmv = _bucket(fb_counts)
        print(f"{name:32s} {ft:6d} {bt:8d} | {bg:4d} {bi:6d} {bs:5d} {bd:5d} {bw:6d} {bmv:5d}")
    print("  mm=aten.mm/addmm/bmm/_grouped_mm; dyn=nonzero/bincount/masked_select (dynamic shape);")
    print("  where=clamp-backward (identical math both paths).  The gmm-mock row emulates each")
    print("  grouped call with E mm calls on CPU, so its real CUDA count is 2 routed + 2 shared")
    print("  GEMMs/fwd.")
    for name, _, fb_counts in rows:
        top = [(op, c) for op, c in fb_counts.most_common(6)]
        print(f"  top fwd+bwd ops {name:32s} " + ", ".join(f"{op.split('.')[1]}×{c}" for op, c in top))

    ref_b, fast_b, gmm_b = (_bucket(r[2]) for r in rows)
    check("fast loop executes fewer GEMM-class aten calls than ref",
          fast_b[1] < ref_b[1], f"ref={ref_b[1]} fast={fast_b[1]}")
    check("fast loop removes dynamic-shape ops (nonzero/bincount)",
          fast_b[4] == 0 and ref_b[4] > 0, f"ref={ref_b[4]} fast={fast_b[4]}")
    check("fast gmm-mock removes dynamic-shape ops too", gmm_b[4] == 0, f"gmm={gmm_b[4]}")

    E, k, n_layers = D12["n_routed_experts"], D12["n_activated_experts"], D12["n_layers"]
    print(f"\n  analytic GEMM launches/layer at T=8192 (all {E} experts used, topk={k}):")
    print(f"    ref              routed {E * 3} fwd + {E * 3 * 2} bwd + shared 3 + 6 + gate 3 = "
          f"{E * 9 + 9 + 3}")
    print(f"    fast loop w13    routed {E * 2} fwd + {E * 2 * 2} bwd + shared 2 + 4 + gate 3 = "
          f"{E * 6 + 6 + 3}")
    print(f"    fast gmm w13     routed 2 + 4 grouped + shared 2 + 4 + gate 3 = 15")
    print(f"    per step (x{n_layers} layers): "
          f"ref {(E * 9 + 9 + 3) * n_layers}, fast loop {(E * 6 + 6 + 3) * n_layers}, "
          f"fast gmm {15 * n_layers}")


# ------------------------------------------------------------------------------ T5 scatter A/B
def test_scatter() -> None:
    print("\n== T5 output scatter strategies (value/grad + CPU micro-bench) ==", flush=True)
    torch.manual_seed(0)
    T, k, dim, E = 256, 2, 64, 16
    idx = torch.randint(0, E, (T, k))
    order, tok, counts = fm._sort_tokens(idx, E)
    base = torch.randn(T * k, dim, dtype=torch.bfloat16)
    outs = [base.clone().requires_grad_(True) for _ in range(3)]
    modes = ("permute", "index_add", "scatter_add")
    ys, gs = [], []
    for mode, o in zip(modes, outs):
        y = fm._scatter_topk(o, tok, order, T, k, dim, mode=mode)
        y.square().sum().backward()
        ys.append(y.detach())
        gs.append(o.grad.detach())
    same_y = all(torch.equal(ys[0], y) for y in ys[1:])
    same_g = all(torch.equal(gs[0], g) for g in gs[1:])
    check("scatter modes agree in value (fp32 accumulate)", same_y,
          f"maxdiff={max((ys[0] - y).abs().max().item() for y in ys[1:]):.1e}")
    check("scatter modes agree in gradient", same_g,
          f"maxdiff={max((gs[0] - g).abs().max().item() for g in gs[1:]):.1e}")

    # same-count sanity: counts comes from scatter_add_, not bincount
    expect = torch.tensor([int((idx == e).sum()) for e in range(E)])
    check("_sort_tokens counts == ones-count", torch.equal(counts.cpu(), expect))

    # CPU micro-bench at d12 scale (T=8192, dim=384, bf16) - the GPU decision is Lead's
    T2, dim2 = 8192, 384
    idx2 = torch.randint(0, E, (T2, 2))
    order2, tok2, _ = fm._sort_tokens(idx2, E)
    outs2 = torch.randn(T2 * 2, dim2, dtype=torch.bfloat16)

    def timed(fn, n=5):
        for _ in range(2):
            fn()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples) * 1e6

    print(f"  CPU scatter micro-bench (T={T2}, dim={dim2}, bf16, median of 5):")
    times = {mode: timed(lambda m=mode: fm._scatter_topk(
        outs2, tok2, order2, T2, 2, dim2, mode=m)) for mode in modes}
    best = min(times.values())
    for mode in modes:
        print(f"    {mode:12s} {times[mode]:9.1f} us   ({times[mode] / best:.2f}x)")
    print("  -> CPU: three modes are within run-to-run noise (~±30%), no clear winner;")
    print("     default 'auto' = permute on CPU (deterministic) / index_add on CUDA (fewest")
    print("     kernels + least HBM traffic).  GPU A/B is in --bench (Lead runs it).")


# -------------------------------------------------------------------------------- T6 no-sync
def test_no_sync_guard() -> None:
    print("\n== T6 assert_no_sync() guard ==", flush=True)
    t = torch.tensor([1.0, 2.0])
    s = torch.tensor(3.0)
    caught = {}
    for name, fn in (("tolist", lambda: t.tolist()), ("item", lambda: s.item()),
                     ("cpu", lambda: t.cpu()), ("numpy", lambda: t.numpy()),
                     ("float(tensor)", lambda: float(s))):
        try:
            with fm.assert_no_sync():
                fn()
            caught[name] = False
        except RuntimeError:
            caught[name] = True
    check("guard catches host-sync methods", all(caught.values()), str(caught))
    try:
        with fm.assert_no_sync():
            a = torch.randn(8, 8)
            _ = (a @ a).sum() + torch.stack([a, a]).sum()
        allowed = True
    except Exception as exc:  # noqa: BLE001
        allowed = False
        print(f"    unexpected: {type(exc).__name__}: {exc}")
    check("guard allows normal tensor math", allowed)
    check("guard restores Tensor methods", t.tolist() == [1.0, 2.0])

    # the fallback loop genuinely is not graph-safe (host group sizes); show the guard catches it
    ref = build_moe(TINY, torch.float32, std=0.3)
    x = torch.randn(2, 6, TINY["dim"], requires_grad=True)
    try:
        with fm.assert_no_sync():
            fm.fast_moe_forward(ref, x).float().sum().backward()
        loop_rejected = False
    except RuntimeError:
        loop_rejected = True
    check("loop fallback is rejected by assert_no_sync (counts.tolist)", loop_rejected)
    probe_ok, probe_err = fm.probe_grouped_mm_capture()
    check("probe_grouped_mm_capture is a no-op without CUDA",
          probe_ok is False and "CUDA" in probe_err, probe_err)


# ------------------------------------------------------------------------------------ T7 bench
def _cuda_timed(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def bench(args) -> int:
    """GPU: loop vs grouped_mm fwd+bwd, loss/grad parity, no-sync audit, optional capture."""
    if not torch.cuda.is_available():
        print("--bench requires CUDA")
        return 2
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"b={args.b} seq={args.seq} | torch._grouped_mm={fm._GROUPED_MM is not None}", flush=True)
    cfg = dict(D12)
    old = M.default_dtype
    M.default_dtype = torch.bfloat16
    try:
        moe = M.MoE(0, M.ModelArgs(**cfg)).cuda()
    finally:
        M.default_dtype = old
    init_random_weights(moe, std=0.02, seed=0)
    T = args.b * args.seq
    xx = torch.randn(args.b, args.seq, cfg["dim"], device="cuda", dtype=torch.bfloat16,
                     requires_grad=True)
    params = [p for p in moe.parameters() if p.requires_grad]
    has_gmm = fm._GROUPED_MM is not None
    ok = True
    gmm_capturable, gmm_err = fm.probe_grouped_mm_capture()
    print(f"torch._grouped_mm available={has_gmm} | capturable={gmm_capturable}"
          + (f" ({gmm_err[:140]})" if gmm_err else ""))

    def zero_grads():
        grads = [p.grad for p in params if p.grad is not None]
        if xx.grad is not None:
            grads.append(xx.grad)
        if grads:
            torch._foreach_zero_(grads)

    def fwd():
        with torch.no_grad():
            return fm.fast_moe_forward(moe, xx)

    def step():
        y = fm.fast_moe_forward(moe, xx)
        loss = y.float().square().mean()
        zero_grads()
        loss.backward()
        return loss

    # loss / grad parity from the same weights + same input
    print("\n== parity: loop vs grouped_mm (bf16, real d12 shape) ==")
    fm.configure(force_loop=True)
    for _ in range(2):
        step()
    zero_grads()
    y_loop = fm.fast_moe_forward(moe, xx)
    loss_loop = y_loop.float().square().mean()
    loss_loop.backward()
    g_loop = {n: (p.grad.detach().clone() if p.grad is not None else None)
              for n, p in moe.named_parameters()}
    l_loop = loss_loop.item()
    zero_grads()

    if has_gmm:
        fm.configure(force_loop=False)
        if fm.path_for(moe, xx.view(T, -1)) != "grouped_mm":
            print("  grouped_mm not selected (dtype/device/quantized weights) - skipping gmm parity")
            has_gmm = False
        else:
            for _ in range(2):
                step()
            zero_grads()
            y_gmm = fm.fast_moe_forward(moe, xx)
            loss_gmm = y_gmm.float().square().mean()
            loss_gmm.backward()
            l_gmm = loss_gmm.item()
            e_y = rel_err(y_gmm, y_loop)
            e_l = abs(l_gmm - l_loop) / max(abs(l_loop), 1e-12)
            worst, worst_name = 0.0, ""
            for n, p in moe.named_parameters():
                if g_loop[n] is None:
                    continue
                e = rel_err(p.grad, g_loop[n])
                if e > worst:
                    worst, worst_name = e, n
            print(f"  y rel={e_y:.3e}  loss {l_loop:.6f} vs {l_gmm:.6f} (rel {e_l:.3e})  "
                  f"worst grad rel={worst:.3e}@{worst_name}")
            parity_ok = e_y < 2e-2 and e_l < 2e-2 and worst < 2e-2
            print(f"  -> {'PASS' if parity_ok else 'FAIL'} (bf16 tol 2e-2)")
            ok = ok and parity_ok
            zero_grads()
    if not has_gmm:
        print("  torch._grouped_mm unavailable - only the loop path can run")

    # timing: eager loop / grouped_mm / static-routing loop
    print("\n== timing (one MoE layer, fwd / fwd+bwd) ==")
    for mode in (["loop", "gmm"] if has_gmm else ["loop"]):
        fm.configure(force_loop=(mode == "loop"))
        t_fwd = _cuda_timed(fwd)
        t_step = _cuda_timed(step)
        print(f"  {mode:4s}: fwd {t_fwd:7.3f} ms | fwd+bwd {t_step:7.3f} ms  "
              f"| x12 layers: fwd {12 * t_fwd:6.1f} / fwd+bwd {12 * t_step:6.1f} ms")
    fm.configure(force_loop=False)

    # grouped_mm audit: informational; torch 2.12's op itself is not capturable
    if has_gmm:
        print("\n== assert_no_sync audit (grouped_mm fwd+bwd) ==")
        zero_grads()
        try:
            path = fm.audit_no_sync(moe, xx)
            print(f"  PASS  path={path}  no tolist/item/cpu/dynamic-shape ops in our fwd+bwd")
        except RuntimeError as exc:
            print(f"  FAIL  {exc}")
            if gmm_capturable:
                ok = False      # our code syncs even though torch's op can be captured
            else:
                print("  (known torch limitation: torch._grouped_mm host-copies offs; "
                      "graph capture uses the static-routing loop below)")
        zero_grads()

    # ---------------------------------------------------------------- static routing
    print("\n== static-routing fused loop (baked group sizes; no .tolist() in fwd/bwd) ==")
    sizes = fm.record_group_sizes(moe, xx)
    fm.install_static_routing(moe, sizes)
    print(f"  group sizes: sum={sum(sizes)} min={min(sizes)} max={max(sizes)} "
          f"| graph_safe={fm.graph_safe(moe, xx)}")
    zero_grads()
    y_static = fm.fast_moe_forward(moe, xx)
    loss_static = y_static.float().square().mean()
    loss_static.backward()
    l_static = loss_static.item()
    e_y = rel_err(y_static, y_loop)
    e_l = abs(l_static - l_loop) / max(abs(l_loop), 1e-12)
    worst, worst_name = 0.0, ""
    for n, p_ in moe.named_parameters():
        if g_loop[n] is None or p_.grad is None:
            continue
        e = rel_err(p_.grad, g_loop[n])
        if e > worst:
            worst, worst_name = e, n
    static_ok = e_y < 1e-3 and e_l < 1e-3 and worst < 1e-3
    print(f"  vs eager loop: y rel={e_y:.3e}  loss rel={e_l:.3e}  worst grad rel={worst:.3e}"
          f"@{worst_name}  -> {'PASS' if static_ok else 'FAIL'}")
    ok = ok and static_ok
    zero_grads()
    t_fwd = _cuda_timed(fwd)
    t_step = _cuda_timed(step)
    print(f"  static: fwd {t_fwd:7.3f} ms | fwd+bwd {t_step:7.3f} ms  "
          f"| x12 layers: fwd {12 * t_fwd:6.1f} / fwd+bwd {12 * t_step:6.1f} ms")
    zero_grads()
    try:
        fm.audit_no_sync(moe, xx)
        print("  assert_no_sync: PASS (fwd+bwd are host-sync free)")
    except RuntimeError as exc:
        print(f"  assert_no_sync: FAIL {exc}")
        ok = False
    zero_grads()

    # scatter strategies on the GPU
    print("\n== GPU scatter micro-bench (T*k rows, dim=384, bf16) ==")
    idx = torch.randint(0, cfg["n_routed_experts"], (T, cfg["n_activated_experts"]), device="cuda")
    order, tok, _ = fm._sort_tokens(idx, cfg["n_routed_experts"])
    outs = torch.randn(T * cfg["n_activated_experts"], cfg["dim"], device="cuda",
                       dtype=torch.bfloat16)
    for mode in ("permute", "index_add", "scatter_add"):
        us = _cuda_timed(lambda m=mode: fm._scatter_topk(
            outs, tok, order, T, cfg["n_activated_experts"], cfg["dim"], mode=m), iters=20) * 1e3
        print(f"  {mode:12s} {us:8.1f} us")

    # optional CUDA-graph capture of the static-routing fwd+bwd
    graph_status = "not-requested"
    if args.capture:
        print("\n== CUDA graph capture: static-routing fused loop fwd+bwd ==")

        def body():
            y = fm.fast_moe_forward(moe, xx)
            loss = y.float().square().mean()
            zero_grads()
            loss.backward()

        try:
            for _ in range(3):
                body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            graph.replay()
            torch.cuda.synchronize()
            ms = _cuda_timed(lambda: graph.replay(), iters=20)
            print(f"  PASS  replay {ms:.3f} ms/layer (x12 = {12 * ms:.1f} ms/step)")
            graph_status = "verified"
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            graph_status = "failed"
            ok = False
        print("  note: grouped_mm captures:" + (" yes" if gmm_capturable else
              " no (torch host-copies offs) -> static routing is the graph-safe path"))

    print(f"\nBENCH eager {'PASS' if ok else 'FAIL'} | graph {graph_status}")
    return 0 if ok else 1


# -------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="CUDA only: loop vs gmm timing/parity")
    ap.add_argument("--capture", action="store_true", help="with --bench: try CUDA graph capture")
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()
    if args.bench:
        return bench(args)

    torch.manual_seed(0)
    print("=" * 96)
    print("fast_moe tests | CPU parity + graph-safety | torch", torch.__version__)
    print("=" * 96)
    test_parity()
    test_image_mask()
    test_static_routing()
    test_install()
    test_op_counts()
    test_scatter()
    test_no_sync_guard()

    failed = [r for r in _RESULTS if not r[1]]
    print("\n" + "=" * 96)
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks PASS")
    print(f"  max relative error: fp32 {_MAXERR[torch.float32]:.2e} (tol 1e-5) | "
          f"bf16 {_MAXERR[torch.bfloat16]:.2e} (tol 2e-2)")
    if failed:
        print("  FAILED:")
        for name, _ok, detail in failed:
            print(f"    - {name}  {detail}")
    print("=" * 96)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
