#!/usr/bin/env python3
"""CPU parser for torch/kineto chrome traces: CUDA kernel -> record_function module attribution.

Why: the 2026-11-10 `--gpu-attrib` run showed that kineto's ``FunctionEvent.cpu_parent`` chain does
not link CUDA kernel events to the enclosing ``record_function`` ranges in torch 2.14 (everything
fell into ``<other>``, ``with_stack`` is empty for kernels too), leaving the biggest single kernel
family (26x cutlass ``128x256`` = 10.08 ms/iter) unattributed.  A chrome trace is a flat timeline,
so this parser reconstructs the nesting itself:

1. prefer **GPU user annotations** (``cat == "gpu_user_annotation"``): they live on the GPU stream
   timeline, so a kernel is assigned to the innermost annotation interval containing it;
2. else map the kernel to its **launching CPU op** through the ``External id``/``correlation`` id
   that both events carry, then walk the containing CPU intervals (same pid/tid, innermost first)
   outwards to the nearest annotation range;
3. else fall back to plain interval containment on the CPU timeline.

Annotations are recognised by ``cat in {user_annotation, gpu_user_annotation}`` or by name prefix
(``proj/``, ``proj_bwd/``, ``head_ce``, ``hc_mixes``, ``ref_linear``).  Output: per-module and
kernel->module ms/iter, plus the unassigned share (the number that decides the next kernel work).

Pure Python (stdlib only) so it runs/tests on CPU; no torch import needed.

Usage:
  python3 tools/trace_attrib.py --trace /tmp/attrib_trace.json --iters 3 \
      --out runs/local-notes/trace_attrib.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import sys
from collections import defaultdict

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_RANGE_CATS = {"cpu_op", "cuda_runtime", "cuda_driver", "user_annotation", "python_function",
                  "gpu_user_annotation"}
DEFAULT_PREFIXES = ("proj/", "proj_bwd/", "head_ce", "hc_mixes", "ref_linear")
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
ID_KEYS = ("External id", "external id", "correlation", "cuda_correlation")


# --------------------------------------------------------------------------- helpers
def _module_name(name: str) -> str:
    if name.startswith("proj_bwd/"):
        return "bwd:" + name[len("proj_bwd/"):]
    if name.startswith("proj/"):
        return name[len("proj/"):]
    return name


def _is_annotation(ev: dict, prefixes: tuple[str, ...]) -> bool:
    cat = ev.get("cat", "")
    if cat in ("user_annotation", "gpu_user_annotation"):
        return True
    name = str(ev.get("name", ""))
    return any(name.startswith(p) for p in prefixes)


def _external_ids(ev: dict) -> list:
    args = ev.get("args") or {}
    ids = []
    for key in ID_KEYS:
        val = args.get(key)
        if val is not None:
            ids.append(val)
    return ids


class _ThreadIndex:
    """Sorted (start) index of CPU timeline ranges per (pid, tid)."""

    def __init__(self, events: list[dict]):
        self.by_thread: dict[tuple, list[dict]] = defaultdict(list)
        for ev in events:
            self.by_thread[(ev.get("pid"), ev.get("tid"))].append(ev)
        self.starts: dict[tuple, list[float]] = {}
        self.prefix_max_end: dict[tuple, list[float]] = {}
        for key, lst in self.by_thread.items():
            lst.sort(key=lambda e: float(e.get("ts", 0.0)))
            self.starts[key] = [float(e["ts"]) for e in lst]
            pref: list[float] = []
            max_end = -1.0
            for e in lst:
                max_end = max(max_end, float(e["ts"]) + float(e.get("dur", 0.0)))
                pref.append(max_end)
            self.prefix_max_end[key] = pref

    def containing(self, pid, tid, ts: float, *, max_scan: int = 20000) -> list[dict]:
        """All ranges on this thread containing ``ts`` (exact: stops when no earlier range can
        reach ts, using a prefix-max-end index — a fixed 'lookback' window would miss long
        annotations such as a 600us head_ce range around a launch at its right edge)."""
        key = (pid, tid)
        lst = self.by_thread.get(key)
        if not lst:
            return []
        idx = bisect.bisect_right(self.starts[key], ts) - 1
        out = []
        scanned = 0
        i = idx
        pref = self.prefix_max_end[key]
        while i >= 0 and scanned < max_scan:
            if pref[i] < ts:
                break
            ev = lst[i]
            start = float(ev["ts"])
            dur = float(ev.get("dur", 0.0))
            if start <= ts <= start + dur:
                out.append(ev)
            scanned += 1
            i -= 1
        return out

    def innermost_all_containing(self, pid, tid, ts: float) -> list[dict]:
        cands = self.containing(pid, tid, ts)
        cands.sort(key=lambda e: float(e.get("dur", 0.0)))
        return cands


def _find_launcher(kernel: dict, id_map: dict, threads: _ThreadIndex) -> tuple[dict | None, str]:
    """Return (launching CPU event, mode) for a kernel event."""
    ids = _external_ids(kernel)
    for eid in ids:
        cands = id_map.get(eid)
        if cands:
            cands = sorted(cands, key=lambda e: (0 if e.get("cat") in LAUNCH_CATS else 1,
                                                 float(e.get("ts", 0.0))))
            return cands[0], "external_id"
    # fallback: innermost containing CPU range on any thread at the kernel start time
    ts = float(kernel.get("ts", 0.0))
    best = None
    for (pid, tid) in threads.by_thread:
        cands = threads.containing(pid, tid, ts, max_scan=200)
        if not cands:
            continue
        cands.sort(key=lambda e: (0 if e.get("cat") in LAUNCH_CATS else 1,
                                  float(e.get("dur", 0.0))))
        if best is None or float(cands[0].get("dur", 0.0)) < float(best.get("dur", 0.0)):
            best = cands[0]
    return (best, "containment") if best is not None else (None, "unassigned")


def _annotation_for_event(ev: dict, threads: _ThreadIndex,
                          prefixes: tuple[str, ...]) -> dict | None:
    """Walk outwards from a CPU event to the nearest enclosing annotation range."""
    pid, tid = ev.get("pid"), ev.get("tid")
    cands = threads.innermost_all_containing(pid, tid, float(ev.get("ts", 0.0)))
    best = None
    for parent in cands:
        if parent is ev:
            continue
        if _is_annotation(parent, prefixes):
            dur = float(parent.get("dur", 0.0))
            if best is None or dur < float(best.get("dur", 0.0)):
                best = parent
    return best


# --------------------------------------------------------------------------- core
def attribute(trace: dict | list, *, iters: int = 1, prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
              top: int = 50) -> dict:
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    evs = [e for e in events
           if isinstance(e, dict) and e.get("ph") == "X"
           and isinstance(e.get("ts"), (int, float)) and isinstance(e.get("dur"), (int, float))
           and float(e["dur"]) > 0]
    kernels = [e for e in evs if e.get("cat") in KERNEL_CATS]
    gpu_ann = [e for e in evs if e.get("cat") == "gpu_user_annotation"]
    cpu = [e for e in evs if e.get("cat") not in KERNEL_CATS | {"gpu_user_annotation"}]

    threads = _ThreadIndex(cpu)
    gpu_threads = _ThreadIndex(gpu_ann)
    id_map: dict = defaultdict(list)
    for ev in cpu:
        for eid in _external_ids(ev):
            id_map[eid].append(ev)

    iters = max(1, int(iters))
    mod_us: dict[str, float] = defaultdict(float)
    kernel_mod_us: dict[tuple[str, str], float] = defaultdict(float)
    kernel_mod_calls: dict[tuple[str, str], int] = defaultdict(int)
    unassigned_us: dict[str, float] = defaultdict(float)
    unassigned_launcher: dict[str, str] = {}
    modes = {"gpu_annotation": 0, "external_id": 0, "containment": 0, "unassigned": 0}
    total_us = 0.0

    for k in kernels:
        dur = float(k["dur"])
        total_us += dur
        name = str(k.get("name", "?"))[:150]
        module = None
        # 1) GPU-side user annotation on the same stream
        if gpu_ann:
            mids = float(k["ts"]) + dur / 2.0
            cands = gpu_threads.innermost_all_containing(k.get("pid"), k.get("tid"), mids)
            for parent in cands:
                if _is_annotation(parent, prefixes):
                    module = _module_name(str(parent.get("name", "")))
                    modes["gpu_annotation"] += 1
                    break
        # 2) launching CPU op -> enclosing annotation
        if module is None:
            launcher, mode = _find_launcher(k, id_map, threads)
            if launcher is not None:
                if _is_annotation(launcher, prefixes):
                    # no launch op recorded; the innermost containing range is already an annotation
                    module = _module_name(str(launcher.get("name", "")))
                    modes["containment"] += 1
                    ann = launcher
                else:
                    ann = _annotation_for_event(launcher, threads, prefixes)
                if ann is not None:
                    if module is None:
                        module = _module_name(str(ann.get("name", "")))
                        modes["external_id" if mode == "external_id" else "containment"] += 1
                else:
                    modes["unassigned"] += 1
                    unassigned_us[name] += dur
                    unassigned_launcher.setdefault(name, str(launcher.get("name", "?"))[:80])
            else:
                modes["unassigned"] += 1
                unassigned_us[name] += dur
                unassigned_launcher.setdefault(name, "")
        if module is None:
            module = "<other>"
        mod_us[module] += dur
        kernel_mod_us[(name, module)] += dur
        kernel_mod_calls[(name, module)] += 1

    def _ms(us: float) -> float:
        return round(us / 1e3 / iters, 4)

    rows = sorted(({"kernel": k, "module": m, "ms_per_iter": _ms(v),
                    "calls_per_iter": round(n / iters, 2)}
                   for (k, m), v in kernel_mod_us.items() for n in [kernel_mod_calls[(k, m)]]),
                  key=lambda r: -r["ms_per_iter"])[:top]
    unassigned = sorted(({"kernel": k, "launcher": unassigned_launcher.get(k, ""),
                          "ms_per_iter": _ms(v)} for k, v in unassigned_us.items()),
                        key=lambda r: -r["ms_per_iter"])[:top]
    assigned_us = total_us - sum(unassigned_us.values())
    return {
        "iters": iters,
        "n_events": len(evs), "n_kernels": len(kernels),
        "n_cpu_ranges": len(cpu), "n_gpu_annotations": len(gpu_ann),
        "total_kernel_ms": _ms(total_us),
        "assigned_ms": _ms(assigned_us),
        "unassigned_ms": _ms(total_us - assigned_us),
        "unassigned_pct": round(100.0 * (total_us - assigned_us) / max(total_us, 1e-9), 2),
        "modes": modes,
        "modules": {m: _ms(v) for m, v in sorted(mod_us.items(), key=lambda kv: -kv[1])},
        "kernel_to_module": rows,
        "top_unassigned": unassigned,
        "prefixes": list(prefixes),
    }


# --------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="attribute CUDA kernels in a chrome trace to "
                                             "record_function ranges (CPU-only parser)")
    ap.add_argument("--trace", required=True, help="chrome trace JSON from prof.export_chrome_trace")
    ap.add_argument("--iters", type=int, default=1, help="profiled iterations (divide sums by this)")
    ap.add_argument("--out", default="", help="optional output JSON path")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--prefix", action="append", default=None,
                    help="annotation name prefix (repeatable; default proj/ proj_bwd/ head_ce hc_mixes ref_linear)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    prefixes = tuple(args.prefix) if args.prefix else DEFAULT_PREFIXES
    trace_path = pathlib.Path(args.trace)
    if not trace_path.exists():
        raise SystemExit(f"trace not found: {trace_path}")
    trace = json.loads(trace_path.read_text())
    res = attribute(trace, iters=args.iters, prefixes=prefixes, top=args.top)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        if not args.quiet:
            print(f"[trace-attrib] wrote {out}")
    if not args.quiet:
        print(f"[trace-attrib] kernels={res['n_kernels']} cpu_ranges={res['n_cpu_ranges']} "
              f"gpu_annotations={res['n_gpu_annotations']} total={res['total_kernel_ms']:.3f} ms/iter")
        print(f"[trace-attrib] assigned={res['assigned_ms']:.3f} ms/iter "
              f"unassigned={res['unassigned_ms']:.3f} ({res['unassigned_pct']:.1f}%) modes={res['modes']}")
        print("\nmodules:")
        for m, v in list(res["modules"].items())[:20]:
            print(f"  {v:8.3f} ms/iter  {m}")
        print("\nkernel -> module (top):")
        for r in res["kernel_to_module"][:20]:
            print(f"  {r['ms_per_iter']:8.3f} x{r['calls_per_iter']:5.1f}  {r['module']:34s} {r['kernel'][:80]}")
        print("\nunassigned kernels (top):")
        for r in res["top_unassigned"][:15]:
            print(f"  {r['ms_per_iter']:8.3f}  launcher={r['launcher'][:40]:40s} {r['kernel'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
