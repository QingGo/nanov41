"""CPU tests for ``tools/trace_attrib.py`` (chrome-trace kernel -> record_function attribution).

The parser is pure-Python (no torch/CUDA), so these tests build small synthetic chrome traces and
check the three attribution paths:
  1. GPU user annotations on the stream timeline (``gpu_user_annotation``);
  2. kernel -> launching CPU op via ``External id`` -> enclosing annotation;
  3. plain interval containment fallback.
Plus nesting (innermost wins), interleaved streams, iters division and the CLI.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("trace_attrib", ROOT / "tools" / "trace_attrib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ta = _load()


# --------------------------------------------------------------------------- synthetic events
def ev(pid, tid, ts, dur, name, cat, args=None):
    out = {"ph": "X", "pid": pid, "tid": tid, "ts": ts, "dur": dur, "name": name, "cat": cat}
    if args:
        out["args"] = args
    return out


def ann(pid, tid, ts, dur, name):
    return ev(pid, tid, ts, dur, name, "user_annotation")


def gpu_ann(pid, tid, ts, dur, name):
    return ev(pid, tid, ts, dur, name, "gpu_user_annotation")


def cpu_op(pid, tid, ts, dur, name):
    return ev(pid, tid, ts, dur, name, "cpu_op")


def launch(pid, tid, ts, dur, ext_id):
    return ev(pid, tid, ts, dur, "cudaLaunchKernel", "cuda_runtime", {"External id": ext_id})


def kernel(pid, tid, ts, dur, name, ext_id=None):
    args = {"External id": ext_id} if ext_id is not None else None
    return ev(pid, tid, ts, dur, name, "kernel", args)


def _trace(events):
    return {"traceEvents": [{"ph": "M", "name": "process_name", "pid": 0}] + events}


# --------------------------------------------------------------------------- 1. external-id path
def test_external_id_chain_and_aggregation():
    events = [
        ann(1, 1, 1000, 600, "proj/head_ce"),
        cpu_op(1, 1, 1100, 100, "aten::mm"),
        launch(1, 1, 1200, 10, 1),
        kernel(0, 7, 1215, 50, "cutlass_head_fwd", 1),
        cpu_op(1, 1, 1300, 100, "aten::mm"),
        launch(1, 1, 1400, 10, 2),
        kernel(0, 7, 1415, 100, "cutlass_head_dX", 2),
        ann(1, 1, 2000, 300, "proj/inner.layers.0.attn.wq_b"),
        launch(1, 1, 2100, 10, 3),
        kernel(0, 7, 2110, 30, "cutlass_wqb_fwd", 3),
        # unassigned: launched outside any annotation range
        cpu_op(1, 1, 3000, 100, "aten::copy_"),
        launch(1, 1, 3050, 10, 4),
        kernel(0, 7, 3060, 20, "copy_kernel", 4),
    ]
    res = ta.attribute(_trace(events), iters=1)
    assert res["modules"]["head_ce"] == 0.15          # 50 + 100 us
    assert res["modules"]["inner.layers.0.attn.wq_b"] == 0.03
    assert res["modules"]["<other>"] == 0.02
    assert res["total_kernel_ms"] == 0.2
    assert res["unassigned_ms"] == 0.02
    assert res["modes"]["external_id"] == 3
    assert res["modes"]["unassigned"] == 1
    assert res["n_kernels"] == 4


# --------------------------------------------------------------------------- 2. nesting
def test_nested_annotations_innermost_wins():
    events = [
        ann(1, 1, 0, 1000, "proj/outer"),
        ann(1, 1, 100, 200, "proj/inner"),
        cpu_op(1, 1, 150, 50, "aten::mm"),
        launch(1, 1, 180, 5, 11),
        kernel(0, 1, 190, 40, "k_inner", 11),
        # a second kernel in the outer range but outside the inner one
        launch(1, 1, 500, 5, 12),
        kernel(0, 1, 510, 10, "k_outer", 12),
    ]
    res = ta.attribute(_trace(events), iters=1)
    assert res["modules"]["inner"] == 0.04
    assert res["modules"]["outer"] == 0.01
    assert res["modes"]["external_id"] == 2


# --------------------------------------------------------------------------- 3. GPU annotations
def test_gpu_annotation_stream_attribution():
    events = [
        gpu_ann(0, 7, 4000, 300, "proj/hc_mixes"),
        kernel(0, 7, 4050, 100, "triton_hc_kernel"),
        gpu_ann(0, 8, 4000, 300, "proj/engram"),
        kernel(0, 8, 4020, 60, "engram_kernel"),
        kernel(0, 9, 4000, 500, "outside_kernel"),   # different stream, no annotation
    ]
    res = ta.attribute(_trace(events), iters=1)
    assert res["modules"]["hc_mixes"] == 0.1
    assert res["modules"]["engram"] == 0.06
    assert res["modules"]["<other>"] == 0.5
    assert res["modes"]["gpu_annotation"] == 2
    assert res["unassigned_ms"] == 0.5
    assert res["unassigned_pct"] == round(100.0 * 500 / 660, 2)


def test_gpu_annotations_interleaved_streams():
    events = [
        gpu_ann(0, 7, 0, 1000, "proj/a"),
        gpu_ann(0, 8, 0, 1000, "proj/b"),
        kernel(0, 7, 100, 50, "k_a"),
        kernel(0, 8, 120, 50, "k_b"),
        kernel(0, 7, 300, 50, "k_a2"),
    ]
    res = ta.attribute(_trace(events), iters=1)
    assert res["modules"]["a"] == 0.1
    assert res["modules"]["b"] == 0.05


# --------------------------------------------------------------------------- 4. containment fallback
def test_containment_fallback_without_external_id():
    events = [
        ann(1, 1, 5000, 200, "proj/fallback"),
        kernel(0, 3, 5100, 10, "no_id_kernel"),
    ]
    res = ta.attribute(_trace(events), iters=1)
    assert res["modules"]["fallback"] == 0.01
    assert res["modes"]["containment"] == 1


# --------------------------------------------------------------------------- 5. iters division
def test_iters_division():
    events = [
        ann(1, 1, 0, 100, "proj/x"),
        launch(1, 1, 20, 5, 1),
        kernel(0, 1, 25, 40, "k", 1),
        launch(1, 1, 40, 5, 2),
        kernel(0, 1, 45, 40, "k", 2),
    ]
    res1 = ta.attribute(_trace(events), iters=1)
    res2 = ta.attribute(_trace(events), iters=2)
    assert res1["total_kernel_ms"] == 0.08
    assert res1["modules"]["x"] == 0.08
    assert res2["total_kernel_ms"] == 0.04
    assert res2["modules"]["x"] == 0.04
    assert res2["kernel_to_module"][0]["calls_per_iter"] == 1.0


# --------------------------------------------------------------------------- 6. degenerate inputs
def test_no_annotations_all_unassigned():
    events = [launch(1, 1, 10, 5, 1), kernel(0, 1, 20, 100, "k", 1)]
    res = ta.attribute(_trace(events), iters=1)
    assert res["assigned_ms"] == 0.0
    assert res["unassigned_ms"] == 0.1
    assert res["unassigned_pct"] == 100.0


def test_empty_trace():
    res = ta.attribute({"traceEvents": []})
    assert res["n_kernels"] == 0 and res["total_kernel_ms"] == 0.0


# --------------------------------------------------------------------------- 7. CLI
def test_cli_writes_json(tmp_path):
    events = [
        ann(1, 1, 1000, 500, "proj/head_ce"),
        launch(1, 1, 1200, 10, 1),
        kernel(0, 7, 1215, 100, "k", 1),
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(_trace(events)))
    out = tmp_path / "attr.json"
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "trace_attrib.py"),
                           "--trace", str(trace), "--iters", "2", "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out.read_text())
    assert res["modules"]["head_ce"] == 0.05
    assert res["iters"] == 2
