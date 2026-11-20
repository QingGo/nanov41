#!/usr/bin/env python3
"""Benchmark Engram training overhead on a 5090 (interleaved A/B, eager path).

Measures, in one process and interleaved to cancel clock/thermal drift:

* baseline d12 training step (fused head CE, bf16, eager),
* d12 + Engram forward/backward (same step, sparse grad records cleared),
* d12 + Engram forward/backward + sparse Adam row update,

and writes a JSON evidence file with medians, overhead %, peak memory and table stats.

Scope note: this is the **eager** path. Naive `torch.cuda.graph` capture is blocked earlier in
the model by the MoE router's CPU `bincount().tolist()`; graph replay will be measured once Engram
is integrated into the R68/R69 graph-training stack (`graph_moe.py` etc.), not here.

    python tools/bench_engram_gpu.py --config d12 --batch 8 --seq 1024 \\
        --out runs/engram_gpu_5090.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nanov41.nano_engram as ne  # noqa: E402
from nanov41.configs import CONFIGS  # noqa: E402
from nanov41.model import NanoV41, init_random_weights  # noqa: E402
from nanov41.patched_env import import_patched  # noqa: E402
from nanov41.train_utils import fused_head_ce  # noqa: E402


def build(config: str, batch: int, seq: int, engram: bool, seed: int = 0):
    cfg = dict(CONFIGS[config], max_batch_size=batch, max_seq_len=max(seq, CONFIGS[config]["max_seq_len"]))
    tokenizer = None
    if engram:
        ne.install()
        cfg = ne.apply_preset(cfg, config)
        tokenizer = ne.load_tokenizer_adapter(_tokenizer_path())
    torch.manual_seed(seed)
    model = NanoV41(cfg, tokenizer=tokenizer, param_dtype="bf16").cuda()
    init_random_weights(model, seed=seed)
    return model, cfg


def _tokenizer_path() -> str:
    import os

    path = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
    if not path:
        raise SystemExit("set NANOV41_DS_TOKENIZER_JSON to the pinned tokenizer.json")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="d12", choices=["d12", "d16", "d20"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/engram_gpu_5090.json")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    M = import_patched()
    M.ParallelHead.forward = lambda self, x, full_logits=False: x  # fused head+CE path
    ids = torch.randint(0, CONFIGS[args.config]["vocab_size"], (args.batch, args.seq), device="cuda")

    def step(model, opt=None, clear_only=False):
        loss = fused_head_ce(model(ids), model.inner.head.weight, ids, z_loss=1e-4)
        loss.backward()
        if opt is not None:
            opt.step()
        elif clear_only:
            for table in ne.engram_tables(model):
                table._records.clear()
                table.zero_row_grad()
        model.zero_grad(set_to_none=True)

    base, _ = build(args.config, args.batch, args.seq, engram=False, seed=args.seed)
    eng, _ = build(args.config, args.batch, args.seq, engram=True, seed=args.seed)
    tables = ne.engram_tables(eng)
    opt = ne.SparseRowAdam(tables, lr=1e-2)

    for _ in range(args.warmup):
        step(base)
        step(eng, opt=opt)
    torch.cuda.synchronize()

    times = {"baseline": [], "engram": [], "engram_sparse": []}
    for _ in range(args.iters):
        for name, fn in (
            ("baseline", lambda: step(base)),
            ("engram", lambda: step(eng, clear_only=True)),
            ("engram_sparse", lambda: step(eng, opt=opt)),
        ):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times[name].append(start.elapsed_time(end))

    base_ms = statistics.median(times["baseline"])
    eng_ms = statistics.median(times["engram"])
    eng_sparse_ms = statistics.median(times["engram_sparse"])
    report = {
        "schema": "nanov41-engram-gpu-bench/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": args.config,
        "batch": args.batch,
        "seq": args.seq,
        "iters": args.iters,
        "baseline_ms": round(base_ms, 2),
        "engram_ms": round(eng_ms, 2),
        "engram_sparse_ms": round(eng_sparse_ms, 2),
        "engram_overhead_pct": round((eng_ms / base_ms - 1) * 100, 2),
        "engram_sparse_overhead_pct": round((eng_sparse_ms / base_ms - 1) * 100, 2),
        "samples_ms": {k: [round(x, 2) for x in v] for k, v in times.items()},
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        "engram_table_params": ne.engram_param_count(eng),
        "engram_tables": [[int(t.num_embeddings), int(t.dim)] for t in tables],
        "engram_tracked_rows": [int(m.size) for m in opt._moments.values()],
        "path": "eager (fused head CE); CUDA-graph capture blocked by MoE router CPU sync",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
