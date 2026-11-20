#!/usr/bin/env python3
"""R93 — full-validation BPB evaluation with per-window records + subsample-error study.

Scores every window of a held-out eval domain (default: the full 56.8M-token ClimbMix val
shard), saves per-window nll/bytes, and derives:
  * full bpb (all windows),
  * K random 3M-token subsets -> sampling error of the 3M-subset estimate (mean/std/deviation
    from full),
  * contiguous segments (e.g. 10 equal blocks) -> within-domain heterogeneity,
  * bootstrap CI on the full bpb.

Usage:
  NANOV41_DS_TOKENIZER_JSON=<tok> python3 tools/r93_fullval_eval.py \
      --panel /root/autodl-tmp/nanochat/p0_eval_40k --domain climbmix_val \
      --ckpt /root/autodl-tmp/nanochat/runs/p1/p1_8b.pt --batch-size 8 \
      --out runs/r93/fullval/p1_raw.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.eval.harness import token_nll_batch  # noqa: E402
from nanov41.eval.provider import ModelProvider  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--domain", default="climbmix_val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ce-chunk", type=int, default=256)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-subsets", type=int, default=8)
    ap.add_argument("--subset-tokens", type=int, default=3_000_000)
    ap.add_argument("--n-segments", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = pathlib.Path(args.panel)
    man = json.loads((panel / "manifest.json").read_text())
    dom = man["domains"][args.domain]
    tb = np.load(panel / man["token_bytes"]["path"])
    shard_files = sorted((panel / args.domain).glob("*.tokens.npy"))
    if not shard_files:
        raise SystemExit(f"no tokens.npy under {panel / args.domain}")
    tokens = np.concatenate([np.load(f) for f in shard_files])
    print(f"[fullval] tokens={tokens.size} shards={len(shard_files)} domain={args.domain}", flush=True)

    win = args.window
    n_win = tokens.size // win          # trailing partial window dropped
    windows = tokens[: n_win * win].reshape(n_win, win).astype(np.int64)
    print(f"[fullval] windows={n_win} scored_tokens={(win-1)*n_win}", flush=True)

    prov = ModelProvider(checkpoint=args.ckpt, dtype=args.dtype, device=args.device)
    nll = np.zeros((n_win, win - 1), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n_win, args.batch_size):
        end = min(start + args.batch_size, n_win)
        x = torch.from_numpy(windows[start:end])
        nll[start:end] = token_nll_batch(prov, x, ce_chunk=args.ce_chunk)
        if (start // args.batch_size) % 20 == 0:
            done = end * (win - 1)
            print(f"[fullval] {done}/{n_win*(win-1)} tokens ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0
    del prov
    torch.cuda.empty_cache()

    tgt = windows[:, 1:]
    bytes_win = tb[tgt].sum(axis=1).astype(np.float64)          # bytes per window
    bits_win = nll.sum(axis=1).astype(np.float64) / math.log(2)  # bits per window
    tot_bits, tot_bytes = bits_win.sum(), bytes_win.sum()
    full_bpb = tot_bits / tot_bytes
    # tokens (predictions) counted for the subset study
    w_tokens = win - 1

    rng = np.random.default_rng(args.seed)
    subset_bpbs = []
    for _ in range(args.n_subsets):
        k = max(1, args.subset_tokens // w_tokens)
        idx = rng.choice(n_win, size=min(k, n_win), replace=False)
        subset_bpbs.append(float(bits_win[idx].sum() / bytes_win[idx].sum()))
    subset_bpbs = np.array(subset_bpbs)

    seg_bpbs = []
    bounds = np.linspace(0, n_win, args.n_segments + 1).astype(int)
    for i in range(args.n_segments):
        a, b = bounds[i], bounds[i + 1]
        seg_bpbs.append(float(bits_win[a:b].sum() / bytes_win[a:b].sum()))

    boot = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n_win, n_win)
        boot.append(float(bits_win[idx].sum() / bytes_win[idx].sum()))
    boot = np.array(boot)

    report = {
        "ckpt": args.ckpt, "domain": args.domain, "window": win, "batch_size": args.batch_size,
        "elapsed_s": elapsed, "windows": int(n_win), "scored_tokens": int((win - 1) * n_win),
        "bytes": float(tot_bytes), "nll_nats": float(bits_win.sum() * math.log(2)),
        "bpb": full_bpb, "bits_per_token": float(tot_bits / ((win - 1) * n_win)),
        "nats_per_token": float(bits_win.sum() * math.log(2) / ((win - 1) * n_win)),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "subset_tokens": args.subset_tokens,
        "subset_bpbs": [float(x) for x in subset_bpbs],
        "subset_mean": float(subset_bpbs.mean()), "subset_std": float(subset_bpbs.std(ddof=1)),
        "subset_vs_full_diff": [float(x - full_bpb) for x in subset_bpbs],
        "segment_bpbs": seg_bpbs,
        "segment_std": float(np.std(seg_bpbs, ddof=1)),
        "write_per_window": True,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".windows.npz"),
                        bits=bits_win.astype(np.float32), bytes=bytes_win.astype(np.float32),
                        bpp=None)
    out.write_text(json.dumps(report, indent=1))
    print(f"[fullval] full bpb={full_bpb:.5f} CI95=[{report['bootstrap_ci95'][0]:.5f},"
          f"{report['bootstrap_ci95'][1]:.5f}] subset3M mean={subset_bpbs.mean():.5f} "
          f"std={subset_bpbs.std(ddof=1):.5f} (n={len(subset_bpbs)}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
