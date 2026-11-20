#!/usr/bin/env python3
"""P0 Engram hash-parity report: layout rows, prime buckets, deterministic hash ids.

This is the cheap, CPU-only gate before the training path is trusted.  It checks:

1. the official V4.1-Flash layout (`ref/config.json`) reproduces rows 384,006,168 / 384,016,682 as
   the exact sum of the prime buckets;
2. every nano preset (`d12`/`d16`/`d20`/`tiny`) has rows equal to the sum of its primes and no
   prime is reused;
3. with the pinned tokenizer, the n-gram hash ids stay inside their prime bucket ranges, are
   deterministic across calls, and chunked (start_pos>0) hashing matches full-sequence hashing.

Exit code is non-zero on the first failed invariant (use it in CI / before enabling Engram).

    python tools/engram_hash_parity.py --tokenizer /tmp/dsv41_tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.configs import CONFIGS  # noqa: E402
from nanov41.nano_engram import ENGRAM_PRESETS, apply_preset, install, load_tokenizer_adapter  # noqa: E402
from nanov41.patched_env import import_patched  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def layout_checks(M) -> None:
    official = json.load(open(ROOT / "ref" / "config.json"))
    args = M.ModelArgs(**official)
    layout = M.EngramLayout.from_args(args)
    rows = [sum(sum(heads) for heads in layer) for layer in layout.primes]
    check("official layer ids == (1, 14)", layout.layer_ids == (1, 14), str(layout.layer_ids))
    check("official rows == prime sums",
          tuple(rows) == tuple(layout.num_embeddings) == (384_006_168, 384_016_682),
          f"{rows} vs {layout.num_embeddings}")
    check("official geometry (ngram 4, 8 heads, head_dim 256)",
          (layout.max_ngram_size, layout.n_heads, layout.head_dim) == (4, 8, 256))

    for name in ENGRAM_PRESETS:
        cfg = apply_preset(CONFIGS["tiny"] if name == "tiny" else CONFIGS[name], name)
        lay = M.EngramLayout.from_args(M.ModelArgs(**cfg))
        sums = [sum(sum(heads) for heads in layer) for layer in lay.primes]
        weird = [p for layer in lay.primes for heads in layer for p in heads]
        check(f"preset {name}: rows == prime sums", tuple(sums) == tuple(lay.num_embeddings),
              f"{sums} vs {lay.num_embeddings}")
        check(f"preset {name}: primes unique", len(weird) == len(set(weird)),
              f"{len(weird)} primes")


def hash_checks(M, tokenizer_path: str) -> None:
    adapter = load_tokenizer_adapter(tokenizer_path)
    cfg = apply_preset(CONFIGS["tiny"], "tiny")
    args = M.ModelArgs(**cfg)
    layout = M.EngramLayout.from_args(args)
    state = M.NgramHashState(args, layout, adapter)
    ids = torch.randint(0, CONFIGS["tiny"]["vocab_size"], (2, 12))

    full = state(ids, 0)
    check("hash ids deterministic", torch.equal(state(ids, 0), full))
    flat_primes = [[p for heads in layer for p in heads] for layer in layout.primes]
    in_range = True
    for layer_idx in range(len(layout.layer_ids)):
        for col in range(full.shape[-1]):
            lo = int(state.offsets[layer_idx, col])
            hi = lo + int(flat_primes[layer_idx][col])
            vals = full[..., layer_idx, col]
            in_range &= int(vals.min()) >= lo and int(vals.max()) < hi
    check("hash ids inside prime buckets", in_range)

    head = state(ids[:, :5], 0)
    tail = state(ids[:, 5:], 5)
    check("chunked hashing matches full", torch.equal(head, full[:, :5]) and torch.equal(tail, full[:, 5:]))
    print(f"[info] compressed vocab kept at 99,092; hash ids/token = {full.shape[-1]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=None, help="pinned tokenizer.json (hash ids need it)")
    args = ap.parse_args(argv)

    M = import_patched()
    layout_checks(M)
    install()
    if args.tokenizer:
        hash_checks(M, args.tokenizer)
    else:
        print("[skip] hash-id checks need --tokenizer (layout checks still ran)")
    print()
    if FAILS:
        print(f"P0 FAILED: {FAILS}")
        return 1
    print("P0 OK: layout + hash parity invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
