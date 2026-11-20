#!/usr/bin/env python3
"""Naive no-KV-cache continuation probe for P2/P2b checkpoints.

This is a smoke test, not a serving engine: the model's forward runs on the full prefix every step
(there is no generate/KV-cache path in this repo yet).  It answers two questions -- does the
checkpoint load and produce finite next-token distributions, and what does a short greedy/sampled
continuation look like after 0.6 B training tokens?

Usage:
    python tools/p2_continue.py --ckpt /tmp/p2b_default.pt \
        --tokenizer /tmp/tok_deepseek-ai_DeepSeek-V4.1-Flash.json --max-new-tokens 24
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--kv-cache", action="store_true", help="use the official incremental KV-cache path")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prompts = args.prompt or [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "人工智能的未来是",
    ]
    from tokenizers import Tokenizer
    from nanov41.eval.provider import ModelProvider

    from nanov41.generate import generate_kv, generate_no_cache
    tok = Tokenizer.from_file(args.tokenizer)
    t0 = time.time()
    provider = ModelProvider(checkpoint=args.ckpt, dtype="float32", device="cpu")
    load_s = time.time() - t0
    print(f"[load] {args.ckpt} in {load_s:.1f}s | params={provider.n_params/1e6:.1f}M "
          f"table={provider.engram_table_params/1e6:.1f}M engram={provider.engram}", flush=True)

    rows = []
    for prompt in prompts:
        ids_prompt = tok.encode(prompt, add_special_tokens=False).ids
        if not ids_prompt:
            continue
        t0 = time.time()
        fn = generate_kv if args.kv_cache else generate_no_cache
        ids = fn(provider.model, ids_prompt, max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature, top_p=args.top_p)
        dt = time.time() - t0
        text = tok.decode(ids, skip_special_tokens=True)
        new_text = tok.decode(ids[len(ids_prompt):],
                              skip_special_tokens=True)
        row = {"prompt": prompt, "continuation": new_text, "full": text,
               "new_tokens": args.max_new_tokens, "seconds": round(dt, 2),
               "tok_per_s": round(args.max_new_tokens / max(dt, 1e-9), 2)}
        rows.append(row)
        print("\n=== prompt ===\n" + prompt + "\n--- continuation ---\n" + new_text +
              f"\n({row['tok_per_s']} tok/s, {'KV-cache' if args.kv_cache else 'no KV cache'})", flush=True)
    out = args.out or str(pathlib.Path("runs/p2b/continuation_samples.json"))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps({
        "ckpt": args.ckpt, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "samples": rows}, indent=2))
    print(f"\n[written] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
