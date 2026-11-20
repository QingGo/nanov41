#!/usr/bin/env python3
"""R95 — build held-out multi-domain eval slices (code / math / arxiv / zh / edu-web) for the
K=40k tokenizer, to complement the existing climbmix_val web domain.

Run on the GPU instance (CPU only; needs `datasets` + network for hf-mirror):
  HF_ENDPOINT=https://hf-mirror.com NANOV41_DS_TOKENIZER_JSON=<tok> \
  python3 tools/r95_build_domain_eval.py [--char-budget 14000000] [--outdir /root/autodl-tmp/nanochat/eval_inputs]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


SPECS = [
    # name, kind, source spec
    ("code_py", "local", ["/root/miniconda3/lib/python3.12", "/root/t141/lib/python3.12/site-packages"]),
    ("math_web", "hf", [("open-web-math/open-web-math", None, "train", "text"),
                        ("EleutherAI/proof-pile-2", "open-web-math", "train", "text")]),
    ("arxiv", "hf", [("ccdv/arxiv-summarization", None, "train", "article"),
                     ("EleutherAI/proof-pile-2", "arxiv", "train", "text")]),
    ("zh_wiki", "hf", [("wikimedia/wikipedia", "20231101.zh", "train", "text")]),
    ("edu_web", "hf", [("HuggingFaceTB/fineweb-edu", "sample-10BT", "train", "text"),
                       ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", "text")]),
]


def local_py_text(paths, budget):
    out = []
    n = 0
    for root in paths:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(t) < 200:
                continue
            out.append(t)
            n += len(t)
            if n >= budget:
                return out, n
    return out, n


def hf_text(cands, budget, log):
    import datasets

    for name, cfg, split, field in cands:
        try:
            log(f"[hf] trying {name} cfg={cfg} split={split} field={field}")
            ds = datasets.load_dataset(name, cfg, split=split, streaming=True)
            out = []
            n = 0
            for rec in ds:
                t = rec.get(field)
                if not isinstance(t, str) or len(t) < 100:
                    continue
                out.append(t)
                n += len(t)
                if n >= budget:
                    break
            if out:
                return out, n, f"{name}:{cfg}"
            log(f"[hf] {name} {cfg} produced no text")
        except Exception as exc:  # noqa: BLE001
            log(f"[hf] {name} cfg={cfg} FAILED: {repr(exc)[:160]}")
    return [], 0, "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-budget", type=int, default=14_000_000)
    ap.add_argument("--outdir", default="/root/autodl-tmp/nanochat/eval_inputs")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    summary = {}
    for name, kind, spec in SPECS:
        t0 = time.time()
        if kind == "local":
            texts, nchars = local_py_text(spec, args.char_budget)
            src = "local-python"
        else:
            texts, nchars, src = hf_text(spec, args.char_budget, log)
        path = outdir / f"{name}.jsonl"
        with open(path, "w") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        summary[name] = {"docs": len(texts), "chars": nchars, "source": src,
                         "seconds": round(time.time() - t0, 1), "path": str(path)}
        log(f"[build] {name}: docs={len(texts)} chars={nchars} src={src} "
            f"({summary[name]['seconds']}s) -> {path}")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    log("R95_DOMAIN_INPUTS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
