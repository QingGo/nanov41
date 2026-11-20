#!/usr/bin/env python3
"""Build the P2 pretraining corpus (0.6 B tokens, Marin-8B phase-1-like mix).

Sources are the same families as the T0 eval fetcher but the **training splits** (never the T0
holdouts): DCLM-Baseline global shards 01-09 (shard 10 is the T0 web holdout), codeparrot-clean
train (eval uses codeparrot-clean-valid), ProofPile-2 train splits (eval uses test).  Tokens are
written as little-endian uint32 arrays, one file per domain, with an EOS document separator.

Usage:
    python tools/p2_make_data.py --out-dir /root/autodl-tmp/p2_data --tokens 600000000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_fetch():
    path = ROOT / "tools" / "fetch_eval_data.py"
    spec = importlib.util.spec_from_file_location("p2_fetch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p2_fetch"] = mod
    spec.loader.exec_module(mod)
    return mod


def _train_sources(fetch):
    Source = fetch.Source
    web_files = [
        f"global-shard_{i:02d}_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"
        for i in range(1, 10)   # 01-09: shard 10 is the T0 web holdout
    ]
    code_files = [
        "file-000000000001.json.gz",
        "file-000000000002.json.gz",
        "file-000000000003.json.gz",
        "file-000000000004.json.gz",
    ]
    return {
        "web": [Source(repo="mlfoundations/dclm-baseline-1.0", files=tuple(web_files),
                       note="DCLM-Baseline shards 01-03 (training; T0 holds out shard 10)")],
        "code": [Source(repo="codeparrot/codeparrot-clean", files=tuple(code_files),
                        text_fields=("content",),
                        note="codeparrot-clean train files (T0 uses -valid)")],
        "math": [Source(repo="EleutherAI/proof-pile-2",
                        file_glob=("algebraic-stack/train/*.jsonl.zst",
                                   "open-web-math/train/*.jsonl.zst"),
                        note="ProofPile-2 train splits (T0 uses test splits)")],
    }


def build_domain(fetch, name, sources, out_dir, target_tokens, tok, eos_id, log):
    out_path = out_dir / f"{name}.u32.bin"
    fh = open(out_path, "wb")
    n_tok = 0
    n_docs = 0
    buf: list[int] = []
    t0 = time.time()

    def flush(force=False):
        nonlocal buf
        if len(buf) >= 4_000_000 or (force and buf):
            np.asarray(buf, dtype="<u4").tofile(fh)
            buf = []

    def add_ids(ids):
        nonlocal n_tok, n_docs
        buf.extend(ids)
        buf.append(eos_id)
        n_tok += len(ids) + 1
        n_docs += 1
        flush()

    for source in sources:
        info = fetch.list_repo_files(source.repo, fetch.DEFAULT_ENDPOINT, "main")
        files = fetch.resolve_files(source, fetch.DEFAULT_ENDPOINT, "main")
        log(f"[{name}] {source.repo}: {len(files)} file(s)")
        for path in files:
            if n_tok >= target_tokens:
                break
            log(f"[{name}] stream {path} (have {n_tok/1e6:.1f}M tokens)")
            try:
                for _, record in fetch.iter_source_records(
                        source, [path], fetch.DEFAULT_ENDPOINT, "main", max_download_mb=512):
                    text = fetch.extract_text(record, source)
                    if not text:
                        continue
                    ids = tok.encode(text, add_special_tokens=False).ids
                    if ids:
                        add_ids(ids)
                    if n_tok >= target_tokens:
                        break
            except Exception as exc:  # noqa: BLE001
                log(f"[warn] {name}/{path}: {type(exc).__name__}: {exc}")
    flush(force=True)
    fh.close()
    info = {"file": out_path.name, "tokens": int(n_tok), "docs": int(n_docs),
            "target_tokens": int(target_tokens), "bytes": os_path_size(out_path),
            "seconds": round(time.time() - t0, 1)}
    log(f"[{name}] done {n_tok/1e6:.2f} M tokens, {n_docs} docs -> {out_path}")
    return info


def os_path_size(p: pathlib.Path) -> int:
    try:
        return int(p.stat().st_size)
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/autodl-tmp/p2_data")
    ap.add_argument("--tokens", type=int, default=600_000_000)
    ap.add_argument("--web", type=float, default=0.926)
    ap.add_argument("--code", type=float, default=0.061)
    ap.add_argument("--math", type=float, default=0.013)
    ap.add_argument("--tokenizer", default="/root/dsv41/tokenizer.json")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    fetch = _load_fetch()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eos_ids = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False).ids
    eos_id = int(eos_ids[0]) if len(eos_ids) == 1 else 1
    log = lambda msg: print(msg, flush=True)
    log(f"tokenizer vocab={tok.get_vocab_size()} eos_id={eos_id}")

    targets = {
        "web": int(args.tokens * args.web),
        "code": int(args.tokens * args.code),
        "math": int(args.tokens * args.math),
    }
    sources = _train_sources(fetch)
    manifest = {"format": "nanov41-p2-corpus/1", "tokens_requested": args.tokens,
                "mix": {"web": args.web, "code": args.code, "math": args.math},
                "tokenizer": args.tokenizer, "eos_id": eos_id,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "domains": {}}
    for name in ("web", "code", "math"):
        manifest["domains"][name] = build_domain(
            fetch, name, sources[name], out_dir, targets[name], tok, eos_id, log)
    manifest["tokens_total"] = sum(d["tokens"] for d in manifest["domains"].values())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[done] total {manifest['tokens_total']/1e6:.2f} M tokens in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
