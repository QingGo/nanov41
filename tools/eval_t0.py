#!/usr/bin/env python3
"""T0 multi-domain held-out BPB evaluator (+ reserved T1 CORE-lite hook).

Reads pre-tokenized eval shards (built by ``tools/make_eval_slices.py``), runs forward-only
evaluation and reports bits-per-byte per domain and per token type (text / assistant /
tool_output), plus a byte-weighted aggregate.  See ``docs/eval-harness.md``.

Examples
--------
    # no model, no GPU, no tokenizer: validates the harness/data contract end-to-end
    python tools/eval_t0.py --dataset runs/eval/data --provider fake

    # d12 checkpoint on a 5090 (bf16, batch 2, 4096-token windows)
    python tools/eval_t0.py --dataset runs/eval/data/test --ckpt runs/d12.pt \
        --device cuda --dtype bf16 --window 4096 --batch-size 2

    # CPU smoke with a random tiny model
    python tools/eval_t0.py --dataset runs/eval/data/smoke --config tiny --random-init

    # only some domains; write the JSON report somewhere explicit
    python tools/eval_t0.py --dataset runs/eval/data --domains web,code --out runs/eval/t0.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.eval.data import EvalDataset  # noqa: E402
from nanov41.eval.harness import (  # noqa: E402
    EvalSettings,
    check_provider_dataset_compat,
    evaluate_dataset,
    format_report_table,
    save_report,
)
from nanov41.eval.provider import make_provider  # noqa: E402

DEFAULT_DATASET = pathlib.Path("runs/eval/data")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("NANOV41_EVAL_DATA", str(DEFAULT_DATASET)),
                    help=f"eval dataset dir with manifest.json (default: $NANOV41_EVAL_DATA or {DEFAULT_DATASET})")
    ap.add_argument("--domains", default="all",
                    help="comma-separated domain names, or 'all' (default)")
    ap.add_argument("--tier", default="t0", choices=["t0", "t1"],
                    help="t0 = multi-domain BPB (default); t1 = also run reserved CORE-lite")
    ap.add_argument("--provider", default="model", choices=["model", "fake"])
    ap.add_argument("--fake-kind", default="uniform", choices=["uniform", "oracle"],
                    help="fake provider logits: uniform -> bpb=log2(vocab), oracle -> 0")
    ap.add_argument("--config", default=None, help="model config name (d12/d16/d20/tiny/...)")
    ap.add_argument("--ckpt", default=None, help="train.py checkpoint (.pt) with cfg+state_dict")
    ap.add_argument("--random-init", action="store_true", help="random weights (smoke tests only)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bf16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=int, default=None,
                    help="eval window in tokens (default: per-domain manifest value)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens-per-domain", type=int, default=None,
                    help="override the manifest's per-domain token budget")
    ap.add_argument("--ce-chunk", type=int, default=256,
                    help="positions per vocab-logsumexp chunk (memory control)")
    ap.add_argument("--nll-mode", default="auto", choices=["auto", "chunked", "fused", "logits"],
                    help="auto/chunked = head+logsumexp tiled over positions (no full logits); "
                         "logits = legacy full-logits path")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="window-level bootstrap resamples for a rough BPB CI (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report JSON path (default: runs/eval/t0-*.json)")
    ap.add_argument("--no-save", action="store_true", help="do not write a report JSON")
    ap.add_argument("--json", action="store_true", help="print the report JSON to stdout only")
    ap.add_argument("--allow-vocab-mismatch", action="store_true",
                    help="toy tests only: do not require provider vocab == dataset vocab")
    ap.add_argument("--list-domains", action="store_true", help="print domains and exit")
    return ap


def _resolve_domains(dataset: EvalDataset, spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return dataset.domain_names()
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in dataset.domains]
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; available: {dataset.domain_names()}")
    return names


def _default_out(dataset: EvalDataset, provider_label: str) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in provider_label)
    root = dataset.root / "reports"
    return root / f"t0-{safe}-{stamp}.json"


def _run_t1(provider, dataset: EvalDataset, args, report: dict) -> None:
    """Reserved CORE-lite hook: runs only if task JSONL data exists."""
    from nanov41.eval.core import CoreDataMissing, evaluate_core_lite
    from nanov41.eval.tokenizer import load_tokenizer

    tokenizer_path = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
    if not tokenizer_path:
        report["t1"] = {"status": "skipped",
                        "detail": "set NANOV41_DS_TOKENIZER_JSON to run CORE-lite"}
        print("[t1] skipped: no NANOV41_DS_TOKENIZER_JSON (reserved interface ready)", file=sys.stderr)
        return
    try:
        tok = load_tokenizer(tokenizer_path)
        t1 = evaluate_core_lite(provider, tok, dataset.root, max_examples=None,
                                log=lambda msg: None if args.json else print(msg))
        report["t1"] = {"status": "ok", **t1}
        if not args.json:
            summary = t1["summary"]
            print(f"[t1] CORE-lite macro accuracy {summary['macro_accuracy']:.4f} "
                  f"over {summary['n_examples']} examples / {summary['n_tasks']} tasks")
    except CoreDataMissing as exc:
        report["t1"] = {"status": "missing", "detail": str(exc)}
        stream = sys.stderr if args.json else sys.stdout
        print("[t1] CORE-lite data missing (reserved interface; see docs/eval-harness.md)",
              file=stream)
        print(f"     {str(exc).splitlines()[0]}", file=stream)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = EvalDataset.open(args.dataset)
    if args.list_domains:
        for name in dataset.domain_names():
            spec = dataset.domain(name)
            print(f"{name}\t{spec.kind}\t{spec.n_tokens} tokens\t{len(spec.shards)} shards\t"
                  f"window={spec.window}")
        return 0
    domains = _resolve_domains(dataset, args.domains)

    if args.provider == "fake":
        provider = make_provider(provider="fake", fake_vocab_size=int(dataset.token_bytes.shape[0]),
                                 fake_kind=args.fake_kind)
        provider_label = f"fake-{args.fake_kind}"
    else:
        if args.nll_mode == "logits":
            os.environ["NANOV41_EVAL_LEGACY_LOGITS"] = "1"
        else:
            os.environ.pop("NANOV41_EVAL_LEGACY_LOGITS", None)
        if args.nll_mode == "fused":
            os.environ["NANOV41_EVAL_FUSED_HEAD"] = "1"
        else:
            os.environ.pop("NANOV41_EVAL_FUSED_HEAD", None)
        if args.ckpt is None and not args.random_init:
            print("error: model provider needs --ckpt <path> (or --random-init for a smoke run), "
                  "or use --provider fake", file=sys.stderr)
            return 2
        if args.random_init and args.config is None:
            print("error: --random-init needs --config (e.g. --config tiny)", file=sys.stderr)
            return 2
        provider = make_provider(
            provider="model", config_name=args.config, checkpoint=args.ckpt,
            random_init=args.random_init, dtype=args.dtype, device=args.device, seed=args.seed,
        )
        setattr(provider, "nll_mode", args.nll_mode)
        info = getattr(provider, "describe", lambda: {})()
        provider_label = f"{info.get('config') or 'model'}-{args.dtype}"

    try:
        check_provider_dataset_compat(provider, dataset, allow_mismatch=args.allow_vocab_mismatch)
        settings = EvalSettings(
            window=args.window,
            batch_size=args.batch_size,
            max_tokens_per_domain=args.max_tokens_per_domain,
            ce_chunk=args.ce_chunk,
            bootstrap=args.bootstrap,
        )
        report = evaluate_dataset(
            provider, dataset, domains=domains, settings=settings,
            log=None if args.json else (lambda msg: print(msg, flush=True)),
        )
        report["dataset"]["domains_requested"] = domains
        report["model_label"] = provider_label
        report["nll_mode"] = args.nll_mode

        if args.tier == "t1":
            _run_t1(provider, dataset, args, report)

        if not args.json:
            print()
            print(format_report_table(report))
        if not args.no_save:
            out = pathlib.Path(args.out) if args.out else _default_out(dataset, provider_label)
            save_report(report, out)
            if not args.json:
                print(f"\n[report] {out}")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
