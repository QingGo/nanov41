"""T0 eval runner: held-out BPB over multi-domain shards.

This is the implementation behind `tools/eval_t0.py`.  It is deliberately boring: no
generation, no sampling, no tokenizer — pre-tokenized windows in, NLL/byte sums out.  The same
code path runs on CPU (tiny config / fake provider) and on a 5090 (d12 checkpoint), which is
what makes eval results comparable across machines.

Determinism: forward-only, fixed non-overlapping windows, no dropout anywhere in the model, no
data shuffling.  GPU reductions can still differ in the last bits across batch sizes / kernel
choices; keep `--batch-size` fixed when comparing runs and prefer bf16 attention only if both
runs use it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import time
from typing import Callable, Sequence

import numpy as np
import torch

from .bpb import LN2, BPBAccumulator
from .mix import mix_bpb
from .data import DEFAULT_WINDOW, EvalDataset, DomainSpec, TYPE_TEXT
from .provider import LogitsProvider, describe_provider

REPORT_FORMAT = "nanov41-eval-report/1"


@dataclasses.dataclass
class EvalSettings:
    """Eval knobs. ``window=None`` means "use the domain's manifest default"."""

    window: int | None = None
    batch_size: int = 1
    max_tokens_per_domain: int | None = None
    ce_chunk: int = 256
    bootstrap: int = 0
    progress_every: int = 0

    def window_for(self, spec: DomainSpec) -> int:
        return int(self.window or spec.window or DEFAULT_WINDOW)

    def max_tokens_for(self, spec: DomainSpec) -> int | None:
        if self.max_tokens_per_domain is not None:
            return int(self.max_tokens_per_domain)
        return spec.max_tokens


@dataclasses.dataclass
class DomainResult:
    name: str
    kind: str
    window: int
    shards: int
    tokens_available: int
    tokens_processed: int
    tokens_dropped: int
    tokens_scored: int
    elapsed_s: float
    metrics: dict
    bootstrap_ci: dict

    @property
    def bpb(self) -> float:
        return float(self.metrics["all"]["bpb"])

    @property
    def tokens_per_second(self) -> float:
        return (self.tokens_processed / self.elapsed_s) if self.elapsed_s > 0 else float("nan")

    def to_dict(self, flops_per_token: float | None = None) -> dict:
        est_flops = (
            float(self.tokens_processed) * float(flops_per_token)
            if flops_per_token
            else None
        )
        return {
            "domain": self.name,
            "kind": self.kind,
            "window": self.window,
            "shards": self.shards,
            "tokens_available": self.tokens_available,
            "tokens_processed": self.tokens_processed,
            "tokens_scored": self.tokens_scored,
            "tokens_dropped": self.tokens_dropped,
            "elapsed_s": round(self.elapsed_s, 4),
            "tokens_per_second": self.tokens_per_second,
            "est_flops": est_flops,
            "metrics": self.metrics,
            "bootstrap_ci": self.bootstrap_ci,
        }


def iter_window_starts(n_tokens: int, window: int, max_tokens: int | None = None) -> list[tuple[int, int]]:
    """Non-overlapping, full-only windows ``[(start, end)]`` plus the dropped tail count.

    The first token of every window is context-only (no in-window prediction); the harness
    scores ``window - 1`` tokens per window.  Trailing tokens that do not fill a window are
    dropped and reported.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    n = n_tokens if max_tokens is None else min(int(n_tokens), int(max_tokens))
    n_windows = n // window
    return [(i * window, (i + 1) * window) for i in range(n_windows)]


def token_nll_batch(provider: LogitsProvider, input_ids: torch.Tensor,
                    ce_chunk: int = 256) -> np.ndarray:
    """Per-position NLL for the next-token labels: returns [B, S-1] float64.

    ``out[b, j] = -log p(input_ids[b, j+1] | input_ids[b, :j+1])``.  The vocab logsumexp is
    computed in position chunks (vocab 129,280 makes full [B, S, V] float32 expensive).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
    if os.environ.get("NANOV41_EVAL_LEGACY_LOGITS", "").lower() not in ("1", "true", "yes"):
        nll_fn = getattr(provider, "nll", None)
        if callable(nll_fn):
            return nll_fn(input_ids)
    logits = provider.logits(input_ids)
    if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0] or logits.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"provider returned logits {tuple(logits.shape)} for input {tuple(input_ids.shape)}"
        )
    if logits.shape[2] != provider.vocab_size:
        raise ValueError(
            f"provider vocab {provider.vocab_size} != logits vocab {logits.shape[2]}"
        )
    b, s, _ = logits.shape
    out = np.empty((b, s - 1), dtype=np.float64)
    for start in range(0, s - 1, ce_chunk):
        end = min(start + ce_chunk, s - 1)
        chunk = logits[:, start:end, :].float()
        target = input_ids[:, start + 1:end + 1].to(logits.device)
        chunk_nll = chunk.logsumexp(-1) - chunk.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        out[:, start:end] = chunk_nll.double().cpu().numpy()
    return out


def _load_shard(dataset: EvalDataset, shard) -> tuple[np.ndarray, np.ndarray]:
    tokens, types = dataset.read_shard(shard)
    if types is None:
        types = np.full(tokens.shape[0], TYPE_TEXT, dtype=np.uint8)
    return tokens, types


def evaluate_domain(provider: LogitsProvider, dataset: EvalDataset, spec: DomainSpec,
                    settings: EvalSettings | None = None,
                    log: Callable[[str], None] | None = None) -> DomainResult:
    settings = settings or EvalSettings()
    window = settings.window_for(spec)
    max_tokens = settings.max_tokens_for(spec)
    batch_size = max(1, int(settings.batch_size))
    acc = BPBAccumulator()

    tokens_available = 0
    tokens_processed = 0
    tokens_dropped = 0
    n_windows = 0
    budget = None if max_tokens is None else int(max_tokens)
    t_start = time.perf_counter()

    for shard in spec.shards:
        tokens, types = _load_shard(dataset, shard)
        n = int(tokens.shape[0])
        tokens_available += n
        allow = n if budget is None else min(n, budget)
        windows = iter_window_starts(n, window, allow)
        if not windows:
            if budget is not None:
                tokens_dropped += allow
                budget -= allow
            continue
        n_used = len(windows) * window
        tokens_processed += n_used
        tokens_dropped += allow - n_used

        for base in range(0, len(windows), batch_size):
            group = windows[base:base + batch_size]
            starts = np.asarray([w[0] for w in group], dtype=np.int64)
            ends = np.asarray([w[1] for w in group], dtype=np.int64)
            # copy the window slabs out of the memmap once; keep dtype int64 for embeddings
            batch_np = np.stack([np.asarray(tokens[a:b], dtype=np.int64) for a, b in zip(starts, ends)])
            batch_types = np.stack([np.asarray(types[a:b], dtype=np.uint8) for a, b in zip(starts, ends)])
            input_ids = torch.from_numpy(batch_np)
            nll = token_nll_batch(provider, input_ids, ce_chunk=settings.ce_chunk)
            labels = batch_np[:, 1:]
            label_types = batch_types[:, 1:]
            for row in range(batch_np.shape[0]):
                acc.add(nll[row], labels[row], label_types[row], dataset.token_bytes)
            n_windows += batch_np.shape[0]
            if log is not None and settings.progress_every and n_windows % settings.progress_every == 0:
                log(f"  [{spec.name}] {n_windows} windows, {tokens_processed} tokens")

        if budget is not None:
            budget -= allow
            if budget < window:
                break

    if tokens_processed == 0:
        raise ValueError(
            f"domain {spec.name!r}: no full {window}-token window could be formed "
            f"({tokens_available} tokens available, budget={max_tokens}); use a smaller --window"
        )
    elapsed = time.perf_counter() - t_start
    metrics = acc.summary()
    ci = acc.bootstrap_ci(n_resamples=settings.bootstrap) if settings.bootstrap else {}
    return DomainResult(
        name=spec.name,
        kind=spec.kind,
        window=window,
        shards=len(spec.shards),
        tokens_available=tokens_available,
        tokens_processed=tokens_processed,
        tokens_dropped=tokens_dropped,
        tokens_scored=int(metrics["all"]["tokens"]),
        elapsed_s=elapsed,
        metrics=metrics,
        bootstrap_ci=ci,
    )


def _git_info() -> dict:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None

    commit = run(["rev-parse", "HEAD"])
    dirty = None
    status = run(["status", "--porcelain"])
    if status is not None:
        dirty = bool(status)
    return {"commit": commit, "dirty": dirty}


def evaluate_dataset(provider: LogitsProvider, dataset: EvalDataset, *,
                     domains: Sequence[str] | None = None,
                     settings: EvalSettings | None = None,
                     log: Callable[[str], None] | None = None) -> dict:
    settings = settings or EvalSettings()
    names = list(domains) if domains is not None else dataset.domain_names()
    if not names:
        raise ValueError("dataset has no domains")
    results: dict[str, DomainResult] = {}
    for name in names:
        spec = dataset.domain(name)
        if log is not None:
            log(f"[t0] {name}: {spec.n_tokens} tokens, {len(spec.shards)} shard(s), "
                f"window={settings.window_for(spec)}, kind={spec.kind}")
        result = evaluate_domain(provider, dataset, spec, settings, log=log)
        results[name] = result
        if log is not None:
            log(f"[t0] {name}: bpb {result.bpb:.4f} over {result.tokens_scored} tokens "
                f"in {result.elapsed_s:.2f}s ({result.tokens_per_second:,.0f} tok/s)")

    flops_per_token = getattr(provider, "flops_per_token", None)
    mixes = {}
    for mix_name, mix in dataset.mixes.items():
        try:
            entry = mix_bpb(results, mix.get("weights", {}), name=mix_name)
        except ValueError as exc:  # malformed mix in a hand-edited manifest
            entry = {"name": mix_name, "error": str(exc), "bpb": float("nan"), "coverage": 0.0}
        entry["source"] = mix.get("source")
        entry["description"] = mix.get("description")
        entry["source_weights"] = mix.get("source_weights", {})
        mixes[mix_name] = entry
    nll_sum = sum(r.metrics["all"]["nll_nats"] for r in results.values())
    bytes_sum = sum(r.metrics["all"]["bytes"] for r in results.values())
    bpb_values = [r.bpb for r in results.values()]
    report = {
        "schema": REPORT_FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_info(),
        "dataset": {
            "root": str(dataset.root),
            "manifest_sha256": dataset.manifest_sha256,
            "domains": {name: results[name].to_dict(flops_per_token) for name in names},
        },
        "provider": describe_provider(provider),
        "mixes": mixes,
        "settings": {
            "window": settings.window,
            "batch_size": settings.batch_size,
            "max_tokens_per_domain": settings.max_tokens_per_domain,
            "ce_chunk": settings.ce_chunk,
            "bootstrap": settings.bootstrap,
        },
        "summary": {
            "domains": len(names),
            "macro_bpb": float(np.mean(bpb_values)),
            "byte_weighted_bpb": float(nll_sum / LN2 / bytes_sum) if bytes_sum else float("nan"),
            "tokens_scored": int(sum(r.tokens_scored for r in results.values())),
            "tokens_processed": int(sum(r.tokens_processed for r in results.values())),
            "elapsed_s": round(sum(r.elapsed_s for r in results.values()), 4),
            "flops_per_token": flops_per_token,
        },
    }
    return report


def _fmt_bpb(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.4f}"


def format_report_table(report: dict) -> str:
    domains = report["dataset"]["domains"]
    header = (
        f"{'domain':<14} {'kind':<11} {'window':>7} {'tokens':>12} {'bpb':>7} "
        f"{'text':>7} {'asst':>7} {'tool':>7} {'tok/s':>10} {'secs':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, entry in domains.items():
        metrics = entry["metrics"]
        lines.append(
            f"{name:<14} {entry['kind']:<11} {entry['window']:>7d} "
            f"{entry['tokens_scored']:>12,d} {_fmt_bpb(metrics['all']['bpb']):>7} "
            f"{_fmt_bpb(metrics.get('text', {}).get('bpb')):>7} "
            f"{_fmt_bpb(metrics.get('assistant', {}).get('bpb')):>7} "
            f"{_fmt_bpb(metrics.get('tool_output', {}).get('bpb')):>7} "
            f"{entry['tokens_per_second']:>10,.0f} {entry['elapsed_s']:>7.2f}"
        )
    mixes = report.get("mixes") or {}
    if mixes:
        lines.append("-" * len(header))
        for name, mix in mixes.items():
            coverage = mix.get("coverage", 0.0) * 100.0
            missing = mix.get("missing") or []
            missing_txt = f"  missing={','.join(missing)}" if missing else ""
            lines.append(
                f"mix {name:<20} bpb {_fmt_bpb(mix.get('bpb')):>7}  "
                f"coverage {coverage:5.1f}%{missing_txt}"
            )
    summary = report["summary"]
    lines.append("-" * len(header))
    lines.append(
        f"macro bpb {_fmt_bpb(summary['macro_bpb'])}  "
        f"byte-weighted bpb {_fmt_bpb(summary['byte_weighted_bpb'])}  "
        f"tokens {summary['tokens_scored']:,}"
    )
    if summary.get("flops_per_token"):
        lines.append(f"eval FLOPs/token {summary['flops_per_token'] / 1e6:.2f} MFLOP "
                     f"(forward, analytic)")
    return "\n".join(lines)


def report_to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def save_report(report: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n")
    return path


def check_provider_dataset_compat(provider: LogitsProvider, dataset: EvalDataset,
                                  *, allow_mismatch: bool = False) -> None:
    """Exact-vocabulary check: token byte table and provider must use the same tokenizer."""
    declared = dataset.manifest.get("tokenizer", {}).get("vocab_size")
    vocab = int(getattr(provider, "vocab_size"))
    table_len = int(dataset.token_bytes.shape[0])
    if declared is not None and int(declared) != table_len:
        raise ValueError(f"manifest tokenizer vocab {declared} != token_bytes length {table_len}")
    if vocab != table_len and not allow_mismatch:
        raise ValueError(
            f"provider vocab_size {vocab} != dataset vocab_size {table_len}; this usually means "
            "the checkpoint and the eval data were built with different tokenizers. Pass "
            "allow_mismatch=True only for toy tests."
        )


__all__ = [
    "EvalSettings",
    "DomainResult",
    "evaluate_domain",
    "evaluate_dataset",
    "format_report_table",
    "iter_window_starts",
    "report_to_json",
    "save_report",
    "token_nll_batch",
    "check_provider_dataset_compat",
]
