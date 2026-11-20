"""Official CORE metric (DCLM / nanochat convention) — 22-task ICL panel.

This is the capability metric nanochat uses for its "time to GPT-2" leaderboard
(GPT-2 CORE = 0.256525; nanochat d24 = 0.2585/0.2626).  Absolute numbers are *tokenizer
independent* (accuracy, not BPB), so our 129k-vocab nanov41 checkpoints can be compared
directly with nanochat's published numbers, provided the task data + prompt convention are
the same.  That convention is ported 1:1 from the MIT-licensed nanochat implementation
(`nanochat/core_eval.py` + `scripts/base_eval.py`, eval_bundle from
https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip):

* 22 tasks from `eval_bundle/core.yaml`, types ``multiple_choice`` / ``schema`` /
  ``language_modeling``; delimiters and few-shot counts per task;
* few-shot examples sampled per-example with ``random.Random(1234 + idx)`` from the other
  examples of the same task;
* multiple-choice / schema: pick the option with the lowest **mean per-token NLL** over the
  continuation span (span found from the common token prefix/suffix across options);
* language modeling: correct iff *every* continuation token is the argmax prediction;
* sequences longer than the model context are left-cropped to ``max_seq_len`` (indices
  shifted down), exactly like the reference evaluator;
* per task centred accuracy = (acc - baseline) / (1 - baseline), baseline from
  ``eval_meta_data.csv`` (percent), CORE = mean over the 22 tasks.

Deviation from nanochat (documented): no BOS token is prepended, because nanov41 was
pretrained without one (`add_special_tokens=False`); contexts are tokenized verbatim.

Usage (remote):
    python tools/eval_core.py --ckpt runs/p5/E-default.pt \
        --bundle /root/autodl-tmp/core_bundle/eval_bundle \
        --max-per-task -1 --out runs/p5/E-default.core.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import random
import re
import time

import numpy as np
import torch

__all__ = [
    "CoreTaskSpec",
    "load_bundle_tasks",
    "load_baselines",
    "evaluate_core",
]

# --------------------------------------------------------------------------- bundle parsing
_TASK_KEY_RE = re.compile(r"^\s*(\w+):\s*(.*)$")


class CoreTaskSpec:
    def __init__(self, label: str, dataset_uri: str, task_type: str, num_fewshot: int,
                 continuation_delimiter: str = " "):
        self.label = label
        self.dataset_uri = dataset_uri
        self.task_type = task_type
        self.num_fewshot = int(num_fewshot)
        self.continuation_delimiter = continuation_delimiter

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (f"CoreTaskSpec({self.label!r}, type={self.task_type!r}, "
                f"fewshot={self.num_fewshot}, delim={self.continuation_delimiter!r})")


def load_bundle_tasks(bundle_dir: str | pathlib.Path) -> list[CoreTaskSpec]:
    """Parse ``core.yaml`` (the bundle's simple key/value list format) into task specs."""
    path = pathlib.Path(bundle_dir) / "core.yaml"
    if not path.exists():
        raise FileNotFoundError(f"CORE bundle config not found: {path}")
    tasks: list[CoreTaskSpec] = []
    block: dict[str, str] = {}

    def flush() -> None:
        if not block.get("label"):
            return
        raw_delim = block.get("continuation_delimiter")
        delim = " "
        if raw_delim:
            # yaml strings in the bundle are double-quoted with escapes, e.g. "\nAnswer: "
            try:
                delim = json.loads(raw_delim) if raw_delim.startswith('"') else raw_delim
            except json.JSONDecodeError:
                delim = raw_delim.strip('"')
        fewshot_raw = block.get("num_fewshot", "[0]").strip()
        m = re.search(r"\[?\s*(\d+)", fewshot_raw)
        tasks.append(CoreTaskSpec(
            label=block["label"],
            dataset_uri=block.get("dataset_uri", ""),
            task_type=block.get("icl_task_type", "multiple_choice"),
            num_fewshot=int(m.group(1)) if m else 0,
            continuation_delimiter=delim,
        ))
        block.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "-":
            flush()
            continue
        m = _TASK_KEY_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val:
                block[key] = val
    flush()
    if len(tasks) != 22:
        raise ValueError(f"expected 22 CORE tasks, parsed {len(tasks)} from {path}")
    return tasks


def load_baselines(bundle_dir: str | pathlib.Path) -> dict[str, float]:
    """Random-baseline accuracy per task label from ``eval_meta_data.csv`` (percent)."""
    path = pathlib.Path(bundle_dir) / "eval_meta_data.csv"
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["Eval Task"]] = float(row["Random baseline"])
    return out


# --------------------------------------------------------------------------- prompt rendering
def _render_choice_prompts_mc(item: dict, delim: str, few: list[dict]) -> list[str]:
    prefix = "".join(f"{ex['query']}{delim}{ex['choices'][ex['gold']]}\n\n" for ex in few)
    return [(prefix + f"{item['query']}{delim}{choice}").strip() for choice in item["choices"]]


def _render_choice_prompts_schema(item: dict, delim: str, few: list[dict]) -> list[str]:
    prefix = "".join(
        f"{ex['context_options'][ex['gold']]}{delim}{ex['continuation']}\n\n" for ex in few)
    return [(prefix + f"{ctx}{delim}{item['continuation']}").strip()
            for ctx in item["context_options"]]


def _render_lm_prompts(item: dict, delim: str, few: list[dict]) -> tuple[str, str]:
    prefix = "".join(
        f"{ex['context'].strip()}{delim}{ex['continuation']}\n\n" for ex in few)
    without = (prefix + f"{item['context'].strip()}{delim}").strip()
    with_cont = prefix + f"{item['context'].strip()}{delim}{item['continuation']}"
    return without, with_cont


# --------------------------------------------------------------------------- token helpers
def _encode(tok, text: str) -> list[int]:
    return [int(x) for x in tok.encode(text, add_special_tokens=False).ids]


def _find_common_length(sequences: list[list[int]], direction: str = "left") -> int:
    min_len = min(len(seq) for seq in sequences)
    idxs = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(idxs):
        token = sequences[0][idx]
        if not all(seq[idx] == token for seq in sequences):
            return i
    return min_len


def _crop_to_max(tokens: list[list[int]], starts: list[int], ends: list[int],
                 max_tokens: int | None) -> tuple[list[list[int]], list[int], list[int]]:
    if not max_tokens:
        return tokens, starts, ends
    out_t, out_s, out_e = [], [], []
    for t, s, e in zip(tokens, starts, ends):
        if len(t) > max_tokens:
            crop = len(t) - max_tokens
            out_t.append(t[-max_tokens:])
            out_s.append(s - crop)
            out_e.append(e - crop)
        else:
            out_t.append(t)
            out_s.append(s)
            out_e.append(e)
    return out_t, out_s, out_e


def _stack(token_seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    seq_len = max(len(x) for x in token_seqs)
    ids = torch.full((len(token_seqs), seq_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(token_seqs):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids


def _mean_loss(losses: np.ndarray, start: int, end: int, row: int) -> float:
    lo = max(start - 1, 0)
    hi = max(end - 1, lo + 1)
    values = losses[row, lo:hi]
    return float(values.mean()) if values.size else float("inf")


def _nll_and_argmax(provider, ids: torch.Tensor, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-position NLL and argmax predictions without materialising [B,S,V] logits."""
    ids = ids.to(provider.device)
    hidden = provider.hidden(ids)
    weight = provider.model.inner.head.weight
    b, s, d = hidden.shape
    nll = np.empty((b, s - 1), dtype=np.float64)
    pred = np.empty((b, s - 1), dtype=np.int64)
    for start in range(0, s - 1, chunk):
        end = min(start + chunk, s - 1)
        h = hidden[:, start:end, :].reshape(-1, d).to(weight.dtype)
        logits = torch.nn.functional.linear(h, weight).float()
        lse = logits.logsumexp(-1)
        target = ids[:, start + 1:end + 1].reshape(-1)
        nll[:, start:end] = (lse - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                             ).reshape(b, end - start).double().cpu().numpy()
        pred[:, start:end] = logits.argmax(-1).reshape(b, end - start).cpu().numpy()
    return nll, pred


# --------------------------------------------------------------------------- task evaluation
@torch.no_grad()
def _prepare_example(tok, data: list[dict], idx: int, spec: CoreTaskSpec,
                     max_seq_len: int | None) -> dict | None:
    """Tokenise one example into (rows, start, end) spans; None if the prompt pair is malformed."""
    item = data[idx]
    delim = spec.continuation_delimiter
    few: list[dict] = []
    if spec.num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        n_few = min(spec.num_fewshot, len(available))
        few = [data[i] for i in rng.sample(available, n_few)]

    if spec.task_type == "multiple_choice":
        prompts = _render_choice_prompts_mc(item, delim, few)
        token_seqs = [_encode(tok, p) for p in prompts]
        start_idx = _find_common_length(token_seqs, "left")
        starts = [start_idx] * len(token_seqs)
        ends = [len(t) for t in token_seqs]
    elif spec.task_type == "schema":
        prompts = _render_choice_prompts_schema(item, delim, few)
        token_seqs = [_encode(tok, p) for p in prompts]
        suffix = _find_common_length(token_seqs, "right")
        ends = [len(t) for t in token_seqs]
        starts = [e - suffix for e in ends]
    elif spec.task_type == "language_modeling":
        without, with_cont = _render_lm_prompts(item, delim, few)
        t_without, t_with = _encode(tok, without), _encode(tok, with_cont)
        if not (len(t_without) < len(t_with) and t_with[:len(t_without)] == t_without):
            return None
        token_seqs = [t_with]
        starts = [len(t_without)]
        ends = [len(t_with)]
    else:
        raise ValueError(f"unsupported CORE task type {spec.task_type!r}")

    token_seqs, starts, ends = _crop_to_max(token_seqs, starts, ends, max_seq_len)
    return {"rows": token_seqs, "starts": starts, "ends": ends,
            "gold": item.get("gold")}


def _score_batch(provider, batch: list[dict], is_lm: bool, pad_id: int) -> list[bool]:
    token_seqs = [seq for ex in batch for seq in ex["rows"]]
    input_ids = _stack(token_seqs, pad_id)
    if is_lm:
        _nll, pred = _nll_and_argmax(provider, input_ids)
        out = []
        row = 0
        for ex in batch:
            si, ei = ex["starts"][0], ex["ends"][0]
            predicted = pred[row, max(si - 1, 0):max(ei - 1, 1)]
            actual = input_ids[row, si:ei].cpu().numpy()
            out.append(predicted.shape == actual.shape and bool(np.array_equal(predicted, actual)))
            row += len(ex["rows"])
        return out

    nll = provider.nll(input_ids)
    out = []
    row = 0
    for ex in batch:
        mean_losses = [_mean_loss(nll, s, e, row + j)
                       for j, (s, e) in enumerate(zip(ex["starts"], ex["ends"]))]
        out.append(int(np.argmin(mean_losses)) == int(ex["gold"]))
        row += len(ex["rows"])
    return out


@torch.no_grad()
def evaluate_task(provider, tok, spec: CoreTaskSpec, bundle_dir: pathlib.Path,
                  max_per_task: int = -1, max_seq_len: int | None = None,
                  batch_examples: int = 16, batch_tokens: int = 65536,
                  max_rows: int = 64, log=None) -> dict:
    data_path = bundle_dir / "eval_data" / spec.dataset_uri
    with open(data_path, encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh if line.strip()]
    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]
    vocab = tok.get_vocab()
    pad_id = int(vocab.get("<｜end▁of▁sentence｜>", vocab.get("<｜▁pad▁｜>", 0)))
    t0 = time.time()
    is_lm = spec.task_type == "language_modeling"

    preps = [_prepare_example(tok, data, i, spec, max_seq_len) for i in range(len(data))]
    order = sorted(range(len(preps)),
                   key=lambda i: max((len(r) for r in preps[i]["rows"]), default=0)
                   if preps[i] is not None else 0)
    correct = [False] * len(preps)
    batch: list[dict] = []
    batch_indices: list[int] = []
    batch_max = 0
    for i in order:
        ex = preps[i]
        if ex is None:
            continue
        max_len = max(len(r) for r in ex["rows"])
        rows_total = sum(len(e["rows"]) for e in batch)
        new_rows = rows_total + len(ex["rows"])
        # bound by examples, engram eval cache rows (cfg["max_batch_size"]) and padded tokens
        if batch and ((len(batch) >= batch_examples)
                      or (rows_total + len(ex["rows"]) > max_rows)
                      or (new_rows * max(batch_max, max_len) > batch_tokens)):
            scored = _score_batch(provider, batch, is_lm, pad_id)
            for ex_i, ok in zip(batch_indices, scored):
                correct[ex_i] = ok
            batch, batch_max, batch_indices = [], 0, []
        batch.append(ex)
        batch_indices.append(i)
        batch_max = max(batch_max, max_len)
    if batch:
        scored = _score_batch(provider, batch, is_lm, pad_id)
        for ex_i, ok in zip(batch_indices, scored):
            correct[ex_i] = ok

    n_correct = sum(correct)
    accuracy = n_correct / len(preps)
    elapsed = time.time() - t0
    if log is not None:
        log(f"[core] {spec.label:32s} {spec.task_type:18s} {spec.num_fewshot}-shot "
            f"acc={accuracy:.4f} ({n_correct}/{len(preps)}) {elapsed:.1f}s "
            f"[{len(preps) / max(elapsed, 1e-9):.1f} ex/s]")
    return {"task_type": spec.task_type, "num_fewshot": spec.num_fewshot,
            "n_examples": len(preps), "n_correct": n_correct,
            "accuracy": accuracy, "elapsed_s": elapsed}


@torch.no_grad()
def evaluate_core(provider, tok, bundle_dir: str | pathlib.Path, *,
                  tasks: list[str] | None = None, max_per_task: int = -1,
                  max_seq_len: int | None = None, batch_examples: int = 16,
                  batch_tokens: int = 65536, max_rows: int = 64, log=None) -> dict:
    """Run the official CORE panel; returns per-task accuracies and the CORE metric."""
    bundle_dir = pathlib.Path(bundle_dir)
    specs = load_bundle_tasks(bundle_dir)
    if tasks:
        wanted = set(tasks)
        unknown = wanted - {s.label for s in specs}
        if unknown:
            raise ValueError(f"unknown CORE tasks: {sorted(unknown)}")
        specs = [s for s in specs if s.label in wanted]
    baselines = load_baselines(bundle_dir)
    results: dict[str, dict] = {}
    centered: dict[str, float] = {}
    t0 = time.time()
    for spec in specs:
        entry = evaluate_task(provider, tok, spec, bundle_dir,
                              max_per_task=max_per_task, max_seq_len=max_seq_len,
                              batch_examples=batch_examples, batch_tokens=batch_tokens,
                              max_rows=max_rows, log=log)
        baseline = baselines[spec.label] / 100.0
        entry["random_baseline"] = baselines[spec.label]
        entry["centred_accuracy"] = (entry["accuracy"] - baseline) / (1.0 - baseline)
        results[spec.label] = entry
        centered[spec.label] = entry["centred_accuracy"]
    core_metric = float(np.mean(list(centered.values()))) if centered else float("nan")
    report = {
        "schema": "nanov41-core/1",
        "bundle": str(bundle_dir),
        "max_per_task": max_per_task,
        "max_seq_len": max_seq_len,
        "tasks": results,
        "centred_results": centered,
        "core_metric": core_metric,
        "elapsed_s": time.time() - t0,
        "reference": {"gpt2_core": 0.256525, "nanochat_d24": 0.2585,
                      "nanochat_best": 0.2690},
    }
    if log is not None:
        log(f"[core] CORE = {core_metric:.4f} over {len(centered)} tasks "
            f"({report['elapsed_s']:.0f}s)")
    return report
