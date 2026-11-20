"""CORE-lite (T1) interface — *reserved*: scoring harness implemented, task data/port pending.

T0 (multi-domain held-out BPB) is deliberately cheap but weak: it cannot tell a model that
models web text from one that can answer questions.  T1 is the cheap capability step: the
DCLM **CORE** panel (22 likelihood-based tasks; nanochat reports CORE 0.2565 for GPT-2-level
models, which is the external boundary object we want to compare against).

What is implemented here is the part that does not depend on task downloads:

* the task registry (`CORE_LITE_TASKS`), a pinned JSONL schema for loaded examples, and a
  data loader that fails loudly with instructions when task data is absent;
* a **log-likelihood continuation scorer** (length-normalised), the same primitive every
  CORE task needs, with injected logits so it is unit-testable on CPU;
* an accuracy/centred-accuracy aggregator.

What is **not** implemented yet (deliberately, to keep the interface honest):

* porting the official task files + few-shot prompt formatting (nanochat/DCLM convention) and
  the exact normalisation used by the reference evaluator — those choices change absolute
  scores, so they must be pinned when the data lands, not guessed;
* MMLU / generation tasks (T3/T4) and RULER/needle (T5).

Expected data layout (created by a future `tools/fetch_core_data.py`)::

    <dataset_root>/core/<task>.jsonl        one JSON object per line:
        {"context": "<few-shot prompt + question>", "choices": ["...", "..."], "label": 0,
         "example_id": "..."}               # example_id optional

`context` is used verbatim (the few-shot builder is part of the data port); `choices` are
tokenised as continuations and scored after the tokenised context.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .harness import token_nll_batch
from .provider import LogitsProvider
from .tokenizer import encode_text

CORE_LITE_TASKS: tuple[str, ...] = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
)

# A score near chance is expected at 200-600M scale on most tasks; the aggregate is the useful
# signal.  Few-shot counts are placeholders until the official task port is pinned.
_FEWSHOT_PLACEHOLDER = 0


class CoreDataMissing(RuntimeError):
    """Raised when CORE-lite task JSONL files are absent (the reserved-interface signal)."""


@dataclasses.dataclass(frozen=True)
class CoreTaskSpec:
    name: str
    n_fewshot: int = _FEWSHOT_PLACEHOLDER
    split: str = "validation"


@dataclasses.dataclass(frozen=True)
class CoreExample:
    task: str
    context: str
    choices: tuple[str, ...]
    label: int
    example_id: str | None = None


CORE_LITE_SPECS: dict[str, CoreTaskSpec] = {name: CoreTaskSpec(name) for name in CORE_LITE_TASKS}


def core_data_dir(dataset_root: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(dataset_root) / "core"


def core_task_path(dataset_root: str | pathlib.Path, spec: CoreTaskSpec) -> pathlib.Path:
    return core_data_dir(dataset_root) / f"{spec.name}.jsonl"


def load_core_examples(dataset_root: str | pathlib.Path, spec: CoreTaskSpec,
                       max_examples: int | None = None) -> list[CoreExample]:
    path = core_task_path(dataset_root, spec)
    if not path.exists():
        raise CoreDataMissing(
            f"CORE-lite data not found: {path}\n"
            "This is the reserved T1 interface; build the task JSONL files first "
            f"(expected fields: context, choices, label; see nanov41/eval/core.py). "
            f"Registered tasks: {', '.join(CORE_LITE_TASKS)}"
        )
    examples: list[CoreExample] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            choices = tuple(str(c) for c in record.get("choices", []))
            context = str(record.get("context", ""))
            if not context or len(choices) < 2:
                raise ValueError(f"{path}:{lineno}: need non-empty context and >=2 choices")
            label = int(record["label"])
            if not 0 <= label < len(choices):
                raise ValueError(f"{path}:{lineno}: label {label} outside [0, {len(choices)})")
            examples.append(CoreExample(
                task=spec.name,
                context=context,
                choices=choices,
                label=label,
                example_id=record.get("example_id"),
            ))
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise CoreDataMissing(f"CORE-lite task file is empty: {path}")
    return examples


# --------------------------------------------------------------------------- scoring
def continuation_nll(provider: LogitsProvider, context_ids: Sequence[int],
                     continuation_ids: Sequence[int],
                     token_bytes: np.ndarray | None = None) -> dict:
    """NLL of a continuation given a context, plus token/byte counts.

    Returns ``{"nll_nats", "tokens", "bytes"}``; ``bytes`` is None unless a token-bytes table
    is supplied (needed for the byte-normalised variant).
    """
    ctx = [int(x) for x in context_ids]
    cont = [int(x) for x in continuation_ids]
    if not cont:
        raise ValueError("continuation_ids must be non-empty")
    ids = ctx + cont
    if max(ids) >= provider.vocab_size:
        raise ValueError(f"token id {max(ids)} >= provider vocab {provider.vocab_size}")
    input_ids = torch.tensor([ids], dtype=torch.long)
    nll = token_nll_batch(provider, input_ids)[0]  # nll[j] = -log p(ids[j+1])
    start = len(ctx) - 1
    values = nll[start:start + len(cont)]
    if values.shape[0] != len(cont):
        raise ValueError("internal: continuation slice was truncated")
    nll_sum = float(values.sum())
    n_bytes = None
    if token_bytes is not None:
        n_bytes = int(np.asarray(token_bytes)[np.asarray(cont, dtype=np.int64)].sum())
    return {"nll_nats": nll_sum, "tokens": len(cont), "bytes": n_bytes}


def score_continuation(provider: LogitsProvider, context_ids: Sequence[int],
                       continuation_ids: Sequence[int], *, normalize: str = "none",
                       token_bytes: np.ndarray | None = None) -> float:
    """Score a continuation; lower is better.

    normalize: "none" (sum NLL), "token" (per continuation token) or "byte" (per UTF-8 byte,
    needs token_bytes).  The normalisation used by the reference CORE evaluator will be pinned
    when the task port lands; exposing it here keeps the choice explicit.
    """
    info = continuation_nll(provider, context_ids, continuation_ids, token_bytes=token_bytes)
    if normalize == "none":
        return info["nll_nats"]
    if normalize == "token":
        return info["nll_nats"] / info["tokens"]
    if normalize == "byte":
        if info["bytes"] is None:
            raise ValueError("normalize='byte' needs the token_bytes table")
        return info["nll_nats"] / max(info["bytes"], 1)
    raise ValueError(f"unknown normalize {normalize!r}")


def choose_continuation(provider: LogitsProvider, context_ids: Sequence[int],
                        choice_id_sequences: Sequence[Sequence[int]], *,
                        normalize: str = "none",
                        token_bytes: np.ndarray | None = None) -> dict:
    """Pick the highest-likelihood choice (lowest normalised NLL). Ties -> lowest index."""
    scores = [
        score_continuation(provider, context_ids, choice, normalize=normalize,
                           token_bytes=token_bytes)
        for choice in choice_id_sequences
    ]
    return {"index": int(np.argmin(scores)), "scores": [float(x) for x in scores]}


def _centred_accuracy(accuracy: float, n_choices: int) -> float:
    chance = 1.0 / n_choices
    if chance >= 1.0:
        return float("nan")
    return (accuracy - chance) / (1.0 - chance)


def evaluate_core_lite(provider: LogitsProvider, tokenizer,
                       dataset_root: str | pathlib.Path, *,
                       specs: Iterable[CoreTaskSpec] | None = None,
                       max_examples: int | None = None,
                       normalize: str = "token",
                       token_bytes: np.ndarray | None = None,
                       log: Callable[[str], None] | None = None) -> dict:
    """Run available CORE-lite tasks (raises `CoreDataMissing` if none are present).

    Contexts/choices are tokenised with the pinned tokenizer (``tokenizer`` is a
    `tokenizers.Tokenizer`-compatible object; use `nanov41.eval.tokenizer.load_tokenizer`).
    """
    chosen = list(specs) if specs is not None else [CORE_LITE_SPECS[n] for n in CORE_LITE_TASKS]
    tasks: dict[str, dict] = {}
    n_total = 0
    n_correct_total = 0
    n_choices_seen: list[int] = []
    for spec in chosen:
        examples = load_core_examples(dataset_root, spec, max_examples=max_examples)
        n_correct = 0
        for example in examples:
            context_ids, _ = encode_text(tokenizer, example.context)
            choice_ids = [encode_text(tokenizer, choice)[0] for choice in example.choices]
            pick = choose_continuation(provider, context_ids, choice_ids, normalize=normalize,
                                       token_bytes=token_bytes)
            if pick["index"] == example.label:
                n_correct += 1
            n_choices_seen.append(len(example.choices))
        accuracy = n_correct / len(examples)
        avg_choices = float(np.mean([len(e.choices) for e in examples]))
        tasks[spec.name] = {
            "n_examples": len(examples),
            "n_correct": n_correct,
            "accuracy": accuracy,
            "centred_accuracy": _centred_accuracy(accuracy, int(round(avg_choices))),
            "n_fewshot": spec.n_fewshot,
            "normalize": normalize,
        }
        n_total += len(examples)
        n_correct_total += n_correct
        if log is not None:
            log(f"[t1] {spec.name}: acc {accuracy:.4f} ({n_correct}/{len(examples)})")
    macro = float(np.mean([t["accuracy"] for t in tasks.values()])) if tasks else float("nan")
    return {
        "schema": "nanov41-core-lite-report/1",
        "normalize": normalize,
        "max_examples": max_examples,
        "tasks": tasks,
        "summary": {
            "n_tasks": len(tasks),
            "n_examples": n_total,
            "accuracy": (n_correct_total / n_total) if n_total else float("nan"),
            "macro_accuracy": macro,
        },
    }


__all__ = [
    "CORE_LITE_TASKS",
    "CORE_LITE_SPECS",
    "CoreTaskSpec",
    "CoreExample",
    "CoreDataMissing",
    "core_data_dir",
    "core_task_path",
    "load_core_examples",
    "continuation_nll",
    "score_continuation",
    "choose_continuation",
    "evaluate_core_lite",
]
