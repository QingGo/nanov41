"""Evaluation harness for nanov41: held-out multi-domain BPB (T0) + reserved CORE-lite (T1).

Quick start (no GPU, no tokenizer, no model)::

    python tools/eval_t0.py --dataset runs/eval/data --provider fake --fake-kind uniform

Building eval data: fetch Marin-referenced held-out raw text with
``tools/fetch_eval_data.py`` (network), then tokenize with ``tools/make_eval_slices.py``
(needs the pinned tokenizer; trajectory domains also need the pinned ``chat_template.jinja``)::

    python tools/make_eval_slices.py --out runs/eval/data --domain web --kind text \\
        --input web_heldout.jsonl --tokenizer /path/tokenizer.json
    python tools/make_eval_slices.py --out runs/eval/data --domain agent --kind trajectory \\
        --input agent_heldout.jsonl --tokenizer /path/tokenizer.json \\
        --chat-template /path/chat_template.jinja

See `docs/eval-harness.md` for the data contract, byte accounting and metric definitions.
"""

from __future__ import annotations

from .bpb import LN2, BPBAccumulator, bpb_from_counts
from .data import (
    ALL_TYPES,
    DEFAULT_WINDOW,
    FORMAT,
    TYPE_ASSISTANT,
    TYPE_IGNORE,
    TYPE_NAMES,
    TYPE_TEXT,
    TYPE_TOOL,
    DatasetWriter,
    DomainSpec,
    EvalDataset,
    ShardSpec,
)
from .mix import REFERENCE_MIXES, mix_bpb, validate_mix
from .harness import (
    DomainResult,
    EvalSettings,
    check_provider_dataset_compat,
    evaluate_dataset,
    evaluate_domain,
    format_report_table,
    iter_window_starts,
    report_to_json,
    save_report,
    token_nll_batch,
)
from .provider import FakeProvider, LogitsProvider, ModelProvider, make_provider

__all__ = [
    # data
    "FORMAT",
    "DEFAULT_WINDOW",
    "TYPE_IGNORE",
    "TYPE_TEXT",
    "TYPE_ASSISTANT",
    "TYPE_TOOL",
    "TYPE_NAMES",
    "ALL_TYPES",
    "DatasetWriter",
    "DomainSpec",
    "EvalDataset",
    "ShardSpec",
    # bpb / harness
    "LN2",
    "BPBAccumulator",
    "bpb_from_counts",
    "DomainResult",
    "EvalSettings",
    "evaluate_dataset",
    "evaluate_domain",
    "format_report_table",
    "iter_window_starts",
    "token_nll_batch",
    "report_to_json",
    "save_report",
    "check_provider_dataset_compat",
    # providers
    "FakeProvider",
    "LogitsProvider",
    "ModelProvider",
    "make_provider",
    # reference mixes
    "REFERENCE_MIXES",
    "mix_bpb",
    "validate_mix",
]
