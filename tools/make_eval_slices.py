#!/usr/bin/env python3
"""Build held-out eval shards (T0) for the nanov41 eval harness.

The eval harness reads only pre-tokenized shards, so this tool is where the pinned
DeepSeek-V4.1-Flash tokenizer (and, for agent trajectories, the pinned ``chat_template.jinja``)
is used.  It writes memory-mappable ``.npy`` shards plus a JSON manifest; see
``docs/eval-harness.md`` for the on-disk contract.

Examples
--------
    # plain held-out text domain (one JSONL with a "text" field per line)
    python tools/make_eval_slices.py --out runs/eval/data --domain web --kind text \
        --input /data/heldout/web.jsonl --tokenizer $NANOV41_DS_TOKENIZER_JSON \
        --max-tokens 20000000 --window 2048

    # agent trajectories (JSONL with "messages" and optional "tools")
    python tools/make_eval_slices.py --out runs/eval/data --domain agent --kind trajectory \
        --input /data/heldout/agent.jsonl --tokenizer $NANOV41_DS_TOKENIZER_JSON \
        --chat-template $NANOV41_DS_CHAT_TEMPLATE --thinking-mode thinking --drop-thinking false

    # tokenizer/template-free smoke data (deterministic synthetic corpus)
    python tools/make_eval_slices.py --out runs/eval/data --domain smoke --kind text \
        --synthetic 64 --tokenizer $NANOV41_DS_TOKENIZER_JSON
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Iterable, Iterator, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.eval import TYPE_TEXT  # noqa: E402
from nanov41.eval.chat import (  # noqa: E402
    ChatTemplate,
    load_chat_template,
    render_and_tag,
)
from nanov41.eval.data import DEFAULT_WINDOW, DatasetWriter  # noqa: E402
from nanov41.eval.tokenizer import (  # noqa: E402
    contract_info,
    encode_text,
    load_tokenizer,
    token_bytes_from_tokenizer,
    validate_token_bytes,
)

MAX_TOKEN_BYTES = 1 << 20  # refuse absurd single records instead of OOM-ing silently


# --------------------------------------------------------------------------- records
def iter_jsonl(paths: Sequence[str], max_records: int | None = None) -> Iterator[dict]:
    """Yield JSON objects from one or more JSONL files (``path`` injected as source)."""
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(record).__name__}")
                record = dict(record)
                record["_source"] = str(path)
                record["_lineno"] = lineno
                yield record
                seen += 1
                if max_records is not None and seen >= max_records:
                    return


_TEXT_TEMPLATES = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "def parse_args(argv):\n    parser = argparse.ArgumentParser()\n    return parser.parse_args(argv)\n",
    "In 1969, the first humans landed on the Moon; the mission returned 21.5 kg of samples.",
    "中文评测文本：大规模语言模型需要在多语言、代码与数学数据上保持稳定的似然估计。",
    "Theorem: for every epsilon > 0 there exists N such that |a_n - L| < epsilon for all n > N.",
    "SELECT user_id, COUNT(*) AS n FROM events WHERE ts > now() - interval '7 days' GROUP BY 1;",
    "The model routes each token to two of sixteen experts; the router is trained with a load-balancing bias.",
    "Los datos de evaluación deben permanecer disjuntos de los datos de entrenamiento para evitar fugas.",
)


def synthetic_text_records(n: int, seed: int = 0) -> Iterator[dict]:
    for i in range(n):
        parts = [str(_TEXT_TEMPLATES[(i + k) % len(_TEXT_TEMPLATES)]) for k in range(2)]
        yield {
            "text": "\n\n".join(parts),
            "_source": f"synthetic:text:{seed}",
            "_lineno": i,
        }


_SYNTHETIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the working tree",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and return the tail of the output",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "max_failures": {"type": "integer"}},
            },
        },
    },
]


def synthetic_trajectory_records(n: int, seed: int = 0) -> Iterator[dict]:
    for i in range(n):
        path = f"nanov41/eval/{'data' if i % 2 else 'chat'}.py"
        yield {
            "messages": [
                {"role": "system", "content": "You are a careful coding agent."},
                {"role": "user", "content": f"Read {path} and summarise the eval data contract."},
                {
                    "role": "assistant",
                    "reasoning_content": "I should read the file before answering.",
                    "content": "",
                    "tool_calls": [{
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": path, "start": 0}},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": (
                        "1  \"\"\"Eval data contract.\"\"\"\n"
                        "2  FORMAT = 'nanov41-eval-data/1'\n"
                        "3  TYPE_TEXT, TYPE_ASSISTANT, TYPE_TOOL = 1, 2, 3\n"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The contract stores uint32 token ids plus a uint8 type mask; "
                        "bytes are counted exactly with a per-token byte table."
                    ),
                },
            ],
            "tools": _SYNTHETIC_TOOLS,
            "_source": f"synthetic:trajectory:{seed}",
            "_lineno": i,
        }


# --------------------------------------------------------------------------- builders
def _finish(ids: list[np.ndarray], types: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not ids:
        raise ValueError("no tokens were produced; check --input / --max-tokens / --max-records")
    return np.concatenate(ids), np.concatenate(types)


def _source_entry(path: str, tokens: int, lines: int) -> dict:
    entry = {"path": path, "tokens": tokens, "records": lines}
    try:
        from nanov41.eval.data import sha256_file

        entry["sha256"] = sha256_file(path)
    except OSError:  # pragma: no cover - synthetic sources have no file
        entry["sha256"] = None
    return entry


def build_text_domain(tok, token_bytes: np.ndarray, records: Iterable[dict], *,
                      max_tokens: int | None, add_eos: bool, text_field: str = "text",
                      log=print) -> dict:
    ids_out: list[np.ndarray] = []
    types_out: list[np.ndarray] = []
    total = 0
    n_records = 0
    bytes_checked = 0
    sources: dict[str, dict] = {}
    eos_id = int(tok.get_vocab().get("<｜end▁of▁sentence｜>", 1))
    for record in records:
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        ids, _ = encode_text(tok, text)
        if ids.size == 0:
            continue
        if ids.size > MAX_TOKEN_BYTES:
            raise ValueError(f"single record has {ids.size} tokens (> {MAX_TOKEN_BYTES})")
        got = int(token_bytes[ids].sum())
        want = len(text.encode("utf-8"))
        if got != want:
            raise ValueError(
                f"byte accounting mismatch ({got} != {want}) at {record.get('_source')}:"
                f"{record.get('_lineno')}; refusing to write a corrupt slice"
            )
        bytes_checked += want
        ids_out.append(ids.astype(np.uint32))
        types_out.append(np.full(ids.size, TYPE_TEXT, dtype=np.uint8))
        if add_eos:
            ids_out.append(np.asarray([eos_id], dtype=np.uint32))
            types_out.append(np.asarray([TYPE_TEXT], dtype=np.uint8))
        total += ids.size + (1 if add_eos else 0)
        n_records += 1
        src = record.get("_source", "input")
        entry = sources.setdefault(src, {"path": src, "tokens": 0, "records": 0})
        entry["tokens"] += int(ids.size)
        entry["records"] += 1
        if max_tokens is not None and total >= max_tokens:
            break
    ids_arr, types_arr = _finish(ids_out, types_out)
    log(f"[text] {n_records} records, {ids_arr.size:,} tokens, {bytes_checked:,} source bytes")
    return {
        "tokens": ids_arr,
        "types": types_arr,
        "sources": [
            {**_source_entry(entry["path"], entry["tokens"], entry["records"])}
            for entry in sources.values()
        ],
    }


def build_trajectory_domain(tok, token_bytes: np.ndarray, template: ChatTemplate,
                            records: Iterable[dict], *, max_tokens: int | None,
                            thinking_mode: str, drop_thinking: bool, drop_reasoning: bool,
                            reasoning_effort: int | str | None,
                            log=print) -> dict:
    ids_out: list[np.ndarray] = []
    types_out: list[np.ndarray] = []
    total = 0
    n_records = 0
    n_assistant = 0
    n_tool = 0
    n_prompt = 0
    mismatches: list[str] = []
    sources: dict[str, dict] = {}
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        tools = record.get("tools")
        tagged = render_and_tag(
            tok,
            template,
            messages,
            tools=tools,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            drop_reasoning=drop_reasoning,
            reasoning_effort=reasoning_effort,
        )
        if not tagged.stats.get("assistant_tokens"):
            raise ValueError(
                f"trajectory at {record.get('_source')}:{record.get('_lineno')} produced no "
                "assistant tokens; check --kind trajectory and the chat template"
            )
        got = int(token_bytes[tagged.ids].sum())
        if got != tagged.stats["bytes"]:
            raise ValueError(
                f"byte accounting mismatch ({got} != {tagged.stats['bytes']}) at "
                f"{record.get('_source')}:{record.get('_lineno')}"
            )
        for key in ("assistant_turn_count_mismatch", "tool_result_count_mismatch"):
            if key in tagged.stats:
                mismatches.append(f"{record.get('_source')}:{record.get('_lineno')}: "
                                  f"{tagged.stats[key]}")
        ids_out.append(tagged.ids.astype(np.uint32))
        types_out.append(tagged.types)
        total += tagged.ids.size
        n_records += 1
        n_assistant += tagged.stats["assistant_tokens"]
        n_tool += tagged.stats["tool_tokens"]
        n_prompt += tagged.stats["prompt_tokens_before_first_assistant"]
        src = record.get("_source", "input")
        entry = sources.setdefault(src, {"path": src, "tokens": 0, "records": 0})
        entry["tokens"] += int(tagged.ids.size)
        entry["records"] += 1
        if max_tokens is not None and total >= max_tokens:
            break
    ids_arr, types_arr = _finish(ids_out, types_out)
    log(
        f"[trajectory] {n_records} records, {ids_arr.size:,} tokens "
        f"(assistant {n_assistant:,}, tool {n_tool:,}, prompt {n_prompt:,})"
    )
    if mismatches:
        log(f"[warn] {len(mismatches)} rendering mismatches, e.g.:")
        for msg in mismatches[:5]:
            log(f"  {msg}")
    return {
        "tokens": ids_arr,
        "types": types_arr,
        "sources": [
            {**_source_entry(entry["path"], entry["tokens"], entry["records"])}
            for entry in sources.values()
        ],
        "mismatches": mismatches,
    }


# --------------------------------------------------------------------------- CLI
def _bool_arg(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset directory (created/updated)")
    ap.add_argument("--domain", required=True, help="domain name, e.g. web/code/agent")
    ap.add_argument("--kind", default="text", choices=["text", "trajectory"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+", help="JSONL file(s) with {'text': ...} or {'messages': ...}")
    src.add_argument("--synthetic", type=int, nargs="?", const=64,
                     help="generate N deterministic synthetic records instead of reading files")
    ap.add_argument("--tokenizer", default=os.environ.get("NANOV41_DS_TOKENIZER_JSON"),
                    help="pinned tokenizer.json (default: $NANOV41_DS_TOKENIZER_JSON)")
    ap.add_argument("--chat-template", default=os.environ.get("NANOV41_DS_CHAT_TEMPLATE"),
                    help="pinned chat_template.jinja for --kind trajectory")
    ap.add_argument("--thinking-mode", default="thinking", choices=["thinking", "chat"])
    ap.add_argument("--drop-thinking", type=_bool_arg, default=True,
                    help="template drop_thinking (default true, matches V4.1 template)")
    ap.add_argument("--drop-reasoning", action="store_true",
                    help="strip reasoning_content/thinking from input messages before rendering")
    ap.add_argument("--reasoning-effort", default=None,
                    help="template reasoning_effort (1-100 or low/medium/high/max)")
    ap.add_argument("--text-field", default="text", help="JSON field holding plain text (kind=text)")
    ap.add_argument("--max-tokens", type=int, default=None, help="stop after ~N tokens (record boundary)")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--shard-tokens", type=int, default=25_000_000,
                    help="split a domain into shards of this many tokens (default 25M ~= 100 MB uint32)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="default eval window recorded in the manifest")
    ap.add_argument("--add-eos", action="store_true", help="insert an EOS between text records")
    ap.add_argument("--seed", type=int, default=0, help="synthetic generator seed")
    ap.add_argument("--expect-vocab", type=int, default=None,
                    help="fail unless the tokenizer vocab is this size (e.g. 129280)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tokenizer:
        print("error: pass --tokenizer or set NANOV41_DS_TOKENIZER_JSON "
              "(the pinned DeepSeek-V4.1-Flash tokenizer.json)", file=sys.stderr)
        return 2
    tok = load_tokenizer(args.tokenizer)
    contract = contract_info(tok, args.tokenizer)
    if args.expect_vocab is not None and contract["vocab_size"] != args.expect_vocab:
        print(f"error: tokenizer vocab {contract['vocab_size']} != --expect-vocab "
              f"{args.expect_vocab}", file=sys.stderr)
        return 2
    token_bytes, byte_info = token_bytes_from_tokenizer(tok)

    template: ChatTemplate | None = None
    if args.kind == "trajectory":
        if not args.chat_template:
            print("error: --kind trajectory needs --chat-template (or $NANOV41_DS_CHAT_TEMPLATE)",
                  file=sys.stderr)
            return 2
        template = load_chat_template(args.chat_template)
        contract["chat_template_sha256"] = template.sha256
        contract["chat_template_path"] = template.path

    writer = DatasetWriter(args.out, token_bytes=token_bytes, tokenizer_meta=contract)

    if args.synthetic is not None:
        records: Iterable[dict] = (
            synthetic_text_records(args.synthetic, seed=args.seed) if args.kind == "text"
            else synthetic_trajectory_records(args.synthetic, seed=args.seed)
        )
        if args.kind == "text":
            validate_token_bytes(tok, token_bytes,
                                 [r["text"] for r in synthetic_text_records(8, seed=args.seed)])
    else:
        records = iter_jsonl(args.input, max_records=args.max_records)
        if args.kind == "text":
            # validate the byte table on the first few real documents before writing anything
            sample = []
            for rec in iter_jsonl(args.input, max_records=4):
                text = rec.get(args.text_field)
                if isinstance(text, str) and text:
                    sample.append(text)
            if sample:
                validate_token_bytes(tok, token_bytes, sample)

    if args.kind == "text":
        built = build_text_domain(tok, token_bytes, records, max_tokens=args.max_tokens,
                                  add_eos=args.add_eos, text_field=args.text_field)
    else:
        assert template is not None
        built = build_trajectory_domain(
            tok, token_bytes, template, records, max_tokens=args.max_tokens,
            thinking_mode=args.thinking_mode, drop_thinking=args.drop_thinking,
            drop_reasoning=args.drop_reasoning, reasoning_effort=args.reasoning_effort,
        )

    # drop stale shards from a previous build of the same domain (files are owned by this tool)
    domain_dir = pathlib.Path(args.out) / args.domain
    if domain_dir.exists():
        for stale in domain_dir.glob("shard-*.npy"):
            stale.unlink()

    eval_meta = {"window": args.window, "max_tokens": args.max_tokens, "weight": 1.0}
    entry = writer.add_domain(
        args.domain, built["tokens"], built["types"], kind=args.kind,
        sources=built["sources"], eval=eval_meta, shard_tokens=args.shard_tokens,
    )
    stats = entry["stats"]
    print(f"[done] domain={args.domain} kind={args.kind} tokens={stats['n_tokens']:,} "
          f"shards={len(entry['shards'])} bytes={stats['n_bytes']:,}")
    for name, type_stats in stats["by_type"].items():
        print(f"  {name:<12} tokens={type_stats['tokens']:>12,} bytes={type_stats['bytes']:>14,}")
    print(f"  tokenizer vocab={contract['vocab_size']} sha256={contract['sha256'][:16]}")
    mixes = writer.manifest.get("mixes", {})
    if mixes:
        print(f"  reference mixes: {', '.join(sorted(mixes))}")
    print(f"  manifest={pathlib.Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
