"""Chat-template rendering and token-level tagging for agent-trajectory eval slices.

The pinned DeepSeek-V4.1-Flash chat template (`chat_template.jinja`, 15,348 B in the model
repo) is *not* vendored here; pass its path at data-build time.  Rendering uses a Jinja2
sandbox the same way HuggingFace does (``raise_exception`` global, ``tojson`` filter), and the
template is recorded by sha256 in the dataset manifest so an eval slice can always be traced
back to the exact template that produced it.

Token tagging does not depend on template internals beyond two facts about the V4.1 format:

* role headers are **single special-token ids** (``<｜System｜>``, ``<｜User｜>``,
  ``<｜Assistant｜>``, ``<｜latest_reminder｜>``), so an assistant turn is exactly the token
  range from an ``<｜Assistant｜>`` id up to the next role-header id;
* tool results are rendered as literal ``<tool_result>...</tool_result>`` blocks, so their
  character spans can be mapped to token spans with the tokenizer's character offsets.

This gives exact, template-agnostic ``ASSISTANT`` / ``TOOL`` masks; tokens outside those spans
(prompt, user turns, system prompt) stay ``TEXT``.  The slices that the eval harness reads are
just ``ids + types`` — no tokenizer or Jinja needed at eval time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re
from typing import Any, Sequence

import numpy as np

from .data import TYPE_ASSISTANT, TYPE_TEXT, TYPE_TOOL
from .tokenizer import char_spans_to_token_spans, encode_text

ASSISTANT_MARKER = "<｜Assistant｜>"
USER_MARKER = "<｜User｜>"
SYSTEM_MARKER = "<｜System｜>"
LATEST_REMINDER_MARKER = "<｜latest_reminder｜>"
BOUNDARY_MARKERS = (ASSISTANT_MARKER, USER_MARKER, SYSTEM_MARKER, LATEST_REMINDER_MARKER)

TOOL_RESULT_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)


class ChatTemplateError(ValueError):
    """Template missing, unrenderable, or incompatible with the pinned tokenizer."""


# --------------------------------------------------------------------------- template
def _raise_exception(message: str) -> None:
    raise ChatTemplateError(str(message))


def _make_env():
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "rendering chat templates needs `jinja2` (pip install jinja2); evaluation of "
            "pre-tokenized shards does not"
        ) from exc
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    return env


@dataclasses.dataclass(frozen=True)
class ChatTemplate:
    """A chat template source plus provenance (path/sha256) for the dataset manifest."""

    source: str
    sha256: str
    path: str | None = None

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "ChatTemplate":
        path = pathlib.Path(path)
        source = path.read_text(encoding="utf-8")
        return cls(source=source, sha256=template_sha256(source), path=str(path))

    def render(self, messages: Sequence[dict], *, tools: Sequence[dict] | None = None,
               thinking_mode: str = "thinking", drop_thinking: bool = True,
               add_generation_prompt: bool = False,
               reasoning_effort: int | str | None = None) -> str:
        if thinking_mode not in ("thinking", "chat"):
            raise ValueError(f"thinking_mode must be 'thinking' or 'chat', got {thinking_mode!r}")
        env = _make_env()
        try:
            template = env.from_string(self.source)
        except Exception as exc:  # untrusted template syntax
            raise ChatTemplateError(f"failed to parse chat template: {exc}") from exc
        try:
            return template.render(
                messages=list(messages),
                tools=list(tools) if tools else None,
                thinking_mode=thinking_mode,
                enable_thinking=(thinking_mode != "chat"),
                drop_thinking=bool(drop_thinking),
                add_generation_prompt=bool(add_generation_prompt),
                reasoning_effort=reasoning_effort,
            )
        except ChatTemplateError:
            raise
        except Exception as exc:
            raise ChatTemplateError(f"failed to render chat template: {exc}") from exc

    def describe(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": len(self.source.encode("utf-8"))}


def template_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_chat_template(path: str | pathlib.Path) -> ChatTemplate:
    return ChatTemplate.from_file(path)


# --------------------------------------------------------------------------- messages
def strip_reasoning(messages: Sequence[dict]) -> list[dict]:
    """Copy messages with ``reasoning_content`` / ``thinking`` removed.

    Use when the held-out trajectories should be scored on visible content only (e.g. teacher
    logs that include hidden CoT you do not want to train on).
    """
    out: list[dict] = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if k not in ("reasoning_content", "thinking")}
        out.append(clean)
    return out


def count_expected_blocks(messages: Sequence[dict]) -> dict:
    """How many assistant turns / tool results the template should render.

    Tool results come from ``role=="tool"`` messages or ``tool_result`` content blocks in user
    messages; the template merges/reorders them but the count is preserved.
    """
    assistant = 0
    tool_results = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            assistant += 1
        elif role == "tool":
            tool_results += 1
        elif role == "user":
            content = msg.get("content")
            if isinstance(content, list):
                tool_results += sum(
                    1 for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                )
    return {"assistant_messages": assistant, "tool_results": tool_results}


# --------------------------------------------------------------------------- tagging
@dataclasses.dataclass
class TaggedTrajectory:
    ids: np.ndarray
    types: np.ndarray
    offsets: np.ndarray
    stats: dict[str, Any]

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])


def role_marker_ids(tok) -> dict[str, int]:
    """Resolve the V4.1 role headers to single token ids."""
    vocab = tok.get_vocab()
    ids: dict[str, int] = {}
    missing: list[str] = []
    for marker in BOUNDARY_MARKERS:
        token_id = vocab.get(marker)
        if token_id is None:
            missing.append(marker)
        else:
            ids[marker] = int(token_id)
    if missing:
        raise ChatTemplateError(
            "tokenizer is missing V4.1 role markers "
            f"{missing}; this is not the pinned DeepSeek-V4.1-Flash tokenizer "
            "(see docs/tokenizer-contract.md)"
        )
    return ids


def tag_trajectory(tok, text: str, *, expected_assistant_messages: int | None = None,
                   expected_tool_results: int | None = None) -> TaggedTrajectory:
    """Tokenize rendered trajectory text and tag ASSISTANT/TOOL/TEXT token spans."""
    ids, offsets = encode_text(tok, text)
    types = np.full(ids.shape[0], TYPE_TEXT, dtype=np.uint8)
    markers = role_marker_ids(tok)

    boundary_ids = np.fromiter(markers.values(), dtype=np.int64, count=len(markers))
    boundary_positions = np.nonzero(np.isin(ids, boundary_ids))[0]
    assistant_positions = np.nonzero(ids == markers[ASSISTANT_MARKER])[0]
    for pos in assistant_positions:
        k = int(np.searchsorted(boundary_positions, int(pos), side="right"))
        end = int(boundary_positions[k]) if k < boundary_positions.size else int(ids.shape[0])
        types[int(pos):end] = TYPE_ASSISTANT

    tool_spans = [(m.start(0), m.end(0)) for m in TOOL_RESULT_RE.finditer(text)]
    tool_token_count = 0
    overlap = 0
    for token_idx in char_spans_to_token_spans(offsets, tool_spans):
        if token_idx.size == 0:
            continue
        overlap += int(np.count_nonzero(types[token_idx] == TYPE_ASSISTANT))
        types[token_idx] = TYPE_TOOL
        tool_token_count += int(token_idx.size)

    stats = {
        "n_tokens": int(ids.shape[0]),
        "assistant_turns": int(assistant_positions.size),
        "tool_results": len(tool_spans),
        "assistant_tokens": int(np.count_nonzero(types == TYPE_ASSISTANT)),
        "tool_tokens": int(np.count_nonzero(types == TYPE_TOOL)),
        "text_tokens": int(np.count_nonzero(types == TYPE_TEXT)),
        "assistant_tool_overlap_tokens": overlap,
        "prompt_tokens_before_first_assistant": (
            int(assistant_positions[0]) if assistant_positions.size else int(ids.shape[0])
        ),
    }
    if expected_assistant_messages is not None and expected_assistant_messages != stats["assistant_turns"]:
        stats["assistant_turn_count_mismatch"] = (
            f"rendered {stats['assistant_turns']} assistant headers for "
            f"{expected_assistant_messages} assistant messages (consecutive assistant turns merge)"
        )
    if expected_tool_results is not None and expected_tool_results != stats["tool_results"]:
        stats["tool_result_count_mismatch"] = (
            f"rendered {stats['tool_results']} <tool_result> blocks for "
            f"{expected_tool_results} expected tool results"
        )
    return TaggedTrajectory(ids=ids, types=types, offsets=offsets, stats=stats)


def render_and_tag(tok, template: ChatTemplate, messages: Sequence[dict], *,
                   tools: Sequence[dict] | None = None, thinking_mode: str = "thinking",
                   drop_thinking: bool = True, add_generation_prompt: bool = False,
                   drop_reasoning: bool = False,
                   reasoning_effort: int | str | None = None) -> TaggedTrajectory:
    """Render one trajectory with the pinned template and tag its token spans."""
    if drop_reasoning:
        messages = strip_reasoning(messages)
    expected = count_expected_blocks(messages)
    text = template.render(
        messages,
        tools=tools,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
    )
    tagged = tag_trajectory(
        tok,
        text,
        expected_assistant_messages=expected["assistant_messages"],
        expected_tool_results=expected["tool_results"],
    )
    tagged.stats["chars"] = len(text)
    tagged.stats["bytes"] = len(text.encode("utf-8"))
    tagged.stats["template_sha256"] = template.sha256
    return tagged


def tag_plain_text(tok, text: str) -> TaggedTrajectory:
    """Tokenize a plain-text eval record: every token is TEXT."""
    ids, offsets = encode_text(tok, text)
    types = np.full(ids.shape[0], TYPE_TEXT, dtype=np.uint8)
    return TaggedTrajectory(
        ids=ids,
        types=types,
        offsets=offsets,
        stats={"n_tokens": int(ids.shape[0]), "chars": len(text),
               "bytes": len(text.encode("utf-8")), "assistant_tokens": 0, "tool_tokens": 0},
    )


def validate_byte_accounting(tok, nodes: TaggedTrajectory, token_bytes: np.ndarray) -> dict:
    """``sum(token_bytes[ids])`` must equal the rendered text's UTF-8 byte length."""
    got = int(token_bytes[nodes.ids].sum()) if nodes.ids.size else 0
    want = int(nodes.stats.get("bytes", 0))
    return {"ok": got == want, "got": got, "want": want}
