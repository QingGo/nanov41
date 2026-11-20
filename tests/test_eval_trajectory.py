# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""End-to-end tokenizer/template tests for the eval data pipeline.

These need the *pinned* DeepSeek-V4.1-Flash artifacts and are skipped otherwise:

    export NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json
    export NANOV41_DS_CHAT_TEMPLATE=/tmp/dsv41_chat_template.jinja
    pytest tests/test_eval_trajectory.py -q

The artifacts are pinned in docs/tokenizer-contract.md (tokenizer.json 6,367,257 B, sha256
c90dfa01..., chat_template.jinja 15,348 B).  They are deliberately not vendored in this repo.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from nanov41.eval import EvalDataset, EvalSettings, FakeProvider, evaluate_dataset
from nanov41.eval import tokenizer as etok
from nanov41.eval.chat import (
    load_chat_template,
    render_and_tag,
    role_marker_ids,
    tag_plain_text,
    tag_trajectory,
    validate_byte_accounting,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
TEMPLATE = os.environ.get("NANOV41_DS_CHAT_TEMPLATE")

requires_tokenizer = pytest.mark.skipif(not TOKENIZER, reason="set NANOV41_DS_TOKENIZER_JSON")
requires_template = pytest.mark.skipif(
    not (TOKENIZER and TEMPLATE), reason="set NANOV41_DS_TOKENIZER_JSON and NANOV41_DS_CHAT_TEMPLATE"
)


@pytest.fixture(scope="module")
def tok():
    return etok.load_tokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def byte_table(tok):
    table, _ = etok.token_bytes_from_tokenizer(tok)
    return table


@pytest.fixture(scope="module")
def template():
    return load_chat_template(TEMPLATE)


SYSTEM = {"role": "system", "content": "You are a careful coding agent."}
USER = {"role": "user", "content": "Find the files in the current directory."}
TOOL_CALL = {
    "role": "assistant",
    "reasoning_content": "I should list files first.",
    "content": "",
    "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "list_files", "arguments": {"path": "."}},
    }],
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"}
FINAL = {"role": "assistant", "content": "There are 2 files: a.py and b.py."}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]


# --------------------------------------------------------------------------- token bytes
@requires_tokenizer
def test_pinned_contract_and_byte_table(tok, byte_table):
    assert tok.get_vocab_size() == 129_280
    assert tok.id_to_token(2) == "<｜▁pad▁｜>"
    assert byte_table.dtype == np.uint8
    assert int(byte_table.max()) <= 255
    samples = [
        "Hello world, 你好世界 👋",
        "def f(x):\n    return x + 1\n",
        "中文测试，标点。🙂",
        "a" * 1000,
        "<｜User｜>hi<｜Assistant｜>hello<｜end▁of▁sentence｜>",
    ]
    stats = etok.validate_token_bytes(tok, byte_table, samples)
    assert stats == {"checked": len(samples), "bad": 0, "examples": []}
    contract = etok.contract_info(tok, TOKENIZER)
    assert contract["pinned_v41_flash"] is True
    assert contract["sha256"] == "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


@requires_tokenizer
def test_plain_text_tagging(tok, byte_table):
    text = "The quick brown fox jumps over the lazy dog."
    tagged = tag_plain_text(tok, text)
    assert tagged.stats["n_tokens"] > 0
    assert np.all(tagged.types == 1)
    assert validate_byte_accounting(tok, tagged, byte_table)["ok"], "byte accounting must be exact"


# --------------------------------------------------------------------------- trajectory tagging
@requires_tokenizer
def test_role_markers_are_single_tokens(tok):
    markers = role_marker_ids(tok)
    assert markers["<｜Assistant｜>"] == 128_804
    assert markers["<｜User｜>"] == 128_803
    ids = tok.encode("<｜User｜>x<｜Assistant｜>y", add_special_tokens=False).ids
    assert markers["<｜User｜>"] in ids and markers["<｜Assistant｜>"] in ids


@requires_template
def test_render_and_tag_assistant_and_tool_spans(tok, byte_table, template):
    tagged = render_and_tag(tok, template, [SYSTEM, USER, TOOL_CALL, TOOL_RESULT, FINAL],
                            tools=TOOLS, drop_thinking=False)
    stats = tagged.stats
    assert stats["assistant_turns"] == 2
    assert stats["tool_results"] == 1
    assert stats["assistant_tokens"] > 0 and stats["tool_tokens"] > 0
    assert stats["text_tokens"] > 0
    assert stats["assistant_tokens"] + stats["tool_tokens"] + stats["text_tokens"] == stats["n_tokens"]
    assert "assistant_turn_count_mismatch" not in stats
    assert "tool_result_count_mismatch" not in stats
    assert validate_byte_accounting(tok, tagged, byte_table)["ok"]

    # every assistant marker starts an ASSISTANT span; user markers stay TEXT
    vocab = tok.get_vocab()
    assistant_pos = np.nonzero(tagged.ids == vocab["<｜Assistant｜>"])[0]
    user_pos = np.nonzero(tagged.ids == vocab["<｜User｜>"])[0]
    for pos in assistant_pos:
        assert tagged.types[pos] == 2
    for pos in user_pos:
        assert tagged.types[pos] == 1
    # end-of-sentence inside an assistant turn is assistant content, not text
    eos_pos = np.nonzero(tagged.ids == vocab["<｜end▁of▁sentence｜>"])[0]
    assert eos_pos.size >= 1
    assert all(tagged.types[p] == 2 for p in eos_pos)


@requires_template
def test_drop_reasoning_shrinks_assistant_span(tok, template):
    conv = [SYSTEM, USER, TOOL_CALL, TOOL_RESULT, FINAL]
    with_reasoning = render_and_tag(tok, template, conv, tools=TOOLS, drop_thinking=False)
    without = render_and_tag(tok, template, conv, tools=TOOLS, drop_thinking=False,
                             drop_reasoning=True)
    assert without.stats["assistant_tokens"] < with_reasoning.stats["assistant_tokens"]
    # the template itself keeps reasoning when asked to; drop_reasoning is what removes it
    rendered = template.render(conv, tools=TOOLS, drop_thinking=False, thinking_mode="thinking")
    assert "I should list files first." in rendered


@requires_tokenizer
def test_tag_plain_trajectory_text(tok):
    """A trajectory rendered *without* a template still tags: no assistant markers -> all TEXT."""
    tagged = tag_trajectory(tok, "plain text with no role markers")
    assert np.all(tagged.types == 1)


# --------------------------------------------------------------------------- dataset builder
@requires_template
def test_make_eval_slices_synthetic_end_to_end(tmp_path):
    out = tmp_path / "data"
    env = dict(os.environ, NANOV41_DS_TOKENIZER_JSON=TOKENIZER, NANOV41_DS_CHAT_TEMPLATE=TEMPLATE)
    for domain, kind in (("web", "text"), ("agent", "trajectory")):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "make_eval_slices.py"),
             "--out", str(out), "--domain", domain, "--kind", kind,
             "--synthetic", "6", "--max-tokens", "20000", "--window", "256",
             "--shard-tokens", "12000", "--drop-thinking", "false"],
            cwd=ROOT, capture_output=True, text=True, env=env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
    ds = EvalDataset.open(out)
    assert "web" in ds.domain_names() and "agent" in ds.domain_names()
    agent = ds.domain("agent")
    stats = agent.stats["by_type"]
    assert stats["assistant"]["tokens"] > 0
    assert stats["tool_output"]["tokens"] > 0
    assert agent.window == 256
    tokens, types = ds.read_shard(agent.shards[0])
    assert types is not None and set(np.unique(types)) <= {1, 2, 3}
    if agent.n_tokens > 256:
        report = evaluate_dataset(FakeProvider(129_280, "uniform"), ds,
                                  domains=["agent"],
                                  settings=EvalSettings(window=256, max_tokens_per_domain=1024))
        metrics = report["dataset"]["domains"]["agent"]["metrics"]
        assert metrics["assistant"]["tokens"] > 0
        assert metrics["tool_output"]["tokens"] > 0


@requires_template
def test_make_eval_slices_missing_tokenizer_fails(tmp_path):
    env = dict(os.environ, NANOV41_DS_TOKENIZER_JSON=TOKENIZER, NANOV41_DS_CHAT_TEMPLATE=TEMPLATE)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_eval_slices.py"),
         "--out", str(tmp_path / "data"), "--domain", "web", "--kind", "text",
         "--synthetic", "4", "--tokenizer", "/nonexistent/tokenizer.json"],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode != 0
    assert "tokenizer.json not found" in (proc.stderr + proc.stdout)
