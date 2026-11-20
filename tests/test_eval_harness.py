# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU tests for the T0 eval harness: data contract, BPB math, CLI, reserved CORE-lite.

Everything here runs without a tokenizer, GPU or checkpoint: shards are written with synthetic
token ids and a hand-made byte table, and logits come from `FakeProvider` whose NLLs are known
in closed form (uniform -> ln(vocab) per token, oracle -> 0).  Tokenizer/template end-to-end
tests live in tests/test_eval_trajectory.py and are skipped when the pinned artifacts are not set.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nanov41.eval import (
    BPBAccumulator,
    DatasetWriter,
    EvalDataset,
    EvalSettings,
    FakeProvider,
    TYPE_ASSISTANT,
    TYPE_IGNORE,
    TYPE_TEXT,
    TYPE_TOOL,
    check_provider_dataset_compat,
    evaluate_dataset,
    format_report_table,
    iter_window_starts,
    token_nll_batch,
)
from nanov41.eval.core import (
    CORE_LITE_SPECS,
    CORE_LITE_TASKS,
    CoreDataMissing,
    choose_continuation,
    evaluate_core_lite,
    load_core_examples,
    score_continuation,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LN2 = math.log(2.0)


# --------------------------------------------------------------------------- helpers
def make_dataset(root: pathlib.Path, *, vocab: int = 512, n_tokens: int = 4096,
                 window: int = 512, seed: int = 0,
                 domains: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
                 token_bytes: np.ndarray | None = None) -> EvalDataset:
    rng = np.random.default_rng(seed)
    token_bytes = np.ones(vocab, dtype=np.uint8) if token_bytes is None else token_bytes
    writer = DatasetWriter(root, token_bytes=token_bytes,
                           tokenizer_meta={"vocab_size": vocab, "pinned_v41_flash": False})
    if domains is None:
        tokens = rng.integers(0, vocab, size=n_tokens, dtype=np.uint32)
        domains = {"toy": (tokens, None)}
    for name, (tokens, types) in domains.items():
        writer.add_domain(name, tokens, types, eval={"window": window})
    return EvalDataset.open(root)


def run_tool(name: str, args: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / name), *args],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=300,
    )


# --------------------------------------------------------------------------- data contract
def test_dataset_writer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 512, size=1000, dtype=np.uint32)
    types = np.full(1000, TYPE_TEXT, dtype=np.uint8)
    types[100:200] = TYPE_ASSISTANT
    types[300:320] = TYPE_TOOL
    ds = make_dataset(tmp_path, domains={
        "web": (tokens[:500], None),
        "agent": (tokens, types),
    })
    assert sorted(ds.domain_names()) == ["agent", "web"]
    assert ds.token_bytes.shape == (512,)

    web = ds.domain("web")
    agent = ds.domain("agent")
    assert web.shards[0].types is None  # all-TEXT mask is not stored
    assert agent.shards[0].types is not None
    loaded, loaded_types = ds.read_shard(agent.shards[0])
    assert np.array_equal(np.asarray(loaded, dtype=np.uint32), tokens)
    assert np.array_equal(np.asarray(loaded_types), types)
    assert agent.stats["by_type"]["assistant"]["tokens"] == 100
    assert agent.stats["by_type"]["tool_output"]["tokens"] == 20
    info = ds.verify(check_hashes=True)
    assert info["ok"] and info["domains"]["agent"]["tokens"] == 1000


def test_dataset_writer_rejects_mixed_token_bytes(tmp_path):
    make_dataset(tmp_path)
    other = (np.arange(512, dtype=np.uint8) % 3 + 1).astype(np.uint8)
    with pytest.raises(ValueError, match="different token_bytes"):
        DatasetWriter(tmp_path, token_bytes=other)


def test_dataset_open_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no eval dataset"):
        EvalDataset.open(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "bogus", "domains": {}}))
    with pytest.raises(ValueError, match="unsupported eval dataset format"):
        EvalDataset.open(tmp_path)


def test_shard_split_and_stats(tmp_path):
    tokens = np.arange(100, dtype=np.uint32) % 512
    writer = DatasetWriter(tmp_path, token_bytes=np.ones(512, np.uint8),
                           tokenizer_meta={"vocab_size": 512})
    entry = writer.add_domain("d", tokens, None, shard_tokens=30)
    assert [s["n_tokens"] for s in entry["shards"]] == [30, 30, 30, 10]
    assert entry["stats"]["n_tokens"] == 100


# --------------------------------------------------------------------------- windowing
@pytest.mark.parametrize("n,window,max_tokens,want", [
    (10, 4, None, [(0, 4), (4, 8)]),
    (8, 4, None, [(0, 4), (4, 8)]),
    (10, 4, 6, [(0, 4)]),
    (3, 4, None, []),
])
def test_iter_window_starts(n, window, max_tokens, want):
    assert iter_window_starts(n, window, max_tokens) == want


def test_iter_window_starts_rejects_tiny_window():
    with pytest.raises(ValueError, match="window must be >= 2"):
        iter_window_starts(10, 1)


# --------------------------------------------------------------------------- NLL / BPB math
def test_token_nll_uniform_is_ln_vocab():
    import torch

    vocab = 512
    ids = torch.randint(0, vocab, (2, 32))
    nll = token_nll_batch(FakeProvider(vocab, "uniform"), ids)
    assert nll.shape == (2, 31)
    assert np.allclose(nll, math.log(vocab), rtol=1e-6, atol=1e-6)


def test_token_nll_oracle_is_zero():
    import torch

    ids = torch.randint(0, 512, (1, 16))
    nll = token_nll_batch(FakeProvider(512, "oracle"), ids)
    assert np.allclose(nll, 0.0, atol=1e-9)


def test_uniform_provider_bpb_is_log2_vocab(tmp_path):
    vocab = 512
    ds = make_dataset(tmp_path, vocab=vocab, n_tokens=4096, window=512)
    report = evaluate_dataset(FakeProvider(vocab, "uniform"), ds,
                              settings=EvalSettings(window=512, batch_size=2))
    entry = report["dataset"]["domains"]["toy"]
    assert entry["tokens_processed"] == 4096
    assert entry["tokens_scored"] == 4096 - 4096 // 512  # first token of each window is context
    assert entry["tokens_dropped"] == 0
    assert entry["metrics"]["all"]["bpb"] == pytest.approx(math.log2(vocab), rel=1e-6)
    assert report["summary"]["domains"] == 1
    assert report["schema"] == "nanov41-eval-report/1"
    table = format_report_table(report)
    assert "toy" in table and "macro bpb" in table


def test_trajectory_type_metrics(tmp_path):
    rng = np.random.default_rng(1)
    n = 4096
    tokens = rng.integers(0, 512, size=n, dtype=np.uint32)
    types = np.full(n, TYPE_TEXT, dtype=np.uint8)
    types[0:2048] = TYPE_ASSISTANT
    types[2048:3072] = TYPE_TOOL
    ds = make_dataset(tmp_path, domains={"agent": (tokens, types)}, window=512)
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=512))
    metrics = report["dataset"]["domains"]["agent"]["metrics"]
    # window starts (0,512,1024,1536,2048,2560,3072,3584) are context-only
    assert metrics["assistant"]["tokens"] == 2048 - 4
    assert metrics["tool_output"]["tokens"] == 1024 - 2
    assert metrics["text"]["tokens"] == 1024 - 2
    assert metrics["assistant"]["bpb"] == pytest.approx(math.log2(512), rel=1e-6)


def test_ignore_type_is_excluded(tmp_path):
    tokens = np.arange(1024, dtype=np.uint32) % 512
    types = np.full(1024, TYPE_IGNORE, dtype=np.uint8)
    types[512:1024] = TYPE_ASSISTANT
    ds = make_dataset(tmp_path, domains={"d": (tokens, types)}, window=256)
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds, settings=EvalSettings(window=256))
    metrics = report["dataset"]["domains"]["d"]["metrics"]
    assert set(metrics) == {"all", "assistant"}
    assert metrics["all"]["tokens"] == 512 - 2
    assert metrics["assistant"]["tokens"] == 512 - 2


def test_accumulator_bootstrap_ci(tmp_path):
    ds = make_dataset(tmp_path, n_tokens=8192, window=512)
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=512, bootstrap=50))
    ci = report["dataset"]["domains"]["toy"]["bootstrap_ci"]["all"]
    assert ci[0] <= ci[1]
    assert ci[0] == pytest.approx(math.log2(512), rel=0.05)


def test_accumulator_unknown_type_raises():
    acc = BPBAccumulator()
    with pytest.raises(ValueError, match="unknown token type"):
        acc.add(np.zeros(4), np.zeros(4, dtype=np.int64), np.full(4, 9, dtype=np.uint8),
                np.ones(512, np.uint8))


def test_domain_shorter_than_window_raises(tmp_path):
    ds = make_dataset(tmp_path, n_tokens=100, window=128)
    with pytest.raises(ValueError, match="no full 128-token window"):
        evaluate_dataset(FakeProvider(512, "uniform"), ds, settings=EvalSettings(window=128))


def test_max_tokens_budget(tmp_path):
    ds = make_dataset(tmp_path, n_tokens=8192, window=512)
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=512, max_tokens_per_domain=1024))
    entry = report["dataset"]["domains"]["toy"]
    assert entry["tokens_processed"] == 1024
    assert entry["tokens_scored"] == 1022


# --------------------------------------------------------------------------- providers / checks
def test_vocab_mismatch_check(tmp_path):
    ds = make_dataset(tmp_path)
    check_provider_dataset_compat(FakeProvider(512), ds)
    with pytest.raises(ValueError, match="provider vocab_size"):
        check_provider_dataset_compat(FakeProvider(256), ds)
    check_provider_dataset_compat(FakeProvider(256), ds, allow_mismatch=True)


def test_model_provider_tiny_cpu():
    pytest.importorskip("torch")
    from nanov41.eval.provider import ModelProvider

    provider = ModelProvider(config_name="tiny", random_init=True, dtype="float32", device="cpu")
    try:
        import torch

        ids = torch.randint(0, provider.vocab_size, (2, 8))
        logits = provider.logits(ids)
        assert tuple(logits.shape) == (2, 8, provider.vocab_size)
        assert provider.flops_per_token and provider.flops_per_token > 0
        assert provider.describe()["config"] == "tiny"
    finally:
        provider.close()


# --------------------------------------------------------------------------- CLI
def test_cli_fake_provider_end_to_end(tmp_path):
    ds_root = tmp_path / "ds"
    make_dataset(ds_root, n_tokens=2048, window=128)
    proc = run_tool("eval_t0.py", ["--dataset", str(ds_root), "--provider", "fake",
                                   "--fake-kind", "uniform", "--window", "128",
                                   "--json", "--no-save"])
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    entry = report["dataset"]["domains"]["toy"]
    assert entry["metrics"]["all"]["bpb"] == pytest.approx(math.log2(512), rel=1e-5)
    assert entry["tokens_processed"] == 2048
    assert report["provider"]["provider"] == "fake"
    assert not (ds_root / "reports").exists()


def test_cli_oracle_provider_zero_bpb(tmp_path):
    ds_root = tmp_path / "ds"
    make_dataset(ds_root, n_tokens=1024, window=128)
    proc = run_tool("eval_t0.py", ["--dataset", str(ds_root), "--provider", "fake",
                                   "--fake-kind", "oracle", "--window", "128",
                                   "--json", "--no-save"])
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["dataset"]["domains"]["toy"]["metrics"]["all"]["bpb"] == pytest.approx(0.0, abs=1e-9)


def test_cli_list_domains(tmp_path):
    make_dataset(tmp_path, domains={"a": (np.zeros(64, dtype=np.uint32), None),
                                    "b": (np.zeros(64, dtype=np.uint32), None)})
    proc = run_tool("eval_t0.py", ["--dataset", str(tmp_path), "--list-domains"])
    assert proc.returncode == 0, proc.stderr
    assert "a" in proc.stdout and "b" in proc.stdout


def test_cli_model_requires_ckpt(tmp_path):
    make_dataset(tmp_path)
    proc = run_tool("eval_t0.py", ["--dataset", str(tmp_path), "--provider", "model",
                                   "--config", "tiny"])
    assert proc.returncode == 2
    assert "--ckpt" in proc.stderr


def test_cli_t1_reserved_json_purity(tmp_path):
    """--json must stay machine-readable even when the reserved T1 data is missing."""
    ds_root = tmp_path / "ds"
    make_dataset(ds_root, n_tokens=512, window=128)
    env = {k: v for k, v in os.environ.items() if k != "NANOV41_DS_TOKENIZER_JSON"}
    proc = run_tool("eval_t0.py", ["--dataset", str(ds_root), "--provider", "fake",
                                   "--window", "128", "--tier", "t1", "--json", "--no-save"],
                    env=env)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)  # would raise if the CLI mixed log lines into stdout
    assert report["t1"]["status"] == "skipped"
    assert "reserved interface" in proc.stderr


def test_cli_unknown_domain(tmp_path):
    make_dataset(tmp_path)
    proc = run_tool("eval_t0.py", ["--dataset", str(tmp_path), "--provider", "fake",
                                   "--domains", "nope"])
    assert proc.returncode != 0
    assert "unknown domains" in (proc.stderr + proc.stdout)


# --------------------------------------------------------------------------- reserved CORE-lite
class _ToyTokenizer:
    """Character-level stand-in: encode() only (enough for CORE continuation scoring)."""

    def encode(self, text, add_special_tokens=False):
        ids = [ord(ch) % 512 for ch in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return SimpleNamespace(ids=ids, offsets=offsets)


def test_core_registry_and_missing_data(tmp_path):
    assert "hellaswag" in CORE_LITE_TASKS and len(CORE_LITE_TASKS) >= 6
    with pytest.raises(CoreDataMissing, match="CORE-lite data not found"):
        load_core_examples(tmp_path, CORE_LITE_SPECS["hellaswag"])


def test_core_loader_and_scoring(tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    with open(core_dir / "hellaswag.jsonl", "w") as f:
        for label in (0, 1):
            f.write(json.dumps({"context": "The sky is", "choices": ["blue", "green"],
                                "label": label}) + "\n")
    examples = load_core_examples(tmp_path, CORE_LITE_SPECS["hellaswag"])
    assert len(examples) == 2 and examples[0].choices == ("blue", "green")

    provider = FakeProvider(512, "uniform")
    info = score_continuation(provider, [1, 2, 3], [4, 5], normalize="token")
    assert info == pytest.approx(math.log(512), rel=1e-6)  # 2 tokens * ln(512), normalised
    pick = choose_continuation(provider, [1, 2, 3], [[4], [5]])
    assert pick["index"] == 0 and len(pick["scores"]) == 2

    report = evaluate_core_lite(provider, _ToyTokenizer(), tmp_path,
                                specs=[CORE_LITE_SPECS["hellaswag"]],
                                max_examples=2, normalize="token")
    # all scores tie -> pick 0 -> example with label 0 is correct
    assert report["tasks"]["hellaswag"]["accuracy"] == pytest.approx(0.5)
    assert report["tasks"]["hellaswag"]["n_examples"] == 2
    assert report["summary"]["n_tasks"] == 1


# --------------------------------------------------------------------------- reference mixes (Marin)
def test_reference_mixes_are_valid_and_sum_to_one():
    from nanov41.eval.mix import REFERENCE_MIXES, validate_mix

    for name, mix in REFERENCE_MIXES.items():
        validate_mix(mix["weights"])
        assert sum(mix["weights"].values()) == pytest.approx(1.0)
        assert mix["source"] and "marin" in mix["source"]
        if mix.get("source_weights"):
            assert sum(mix["source_weights"].values()) == pytest.approx(1.0)


def test_validate_mix_rejects_bad_weights():
    from nanov41.eval.mix import validate_mix

    with pytest.raises(ValueError, match="sum to 1"):
        validate_mix({"web": 0.5})
    with pytest.raises(ValueError, match="finite and >= 0"):
        validate_mix({"web": -0.1, "code": 1.1})
    with pytest.raises(ValueError, match="must not be empty"):
        validate_mix({})


def test_mix_bpb_token_weighted_math():
    from nanov41.eval.mix import mix_bpb

    def result(nll, tokens, nbytes):
        return SimpleNamespace(metrics={"all": {
            "nll_nats": nll, "tokens": tokens, "bytes": nbytes,
            "bpb": nll / LN2 / nbytes,
        }})

    results = {"web": result(100.0, 1000, 4000), "code": result(200.0, 1000, 3000)}
    out = mix_bpb(results, {"web": 0.75, "code": 0.25})
    # token-proportion mixture: weighted NLL/token divided by weighted bytes/token
    expected = (0.75 * (100.0 / 1000) + 0.25 * (200.0 / 1000)) / LN2 / (0.75 * 4 + 0.25 * 3)
    assert out["bpb"] == pytest.approx(expected)
    assert out["coverage"] == pytest.approx(1.0)
    assert out["missing"] == []

    partial = mix_bpb({"web": results["web"]}, {"web": 0.75, "code": 0.25})
    assert partial["coverage"] == pytest.approx(0.75)
    assert partial["missing"] == ["code"]
    assert partial["bpb"] == pytest.approx(100.0 / 1000 / LN2 / 4)  # renormalised to web only


def test_auto_reference_mix_phase1_in_dataset(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 512, size=512, dtype=np.uint32)
    ds = make_dataset(tmp_path, domains={d: (tokens, None) for d in ("web", "code", "math")},
                      n_tokens=512, window=128)
    assert "marin-8b-phase1" in ds.mixes
    assert "marin-8b-jellyfish" not in ds.mixes  # needs the six jellyfish domains
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=128))
    mix = report["mixes"]["marin-8b-phase1"]
    assert mix["coverage"] == pytest.approx(1.0)
    assert mix["missing"] == []
    assert mix["bpb"] == pytest.approx(math.log2(512), rel=1e-6)
    assert mix["source"] and mix["source_weights"]
    assert "mix marin-8b-phase1" in format_report_table(report)


def test_auto_reference_mix_jellyfish_full_coverage(tmp_path):
    tokens = np.arange(256, dtype=np.uint32) % 512
    writer = DatasetWriter(tmp_path, token_bytes=np.ones(512, np.uint8),
                           tokenizer_meta={"vocab_size": 512})
    for domain in ("web", "code", "math", "academic", "wikipedia", "stackexchange"):
        writer.add_domain(domain, tokens, None, eval={"window": 64})
    ds = EvalDataset.open(tmp_path)
    assert set(ds.mixes["marin-8b-jellyfish"]["weights"]) == {
        "web", "academic", "math", "stackexchange", "code", "wikipedia"
    }
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=64))
    assert report["mixes"]["marin-8b-jellyfish"]["coverage"] == pytest.approx(1.0)


def test_manual_partial_mix_reported_with_coverage(tmp_path):
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 512, size=256, dtype=np.uint32)
    writer = DatasetWriter(tmp_path, token_bytes=np.ones(512, np.uint8),
                           tokenizer_meta={"vocab_size": 512})
    writer.add_domain("web", tokens, None, eval={"window": 64})
    writer.add_domain("code", tokens, None, eval={"window": 64})
    writer.set_mix("custom", {"web": 0.7, "code": 0.2, "math": 0.1}, source="unit-test")
    ds = EvalDataset.open(tmp_path)
    report = evaluate_dataset(FakeProvider(512, "uniform"), ds,
                              settings=EvalSettings(window=64))
    mix = report["mixes"]["custom"]
    assert mix["coverage"] == pytest.approx(0.9)
    assert mix["missing"] == ["math"]
    assert mix["bpb"] == pytest.approx(math.log2(512), rel=1e-6)
