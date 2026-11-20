# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""P1: training-capable Engram — bf16 table, deterministic row grads, sparse Adam, train.py wiring.

The non-gated tests exercise the table/optimizer directly on CPU (no tokenizer needed).  The gated
tests build the real tiny model with the reference hash state and run forward/backward + a train.py
smoke, which needs the pinned tokenizer:

    NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json pytest tests/test_engram_train.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import nanov41.nano_engram as ne

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
requires_tokenizer = pytest.mark.skipif(not TOKENIZER, reason="set NANOV41_DS_TOKENIZER_JSON")


# --------------------------------------------------------------------------- deterministic reduction
def test_segment_sum_rows_matches_loop():
    torch.manual_seed(0)
    ids = torch.tensor([5, 3, 5, 5, 3, 7, 3])
    grads = torch.randn(7, 4)
    rows, sums = ne.segment_sum_rows(ids, grads)
    assert rows.tolist() == [3, 5, 7]
    for i, row in enumerate(rows.tolist()):
        want = grads[(ids == row)].sum(dim=0)
        assert torch.allclose(sums[i], want, atol=1e-6), (row, sums[i], want)


def test_segment_sum_rows_empty():
    rows, sums = ne.segment_sum_rows(torch.empty(0, dtype=torch.long), torch.empty(0, 4))
    assert rows.numel() == 0 and sums.shape == (0, 4)


# --------------------------------------------------------------------------- table gradients
def test_table_backward_matches_dense_autograd():
    torch.manual_seed(0)
    table = ne.NanoEngramTable(64, 8)
    ids = torch.randint(0, 64, (3, 5, 4))
    coef = torch.randn(3, 5, 4, 8)
    values = table(ids)
    (values * coef).sum().backward()
    table.collect_row_grads()
    rows, grads = table.merged_row_grads()

    ref = table.weight.detach().clone().requires_grad_(True)
    (ref[ids] * coef).sum().backward()
    assert rows.numel() == torch.unique(ids).numel()
    for i, row in enumerate(rows.tolist()):
        assert torch.allclose(grads[i], ref.grad[row], atol=1e-6), row
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[torch.unique(ids)] = False
    assert torch.count_nonzero(ref.grad[untouched]) == 0


def test_table_eval_mode_does_not_record():
    table = ne.NanoEngramTable(32, 8)
    with torch.no_grad():
        out = table(torch.randint(0, 32, (2, 4)))
    assert out.shape == (2, 4, 8)
    assert table._records == []
    assert table.merged_row_grads()[0].numel() == 0


def test_table_forward_out_of_range_zeroed():
    table = ne.NanoEngramTable(8, 4)
    with torch.no_grad():
        out = table(torch.tensor([[0, 7, 99, -1]]))
    assert torch.count_nonzero(out[0, 2]) == 0 and torch.count_nonzero(out[0, 3]) == 0


# --------------------------------------------------------------------------- sparse Adam
def _one_step(table: ne.NanoEngramTable, ids: torch.Tensor, coef: torch.Tensor, opt: ne.SparseRowAdam):
    values = table(ids)
    (values * coef).sum().backward()
    opt.step()


def test_sparse_adam_updates_only_touched_rows_and_is_deterministic():
    ids = torch.randint(0, 64, (2, 4, 3))
    coef = torch.randn(2, 4, 3, 8)
    torch.manual_seed(7)
    t1, t2 = ne.NanoEngramTable(64, 8), ne.NanoEngramTable(64, 8)
    t2.weight.data.copy_(t1.weight.data)
    init = t1.weight.data.clone()
    opt1, opt2 = ne.SparseRowAdam([t1], lr=1e-2), ne.SparseRowAdam([t2], lr=1e-2)
    _one_step(t1, ids, coef, opt1)
    _one_step(t2, ids, coef, opt2)
    touched = torch.unique(ids)
    assert torch.equal(t1.weight.data, t2.weight.data), "sparse Adam must be deterministic"
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[touched] = False
    assert torch.equal(t1.weight.data[untouched], init[untouched]), "untouched rows must not move"
    assert not torch.equal(t1.weight.data[touched], init[touched])


def test_sparse_adam_state_dict_roundtrip():
    ids = torch.randint(0, 64, (2, 4, 3))
    coef = torch.randn(2, 4, 3, 8)
    torch.manual_seed(3)
    t1, t2 = ne.NanoEngramTable(64, 8), ne.NanoEngramTable(64, 8)
    t2.weight.data.copy_(t1.weight.data)
    opt1, opt2 = ne.SparseRowAdam([t1], lr=1e-2), ne.SparseRowAdam([t2], lr=1e-2)
    _one_step(t1, ids, coef, opt1)
    _one_step(t2, ids, coef, opt2)

    # checkpoint t2's table + optimizer state, then continue both with a fresh batch
    saved_table = t2.weight.data.clone()
    saved_state = opt2.state_dict()
    ids2 = torch.randint(0, 64, (2, 4, 3))
    coef2 = torch.randn(2, 4, 3, 8)
    _one_step(t1, ids2, coef2, opt1)

    t3 = ne.NanoEngramTable(64, 8)
    t3.weight.data.copy_(saved_table)
    opt3 = ne.SparseRowAdam([t3], lr=1e-2)
    opt3.load_state_dict(saved_state)
    _one_step(t3, ids2, coef2, opt3)
    assert torch.equal(t1.weight.data, t3.weight.data)


def test_apply_preset_unknown():
    with pytest.raises(ValueError, match="unknown Engram preset"):
        ne.apply_preset({"dim": 64}, "nope")


# --------------------------------------------------------------------------- model integration (gated)
def _tiny_engram_cfg():
    from nanov41.configs import CONFIGS

    return ne.apply_preset(CONFIGS["tiny"], "tiny")


def _build_tiny_model(seed: int = 0):
    from nanov41.model import NanoV41, init_random_weights

    ne.install()
    adapter = ne.load_tokenizer_adapter(TOKENIZER)
    torch.manual_seed(seed)
    model = NanoV41(_tiny_engram_cfg(), tokenizer=adapter, param_dtype="float32")
    init_random_weights(model, seed=seed)
    return model


@requires_tokenizer
def test_tiny_model_forward_backward_and_sparse_step():
    model = _build_tiny_model(seed=0)
    tables = ne.engram_tables(model)
    assert len(tables) == 1
    ids = torch.randint(0, 512, (2, 16))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                             ids[:, 1:].reshape(-1))
    loss.backward()
    opt = ne.SparseRowAdam(tables, lr=1e-2)
    before = tables[0].weight.data.clone()
    opt.step()
    assert torch.isfinite(loss)
    touched = (tables[0].weight.data != before).any(dim=1)
    assert int(touched.sum()) > 0
    assert opt._moments[id(tables[0])].size >= int(touched.sum())
    # deterministic across two identical runs
    t_a = tables[0].weight.data.clone()
    model_b = _build_tiny_model(seed=0)
    tables_b = ne.engram_tables(model_b)
    logits_b = model_b(ids)
    loss_b = torch.nn.functional.cross_entropy(
        logits_b[:, :-1].reshape(-1, logits_b.size(-1)).float(), ids[:, 1:].reshape(-1))
    loss_b.backward()
    ne.SparseRowAdam(tables_b, lr=1e-2).step()
    assert torch.equal(t_a, tables_b[0].weight.data)


@requires_tokenizer
def test_train_py_engram_smoke(tmp_path):
    out = tmp_path / "tiny_engram.pt"
    env = dict(os.environ, NANOV41_DS_TOKENIZER_JSON=TOKENIZER)
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), "--config", "tiny", "--engram", "tiny",
         "--steps", "2", "--batch-size", "2", "--seq-len", "16", "--device", "cpu",
         "--log-every", "1", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[engram] preset=tiny" in proc.stdout
    assert "[done]" in proc.stdout
    assert out.exists()
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["cfg"]["engram_layer_ids"] == [1]
    assert "engram_optimizer" in payload and payload["engram_optimizer"] is not None
    assert any("engram.embed.weight" in key for key in payload["state_dict"])


@requires_tokenizer
def test_engram_checkpoint_loads_in_eval_provider(tmp_path):
    from nanov41.eval.provider import ModelProvider

    out = tmp_path / "tiny_engram.pt"
    env = dict(os.environ, NANOV41_DS_TOKENIZER_JSON=TOKENIZER)
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), "--config", "tiny", "--engram", "tiny",
         "--steps", "1", "--batch-size", "2", "--seq-len", "16", "--device", "cpu",
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    provider = ModelProvider(checkpoint=out, dtype="float32", device="cpu")
    try:
        assert provider.engram is True
        assert provider.engram_table_params == 521 * 32
        logits = provider.logits(torch.randint(0, 512, (2, 8)))
        assert tuple(logits.shape) == (2, 8, 512)
    finally:
        provider.close()


@requires_tokenizer
def test_train_py_engram_requires_tokenizer():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"), "--config", "tiny", "--engram", "tiny",
         "--steps", "1", "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "NANOV41_DS_TOKENIZER_JSON"},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "needs --tokenizer" in proc.stderr
