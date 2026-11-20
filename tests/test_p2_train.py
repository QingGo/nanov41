"""CPU tests for the aligned trainer plumbing that does not need a GPU.

``tools/p2_train.py`` builds a CUDA model in ``main()``, so the portable pieces are tested here:
the LR schedules, the mixed-dtype gradient zeroing helper, the domain-mix loader format and the
``--help``/argparse surface (catches import-time regressions).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_p2_train():
    path = ROOT / "tools" / "p2_train.py"
    spec = importlib.util.spec_from_file_location("p2_train_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p2_train_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lr_at_schedules() -> None:
    p2 = _load_p2_train()
    lr_at = p2.lr_at
    # warmup ramps linearly from peak/warmup to peak
    assert lr_at(0, 1000, 1e-3, warmup=10) == pytest.approx(1e-3 / 10)
    assert lr_at(9, 1000, 1e-3, warmup=10) == pytest.approx(1e-3)
    # constant holds the peak
    assert lr_at(500, 1000, 1e-3, warmup=10, schedule="constant") == pytest.approx(1e-3)
    # wsd is flat before decay then cosine to min ratio
    assert lr_at(500, 1000, 1e-3, warmup=10, schedule="wsd", decay_start=0.8) == pytest.approx(1e-3)
    assert lr_at(999, 1000, 1e-3, warmup=10, schedule="wsd", decay_start=0.8,
                 min_ratio=0.1) == pytest.approx(0.1e-3, rel=1e-3)
    # cosine ends at 10% of peak
    assert lr_at(999, 1000, 1e-3, warmup=10) == pytest.approx(0.1e-3, rel=1e-3)


def test_zero_grads_mixed_dtype() -> None:
    p2 = _load_p2_train()
    ps = [torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16)),
          torch.nn.Parameter(torch.randn(4, dtype=torch.float32)),
          torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))]
    for i, p in enumerate(ps):
        p.grad = torch.ones_like(p) * (i + 1)
    ps[0].grad = None  # unused param must be skipped, not crash
    p2._zero_grads(ps)
    assert torch.count_nonzero(ps[1].grad) == 0
    assert torch.count_nonzero(ps[2].grad) == 0


def test_token_mix_loader_roundtrip(tmp_path: pathlib.Path) -> None:
    p2 = _load_p2_train()
    data = tmp_path / "data"
    data.mkdir()
    arr = np.arange(10_000, dtype="<u4")
    for domain, weight in (("web", 0.8), ("code", 0.2)):
        arr.tofile(data / f"{domain}.u32.bin")
    (data / "manifest.json").write_text(json.dumps(
        {"format": "nanov41-p2-corpus/1",
         "domains": {"web": {"tokens": 10_000}, "code": {"tokens": 10_000}}}))
    loader = p2.TokenMixLoader(data, seq=64, batch=4, mix={"web": 0.8, "code": 0.2}, seed=3)
    batch = loader.next_batch()
    assert batch.shape == (4, 64) and batch.dtype == np.int64
    assert loader.tokens_per_step == 256
    state = loader.state_dict()
    a = loader.next_batch()
    loader2 = p2.TokenMixLoader(data, seq=64, batch=4, mix={"web": 0.8, "code": 0.2}, seed=3)
    loader2.load_state_dict(state)
    b = loader2.next_batch()
    assert np.array_equal(a, b)


def test_cli_help_and_aligned_build_helper() -> None:
    proc = subprocess.run([sys.executable, "tools/p2_train.py", "--help"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--grad-accum", "--optimizer", "--config-json", "--balance-loss",
                 "--save-optimizer", "--resume", "--peak-tflops"):
        assert flag in proc.stdout, flag

    spec = importlib.util.spec_from_file_location("aligned_build_test",
                                                  ROOT / "tools" / "aligned_build.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    name = ab.install_aligned(with_engram=True)
    from nanov41.configs import CONFIGS
    import nanov41.nano_engram as ne
    assert CONFIGS[name]["n_routed_experts"] == 128
    assert CONFIGS[name]["moe_inter_dim"] == 256
    assert ne.ENGRAM_PRESETS[name]["engram_layer_ids"] == [1, 4]
    acct = ab.aligned_accounting()
    assert acct["fwd_flops_per_token_est"] > 0


def test_climbmix_converter_streams_and_caps(tmp_path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location("climbmix_to_p2_test",
                                                  ROOT / "tools" / "climbmix_to_p2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "shards"
    src.mkdir()
    for i in range(3):
        np.arange(1000, dtype=np.uint16).astype(np.uint16).tofile(src / f"shard_{i}.bin")
        np.save(src / f"shard_{i}.npy", np.arange(1000, dtype=np.uint16))
    out = tmp_path / "out"
    manifest = mod.convert(src, out, tokens=2500)
    assert manifest["tokens_total"] == 2500
    assert (out / "manifest.json").exists()
    arr = np.fromfile(out / "web.u32.bin", dtype="<u4")
    assert arr.shape == (2500,) and arr.dtype == np.uint32


def test_ab_report_renders(tmp_path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location("m1_ab_report_test",
                                                  ROOT / "tools" / "m1_ab_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = tmp_path
    for stem, loss in (("A-v41", 4.1), ("B-adamw", 4.2)):
        (d / f"{stem}.train.json").write_text(json.dumps({
            "tokens_per_step": 524288, "steps": 954, "avg_logged_loss": loss,
            "tokens_per_sec": 100000.0, "raw_bf16_mfu": 0.5, "train_seconds": 3000.0}))
        (d / f"{stem}.log").write_text(
            "[step 1/954] loss=5.00000 lr=8.22e-04 tok/s=90.0k mfu=45.0% gn=1.00\n"
            f"[step 20/954] loss={loss:.5f} lr=8.22e-04 tok/s=95.0k mfu=47.5% gn=0.90\n")
    results = {stem: mod.load_arm(d, stem) for _, stem, _ in mod.ARMS}
    text = mod.render(results, price=2.78)
    assert "V4.1 三类优化器" in text and "vanilla AdamW 基线" in text
    assert "4.10000" in text and "4.20000" in text
    assert "¥" in text and "loss" in text
