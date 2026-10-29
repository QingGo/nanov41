# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Tests for the precision-discounted MFU accounting and the V4.1-Flash scaling check."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mfu = _load("mfu_accounting")
v41 = _load("v41_scaling_check")


# --------------------------------------------------------------------------- accounting
def test_precision_weights_match_v41_report() -> None:
    """DeepSeek-V4.1-Flash Figure 2 caption: BF16/FP8/FP4 weighted by 1 / 0.5 / 0.25."""
    assert mfu.PRECISION_WEIGHTS["bf16"] == 1.0
    assert mfu.PRECISION_WEIGHTS["fp8"] == 0.5
    assert mfu.PRECISION_WEIGHTS["fp4"] == 0.25


def test_discounted_flops() -> None:
    assert mfu.discounted_flops({"fp8": 4.0, "bf16": 2.0}) == pytest.approx(4.0)
    assert mfu.discounted_flops({"fp4": 4.0, "fp8": 2.0, "bf16": 1.0}) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        mfu.discounted_flops({"int4": 1.0})


def test_r50_preset_accounting() -> None:
    """R50 best: 60.43 ms/step, fp8 head + bf16 rest, raw 114.93 TFLOP/s -> 54.86% raw."""
    preset = mfu.PRESETS["r50_b16"]
    result = mfu.accounting(preset["mix"], preset["time_ms"])
    assert result["raw_flops_tflop"] == pytest.approx(6.945, abs=1e-3)
    assert result["raw_mfu_pct"] == pytest.approx(54.86, abs=0.05)
    assert result["discounted_flops_tflop"] == pytest.approx(4.504, abs=1e-3)
    assert result["discounted_mfu_pct"] == pytest.approx(35.58, abs=0.05)
    assert result["wall_ms_at_60pct"] == pytest.approx(35.83, abs=0.05)


def test_all_fp8_moe_what_if() -> None:
    preset = mfu.PRESETS["r50_b16_all_fp8_moe"]
    result = mfu.accounting(preset["mix"], preset["time_ms"])
    assert result["discounted_mfu_pct"] == pytest.approx(29.35, abs=0.05)
    assert result["wall_ms_at_60pct"] == pytest.approx(29.56, abs=0.05)


# --------------------------------------------------------------------------- scaling check
def test_full_model_breakdown_matches_report() -> None:
    """The derived activation split must add up to the report's 16B decode activation."""
    full = v41.full_model_breakdown()
    active = (
        full["routed_active_params"]
        + full["shared_active_params"]
        + full["other_active_params"]
    )
    assert active == pytest.approx(16e9, rel=1e-12)
    assert full["raw_decode_flops_per_token"] == pytest.approx(32e9)
    assert full["effective_decode_flops_per_token"] == pytest.approx(11.045e9, rel=1e-3)
    # Experts (routed+shared active) are fp4, the non-expert active part is fp8.
    assert full["routed_active_params"] / 1e9 == pytest.approx(8.4935, rel=1e-3)
    assert full["routed_expert_params"] / 1e9 == pytest.approx(543.58, rel=1e-3)


def test_nano_scaling_matches_decode_flops() -> None:
    """The check requested for the release notes: activated-params scaling == decode-FLOPs scaling."""
    report = v41.build_report()
    d12 = report["nano"]["d12"]
    assert d12["ratios"]["activated_params"] == pytest.approx(226.47, rel=1e-4)
    assert d12["ratios"]["raw_decode_flops"] == pytest.approx(226.47, rel=1e-4)
    assert d12["checks"]["raw_ratio_matches_activated_params"] is True
    # Total (backbone) params are NOT the right scaling metric for compute.
    assert d12["ratios"]["total_params"] == pytest.approx(2854.1, rel=1e-3)
    assert d12["checks"]["total_params_matches_decode_flops"] is False
    assert d12["ratios"]["effective_decode_flops"] == pytest.approx(120.51, rel=1e-3)
    assert d12["ratios"]["raw_prefill_flops"] == pytest.approx(113.23, rel=1e-3)

    d8 = report["nano"]["d8"]
    assert d8["ratios"]["activated_params"] == pytest.approx(405.06, rel=1e-4)
    assert d8["ratios"]["raw_decode_flops"] == pytest.approx(405.06, rel=1e-4)
    assert d8["ratios"]["total_params"] == pytest.approx(6763.0, rel=1e-3)
