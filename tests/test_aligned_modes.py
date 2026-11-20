"""Aligned-config CSA2 mode consistency (spec ``layer_modes`` <-> engine source lists).

The M0 config expressed the CSA2 modes only in the human-readable ``layer_modes`` list while
the engine sources were derived from ``compress_ratios`` changes -- which silently dropped the
decoder **Reindex** layer.  The M2 config is generated with ``--modes official`` so the engine
sources are derived from ``layer_modes`` (Full -> kv+index source, Reindex -> index source
only, Reuse -> neither).  These tests pin that mapping for both committed JSONs and check the
reference ``ModelArgs`` accepts them.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
M1_JSON = ROOT / "docs" / "aligned" / "d12_aligned.json"
M2_JSON = ROOT / "docs" / "aligned" / "d12_aligned_m2.json"


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _load_generator():
    spec = importlib.util.spec_from_file_location("aligned_gen_modes",
                                                  ROOT / "tools" / "make_aligned_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ratio_sources(compress: list[int]) -> tuple[list[int], list[int]]:
    kv, prev = [], None
    for i, r in enumerate(compress):
        if r != 0 and r != prev:
            kv.append(i)
        prev = r
    return kv, list(kv)


def test_generator_mode_mapping_helpers() -> None:
    gen = _load_generator()
    for layers in (8, 12, 16, 20):
        modes = gen._layer_modes(layers)
        assert len(modes) == layers
        assert modes[0] == "swa"
        kv, idx = gen.mode_source_layers(modes)
        assert kv == [i for i, m in enumerate(modes) if m == "full"]
        assert set(kv) <= set(idx)
        reindex = [i for i, m in enumerate(modes) if m == "reindex"]
        assert reindex == sorted(set(idx) - set(kv))
        # decoder half must contain at least one Full layer (CED global-KV owner)
        assert any(m == "full" for m in modes[layers // 2:])


def test_m1_config_mode_consistency() -> None:
    data = _load(M1_JSON)
    cfg, modes = data["engine_config"], data["layer_modes"]
    assert len(modes) == cfg["n_layers"]
    kv, idx = _ratio_sources(cfg["compress_ratios"])
    assert list(cfg["kv_source_layers"]) == kv
    assert list(cfg["index_source_layers"]) == idx
    # M1 deliberately has no Reindex layer (optimizer A/B baseline keeps the M0 approximation)
    assert not set(idx) - set(kv)


def test_m2_config_matches_official_modes() -> None:
    data = _load(M2_JSON)
    cfg, modes = data["engine_config"], data["layer_modes"]
    gen = _load_generator()
    assert modes == gen._layer_modes(cfg["n_layers"])
    kv, idx = gen.mode_source_layers(modes)
    assert list(cfg["kv_source_layers"]) == kv
    assert list(cfg["index_source_layers"]) == idx
    reindex = [i for i, m in enumerate(modes) if m == "reindex"]
    assert reindex, "M2 config must contain the decoder Reindex layer"
    assert set(reindex) == set(idx) - set(kv)
    # Reindex layers must have a positive compression ratio (they read a main-KV cache)
    for layer in idx:
        assert cfg["compress_ratios"][layer] > 0
    # CED: decoder Full layer is the first decoder layer and owns the global KV
    dec_start = cfg["n_layers"] // 2
    assert cfg["kv_source_layers"][-1] == dec_start
    assert data["hierarchical_indexer"]["candidate_source_layer"] == dec_start
    # pretraining engine keeps the hierarchical candidate pool disabled (post-training only)
    assert cfg["candidate_source_layer"] == -1

    # accounting delta vs the M1 baseline is just the extra Reindex indexer (10,240 params)
    m1 = _load(M1_JSON)
    d_backbone = data["accounting"]["backbone_params"] - m1["accounting"]["backbone_params"]
    d_active = data["accounting"]["active_params"] - m1["accounting"]["active_params"]
    d_flops = (data["accounting"]["fwd_flops_per_token_est"]
               - m1["accounting"]["fwd_flops_per_token_est"])
    assert d_backbone == d_active == 10_240
    assert d_flops == 2 * d_active
    assert abs(data["accounting"]["engram_nonembed_ratio"] - 0.355) < 0.01


def test_m2_engine_config_parses_as_model_args() -> None:
    from nanov41.patched_env import import_patched

    M = import_patched()
    cfg = _load(M2_JSON)["engine_config"]
    args = M.ModelArgs(**cfg)
    assert args.n_layers == 12
    assert tuple(args.kv_source_layers) == (1, 6)
    assert tuple(args.index_source_layers) == (1, 6, 10)
    # every layer listed in compress_ratios, MTP disabled
    assert len(args.compress_ratios) == args.n_layers
