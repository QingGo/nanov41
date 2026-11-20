#!/usr/bin/env python3
"""Register the committed aligned d12 config (docs/aligned/d12_aligned.json) with the engine.

The aligned config is *generated* by ``tools/make_aligned_config.py`` and validated by
``tests/test_aligned_configs.py``; the tools (p2_train, smoke/bench scripts) consume it here at
runtime instead of duplicating the numbers in ``nanov41/configs.py``.  Usage::

    from tools.aligned_build import install_aligned
    name = install_aligned("docs/aligned/d12_aligned.json")   # -> "d12_aligned"
    model, cfg = bench_graph_train.build(b, seq, config_name=name, engram=name)
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ALIGNED_JSON = ROOT / "docs" / "aligned" / "d12_aligned.json"

#: fields the reference ModelArgs needs to build a model without Engram modules disabled
_ENGRAM_KEYS = ("engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
                "engram_vocab_size", "engram_n_heads", "engram_head_dim")


def load_aligned(path: str | pathlib.Path | None = None) -> dict:
    path = pathlib.Path(path) if path else DEFAULT_ALIGNED_JSON
    return json.loads(path.read_text())


def install_aligned(path: str | pathlib.Path | None = None, name: str = "d12_aligned",
                    with_engram: bool = True) -> str:
    """Register ``CONFIGS[name]`` (and the Engram preset ``ENGRAM_PRESETS[name]``) in memory."""
    data = load_aligned(path)
    cfg = dict(data["engine_config"])
    if not with_engram:
        for key in _ENGRAM_KEYS:
            cfg[key] = [] if key in ("engram_layer_ids", "engram_num_embeddings") else 0
        cfg["engram_layer_ids"] = []
        cfg["engram_num_embeddings"] = []
        cfg["engram_compressed_vocab_size"] = 0
    from nanov41.configs import CONFIGS
    CONFIGS[name] = cfg
    if with_engram:
        import nanov41.nano_engram as ne
        eng = data["engram"]
        ne.ENGRAM_PRESETS[name] = {
            "engram_layer_ids": list(eng["engram_layer_ids"]),
            "engram_max_ngram_size": int(eng["engram_max_ngram_size"]),
            "engram_n_heads": int(eng["engram_n_heads"]),
            "engram_head_dim": int(eng["engram_head_dim"]),
            "engram_vocab_size": int(eng["engram_vocab_size"]),
            "engram_num_embeddings": [int(x) for x in eng["engram_num_embeddings"]],
        }
    return name


def aligned_accounting(path: str | pathlib.Path | None = None) -> dict:
    """The generator's parameter/FLOPs accounting (``accounting`` section of the JSON)."""
    return dict(load_aligned(path)["accounting"])


def aligned_optimizer_spec(path: str | pathlib.Path | None = None) -> dict:
    return dict(load_aligned(path)["optimizer"])


def main() -> int:
    data = load_aligned()
    print(json.dumps({"accounting": data["accounting"],
                      "optimizer": data["optimizer"],
                      "engram": data["engram"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
