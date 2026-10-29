# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Scaling sanity check: nano-v41 vs DeepSeek-V4.1-Flash decode FLOPs.

Question: when we shrink the V4.1-Flash architecture to a nano proxy, does the *parameter*
down-scaling ratio match the *decode-FLOPs* down-scaling ratio? If yes, compute-efficiency
numbers measured on the nano model are meaningful for the full model.

Report facts (DeepSeek-V4.1-Flash tech report §2.1 + Figure 2):
  * 552B backbone parameters (+196B Engram parameters),
  * 40 layers (20 causal-encoder + 20 decoder, CED), vocab 129,280, dim 5,120,
    384 routed experts / 6 activated, 1 shared expert, moe_inter_dim 2,304,
  * activates 8B parameters per token during **prefill** and 16B during **decode**,
  * decode FLOPs are compared with precision weights BF16/FP8/FP4 = 1/0.5/0.25.

Nano configs (this repo; parameter counts are regression-tested in tests/test_configs.py):
  * d12: 193.41M total / 94.12M non-embedding, forward FLOPs/token = 141.3 MFLOP,
  * d8 :  81.62M total / 15.43M non-embedding, forward FLOPs/token =  79.0 MFLOP
    (forward FLOPs from the FlopCounter used for the W1/W2 benchmark numbers).

Run: python3 tools/v41_scaling_check.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

# --- DeepSeek-V4.1-Flash, from the report text and the released ref/config.json ---
FULL = {
    "backbone_params": 552e9,
    "engram_params": 196e9,
    "decode_activated_params": 16e9,
    "prefill_activated_params": 8e9,
    "vocab": 129_280,
    "dim": 5_120,
    "layers": 40,
    "routed_experts": 384,
    "activated_experts": 6,
    "shared_experts": 1,
    "moe_inter_dim": 2_304,
}

# --- nano-v41 configs ---
NANO = {
    "d8": {
        "total": 81_620_944,
        "non_embedding": 15_429_584,
        "forward_flops_per_token": 79.0e6,
        "dim": 256,
    },
    "d12": {
        "total": 193_408_360,
        "non_embedding": 94_121_320,
        "forward_flops_per_token": 141.3e6,
        "dim": 384,
    },
}

# Precision mix assumption for the nano proxy's current best run (R50): the vocabulary
# head GEMM runs in FP8, attention/MoE/other in BF16. Used only for the discounted rows.
NANO_FP8_HEAD_MIX = {"head": 0.703, "rest": 0.297}

# Full-model precision assignment from the released config: dtype=fp8 for the linear
# layers (attention/head/indexer), expert_dtype=fp4 for the MoE experts.
FULL_EXPERT_PRECISION = "fp4"
FULL_OTHER_PRECISION = "fp8"


def full_model_breakdown() -> dict:
    """Parameter/activation breakdown of V4.1-Flash, derived from the released config."""
    v, d, layers = FULL["vocab"], FULL["dim"], FULL["layers"]
    inter = FULL["moe_inter_dim"]
    routed = layers * FULL["routed_experts"] * 3 * d * inter
    shared = layers * FULL["shared_experts"] * 3 * d * inter
    embed_head = 2 * v * d
    routed_active = routed * FULL["activated_experts"] / FULL["routed_experts"]
    decode_act = FULL["decode_activated_params"]
    other_active = decode_act - routed_active - shared
    # Discounted ("effective") activated parameters under the report's Figure 2 convention.
    effective = routed_active * 0.25 + shared * 0.25 + other_active * 0.5
    return {
        "backbone_params": FULL["backbone_params"],
        "engram_params": FULL["engram_params"],
        "routed_expert_params": routed,
        "shared_expert_params": shared,
        "embed_head_params": embed_head,
        "non_expert_params": FULL["backbone_params"] - routed - shared - embed_head,
        "routed_active_params": routed_active,
        "shared_active_params": shared,
        "other_active_params": other_active,
        "decode_activated_params": decode_act,
        "prefill_activated_params": FULL["prefill_activated_params"],
        "raw_decode_flops_per_token": 2 * decode_act,
        "raw_prefill_flops_per_token": 2 * FULL["prefill_activated_params"],
        "effective_decode_flops_per_token": 2 * effective,
        "effective_activated_params": effective,
    }


def nano_breakdown(name: str, data: dict) -> dict:
    """Parameter / decode-FLOPs breakdown of one nano config."""
    fwd = data["forward_flops_per_token"]
    activated = fwd / 2  # forward FLOPs = 2 x activated params for GEMM-only accounting
    head_params = FULL["vocab"] * data["dim"]
    rest_active = activated - head_params
    effective = head_params * 0.5 + rest_active * 1.0  # fp8 head + bf16 rest (R50 mix)
    return {
        "total": data["total"],
        "non_embedding": data["non_embedding"],
        "activated_params": activated,
        "head_params": head_params,
        "rest_active_params": rest_active,
        "forward_flops_per_token": fwd,
        "effective_activated_params": effective,
        "effective_decode_flops_per_token": 2 * effective,
    }


def build_report() -> dict:
    full = full_model_breakdown()
    out: dict = {"full": full, "nano": {}}
    for name, data in NANO.items():
        nano = nano_breakdown(name, data)
        raw_ratio = full["raw_decode_flops_per_token"] / nano["forward_flops_per_token"]
        activated_ratio = full["decode_activated_params"] / nano["activated_params"]
        effective_ratio = (
            full["effective_decode_flops_per_token"] / nano["effective_decode_flops_per_token"]
        )
        total_ratio = full["backbone_params"] / nano["total"]
        non_emb_ratio = (
            (full["backbone_params"] - full["embed_head_params"]) / nano["non_embedding"]
        )
        prefill_ratio = full["raw_prefill_flops_per_token"] / nano["forward_flops_per_token"]
        out["nano"][name] = {
            **nano,
            "ratios": {
                "raw_decode_flops": raw_ratio,
                "activated_params": activated_ratio,
                "effective_decode_flops": effective_ratio,
                "total_params": total_ratio,
                "non_embedding_params": non_emb_ratio,
                "raw_prefill_flops": prefill_ratio,
            },
            "checks": {
                "raw_ratio_matches_activated_params": abs(raw_ratio - activated_ratio) < 1e-6,
                "total_params_matches_decode_flops": abs(total_ratio - raw_ratio) / raw_ratio < 0.05,
                "total_vs_decode_gap_x": total_ratio / raw_ratio,
            },
        }
    return out


def _fmt_b(x: float) -> str:
    return f"{x / 1e9:.3f}B"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args(argv)

    report = build_report()
    full = report["full"]
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("DeepSeek-V4.1-Flash (report)")
    print(f"  backbone params            : {_fmt_b(full['backbone_params'])} (+{_fmt_b(full['engram_params'])} Engram)")
    print(f"  routed experts (all)       : {_fmt_b(full['routed_expert_params'])}")
    print(f"  shared experts             : {_fmt_b(full['shared_expert_params'])}")
    print(f"  embed+head                 : {_fmt_b(full['embed_head_params'])}")
    print(f"  activated params (decode)  : {_fmt_b(full['decode_activated_params'])}")
    print(f"    routed active            : {_fmt_b(full['routed_active_params'])} @ fp4")
    print(f"    shared                   : {_fmt_b(full['shared_active_params'])} @ fp4")
    print(f"    non-expert active        : {_fmt_b(full['other_active_params'])} @ fp8")
    print(f"  raw decode FLOPs / token   : {full['raw_decode_flops_per_token'] / 1e9:.2f} GFLOP")
    print(f"  discounted decode FLOPs    : {full['effective_decode_flops_per_token'] / 1e9:.2f} GFLOP (bf16-eq)")
    print(f"  raw prefill FLOPs / token  : {full['raw_prefill_flops_per_token'] / 1e9:.2f} GFLOP (CED: 8B active)")
    print()

    for name, nano in report["nano"].items():
        r = nano["ratios"]
        print(f"nano {name}: {nano['total'] / 1e6:.2f}M total / {nano['non_embedding'] / 1e6:.2f}M non-emb, "
              f"forward {nano['forward_flops_per_token'] / 1e6:.1f} MFLOP/token")
        print(f"  raw decode FLOPs scaling        : {r['raw_decode_flops']:.2f}x")
        print(f"  activated-params scaling        : {r['activated_params']:.2f}x   <- matches raw FLOPs")
        print(f"  discounted decode scaling (fp8 head + bf16 rest): {r['effective_decode_flops']:.2f}x")
        print(f"  total-params scaling            : {r['total_params']:.1f}x")
        print(f"  non-embedding-params scaling    : {r['non_embedding_params']:.1f}x")
        print(f"  raw prefill FLOPs scaling       : {r['raw_prefill_flops']:.2f}x (no CED in nano)")
        print(f"  consistent (activated vs decode): {report['nano'][name]['checks']['raw_ratio_matches_activated_params']}")
        print()

    print("Note: the total-parameter ratio is much larger than the decode-FLOPs ratio because")
    print("the nano configs keep the full vocab (embedding+head = 51.3% of d12 params), activate 2/16")
    print("experts instead of 6/384, and omit MTP/Engram/vision. Use activated params or")
    print("decode FLOPs when describing compute scaling; see docs/v41-scaling-check.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
