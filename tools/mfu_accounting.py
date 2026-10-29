# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Precision-discounted FLOP / MFU accounting (DeepSeek-V4.1-Flash report convention).

The DeepSeek-V4.1-Flash technical report compares decode FLOPs across models by weighting
BF16 / FP8 / FP4 operations by **1 / 0.5 / 0.25** (Figure 2 caption). This module applies
the same rule to MFU, which avoids "swapping the denominator" per precision mix:

    F_eq   = sum_p alpha[p] * F[p]                 # discounted (bf16-equivalent) FLOPs
    MFU_eq = F_eq / (t * P_bf16)                   # fixed bf16 peak (209.5 TFLOPS on RTX 5090)

Why this is the honest form: an FP8 FLOP is worth half a BF16 FLOP of hardware time (the
peak is 2x), so counting raw FLOPs inflates utilisation when precision drops. With a fixed
denominator, switching a component to FP8 only improves MFU if the wall time really drops
enough; otherwise the metric stays flat or gets worse.

Examples (R50 best, b=16x1024, graph replay + optimizer, 60.43 ms/step, raw 114.93 TFLOP/s):

    python3 tools/mfu_accounting.py --preset r50_b16
    python3 tools/mfu_accounting.py --time-ms 60.43 --flop fp8:4.882 --flop bf16:2.063
    python3 tools/mfu_accounting.py --preset r50_b16_all_fp8_moe   # 60% wall for that mix

Formula references: the report's Figure 2 caption (precision weights) and its accuracy
metric descriptions. "MFU" here means bf16-equivalent utilization, not the vendor peak of
the precision being used.
"""

from __future__ import annotations

import argparse
import json
import sys

# BF16 / FP8 / FP4 = 1 / 0.5 / 0.25, exactly as in the V4.1-Flash report (Figure 2 caption).
PRECISION_WEIGHTS = {
    "bf16": 1.0,
    "fp16": 1.0,
    "fp32": 1.0,
    "fp8": 0.5,
    "int8": 0.5,
    "fp4": 0.25,
}

PEAK_BF16_5090 = 209.5  # TFLOP/s, RTX 5090 bf16 (fp32 accumulate), vendor spec

# Measured precision mix per step, in TFLOP (from the internal W2 R50 run; the head GEMM
# runs in FP8, everything else in BF16). Token count = 16,384 tokens/step.
PRESETS = {
    "r50_b16": {
        "time_ms": 60.43,
        "mix": {"fp8": 4.882, "bf16": 2.063},
        "note": "R50 best: b=16x1024, fp8 head + bf16 MoE/attn, step incl. optimizer",
    },
    "r50_b16_all_fp8_moe": {
        "time_ms": 60.43,
        "mix": {"fp8": 6.459, "bf16": 0.486},
        "note": "what-if: same wall time, MoE also fp8 (head+MoE fp8, attn/other bf16)",
    },
}


def discounted_flops(mix: dict[str, float]) -> float:
    """Discounted FLOPs for a precision mix ``{precision: TFLOP}``."""
    total = 0.0
    for precision, flops in mix.items():
        try:
            weight = PRECISION_WEIGHTS[precision]
        except KeyError:
            raise ValueError(
                f"unknown precision {precision!r}; expected one of {sorted(PRECISION_WEIGHTS)}"
            ) from None
        total += weight * float(flops)
    return total


def mfu_equivalent(flops_eq_tflop: float, time_s: float, peak_bf16: float = PEAK_BF16_5090) -> float:
    """bf16-equivalent MFU (fraction): discounted FLOPs / (time * fixed bf16 peak)."""
    return flops_eq_tflop / (time_s * peak_bf16)


def mfu_raw(flops_tflop: float, time_s: float, peak_bf16: float = PEAK_BF16_5090) -> float:
    """Raw-FLOPs MFU (fraction) against the bf16 peak: cross-round throughput, precision-inflated."""
    return flops_tflop / (time_s * peak_bf16)


def wall_ms_at_mfu(
    flops_eq_tflop: float, target_mfu: float = 0.60, peak_bf16: float = PEAK_BF16_5090
) -> float:
    """Wall time at which a given mix reaches ``target_mfu`` under the discounted convention."""
    return flops_eq_tflop / (target_mfu * peak_bf16) * 1e3


def accounting(mix: dict[str, float], time_ms: float, peak_bf16: float = PEAK_BF16_5090) -> dict:
    """Full accounting breakdown for one measurement."""
    raw = sum(mix.values())
    eq = discounted_flops(mix)
    time_s = time_ms / 1e3
    return {
        "time_ms": time_ms,
        "mix_tflop": dict(mix),
        "raw_flops_tflop": raw,
        "discounted_flops_tflop": eq,
        "discount_ratio": eq / raw if raw else 0.0,
        "raw_mfu_pct": 100 * mfu_raw(raw, time_s, peak_bf16),
        "discounted_mfu_pct": 100 * mfu_equivalent(eq, time_s, peak_bf16),
        "peak_bf16_tflops": peak_bf16,
        "wall_ms_at_60pct": wall_ms_at_mfu(eq, 0.60, peak_bf16),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="named measurement preset")
    ap.add_argument("--time-ms", type=float, help="wall time per step (ms)")
    ap.add_argument("--flop", action="append", metavar="PRECISION:TFLOP",
                    help="precision mix component, e.g. --flop fp8:4.882 --flop bf16:2.063")
    ap.add_argument("--peak", type=float, default=PEAK_BF16_5090,
                    help=f"fixed bf16 peak in TFLOP/s (default {PEAK_BF16_5090})")
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args(argv)

    if args.preset:
        preset = PRESETS[args.preset]
        time_ms = preset["time_ms"]
        mix = dict(preset["mix"])
        note = preset["note"]
    else:
        if not args.time_ms or not args.flop:
            ap.error("provide --preset, or --time-ms plus at least one --flop PRECISION:TFLOP")
        time_ms = args.time_ms
        mix = {}
        for item in args.flop:
            precision, _, value = item.partition(":")
            if not value:
                ap.error(f"bad --flop {item!r}; expected PRECISION:TFLOP")
            mix[precision] = mix.get(precision, 0.0) + float(value)
        note = ""

    result = accounting(mix, time_ms, args.peak)
    result["note"] = note
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"time: {result['time_ms']:.3f} ms | mix: {result['mix_tflop']} TFLOP")
    if note:
        print(f"note: {note}")
    print(f"raw FLOPs           : {result['raw_flops_tflop']:.3f} TFLOP "
          f"-> raw/b16 MFU {result['raw_mfu_pct']:.2f}%")
    print(f"discounted FLOPs    : {result['discounted_flops_tflop']:.3f} TFLOP "
          f"(x{result['discount_ratio']:.3f})")
    print(f"discounted MFU      : {result['discounted_mfu_pct']:.2f}% "
          f"(fixed {result['peak_bf16_tflops']} TFLOP/s denominator)")
    print(f"60% wall (eq)       : {result['wall_ms_at_60pct']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
