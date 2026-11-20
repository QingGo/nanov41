#!/usr/bin/env python3
"""Render the M1 aligned-d12 A/B (AdamW baseline vs V4.1 optimizer suite) into markdown.

Reads the two ``*.train.json`` summaries + the tail of each ``*.log`` produced by
``tools/m1_ab_run.sh`` and writes a comparison table (loss, tok/s, raw MFU, step time, cost
estimate).  Usage::

    python3 tools/m1_ab_report.py --dir runs/aligned \\
        --tokens 500000000 --price 2.78 --out docs/m1-ab-report.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ARMS = (("v41", "A-v41", "V4.1 三类优化器"), ("adamw", "B-adamw", "vanilla AdamW 基线"))
STEP_RE = re.compile(r"\[step (\d+)/(\d+)\] loss=([\d.]+) lr=([\d.eE+-]+) "
                     r"tok/s=([\d.]+)k mfu=([\d.]+)%(?: gn=([\d.]+))?(?: imb=([\d.]+))?")
# head share of train FLOPs for the aligned d12 config (prediction head 132.4M / 183.0M fwd).
HEAD_FLOPS_SHARE = 0.7234
HEAD_ALPHA = {"bf16": 1.0, "fp8": 0.5, "fp4": 0.25}


def load_arm(directory: pathlib.Path, stem: str) -> dict | None:
    summary = directory / f"{stem}.train.json"
    if not summary.exists():
        return None
    out = json.loads(summary.read_text())
    log = directory / f"{stem}.log"
    tail: list[float] = []
    steps: list[dict] = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = STEP_RE.search(line)
            if m:
                rec = {"step": int(m.group(1)), "loss": float(m.group(3)),
                       "tok_s": float(m.group(5)), "mfu": float(m.group(6)) / 100.0}
                if m.group(7):
                    rec["gn"] = float(m.group(7))
                if m.group(8):
                    rec["imb"] = float(m.group(8))
                steps.append(rec)
                tail.append(float(m.group(3)))
    out["log_steps"] = steps
    if tail:
        out.setdefault("tail_loss", sum(tail[-20:]) / min(20, len(tail)))
    for key, src in (("gn_tail_avg", "gn"), ("imb_tail_avg", "imb")):
        vals = [rec[src] for rec in steps if src in rec][-20:]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def fmt(value, spec: str, dash: str = "—") -> str:
    return dash if value is None else format(value, spec)


def render(results: dict[str, dict | None], *, tokens: int = 500_000_000,
           price: float = 2.78, steps: int | None = None,
           head_precision: str = "bf16", seq: int = 8192) -> str:
    alpha = HEAD_ALPHA[head_precision]
    disc_mul = (1.0 - HEAD_FLOPS_SHARE) + HEAD_FLOPS_SHARE * alpha
    lines = ["# M1 aligned d12 A/B：AdamW vs V4.1 三类优化器", ""]
    lines.append(f"协议：`docs/aligned/d12_aligned.json`（dim 512, 12L enc6+dec6, 128 experts top-2, "
                 f"Engram 220M, seq {seq}, 2^19 tokens/step, μP LR, constant schedule, "
                 f"loss-free balancing 1e-4 + bias 1e-3）。**两臂唯一变量是优化器。**")
    lines.append(f"数值口径：head = {head_precision}（discounted α_head={alpha}，"
                 f"disc = raw × {disc_mul:.4f}）；raw 与 discounted 两列不可互相比较。")
    lines += ["", "| 指标 | " + " | ".join(name for _, _, name in ARMS) + " |",
              "|---|---:|---:|"]
    rows = [
        ("tokens/step", "tokens_per_step", ""),
        ("steps", "steps", ""),
        ("平均 logged loss", "avg_logged_loss", ".5f"),
        ("尾 20 log loss", "tail_loss", ".5f"),
        ("tok/s", "tokens_per_sec", ",.0f"),
        ("raw MFU（ΣF/(t·P_bf16)）", "raw_bf16_mfu", ".2%"),
        ("discounted MFU（主口径）", "_disc_mfu", ".2%"),
        ("尾 20 log gn", "gn_tail_avg", ".2f"),
        ("尾 20 log imb", "imb_tail_avg", ".2f"),
        ("train 秒", "train_seconds", ".1f"),
        ("step 毫秒", None, ".1f"),
    ]
    for label, key, spec in rows:
        cells = []
        for _, stem, _ in ARMS:
            r = results.get(stem)
            if r is None:
                cells.append("—")
                continue
            if key is None:
                value = (r["train_seconds"] * 1000.0 / r["steps"]) if r.get("steps") else None
            elif key == "_disc_mfu":
                raw = r.get("raw_bf16_mfu")
                value = None if raw is None else raw * disc_mul
            else:
                value = r.get(key)
            cells.append(fmt(value, spec))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # cost estimate
    cost_cells = []
    for _, stem, _ in ARMS:
        r = results.get(stem)
        if r is None or not r.get("train_seconds"):
            cost_cells.append("—")
            continue
        cost = r["train_seconds"] / 3600.0 * price
        cost_cells.append(f"¥{cost:.2f}")
    lines.append("| 估算成本 @¥" + f"{price:.2f}/h | " + " | ".join(cost_cells) + " |")
    lines += ["",
              "## 曲线（每 log 点）", "",
              "| step | " + " | ".join(f"{name} loss" for _, _, name in ARMS) + " |",
              "|---:|" + "---:|" * len(ARMS)]
    stepwise: dict[int, dict[str, float]] = {}
    for _, stem, _ in ARMS:
        r = results.get(stem)
        if not r:
            continue
        for pt in r.get("log_steps", []):
            stepwise.setdefault(pt["step"], {})[stem] = pt["loss"]
    for step in sorted(stepwise)[:: max(1, len(stepwise) // 12)] if stepwise else []:
        cells = [f"{stepwise[step].get(stem, float('nan')):.5f}"
                 if stem in stepwise[step] else "—" for _, stem, _ in ARMS]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    missing = [name for _, stem, name in ARMS if results.get(stem) is None]
    if missing:
        lines += ["", f"> 缺少结果：{', '.join(missing)}（运行未完成或未启动）。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/aligned")
    ap.add_argument("--tokens", type=int, default=500_000_000)
    ap.add_argument("--price", type=float, default=2.78, help="5090 ¥/hour")
    ap.add_argument("--out", default="")
    ap.add_argument("--head-precision", choices=sorted(HEAD_ALPHA), default="bf16",
                    help="head precision used in BOTH arms (affects discounted MFU only)")
    ap.add_argument("--seq", type=int, default=8192)
    args = ap.parse_args()
    directory = pathlib.Path(args.dir)
    results = {stem: load_arm(directory, stem) for _, stem, _ in ARMS}
    text = render(results, tokens=args.tokens, price=args.price,
                  head_precision=args.head_precision, seq=args.seq)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text)
        print(f"\n[report] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
