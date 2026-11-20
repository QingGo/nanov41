#!/usr/bin/env python3
"""R93 — P1 8B monitoring figure: loss / LR / throughput / grad-norm from p1_8b.series.json,
plus panel-bpb mark evaluations (eval_t*.json) once they exist.

Usage: python3 tools/r93_p1_plot.py [--series runs/r93/p1/p1_8b.series.json] [--out runs/r93/p1/p1_loss.png]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXPECT_BPB = lambda d_tokens: 1.2106 * (np.maximum(np.asarray(d_tokens, dtype=float), 1e7) / 1e9) ** -0.154  # noqa: E731


def load(path: pathlib.Path) -> list[dict]:
    d = json.loads(path.read_text())
    return [p for p in d.get("points", []) if "loss" in p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="runs/r93/p1/p1_8b.series.json")
    ap.add_argument("--out", default="runs/r93/p1/p1_loss.png")
    ap.add_argument("--total-steps", type=int, default=3814)
    ap.add_argument("--tokens-per-step", type=float, default=2_097_152)
    args = ap.parse_args()
    pts = load(pathlib.Path(args.series))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not pts:
        print("no points yet")
        return 1
    steps = np.array([p["step"] for p in pts], dtype=float)
    toks = steps * args.tokens_per_step
    loss = np.array([p["loss"] for p in pts], dtype=float)
    lr = np.array([p.get("lr", np.nan) for p in pts], dtype=float)
    tps = np.array([p.get("tok_s", np.nan) for p in pts], dtype=float)
    mfu = np.array([p.get("mfu_pct", np.nan) for p in pts], dtype=float)
    gn = np.array([p.get("gn", np.nan) if p.get("gn") is not None else np.nan for p in pts], dtype=float)
    elapsed = np.array([p.get("elapsed_s", np.nan) for p in pts], dtype=float)

    # mark evals (bpb) if already produced by the eval loop
    evals = []
    for ev in sorted(pathlib.Path(args.series).parent.glob("eval_t*.json")):
        try:
            d = json.loads(ev.read_text())
            m = ((d.get("dataset", {}).get("domains") or {}).get("climbmix_val") or {}).get("metrics", {}).get("all", {})
            tag = ev.stem.replace("eval_t", "")
            d_tok = float(tag) if tag.replace(".", "").isdigit() else None
            if d_tok:
                evals.append((d_tok, m.get("bpb")))
        except Exception:  # noqa: BLE001
            pass
    evals.sort()
    eval_all = []
    for ev in sorted(pathlib.Path(args.series).parent.glob("eval_final.json")):
        try:
            d = json.loads(ev.read_text())
            m = ((d.get("dataset", {}).get("domains") or {}).get("climbmix_val") or {}).get("metrics", {}).get("all", {})
            eval_all.append(("final", m.get("bpb")))
        except Exception:  # noqa: BLE001
            pass

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
    tok_max = args.total_steps * args.tokens_per_step / 1e9

    ax = axs[0][0]
    ax.plot(toks / 1e9, loss, lw=1.6, color="C0", label="train loss (nats/token)")
    # expected VAL loss curve from the P0b-measured bpb curve (val, after cooldown -> upper ref)
    grid = np.linspace(max(toks[0] / 1e9, 0.05), tok_max, 200) * 1e9
    ax.plot(grid / 1e9, EXPECT_BPB(grid) * 3.0856, "--", color="0.55",
            label="expected val (1.21·D^-0.154 × 3.0856)")
    for d_tok, bpb in evals:
        ax.axvline(d_tok / 1e9, color="C2", alpha=0.25, lw=0.8)
        ax.annotate(f"bpb {bpb:.3f}", (d_tok / 1e9, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 0),
                    color="C2", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("tokens (B)"); ax.set_ylabel("loss")
    ax.set_title(f"P1 8B training — step {int(steps[-1])}/{args.total_steps}, "
                 f"{toks[-1]/1e9:.3f}B tokens")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[0][1]
    ax.plot(steps, lr, lw=1.4, color="C3")
    ax.set_xlabel("step"); ax.set_ylabel("LR")
    ax.set_title("learning rate (WSD: warmup 30 / decay-start 0.9 / min 0.1)")
    ax.grid(alpha=0.3)

    ax = axs[1][0]
    ax.plot(steps, tps / 1e3, lw=1.2, color="C4", label="tok/s (k)")
    ax.set_xlabel("step"); ax.set_ylabel("k tok/s", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")
    ax2 = ax.twinx()
    ax2.plot(steps, mfu, lw=1.2, color="C5", label="MFU %")
    ax2.set_ylabel("MFU %", color="C5"); ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("throughput"); ax.grid(alpha=0.3)

    ax = axs[1][1]
    ax.plot(steps, gn, lw=1.0, color="C6")
    ax.set_xlabel("step"); ax.set_ylabel("grad norm (clipped at 1.0)")
    ax.set_title("grad norm / stability")
    ax.grid(alpha=0.3)

    if evals:
        txt = "  ".join(f"{d/1e9:.1f}B:bpb {b:.3f}" for d, b in evals)
        fig.text(0.5, 0.005, "mark evals: " + txt, ha="center", fontsize=9, color="C2")
    for e in eval_all:
        fig.text(0.5, 0.985, f"final bpb {e[1]:.4f}", ha="center", fontsize=9, color="C2")
    fig.tight_layout(rect=(0, 0.012, 1, 0.99))
    fig.savefig(out, dpi=110)
    print(f"[plot] wrote {out} ({len(pts)} points, {len(evals)} mark evals, last step {int(steps[-1])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
