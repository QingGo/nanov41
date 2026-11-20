#!/usr/bin/env python3
"""Aggregate P2 arm train/eval JSONs into a comparison table."""
from __future__ import annotations

import argparse
import json
import pathlib


def _load(p: pathlib.Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/p2")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    rows = []
    for train in sorted(d.glob("*.train.json")):
        arm = train.name[: -len(".train.json")]
        t = _load(train) or {}
        t0 = _load(d / f"{arm}.t0.json") or {}
        provider = t0.get("provider", {})
        summary = t0.get("summary", {})
        mixes = t0.get("mixes", {})
        if isinstance(mixes, list):
            mixes = {m.get("name", str(i)): m.get("bpb") for i, m in enumerate(mixes) if isinstance(m, dict)}
        def _mix_bpb(key):
            v = mixes.get(key) if isinstance(mixes, dict) else None
            return v.get("bpb") if isinstance(v, dict) else v
        rows.append({
            "arm": arm,
            "config": t.get("config"),
            "engram": t.get("engram"),
            "tokens": t.get("tokens"),
            "train_seconds": t.get("train_seconds"),
            "tok_per_s": round(t.get("tokens", 0) / max(1.0, t.get("train_seconds", 1.0)), 0),
            "train_loss": round(t.get("avg_logged_loss", float("nan")), 4),
            "n_params": provider.get("n_params"),
            "table_params": provider.get("engram_table_params", t.get("table_params", 0)),
            "macro_bpb": summary.get("macro_bpb"),
            "byte_weighted_bpb": summary.get("byte_weighted_bpb"),
            "mix_phase1": _mix_bpb("marin-8b-phase1"),
            "mix_jellyfish": _mix_bpb("marin-8b-jellyfish"),
        })
    headers = ["arm", "config", "engram", "tokens", "train_seconds", "tok_per_s", "train_loss",
               "n_params", "table_params", "macro_bpb", "byte_weighted_bpb", "mix_phase1", "mix_jellyfish"]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for r in rows:
        def fmt(k):
            v = r.get(k)
            if isinstance(v, float):
                return f"{v:.4f}"
            if isinstance(v, int) and v > 1e6:
                return f"{v/1e6:.2f}M"
            return str(v)
        print(" | ".join(fmt(h) for h in headers))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
