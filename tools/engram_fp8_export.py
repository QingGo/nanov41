#!/usr/bin/env python3
"""Export nano-V41 bf16 Engram tables to the vLLM official checkpoint layout.

vLLM's ``ParallelEngramEmbedding`` stores each hash head as a *row table* of
``float8_e4m3fn`` values plus one ``ue8m0`` (E8M0) scale byte per 32-element
block.  This script converts our training tables (bf16 rows, width 32) into
``.weight`` (fp8) + ``.scale`` (uint8 E8M0 bytes) tensors and reports the
round-trip error.  It is deliberately independent of a GPU/vLLM install.

Usage:
    python tools/engram_fp8_export.py --ckpt /tmp/p2b_default.pt \
        --out /tmp/p2b_engram_fp8

The generated safetensors keys use the official checkpoint spelling, i.e.
``layers.{i}.engram.embed.weight`` / ``layers.{i}.engram.embed.scale``; the
vLLM DeepSeek-V4.1 weight mapper renames them to
``...engram.embed_tokens.weight`` / ``...weight_scale_inv``.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file

E4M3_MAX = 448.0


def quantize_ue8m0(weight: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Row-wise block FP8 quantisation with E8M0 (power-of-two) scales.

    Returns (fp8 [rows, dim], scale_bytes [rows, dim//block] uint8, stats).
    The dequantisation used by vLLM is ``fp8_value * 2**(byte - 127)``.
    """
    assert weight.ndim == 2 and weight.shape[1] % block == 0
    w = weight.float()
    rows, dim = w.shape
    blocks = w.view(rows, dim // block, block)
    max_abs = blocks.abs().amax(dim=-1)  # [rows, nblocks]
    # Smallest power-of-two scale such that max_abs / scale <= 448.
    ratio = (max_abs / E4M3_MAX).clamp_min(2.0**-126)
    exponent = torch.ceil(torch.log2(ratio)).to(torch.int32)
    exponent = exponent.clamp(-127, 127)
    exponent = torch.where(max_abs == 0, torch.zeros_like(exponent), exponent)
    scale = torch.pow(torch.tensor(2.0), exponent.float()).unsqueeze(-1)  # [rows, nb, 1]
    q3 = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    deq = q3.float() * scale
    q = q3.reshape(rows, dim)
    err = (deq - blocks).abs()
    rel = err / blocks.abs().clamp_min(1e-12)
    stats = {
        "rows": rows,
        "dim": dim,
        "block": block,
        "max_abs": float(max_abs.max()),
        "mean_abs": float(w.abs().mean()),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "mean_rel_err": float(rel.mean()),
        "p99_rel_err": float(torch.quantile(rel.flatten()[:: max(1, rel.numel() // 1_000_000)].float(), 0.99)),
        "scale_exp_min": int(exponent.min()),
        "scale_exp_max": int(exponent.max()),
    }
    byte = (exponent + 127).clamp(0, 255).to(torch.uint8)
    return q, byte, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    report = {"ckpt": args.ckpt, "block": args.block, "tables": {}}

    for name in sorted(sd):
        if not name.endswith(".engram.embed.weight"):
            continue
        w = sd[name]
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        q, scale_byte, stats = quantize_ue8m0(w, args.block)
        official = name.removeprefix("inner.").removesuffix(".weight")
        tensors[official + ".weight"] = q
        tensors[official + ".scale"] = scale_byte
        report["tables"][official] = stats
        print(f"[{official}] {tuple(w.shape)} -> fp8 {tuple(q.shape)} + scale {tuple(scale_byte.shape)} | "
              f"max_abs={stats['max_abs']:.4f} mean_rel_err={stats['mean_rel_err']:.2%} "
              f"p99_rel_err={stats['p99_rel_err']:.2%}", flush=True)

    save_file(tensors, str(out / "engram_tables.safetensors"))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[saved] {out/'engram_tables.safetensors'} ({sum(t.numel() for t in tensors.values())/1e6:.1f}M elements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
