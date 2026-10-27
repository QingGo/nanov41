#!/usr/bin/env python3
"""Speedrun-style training entry point (W1 skeleton).

Deliberately small and hackable (nanochat discipline): one file, one model, one loop.
The real pretokenized-data pipeline (DeepSeek tokenizer + FineWeb-Edu/ClimbMix shards)
lands in W2/W3; for now `--data synthetic` runs a deterministic learnable pattern so the
smoke test can show the loss actually going down.

Examples
--------
    # CPU smoke: tiny config, learnable synthetic pattern
    python train.py --config tiny_all_topk --steps 40 --batch-size 4 --seq-len 32

    # inspect a real target config (params/FLOPs estimate, no training)
    python train.py --config d8 --steps 0
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.configs import CONFIGS  # noqa: E402
from nanov41.model import NanoV41, init_random_weights  # noqa: E402
from nanov41.patched_env import import_patched  # noqa: E402
from nanov41.train_utils import fused_head_ce  # noqa: E402


def make_batch(bs: int, seq: int, vocab: int, step: int, device) -> torch.Tensor:
    """Deterministic learnable pattern: token_t = (base + t * stride + step) % vocab.

    A tiny model can fit this within tens of steps, so the smoke loss visibly falls.
    """
    base = (step * 7) % max(vocab - seq, 1)
    t = torch.arange(seq)
    stride = 3
    ids = (base + t * stride) % vocab
    return ids.unsqueeze(0).expand(bs, -1).contiguous().to(device)


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    ratio = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * ratio))


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name.endswith("bias") or "norm" in name) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny_all_topk", choices=sorted(CONFIGS))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # Marin's cooldown lesson: a small z-loss keeps the lm_head norm from exploding during deep
    # LR cooldowns (their Spoonbill run: loss crept up because lm_head norm grew). 1e-4 is their value.
    ap.add_argument("--z-loss", type=float, default=1e-4)
    ap.add_argument("--dtype", default="float32", choices=["bf16", "float32"])
    # fused head+CE：分块算 logits，不落全量 [T, V]（vocab 129280 时省 2–4 GB，b≥16 才装得下）
    ap.add_argument("--fused-head-ce", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    assert args.seq_len <= cfg["max_seq_len"], "seq_len > config max_seq_len"
    assert args.batch_size <= cfg["max_batch_size"], "batch_size > config max_batch_size"

    torch.manual_seed(args.seed)
    model = NanoV41(cfg, param_dtype=args.dtype).to(args.device)
    init_random_weights(model, seed=args.seed)
    n_params = model.num_params()
    fused_ce = (args.fused_head_ce == "on"
                or (args.fused_head_ce == "auto" and args.device != "cpu" and args.dtype == "bf16"))
    if fused_ce:
        M = import_patched()
        M.ParallelHead.forward = lambda self, x, full_logits=False: x   # head 只出 hidden
        head_weight = model.inner.head.weight

    batch_tokens = args.batch_size * args.seq_len
    flops_per_step = 6 * model.num_params_non_embedding() * batch_tokens
    print(
        f"[model] {args.config}: params={n_params/1e6:.2f}M "
        f"(non-emb {model.num_params_non_embedding()/1e6:.2f}M) layers={cfg['n_layers']} "
        f"dim={cfg['dim']} experts={cfg['n_routed_experts']} ratios={cfg['compress_ratios']}",
        flush=True,
    )
    print(f"[budget] batch={args.batch_size}x{args.seq_len}={batch_tokens} tok/step, "
          f"~{flops_per_step/1e9:.2f} GFLOP/step, dtype={args.dtype}", flush=True)
    if args.steps == 0:
        return 0

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        ids = make_batch(args.batch_size, args.seq_len, cfg["vocab_size"], step, args.device)
        if fused_ce:
            loss = fused_head_ce(model(ids), head_weight, ids, z_loss=args.z_loss)
            flat = None
        else:
            logits = model(ids)
            flat = logits[:, :-1].reshape(-1, logits.size(-1)).float()
            loss = F.cross_entropy(flat, ids[:, 1:].reshape(-1))
            if args.z_loss:
                z = flat.logsumexp(-1)
                loss = loss + args.z_loss * z.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            tok_s = ((step + 1) * batch_tokens) / max(time.time() - t0, 1e-9)
            z_msg = ""
            if args.z_loss and flat is not None:
                z_msg = f"  z {flat.logsumexp(-1).mean().item():.2f}"
            print(f"step {step:4d}  loss {loss.item():.4f}{z_msg}  lr {lr:.2e}  {tok_s:8.1f} tok/s", flush=True)

    first, last = losses[0], losses[-1]
    print(f"[done] loss {first:.4f} -> {last:.4f} (delta {last - first:+.4f}) in {time.time() - t0:.1f}s")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg, "state_dict": model.state_dict(), "step": args.steps, "args": vars(args)}, out)
        print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
