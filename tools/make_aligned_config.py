#!/usr/bin/env python3
"""Generate V4.1-Flash-aligned nano model configs (M0 of the alignment program).

Source of truth: `docs/v41-aligned-spec.md` (from arXiv:2609.19969 + `ref/config.json`).
The official model is 552B backbone / 196B Engram, 40 layers (20 encoder + 20 decoder CED),
dim 5120, 128 routed experts top-6 (activation 1.5625%), moe_inter = 0.45*dim, Engram at
layers [1,14] with ngram {2,3,4}, 8 heads, 256-dim/head, tables ~16M rows/head.

This generator produces, for a chosen nano scale:
1. `engine_config` — what the *current* nanov41 engine can run today (decoder-only
   approximation of the aligned structure; CED / CSA2 modes / hierarchical indexer / fp8 are
   recorded in `aligned_spec` and land in milestones M2/M3).
2. `aligned_spec` — the target structural + recipe spec (CED split with layer modes, engram
   geometry, optimizer groups, fp8 scope, sequence length, LR/batch rules).
3. Accounting — backbone/Engram param split, activation fraction, active params, fwd FLOPs
   estimate, memory estimate.

Decisions locked by the user (2026-11-09): dim=512, seq=8192, Engram params ≈ 35% of backbone,
LR scaled by muP eta*(5120/dim)^0.5.

Usage:
    python tools/make_aligned_config.py --layers 12 --dim 512 --seq 8192 \
        --out docs/aligned/d12_aligned.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- official V4.1-Flash constants (report 4.2.1 / 2.4.2 / ref/config.json) --------------
OFFICIAL = dict(
    backbone_params=552e9,
    engram_params=196e9,
    dim=5120,
    layers=40,
    n_heads=64,
    head_dim=512,
    rope_head_dim=64,
    q_lora_rank=1280,
    o_groups=8,
    o_lora_rank=1024,
    window_size=128,
    experts=384,
    activated_experts=6,
    moe_inter_dim=2304,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    base_lr=2.6e-4,
    eng_layers=(1, 14),
    eng_ngram=4,          # orders {2,3,4} -> max_ngram_size 4
    eng_heads=8,
    eng_dim_total=2048,   # per order, i.e. 256/head
    eng_rows_per_head=16e6,
    engram_ngram=4,
    seq_len=65536,
)

_COMMON = dict(
    dtype="bf16", expert_dtype=None, score_func="sqrtsoftplus", gate_temp=1.0,
    norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0, norm_eps=1e-20,
    hc_mult=4, hc_eps=1e-6, compress_rope_theta=1.6e5, rope_theta=1e4,
    rope_factor=16, beta_fast=32, beta_slow=1, original_seq_len=0,
    candidate_source_layer=-1, candidate_topk_blocks=0, candidate_block_size=0,
    vision_n_layers=0, n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=[],
    engram_pad_id=2, engram_compressed_vocab_size=99092,
)


def _norm_uniform(frac: float, max_den: int = 64) -> tuple[int, int]:
    """Closest integer (k, n) with k/n == activation fraction, n <= max_den."""
    best = min(((n, round(frac * n)) for n in range(2, max_den + 1)),
               key=lambda t: abs(t[1] / t[0] - frac))
    return best[1], best[0]  # (topk, experts)


def _primes_rows(layers: list[int], max_ngram: int, n_heads: int, vocab: int) -> list[int]:
    """Exact row counts per Engram layer = sum of distinct prime bucket sizes (ref/engram.py)."""
    try:
        from sympy import isprime
    except Exception:  # noqa: BLE001
        return []

    def next_prime(start: int, seen: set[int]) -> int:
        cand = start + 1
        while not isprime(cand) or cand in seen:
            cand += 1
        return cand

    seen: set[int] = set()
    rows = []
    for _ in layers:
        total = 0
        for _ in range(max_ngram - 1):
            current = vocab - 1
            for _ in range(n_heads):
                current = next_prime(current, seen)
                seen.add(current)
                total += current
        rows.append(total)
    return rows


def make_engram_geometry(backbone_params: float, ratio: float, layers: list[int],
                         max_ngram: int, n_heads: int, head_dim: int,
                         vocab_lo: int = 10_000, vocab_hi: int = 4_000_000) -> dict:
    """Pick `engram_vocab_size` so that Engram table params ≈ ratio * backbone params."""
    target = ratio * backbone_params
    n_buckets = len(layers) * (max_ngram - 1) * n_heads
    lo, hi = vocab_lo, vocab_hi
    best = None
    for _ in range(40):
        mid = (lo + hi) // 2
        rows = _primes_rows(layers, max_ngram, n_heads, mid)
        params = (sum(rows) * head_dim) if rows else n_buckets * mid * head_dim
        if best is None or abs(params - target) < abs(best[2] - target):
            best = (mid, rows, params)
        if params < target:
            lo = mid + 1
        else:
            hi = mid - 1
    vocab, rows, params = best
    return {"engram_vocab_size": vocab, "engram_num_embeddings": rows,
            "engram_max_ngram_size": max_ngram, "engram_n_heads": n_heads,
            "engram_head_dim": head_dim, "engram_layer_ids": layers,
            "engram_params": int(params), "target_ratio": ratio,
            "achieved_ratio": params / backbone_params}


def mode_source_layers(modes: list[str]) -> tuple[list[int], list[int]]:
    """CSA2 mode pattern -> (kv_source_layers, index_source_layers).

    ``full``   = own main KV + own indexer (Full mode)
    ``reindex``= reuse main/index K, recompute Top-K (Reindex mode)
    ``reuse``  = reuse both (Reuse mode); ``swa`` = pure sliding-window layer
    """
    kv = [i for i, m in enumerate(modes) if m == "full"]
    idx = [i for i, m in enumerate(modes) if m in ("full", "reindex")]
    return kv, idx


def build_engine_config(layers: int, dim: int, seq: int, engram: dict, *,
                        mode_lists: dict | None = None,
                        experts: int = 128, topk: int = 2,
                        dense_h: int | None = None,
                        out: pathlib.Path | None = None) -> dict:
    """Decoder-only engine approximation of the aligned structure (current engine limits).

    ``dense_h`` builds a plain SwiGLU dense FFN instead of MoE (patch P6 in
    tools/make_patched_ref.py); the routed-expert fields are then unused placeholders.
    """
    head_dim = 64
    n_heads = max(1, dim // head_dim)
    head_dim = dim // n_heads
    if dense_h is not None:
        assert dense_h > 0
        experts, topk = 1, 1
    else:
        assert topk / experts == 1 / 64, (
            f"activation fraction must match official 6/384=1/64, got {topk}/{experts}")
    enc = layers // 2
    dec = layers - enc
    # engine approximation: 1 SWA layer + encoder CSA2 group (ratio 2) + decoder groups (ratio 1)
    n_swa = max(1, round(0.05 * layers))
    n_enc_csa = max(1, enc - n_swa)
    n_dec = dec
    compress = [0] * n_swa + [2] * n_enc_csa + [1] * n_dec
    # first layer of each ratio group is the KV/index source; an explicit CSA2 mode pattern
    # (encoder Full+Reuse, decoder Full/Reindex+Reuse) can override this so Reindex layers are
    # actually built (they need an indexer but no compressor).
    if mode_lists is not None:
        kv_sources = list(mode_lists["kv_source_layers"])
        idx_sources = list(mode_lists["index_source_layers"])
    else:
        kv_sources, idx_sources = [], []
        prev = None
        for i, r in enumerate(compress):
            if r != 0 and r != prev:
                kv_sources.append(i)
                idx_sources.append(i)
            prev = r
    q_lora = max(32, (dim // 4 + 15) // 16 * 16)
    o_lora = max(32, (dim // 5 + head_dim - 1) // head_dim * head_dim)
    moe_inter = dense_h if dense_h is not None else max(64, int(round(0.45 * dim / 64.0)) * 64)
    vocab = 129_280
    cfg = dict(
        _COMMON,
        vocab_size=vocab, dim=dim, moe_inter_dim=moe_inter, n_layers=layers,
        n_heads=n_heads, n_routed_experts=experts, n_shared_experts=1,
        n_activated_experts=topk, q_lora_rank=q_lora, head_dim=head_dim,
        rope_head_dim=max(8, head_dim // 4), o_groups=min(8, n_heads),
        o_lora_rank=o_lora, window_size=OFFICIAL["window_size"],
        compress_ratios=compress, kv_source_layers=kv_sources,
        index_source_layers=idx_sources,
        index_n_heads=max(1, n_heads // 2),
        index_head_dim=max(8, head_dim // 4),
        index_topk=max(8, int(round(OFFICIAL["index_topk"] * seq / OFFICIAL["seq_len"] / 8) * 8)),
        hc_sinkhorn_iters=20, max_batch_size=8, max_seq_len=seq,
    )
    if dense_h is not None:
        cfg.update(ffn_impl="dense", dense_inter_dim=dense_h)
    else:
        cfg.update(ffn_impl="moe")
    cfg.update(engram)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    return cfg


_ENGRAM_MODEL_KEYS = ("engram_vocab_size", "engram_num_embeddings", "engram_max_ngram_size",
                      "engram_n_heads", "engram_head_dim", "engram_layer_ids")
_ENGRAM_META_KEYS = ("engram_params", "target_ratio", "achieved_ratio")


def account(cfg: dict, engram_meta: dict | None = None) -> dict:
    """Build the model and split params into backbone / engram / active / inactive experts."""
    import torch  # noqa: F401
    import sys
    sys.path.insert(0, str(ROOT))
    from nanov41.model import NanoV41

    engine_cfg = {k: v for k, v in cfg.items()
                  if k not in _ENGRAM_META_KEYS and k not in _ENGRAM_MODEL_KEYS}
    model = NanoV41(engine_cfg, tokenizer=None, param_dtype="float32")
    named = dict(model.named_parameters())
    total = sum(p.numel() for p in named.values())
    expert_parts = [p for n, p in named.items() if ".ffn.experts." in n]
    experts_total = sum(p.numel() for p in expert_parts)
    embed = named["inner.embed.weight"].numel()
    head = named["inner.head.weight"].numel()
    ffn_impl = cfg.get("ffn_impl", "moe")
    if ffn_impl == "dense":
        inactive = 0.0
        active = total - embed
        activation_fraction = 1.0
    else:
        n_experts = cfg["n_routed_experts"]
        topk = cfg["n_activated_experts"]
        inactive = experts_total * (1.0 - topk / n_experts)
        active = total - embed - inactive
        activation_fraction = topk / n_experts
    engram_params = int((engram_meta or {}).get("engram_params", 0))
    backbone = total
    return {
        "backbone_params": backbone,
        "backbone_non_embedding": total - embed - head,
        "embed_plus_head": embed + head,
        "experts_total": experts_total,
        "inactive_expert_params": int(inactive),
        "active_params": int(active),
        "engram_params": engram_params,
        "total_with_engram": backbone + engram_params,
        "ffn_impl": ffn_impl,
        "activation_fraction": activation_fraction,
        "fwd_flops_per_token_est": 2 * active,
        "engram_backbone_ratio": engram_params / backbone,
        "engram_nonembed_ratio": engram_params / (total - embed - head),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--engram-ratio", type=float, default=0.355,
                    help="Engram params / backbone params (official 196/552)")
    ap.add_argument("--out", default="docs/aligned/d12_aligned.json")
    ap.add_argument("--modes", default="ratio", choices=["ratio", "official"],
                    help="engine CSA2 sources: ratio (M0 approximation, no Reindex fields) or "
                         "official (derive from the layer_modes pattern; adds Reindex layers)")
    ap.add_argument("--engram-rows-scale", type=float, default=1.0,
                    help="extra multiplier on the engram param target (for ablations)")
    ap.add_argument("--dense-h", type=int, default=0,
                    help="build a plain SwiGLU dense FFN of hidden width H instead of MoE")
    ap.add_argument("--engram-from", default="",
                    help="copy the Engram geometry from this existing config JSON (e.g. "
                         "docs/aligned/d12_aligned.json) instead of deriving it from the backbone")
    args = ap.parse_args()

    modes = _layer_modes(args.layers)
    mode_lists = None
    if args.modes == "official":
        kv_s, idx_s = mode_source_layers(modes)
        mode_lists = {"kv_source_layers": kv_s, "index_source_layers": idx_s}
    dense_h = args.dense_h or None
    base_cfg = build_engine_config(args.layers, args.dim, args.seq, {},
                                   experts=args.experts, topk=args.topk,
                                   mode_lists=mode_lists, dense_h=dense_h)
    base = account(base_cfg)
    backbone = base["backbone_params"]
    if args.engram_from:
        # copy the production Engram geometry verbatim (same table for MoE and dense arms)
        _src = json.loads(pathlib.Path(args.engram_from).read_text())
        engram = dict(_src["engram"])
        engram.setdefault("target_ratio", 0.0)
        engram.setdefault("achieved_ratio",
                          engram.get("engram_params", 0) / max(1, base["backbone_non_embedding"]))
    else:
        # official engram head dim = 256 = dim_official/20; ratio base = non-embedding backbone
        # (official embed+head is 0.24% of the backbone; at nano scale they are ~20%, so using the
        # full backbone as the base would inflate Engram 5x)
        eng_head_dim = max(16, 2 ** round(math.log2(args.dim / 20)))
        engram = make_engram_geometry(base["backbone_non_embedding"] * args.engram_rows_scale,
                                      args.engram_ratio,
                                      layers=[1, max(1, round(0.35 * args.layers))],
                                      max_ngram=OFFICIAL["engram_ngram"],
                                      n_heads=OFFICIAL["eng_heads"],
                                      head_dim=eng_head_dim)
    cfg = build_engine_config(args.layers, args.dim, args.seq,
                              {k: engram[k] for k in _ENGRAM_MODEL_KEYS},
                              experts=args.experts, topk=args.topk,
                              mode_lists=mode_lists, dense_h=dense_h)
    acct = account(cfg, engram)

    # recipe rules (user decisions 2026-11-09)
    lr = OFFICIAL["base_lr"] * math.sqrt(OFFICIAL["dim"] / args.dim)        # muP eta ~ 1/sqrt(dim)
    muon_lr = lr                                                            # update RMS rescaled to 0.18
    adamw_lr = lr
    sinkhorn_lr = lr                                                        # gamma=0.18 inside the update
    engram_lr = 5.0 * lr

    aligned = {
        "schema": "nanov41-v41-aligned/1",
        "source": "arXiv:2609.19969 + ref/config.json (see docs/v41-aligned-spec.md)",
        "official": OFFICIAL,
        "scale": {"dim": args.dim, "layers": args.layers, "seq_len": args.seq,
                  "down_scale": OFFICIAL["dim"] / args.dim},
        "ced": {
            "encoder_layers": list(range(args.layers // 2)),
            "decoder_layers": list(range(args.layers // 2, args.layers)),
            "decoder_global_kv_from_layer": args.layers // 2 - 1,
            "prefill_skips_decoder_global_kv": True,
        },
        "layer_modes": modes,
        "engine_mode_sources": {
            "kv_source_layers": cfg["kv_source_layers"],
            "index_source_layers": cfg["index_source_layers"],
            "reindex_layers": [i for i, m in enumerate(modes) if m == "reindex"],
        },
        "hierarchical_indexer": {
            "enabled": True, "stage": "post_training",
            "candidate_source_layer": next(
                (i for i, m in enumerate(modes[args.layers // 2:], start=args.layers // 2)
                 if m == "full"), -1),
            "candidate_topk_blocks": 128, "candidate_block_size": 8,
            "note": "official 2048 blocks x 8 pos for 64K context; scaled to seq/4 candidates; "
                    "candidate_source_layer is the first decoder Full layer (set engine "
                    "candidate_source_layer<0 during pretraining)",
        },
        "engram": {k: engram[k] for k in ("engram_layer_ids", "engram_max_ngram_size",
                                          "engram_n_heads", "engram_head_dim",
                                          "engram_vocab_size", "engram_num_embeddings",
                                          "engram_params", "achieved_ratio")},
        "optimizer": {
            "muon": {"params": "backbone linear matrices + engram projections",
                     "headwise": ["Q", "K"], "momentum": 0.95, "weight_decay": 0.1,
                     "update_rms_rescale": 0.18, "lr": muon_lr},
            "adamw": {"params": "norms (wd), biases/scalings (no wd)",
                      "betas": [0.9, 0.95], "eps": 1e-20, "weight_decay": 0.1, "lr": adamw_lr},
            "sinkhorn": {"params": "engram tables + token embedding + head",
                         "momentum": 0.95, "gamma": 0.18, "K": 11, "tau": 1e-3,
                         "eps": 1e-20, "weight_decay": 0.0, "lr": sinkhorn_lr,
                         "engram_lr_scale": 5.0, "engram_lr": engram_lr},
        },
        "precision": {"pretrain": {"linear": "fp8", "engram_embedding": "fp8",
                                   "norms_attention_scores": "bf16"},
                      "post_training": {"main_kv": "fp4_mxfp4_qat", "indexer_qk": "fp4_qat",
                                        "swa_kv": "fp8"}},
        "lr_rule": {"base": OFFICIAL["base_lr"], "muP": "eta*(dim_official/dim)^0.5",
                    "value": lr, "warmup_steps": 2000,
                    "schedule": "constant -> cosine 28T-40T (official, scaled)",
                    "final_frac": 0.1},
        "batch_rule": {"official": 100.6e6, "nano": "2^19 tokens (Power-Lines scalable)",
                       "tokens_per_step": 2 ** 19, "grad_accum_required": True},
        "misc": {"aux_free_bias_speed": 1e-3, "sequence_balance_loss": 1e-4,
                 "sample_level_attention_mask": True, "mtp": "omitted (V4.1)",
                 "dspark": "post_pretrain_stage", "vision": "deferred"},
        "accounting": acct,
        "engine_config": cfg,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aligned, indent=2))
    print(f"[aligned] wrote {out}")
    if cfg.get("ffn_impl") == "dense":
        print(f"[aligned] dim={args.dim} layers={args.layers} seq={args.seq} "
              f"ffn=DENSE h={cfg['dense_inter_dim']}")
    else:
        print(f"[aligned] dim={args.dim} layers={args.layers} seq={args.seq} "
              f"experts={cfg['n_routed_experts']}/{cfg['n_activated_experts']} "
              f"moe_inter={cfg['moe_inter_dim']}")
    for k in ("backbone_params", "backbone_non_embedding", "active_params", "engram_params",
              "total_with_engram", "activation_fraction", "engram_backbone_ratio",
              "engram_nonembed_ratio"):
        v = acct[k]
        print(f"  {k:24s} {v:,.4f}" if isinstance(v, float) else f"  {k:24s} {v:,}")
    print(f"  fwd FLOPs/token (2*active) {acct['fwd_flops_per_token_est']:,}")
    print(f"  LR (muP) = {lr:.3e}  | Muon {muon_lr:.3e} | Sinkhorn {sinkhorn_lr:.3e} "
          f"| Engram {engram_lr:.3e}")
    print(f"  layer_modes = {modes}")
    print(f"  kv_source_layers = {cfg['kv_source_layers']}  "
          f"index_source_layers = {cfg['index_source_layers']}")
    return 0


def _layer_modes(layers: int) -> list[str]:
    """Official mode pattern scaled to `layers` (encoder 2/40 SWA, then Full+Reuse, decoder
    Full/Reindex+Reuse)."""
    n_swa = max(1, round(0.05 * layers))
    enc = layers // 2
    modes = ["swa"] * n_swa
    # encoder CSA2 m=2: groups of 6 -> Full + 5 Reuse
    g = max(1, enc - n_swa)
    modes += ["full"] + ["reuse"] * (g - 1)
    dec = layers - len(modes)
    # decoder CSA2 m=1: 4-layer groups; first group Full, rest Reindex, each + Reuse
    group = 4
    first = True
    while dec > 0:
        n = min(group, dec)
        modes.append("full" if first else "reindex")
        modes += ["reuse"] * (n - 1)
        dec -= n
        first = False
    return modes[:layers]


if __name__ == "__main__":
    raise SystemExit(main())
