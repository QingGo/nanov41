#!/usr/bin/env python3
"""Export a nano-V41 training checkpoint to a vLLM DeepSeek-V4.1 HF checkpoint dir.

Scope: this is the *conversion prototype* for the single-GPU (TP=1, no
quantization) bring-up path used by the eval/OPD harness.

- keys: strip the training wrapper's ``inner.`` prefix and emit official
  checkpoint spelling (``layers.*``, ``embed.weight``, ``head.weight``,
  ``norm.weight``).  vLLM's ``_make_deepseek_v4_weights_mapper`` then maps
  them to ``model.layers.*``, ``model.embed_tokens.weight``, ``lm_head.weight``
  and ``...engram.embed_tokens.weight[_scale_inv]``.
- Engram tables: bf16 rows -> ``float8_e4m3fn`` rows + one ``ue8m0`` scale
  byte per 32-wide block (see tools/engram_fp8_export.py for the quantiser).
- all other weights are exported as bf16/float32; the config deliberately has
  **no quantization_config**, so vLLM instantiates bf16 linear/attention/MoE
  layers.  If a vLLM version insists on the official FP8 linear layout, the
  next step is to extend this script with block-FP8 linear quantisation.
- config: a small ``text_config`` built from the nano training cfg; tokenizer
  files are copied next to it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file

from engram_fp8_export import quantize_ue8m0

CFG_MAP = {
    "vocab_size": "vocab_size",
    "dim": "hidden_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "head_dim": "head_dim",
    "rope_head_dim": "qk_rope_head_dim",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "n_activated_experts": "num_experts_per_tok",
    "route_scale": "routed_scaling_factor",
    "norm_eps": "rms_norm_eps",
    "swiglu_limit": "swiglu_limit",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "window_size": "sliding_window",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "compress_rope_theta": "compress_rope_theta",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_pad_id": "engram_pad_token_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "compress_ratios": "compress_ratios",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
}


def build_config(cfg: dict) -> dict:
    text = {
        "model_type": "deepseek_v41_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": int(cfg.get("max_seq_len", 1024)),
        "num_key_value_heads": 1,
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": float(cfg.get("rope_factor", 1)),
            "beta_fast": float(cfg.get("beta_fast", 32)),
            "beta_slow": float(cfg.get("beta_slow", 1)),
            "original_max_position_embeddings": int(cfg.get("original_seq_len") or 65536),
        },
        "scoring_func": cfg.get("score_func", "sqrtsoftplus"),
        "topk_method": "noaux_tc",
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "num_nextn_predict_layers": int(cfg.get("n_mtp_layers", 0) or 0),
        "dspark_block_size": int(cfg.get("dspark_block_size", 0) or 0),
        "dspark_target_layer_ids": list(cfg.get("dspark_target_layer_ids") or []),
        "expert_dtype": "bf16",
    }
    for src, dst in CFG_MAP.items():
        if src in cfg:
            text[dst] = cfg[src]
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 129264,
        "text_config": text,
    }


def convert_state_dict(sd: dict, block: int = 32) -> tuple[dict, dict]:
    out: dict[str, torch.Tensor] = {}
    stats = {"tables": {}, "n_tensors": 0, "skipped": []}
    for name, tensor in sd.items():
        if not name.startswith("inner."):
            stats["skipped"].append(name)
            continue
        key = name.removeprefix("inner.")
        if key.startswith("engram_hash.") or key.endswith(".engram_hash"):
            # runtime hash constants, not model parameters
            continue
        if key.endswith(".engram.embed.weight"):
            q, scale_byte, st = quantize_ue8m0(tensor, block)
            base = key.removesuffix(".weight")
            out[base + ".weight"] = q.contiguous()
            out[base + ".scale"] = scale_byte.contiguous()
            stats["tables"][base] = st
            stats["n_tensors"] += 2
            continue
        if key == "head.weight":
            out[key] = tensor.to(torch.bfloat16).contiguous()
        else:
            out[key] = tensor.contiguous()
        stats["n_tensors"] += 1
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    ap.add_argument("--tokenizer-config", default="")
    ap.add_argument("--chat-template", default="")
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = ckpt["cfg"], ckpt["state_dict"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state, stats = convert_state_dict(sd, args.block)
    save_file(state, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(build_config(cfg), indent=2) + "\n")
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    if args.tokenizer_config:
        shutil.copyfile(args.tokenizer_config, out / "tokenizer_config.json")
    if args.chat_template:
        shutil.copyfile(args.chat_template, out / "chat_template.jinja")
    report = {
        "ckpt": args.ckpt,
        "out": str(out),
        "n_state_tensors": len(state),
        "size_mb": sum(t.numel() * t.element_size() for t in state.values()) / 1e6,
        "engram_tables": stats["tables"],
        "skipped": stats["skipped"],
        "note": "no quantization_config; bf16 linears + fp8/ue8m0 Engram table (TP=1 bring-up prototype)",
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("out", "n_state_tensors", "size_mb")}, indent=1))
    for name, st in stats["tables"].items():
        print(f"  {name}: rows={st['rows']} fp8+scale, mean_rel_err={st['mean_rel_err']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
