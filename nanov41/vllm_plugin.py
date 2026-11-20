# SPDX-License-Identifier: MIT
"""vLLM fallback model plugin for NanoV41 (experiment).

Instead of patching vLLM's production DeepSeek-V4.1 kernels (compressor / mHC /
o-proj / sparse MLA / MoE all bake in 512/64/5120/1024 shapes), this plugin
registers a *generic* vLLM model that reuses our reference forward.  It is a
correctness/fallback path, not a fast path: the model keeps its own reference
KV/hash caches and works for a single request at a time (``max_num_seqs=1``).

The engine still handles scheduling, sampling and request management; the
model forward is our generic NanoV41 forward.  The cache is advanced with the
official incremental ``start_pos`` interface, so it is the same path as
``nanov41/generate.py`` (which is bit-identical to full-prefix decoding).
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import torch
from torch import nn

try:  # imported inside vLLM only
    from vllm.model_executor.models.registry import ModelRegistry
except Exception:  # pragma: no cover - allows plain import outside vLLM
    ModelRegistry = None

from .generate import reset_caches
from .model import NanoV41, inner_forward
from .nano_engram import install as install_nano_engram


class NanoV41ForCausalLM(nn.Module):
    """Minimal vLLM model: generic NanoV41 forward + reference KV/hash caches."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        model_path = getattr(self.config, "_name_or_path", "") or "."
        cfg_path = os.path.join(model_path, "nano_cfg.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tok_path = cfg.get("tokenizer_json") or os.path.join(model_path, "tokenizer.json")
        tokenizer = None
        if cfg.get("engram_layer_ids"):
            from .nano_engram import load_tokenizer_adapter
            tokenizer = load_tokenizer_adapter(tok_path)
        # Use the nano bf16 Engram table, exactly like ModelProvider does.
        install_nano_engram()
        model_cfg = {k: v for k, v in cfg.items() if k != "ckpt_path"}
        self.model = NanoV41(model_cfg, tokenizer=tokenizer, param_dtype="bfloat16")
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.ndim == 0:
                    continue
                if name.endswith(("norm.weight", "q_norm.weight", "kv_norm.weight", "k_norm.weight")):
                    p.fill_(1.0)
                elif name.endswith("attn_sink") or name.endswith(".bias"):
                    p.zero_()
                else:
                    p.normal_(0.0, 0.02)
        self.model.eval()
        self._ckpt_path = cfg.get("ckpt_path")
        self._ckpt_loaded = False
        self._last_start = None
        self.supports_pp = False
        self._vocab_size = int(cfg["vocab_size"])
        # vLLM's scheduler requires at least one cacheable attention group.  Our
        # reference model keeps its own caches, so this dummy layer only declares
        # a tiny KV-cache spec; its forward is never called by the plugin.
        try:
            from vllm.model_executor.layers.attention.attention import Attention
            self._dummy_attn = Attention(
                num_heads=1, head_size=128, scale=1.0, num_kv_heads=1,
                prefix="layers.0.nano_attn",
            )
        except Exception as _exc:
            self._dummy_attn = None
        # vLLM's runner looks for these attributes on some paths.
        self.logits_processor = None
        try:
            from vllm.model_executor.layers.logits_processor import LogitsProcessor
            self.logits_processor = LogitsProcessor(self._vocab_size)
        except Exception:
            pass

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.inner.embed(input_ids)

    def get_input_embeddings(self, *args, **kwargs):
        if args:
            return self.embed_input_ids(args[0])
        return self.model.inner.embed.weight

    @torch.no_grad()
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        """Run the generic reference forward and return per-token logits.

        vLLM's model runner calls ``compute_logits`` on the returned tensor; this
        plugin returns logits directly and ``compute_logits`` is the identity.
        Single-request/no-paged-cache: our reference caches live inside the model.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        start_pos = int(positions[0].item()) if positions is not None and positions.numel() else 0
        if os.environ.get("NANO_VLLM_DEBUG"):
            print(f"[nano-vllm] fwd ids={input_ids.flatten()[:12].tolist()} pos={positions.flatten()[:12].tolist()} start={start_pos} shape={tuple(input_ids.shape)}", flush=True)
        if start_pos == 0:
            reset_caches(self.model.inner)
        logits = inner_forward(self.model.inner, input_ids, start_pos=start_pos, full_logits=True)
        if os.environ.get("NANO_VLLM_DEBUG") and input_ids.shape[1] == 5 and start_pos == 0:
            flat = logits.reshape(-1, logits.shape[-1])[-1].float()
            vals, idx = flat.topk(5)
            print(f"[nano-vllm] prefill last argmax={idx.tolist()} vals={vals.tolist()}", flush=True)
        # vLLM indexes the forward output with logits_indices, so return a 2-D
        # [num_tokens, vocab] tensor (the plugin's "hidden states" are logits;
        # compute_logits is the identity).
        return logits.reshape(-1, logits.shape[-1])

    def compute_logits(self, hidden_states: torch.Tensor):
        # Our generic forward returns [batch, tokens, vocab]; vLLM wants
        # [num_tokens, vocab].
        if hidden_states.dim() == 3:
            return hidden_states.reshape(-1, hidden_states.shape[-1])
        return hidden_states

    def _load_ckpt(self) -> None:
        if not self._ckpt_path or self._ckpt_loaded:
            return
        try:
            from safetensors.torch import load_file
            sd = load_file(self._ckpt_path)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self._ckpt_loaded = True
            print(f"[nano_vllm] ckpt loaded missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[nano_vllm] checkpoint load failed: {exc}", flush=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # vLLM's dummy loader calls this after construction; load our real
        # checkpoint here so a dummy/random init cannot overwrite it.
        del weights
        self._load_ckpt()
        # Tell vLLM's weight tracker that every parameter was initialized by
        # our own checkpoint load (otherwise real-checkpoint loading raises).
        return {name for name, _ in self.named_parameters()}


if ModelRegistry is not None:
    ModelRegistry.register_model(
        "NanoV41ForCausalLM", "nanov41.vllm_plugin:NanoV41ForCausalLM"
    )
