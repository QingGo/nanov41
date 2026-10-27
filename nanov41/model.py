"""Training wrapper around the official DeepSeek-V4.1 reference implementation.

We do not reimplement the architecture: `NanoV41` builds the official `Transformer`
(a patched copy — see `tools/make_patched_ref.py`) so the model math stays exactly the
released reference. What this module adds:

- `forward()` returns per-position logits (`full_logits=True` patch) and a causal-LM loss,
- `build_pristine()` for the parity harness (untouched official code, forward-only),
- parameter counting / config plumbing.

The CED ("causal encoder-decoder") structure of V4.1 is expressed by the compress_ratio
groups in the config: every layer attends a sliding window of raw KV, and a group's first
layer (a `kv_source_layer`) additionally pools every `ratio` tokens into one compressed
KV latent that later layers in the group share. In the released config the last group has
ratio 1, which is how decoder layers get a per-token global KV projection.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patched_env import import_official, import_patched


class NanoV41(nn.Module):
    """A trainable DeepSeek-V4.1-architecture model at nano scale."""

    def __init__(self, cfg: dict, tokenizer=None, param_dtype: str = "bf16"):
        super().__init__()
        self.cfg = dict(cfg)
        official = import_patched()
        self.args = official.ModelArgs(**self.cfg)
        self.inner = _build_transformer(official, self.args, tokenizer, param_dtype)
        if param_dtype == "float32":
            _cast_floating(self.inner, torch.float32)
        self.hc_mult = self.args.hc_mult

    # ------------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_params_non_embedding(self) -> int:
        emb = self.inner.embed.weight.numel() + self.inner.head.weight.numel()
        return self.num_params() - emb

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        """input_ids/labels: [b, s]. Returns logits [b, s, vocab] or the causal-LM loss."""
        logits = inner_forward(self.inner, input_ids, full_logits=True)
        if labels is None:
            return logits
        return causal_lm_loss(logits, labels)

    @torch.no_grad()
    def logits_for(self, input_ids: torch.Tensor) -> torch.Tensor:
        return inner_forward(self.inner, input_ids, full_logits=True)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, temperature: float = 0.0) -> torch.Tensor:
        """Greedy/temperature autoregressive decode using the official incremental path."""
        ids = input_ids
        for _ in range(max_new_tokens):
            _out_ids, logits, _hidden = inner_forward(self.inner, ids)
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = probs.multinomial(1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def init_random_weights(model: nn.Module, std: float = 0.02, seed: int = 0) -> None:
    """The official classes allocate with `torch.empty` (weights come from a checkpoint).
    Parity/smoke runs need a sane random init: norms=1, attention sinks / router bias=0,
    everything else N(0, std)."""
    gen = torch.Generator().manual_seed(seed)
    norm_suffixes = ("norm.weight", "q_norm.weight", "kv_norm.weight", "k_norm.weight")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 0:
                continue
            if name.endswith(norm_suffixes) or name == "norm.weight":
                p.fill_(1.0)
            elif name.endswith("attn_sink") or name.endswith(".bias"):
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype) * std)


def inner_forward(inner, input_ids: torch.Tensor, start_pos: int = 0, full_logits: bool = False):
    """Call the official forward with the torch default device pinned to the input device.

    The official code builds its window/index index tensors with bare `torch.arange` and relies
    on `torch.set_default_device("cuda")` from its __main__ block, so non-CPU runs need this
    context or they mix CPU index tensors with GPU KV tensors.
    """
    kwargs = {}
    if "full_logits" in inspect.signature(inner.forward).parameters:
        kwargs["full_logits"] = full_logits
    with torch.device(input_ids.device):
        return inner.forward(input_ids, start_pos, None, None, **kwargs)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Standard next-token CE over the full sequence."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )


def _build_transformer(official, args, tokenizer, param_dtype: str):
    """Build the official Transformer with consistent dtypes.

    The reference relies on `torch.set_default_dtype` from its __main__ block (embeddings /
    norms / gates use the torch default dtype) while `Linear` uses the module-global
    `default_dtype` derived from `ModelArgs.dtype`. We set both consistently, and allow a
    float32 build for fast/clean CPU parity runs.
    """
    torch_dtype = torch.float32 if param_dtype == "float32" else torch.bfloat16
    prev = official.default_dtype
    official.default_dtype = torch_dtype
    try:
        with official.set_dtype(torch_dtype):
            return official.Transformer(args, tokenizer=tokenizer)
    finally:
        official.default_dtype = prev


def build_pristine(cfg: dict, tokenizer=None, param_dtype: str = "bf16"):
    """Untouched official reference (forward-only, inference_mode). Used by parity tests."""
    official = import_official()
    args = official.ModelArgs(**dict(cfg))
    model = _build_transformer(official, args, tokenizer, param_dtype)
    if param_dtype == "float32":
        _cast_floating(model, torch.float32)
    return model


def _cast_floating(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast all floating params/buffers (the reference builds them in bf16). Complex RoPE
    buffers are left untouched."""
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype)


@torch.no_grad()
def copy_weights(dst: nn.Module, src: nn.Module) -> None:
    """Load a state dict across the patched/pristine copies (module trees are identical)."""
    dst.load_state_dict(src.state_dict(), strict=True)
