"""Logits providers for the eval harness.

The harness only needs one thing from a model: ``logits(input_ids) -> [B, S, vocab]``.  The
indirection buys three things:

* **testable math** — `FakeProvider` yields exactly-known NLLs (uniform logits give
  ``ln(vocab)`` per token; oracle logits give 0), so BPB accounting is unit-tested without a
  model or a GPU;
* **CPU/GPU symmetry** — the same harness code runs a random-init tiny model on a laptop and a
  d12 checkpoint on a 5090;
* **honest accounting** — the provider reports forward FLOPs/token, so eval throughput can be
  converted to an estimated eval cost instead of being hand-waved.

The model provider is a thin wrapper around `NanoV41` and uses the same patched-reference
forward as training (`full_logits=True`), so eval numbers are produced by the training port,
not a separate implementation.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any
from typing import Protocol

import torch


class LogitsProvider(Protocol):
    vocab_size: int
    flops_per_token: float | None

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


class FakeProvider:
    """Analytically-known logits for harness tests and CLI smoke runs.

    kind="uniform": all logits 0 -> every target has NLL = ln(vocab_size).
    kind="oracle":  position i is a one-hot on input_ids[i+1] -> NLL ~ 0 (upper bound check).
    """

    def __init__(self, vocab_size: int, kind: str = "uniform"):
        if kind not in ("uniform", "oracle"):
            raise ValueError(f"unknown fake provider kind {kind!r}")
        self.vocab_size = int(vocab_size)
        self.kind = kind
        self.flops_per_token = None

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        if self.kind == "oracle" and s > 1:
            tgt = input_ids[:, 1:]
            logits[:, :-1, :].scatter_(2, tgt.unsqueeze(-1), 30.0)
        return logits

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


def checkpoint_sha256(path: str | pathlib.Path) -> str | None:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flops_per_token_for_model(model, cfg: dict) -> float:
    """Analytic forward FLOPs/token: 2 * (head + activated backbone params).

    Same convention as tools/bench_graph_train.py (d12: 138.9 MFLOP analytic vs 141.3 MFLOP
    measured with a FLOP counter; 1.7% low because the counter also sees attention/indexer).
    """
    V = int(cfg["vocab_size"])
    head = V * int(cfg["dim"])
    experts = (
        int(cfg["n_layers"])
        * int(cfg["n_routed_experts"])
        * 3
        * int(cfg["dim"])
        * int(cfg["moe_inter_dim"])
    )
    inactive = experts * (1.0 - int(cfg["n_activated_experts"]) / int(cfg["n_routed_experts"]))
    active = int(model.num_params_non_embedding()) - inactive
    return 2.0 * (head + active)


class ModelProvider:
    """NanoV41 logits provider with optional checkpoint loading."""

    def __init__(self, *, config_name: str | None = None, cfg: dict | None = None,
                 dtype: str = "float32", device: str = "cpu",
                 checkpoint: str | pathlib.Path | None = None,
                 random_init: bool = False, seed: int = 0):
        from ..configs import CONFIGS
        from ..model import NanoV41, init_random_weights

        if dtype not in ("float32", "bf16"):
            raise ValueError(f"dtype must be float32 or bf16, got {dtype!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.config_name = config_name
        self.checkpoint = str(checkpoint) if checkpoint else None

        ckpt = None
        if checkpoint is not None:
            ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                raise ValueError(
                    f"{checkpoint}: expected a train.py checkpoint dict with 'state_dict'"
                )
            if cfg is None:
                cfg = dict(ckpt.get("cfg") or {})
                if not cfg:
                    raise ValueError(f"{checkpoint}: checkpoint has no 'cfg'; pass --config")
            state_dict = ckpt["state_dict"]
        else:
            if not random_init:
                raise ValueError("pass a checkpoint or random_init=True")
            if cfg is None:
                if config_name is None:
                    raise ValueError("pass cfg or config_name")
                cfg = CONFIGS[config_name]
            state_dict = None
        if config_name is None and cfg is not None:
            # best-effort label: match a known config by equality
            from ..configs import CONFIGS

            for name, known in CONFIGS.items():
                if known == cfg:
                    config_name = name
                    break
        self.config_name = config_name
        self.cfg = dict(cfg)
        self.engram = bool(self.cfg.get("engram_layer_ids"))
        tokenizer = None
        if self.engram:
            from ..nano_engram import install, load_tokenizer_adapter

            tok_path = os.environ.get("NANOV41_DS_TOKENIZER_JSON")
            if not tok_path:
                raise ValueError(
                    "this checkpoint has Engram layers; set NANOV41_DS_TOKENIZER_JSON to the "
                    "pinned DeepSeek-V4.1-Flash tokenizer.json so the n-gram hash map can be built"
                )
            install()
            tokenizer = load_tokenizer_adapter(tok_path)
        if self.dtype == "float32":
            param_dtype = "float32"
        else:
            param_dtype = "bf16"
        # The official classes allocate with torch.empty; initialise first so random_init is
        # meaningful and a missing/unexpected checkpoint key can never leave NaNs behind.
        torch.manual_seed(seed)
        self.model = NanoV41(self.cfg, tokenizer=tokenizer, param_dtype=param_dtype)
        init_random_weights(self.model, seed=seed)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.vocab_size = int(self.cfg["vocab_size"])
        self.n_params = int(self.model.num_params())
        self.n_params_non_embedding = int(self.model.num_params_non_embedding())
        self.flops_per_token = _flops_per_token_for_model(self.model, self.cfg)
        self.checkpoint_sha256 = checkpoint_sha256(checkpoint) if checkpoint else None
        if self.engram:
            from ..nano_engram import engram_param_count

            self.engram_table_params = engram_param_count(self.model)
        else:
            self.engram_table_params = 0

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        return self.model.logits_for(input_ids)

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final normalized hidden states [B,S,D] without materialising [B,S,V] logits."""
        from ..model import inner_forward
        input_ids = input_ids.to(self.device)
        inner = self.model.inner
        original = inner.head.forward
        inner.head.forward = lambda x, full_logits=False: x  # type: ignore[method-assign]
        try:
            hidden = inner_forward(inner, input_ids, full_logits=True)
        finally:
            inner.head.forward = original  # type: ignore[method-assign]
        return hidden

    @torch.no_grad()
    def nll(self, input_ids: torch.Tensor, chunk: int = 512) -> "np.ndarray":
        """Per-position NLL [B,S-1] without a full [B,S,V] logits allocation.

        The head GEMM + logsumexp are tiled over positions: hidden [B,S,D] is small
        (0.4 MB/window) and only `chunk x vocab` logits exist at a time.  This is the
        eval-side counterpart of the training fused head and a drop-in faster path for
        the T0 harness.
        """
        import numpy as np
        input_ids = input_ids.to(self.device)
        hidden = self.hidden(input_ids)
        b, s, d = hidden.shape
        weight = self.model.inner.head.weight
        if os.environ.get("NANOV41_EVAL_FUSED_HEAD", "").lower() in ("1", "true", "yes"):
            try:
                from ..fused_ce_head_v3 import fused_head_nll_v3
                return fused_head_nll_v3(hidden, weight, input_ids).double().cpu().numpy()
            except Exception as exc:  # noqa: BLE001
                print(f"[nanov41] fused-head NLL fell back to chunked path: {exc}", file=__import__("sys").stderr, flush=True)
        out = np.empty((b, s - 1), dtype=np.float64)
        for start in range(0, s - 1, chunk):
            end = min(start + chunk, s - 1)
            h = hidden[:, start:end, :].reshape(-1, d).to(weight.dtype)
            tgt = input_ids[:, start + 1:end + 1].reshape(-1)
            logits = torch.nn.functional.linear(h, weight).float()
            nll = logits.logsumexp(-1) - logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out[:, start:end] = nll.reshape(b, end - start).double().cpu().numpy()
        return out

    def close(self) -> None:
        del self.model
        self.model = None  # type: ignore[assignment]
        if self.device.type == "cuda":  # pragma: no cover - depends on host
            torch.cuda.empty_cache()

    def describe(self) -> dict:
        return {
            "provider": "model",
            "config": self.config_name,
            "vocab_size": self.vocab_size,
            "n_params": self.n_params,
            "n_params_non_embedding": self.n_params_non_embedding,
            "dtype": self.dtype,
            "device": str(self.device),
            "flops_per_token": self.flops_per_token,
            "engram": self.engram,
            "engram_table_params": self.engram_table_params,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "nll_mode": getattr(self, "nll_mode", "auto"),
        }


def make_provider(*, provider: str, config_name: str | None = None,
                  checkpoint: str | pathlib.Path | None = None,
                  random_init: bool = False, dtype: str = "float32", device: str = "cpu",
                  seed: int = 0, fake_vocab_size: int | None = None,
                  fake_kind: str = "uniform") -> LogitsProvider:
    if provider == "fake":
        if fake_vocab_size is None:
            raise ValueError("fake provider needs fake_vocab_size (e.g. from the dataset)")
        return FakeProvider(fake_vocab_size, kind=fake_kind)
    if provider != "model":
        raise ValueError(f"unknown provider {provider!r}")
    return ModelProvider(config_name=config_name, checkpoint=checkpoint, random_init=random_init,
                         dtype=dtype, device=device, seed=seed)


def describe_provider(provider: LogitsProvider) -> dict:
    if isinstance(provider, FakeProvider):
        return {"provider": "fake", "fake_kind": provider.kind,
                "vocab_size": provider.vocab_size}
    if isinstance(provider, ModelProvider):
        return provider.describe()
    return {"provider": type(provider).__name__, "vocab_size": getattr(provider, "vocab_size", None)}

