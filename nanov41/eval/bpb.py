"""Bits-per-byte (BPB) accounting and window bookkeeping.

Definition used throughout the harness::

    BPB = sum_{scored tokens} NLL_nats(token | context) / (ln 2 * UTF-8 bytes of scored tokens)

This is tokenizer-independent (nanochat/GPT-style bits-per-byte), so it can be compared across
models that use the same *evaluation text* even if their tokenizers differ, and it stays
meaningful when a metric covers only a token subset (assistant turns, tool output, ...).

Two rules keep the number honest and cheap:

* non-overlapping windows of ``window`` tokens; the first token of each window is not scored
  because it has no in-window context.  The dropped fraction is ``1/window`` of the tokens (and
  reported); increasing the window shrinks it.
* bytes are counted *only for the tokens that are scored* (the per-token byte table is exact),
  so masking a subset does not silently change the denominator.

`BPBAccumulator` also stores per-window records so a window-level bootstrap CI can be reported
(window-level resampling is a rough but useful stability check; windows are not independent).
"""

from __future__ import annotations

import math

import numpy as np

from .data import ALL_TYPES, TYPE_IGNORE, TYPE_NAMES

LN2 = math.log(2.0)


def _new_counts() -> dict:
    return {"nll_nats": 0.0, "tokens": 0, "bytes": 0}


def finalize_counts(counts: dict) -> dict:
    """Turn raw sums into a BPB record (in place-free: returns a new dict)."""
    nll = float(counts["nll_nats"])
    n_tokens = int(counts["tokens"])
    n_bytes = int(counts["bytes"])
    out = {
        "nll_nats": nll,
        "tokens": n_tokens,
        "bytes": n_bytes,
        "bpb": (nll / LN2 / n_bytes) if n_bytes else float("nan"),
        "bits_per_token": (nll / LN2 / n_tokens) if n_tokens else float("nan"),
        "bytes_per_token": (n_bytes / n_tokens) if n_tokens else float("nan"),
    }
    return out


class BPBAccumulator:
    """Accumulate NLL/bytes per token type over evaluation windows."""

    def __init__(self) -> None:
        self.counts: dict[str, dict] = {"all": _new_counts()}
        for t in ALL_TYPES:
            self.counts[TYPE_NAMES[t]] = _new_counts()
        # per-window records: type name -> list[(nll, tokens, bytes)]
        self.records: dict[str, list[tuple[float, int, int]]] = {}

    def add(self, nll: np.ndarray, label_ids: np.ndarray, types: np.ndarray,
            token_bytes: np.ndarray) -> None:
        """Add one window of label positions.

        ``nll[i]`` is -log p(label_ids[i]); ``types[i]`` is the type of the *label* token.
        Arrays must be the same length (``window - 1`` for a full window).
        """
        nll = np.asarray(nll, dtype=np.float64)
        label_ids = np.asarray(label_ids, dtype=np.int64)
        types = np.asarray(types, dtype=np.int64)
        if not (nll.shape == label_ids.shape == types.shape):
            raise ValueError(
                f"window arrays disagree: nll {nll.shape}, labels {label_ids.shape}, types {types.shape}"
            )
        valid = types != TYPE_IGNORE
        if not np.any(valid):
            return
        bytes_all = token_bytes[label_ids] if label_ids.size else np.zeros(0, dtype=np.int64)
        self._add_to("all", float(nll[valid].sum()), int(valid.sum()),
                     int(bytes_all[valid].sum()))
        for t in np.unique(types[valid]):
            t = int(t)
            if t == TYPE_IGNORE:
                continue
            name = TYPE_NAMES.get(t)
            if name is None:
                raise ValueError(f"unknown token type {t}")
            mask = types == t
            self._add_to(name, float(nll[mask].sum()), int(mask.sum()),
                         int(bytes_all[mask].sum()))

    def _add_to(self, name: str, nll: float, tokens: int, nbytes: int) -> None:
        c = self.counts[name]
        c["nll_nats"] += nll
        c["tokens"] += tokens
        c["bytes"] += nbytes
        self.records.setdefault(name, []).append((nll, tokens, nbytes))

    # ---------------------------------------------------------------- reporting
    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for name, counts in self.counts.items():
            if name != "all" and not counts["tokens"]:
                continue
            out[name] = finalize_counts(counts)
        return out

    def bootstrap_ci(self, *, n_resamples: int = 1000, level: float = 0.95,
                     seed: int = 0) -> dict[str, list[float]]:
        """Percentile CI of BPB by resampling windows (rough: windows are correlated)."""
        if n_resamples <= 0:
            return {}
        rng = np.random.default_rng(seed)
        alpha = (1.0 - level) / 2.0
        out: dict[str, list[float]] = {}
        for name, records in self.records.items():
            if len(records) < 10:
                continue
            arr = np.asarray(records, dtype=np.float64)  # [N, 3]
            n = arr.shape[0]
            idx = rng.integers(0, n, size=(n_resamples, n))
            nll = arr[idx, 0].sum(axis=1)
            nbytes = arr[idx, 2].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                stats = nll / LN2 / nbytes
            stats = stats[np.isfinite(stats)]
            if stats.size == 0:
                continue
            out[name] = [
                float(np.quantile(stats, alpha)),
                float(np.quantile(stats, 1.0 - alpha)),
            ]
        return out


def bpb_from_counts(nll_nats: float, n_bytes: int) -> float:
    return (nll_nats / LN2 / n_bytes) if n_bytes else float("nan")
