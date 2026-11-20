"""Reference data mixes (Marin-8B) and mix-weighted BPB.

Our training mix is **not frozen yet**; the mix numbers here are *reference summaries*, not
objectives.  They answer "how would this model look on Marin-8B's token distribution?" while the
primary metrics stay the per-domain BPB values.  The only reason to use Marin is that it is a
fully open model that publishes both its pretraining mixture and its cooldown mixture:

https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md

* **Phase 1/2 (Kestrel/Ocelot, 0 → 3.78 T tokens)** — "DCLM 7B mix", ratios roughly
  proportional to token count: DCLM-Baseline 92.6 %, StarCoder Data 6.1 %, ProofPile-2 1.3 %.
* **Phase 3 (Jellyfish cooldown, 3.78 T → 4.78 T tokens)**: Dolmino DCLM HQ 67.8 %,
  Dolma peS2o 10.8 %, FineMath-3+ 6.3 %, Dolma Arxiv 5.2 %, Dolma StackExchange 3.2 %,
  StarCoder 2.2 %, Dolma Algebraic Stack 2.1 %, Dolma Open Web Math 0.9 %, Dolma Megawika 0.8 %,
  Dolma Wikipedia 0.7 %.

The jellyfish numbers are *aggregated* to the eval-domain granularity our harness uses
(web/academic/math/code/stackexchange/wikipedia); the original per-source table is kept in
``source_weights`` so the aggregation is auditable.  Weights are token proportions, so a mix BPB
is computed as (weighted mean NLL per token) / ln2 / (weighted mean bytes per token), not as a
naive average of per-domain BPB.

Missing domains are reported as ``missing`` and shrink ``coverage``; the remaining weights are
renormalised for the reported number.  Never compare mix BPB across datasets with different
coverage.
"""

from __future__ import annotations

import math
from typing import Mapping

from .bpb import LN2

MARIN_SOURCE = "https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md"

REFERENCE_MIXES: dict[str, dict] = {
    "marin-8b-phase1": {
        "weights": {"web": 0.926, "code": 0.061, "math": 0.013},
        "source_weights": {
            "DCLM Baseline": 0.926,
            "StarCoder Data": 0.061,
            "ProofPile 2": 0.013,
        },
        "source": MARIN_SOURCE,
        "description": (
            "Marin-8B phase 1/2 (Kestrel/Ocelot, 0-3.78T tokens): DCLM 7B mix at ratios roughly "
            "proportional to token count."
        ),
    },
    "marin-8b-jellyfish": {
        "weights": {
            "web": 0.678,          # Dolmino DCLM HQ
            "academic": 0.160,     # Dolma peS2o 10.8 + Dolma Arxiv 5.2
            "math": 0.093,         # FineMath-3+ 6.3 + Algebraic Stack 2.1 + Open Web Math 0.9
            "stackexchange": 0.032,
            "code": 0.022,         # StarCoder
            "wikipedia": 0.015,    # Dolma Megawika 0.8 + Dolma Wikipedia 0.7
        },
        "source_weights": {
            "Dolmino DCLM HQ": 0.678,
            "Dolma peS2o": 0.108,
            "FineMath-3+": 0.063,
            "Dolma Arxiv": 0.052,
            "Dolma StackExchange": 0.032,
            "StarCoder": 0.022,
            "Dolma Algebraic Stack": 0.021,
            "Dolma Open Web Math": 0.009,
            "Dolma Megawika": 0.008,
            "Dolma Wikipedia": 0.007,
        },
        "source": MARIN_SOURCE,
        "description": (
            "Marin-8B phase 3 (Jellyfish cooldown, 3.78T-4.78T tokens): 70% high-quality web + "
            "30% Dolmino/FineMath mix, aggregated to eval domains (original table in source_weights)."
        ),
    },
}


def validate_mix(weights: Mapping[str, float], *, tol: float = 1e-3) -> None:
    """Raise ValueError unless weights are positive and sum to 1 (±tol)."""
    if not weights:
        raise ValueError("mix weights must not be empty")
    total = 0.0
    for domain, weight in weights.items():
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"mix domain names must be non-empty strings, got {domain!r}")
        try:
            w = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mix weight for {domain!r} is not a number: {weight!r}") from exc
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"mix weight for {domain!r} must be finite and >= 0, got {w}")
        total += w
    if abs(total - 1.0) > tol:
        raise ValueError(f"mix weights must sum to 1.0 (±{tol}), got {total:.6f}")


def covered_mixes(domain_names) -> dict[str, dict]:
    """Reference mixes whose domains are all present in ``domain_names``."""
    names = set(domain_names)
    out = {}
    for mix_name, mix in REFERENCE_MIXES.items():
        validate_mix(mix["weights"])
        if set(mix["weights"]) <= names:
            out[mix_name] = mix
    return out


def mix_bpb(results: Mapping[str, object], weights: Mapping[str, float], *,
            name: str | None = None) -> dict:
    """Token-mix BPB from per-domain results.

    ``results`` maps domain -> object with ``.metrics["all"]`` containing ``nll_nats``,
    ``tokens`` and ``bytes`` (``DomainResult`` satisfies this).  Returns a report dict with
    ``bpb``, ``coverage`` (sum of weights of present domains), ``missing`` domains and the
    per-domain contributions used in the weighted mean.
    """
    validate_mix(weights)
    missing: list[str] = []
    present: list[tuple[str, float, object]] = []
    for domain, weight in weights.items():
        result = results.get(domain)
        if result is None:
            missing.append(domain)
            continue
        present.append((domain, float(weight), result))
    coverage = sum(w for _, w, _ in present)
    if coverage <= 0:
        return {
            "name": name,
            "bpb": float("nan"),
            "coverage": 0.0,
            "missing": missing,
            "contributions": {},
            "method": "token_mix_weighted_nll_per_byte",
        }
    mix_nll_per_token = 0.0
    mix_bytes_per_token = 0.0
    contributions = {}
    for domain, weight, result in present:
        metrics = result.metrics["all"]  # type: ignore[attr-defined]
        tokens = float(metrics["tokens"])
        if tokens <= 0:
            raise ValueError(f"domain {domain!r} has no scored tokens")
        nll_per_token = float(metrics["nll_nats"]) / tokens
        bytes_per_token = float(metrics["bytes"]) / tokens
        norm = weight / coverage
        mix_nll_per_token += norm * nll_per_token
        mix_bytes_per_token += norm * bytes_per_token
        contributions[domain] = {
            "weight": weight,
            "normalized_weight": norm,
            "bpb": float(metrics["bpb"]),
            "tokens": int(metrics["tokens"]),
            "bytes": int(metrics["bytes"]),
        }
    bpb = mix_nll_per_token / LN2 / mix_bytes_per_token
    return {
        "name": name,
        "bpb": bpb,
        "coverage": coverage,
        "missing": missing,
        "contributions": contributions,
        "method": "token_mix_weighted_nll_per_byte",
    }


__all__ = [
    "MARIN_SOURCE",
    "REFERENCE_MIXES",
    "validate_mix",
    "covered_mixes",
    "mix_bpb",
]
