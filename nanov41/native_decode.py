# SPDX-License-Identifier: MIT
"""One-call enabler for the native decode path (Engram FP8 kernel + small-M MoE)."""
from __future__ import annotations

from .engram_decode import enable_engram_decode
from .moe_decode import enable_moe_decode


def enable_native_decode(model, *, engram: bool = True, moe: bool = True,
                         moe_max_tokens: int = 8, mhc: bool = True, attn: bool = False,
                         dtype=None) -> dict:
    """Patch an inference copy of NanoV41 with the native decode paths.

    Use after loading weights and calling ``model.eval()``.  The model's forward/generate path is
    then: reference attention/compressor + FP8 Engram gather/gate + small-M batched MoE + fused
    mHC.  Returns the number of patched Engram/MoE modules.

    Fused mHC is on by default since the real-checkpoint validation
    (``runs/p2b/real_decode_benchmark.json``): 100 % token agreement with the torch mHC path over
    160 greedy tokens and T0 held-out BPB 1.922632 vs 1.922638 (-6e-6).  Pass ``mhc=False`` for
    a numerically-pristine reference path.

    ``attn=True`` installs the fused decode attention fast path
    (``runs/p2b/attn_decode_verify.json``: 100 % argmax agreement over 24 decode steps, 1.20x
    over the native stack end-to-end).  It is opt-in until longer-context validation lands.
    """
    out = {"engram": 0, "moe": 0, "mhc": 0, "attn": 0}
    if engram:
        out["engram"] = enable_engram_decode(model)
    if moe:
        import torch
        out["moe"] = enable_moe_decode(model, max_tokens=moe_max_tokens,
                                       dtype=dtype or torch.bfloat16)
    if mhc:
        from .mhc_decode import enable_mhc_decode
        stats = enable_mhc_decode(model)
        out["mhc"] = len(stats)
    if attn:
        from .attn_decode import enable_attn_decode
        out["attn"] = enable_attn_decode(model)
    return out


__all__ = ["enable_native_decode"]
