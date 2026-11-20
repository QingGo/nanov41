# SPDX-License-Identifier: MIT
"""Decode mHC path: use the R69 fused mHC kernels (mixes/pre/post) for small token counts.

The reference Block does hc_mixes -> hc_pre -> hc_post as separate elementwise/GEMM steps.
`fused_mhc_v3` already fuses them for training; it also works on a decode token (T=1) and cuts
the per-token launch/read traffic substantially.  There is no separate small-M kernel yet; this
wrapper is the opt-in bridge until a T<=8-specific kernel lands.

Parity, real checkpoint (P2b/E-default_s1235, RTX 4090, bf16, 5 prompts x 32 greedy tokens):
**100 % token agreement** with the torch mHC path; next-token logits differ by mean 0.0126
(max 1.08 over 40 prefix-aligned steps).  T0 held-out BPB is 1.922632 vs 1.922638 (-6e-6), i.e.
no measurable quality cost, so fused mHC is now the default in ``enable_native_decode``.  Fused
v4 (the training block chain, ``NANOV41_MHC_V4=1``) is bit-identical to v3 at decode (logit
diff 0.0) and marginally faster in timing noise, so the v3 kernels are kept.
Raw: ``runs/p2b/real_decode_benchmark.json``, ``runs/p2b/real_decode_mhc_extra.json``.

There is still no T<=8-specific mHC kernel: the fused v3 kernels are training-tuned (5 launches
per block) and mHC+norm+residual is ~30 % of a native decode step, so a decode kernel is the
next mHC item (after the launch-bound profile in ``runs/p2b/real_decode_profile.json``).
"""
from __future__ import annotations


def enable_mhc_decode(model=None) -> dict:
    """Install the R69 fused mHC kernels (v3; v4 is bit-identical at decode)."""
    from . import fused_mhc_v3
    if not fused_mhc_v3.available():
        raise RuntimeError("fused_mhc_v3 not available (needs CUDA+Triton)")
    return fused_mhc_v3.install()


__all__ = ["enable_mhc_decode"]
