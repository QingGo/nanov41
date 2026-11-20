"""fused_mhc_v4 — mHC as a **single pass over the residual stream** (R63).

## Why (measured problem)

R55 census (b=16, seq=1024, d12): mHC costs **9.48 ms/step** of a 57.17 ms device budget.
Splitting it by kernel shows it is *bandwidth bound*, not compute bound:

    _pm_pre_fwd  2.59   _pm_pre_bwd  2.98   _post_fwd 0.71   _post_bwd 1.28
    _mixes_pre_fwd 0.40 _mixes_pre_bwd 0.91 (sinkhorn fwd 0.16 / bwd 0.45)

`fused_mhc_v3` fuses *inside* one mHC cell, but the cell **boundary at the end of a block is
still three separate ops**: `_post(h2, y)` materialises the next residual stream, then the
next block's `_mixes_pre(x)` reads it again, and their two backward kernels each write a
full `[T, hc*d]` gradient that autograd then sums with a third full-size kernel.

## Measured outcome (R63, 5090, isolated 12-layer chain at T=16384, graph replay)

    variant                       fwd+bwd ms   vs v3f
    v3f (today)                      12.56       —
    split (this module, default)     11.15     -1.71   <- shipped
    cross-layer fused chain          15.37     +2.50   <- negative, kept for the record
    cross-layer + 1x-res fwd         22.68     +9.82   <- negative (PMPF_V2 kernel is 2.3x slower)

The decisive finding: the v3 `_pm_pre` (post+mixes+pre in ONE kernel) is **~2x slower per
call than `_post_fwd` + `_mixes_pre_fwd` together** (204-414 us across runs vs 48+33 us at
T=16384; the profiler swings up to 2x on a shared host, the CUDA-event totals do not), because the fused kernel has to build `y` four times (once per hc copy, with the
residual re-read each time) and then run four M=16/N=32 `tl.dot`s, while the split pair runs
one dot chain over the full K and a pure elementwise post.  Fusing also *looked* cheaper
because it hides one autograd grad-accumulation; unfusing costs +1 add of [T, 4, 384]
(~62 us) and saves ~205 us of kernel time.

## mode="split" (default, ships) — one-line change to `Block.forward`

    mixes, li       = mixes_pre(x, hc_attn_fn, pre_mix)     # unchanged
    y               = hc_post(attn_out, x, post, comb)      # was: fused post+mixes+pre
    mixes2, li2     = mixes_pre(y, hc_ffn_fn, attn_pre)     #   in _pm_pre
    y2              = hc_post(ffn_out, y, post2, comb2)     # unchanged

Same math, same kernels that already exist, no new CUDA code, no `Transformer.forward`
rewrite, no guards (works for engram/MTP/vision too).  Single-step parity vs v3f: loss
rel 7.1e-7, grad 5.8e-4 (worst-parameter max|d|/max|g|) — fp32 re-association only; the
forward is bitwise identical at model shapes, and CPU-vs-reference is bitwise 0.0.

## mode="cross" (opt-in, negative result) — what the earlier design did

The reference block chain is

    block L:  ... ffn -> hc_post(h2_L, y_L) = y2_L   # -> next block's residual
    block L+1: hc_mixes(y2_L); hc_pre(y2_L, pre_mix)  # -> attn input

so `hc_post` of block L and `hc_mixes+hc_pre` of block L+1 are **adjacent in the graph and
share the same tensor**. v4 drives the block loop so that this pair becomes *one* call of
the already-proven `_PostMixesPre` op (`_pm_pre_fwd_v2_kernel` / `_pm_pre_bwd_kernel`).

Per layer that removes
  * fwd: one full read of the residual stream (50.3 MB at T=16384) + one kernel,
  * bwd: one full write of a `[T, hc*d]` gradient + **the autograd grad accumulation**
    (read 2x + write once, 151 MB) because after the refactor every residual tensor has
    exactly one consumer,
  * and it makes the fused fwd read the residual **once** instead of 4x (`PMPF_V2`: the v3
    default kernel re-reads RES per `j`).

Per-layer traffic (T=16384, MB, fwd+bwd): 1234.7 -> 730.0, i.e. **-41%**.

No numerics change: the fused op computes `y` in fp32, rounds to bf16 exactly like the
stored tensor, and `li = pmix . y_bf16` from that same rounded value, so v3f and v4 agree
bit-for-bit (verified by `tools/r63_mhc_attn.py --mode mhc`).

## Scope / guards

`install(M)` patches `M.Transformer.forward` (the block loop) and leaves every other path
alone. It refuses (falls back to the installed baseline forward) when the model uses
anything the fused driver does not reproduce: engram, MTP/dspark target layers, vision
images. Opt-in only: `NANOV41_MHC_V4=1`.

Env knobs
    NANOV41_MHC_V4=1            install (default 0 -> nothing changes)
    NANOV41_V4_MODE=split       `split` (default, ships) or `cross` (neg result, needs the
                                Transformer.forward rewrite; guarded fallback for engram/
                                MTP/vision models)
    NANOV41_V4_SINGLE_READ=0    mode=cross only: 1x-residual-read fused fwd (measured 2.3x
                                slower than the 4x-read kernel on 5090 -> keep 0)
    NANOV41_V4_PMPF_BK/BMPF_S   mode=cross only: tiles of the fused fwd kernel
"""
from __future__ import annotations

import os

import torch

from . import fused_mhc_v3 as v3

__all__ = ["install", "uninstall", "available", "set_single_read_fwd", "get_stats"]

_ORIG_FORWARD = None      # Transformer.forward captured at install time
_CLS = None
_STATE = {"installed": False, "single_read": True, "forward_patched": False,
          "calls": 0, "fallbacks": 0}


def _flag(name, default="0"):
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


def set_single_read_fwd(enabled: bool = True) -> bool:
    """Fused fwd reads the residual once (v3 `PMPF_V2`) instead of once per hc copy."""
    _STATE["single_read"] = bool(enabled)
    v3.PMPF_V2 = bool(enabled)
    return _STATE["single_read"]


def _apply_tiles() -> None:
    v3.PMPF_BK2 = int(os.environ.get("NANOV41_V4_PMPF_BK", v3.PMPF_BK2))
    v3.PMPF_BM = int(os.environ.get("NANOV41_V4_PMPF_BM", v3.PMPF_BM))
    v3.PMPF_S = int(os.environ.get("NANOV41_V4_PMPF_S", v3.PMPF_S))


# --------------------------------------------------------------------------- the driver
def _fused_blocks(self, h, start_pos, pre_mix, image_mask):
    """Block loop with the block-boundary mHC pair fused into one `_PostMixesPre`.

    `h` is the residual stream `[b, s, hc, d]` entering layer 0; returns the residual
    stream leaving the last layer (what `Block.forward` would have returned) **and** the
    last layer's `ffn_pre`, which the model's final `hc_pre` collapse needs.
    """
    layers = self.layers
    pend = None                                  # (x1, res, post, comb) pending hc_post
    res = h
    for i, layer in enumerate(layers):
        fn_attn = layer.hc_attn_fn
        if pend is None:
            mixes, li = v3.mhc_mixes_pre(res, fn_attn, pre_mix, layer.norm_eps)
        else:
            x1p, resp, postp, combp = pend
            res, mixes, li = v3.mhc_post_mixes_pre(
                x1p, resp, postp, combp, pre_mix, fn_attn, layer.norm_eps)
        attn_pre, attn_post, attn_comb = v3.mhc_split_sinkhorn(
            mixes, layer.hc_attn_scale, layer.hc_attn_base,
            layer.hc_sinkhorn_iters, layer.hc_eps)
        x = layer.attn_norm(li)
        x = layer.attn(x, start_pos)
        y, mixes2, li2 = v3.mhc_post_mixes_pre(
            x, res, attn_post, attn_comb, attn_pre, layer.hc_ffn_fn, layer.norm_eps)
        ffn_pre, ffn_post, ffn_comb = v3.mhc_split_sinkhorn(
            mixes2, layer.hc_ffn_scale, layer.hc_ffn_base,
            layer.hc_sinkhorn_iters, layer.hc_eps)
        x = layer.ffn_norm(li2)
        x = layer.ffn(x, image_mask)
        pend = (x, y, ffn_post, ffn_comb)        # hc_post deferred to the next layer
        pre_mix = ffn_pre
    x1p, resp, postp, combp = pend
    return v3.mhc_post(x1p, resp, postp, combp), pre_mix


def _block_forward_split(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """`Block.forward` with the mid-layer mHC cell split back into post + mixes_pre."""
    mixes, li = v3.mhc_mixes_pre(x, self.hc_attn_fn, pre_mix, self.norm_eps)
    attn_pre, attn_post, attn_comb = v3.mhc_split_sinkhorn(
        mixes, self.hc_attn_scale, self.hc_attn_base,
        self.hc_sinkhorn_iters, self.hc_eps)
    h = self.attn_norm(li)
    h = self.attn(h, start_pos, *attn_args)
    y = v3.mhc_post(h, x, attn_post, attn_comb)
    mixes2, li2 = v3.mhc_mixes_pre(y, self.hc_ffn_fn, attn_pre, self.norm_eps)
    ffn_pre, ffn_post, ffn_comb = v3.mhc_split_sinkhorn(
        mixes2, self.hc_ffn_scale, self.hc_ffn_base,
        self.hc_sinkhorn_iters, self.hc_eps)
    h2 = self.ffn_norm(li2)
    h2 = self.ffn(h2, image_mask)
    y2 = v3.mhc_post(h2, y, ffn_post, ffn_comb)
    return y2, ffn_pre


def _make_forward(M):
    def forward(self, input_ids, start_pos=0, images=None, token_types=None,
                full_logits=False):
        # ---- guards: reproduce the reference loop only for the plain text path ----
        if (images is not None or getattr(self, "engram_hash", None) is not None
                or getattr(self, "target_layer_ids", None) or getattr(self, "vision", None)):
            _STATE["fallbacks"] += 1
            return _ORIG_FORWARD(self, input_ids, start_pos, images, token_types,
                                 full_logits)
        _STATE["calls"] += 1
        image_mask = None if token_types is None else token_types >= 0
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, self.hc_mult)
        h, pre_mix = _fused_blocks(self, h, start_pos, pre_mix, image_mask)
        h = self.layers[-1].hc_pre(h, pre_mix)
        logits = self.head(self.norm(h), full_logits=full_logits)
        if full_logits:
            return logits
        output_ids = M.sample(logits, self.temperature)
        return output_ids, logits, None

    return forward


# --------------------------------------------------------------------------- install
def install(M=None, model=None) -> dict:
    """Install the fused block chain.  Idempotent. Opt-in (`NANOV41_MHC_V4=1`)."""
    global _ORIG_FORWARD, _CLS
    if M is None:
        from .patched_env import import_patched
        M = import_patched()
    if not available():
        return {"installed": False, "reason": "triton/cuda unavailable"}
    v3.install(M)                                  # hc_* ops (idempotent)
    _apply_tiles()
    mode = os.environ.get("NANOV41_V4_MODE", "split").lower()
    # K1: the production shell scripts export NANOV41_V4_MODE=split unconditionally, so v5 also
    # has its own gate: NANOV41_MHC_V5=1 (or =block for the mid-block-only variant).
    _v5sel = os.environ.get("NANOV41_MHC_V5", "0").lower()
    if _v5sel not in ("0", "false", "no", "off"):
        mode = "v5block" if _v5sel in ("block", "v5block") else "v5"
    # R72/K1: `v5` = single-pass mHC cell (one kernel fwd, one kernel bwd, no boundary grad add).
    # Delegated here so the existing harness hook (`NANOV41_MHC_V4=1 NANOV41_V4_MODE=v5`) works
    # without touching tools/bench_graph_train.py.  See nanov41/fused_mhc_v5.py.
    if mode in ("v5", "v5block", "single", "single_pass"):
        from . import fused_mhc_v5 as v5
        out = v5.install(M, driver=(mode != "v5block"))
        print(f"[mhc-v5] installed mode={mode} fwd_single={v5.FWD_SINGLE} "
              f"sinkhorn={v5.FUSE_SINKHORN} recompute={v5.BWD_RECOMPUTE} "
              f"driver={out.get('driver')}", flush=True)
        _STATE.update({"mode": mode, "installed": True, "forward_patched": True,
                       "v5": out})
        return dict(_STATE)
    # PMPF_V2 (1x-residual-read fused fwd) is 2.3x SLOWER on 5090 (R63); only mode=cross
    # can even reach it, so keep it off unless explicitly asked.
    set_single_read_fwd(_flag("NANOV41_V4_SINGLE_READ", "0"))
    cls = getattr(M, "Block", None)
    if cls is None:
        raise TypeError("fused_mhc_v4.install() expects the patched model module")
    if mode == "cross":
        tcls = getattr(M, "Transformer", None)
        if tcls is None:
            raise TypeError("mode=cross needs M.Transformer")
        if _STATE["forward_patched"] and _CLS is tcls:
            return dict(_STATE)
        _CLS = tcls
        _ORIG_FORWARD = tcls.forward
        tcls.forward = _make_forward(M)
        _STATE["mode"] = "cross"
    else:
        _CLS = cls
        _ORIG_FORWARD = cls.__dict__.get("forward")
        cls.forward = _block_forward_split
        _STATE["mode"] = "split"
    _STATE["forward_patched"] = True
    _STATE["installed"] = True
    return dict(_STATE)


def uninstall(M=None) -> None:
    global _ORIG_FORWARD
    if _CLS is not None and _ORIG_FORWARD is not None:
        _CLS.forward = _ORIG_FORWARD
    _ORIG_FORWARD = None
    _STATE["forward_patched"] = False
    _STATE["installed"] = False
    _ORIG_FORWARD = None


def available() -> bool:
    return v3.available()


def get_stats() -> dict:
    return dict(_STATE)
