#!/usr/bin/env python3
"""Fused CE Plan-A (v2): chunked head GEMM + online-softmax CE, **in both directions**.

Why this file exists
-------------------
``nanov41/triton_ce.py`` already ships the two Plan-A kernels:
  * ``ce_fwd``  — reads bf16 logits once, computes the per-row online max/sumexp (fp32),
                  returns ``(loss_sum, lse)`` **without materialising log_softmax**;
  * ``ce_bwd``  — reads bf16 logits + the saved ``lse`` and writes
                  ``dlogits = (softmax - onehot) * scale`` **without materialising a softmax tensor**.

``train_utils._FusedHeadCE`` uses ``ce_fwd`` in the forward pass, but its **backward still calls
``torch.softmax(lc, dim=-1)`` + ``scatter_``** (a full ``[chunk, V]`` bf16 softmax buffer + gather/
scatter traffic).  This module completes Plan A for the bf16-head path:

  forward : chunked ``F.linear`` -> ``triton_ce.ce_fwd`` (loss + lse, no softmax buffer)
  backward: chunked ``F.linear`` (recompute, as today) -> ``triton_ce.ce_bwd``
            (dlogits from lse, no softmax buffer, no scatter) -> ``dX`` / ``dW`` matmuls

Accounting: fwd saves one ``[chunk,V]`` softmax write; bwd saves one softmax materialisation and
the scatter/gather pass.  Expected win from the 2026-11-10 calibration: **~1–2 ms/replay** on top
of the existing ``ce_fwd`` path (the full Plan-A ceiling is ~3–4.5 ms vs the pre-triton baseline,
see ``runs/local-notes/fused_ce_design.md``).

Activation (default path untouched)
-----------------------------------
``NANOV41_CE_TRITON_V2=1`` + ``install()`` (or ``maybe_install()``) monkeypatches
``nanov41.train_utils.fused_head_ce``; the wrapper only takes over when
``trition_ce.available()`` (CUDA + triton) and ``z_loss == 0`` — every other call falls through to
the original implementation unchanged.  The production integration is a 3-line dispatch inside
``train_utils.fused_head_ce`` (see ``runs/local-notes/fused_ce_impl.md``); this module exists so
the kernel path can be validated on CPU/GPU without editing the training file.

CUDA-graph safety: all loops are Python ``range`` with fixed sizes, every kernel launch has a
static grid, outputs are allocated by the caller before the kernel, and there is no host sync
inside forward/backward (``count`` is a device-side fp32 scalar, exactly like ``_FusedHeadCE``).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from . import triton_ce as _triton_ce

ENV = "NANOV41_CE_TRITON_V2"
DEFAULT_CHUNK_ENV = "NANOV41_CE_CHUNK"

_ORIG = {}
_INSTALLED = False


def available() -> bool:
    """True when the Triton CE kernels can run on this machine (CUDA + triton)."""
    try:
        return bool(_triton_ce.available())
    except Exception:  # noqa: BLE001
        return False


def enabled() -> bool:
    """Env gate only (default off -> the training path is literally unchanged)."""
    return os.environ.get(ENV, "0").strip().lower() not in ("0", "false", "no", "off", "")


def _default_chunk(n: int) -> int:
    return int(os.environ.get(DEFAULT_CHUNK_ENV, "2048"))


# =========================================================================== shift + reference
def shift_targets(ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """``[b,s]`` token ids -> flat ``[b*s]`` next-token targets (last position of every row ignored)."""
    b, s = ids.shape
    target = torch.full((b * s,), ignore_index, dtype=torch.long, device=ids.device)
    target.view(b, s)[:, :-1] = ids[:, 1:]
    return target


def reference_online_ce(logits: torch.Tensor, target: torch.Tensor,
                        ignore_index: int = -100) -> dict:
    """Pure-torch reference of the Plan-A algorithm (CPU-testable; mirrors the Triton kernels).

    Forward (online softmax, one pass over V):
        m = -inf, s = 0
        for V in blocks:  new_m = max(m, block_max); s = s*exp(m-new_m) + sum(exp(block-new_m)); m = new_m
        lse = m + log(s);  loss_row = lse - logit[target]  (0 for ignored rows)

    Backward (recompute p from lse, no softmax buffer):
        p = exp(logit - lse);  dlogits = (p - onehot[target]) * scale  (0 for ignored rows)

    Returns ``{"loss_sum", "lse", "target", "count", "dlogits" (unscaled p-onehot)}``.
    """
    n, V = logits.shape
    logits_f = logits.float()
    target = target.reshape(-1)
    valid = target != ignore_index
    tgt_safe = target.clamp_min(0)

    m = torch.full((n,), -float("inf"), device=logits.device, dtype=torch.float32)
    s = torch.zeros(n, device=logits.device, dtype=torch.float32)
    block = 4096
    for off in range(0, V, block):
        l = logits_f[:, off:off + block]
        blk_max = l.max(dim=-1).values
        new_m = torch.maximum(m, blk_max)
        s = s * torch.exp(m - new_m) + torch.exp(l - new_m.unsqueeze(-1)).sum(dim=-1)
        m = new_m
    lse = m + torch.log(s)

    tgt_logit = logits_f.gather(1, tgt_safe.unsqueeze(-1)).squeeze(-1)
    loss_rows = torch.where(valid, lse - tgt_logit, torch.zeros_like(lse))
    d = torch.exp(logits_f - lse.unsqueeze(-1))
    d.scatter_(1, tgt_safe.unsqueeze(-1), d.gather(1, tgt_safe.unsqueeze(-1)) - 1.0)
    d = torch.where(valid.unsqueeze(-1), d, torch.zeros_like(d))
    return {"loss_sum": loss_rows.sum(), "lse": lse, "target": target,
            "count": valid.sum(), "dlogits": d}


# =========================================================================== autograd Function
class _FusedCEv2(torch.autograd.Function):
    """head GEMM (chunked, bf16) + Plan-A CE in fwd and bwd. Mirrors ``_FusedHeadCE`` semantics."""

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk: int, ignore_index: int = -100):
        b, s, D = hidden.shape
        V = weight.shape[0]
        n = b * s
        ctx_aligned = chunk % s == 0          # chunk multiple of seq -> static strided masks
        hf = hidden.reshape(n, D)
        target = shift_targets(ids, ignore_index)
        wb = weight.to(torch.bfloat16)
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        count = torch.zeros((), dtype=torch.float32, device=hidden.device)
        use_triton = available() and hidden.is_cuda
        lse_all = torch.empty(n, dtype=torch.float32, device=hidden.device)
        for i in range(0, n, chunk):
            lc = F.linear(hf[i:i + chunk].to(torch.bfloat16), wb)
            tc = target[i:i + chunk]
            if use_triton:
                rows, lse = _triton_ce.ce_fwd(lc, tc, ignore_index)
                total = total + rows
                lse_all[i:i + chunk] = lse
            else:
                total = total + F.cross_entropy(lc.float(), tc, reduction="sum",
                                                ignore_index=ignore_index)
            count = count + (tc != ignore_index).sum()
        loss = total / count
        ctx.save_for_backward(hidden, weight, target, lse_all)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.count = count
        ctx.n = n
        ctx.s = s
        ctx.aligned = ctx_aligned
        ctx.use_triton = use_triton
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, target, lse_all = ctx.saved_tensors
        b, s, D = hidden.shape
        n = b * s
        hf = hidden.reshape(n, D)
        wb = weight.to(torch.bfloat16)
        gh = torch.empty_like(hf)
        gw = torch.zeros_like(weight)                 # fp32 master accumulation
        scale = grad_out.float() / ctx.count
        for i in range(0, n, ctx.chunk):
            tc = target[i:i + ctx.chunk]
            xc = hf[i:i + ctx.chunk].to(torch.bfloat16)
            lc = F.linear(xc, wb)                     # recompute logits (same as _FusedHeadCE)
            if ctx.use_triton:
                d = _triton_ce.ce_bwd(lc, tc, lse_all[i:i + ctx.chunk], scale,
                                      ctx.ignore_index)   # bf16 (p-onehot)*scale, ignore rows 0
            else:
                df = torch.softmax(lc.float(), dim=-1)
                tgt_safe = tc.clamp_min(0).unsqueeze(1)
                df.scatter_(1, tgt_safe, df.gather(1, tgt_safe) - 1.0)
                df = torch.where((tc == ctx.ignore_index).unsqueeze(-1),
                                 torch.zeros_like(df), df) * scale
                d = df.to(wb.dtype)
            gh[i:i + ctx.chunk] = torch.matmul(d, wb)          # d already scaled
            gw += torch.matmul(d.t(), xc).float()
        return gh.view_as(hidden), gw, None, None, None


# =========================================================================== public entry points
def _fp8_head_requested() -> bool:
    """p2/A-B runs may select the fp8-v2 head; that path must keep winning over the bf16 v2 path."""
    return os.environ.get("NANOV41_HEAD_FP8", "0").strip().lower() in ("2", "v2")


def fused_head_ce_v2(hidden: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor,
                     chunk: int | None = None, ignore_index: int = -100,
                     z_loss: float = 0.0) -> torch.Tensor:
    """Same contract as ``train_utils.fused_head_ce`` (hidden [b,s,D], weight [V,D], ids [b,s]).

    ``z_loss != 0`` is not implemented in v2 (falls back to the original path in ``install()``);
    callers that need z-loss should keep using the original function.
    """
    if z_loss:
        raise NotImplementedError("fused_ce_triton v2 does not implement z_loss yet")
    chunk = int(chunk) if chunk is not None else _default_chunk(hidden.shape[0] * hidden.shape[1])
    return _FusedCEv2.apply(hidden, weight, ids, chunk, ignore_index)


def install() -> bool:
    """Monkeypatch ``train_utils.fused_head_ce`` with the env-gated v2 path. Idempotent.

    Only activates when ``NANOV41_CE_TRITON_V2`` is truthy **and** the Triton kernels are usable;
    otherwise nothing changes.  Returns True when the v2 path is active.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if not enabled() or not available():
        return False
    from . import train_utils as _tu
    _ORIG["fused_head_ce"] = _tu.fused_head_ce
    orig = _tu.fused_head_ce

    def _dispatch(hidden, weight, ids, chunk=None, ignore_index=-100, z_loss=0.0):
        if z_loss == 0 and enabled() and available() and not _fp8_head_requested():
            return fused_head_ce_v2(hidden, weight, ids, chunk, ignore_index)
        return orig(hidden, weight, ids, chunk, ignore_index, z_loss)

    _tu.fused_head_ce = _dispatch
    _INSTALLED = True
    print(f"[fused_ce_triton] v2 installed (env {ENV}=1, triton CE available)", flush=True)
    return True


def maybe_install() -> bool:
    """Alias used by launch scripts: install iff the env gate is on."""
    return install()


def uninstall() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    from . import train_utils as _tu
    _tu.fused_head_ce = _ORIG.pop("fused_head_ce")
    _INSTALLED = False


__all__ = ["ENV", "available", "enabled", "shift_targets", "reference_online_ce",
           "fused_head_ce_v2", "install", "maybe_install", "uninstall"]
