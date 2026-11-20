"""Fused head CE with **forward-precomputed dX/dW** (no backward logits recompute).

Context (2026-11-10 attribution): the biggest single kernel family is the head GEMM
``cutlass 128x256`` = 10.23 ms/iter at b1/s8192, of which 97% is (a) the forward head GEMM
(4.96 ms, 218 TFLOP/s ≈ bf16 peak) and (b) the **backward logits recompute** (4.95 ms).  The
recompute exists only because ``_FusedHeadCE`` / ``_FP8HeadCEv2`` do not keep logits (they would
cost 1.06-2.12 GiB at 8k tokens).  This module removes the recompute by doing the gradient GEMMs
while the logits are still in hand during the forward pass:

    forward : per chunk  logits -> CE loss/lse -> dlogits -> (dX chunk, dW accumulation)
              saves dX [n, D] bf16 (~8.4 MiB @8k / 16.8 MiB @16k) and dW [V, D] fp32 (~265 MB)
    backward: only rescales the saved grads by the scalar ``grad_out``

Why this is cheaper than keeping logits: we store **gradients**, not logits — dX is n*D (8.4 MiB)
and dW is V*D (265 MB fp32, or 132 MB bf16 via ``NANOV41_HEAD_FWDGRAD_DW=bf16``), instead of
n*V logits (1.06 GiB fp8 / 2.12 GiB bf16).  Saved compute: one logits GEMM (bf16 path also saves
the separate ``torch.softmax`` materialisation, replaced by the single-pass ``triton_ce.ce_bwd``).

Safety: the backward only rescales, which is exact only for a **scalar** ``grad_out`` (the fused
head CE always returns a mean loss, so autograd hands us a 1-element tensor).  A non-scalar
grad_out raises instead of silently returning a wrong gradient (the dispatcher keeps the old path
available; tested).

Env (default off, behaviour of the default path unchanged):
  NANOV41_HEAD_FWDGRAD=1        enable (picked up by nanov41.train_utils.fused_head_ce)
  NANOV41_HEAD_FWDGRAD_DW=bf16  store dW as bf16 (half the memory, a little grad precision loss)
  NANOV41_HEAD_WCACHE=1         cache the fp8 head-weight quantisation per **optimizer** step
                                (see fp8_head_v2's module docstring): the [V,D] weight is
                                quantised once per step by ``refresh_head_weight_cache()`` -- called
                                by the training loop right after ``bundle.step()``/``opt.step()``,
                                i.e. outside the captured CUDA graph -- instead of once per
                                micro-batch inside the graph replay.  Bitwise identical within a
                                step; requires the caller to invoke the refresh after each step.

The fp8 head (``NANOV41_HEAD_FP8=v2``, WALL main path) also recomputes logits in backward, so it
gets its own fused-gradient variant here and is preferred when that env is set.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from . import triton_ce as _triton_ce

_CE_FWD = None  # late import guard for torch/CPU-only environments


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def enabled() -> bool:
    return _flag("NANOV41_HEAD_FWDGRAD", "0")


def _dw_bf16() -> bool:
    return os.environ.get("NANOV41_HEAD_FWDGRAD_DW", "fp32").strip().lower() in ("bf16", "bfloat16")


def available() -> bool:
    return True


def _shift_targets(ids: torch.Tensor, ignore_index: int) -> torch.Tensor:
    b, s = ids.shape
    target = torch.full((b * s,), ignore_index, dtype=torch.long, device=ids.device)
    target.view(b, s)[:, :-1] = ids[:, 1:]
    return target


def _count_valid(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return (target != ignore_index).sum().to(torch.float32)


def _scalar_scale(grad_out: torch.Tensor) -> torch.Tensor:
    """0-dim fp32 scale; raises for non-scalar grad_out (the rescale-only backward is exact only
    for a mean-loss scalar)."""
    if grad_out.numel() != 1:
        raise RuntimeError(
            "NANOV41_HEAD_FWDGRAD requires a scalar grad_out (mean CE loss); got "
            f"{tuple(grad_out.shape)}. Unset NANOV41_HEAD_FWDGRAD for non-scalar use.")
    return grad_out.reshape(()).float()


def _dlogits_torch(lc: torch.Tensor, tc: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """CPU/reference dlogits: (softmax - onehot), ignore rows zeroed.  The 1/count loss scale is
    applied by the backward (same convention as the triton ``ce_bwd(..., scale=1.0)`` path)."""
    p = torch.softmax(lc.float(), dim=-1)
    idx = tc.clamp_min(0).unsqueeze(1)
    p.scatter_(1, idx, p.gather(1, idx) - 1.0)
    p = p * (tc != ignore_index).unsqueeze(-1).to(p.dtype)   # static mask (graph-safe)
    return p.to(lc.dtype)


def _zloss_setup(ctx, lse):
    """Pre-compute the analytic z-loss factors for the backward correction.

    z-loss:  L_z = z_loss * mean_i(lse_i**2)  ->  dL_z/z_logits_ij = (2*z_loss/n) * lse_i * q_ij.

    Instead of adding that term to the (bf16/fp8) dlogits -- where it is below the dtype ulp and is
    rounded away while still costing a full fp32 [chunk,V] pass (measured: no speed-up vs the old
    fallback) -- the fwdgrad path applies the z-term **after** the gradient GEMMs in fp32:

        gh = d_ce @ W                       (forward-precomputed CE gradient)
        dL_z/dx_i = (1/count) * a_i * q_i @ W = (1/count) * a_i * (gh_i + W[y_i])      [exact]
        dL_z/dW   = (1/count) * sum_i a_i q_i x_i^T
                  = (1/count) * [ abar * gw + sum_i a_i e_{y_i} x_i^T ]                [abar approx]

    with a_i = 2*z_loss*count/n * lse_i and abar = mean_i a_i over the valid rows.  `q_i = d_ce_i +
    e_{y_i}` is what makes the dX term exact; the only approximation is replacing the per-row factor
    of the dW term by its mean (error O(std(a)/mean(a)) of a term that is ~0.2% of the CE gradient,
    i.e. ~1e-4 relative -- far below any parity gate).  Cost: two O(n*D) elementwise/gather ops.
    Ignored rows (last position of each sequence, target=ignore_index) get a_i = 0: their z-term
    contribution is O(1/n) of the total, while reconstructing it would need a real softmax.
    """
    if not ctx.z_loss:
        return
    a = (2.0 * ctx.z_loss * ctx.count / float(ctx.n)) * lse          # [n] fp32 (pre-1/count)
    av = a.view(ctx.b, ctx.s)
    av[:, -1] = 0.0                                                   # static ignore stride
    ctx.z_a = a
    ctx.z_abar = a.sum() / ctx.count.clamp_min(1)


def _zloss_grad_correction(ctx, gh, hidden, weight):
    """Return (a [n] fp32, wy = W[y] [n,D] fp32) for the backward correction (None without z-loss)."""
    if not ctx.z_loss:
        return None, None
    idx = ctx.z_target.clamp_min(0)
    wy = weight.index_select(0, idx).float()
    return ctx.z_a, wy


class _BF16HeadCEFwdGrad(torch.autograd.Function):
    """bf16 head + CE with forward-precomputed grads (DISC / bf16-head path); z-loss supported."""

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk, ignore_index, z_loss):
        b, s, D = hidden.shape
        n = b * s
        hf = hidden.reshape(n, D)
        target = _shift_targets(ids, ignore_index)
        wb = weight.to(torch.bfloat16)
        count = _count_valid(target, ignore_index)
        use_triton = (_triton_ce is not None and hidden.is_cuda
                      and _triton_ce.available())
        gh = torch.empty(n, D, dtype=torch.bfloat16, device=hidden.device)
        gw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        z2 = (torch.zeros((), dtype=torch.float32, device=hidden.device)
              if z_loss else None)
        lse_all = (torch.empty(n, dtype=torch.float32, device=hidden.device)
                   if z_loss else None)
        for i in range(0, n, chunk):
            tc = target[i:i + chunk]
            xc = hf[i:i + chunk].to(torch.bfloat16)
            lc = F.linear(xc, wb)                       # bf16 logits (this is the forward GEMM)
            if use_triton:
                s_rows, lse = _triton_ce.ce_fwd(lc, tc, ignore_index)
                total = total + s_rows
                # single pass softmax-onehot from lse: no [chunk,V] softmax materialisation
                d = _triton_ce.ce_bwd(lc, tc, lse, 1.0, ignore_index)
            else:
                row = F.cross_entropy(lc, tc, reduction="none", ignore_index=ignore_index)
                total = total + row.float().sum()
                d = _dlogits_torch(lc, tc, ignore_index)
                if z_loss:
                    lse = torch.logsumexp(lc.float(), -1)
            if z_loss:
                lse_all[i:i + chunk] = lse
                z2 = z2 + lse.pow(2).sum()              # mean over all n positions (as reference)
            # gradient GEMMs now, while logits are hot (they replace the backward ones)
            gh[i:i + chunk] = torch.matmul(d, wb).to(torch.bfloat16)
            gw += torch.matmul(d.t(), xc).float()
        loss = total / count.clamp_min(1)
        if z_loss:
            loss = loss + z_loss * z2 / float(n)
        ctx.save_for_backward(gh, gw, hidden, weight, target) if z_loss else \
            ctx.save_for_backward(gh, gw, hidden)
        ctx.chunk, ctx.count, ctx.n, ctx.b, ctx.s = chunk, count, n, b, s
        ctx.z_loss = float(z_loss)
        ctx.z_target = target
        if z_loss:
            _zloss_setup(ctx, lse_all)
        ctx.hidden_shape = hidden.shape
        ctx.dw_bf16 = _dw_bf16()
        ctx.dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        scale = _scalar_scale(grad_out) / ctx.count.clamp_min(1)
        if ctx.z_loss:
            gh, gw, hidden, weight, target = ctx.saved_tensors
            a, wy = _zloss_grad_correction(ctx, gh, hidden, weight)
            n, D = gh.shape
            gx = (gh.float() * (1.0 + a).unsqueeze(1) + a.unsqueeze(1) * wy) * scale
            gx = gx.to(ctx.dtype).reshape(ctx.hidden_shape)
            gW = gw.float() * (1.0 + ctx.z_abar) if not ctx.dw_bf16 else \
                gw.to(torch.bfloat16).float() * (1.0 + ctx.z_abar)
            gW = gW * scale
            gW.index_add_(0, ctx.z_target.clamp_min(0),
                          (a.unsqueeze(1) * hidden.reshape(n, D).float()) * scale)
            return gx, gW.to(ctx.weight_dtype), None, None, None, None
        gh, gw, hidden = ctx.saved_tensors
        gx = (gh.float() * scale).to(ctx.dtype).reshape(ctx.hidden_shape)
        if ctx.dw_bf16:
            gw_saved = gw.to(torch.bfloat16).float()
        else:
            gw_saved = gw
        gW = (gw_saved * scale).to(ctx.weight_dtype)   # gw is already [V, D]
        return gx, gW, None, None, None, None


class _FP8HeadCEFwdGrad(torch.autograd.Function):
    """fp8-v2 head + CE with forward-precomputed grads (WALL path)."""

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk, ignore_index, z_loss):
        from . import fp8_head_v2 as _f8

        b, s, D = hidden.shape
        n = b * s
        v = weight.shape[0]
        hc = hidden.reshape(n, D)
        target = _shift_targets(ids, ignore_index)
        dwb = _dw_bf16()
        # ---- head-weight fp8 quantisation: per-micro-batch (default) or cached per opt step
        # (NANOV41_HEAD_WCACHE=1; static buffers refreshed outside the CUDA graph, bitwise equal
        # within a step because the weight only changes when the optimizer steps).
        wcache = _f8.wcache_enabled()
        if wcache:
            if not _f8.weight_cache().ready:
                _f8.refresh_weight_cache(weight)   # first forward (eager warm-up), before capture
            wc = _f8.weight_cache()
            wq_row, sw_f, wq_dx, sw_col = wc.wq_row, wc.sw_f, wc.wq_dx, wc.sw_col
        else:
            wb = weight.to(torch.bfloat16)
            # weight quantisation exactly like _FP8HeadCEv2.forward (fwd and dX/dW reuse both copies)
            wq_row, sw_row = _f8._q_row(wb)
            sw_f = sw_row.reshape(1, -1).contiguous()
            wq_col, sw_col = _f8._q_col(wb)
            wphys = wq_col.t().contiguous()
            wq_dx = wphys.t()
        count = _count_valid(target, ignore_index)
        dev = hidden.device
        gh = torch.empty(n, D, dtype=torch.bfloat16, device=dev)
        # dW^T [D,V]: fp32 accumulation (1.06 GB per micro-batch of add traffic).
        # WCACHE: bf16 accumulation when NANOV41_HEAD_FWDGRAD_DW=bf16 (half the add traffic; the
        # value is rounded to bf16 by the backward anyway) and skip the bf16->fp32 part copy by
        # adding the bf16 GEMM output directly (identical arithmetic, one less [D,V] materialise).
        if wcache:
            gwT = torch.zeros(D, v, dtype=torch.bfloat16 if dwb else torch.float32, device=dev)
        else:
            gwT = torch.zeros(D, v, dtype=torch.float32, device=dev)

        total = torch.zeros((), dtype=torch.float32, device=dev)
        z2 = (torch.zeros((), dtype=torch.float32, device=dev) if z_loss else None)
        lse_all = (torch.empty(n, dtype=torch.float32, device=dev) if z_loss else None)
        hq = hc.to(torch.bfloat16)
        ones = _f8._ones(dev)
        for i in range(0, n, chunk):
            xc = hq[i:i + chunk]
            tc = target[i:i + chunk]
            xq, sx = _f8._q_row(xc)
            lc = torch._scaled_mm(xq, wq_row.t(), scale_a=sx, scale_b=sw_f,
                                  out_dtype=torch.bfloat16)          # forward logits GEMM
            s_rows, lse, amax = _triton_ce.ce_fwd(lc, tc, ignore_index, need_amax=True)
            total = total + s_rows
            if z_loss:
                lse_all[i:i + chunk] = lse
                z2 = z2 + lse.pow(2).sum()
            # single-pass fp8 dlogits (same kernel the old backward used; now in forward)
            dfp8, sd = _triton_ce.ce_bwd_fp8(lc, tc, lse, amax, ignore_index)
            sd = sd.reshape(-1, 1)
            # dX = (d·sd) @ W  (A fp8 rows scaled by sd, B=W with column scale)
            dx = torch._scaled_mm(dfp8, wq_dx, scale_a=sd.contiguous(),
                                  scale_b=sw_col, out_dtype=torch.bfloat16)
            gh[i:i + chunk] = dx
            # dW = d^T @ x = dfp8^T @ (sd ⊙ x)  (TensorWise, same trick as _FP8HeadCEv2.backward)
            xp = (xc.float() * sd).to(torch.bfloat16)
            xq_t, sx_t = _f8._q_per_tensor(xp)
            a = xq_t.t().contiguous()
            part = torch._scaled_mm(a, dfp8, scale_a=sx_t, scale_b=ones,
                                    out_dtype=torch.bfloat16)
            if wcache:
                gwT += part             # bf16 operand: no part.float() [D,V] materialisation
            else:
                gwT += part.float()
        loss = total / count.clamp_min(1)
        if z_loss:
            loss = loss + z_loss * z2 / float(n)
        ctx.save_for_backward(gh, gwT, hidden, weight, target) if z_loss else \
            ctx.save_for_backward(gh, gwT, hidden)
        ctx.count, ctx.n, ctx.b, ctx.s = count, n, b, s
        ctx.z_loss = float(z_loss)
        ctx.z_target = target
        if z_loss:
            _zloss_setup(ctx, lse_all)
        ctx.hidden_shape = hidden.shape
        ctx.dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.dw_bf16 = dwb
        ctx.wcache = wcache
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        scale = _scalar_scale(grad_out) / ctx.count.clamp_min(1)
        if ctx.z_loss:
            gh, gwT, hidden, weight, target = ctx.saved_tensors
            n, D = gh.shape
            a, wy = _zloss_grad_correction(ctx, gh, hidden, weight)
            gx = (gh.float() * (1.0 + a).unsqueeze(1) + a.unsqueeze(1) * wy) * scale
            gx = gx.to(ctx.dtype).reshape(ctx.hidden_shape)
            gw_saved = gwT.to(torch.bfloat16).float() if ctx.dw_bf16 else gwT.float()
            gW = (gw_saved.t() * (1.0 + ctx.z_abar) * scale).contiguous()   # [V, D]
            gW.index_add_(0, ctx.z_target.clamp_min(0),
                          (a.unsqueeze(1) * hidden.reshape(n, D).float()) * scale)
            return gx, gW.to(ctx.weight_dtype), None, None, None, None
        gh, gwT, hidden = ctx.saved_tensors
        gx = (gh.float() * scale).to(ctx.dtype).reshape(ctx.hidden_shape)
        if ctx.dw_bf16 and getattr(ctx, "wcache", False):
            # gwT is already bf16: fuse the up-cast + transpose + rescale in one elementwise pass
            gW = (gwT.t() * scale).to(ctx.weight_dtype)
        else:
            gw_saved = gwT.to(torch.bfloat16).float() if ctx.dw_bf16 else gwT
            gW = (gw_saved.t() * scale).to(ctx.weight_dtype)   # gw is already [V, D]
        return gx, gW, None, None, None, None


def fused_head_ce_fwdgrad(hidden, weight, ids, chunk=2048, ignore_index=-100, z_loss=0.0):
    """Env-gated entry point used by ``train_utils.fused_head_ce``.

    ``z_loss > 0`` is supported by BOTH fast paths: the z-term is applied as an analytic fp32
    correction to the saved dX/dW in the backward (see ``_zloss_setup``), not mixed into the
    bf16/fp8 dlogits where it would be rounded away.  Measured intent: keep the z-loss step-time
    penalty near zero instead of the old +17.7% fallback.
    """
    fp8 = os.environ.get("NANOV41_HEAD_FP8", "").strip().lower()
    if fp8 in ("2", "v2"):
        from . import fp8_head_v2 as _f8
        if _f8.available() and hidden.is_cuda:
            return _FP8HeadCEFwdGrad.apply(hidden, weight, ids, chunk, ignore_index, z_loss)
    return _BF16HeadCEFwdGrad.apply(hidden, weight, ids, chunk, ignore_index, z_loss)


# --------------------------------------------------------------------------------------------
# Per-optimizer-step head-weight quantisation cache (NANOV41_HEAD_WCACHE=1, default OFF)
# --------------------------------------------------------------------------------------------
# The [V, D] fp8 head-weight quantisation is repeated for every micro-batch inside the captured
# CUDA graph.  With NANOV41_HEAD_WCACHE=1 it is cached in static buffers that are refreshed once
# per optimizer step (outside the graph).  The caller must invoke the refresh right after the
# optimizer step:  ``head_fwdgrad.maybe_refresh_head_weight_cache()``.
def wcache_enabled() -> bool:
    from . import fp8_head_v2 as _f8
    return _f8.wcache_enabled()


def refresh_head_weight_cache(weight=None) -> bool:
    """Refill the cached fp8 head-weight buffers from ``weight`` [V, D] (or the last weight seen).

    Call this **after every optimizer step** (outside the captured CUDA graph) when
    ``NANOV41_HEAD_WCACHE=1``.  Returns True when the cache was refreshed.
    """
    from . import fp8_head_v2 as _f8
    if not _f8.wcache_enabled():
        return False
    return _f8.refresh_weight_cache(weight)


def maybe_refresh_head_weight_cache(weight=None) -> bool:
    """Cheap no-op unless ``NANOV41_HEAD_WCACHE=1`` (the one-line training-loop wiring)."""
    if not _flag("NANOV41_HEAD_WCACHE", "0"):
        return False
    from . import fp8_head_v2 as _f8
    if not _f8.weight_cache().ready:   # no fused fp8 head forward yet -> nothing to refresh
        return False
    return _f8.refresh_weight_cache(weight)


def wcache_state() -> dict:
    from . import fp8_head_v2 as _f8
    c = _f8.weight_cache()
    return {"enabled": _f8.wcache_enabled(), "ready": c.ready, "version": c.version,
            "refreshes": c.refreshes,
            "shape": list(c.wq_row.shape) if c.wq_row is not None else None}
