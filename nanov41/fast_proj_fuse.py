"""Projection-fusion patches for the d12 training path (R56, experimental, env-gated).

Two independent fusions that reduce small-GEMM count without changing the math:

1. `NANOV41_ATTN_PROJ_FUSED=1` — `Attention.forward`: `wq_a(x)` and `wkv(x)` share the same
   input x, so concatenate their weights (`[q_lora_rank + head_dim, dim]`) and run **one** GEMM
   instead of two. Autograd's cat-backward then also fuses the two dX GEMMs and the two dW GEMMs
   into one each, i.e. 6 small GEMMs/layer -> 3 + one small cat.
   Requires `nanov41.fused_rope_attn` to be installed first (it patches Attention.forward with
   the R49 rope kernel); this module replaces that forward with a fused-projection variant that
   keeps the same rope calls.

2. `NANOV41_SHARED_EXPERT_FUSED=1` — the shared expert's `w1(x)`/`w3(x)` also share x: fuse them
   into one GEMM (SwiGLU still uses `fused_swiglu.swiglu_fused` when available, else torch).
   Only the `shared_experts` instance is patched (routed experts go through the grouped kernels).

3. `NANOV41_PROJ_FUSE_NATIVE=1` — instead of `torch.cat`-ing the weights at every forward, replace
   the two parameters with one native Parameter (`wqkv`, `w13`) and slice it in forward. Removes
   24 cat kernels/step and their autograd nodes; same math as (1)+(2). Also non-bitwise vs the
   two-GEMM path.

Measured on RTX 5090 (b=16, R50-best graph, 2026-10-30): the cat-based fusions give
59.38 -> 59.00 ms (-0.39 ms / -0.66%); native Parameters remove another 24 cat kernels and
give 61.46 -> 60.92 ms in a second same-session sweep (-0.54 ms / -0.88% vs baseline, and
0.07 ms better than the cat variant). Step-1 loss differs from the default two-GEMM path by
7.1e-5 relative (cuBLAS tiling changes fp32 accumulation order) but native == cat bitwise, so
the default stays OFF and this is an opt-in, parity-tolerant path. See
runs/r56_projection_fusion_5090.json and runs/r57_native_param_fusion_5090.json.

Install after the model is built:
    import nanov41.fast_proj_fuse as fpf
    fpf.install(model, M)
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F

_MOD = None
_ORIG_EXPERT_FORWARD = None


# --------------------------------------------------------------------------- attention
def _attn_forward(self, x: torch.Tensor, start_pos: int):
    M = _MOD
    from .fused_rope_attn import rope_tail

    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    qrank = self.q_lora_rank

    # fused (wq_a, wkv): same input x -> one GEMM; cat-backward fuses dX/dW too.
    # native mode stores one [qrank+head_dim, dim] Parameter and slices it (no cat kernel).
    w = self.wqkv if getattr(self, "wqkv", None) is not None else torch.cat(
        [self.wq_a.weight, self.wkv.weight], dim=0)
    y = F.linear(x, w)
    qr = self.q_norm(y[..., :qrank])
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = rope_tail(q, freqs_cis)

    kv, topk_idxs = self._window_kv_fused(x, freqs_cis, start_pos, y[..., qrank:])
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
        kv = torch.cat([kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

    o = M.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    o = rope_tail(o, freqs_cis, True)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return self.wo_b(o.flatten(2))


def _window_kv_fused(self, x, freqs_cis, start_pos, kv_pre: torch.Tensor):
    """`fused_rope_attn._window_kv` but taking the pre-norm K/V (already projected)."""
    M = _MOD
    from .fused_rope_attn import rope_tail

    bsz, seqlen, _ = x.size()
    win = self.window_size
    kv = self.kv_norm(kv_pre)
    kv = rope_tail(kv, freqs_cis)
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
        if seqlen <= win:
            self.window_kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1
            )
        window_kv = kv
    else:  # decode: one token into the ring buffer, attend over the whole window
        self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        window_kv = self.window_kv_cache[:bsz]
    return window_kv, M.get_window_topk_idxs(win, bsz, seqlen, start_pos)


# --------------------------------------------------------------------------- shared expert
def _shared_forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
    if weights is not None or not x.is_cuda or _ORIG_EXPERT_FORWARD is None:
        return _ORIG_EXPERT_FORWARD(self, x, weights)
    limit = float(getattr(self, "swiglu_limit", 0.0) or 0.0)
    if getattr(self, "w13", None) is not None:
        w = self.w13
    else:
        w = torch.cat([self.w1.weight, self.w3.weight], dim=0)
    y = F.linear(x, w)
    gate, up = y.chunk(2, dim=-1)
    try:
        from .fused_swiglu import swiglu_fused

        h = swiglu_fused(gate, up, limit)
    except Exception:  # noqa: BLE001 - fall back to the identical torch math
        gate = gate.float()
        up = up.float()
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        h = F.silu(gate) * up
    return self.w2(h.type_as(self.w2.weight))


# --------------------------------------------------------------------------- install
def _make_native(model, M) -> dict:
    """Replace the paired parameters with one native Parameter (runs after model build)."""
    import torch.nn as nn

    n_attn = 0
    for mod in model.modules():
        if isinstance(mod, M.Attention) and getattr(mod, "wqkv", None) is None:
            w = torch.cat([mod.wq_a.weight.detach(), mod.wkv.weight.detach()], dim=0).contiguous()
            mod.wqkv = nn.Parameter(w.clone())
            mod.wq_a = None
            mod.wkv = None
            n_attn += 1
    n_exp = 0
    for mod in model.modules():
        if hasattr(mod, "shared_experts") and getattr(mod.shared_experts, "w13", None) is None:
            e = mod.shared_experts
            w = torch.cat([e.w1.weight.detach(), e.w3.weight.detach()], dim=0).contiguous()
            e.w13 = nn.Parameter(w.clone())
            e.w1 = None
            e.w3 = None
            n_exp += 1
    return {"native_attn": n_attn, "native_shared": n_exp}


def install(model, M) -> dict:
    """Patch the built model in place; returns which fusions were enabled."""
    global _MOD, _ORIG_EXPERT_FORWARD
    _MOD = M
    enabled = {}
    native = os.environ.get("NANOV41_PROJ_FUSE_NATIVE", "0").lower() not in ("0", "false", "no", "off")

    if os.environ.get("NANOV41_ATTN_PROJ_FUSED", "0").lower() not in ("0", "false", "no", "off"):
        M.Attention.forward = _attn_forward
        M.Attention._window_kv_fused = _window_kv_fused
        enabled["attn_proj"] = True

    if os.environ.get("NANOV41_SHARED_EXPERT_FUSED", "0").lower() not in ("0", "false", "no", "off"):
        # remember the currently installed Expert.forward (fused_swiglu's) for the routed fallback
        _ORIG_EXPERT_FORWARD = M.Expert.forward
        n = 0
        for mod in model.modules():
            if hasattr(mod, "shared_experts"):
                mod.shared_experts.forward = types.MethodType(_shared_forward, mod.shared_experts)
                n += 1
        enabled["shared_expert"] = n
    if native:
        # native mode needs both fusions installed first (forward already handles the params)
        M.Attention.forward = _attn_forward
        M.Attention._window_kv_fused = _window_kv_fused
        _ORIG_EXPERT_FORWARD = M.Expert.forward
        for mod in model.modules():
            if hasattr(mod, "shared_experts"):
                mod.shared_experts.forward = types.MethodType(_shared_forward, mod.shared_experts)
        enabled.update(_make_native(model, M))
        enabled["native"] = True
    return enabled
