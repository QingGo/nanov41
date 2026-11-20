# SPDX-License-Identifier: MIT
"""Decode-side Engram kernels (incremental hash + FP8/E8M0 row gather + gate).

The training path computes all n-gram hash ids for a whole sequence and gathers a bf16
table.  At decode time the context is one token, but the n-gram needs the previous
``max_ngram_size - 1`` *compressed* token ids.  This module keeps that tiny lookback cache
and provides:

- ``EngramDecodeCache`` / ``decode_hash_ids``: same hash formula as ``NgramHashState`` but
  for the current token(s), with start-of-sequence/dead-token masking;
- ``quantize_table_ue8m0``: bf16 table -> e4m3 rows + one E8M0 byte per 32-wide block,
  matching the official/vLLM Engram layout;
- ``gather_dequant_torch`` + Triton kernel: row gather + dequant of the FP8 table;
- ``engram_gate_decode``: gather -> wkv -> q/k gate math -> residual add (PyTorch gate for
  now; the kernel hot path is hash+gather+dequant).

The kernel is intentionally small (one program per BLOCK_R rows) because decode has 8--64
rows per step, not thousands.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

E4M3_MAX = 448.0
DEAD = -1


class EngramDecodeCache:
    """Per-layer compressed-token lookback cache (small: B x max_seq int64)."""

    def __init__(self, batch: int, max_seq: int, pad_id: int, device="cpu"):
        self.cache = torch.full((batch, max_seq), pad_id, dtype=torch.int64, device=device)
        self.pad_id = int(pad_id)
        self.device = torch.device(device)

    def append(self, compressed: torch.Tensor, start_pos: int) -> None:
        b, l = compressed.shape
        self.cache[:b, start_pos:start_pos + l] = compressed.to(self.cache.device)

    def window(self, start_pos: int, length: int = 1) -> torch.Tensor:
        return self.cache[:, start_pos:start_pos + length]


def decode_hash_ids(cache: torch.Tensor, start_pos: int, compressed: torch.Tensor,
                    multipliers: torch.Tensor, primes: torch.Tensor,
                    offsets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Hash ids for the new token(s) at ``start_pos..start_pos+length-1``.

    Shapes: cache [B, max_seq], compressed [B, L], multipliers [L_layers, max_ngram],
    primes [L_layers, max_ngram-1, n_heads], offsets same as primes.
    Returns [B, L, L_layers, n_hash_cols] (same as NgramHashState).
    """
    b, l = compressed.shape
    cache[:b, start_pos:start_pos + l] = compressed
    positions = torch.arange(start_pos, start_pos + l, device=compressed.device).expand(b, l)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    max_ngram = multipliers.shape[1]
    for shift in range(max_ngram):
        source = cache[:b].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == DEAD)
        tokens.append(torch.where(blocked, pad_id, source))
    tokens = torch.stack(tokens, dim=-1)                     # [B,L,max_ngram]
    products = tokens.unsqueeze(2) * multipliers              # [B,L,Layers,max_ngram]
    rolling, hashes = products[..., 0], []
    for i in range(1, max_ngram):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


def quantize_table_ue8m0(weight: torch.Tensor, block: int = 32):
    """bf16/float table [rows, dim] -> (fp8 e4m3 [rows,dim], uint8 scales [rows,dim//block])."""
    w = weight.float()
    rows, dim = w.shape
    blocks = w.view(rows, dim // block, block)
    max_abs = blocks.abs().amax(dim=-1)
    ratio = (max_abs / E4M3_MAX).clamp_min(2.0 ** -126)
    exponent = torch.ceil(torch.log2(ratio)).to(torch.int32).clamp(-127, 127)
    exponent = torch.where(max_abs == 0, torch.zeros_like(exponent), exponent)
    scale = torch.pow(torch.tensor(2.0, device=w.device), exponent.float()).unsqueeze(-1)
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(rows, dim)
    return q.contiguous(), (exponent + 127).clamp(0, 255).to(torch.uint8).contiguous()


def fp8_scale_to_float(scale_byte: torch.Tensor) -> torch.Tensor:
    return (scale_byte.to(torch.int32) << 23).view(torch.float32)


def gather_dequant_torch(weight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Torch fallback for row gather + E8M0 dequant. ids: [N] (any shape), out [*ids.shape, dim]."""
    flat = ids.reshape(-1)
    if weight.device.type == "cuda":
        vals = weight[flat].float()
    else:
        # CPU torch does not implement index select on float8; dequant via a 256-entry LUT.
        lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        vals = lut[weight.view(torch.uint8)[flat].long()]
    sc = fp8_scale_to_float(scales[flat]).repeat_interleave(weight.shape[1] // scales.shape[1], dim=-1)
    out = (vals * sc).to(torch.bfloat16)
    return out.view(*ids.shape, weight.shape[1])


if HAS_TRITON:
    @triton.jit
    def _gather_dequant_kernel(W, S, IDS, OUT, N, D: tl.constexpr,
                               BLOCK_R: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = rows < N
        ids = tl.load(IDS + rows, mask=mask, other=0).to(tl.int64)
        cols = tl.arange(0, D)
        vals = tl.load(W + ids[:, None] * D + cols[None, :], mask=mask[:, None], other=0.0)
        scale_byte = tl.load(S + ids, mask=mask, other=127).to(tl.int32)
        scale = (scale_byte << 23).to(tl.float32, bitcast=True)
        tl.store(OUT + rows[:, None] * D + cols[None, :],
                 (vals.to(tl.float32) * scale[:, None]).to(tl.bfloat16),
                 mask=mask[:, None])


def gather_dequant(weight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor,
                   block: int = 32) -> torch.Tensor:
    """FP8/E8M0 row gather + dequant; uses the Triton kernel on CUDA, torch fallback otherwise."""
    if (not HAS_TRITON or not weight.is_cuda or scales.shape[1] != 1
            or weight.shape[1] % 16 != 0):
        return gather_dequant_torch(weight, scales, ids)
    flat = ids.reshape(-1)
    out = torch.empty((flat.numel(), weight.shape[1]), dtype=torch.bfloat16, device=weight.device)
    _gather_dequant_kernel[(triton.cdiv(flat.numel(), 64),)](
        weight, scales, flat, out, flat.numel(), weight.shape[1], 64, num_warps=4)
    return out.view(*ids.shape, weight.shape[1])


def engram_gate_decode(x: torch.Tensor, hash_ids: torch.Tensor, weight: torch.Tensor,
                       scales: torch.Tensor, wkv_weight: torch.Tensor,
                       q_weight: torch.Tensor, k_weight: torch.Tensor,
                       hc_mult: int, dim: int, eps: float, clamp_value: float) -> torch.Tensor:
    """Decode Engram gate: gather table -> wkv -> q/k gate -> residual add.

    Numerics mirror the reference ``Engram.forward`` exactly: the table gather and the ``wkv``
    projection run in the model dtype, the gate reduction runs in float32, and the residual add
    is done in float32 before casting back.  That leaves the *intended* FP8/E8M0 table
    quantisation as the only decode-path difference vs the reference.

    x: [B, L, hc_mult, dim], hash_ids: [B, L, n_hash_cols] for one layer.
    """
    embed = gather_dequant(weight, scales, hash_ids).to(wkv_weight.dtype)  # [B,L,heads,dim]
    from .linear_decode import linear_decode
    # batch-invariant projection: the reference F.linear uses a different cuBLAS kernel for M=1 vs
    # M>1, which makes batched decode drift away from the single-sequence runtime (see
    # nanov41/linear_decode.py).
    kv = linear_decode(embed.flatten(-2), wkv_weight)  # [B,L,(hc+1)*dim]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    value = value.float().unsqueeze(-2)
    gate_weight = q_weight.float() * k_weight.float()            # [hc, dim]; only the product is used
    h = x.float()
    h2 = h.square().mean(-1)
    k2 = key.square().mean(-1)
    rstd = torch.rsqrt(h2 + eps) * torch.rsqrt(k2 + eps)
    dot = (h * gate_weight * key).sum(-1) * rstd * (dim ** -0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value).to(x.dtype)


__all__ = ["EngramDecodeCache", "decode_hash_ids", "quantize_table_ue8m0",
           "fp8_scale_to_float", "gather_dequant_torch", "gather_dequant",
           "engram_gate_decode", "HAS_TRITON"]


# --------------------------------------------------------------------------- model integration
def _engram_decode_forward(module, x, hash_ids, token_mask=None):
    """Drop-in Engram.forward using the FP8 gather + gate path."""
    if token_mask is not None:
        return module._original_forward(x, hash_ids, token_mask)
    return engram_gate_decode(
        x, hash_ids, module._decode_fp8, module._decode_scales, module.wkv.weight,
        module.q_weight, module.k_weight, module.hc_mult, module.dim,
        module.eps, module.clamp_value,
    )


def enable_engram_decode(model, block: int = 32, max_rows: int | None = None) -> int:
    """Quantize every Engram table to FP8/E8M0 and patch Engram.forward to use the decode path.

    This changes the model's table numerics (FP8) for *all* sequence lengths, so use it on an
    inference copy.  Returns the number of patched Engram layers.
    """
    from .nano_engram import NanoEngramTable

    inner = getattr(model, "inner", model)
    n = 0
    for layer in getattr(inner, "layers", []):
        eng = getattr(layer, "engram", None)
        if eng is None:
            continue
        table = getattr(eng, "embed", None)
        if table is None or not isinstance(table, NanoEngramTable):
            continue
        w = table.weight.detach()
        if max_rows is not None and w.shape[0] > max_rows:
            w = w[:max_rows]
        q, s = quantize_table_ue8m0(w, block)
        eng._decode_fp8 = q
        eng._decode_scales = s
        eng._original_forward = eng.forward
        eng.forward = _engram_decode_forward.__get__(eng, type(eng))
        n += 1
    return n


def dequantize_tables(model) -> None:
    """Replace each enabled FP8 table with its dequantized bf16 rows (validation/reference helper)."""
    from .nano_engram import NanoEngramTable

    inner = getattr(model, "inner", model)
    for layer in getattr(inner, "layers", []):
        eng = getattr(layer, "engram", None)
        if eng is None or not hasattr(eng, "_decode_fp8"):
            continue
        table = eng.embed
        w = gather_dequant_torch(eng._decode_fp8, eng._decode_scales,
                                 torch.arange(eng._decode_fp8.shape[0], device=eng._decode_fp8.device))
        if isinstance(table, NanoEngramTable):
            table.weight.data.copy_(w)
        eng.forward = eng._original_forward
        del eng._decode_fp8
        del eng._decode_scales
        del eng._original_forward


__all__ += ["enable_engram_decode", "dequantize_tables"]
