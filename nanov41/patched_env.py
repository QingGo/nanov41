"""Torch fallbacks for the official DeepSeek-V4.1 reference kernels.

The official HF repo ships `inference/kernel.py` as TileLang kernels (FP8/FP4 quant,
sparse attention, hyper-connection Sinkhorn) that only JIT-compile on the vendor GPUs.
For training and for CPU/Mac parity work we install pure-torch stand-ins with the same
signatures/semantics, then import the official `ref/model.py` unchanged.

Design notes:
- quantization is a no-op here (`act_quant` / `fp4_act_quant`): a training port runs in
  bf16, and for parity we compare the *same* code path, so dropping quant precision is
  fine as long as both sides use these fallbacks.
- `sparse_attn` reproduces the kernel's math exactly: gather KV rows by index (-1 =
  invalid), softmax over the gathered slots, plus a learnable `attn_sink` added only to
  the denominator (equivalent to an extra key whose value is zero).
- `hc_split_sinkhorn` reproduces the kernel's normalization order (softmax rows ->
  column normalize -> (iters-1) x [row, column]).
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

import torch
import torch.nn.functional as F

REF_DIR = pathlib.Path(__file__).resolve().parent.parent / "ref"
PATCHED_DIR = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------- kernels
def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """No-op stand-in (training runs in bf16). Keeps the official call signatures."""
    if inplace:
        return x
    return x, None


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if inplace:
        return x
    return x, None


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    raise NotImplementedError(
        "fp8 weights are not supported by the torch fallback; build the model with dtype='bf16'."
    )


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    raise NotImplementedError(
        "fp4 weights are not supported by the torch fallback; build the model with expert_dtype=None."
    )


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q: [b,m,h,d]  kv: [b,n,d]  attn_sink: [h]  topk_idxs: [b,m,topk] (-1 invalid).

    Softmax over the gathered slots; the sink enters only the denominator. Rows whose
    indices are all invalid produce zeros (same convention as the TileLang kernel).
    """
    b, m, h, d = q.shape
    if topk_idxs.numel() == 0:
        return q.new_zeros(b, m, h, d)
    valid = topk_idxs >= 0
    safe = topk_idxs.clamp_min(0)
    n = kv.size(1)
    flat = kv.reshape(b * n, d)
    flat_idx = (torch.arange(b, device=kv.device).view(b, 1, 1) * n + safe).reshape(-1)
    kvg = flat.index_select(0, flat_idx).reshape(b, m, topk_idxs.size(-1), d)  # [b,m,topk,d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), kvg.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = scores.amax(dim=-1)  # [b,m,h]
    finite = torch.isfinite(mx)
    mx_safe = torch.where(finite, mx, torch.zeros_like(mx))
    p = torch.where(valid.unsqueeze(2), (scores - mx_safe.unsqueeze(-1)).exp(), torch.zeros_like(scores))
    denom = p.sum(dim=-1) + torch.exp(attn_sink.view(1, 1, h) - mx_safe)
    out = torch.einsum("bmht,bmtd->bmhd", p, kvg.float()) / denom.unsqueeze(-1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """mixes: [b,s,(2+hc)*hc] -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]."""
    b, s, _ = mixes.size()
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = m[..., 2 * hc :].reshape(b, s, hc, hc) * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# --------------------------------------------------------------------------- stubs
def _make_stub_modules() -> dict[str, types.ModuleType]:
    kernel = types.ModuleType("kernel")
    kernel._nanov41_stub = True
    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_act_quant
    kernel.fp8_gemm = fp8_gemm
    kernel.fp4_gemm = fp4_gemm
    kernel.sparse_attn = sparse_attn
    kernel.hc_split_sinkhorn = hc_split_sinkhorn

    vision = types.ModuleType("vision")
    vision._nanov41_stub = True

    class _UnavailableVision(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("vision encoder is not part of the text-only training port")

    vision.ViT = _UnavailableVision
    vision.Aligner = _UnavailableVision

    image_processor = types.ModuleType("image_processor")
    image_processor._nanov41_stub = True
    image_processor.IMAGE = -2
    image_processor.IMAGE_START = -3
    image_processor.IMAGE_END = -4
    image_processor.IMAGE_NEW_LINE = -5

    return {"kernel": kernel, "vision": vision, "image_processor": image_processor}


def install() -> None:
    """Install the fallbacks and put the official reference dir on sys.path (idempotent)."""
    if "kernel" in sys.modules and getattr(sys.modules["kernel"], "_nanov41_stub", False):
        pass
    else:
        sys.modules.update(_make_stub_modules())
    # REF_DIR is needed for the official code's absolute sibling imports (`engram`, `kernel`).
    # PATCHED_DIR is NOT put on sys.path: it contains nanov41/model.py, which would shadow the
    # official `model` module for top-level imports. We load both by explicit path instead.
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))


def _load_by_path(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_official():
    """Import the untouched official reference model (forward-only; needs `install()`)."""
    install()
    return _load_by_path("dsv41_ref_model", REF_DIR / "model.py")


def import_patched():
    """Import the training-patched copy (see tools/make_patched_ref.py for the diff)."""
    install()
    return _load_by_path("_ref_patched_model", PATCHED_DIR / "_ref_patched_model.py")
