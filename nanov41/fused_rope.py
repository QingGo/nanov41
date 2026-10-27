"""fused_rope — 官方 `apply_rotary_emb` 的融合实现（R45，fwd 1 kernel + 可追踪原地写）。

官方实现（`_ref_patched_model.py:402`）每次调用 3–4 个 aten kernel 且全程 fp32 中间量：

    x = torch.view_as_complex(x.float()...)   # bf16 -> fp32 copy
    x = torch.view_as_real(x * freqs_cis)     # complex64 复数乘（fp32）
    y.copy_(x)                                # fp32 -> bf16 copy（原地写回切片）

d12 训练里它被调用 **42 次/step**（每层 q / window-kv / output 各一次，压缩层再多一次 latent），
每次约 140 MB 流量，torch 实测 ~112 µs/次（R45 归因：4.73 ms/step，非 GEMM 长尾最大单项之一）。

为什么不能"直接原地写回"：Triton 写内存对 autograd 不可见，旋转的梯度会丢。官方用
`y.copy_(x)` 让 autograd 记录这次原地写；我们保留同样的结构，但把"算旋转"压成 1 个 kernel：

    r = RopeFn.apply(x_slice, freqs, inverse)   # 1 kernel：读 x、写 r（bf16，25 MB 流量）
    x.copy_(r)                                  # 官方同样的可追踪原地写（25 MB 流量）

⇒ 每次 50 MB，比官方 ~140 MB 少 64%；反传 = 反向旋转（同一个 kernel）。

支持 `[b, s, d]`（H=1）与 `[b, s, h, d]`（4D）；`freqs_cis` 允许非连续（`[::ratio]` 行跨度）。
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = ["apply_rotary_emb_fused", "apply_rotary_emb_ref", "install", "uninstall", "available"]

_INSTALLED = None


def apply_rotary_emb_ref(x, freqs_cis, inverse=False):
    """官方实现逐行复刻（CPU fallback / 测试用）。"""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def _freqs_2d(freqs_cis: torch.Tensor):
    """复数 freqs [n, rd/2] -> (fp32 [n, rd] 视图或拷贝, 行跨度(元素))。"""
    assert freqs_cis.is_complex()
    k = freqs_cis.shape[-1]
    rd = 2 * k
    base = freqs_cis.reshape(-1, k)
    row_stride = base.stride(0) * 2  # complex 元素跨度 -> fp32 元素跨度
    flat = torch.view_as_real(base)
    if flat.stride(-1) != 1 or flat.stride(-2) != 2:
        flat = flat.contiguous()
        row_stride = rd
    return flat.reshape(-1, rd), row_stride


def _layout(x: torch.Tensor):
    """(B, S, H, 父行跨度 SP, batch stride SB, token stride ST)。"""
    if x.dim() == 3:
        b, s, _ = x.shape
        return b, s, 1, x.stride(1), x.stride(0), x.stride(1)
    if x.dim() == 4:
        b, s, h, _ = x.shape
        return b, s, h, x.stride(2), x.stride(0), x.stride(1)
    raise ValueError(f"apply_rotary_emb: unsupported ndim {x.dim()}")


if HAS_TRITON:

    @triton.jit
    def _rope_copy_kernel(
        X, F, R, T,
        SB, ST, SP,      # 输入 strides（batch / token / head 父行跨度）
        OB, OT, OP,      # 输出 strides
        RD: tl.constexpr,   # 旋转维度（偶数）
        H: tl.constexpr,    # 每 token 头数（3D 形态为 1）
        BR: tl.constexpr,   # pair 数（RD//2 向上取 2 的幂）
        FS,                 # freqs 行跨度（fp32 元素）
        INVERSE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid % T
        b = pid // T
        j = tl.arange(0, BR)
        pmask = j < RD // 2
        f_off = t * FS + 2 * j
        fre = tl.load(F + f_off, mask=pmask, other=0.0)
        fim = tl.load(F + f_off + 1, mask=pmask, other=0.0)
        if INVERSE:
            fim = -fim
        for h in tl.static_range(H):
            ioff = b * SB + t * ST + h * SP
            ooff = b * OB + t * OT + h * OP
            re = tl.load(X + ioff + 2 * j, mask=pmask, other=0.0).to(tl.float32)
            im = tl.load(X + ioff + 2 * j + 1, mask=pmask, other=0.0).to(tl.float32)
            tl.store(R + ooff + 2 * j, (re * fre - im * fim).to(R.dtype.element_ty), mask=pmask)
            tl.store(R + ooff + 2 * j + 1, (re * fim + im * fre).to(R.dtype.element_ty), mask=pmask)


def _launch(src, dst, freqs_cis, inverse):
    b, s, h, sp, sb, st = _layout(src)
    _, _, _, odp, odb, odt = _layout(dst)
    fr, fs = _freqs_2d(freqs_cis)
    rd = src.shape[-1]
    br = triton.next_power_of_2(rd // 2)
    _rope_copy_kernel[(b * s,)](
        src, fr, dst, s,
        sb, st, sp, odb, odt, odp,
        RD=rd, H=h, BR=br, FS=fs, INVERSE=bool(inverse),
    )


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis, inverse):
        r = torch.empty_like(x)  # 同 shape 的连续新张量（不是 view）
        _launch(x, r, freqs_cis, inverse)
        ctx.save_for_backward(freqs_cis)
        ctx.inverse = bool(inverse)
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        ctx.freq_dtype = freqs_cis.dtype
        return r

    @staticmethod
    def backward(ctx, dr):
        (freqs_cis,) = ctx.saved_tensors
        # dr 可能是非连续的切片梯度（copy_ 的 view 反传）；先连续化再旋转
        drc = dr.contiguous() if not dr.is_contiguous() else dr
        dx = torch.empty(ctx.shape, dtype=ctx.dtype, device=drc.device)
        _launch(drc, dx, freqs_cis, not ctx.inverse)
        return dx, None, None


def apply_rotary_emb_fused(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """与官方语义一致的原地旋转（fwd 1 kernel + 可追踪 copy_）。"""
    if not HAS_TRITON or not x.is_cuda:
        return apply_rotary_emb_ref(x, freqs_cis, inverse)
    r = _RopeFn.apply(x, freqs_cis, bool(inverse))
    x.copy_(r)  # 官方同样的原地写，autograd 可追踪
    return x


def _rope_forward(x, freqs_cis, inverse=False):
    return apply_rotary_emb_fused(x, freqs_cis, inverse)


def install(M):
    """把 `M.apply_rotary_emb` 换成融合实现（幂等）。"""
    global _INSTALLED
    if _INSTALLED is not None and _INSTALLED[0] is M:
        return _INSTALLED[1]
    prev = getattr(M, "apply_rotary_emb", None)
    M.apply_rotary_emb = _rope_forward
    _INSTALLED = (M, prev)
    return prev


def uninstall(M=None):
    global _INSTALLED
    if _INSTALLED is None:
        return
    mod, prev = _INSTALLED
    if prev is not None:
        mod.apply_rotary_emb = prev
    _INSTALLED = None


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
