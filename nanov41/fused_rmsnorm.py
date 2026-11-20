"""fused_rmsnorm — 官方 `RMSNorm.forward` 的 fwd+bwd 融合 Triton 实现（R45）。

官方 `RMSNorm.forward`（`_ref_patched_model.py:298`）在 d12 训练里被调用 **25 次/step**
（12 层 × attn_norm/ffn_norm + 2 个 compressor norm + 1 个 final norm；indexer k_norm 另计），
每次是 6–7 个 aten kernel、且全程 fp32 中间 buffer：

    x = x.float()                    # bf16 -> fp32  copy   （读 12.6 MB 写 25.2 MB）
    var = x.square().mean(-1)        # mul + reduce
    x = x * rsqrt(var + eps)         # add + rsqrt + mul
    return (self.weight * x).to(dtype)   # mul + cast

按 [16384,384] 估算每次 ~260 MB 流量（≈0.16 ms），25 次 ≈ 4 ms，是本轮"非 GEMM 长尾"里
最大的一块。本模块把整段压成 **1 个 fwd kernel + 2 个 bwd kernel（dw 两段确定性归约）**，
流量降到 ~50 MB/次（读 x、写 y；bwd 再读 dy、写 dx）。

数值语义与官方逐项对齐：平方/均值/rsqrt/权重乘全部 fp32，输出 cast 回 `x.dtype`；
`dw` 的 fp32 部分和按 program id 定序归约（**不用 atomic**）⇒ 三次 replay 逐位一致，
满足 parity 0.0 的验收口径。

用法：
    import nanov41.fused_rmsnorm as fr
    fr.install(M)          # M = patched_env.import_patched()
    fr.uninstall(M)
"""
from __future__ import annotations

import os

import torch

from . import fp8_cast_registry as _castreg

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = ["rmsnorm_fused", "install", "uninstall", "available", "rmsnorm_ref"]

# rows per program: 8 rows × up to 512 cols = 8 KB bf16 tile（小 kernel，多了反而慢）
BM = int(os.environ.get("NANOV41_RMS_BM", "8"))
_DW_BM = int(os.environ.get("NANOV41_RMS_DW_BM", "64"))
_DW_BK = 128
_INSTALLED = None


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """official forward, for tests / CPU fallback."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xh = xf * torch.rsqrt(var + eps)
    return (weight.float() * xh).to(dtype)


def _torch_bwd(dy, x, weight, eps):
    """CPU fallback backward（fp32 参考）。"""
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    xh = xf * r
    dyf = dy.float()
    d_hat = dyf * weight.float()
    s = (d_hat * xh).mean(-1, keepdim=True)
    dx = (r * (d_hat - xh * s)).to(x.dtype)
    dw = (dyf.reshape(-1, dyf.shape[-1]) * xh.reshape(-1, xh.shape[-1])).sum(0).to(weight.dtype)
    return dx, dw


if HAS_TRITON:

    @triton.jit
    def _rms_fwd_kernel(
        X, W, Y, R, T,
        D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        cols = tl.arange(0, BD)
        cmask = cols < D
        x = tl.load(X + rows[:, None] * D + cols[None, :],
                    mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=1) / D
        r = tl.rsqrt(ss + EPS)
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        y = x * r[:, None] * w[None, :]
        tl.store(Y + rows[:, None] * D + cols[None, :], y.to(Y.dtype.element_ty),
                 mask=rmask[:, None] & cmask[None, :])
        tl.store(R + rows, r, mask=rmask)

    @triton.jit
    def _rms_fwd_kernel_fp8(
        X, W, Y, R, Y8, S8,
        T, D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
        MAX: tl.constexpr,
    ):
        """Same math as _rms_fwd_kernel, plus a fused per-row FP8 copy (V4 Fig.5 pattern).

        The fp8 value is quantised from the **bf16-rounded output** (exactly what a downstream
        standalone quantiser would see), so the consumer's numerics are preserved.
        """
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        cols = tl.arange(0, BD)
        cmask = cols < D
        m2 = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rows[:, None] * D + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=1) / D
        r = tl.rsqrt(ss + EPS)
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        y = x * r[:, None] * w[None, :]
        yb = y.to(tl.bfloat16)
        tl.store(Y + rows[:, None] * D + cols[None, :], yb, mask=m2)
        tl.store(R + rows, r, mask=rmask)
        yf = yb.to(tl.float32)
        amax = tl.max(tl.abs(yf), axis=1)
        s = tl.maximum(amax, 1e-12) / MAX
        tl.store(S8 + rows, s, mask=rmask)
        q = tl.minimum(tl.maximum(yf / s[:, None], -MAX), MAX).to(tl.float8e4nv)
        tl.store(Y8 + rows[:, None] * D + cols[None, :], q, mask=m2)

    @triton.jit
    def _rms_bwd_kernel(
        X, DY, W, R, DX, DW_PART, T,
        D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        cols = tl.arange(0, BD)
        cmask = cols < D
        m2 = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rows[:, None] * D + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * D + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        r = tl.load(R + rows, mask=rmask, other=0.0)
        xh = x * r[:, None]
        d_hat = dy * w[None, :]
        s = tl.sum(d_hat * xh, axis=1) / D
        dx = r[:, None] * (d_hat - xh * s[:, None])
        tl.store(DX + rows[:, None] * D + cols[None, :], dx.to(DX.dtype.element_ty), mask=m2)
        dwp = tl.sum(dy * xh, axis=0)  # [D]
        tl.store(DW_PART + pid * D + cols, dwp, mask=cmask)

    @triton.jit
    def _rms_bwd_kernel_fp8(
        X, DY, W, R, DX, DW_PART, DX8, S8, T,
        D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr, MAX: tl.constexpr,
    ):
        """Same math as _rms_bwd_kernel + fused per-row FP8 copy of dX (feeds the next linear's dgrad)."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        cols = tl.arange(0, BD)
        cmask = cols < D
        m2 = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rows[:, None] * D + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * D + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        r = tl.load(R + rows, mask=rmask, other=0.0)
        xh = x * r[:, None]
        d_hat = dy * w[None, :]
        s = tl.sum(d_hat * xh, axis=1) / D
        dx = r[:, None] * (d_hat - xh * s[:, None])
        dxb = dx.to(tl.bfloat16)
        tl.store(DX + rows[:, None] * D + cols[None, :], dxb, mask=m2)
        dwp = tl.sum(dy * xh, axis=0)
        tl.store(DW_PART + pid * D + cols, dwp, mask=cmask)
        dxf = dxb.to(tl.float32)
        amax = tl.max(tl.abs(dxf), axis=1)
        sa = tl.maximum(amax, 1e-12) / MAX
        tl.store(S8 + rows, sa, mask=rmask)
        q = tl.minimum(tl.maximum(dxf / sa[:, None], -MAX), MAX).to(tl.float8e4nv)
        tl.store(DX8 + rows[:, None] * D + cols[None, :], q, mask=m2)

    @triton.jit
    def _dw_reduce_kernel(
        DW_PART, DW, NP, D: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
    ):
        cols = tl.program_id(0) * BD + tl.arange(0, BD)
        cmask = cols < D
        acc = tl.zeros((BD,), dtype=tl.float32)
        for p0 in range(0, NP, BK):
            pid = p0 + tl.arange(0, BK)
            pmask = pid < NP
            v = tl.load(DW_PART + pid[:, None] * D + cols[None, :],
                        mask=pmask[:, None] & cmask[None, :], other=0.0)
            acc += tl.sum(v, axis=0)
        tl.store(DW + cols, acc.to(DW.dtype.element_ty), mask=cmask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        d = x.shape[-1]
        x2 = x.reshape(-1, d)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        t = x2.shape[0]
        # 注意：输出必须是与输入同 shape 的**新张量本体**（不能返回 view）——
        # 否则下游对 `norm_out[..., -rd:]` 的原地写会被 autograd 判定为
        # "view created inside a custom Function" 而报错。
        y = torch.empty_like(x)
        r = torch.empty(t, dtype=torch.float32, device=x.device)
        bd = triton.next_power_of_2(d)
        if _castreg.enabled():
            bm8 = int(os.environ.get("NANOV41_RMS_FP8_BM", "4"))
            bm8 = max(1, min(bm8, BM))
            y8 = torch.empty((t, d), dtype=torch.float8_e4m3fn, device=x.device)
            s8 = torch.empty(t, dtype=torch.float32, device=x.device)
            _rms_fwd_kernel_fp8[(triton.cdiv(t, bm8),)](
                x2, weight, y, r, y8, s8, t, D=d, BD=bd, BM=bm8, EPS=eps, MAX=448.0)
            _castreg.register(y, y8, s8.view(t, 1))
        else:
            _rms_fwd_kernel[(triton.cdiv(t, BM),)](x2, weight, y, r, t, D=d, BD=bd, BM=BM, EPS=eps)
        ctx.save_for_backward(x2, weight, r)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, weight, r = ctx.saved_tensors
        d = x2.shape[-1]
        t = x2.shape[0]
        dy2 = dy.reshape(-1, d)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        bd = triton.next_power_of_2(d)
        fuse = _castreg.enabled()
        if fuse:
            bm8 = max(1, min(int(os.environ.get("NANOV41_RMS_FP8_BM", "4")), BM))
            grid = (triton.cdiv(t, bm8),)
            dx8 = torch.empty((t, d), dtype=torch.float8_e4m3fn, device=x2.device)
            s8 = torch.empty(t, dtype=torch.float32, device=x2.device)
        else:
            bm8 = BM
            grid = (triton.cdiv(t, BM),)
            dx8 = s8 = None
        npart = grid[0]
        dw_part = torch.empty(npart, d, dtype=torch.float32, device=x2.device)
        if fuse:
            _rms_bwd_kernel_fp8[grid](x2, dy2, weight, r, dx, dw_part, dx8, s8, t,
                                      D=d, BD=bd, BM=bm8, MAX=448.0)
        else:
            _rms_bwd_kernel[grid](x2, dy2, weight, r, dx, dw_part, t, D=d, BD=bd, BM=BM)
        dw = torch.empty(d, dtype=torch.float32, device=x2.device)
        _dw_reduce_kernel[(triton.cdiv(d, _DW_BK),)](dw_part, dw, npart, D=d, BD=_DW_BK, BK=_DW_BK)
        dx_ret = dx.view(dy.shape).to(dy.dtype)
        if fuse:
            _castreg.register(dx_ret, dx8, s8.view(t, 1))
        return dx_ret, dw.to(weight.dtype), None


def rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """drop-in for official `RMSNorm.forward`; CPU / no-triton 时退回 torch 实现。"""
    if not HAS_TRITON or not x.is_cuda:
        y = rmsnorm_ref(x, weight, eps)
        if x.requires_grad or weight.requires_grad:
            # 训练用不到 CPU 反传；这里给出可导近似，保持 API 干净
            class _Cpu(torch.autograd.Function):
                @staticmethod
                def forward(c, x, w):
                    c.save_for_backward(x, w)
                    return rmsnorm_ref(x, w, eps)

                @staticmethod
                def backward(c, dy):
                    x, w = c.saved_tensors
                    return _torch_bwd(dy, x, w, eps)

            return _Cpu.apply(x, weight)
        return y
    return _RMSNormFn.apply(x, weight, eps)


def _rms_forward(self, x):
    return rmsnorm_fused(x, self.weight, self.eps)


def install(M):
    """把 `M.RMSNorm.forward` 换成融合实现（幂等）。"""
    global _INSTALLED
    prev = getattr(M.RMSNorm, "forward", None)
    if _INSTALLED is not None and _INSTALLED[0] is M.RMSNorm:
        return _INSTALLED[1]
    M.RMSNorm.forward = _rms_forward
    _INSTALLED = (M.RMSNorm, prev)
    return prev


def uninstall(M=None):
    global _INSTALLED
    if _INSTALLED is None:
        return
    cls, prev = _INSTALLED
    if prev is not None:
        cls.forward = prev
    _INSTALLED = None


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
