"""fused_swiglu — 官方 `Expert.forward` 的 SwiGLU 段 fwd+bwd 融合（R45）。

官方实现（`_ref_patched_model.py:856`，d12 的 `swiglu_limit=10`）：

    gate = self.w1(x).float()          # bf16 -> fp32 copy
    up   = self.w3(x).float()          # bf16 -> fp32 copy
    up   = torch.clamp(up, -10, 10)    # clamp
    gate = torch.clamp(gate, max=10)   # clamp
    x    = F.silu(gate) * up           # silu + mul（fp32）
    return self.w2(x.to(dtype))        # fp32 -> bf16 copy

共享专家每层都走这段（12 次/step），六个 elementwise kernel、全程 fp32 中间量。
本模块把 clamp+silu+mul+cast 压成 1 个 kernel（fwd），反传同样是 1 个 kernel
（直接给出 d(gate)/d(up)，带 clamp 饱和 mask），数值语义与官方逐步对齐（fp32 计算后 cast）。

用法：
    import nanov41.fused_swiglu as fsw
    fsw.install(M)     # 只替换 `weights is None` 的共享专家路径（routed 走 grouped kernel）
"""
from __future__ import annotations

import os

import torch

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = ["swiglu_fused", "swiglu_ref", "install", "uninstall", "available"]

BLOCK = int(os.environ.get("NANOV41_SWIGLU_BLOCK", "2048"))
_INSTALLED = None


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """官方前向的 torch 复刻（CPU fallback / 测试用）；返回 bf16/fp32 与输入同 dtype。"""
    dtype = gate.dtype
    g = gate.float()
    u = up.float()
    if limit > 0:
        u = torch.clamp(u, -limit, limit)
        g = torch.clamp(g, max=limit)
    return (torch.nn.functional.silu(g) * u).to(dtype)


def _torch_bwd(gate, up, dout, limit):
    g = gate.float()
    u = up.float()
    do = dout.float()
    if limit > 0:
        u = torch.clamp(u, -limit, limit)
        g = torch.clamp(g, max=limit)
    sig = torch.sigmoid(g)
    silu = g * sig
    dy_dg = u * (sig * (1.0 + g * (1.0 - sig)))
    dy_du = silu
    dgate = do * dy_dg
    dup = do * dy_du
    if limit > 0:
        # 注意：PyTorch clamp 的反传是**严格**不等式（边界上 grad=0，已实测 2.14）：
        #   clamp(max=L) -> grad 仅当 x < L；clamp(min=-L,max=L) -> 仅当 -L < x < L
        gp = gate.float()
        up_p = up.float()
        dgate = torch.where(gp < limit, dgate, torch.zeros_like(dgate))
        dup = torch.where((up_p > -limit) & (up_p < limit), dup, torch.zeros_like(dup))
    return dgate.to(gate.dtype), dup.to(up.dtype)


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(G, U, O, N, LIM, HAS_LIM: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        g = tl.load(G + idx, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + idx, mask=mask, other=0.0).to(tl.float32)
        if HAS_LIM:
            u = tl.minimum(tl.maximum(u, -LIM), LIM)
            g = tl.minimum(g, LIM)
        y = g * tl.sigmoid(g) * u
        tl.store(O + idx, y.to(O.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(G, U, DO, DG, DU, N, LIM, HAS_LIM: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        gp = tl.load(G + idx, mask=mask, other=0.0).to(tl.float32)
        up_ = tl.load(U + idx, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DO + idx, mask=mask, other=0.0).to(tl.float32)
        g = gp
        u = up_
        if HAS_LIM:
            u = tl.minimum(tl.maximum(u, -LIM), LIM)
            g = tl.minimum(g, LIM)
        sig = tl.sigmoid(g)
        silu = g * sig
        dy_dg = u * (sig * (1.0 + g * (1.0 - sig)))
        dg = do * dy_dg
        du = do * silu
        if HAS_LIM:
            dg = tl.where(gp < LIM, dg, 0.0)
            du = tl.where((up_ > -LIM) & (up_ < LIM), du, 0.0)
        tl.store(DG + idx, dg.to(DG.dtype.element_ty), mask=mask)
        tl.store(DU + idx, du.to(DU.dtype.element_ty), mask=mask)


def _launch_fwd(gate, up, out, limit):
    n = gate.numel()
    _swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, float(limit),
                                                 HAS_LIM=limit > 0, BLOCK=BLOCK)


def _launch_bwd(gate, up, dout, dgate, dup, limit):
    n = gate.numel()
    _swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, dout, dgate, dup, n, float(limit),
                                                 HAS_LIM=limit > 0, BLOCK=BLOCK)


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, limit):
        out = torch.empty_like(gate)
        _launch_fwd(gate, up, out, limit)
        ctx.save_for_backward(gate, up)
        ctx.limit = float(limit)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        _launch_bwd(gate, up, dout, dgate, dup, ctx.limit)
        return dgate, dup, None


def swiglu_fused(gate: torch.Tensor, up: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    if not HAS_TRITON or not gate.is_cuda:
        out = swiglu_ref(gate, up, limit)
        if gate.requires_grad or up.requires_grad:
            class _Cpu(torch.autograd.Function):
                @staticmethod
                def forward(c, g, u):
                    c.save_for_backward(g, u)
                    return swiglu_ref(g, u, limit)

                @staticmethod
                def backward(c, do):
                    g, u = c.saved_tensors
                    a, b = _torch_bwd(g, u, do, limit)
                    return a, b

            return _Cpu.apply(gate, up)
        return out
    return _SwiGLUFn.apply(gate, up, float(limit))


# --------------------------------------------------------------------------- model patch
_ORIG = {}


def _expert_forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
    """替换官方 `Expert.forward`：weights is None 时（共享专家）走融合 SwiGLU。"""
    if weights is not None or not HAS_TRITON or not x.is_cuda:
        return _ORIG["Expert"](self, x, weights)
    gate = self.w1(x)
    up = self.w3(x)
    h = swiglu_fused(gate, up, float(getattr(self, "swiglu_limit", 0.0) or 0.0))
    return self.w2(h)


def install(M):
    global _INSTALLED
    if _INSTALLED is not None and _INSTALLED[0] is M.Expert:
        return _INSTALLED[1]
    prev = M.Expert.__dict__.get("forward")
    _ORIG["Expert"] = prev
    M.Expert.forward = _expert_forward
    _INSTALLED = (M.Expert, prev)
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
