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

from . import fp8_cast_registry as _castreg

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


    @triton.jit
    def _swiglu_fwd_kernel_fp8(G, U, O, Y8, S8, T,
                               INT: tl.constexpr, BI: tl.constexpr, LIM, HAS_LIM: tl.constexpr,
                               BR: tl.constexpr, MAX: tl.constexpr):
        """Row-based SwiGLU fwd + fused per-row FP8 copy (feeds the following w2 linear)."""
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        rmask = rows < T
        cols = tl.arange(0, BI)
        cmask = cols < INT
        m2 = rmask[:, None] & cmask[None, :]
        idx = rows[:, None] * INT + cols[None, :]
        g = tl.load(G + idx, mask=m2, other=0.0).to(tl.float32)
        u = tl.load(U + idx, mask=m2, other=0.0).to(tl.float32)
        if HAS_LIM:
            u = tl.minimum(tl.maximum(u, -LIM), LIM)
            g = tl.minimum(g, LIM)
        y = g * tl.sigmoid(g) * u
        yb = y.to(tl.bfloat16)
        tl.store(O + idx, yb, mask=m2)
        yf = yb.to(tl.float32)
        amax = tl.max(tl.abs(yf), axis=1)
        s = tl.maximum(amax, 1e-12) / MAX
        tl.store(S8 + rows, s, mask=rmask)
        q = tl.minimum(tl.maximum(yf / s[:, None], -MAX), MAX).to(tl.float8e4nv)
        tl.store(Y8 + idx, q, mask=m2)

    @triton.jit
    def _swiglu_bwd_kernel_fp8(G, U, DO, DG, DU, DG8, DU8, SG, SU, T,
                               INT: tl.constexpr, BI: tl.constexpr, LIM, HAS_LIM: tl.constexpr,
                               BR: tl.constexpr, MAX: tl.constexpr):
        """Row-based SwiGLU bwd + fused FP8 copies of dGate/dUp (feed the w1/w3 dgrad)."""
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        rmask = rows < T
        cols = tl.arange(0, BI)
        cmask = cols < INT
        m2 = rmask[:, None] & cmask[None, :]
        idx = rows[:, None] * INT + cols[None, :]
        gp = tl.load(G + idx, mask=m2, other=0.0).to(tl.float32)
        up_ = tl.load(U + idx, mask=m2, other=0.0).to(tl.float32)
        do = tl.load(DO + idx, mask=m2, other=0.0).to(tl.float32)
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
        dgb = dg.to(tl.bfloat16)
        dub = du.to(tl.bfloat16)
        tl.store(DG + idx, dgb, mask=m2)
        tl.store(DU + idx, dub, mask=m2)
        dgf = dgb.to(tl.float32)
        ag = tl.max(tl.abs(dgf), axis=1)
        sg = tl.maximum(ag, 1e-12) / MAX
        tl.store(SG + rows, sg, mask=rmask)
        qg = tl.minimum(tl.maximum(dgf / sg[:, None], -MAX), MAX).to(tl.float8e4nv)
        tl.store(DG8 + idx, qg, mask=m2)
        duf = dub.to(tl.float32)
        au = tl.max(tl.abs(duf), axis=1)
        su = tl.maximum(au, 1e-12) / MAX
        tl.store(SU + rows, su, mask=rmask)
        qu = tl.minimum(tl.maximum(duf / su[:, None], -MAX), MAX).to(tl.float8e4nv)
        tl.store(DU8 + idx, qu, mask=m2)

def _launch_fwd(gate, up, out, limit):
    n = gate.numel()
    if _castreg.enabled():
        inter = gate.shape[-1]
        t = n // inter
        br = int(os.environ.get("NANOV41_SWIGLU_FP8_BR", "8"))
        bi = triton.next_power_of_2(inter)
        y8 = torch.empty((t, inter), dtype=torch.float8_e4m3fn, device=gate.device)
        s8 = torch.empty(t, dtype=torch.float32, device=gate.device)
        _swiglu_fwd_kernel_fp8[(triton.cdiv(t, br),)](
            gate.reshape(t, inter), up.reshape(t, inter), out.reshape(t, inter), y8, s8, t,
            INT=inter, BI=bi, LIM=float(limit), HAS_LIM=limit > 0, BR=br, MAX=448.0)
        _castreg.register(out, y8, s8.view(t, 1))
        return
    _swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, float(limit),
                                                 HAS_LIM=limit > 0, BLOCK=BLOCK)


def _launch_bwd(gate, up, dout, dgate, dup, limit):
    n = gate.numel()
    if _castreg.enabled():
        inter = gate.shape[-1]
        t = n // inter
        br = int(os.environ.get("NANOV41_SWIGLU_FP8_BR", "8"))
        bi = triton.next_power_of_2(inter)
        z = lambda: torch.empty((t, inter), dtype=torch.float8_e4m3fn, device=gate.device)
        dg8, du8 = z(), z()
        sg8 = torch.empty(t, dtype=torch.float32, device=gate.device)
        su8 = torch.empty(t, dtype=torch.float32, device=gate.device)
        _swiglu_bwd_kernel_fp8[(triton.cdiv(t, br),)](
            gate.reshape(t, inter), up.reshape(t, inter), dout.reshape(t, inter),
            dgate.reshape(t, inter), dup.reshape(t, inter), dg8, du8, sg8, su8, t,
            INT=inter, BI=bi, LIM=float(limit), HAS_LIM=limit > 0, BR=br, MAX=448.0)
        _castreg.register(dgate, dg8, sg8.view(t, 1))
        _castreg.register(dup, du8, su8.view(t, 1))
        return
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
