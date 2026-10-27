"""mHC mixes 的核心算子融合：`vector_norm + bf16 linear + rsqrt 乘` 合成一个 Triton kernel（含解析反向）。

## 为什么

5090/b=16 的 mHC 子项（S16 §27）：hc_mixes **15.0 ms/步**（占 mHC 85%），其中 split 段已在
`triton_mhc_split.py` 融合；剩下的 **norm + bf16 GEMM + rsqrt 乘** 是 3 个 kernel ×2（前向+反向各若干），
×24 次/步。这段数学非常轻（每行 1 个平方和 + 24 个 dot + 1 个缩放），是典型的"小 kernel 太多"。

官方对应的融合是 `mhc_pre_big_fuse_fwd`，但它需要 **DeepGEMM 的 split-K partials** 且**没有训练用 backward**
（详见 S16 §24/§27 的 A/B 与边界论证），所以在我们的 torch/Triton 链路里自研这一小段是必要且已量化收益的。

## 数学（与 level3 路径一致）

    ss_i = Σ_j x_ij²
    r_i  = rsqrt(ss_i / D + eps)
    u_i  = x_i @ FNᵀ            （FN 为 bf16，fp32 累加）
    m_i  = u_i · r_i            （fp32）

反向（给定 dm）：
    du = dm · r
    dss = -0.5 · r³ / D · Σ_c (dm_c · u_c)
    dx  = du @ FN + 2·dss·x
    dFN = duᵀ @ X               （torch 侧一次 bf16 matmul，[24,T]×[T,K]）
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

MIX = 24          # 2*hc + hc*hc
MIXP = 32         # tl.dot 的 N 需要 2 的幂
BM = 128
BK = 128


if HAS_TRITON:

    @triton.jit
    def _core_fwd(X, FNT, OUT, R, T, D: tl.constexpr, K: tl.constexpr,
                  MIX: tl.constexpr, MIXP: tl.constexpr, BM: tl.constexpr,
                  BK: tl.constexpr, EPS: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        mask = rows < T
        rm = tl.arange(0, MIXP)
        rc = tl.arange(0, BK)
        acc = tl.zeros((BM, MIXP), dtype=tl.float32)
        ss = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            cols = k0 + rc
            x = tl.load(X + rows[:, None] * K + cols[None, :], mask=mask[:, None], other=0.0)
            f = tl.load(FNT + cols[:, None] * MIXP + rm[None, :])          # [BK, MIXP] bf16
            acc = tl.dot(x, f, acc)
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
        r = tl.rsqrt(ss / D + EPS)
        out = acc * r[:, None]
        mstore = mask[:, None] & (rm[None, :] < MIX)
        tl.store(OUT + rows[:, None] * MIX + rm[None, :], out, mask=mstore)
        tl.store(R + rows, r, mask=mask)

    @triton.jit
    def _core_bwd(X, FNT, MIXES, R, DM, DX, DFN_PART, T,
                  D: tl.constexpr, K: tl.constexpr, MIX: tl.constexpr,
                  MIXP: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        mask = rows < T
        rm = tl.arange(0, MIXP)
        rc = tl.arange(0, BK)
        r = tl.load(R + rows, mask=mask, other=1.0)
        dm = tl.load(DM + rows[:, None] * MIX + rm[None, :],
                     mask=mask[:, None] & (rm[None, :] < MIX), other=0.0)
        mm = tl.load(MIXES + rows[:, None] * MIX + rm[None, :],
                     mask=mask[:, None] & (rm[None, :] < MIX), other=0.0)
        u = mm / r[:, None]
        dot_sum = tl.sum(dm * u, axis=1)                       # [BM]
        dss = -0.5 * r * r * r / D * dot_sum
        du = dm * r[:, None]                                   # [BM, MIXP]
        for k0 in range(0, K, BK):
            cols = k0 + rc
            x = tl.load(X + rows[:, None] * K + cols[None, :], mask=mask[:, None], other=0.0)
            f = tl.load(FNT + cols[:, None] * MIXP + rm[None, :])          # [BK, MIXP]
            dxk = tl.dot(du.to(tl.bfloat16), tl.trans(f)) + 2.0 * dss[:, None] * x.to(tl.float32)
            tl.store(DX + rows[:, None] * K + cols[None, :], dxk.to(DX.dtype.element_ty),
                     mask=mask[:, None])
            dfnk = tl.dot(tl.trans(du.to(tl.bfloat16)), x)                 # [MIXP, BK]
            tl.store(DFN_PART + pid * MIXP * K + rm[:, None] * K + cols[None, :],
                     dfnk.to(tl.float32))

    def core_fwd(x, fnt, eps=1e-6):
        """x [T, K] bf16；fnt [K, 32] bf16（前 24 列是 fnᵀ，其余 0）→ mixes [T,24] fp32, r [T] fp32。"""
        t, k = x.shape
        out = torch.empty(t, MIX, dtype=torch.float32, device=x.device)
        r = torch.empty(t, dtype=torch.float32, device=x.device)
        _core_fwd[(triton.cdiv(t, BM),)](x, fnt, out, r, t, D=k, K=k, MIX=MIX, MIXP=MIXP,
                                         BM=BM, BK=BK, EPS=eps, num_warps=4, num_stages=2)
        return out, r

    def core_bwd(x, fnt, mixes, r, dm, eps=1e-6):
        t, k = x.shape
        nprog = triton.cdiv(t, BM)
        dx = torch.empty_like(x)
        dfn_part = torch.empty(nprog, MIXP, k, dtype=torch.float32, device=x.device)
        _core_bwd[(nprog,)](x, fnt, mixes, r, dm, dx, dfn_part, t, D=k, K=k, MIX=MIX,
                            MIXP=MIXP, BM=BM, BK=BK, num_warps=4, num_stages=2)
        return dx, dfn_part.sum(0)[:MIX]


class _CoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn):
        """x [T, K] bf16（已 flatten），fn [24, K]（任意 dtype，内部转 bf16 对齐 level3）→ mixes [T,24] fp32。"""
        fnt = torch.zeros(fn.size(1), MIXP, dtype=torch.bfloat16, device=fn.device)
        fnt[:, :MIX] = fn.to(torch.bfloat16).t()
        fnt = fnt.contiguous()
        out, r = core_fwd(x.contiguous(), fnt)
        ctx.save_for_backward(x, fnt, out, r)
        ctx.fn_dtype = fn.dtype
        return out

    @staticmethod
    def backward(ctx, dm):
        x, fnt, mixes, r = ctx.saved_tensors
        dx, dfn = core_bwd(x, fnt, mixes, r, dm.contiguous())
        return dx, dfn.to(ctx.fn_dtype)


def hc_mixes_core(x_flat, fn, eps=1e-6):
    """x_flat: [b, s, hc*d] 或 [T, hc*d] bf16；fn: [24, hc*d] → mixes [b, s, 24] / [T, 24] fp32。"""
    shape = x_flat.shape
    x2 = x_flat.reshape(-1, shape[-1])
    out = _CoreFn.apply(x2, fn)
    return out.view(*shape[:-1], MIX)


def available():
    return HAS_TRITON and torch.cuda.is_available()
