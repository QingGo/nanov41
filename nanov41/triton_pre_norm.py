"""mHC `hc_pre` + `RMSNorm` 的融合：一次 collapse（Σ_h p_h·x_h）后直接做 RMSNorm。

## 为什么（S16 §34）
b=16 时 mHC 仍占 ~11–15 ms/步，其中 `hc_pre`（加权求和）与紧随其后的 `attn_norm`/`ffn_norm`
（RMSNorm：float 上采样 + square + mean + rsqrt + mul + cast ≈5 个小 kernel）都是纯搬运。
ref 的结构是固定的两行：
    x = self.hc_pre(x, pre_mix)     # Σ_h p_h x_h，fp32 累加
    x = self.attn_norm(x)           # x.float(); var=mean(x²); x*rsqrt(var+eps); weight*x
融合后：1 个 fwd kernel（读 x 一次、不物化 pre-norm 的 [T,d]）+ 1 个 bwd kernel。

## 数学
    y = Σ_h p_h · x_h                         （fp32）
    var = mean(y²); r = rsqrt(var + eps)
    out = (weight · y · r).to(bf16)
反向（d = dout·weight）：
    dy_i = d_i·r − (Σ_k d_k y_k)·r³·y_i / D
    dres_h = p_h · dy ;  dpre_h = Σ_i dy_i · x_{h,i} ;  dweight = Σ_t dout·y·r
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

BM = 16
BD = 512


if HAS_TRITON:

    @triton.jit
    def _pre_norm_fwd(X, PIN, W, OUT, T, HC: tl.constexpr, D: tl.constexpr,
                      BD: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        maskr = rows < T
        rd = tl.arange(0, BD)
        maskd = rd < D
        rh = tl.arange(0, HC)
        p = tl.load(PIN + rows[:, None] * HC + rh[None, :], mask=maskr[:, None], other=0.0)
        w = tl.load(W + rd, mask=maskd, other=0.0).to(tl.float32)
        y = tl.zeros((BM, BD), dtype=tl.float32)
        for h in tl.static_range(HC):
            ph = tl.sum(tl.where(rh[None, :] == h, p, 0.0), axis=1)
            xh = tl.load(X + rows[:, None] * (HC * D) + h * D + rd[None, :],
                         mask=maskr[:, None] & maskd[None, :], other=0.0).to(tl.float32)
            y += ph[:, None] * xh
        var = tl.sum(y * y, axis=1) / D
        r = tl.rsqrt(var + EPS)
        out = w[None, :] * y * r[:, None]
        tl.store(OUT + rows[:, None] * D + rd[None, :], out.to(OUT.dtype.element_ty),
                 mask=maskr[:, None] & maskd[None, :])

    @triton.jit
    def _pre_norm_bwd(X, PIN, W, DOUT, DRES, DPRE, DW_PART, T,
                      HC: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                      BM: tl.constexpr, EPS: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        maskr = rows < T
        rd = tl.arange(0, BD)
        maskd = rd < D
        rh = tl.arange(0, HC)
        md = maskr[:, None] & maskd[None, :]
        p = tl.load(PIN + rows[:, None] * HC + rh[None, :], mask=maskr[:, None], other=0.0)
        w = tl.load(W + rd, mask=maskd, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + rows[:, None] * D + rd[None, :], mask=md, other=0.0).to(tl.float32)
        y = tl.zeros((BM, BD), dtype=tl.float32)
        for h in tl.static_range(HC):
            ph = tl.sum(tl.where(rh[None, :] == h, p, 0.0), axis=1)
            xh = tl.load(X + rows[:, None] * (HC * D) + h * D + rd[None, :], mask=md,
                         other=0.0).to(tl.float32)
            y += ph[:, None] * xh
        var = tl.sum(y * y, axis=1) / D
        r = tl.rsqrt(var + EPS)
        d = dout * w[None, :]
        sdy = tl.sum(d * y, axis=1)                       # [BM]
        dy = d * r[:, None] - (sdy * r * r * r / D)[:, None] * y
        dwp = tl.sum(dout * y, axis=0) * r[0] if False else tl.sum(dout * y * r[:, None], axis=0)
        tl.store(DW_PART + pid * BD + rd, dwp, mask=maskd)
        for h in tl.static_range(HC):
            ph = tl.sum(tl.where(rh[None, :] == h, p, 0.0), axis=1)
            xh = tl.load(X + rows[:, None] * (HC * D) + h * D + rd[None, :], mask=md,
                         other=0.0).to(tl.float32)
            dres = ph[:, None] * dy
            tl.store(DRES + rows[:, None] * (HC * D) + h * D + rd[None, :],
                     dres.to(DRES.dtype.element_ty), mask=md)
            dpre = tl.sum(dy * xh, axis=1)
            tl.store(DPRE + rows[:, None] * HC + h, dpre, mask=maskr)

    class _PreNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, pre_mix, weight, eps):
            b, s, hc, d = x.shape
            t = b * s
            xf = x.reshape(t, hc * d).contiguous()
            out = torch.empty(t, d, dtype=x.dtype, device=x.device)
            _pre_norm_fwd[(triton.cdiv(t, BM),)](xf, pre_mix.reshape(t, hc).contiguous(), weight,
                                                 out, t, HC=hc, D=d, BD=BD, BM=BM, EPS=eps,
                                                 num_warps=4, num_stages=2)
            ctx.save_for_backward(xf, pre_mix, weight)
            ctx.shape, ctx.eps = x.shape, eps
            return out.view(b, s, d)

        @staticmethod
        def backward(ctx, dout):
            xf, pre_mix, weight = ctx.saved_tensors
            b, s, hc, d = ctx.shape
            t = b * s
            nprog = triton.cdiv(t, BM)
            dres = torch.empty(t, hc * d, dtype=xf.dtype, device=xf.device)
            dpre = torch.empty(t, hc, dtype=torch.float32, device=xf.device)
            dwp = torch.empty(nprog, BD, dtype=torch.float32, device=xf.device)
            _pre_norm_bwd[(nprog,)](xf, pre_mix.reshape(t, hc).contiguous(), weight,
                                    dout.reshape(t, d).contiguous(), dres, dpre, dwp, t,
                                    HC=hc, D=d, BD=BD, BM=BM, EPS=ctx.eps,
                                    num_warps=4, num_stages=2)
            dw = dwp.sum(0)[:d].to(weight.dtype)
            return dres.view(b, s, hc, d), dpre.view(b, s, hc), dw, None

    def hc_pre_norm(x, pre_mix, weight, eps=1e-6):
        return _PreNormFn.apply(x, pre_mix, weight, eps)
else:  # pragma: no cover
    def hc_pre_norm(*a, **k):
        raise RuntimeError("triton 不可用")


def available():
    return HAS_TRITON and torch.cuda.is_available()
