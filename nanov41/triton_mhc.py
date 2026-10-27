"""Triton 融合 mHC hc_post：一个 kernel 算完 out = post_j*x + Σ_i comb[i,j]*residual_i。

## 为什么值得

5090 graph 步的分组件消融（b=8/seq1024，总 88.4 ms）：
  CE+head 25.7 | mHC 20.6 | MoE 18.3 | attention 17.8 | 其它 6.0
mHC 里 `hc_post` 占 **14.8 ms**、`hc_mixes` 12.9 ms。hc_post 的数学很轻（hc=4 的 4×4 组合），
但现在的实现是：`cat` 出 [N,4,5] 与 [N,5,384]，再做 **batch 数 8192、M=4、K=5 的 bmm**
（cuBLAS 在这种形状上效率极低），反向再来一遍 + cat 的 split。

融合 kernel：每行一个 program，d=384 一次进 block（BD=512 掩码），静态展开 hc=4：
  前向：读 residual[4,384] (3 KB) + x[384] + comb[16] + post[4] → 写 out[4,384]
  反向：一次读 dout/residual/x/comb/post → 写 dx、dresidual、dcomb、dpost（都是 per-row，无需 atomics）
预计 14.8 ms → ~1.5 ms（按 57 MB/次 × 24 次 ≈ 1.4 GB 带宽估算）。
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

import os

HC = 4
BD = int(os.environ.get("NANOV41_MHC_BD", "512"))    # >= hidden(384)；512 浪费 25% 带宽，384 也合法
ROWS = int(os.environ.get("NANOV41_MHC_ROWS", "1"))  # 每个 program 处理几行
BD2 = 128     # 反向里 d-block（也用于 out 写回的 block 维度）


if HAS_TRITON:

    @triton.jit
    def _post_fwd_kernel(OUT, X, RES, COMB, POST, N, D,
                         HC: tl.constexpr, BD: tl.constexpr):
        """每行一个 program：out[j,d] = post_j*x[d] + Σ_i comb[i,j]*res[i,d]。"""
        n = tl.program_id(0)
        rd = tl.arange(0, BD)
        mask = rd < D
        x = tl.load(X + n * D + rd, mask=mask, other=0.0).to(tl.float32)
        res_base = RES + n * HC * D
        # 先把 4 条 residual 行读进来（静态展开，L1 命中后续重复读）
        r0 = tl.load(res_base + 0 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(res_base + 1 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(res_base + 2 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(res_base + 3 * D + rd, mask=mask, other=0.0).to(tl.float32)
        out_base = OUT + n * HC * D
        for j in tl.static_range(HC):
            post_j = tl.load(POST + n * HC + j)
            c0 = tl.load(COMB + n * HC * HC + 0 * HC + j)
            c1 = tl.load(COMB + n * HC * HC + 1 * HC + j)
            c2 = tl.load(COMB + n * HC * HC + 2 * HC + j)
            c3 = tl.load(COMB + n * HC * HC + 3 * HC + j)
            v = post_j * x + c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3
            tl.store(out_base + j * D + rd, v.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _post_bwd_kernel(DOUT, X, RES, COMB, POST, DX, DRES, DCOMB, DPOST, N, D,
                         HC: tl.constexpr, BD: tl.constexpr):
        n = tl.program_id(0)
        rd = tl.arange(0, BD)
        mask = rd < D
        x = tl.load(X + n * D + rd, mask=mask, other=0.0).to(tl.float32)
        res_base = RES + n * HC * D
        dout_base = DOUT + n * HC * D
        r0 = tl.load(res_base + 0 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(res_base + 1 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(res_base + 2 * D + rd, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(res_base + 3 * D + rd, mask=mask, other=0.0).to(tl.float32)
        g0 = tl.load(dout_base + 0 * D + rd, mask=mask, other=0.0).to(tl.float32)
        g1 = tl.load(dout_base + 1 * D + rd, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(dout_base + 2 * D + rd, mask=mask, other=0.0).to(tl.float32)
        g3 = tl.load(dout_base + 3 * D + rd, mask=mask, other=0.0).to(tl.float32)
        # dx = Σ_j dout[j]*post_j
        p0 = tl.load(POST + n * HC + 0)
        p1 = tl.load(POST + n * HC + 1)
        p2 = tl.load(POST + n * HC + 2)
        p3 = tl.load(POST + n * HC + 3)
        dx = g0 * p0 + g1 * p1 + g2 * p2 + g3 * p3
        tl.store(DX + n * D + rd, dx.to(DX.dtype.element_ty), mask=mask)
        # dres_i = Σ_j comb[i,j]*dout[j]
        dres_base = DRES + n * HC * D
        for i in tl.static_range(HC):
            c0 = tl.load(COMB + n * HC * HC + i * HC + 0)
            c1 = tl.load(COMB + n * HC * HC + i * HC + 1)
            c2 = tl.load(COMB + n * HC * HC + i * HC + 2)
            c3 = tl.load(COMB + n * HC * HC + i * HC + 3)
            v = c0 * g0 + c1 * g1 + c2 * g2 + c3 * g3
            tl.store(dres_base + i * D + rd, v.to(DRES.dtype.element_ty), mask=mask)
        # dcomb[i,j] = Σ_d dout[j,d]*res[i,d]；dpost_j = Σ_d dout[j,d]*x[d]
        dcomb_base = DCOMB + n * HC * HC
        gs = (g0, g1, g2, g3)
        rs = (r0, r1, r2, r3)
        for i in tl.static_range(HC):
            for j in tl.static_range(HC):
                val = tl.sum(gs[j] * rs[i], axis=0)
                tl.store(dcomb_base + i * HC + j, val.to(DCOMB.dtype.element_ty))
        for j in tl.static_range(HC):
            tl.store(DPOST + n * HC + j, tl.sum(gs[j] * x, axis=0).to(DPOST.dtype.element_ty))


def _fwd(x, residual, post, comb):
    N, hc, D = residual.shape
    assert hc == HC
    out = torch.empty_like(residual)
    _post_fwd_kernel[(N,)](out, x, residual, comb, post, N, D, HC=HC, BD=BD,
                           num_warps=4, num_stages=2)
    return out


class _PostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, post, comb):
        out = _fwd(x, residual, post, comb)
        ctx.save_for_backward(x, residual, post, comb)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, residual, post, comb = ctx.saved_tensors
        N, hc, D = residual.shape
        dx = torch.empty_like(x)
        dres = torch.empty_like(residual)
        dcomb = torch.empty_like(comb)
        dpost = torch.empty_like(post)
        _post_bwd_kernel[(N,)](dout.contiguous(), x, residual, comb, post, dx, dres,
                               dcomb, dpost, N, D, HC=HC, BD=BD,
                               num_warps=4, num_stages=2)
        return dx, dres, dpost, dcomb


def hc_post_triton(x, residual, post, comb):
    """drop-in：x [N,D] bf16、residual [N,hc,D] bf16、post [N,hc] fp32、comb [N,hc,hc] fp32。

    与 ref 语义完全一致（y[...,j,:] = post_j*x + Σ_i comb[i,j]*residual[...,i,:]）。
    注意 ref/fast_mhc 的 post 形状是 [b,s,hc]，这里需要 [N,hc]；调用方负责 view。
    """
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
    out = _PostFn.apply(x, residual, post, comb)
    return out


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
