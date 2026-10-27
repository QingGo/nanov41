"""mHC split 的 Triton 融合 kernel：一次算出 (pre, post, comb_mix_raw)，并带解析反向。

## 为什么（测量驱动 + 官方优先的边界）

5090、b=16 的 mHC 子项消融：**hc_mixes = 15.0 ms/步**、hc_pre 2.5、hc_post 4.5、sinkhorn 0.4。
hc_mixes 里 "norm + bf16 GEMM + rsqrt 乘 + split（sigmoid/softmax/affine）" 是 8–10 个极小 torch 算子 ×
24 次/步，属于典型的 launch/occupancy bound。

官方 `mhc_pre_big_fuse_fwd` 把这一段全融了，但**它没有训练用的 backward**（官方只有未融合各段的 bwd：
slice/split/sinkhorn/apply_mix），而我们已实测"官方未融合链路 + torch split-K GEMM"并不比现有实现快
（S16 §24）。因此这里**只把 split 段自研成融合 kernel**（fwd 1 个 + bwd 1 个），sinkhorn 仍用官方
TileLang kernel（`tl_sinkhorn.sinkhorn_tl`，含解析反向）——这是"官方优先"原则下最小必要的自研。

## 语义（必须与 ref `hc_split_sinkhorn` 的前半段逐位一致）

    pre  = sigmoid(m[:hc] * scale0 + base[:hc]) + eps
    post = 2 * sigmoid(m[hc:2hc] * scale1 + base[hc:2hc])
    comb_mix = m[2hc:] * scale2 + base[2hc:]        # 未 softmax；官方 sinkhorn kernel 内部先 softmax

反向（给定 grad_pre / grad_post / grad_comb_mix）：
    dz0 = g_pre * σ0(1-σ0)，σ0 = pre - eps
    dz1 = g_post * σ1(1-σ1)，σ1 = post/2      （post = 2σ1 ⇒ dpost/dz1 = 2σ1(1-σ1)）
    dz2 = g_comb
    dm[c] = dz[c] * scale[group(c)]；dbase[c] = dz[c]；dscale[g] = Σ_rows dz[c]·m[c]
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

HC = 4
MIX = 24          # 2*hc + hc*hc（pre 4 + post 4 + comb 16）
BM = 128


if HAS_TRITON:

    @triton.jit
    def _split_fwd(MIXES, SCALE, BASE, PRE, POST, COMB, T,
                   HC: tl.constexpr, MIX: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        mask = rows < T
        rc = tl.arange(0, HC)
        rz = tl.arange(0, HC * HC)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        # pre
        m0 = tl.load(MIXES + rows[:, None] * MIX + rc[None, :], mask=mask[:, None], other=0.0)
        b0 = tl.load(BASE + rc)
        z0 = m0 * s0 + b0[None, :]
        pre = tl.sigmoid(z0) + EPS
        tl.store(PRE + rows[:, None] * HC + rc[None, :], pre, mask=mask[:, None])
        # post
        m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rc)[None, :], mask=mask[:, None],
                     other=0.0)
        b1 = tl.load(BASE + HC + rc)
        z1 = m1 * s1 + b1[None, :]
        post = 2.0 * tl.sigmoid(z1)
        tl.store(POST + rows[:, None] * HC + rc[None, :], post, mask=mask[:, None])
        # comb_mix（未 softmax）
        m2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + rz)[None, :], mask=mask[:, None],
                     other=0.0)
        b2 = tl.load(BASE + 2 * HC + rz)
        comb = m2 * s2 + b2[None, :]
        tl.store(COMB + rows[:, None] * (HC * HC) + rz[None, :], comb, mask=mask[:, None])

    @triton.jit
    def _split_bwd(MIXES, SCALE, PRE, POST, GPRE, GPOST, GCOMB,
                   DMIX, DB_PART, DS_PART, T,
                   HC: tl.constexpr, MIX: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        mask = rows < T
        rc = tl.arange(0, HC)
        rz = tl.arange(0, HC * HC)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        m0 = tl.load(MIXES + rows[:, None] * MIX + rc[None, :], mask=mask[:, None], other=0.0)
        m1 = tl.load(MIXES + rows[:, None] * MIX + (HC + rc)[None, :], mask=mask[:, None],
                     other=0.0)
        m2 = tl.load(MIXES + rows[:, None] * MIX + (2 * HC + rz)[None, :], mask=mask[:, None],
                     other=0.0)
        gpre = tl.load(GPRE + rows[:, None] * HC + rc[None, :], mask=mask[:, None], other=0.0)
        pre = tl.load(PRE + rows[:, None] * HC + rc[None, :], mask=mask[:, None], other=0.0)
        gpost = tl.load(GPOST + rows[:, None] * HC + rc[None, :], mask=mask[:, None], other=0.0)
        post = tl.load(POST + rows[:, None] * HC + rc[None, :], mask=mask[:, None], other=0.0)
        gcomb = tl.load(GCOMB + rows[:, None] * (HC * HC) + rz[None, :], mask=mask[:, None],
                        other=0.0)
        sig0 = pre - EPS
        dz0 = gpre * sig0 * (1.0 - sig0)                 # dL/dz0
        sig1 = post * 0.5
        dz1 = gpost * 2.0 * sig1 * (1.0 - sig1)          # dL/dz1
        dz2 = gcomb                                       # dL/dz2
        tl.store(DMIX + rows[:, None] * MIX + rc[None, :], m0 * 0.0 + dz0 * s0,
                 mask=mask[:, None])
        tl.store(DMIX + rows[:, None] * MIX + (HC + rc)[None, :], dz1 * s1, mask=mask[:, None])
        tl.store(DMIX + rows[:, None] * MIX + (2 * HC + rz)[None, :], dz2 * s2,
                 mask=mask[:, None])
        # 每 program 的部分和（外层 torch 再 sum ⇒ 确定性）
        db0 = tl.sum(dz0, axis=0)                         # [HC]
        db1 = tl.sum(dz1, axis=0)
        db2 = tl.sum(dz2, axis=0)                         # [HC*HC]
        tl.store(DB_PART + pid * MIX + rc, db0)
        tl.store(DB_PART + pid * MIX + HC + rc, db1)
        tl.store(DB_PART + pid * MIX + 2 * HC + rz, db2)
        ds0 = tl.sum(dz0 * m0)
        ds1 = tl.sum(dz1 * m1)
        ds2 = tl.sum(dz2 * m2)
        # arange 长度必须 2 的幂：用 4 槽、只存前 3 个
        tile = tl.arange(0, 4)
        ds = tl.where(tile == 0, ds0, tl.where(tile == 1, ds1, tl.where(tile == 2, ds2, 0.0)))
        tl.store(DS_PART + pid * 3 + tile, ds, mask=tile < 3)

    def split_fwd(mixes, scale, base, hc=HC, eps=1e-6):
        m = mixes.reshape(-1, mixes.size(-1)).contiguous()
        t = m.size(0)
        pre = torch.empty(t, hc, dtype=torch.float32, device=m.device)
        post = torch.empty_like(pre)
        comb = torch.empty(t, hc * hc, dtype=torch.float32, device=m.device)
        _split_fwd[(triton.cdiv(t, BM),)](m, scale, base, pre, post, comb, t,
                                          HC=hc, MIX=2 * hc + hc * hc, BM=BM, EPS=eps, num_warps=4)
        return pre, post, comb, m

    def split_bwd(m, scale, pre, post, gpre, gpost, gcomb, hc=HC, eps=1e-6):
        t = m.size(0)
        np_ = triton.cdiv(t, BM)
        dmix = torch.empty_like(m)
        db = torch.empty(np_, 2 * hc + hc * hc, dtype=torch.float32, device=m.device)
        ds = torch.empty(np_, 3, dtype=torch.float32, device=m.device)
        _split_bwd[(np_,)](m, scale, pre, post, gpre, gpost, gcomb, dmix, db, ds, t,
                           HC=hc, MIX=2 * hc + hc * hc, BM=BM, EPS=eps, num_warps=4)
        return dmix, db.sum(0), ds.sum(0)
else:  # pragma: no cover
    def split_fwd(*a, **k):
        raise RuntimeError("triton 不可用")

    def split_bwd(*a, **k):
        raise RuntimeError("triton 不可用")


class _SplitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, hc, eps):
        b, s_, _ = mixes.shape
        pre, post, comb, m = split_fwd(mixes, scale, base, hc, eps)
        ctx.save_for_backward(m, scale, pre, post)
        ctx.hc, ctx.eps, ctx.shape = hc, eps, mixes.shape
        return (pre.view(b, s_, hc), post.view(b, s_, hc),
                comb.view(b, s_, hc, hc))

    @staticmethod
    def backward(ctx, gpre, gpost, gcomb):
        m, scale, pre, post = ctx.saved_tensors
        dmix, dbase, dscale = split_bwd(m, scale, pre, post,
                                        gpre.reshape(-1, ctx.hc).contiguous(),
                                        gpost.reshape(-1, ctx.hc).contiguous(),
                                        gcomb.reshape(-1, ctx.hc * ctx.hc).contiguous(),
                                        ctx.hc, ctx.eps)
        return dmix.view(ctx.shape), dscale, dbase, None, None


def hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """与 `tl_sinkhorn.hc_split_sinkhorn_tl` 同语义；split 段用 Triton 融合 kernel，sinkhorn 仍用官方。"""
    from .tl_sinkhorn import sinkhorn_tl

    b, s, _ = mixes.size()
    hc = hc_mult
    pre, post, comb_mix = _SplitFn.apply(mixes.float(), hc_scale, hc_base, hc, eps)
    comb = sinkhorn_tl(comb_mix.reshape(b, s, hc, hc), repeat=sinkhorn_iters, eps=eps)
    return pre, post, comb


def available():
    return HAS_TRITON and torch.cuda.is_available()
