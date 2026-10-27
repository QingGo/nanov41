"""官方 mHC 训练链路的 5090 落地方案（按 TileKernels 官方测试的权威用法）。

## 官方训练配方（`tests/mhc/test_pre_big_fuse.py::big_fuse_reference`）
    1) prenorm GEMM（线上是 DeepGEMM，split-K 出 partials：`gemm_out_mul [n_splits,T,1,24]` +
       `gemm_out_sqrsum [n_splits,T,1]`）
    2) `mhc_reduce_partials_and_rmsnorm_fwd(...)` → `mixes [T,24]`（reduce + RMSNorm）
    3) `mhc_pre_split_mixes_fwd(...)` → `pre [T,4]`,`post [T,4]`,`comb_mix [T,16]`
    4) `mhc_sinkhorn_fwd(comb_mix.view(T,4,4), comb_norm, repeat, eps)`
    5) `mhc_pre_apply_mix_fwd(residual[T,4,d], pre, layer_input[T,d])`
反向官方同样给全了：`mhc_pre_apply_mix_bwd` / `mhc_sinkhorn_bwd` / `mhc_pre_split_mixes_bwd` /
`mhc_reduce_partials_and_rmsnorm_bwd`，只有 GEMM 的 dw/dx 需要我们自己算（DeepGEMM 在 sm120 上没有 kernel）。

## 5090 落地的关键发现
官方 kernel 支持 **`n_splits=1`**（Ascend 路径就是单 partial）。⇒ 在 5090 上可以：
    `gemm_out_mul = F.linear(x_float, fn)`、`gemm_out_sqrsum = x.square().sum(-1)`
用 cuBLAS fp32 GEMM 直接产出 partials，**不需要 DeepGEMM、也不需要自己写 split-K kernel**，
其余 4 步全部用官方 TileLang kernel（fwd+bwd 都有）。

本模块先实现 **forward**（本轮验证数值与速度）；backward 下一步接上。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_TK = os.environ.get("TILE_KERNELS_PATH", "")
if os.path.isdir(_TK) and _TK not in sys.path:
    sys.path.insert(0, _TK)

_AVAILABLE = False
try:  # pragma: no cover - 依赖远端 TileKernels/tilelang
    from tile_kernels.mhc.norm_fn_kernel import (
        mhc_reduce_partials_and_rmsnorm_bwd,
        mhc_reduce_partials_and_rmsnorm_fwd,
    )
    from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_bwd, mhc_pre_apply_mix_fwd
    from tile_kernels.mhc.pre_split_mixes_kernel import (
        mhc_pre_split_mixes_bwd,
        mhc_pre_split_mixes_fwd,
    )
    from tile_kernels.mhc.sinkhorn_kernel import mhc_sinkhorn_bwd, mhc_sinkhorn_fwd
    _AVAILABLE = True
except Exception:  # noqa: BLE001
    pass


def available() -> bool:
    return _AVAILABLE


def pre_forward(residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
                base: torch.Tensor, *, sinkhorn_iters: int, eps: float,
                rms_eps: float = 1e-6, post_mult: float = 2.0):
    """residual [b,s,hc,d] bf16；fn [24, hc*d] fp32 → (layer_input, pre, post, comb_norm)。"""
    assert _AVAILABLE, "TileKernels 不可用"
    b, s, hc, d = residual.shape
    t = b * s
    g = hc * d
    x = residual.reshape(t, g).float()
    out_mul = F.linear(x, fn)                              # [T,24] fp32（cuBLAS）
    sqrsum = x.square().sum(-1, keepdim=True)              # [T,1] fp32
    om = out_mul.view(1, t, 1, out_mul.size(-1))
    ss = sqrsum.view(1, t, 1)
    om2 = torch.empty(t, 1, out_mul.size(-1), dtype=torch.float32, device=x.device)
    ss2 = torch.empty(t, 1, dtype=torch.float32, device=x.device)
    mixes = torch.empty(t, out_mul.size(-1), dtype=torch.float32, device=x.device)
    mhc_reduce_partials_and_rmsnorm_fwd(om, ss, om2, ss2, mixes,
                                       rms_group_size=g, rms_eps=rms_eps, n_splits=1)
    pre = torch.empty(t, hc, dtype=torch.float32, device=x.device)
    post = torch.empty(t, hc, dtype=torch.float32, device=x.device)
    comb = torch.empty(t, hc * hc, dtype=torch.float32, device=x.device)
    mhc_pre_split_mixes_fwd(mixes, scale, base, pre, post, comb, post_mult, eps)
    comb_norm = torch.empty(t, hc, hc, dtype=torch.float32, device=x.device)
    mhc_sinkhorn_fwd(comb.view(t, hc, hc), comb_norm, sinkhorn_iters, eps)
    li = torch.empty(t, d, dtype=torch.bfloat16, device=x.device)
    mhc_pre_apply_mix_fwd(residual.view(t, hc, d), pre, li)
    return (li.view(b, s, d), pre.view(b, s, hc), post.view(b, s, hc),
            comb_norm.view(b, s, hc, hc))


class _OfficialMhcPre(torch.autograd.Function):
    """官方 mHC 训练链（fwd 用官方 4 个 kernel + cuBLAS partials；bwd 用官方 4 个 bwd kernel）。"""

    @staticmethod
    def forward(ctx, residual, fn, scale, base, sinkhorn_iters, eps, rms_eps, post_mult):
        b, s, hc, d = residual.shape
        t = b * s
        g = hc * d
        x = residual.reshape(t, g).float()
        out_mul = F.linear(x, fn)                                  # cuBLAS fp32
        sqrsum = x.square().sum(-1, keepdim=True)
        om = out_mul.view(1, t, 1, out_mul.size(-1))
        ss = sqrsum.view(1, t, 1)
        om2 = torch.empty(t, 1, out_mul.size(-1), dtype=torch.float32, device=x.device)
        ss2 = torch.empty(t, 1, dtype=torch.float32, device=x.device)
        mixes = torch.empty(t, out_mul.size(-1), dtype=torch.float32, device=x.device)
        mhc_reduce_partials_and_rmsnorm_fwd(om, ss, om2, ss2, mixes,
                                           rms_group_size=g, rms_eps=rms_eps, n_splits=1)
        pre = torch.empty(t, hc, dtype=torch.float32, device=x.device)
        post = torch.empty(t, hc, dtype=torch.float32, device=x.device)
        comb = torch.empty(t, hc * hc, dtype=torch.float32, device=x.device)
        mhc_pre_split_mixes_fwd(mixes, scale, base, pre, post, comb, post_mult, eps)
        comb_norm = torch.empty(t, hc, hc, dtype=torch.float32, device=x.device)
        mhc_sinkhorn_fwd(comb.view(t, hc, hc), comb_norm, sinkhorn_iters, eps)
        li = torch.empty(t, d, dtype=torch.bfloat16, device=x.device)
        mhc_pre_apply_mix_fwd(residual.reshape(t, hc, d), pre, li)
        ctx.save_for_backward(residual, fn, scale, base, out_mul, sqrsum, mixes, pre, post, comb)
        ctx.cfg = (sinkhorn_iters, eps, rms_eps, post_mult)
        return li.view(b, s, d), pre.view(b, s, hc), post.view(b, s, hc), comb_norm.view(b, s, hc, hc)

    @staticmethod
    def backward(ctx, g_li, g_pre, g_post, g_comb_norm):
        residual, fn, scale, base, out_mul, sqrsum, mixes, pre, post, comb = ctx.saved_tensors
        iters, eps, rms_eps, post_mult = ctx.cfg
        b, s, hc, d = residual.shape
        t, g = b * s, hc * d
        # 1) apply_mix bwd（只剩 residual 梯度；pre_mix 梯度用 torch 算）
        x_grad = torch.zeros(t, hc, d, dtype=residual.dtype, device=residual.device)
        mhc_pre_apply_mix_bwd(g_li.reshape(t, d).contiguous(), residual.reshape(t, hc, d),
                              pre, x_grad)
        pre_grad = (g_li.reshape(t, d).float().unsqueeze(1) *
                    residual.reshape(t, hc, d).float()).sum(-1)
        # 2) sinkhorn bwd
        comb_grad = torch.zeros(t, hc, hc, dtype=torch.float32, device=residual.device)
        mhc_sinkhorn_bwd(g_comb_norm.reshape(t, hc, hc).contiguous(),
                         comb.view(t, hc, hc), comb_grad, iters, eps)
        # 3) split_mixes bwd
        mixes_grad = torch.zeros_like(mixes)
        scale_grad = torch.zeros_like(scale)
        base_grad = torch.zeros_like(base)
        mhc_pre_split_mixes_bwd(pre_grad.contiguous(),
                                g_post.reshape(t, hc).contiguous().float(),
                                comb_grad.view(t, hc * hc),
                                mixes, post, scale, base,
                                mixes_grad, scale_grad, base_grad, post_mult)
        # 4) reduce_partials_and_rmsnorm bwd
        mix_n = out_mul.size(-1)
        om_grad = torch.zeros(t, 1, mix_n, dtype=torch.float32, device=residual.device)
        ss_grad = torch.zeros(t, 1, dtype=torch.float32, device=residual.device)
        mhc_reduce_partials_and_rmsnorm_bwd(
            mixes_grad, out_mul.view(t, 1, mix_n), sqrsum.view(t, 1),
            om_grad, ss_grad, rms_group_size=g, rms_eps=rms_eps)
        # 5) GEMM 的 dx / dW（cuBLAS）
        x = residual.reshape(t, g).float()
        omg = om_grad.view(t, mix_n)                                   # [T,24]
        d_fn = omg.t() @ x
        dx = omg @ fn + 2.0 * ss_grad.view(t, 1) * x
        if os.environ.get("MHC_DEBUG", "0").lower() not in ("0", "false", "no", "off"):
            print(f"[mhc bwd] x_grad max {x_grad.float().abs().max().item():.3e} | "
                  f"omg max {omg.abs().max().item():.3e} | ss_grad max {ss_grad.abs().max().item():.3e} | "
                  f"dx max {(omg @ fn).abs().max().item():.3e} | "
                  f"x max {x.abs().max().item():.3e} | mixes_grad max {mixes_grad.abs().max().item():.3e}", flush=True)
        dx = dx.view(t, hc, d) + x_grad.float()
        if os.environ.get("MHC_DEBUG", "0").lower() not in ("0", "false", "no", "off"):
            print(f"[mhc bwd] final dx max {dx.abs().max().item():.3e}", flush=True)
        out_grad_res = dx.view(b, s, hc, d).to(residual.dtype)
        if os.environ.get("MHC_DEBUG", "0").lower() not in ("0", "false", "no", "off"):
            print(f"[mhc bwd] returned dres max {out_grad_res.float().abs().max().item():.3e} "
                  f"contig {out_grad_res.is_contiguous()} dtype {out_grad_res.dtype}", flush=True)
        return out_grad_res, d_fn.to(fn.dtype), scale_grad, base_grad, None, None, None, None


def pre_forward_bwd(residual, fn, scale, base, *, sinkhorn_iters, eps, rms_eps=1e-6, post_mult=2.0):
    return _OfficialMhcPre.apply(residual, fn, scale, base, sinkhorn_iters, eps, rms_eps, post_mult)


# ---------------------------------------------------------------------------
# 拆成两段（与 ref 模型的结构一致）：mixes（pre/post/comb） 与 apply_mix（collapse）
# ref 的 Block.forward 里，hc_pre 用的 pre_mix 是**上一层 hc_mixes** 的输出，
# 因此训练链路必须拆开（不能用"同一次 mixes 的 pre"去 apply）。
class _OfficialMhcMixes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, fn, scale, base, sinkhorn_iters, eps, rms_eps, post_mult):
        b, s, hc, d = residual.shape
        t, g = b * s, hc * d
        x = residual.reshape(t, g).float()
        out_mul = F.linear(x, fn)
        sqrsum = x.square().sum(-1, keepdim=True)
        om = out_mul.view(1, t, 1, out_mul.size(-1))
        ss = sqrsum.view(1, t, 1)
        om2 = torch.zeros(t, 1, out_mul.size(-1), dtype=torch.float32, device=x.device)
        ss2 = torch.zeros(t, 1, dtype=torch.float32, device=x.device)
        mixes = torch.zeros(t, out_mul.size(-1), dtype=torch.float32, device=x.device)
        mhc_reduce_partials_and_rmsnorm_fwd(om, ss, om2, ss2, mixes,
                                           rms_group_size=g, rms_eps=rms_eps, n_splits=1)
        pre = torch.zeros(t, hc, dtype=torch.float32, device=x.device)
        post = torch.zeros(t, hc, dtype=torch.float32, device=x.device)
        comb = torch.zeros(t, hc * hc, dtype=torch.float32, device=x.device)
        mhc_pre_split_mixes_fwd(mixes, scale, base, pre, post, comb, post_mult, eps)
        comb_norm = torch.zeros(t, hc, hc, dtype=torch.float32, device=x.device)
        mhc_sinkhorn_fwd(comb.view(t, hc, hc), comb_norm, sinkhorn_iters, eps)
        ctx.save_for_backward(residual, fn, scale, base, out_mul, sqrsum, mixes, pre, post, comb)
        ctx.cfg = (sinkhorn_iters, eps, rms_eps, post_mult)
        return pre.view(b, s, hc), post.view(b, s, hc), comb_norm.view(b, s, hc, hc)

    @staticmethod
    def backward(ctx, g_pre, g_post, g_comb_norm):
        residual, fn, scale, base, out_mul, sqrsum, mixes, pre, post, comb = ctx.saved_tensors
        iters, eps, rms_eps, post_mult = ctx.cfg
        b, s, hc, d = residual.shape
        t, g = b * s, hc * d
        comb_grad = torch.zeros(t, hc, hc, dtype=torch.float32, device=residual.device)
        mhc_sinkhorn_bwd(g_comb_norm.reshape(t, hc, hc).contiguous(), comb.view(t, hc, hc),
                         comb_grad, iters, eps)
        mixes_grad = torch.zeros_like(mixes)
        scale_grad = torch.zeros_like(scale)
        base_grad = torch.zeros_like(base)
        mhc_pre_split_mixes_bwd(g_pre.reshape(t, hc).contiguous().float(),
                                g_post.reshape(t, hc).contiguous().float(),
                                comb_grad.view(t, hc * hc), mixes, post, scale, base,
                                mixes_grad, scale_grad, base_grad, post_mult)
        mix_n = out_mul.size(-1)
        om_grad = torch.zeros(t, 1, mix_n, dtype=torch.float32, device=residual.device)
        ss_grad = torch.zeros(t, 1, dtype=torch.float32, device=residual.device)
        mhc_reduce_partials_and_rmsnorm_bwd(mixes_grad, out_mul.view(t, 1, mix_n),
                                           sqrsum.view(t, 1), om_grad, ss_grad,
                                           rms_group_size=g, rms_eps=rms_eps)
        x = residual.reshape(t, g).float()
        omg = om_grad.view(t, mix_n)
        d_fn = omg.t() @ x
        dx = (omg @ fn + 2.0 * ss_grad.view(t, 1) * x).view(b, s, hc, d)
        return dx.to(residual.dtype), d_fn.to(fn.dtype), scale_grad, base_grad, None, None, None, None


class _OfficialMhcApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, pre):
        b, s, hc, d = residual.shape
        t = b * s
        pre2 = pre.reshape(t, hc).float().contiguous()
        li = torch.zeros(t, d, dtype=torch.bfloat16, device=residual.device)
        mhc_pre_apply_mix_fwd(residual.reshape(t, hc, d), pre2, li)
        ctx.save_for_backward(residual, pre)
        return li.view(b, s, d)

    @staticmethod
    def backward(ctx, g_li):
        residual, pre = ctx.saved_tensors
        b, s, hc, d = residual.shape
        t = b * s
        x_grad = torch.zeros(t, hc, d, dtype=residual.dtype, device=residual.device)
        mhc_pre_apply_mix_bwd(g_li.reshape(t, d).contiguous(),
                              residual.reshape(t, hc, d), pre.reshape(t, hc).float().contiguous(),
                              x_grad)
        pre_grad = (g_li.reshape(t, d).float().unsqueeze(1) *
                    residual.reshape(t, hc, d).float()).sum(-1).view(b, s, hc).to(pre.dtype)
        return x_grad.view(b, s, hc, d), pre_grad


def mixes_official(residual, fn, scale, base, *, sinkhorn_iters, eps, rms_eps=1e-6, post_mult=2.0):
    return _OfficialMhcMixes.apply(residual, fn, scale, base, sinkhorn_iters, eps, rms_eps, post_mult)


def apply_mix_official(residual, pre):
    return _OfficialMhcApply.apply(residual, pre)
