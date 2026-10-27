"""fused_woa — attention 输出低秩投影 `wo_a` einsum 的替换实现（R47）。

官方 `Attention.forward` 末段：

    o = o.view(bsz, seqlen, n_local_groups, -1)          # [b,s,2,256]
    act_quant(o, ...)                                    # no-op（训练 bf16）
    wo_a = self.wo_a.weight.view(g, o_lora_rank, -1)     # [2,96,256]
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)          # [b,s,2,96]
    x = self.wo_b(o.flatten(2))                          # [b,s,192] @ [192,384]

R47 的 aten 普查发现这个 einsum 的 **fwd 是 1.56 ms eager（12 次/step）**，而我们算它只有 1.61 GFLOP/layer
⇒ cuBLAS 在该 shape 上效率 <1%（torch 把它降成 [g,d,bs]×[g,bs,r] 的怪 bmm）。
本模块用 1 个 Triton kernel 直接算 [g, (b·s), d] × [g, d, r]（M=16384 大 GEMM 的形态），
反传用两个 bmm（形状友好，已是 0.3/0.2 ms 级）。

安装方式：包一层 `torch.einsum`，只拦截 `"bsgd,grd->bsgr"` 这一个 pattern（其它原样透传），
因此不需要复制/改写官方 Attention.forward。

用法：
    import nanov41.fused_woa as fwa
    fwa.install(M)      # 幂等；M 只用来确认是 patched ref 模块
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

__all__ = ["wo_a_forward", "wo_a_ref", "install", "uninstall", "available"]

BM = int(os.environ.get("NANOV41_WOA_BM", "64"))
BK = int(os.environ.get("NANOV41_WOA_BK", "64"))
# R48: wo_a 的 dW 是 K=bs(16384) 的瘦长归约（eager 1.56 ms/step），用 split-K + 定序归约替代 bmm
SPLIT = int(os.environ.get("NANOV41_WOA_SPLIT", "0") or 0)
_INSTALLED = None
_ORIG_EINSUM = None


def wo_a_ref(o, w):
    """官方 einsum 的等价（测试用）。o [b,s,g,d], w [g,r,d] -> [b,s,g,r]。"""
    return torch.einsum("bsgd,grd->bsgr", o, w)


if HAS_TRITON:

    @triton.jit
    def _woa_fwd_kernel(O, W, C, BS_,
                        G: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                        BR: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        gr = tl.program_id(1)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < BS_
        ra = tl.arange(0, BR)
        rm = ra < R
        acc = tl.zeros((BM, BR), dtype=tl.float32)
        for k0 in tl.static_range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(O + rows[:, None] * (G * D) + gr * D + rk[None, :],
                        mask=rmask[:, None], other=0.0)
            # W[gr, r, d] -> tile [BK, BR]：wt[k, r] = W[gr*R*D + r*D + k]
            wt = tl.load(W + gr * (R * D) + ra[None, :] * D + rk[:, None],
                         mask=rm[None, :], other=0.0)
            acc += tl.dot(a, wt)
        tl.store(C + rows[:, None] * (G * R) + gr * R + ra[None, :],
                 acc.to(C.dtype.element_ty), mask=rmask[:, None] & rm[None, :])

    @triton.jit
    def _woa_dw_kernel(PART, DP, OP, BS_,
                       G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                       S: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
                       RT: tl.constexpr, DT: tl.constexpr,
                       BK: tl.constexpr, CHUNK: tl.constexpr):
        """dw[g,r,d] 的第 s 段部分和：Σ_{bs∈chunk} dp[bs,r]·op[bs,d]。"""
        pid = tl.program_id(0)
        gr = tl.program_id(1)
        s = pid // (RT * DT)
        t = pid % (RT * DT)
        rt = t // DT
        dt = t % DT
        rr = rt * BR + tl.arange(0, BR)
        dd = dt * BD + tl.arange(0, BD)
        rm = rr < R
        dm = dd < D
        acc = tl.zeros((BR, BD), dtype=tl.float32)
        kbase = s * CHUNK
        for k0 in range(0, CHUNK, BK):
            rows = kbase + k0 + tl.arange(0, BK)
            rmask = rows < BS_
            a = tl.load(DP + gr * (BS_ * R) + rows[:, None] * R + rr[None, :],
                        mask=rmask[:, None] & rm[None, :], other=0.0)
            b = tl.load(OP + gr * (BS_ * D) + rows[:, None] * D + dd[None, :],
                        mask=rmask[:, None] & dm[None, :], other=0.0)
            acc += tl.dot(tl.trans(a), b)
        tl.store(PART + s * (G * R * D) + gr * (R * D) + rr[:, None] * D + dd[None, :],
                 acc, mask=rm[:, None] & dm[None, :])

    @triton.jit
    def _woa_dw_reduce_kernel(PART, DW, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                              S: tl.constexpr, BD: tl.constexpr):
        gr = tl.program_id(0)
        pid = tl.program_id(1)
        cols = pid * BD + tl.arange(0, BD)
        cm = cols < R * D
        acc = tl.zeros((BD,), dtype=tl.float32)
        for s in range(0, S):
            acc += tl.load(PART + s * (G * R * D) + gr * (R * D) + cols, mask=cm, other=0.0)
        tl.store(DW + gr * (R * D) + cols, acc.to(DW.dtype.element_ty), mask=cm)

    def _launch_dw(dp, op, out):
        # permute(2,0,1,3).reshape(...) 得到的 [G,BS,R] 可能是**非连续视图**（stride 192 等），
        # kernel 假设行连续 → 必须先 contiguous（否则读到错位数据，且 graph/eager parity 抓不到）
        dp = dp.contiguous()
        op = op.contiguous()
        g, bs, r = dp.shape
        d = op.shape[2]
        S = SPLIT
        chunk = (bs + S - 1) // S
        BR = 64 if r % 64 == 0 else triton.next_power_of_2(r)
        BD = 64
        rt = triton.cdiv(r, BR)
        dt = triton.cdiv(d, BD)
        part = torch.empty(S, g, r, d, dtype=torch.float32, device=dp.device)
        _woa_dw_kernel[(S * rt * dt, g)](
            part, dp, op, bs, G=g, R=r, D=d, S=S, BR=BR, BD=BD, RT=rt, DT=dt,
            BK=64, CHUNK=chunk, num_warps=4, num_stages=2)
        out3 = torch.empty(g, r, d, dtype=out.dtype, device=out.device)
        _woa_dw_reduce_kernel[(g, triton.cdiv(r * d, 1024))](
            part, out3, G=g, R=r, D=d, S=S, BD=1024, num_warps=4)
        return out3

    def _launch_fwd(o, w, out):
        b, s, g, d = o.shape
        r = w.shape[1]
        bs = b * s
        _woa_fwd_kernel[(triton.cdiv(bs, BM), g)](
            o, w, out, bs, G=g, D=d, R=r, BR=triton.next_power_of_2(r),
            BM=BM, BK=BK, num_warps=4, num_stages=2)


class _WoAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, w):
        b, s, g, d = o.shape
        r = w.shape[1]
        out = torch.empty(b, s, g, r, dtype=o.dtype, device=o.device)
        _launch_fwd(o, w, out)
        ctx.save_for_backward(o, w)
        return out

    @staticmethod
    def backward(ctx, dy):
        o, w = ctx.saved_tensors
        b, s, g, d = o.shape
        r = w.shape[1]
        # [g, bs, *] 视图（permute 后 reshape 会拷一次，量很小）
        op = o.permute(2, 0, 1, 3).reshape(g, b * s, d)
        dp = dy.permute(2, 0, 1, 3).reshape(g, b * s, r)
        do3 = torch.bmm(dp, w)                                   # [g, bs, d]
        if SPLIT > 1:
            dw = _launch_dw(dp, op, torch.empty(g, r, d, dtype=o.dtype, device=o.device))
        else:
            dw = torch.bmm(dp.transpose(1, 2), op)               # [g, r, d]
        do = do3.view(g, b, s, d).permute(1, 2, 0, 3).contiguous()
        return do, dw


def wo_a_forward(o, w):
    # 只在 bf16+CUDA 上走 Triton（fp32 输入时 Triton 的 tl.dot 默认走 tf32，精度不够；
    # float32 构建（parity harness）应当继续走官方 einsum）
    if not HAS_TRITON or not o.is_cuda or o.dtype != torch.bfloat16:
        return wo_a_ref(o, w)
    if not o.is_contiguous():
        o = o.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    return _WoAFn.apply(o, w)


def _einsum_patched(eq, *ops, **kw):
    if (eq == "bsgd,grd->bsgr" and len(ops) == 2 and ops[0].is_cuda
            and ops[0].dim() == 4 and ops[0].dtype == torch.bfloat16):
        return wo_a_forward(ops[0], ops[1])
    return _ORIG_EINSUM(eq, *ops, **kw)


def install(M=None):
    """安装 einsum 拦截（只拦 wo_a 的 pattern）。"""
    global _INSTALLED, _ORIG_EINSUM
    if _INSTALLED is not None:
        return _INSTALLED
    if _ORIG_EINSUM is None:
        _ORIG_EINSUM = torch.einsum
    torch.einsum = _einsum_patched
    _INSTALLED = _ORIG_EINSUM
    return _INSTALLED


def uninstall(M=None):
    global _INSTALLED
    if _INSTALLED is None:
        return
    torch.einsum = _INSTALLED
    _INSTALLED = None


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()
