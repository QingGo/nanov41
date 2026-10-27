"""窗口 attention 的 Triton 融合内核（band + sink），前向算 numerator/denominator，反向直出 dq/dkv。

## 为什么

5090 消融：attention 19.3 ms/步（占 5.8% FLOPs）。当前 `fast_attn` 每层用 8 个 query tile、
每 tile ~6 个 torch 算子（bmm → mask → amax → exp → sum → cast → bmm），合计 ~50 个 kernel/层、
外加每 tile 的 dtype cast。图模式下 kernel 数不再有 CPU 成本，但每个 kernel 的 GPU 侧排空/占用率
损失 + 中间张量往返（scores/probs 各 [b,h,1024,128] fp16 = 33 MB）仍然可观。

融合后：**每层 1 个 fwd kernel + 1 个 bwd kernel**，scores/probs 只在寄存器/共享内存里，
不落 HBM；sink 折进同一个 softmax（用 m_safe = max(max s, sink) 的稳定化）。

## 数学（与 fast_attn / ref 一致）

对每个 query 行 i，key 集合 S = 窗口 band 里有效的 key（compress 部分仍由 torch 路径负责）：
    s_j = softmax_scale * q_i · k_j
    m_i = max(max_j s_j, sink_h)            （detach）
    num_i = Σ_j exp(s_j - m_i) v_j
    den_i = Σ_j exp(s_j - m_i)
本模块只输出 (num_i, den_i, m_i)，由调用方与 compress 部分按同一个 softmax 合并
（compress 部分用自己的局部最大值，合并时用 exp 重标定，数学等价）。

反向：给定 d_num, d_den（合并后的梯度），m 视为常数（softmax 平移不变）：
    dP = d_num @ Vᵀ + d_den[:, None]
    dS = P * (dP - Σ_k P·dP)        （逐行）
    dQ = (dS @ K) · scale
    dK = (dSᵀ @ Q) · scale          → scatter-add 到 dfp32 缓冲（窗口行被多个 tile 共享）
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

BM = 64        # query rows / program
BW = 128       # window keys / program（win + BM - 1 ≤ 127）


if HAS_TRITON:

    @triton.jit
    def _win_fwd(Q, KV, SINK, NUM, DEN, MX, B, NB, M, N, WIN,
                 SCALE: tl.constexpr, BM: tl.constexpr, BW: tl.constexpr,
                 D: tl.constexpr, H: tl.constexpr):
        pid_t = tl.program_id(0)        # tile index（全局 b*NB + tile）
        h = tl.program_id(1)
        b = pid_t // NB
        it = pid_t % NB
        i0 = it * BM

        rd = tl.arange(0, D)
        rq = i0 + tl.arange(0, BM)                    # 全局 query 行
        # 窗口 keys：[lo, lo+K_W)，K_W <= WK
        lo = tl.maximum(0, i0 - WIN + 1)
        k_w = tl.minimum(M, i0 + BM) - lo            # 有效窗口 key 数（≤ WIN+BM-1）
        rj = tl.arange(0, BW)                        # 窗口内偏移
        kj = lo + rj                                 # 全局 key 行

        q = tl.load(Q + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                    mask=(rq[:, None] < M), other=0.0).to(tl.float32)          # [BM,D]
        k = tl.load(KV + (b * N + kj[:, None]) * D + rd[None, :],
                    mask=(rj[:, None] < k_w) & (kj[:, None] < M), other=0.0)   # [BW,D] bf16
        s = tl.dot(q.to(tl.bfloat16), tl.trans(k)) * SCALE                     # [BM,BW] fp32

        # band 掩码：key j 对 query i 有效 ⇔ (i - WIN < j <= i) 且 j < M
        valid = (kj[None, :] <= rq[:, None]) & (kj[None, :] > (rq[:, None] - WIN)) \
            & (rj[None, :] < k_w) & (rq[:, None] < M)
        s = tl.where(valid, s, float("-inf"))
        sink = tl.load(SINK + h).to(tl.float32)
        mx = tl.maximum(tl.max(s, axis=1), sink)
        mx_safe = tl.where(mx == float("-inf"), 0.0, mx)
        p = tl.exp(s - mx_safe[:, None])                                      # [BM,BW]
        den = tl.sum(p, axis=1)   # 不含 sink：sink 项在合并处只加一次（见 _FusedSparseAttn）
        num = tl.dot(p.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)      # [BM,D]
        tl.store(NUM + ((b * M + rq[:, None]) * H + h) * D + rd[None, :], num,
                 mask=(rq[:, None] < M))
        tl.store(DEN + (b * M + rq) * H + h, den, mask=(rq < M))
        tl.store(MX + (b * M + rq) * H + h, mx_safe, mask=(rq < M))

    @triton.jit
    def _win_bwd(DOUT, OUT, DEN, Q, KV, MX, SINK, DQ, DKT, BSINKF, B, NB, M, N, WIN,
                 SCALE: tl.constexpr, BM: tl.constexpr, BW: tl.constexpr,
                 D: tl.constexpr, H: tl.constexpr):
        """窗口部分反向：dS = (P/D)·(dout·(v − out))，再出 dq/dk（dk 用原子累加）/dsink。"""
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        b = pid_t // NB
        it = pid_t % NB
        i0 = it * BM
        rd = tl.arange(0, D)
        rq = i0 + tl.arange(0, BM)
        lo = tl.maximum(0, i0 - WIN + 1)
        k_w = tl.minimum(M, i0 + BM) - lo
        rj = tl.arange(0, BW)
        kj = lo + rj
        maskq = rq[:, None] < M
        q = tl.load(Q + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                    mask=maskq, other=0.0)
        k = tl.load(KV + (b * N + kj[:, None]) * D + rd[None, :],
                    mask=(rj[:, None] < k_w) & (kj[:, None] < M), other=0.0)
        mx = tl.load(MX + (b * M + rq) * H + h, mask=(rq < M), other=0.0)
        den = tl.load(DEN + (b * M + rq) * H + h, mask=(rq < M), other=1.0)
        dout = tl.load(DOUT + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                       mask=maskq, other=0.0).to(tl.float32)
        out = tl.load(OUT + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                      mask=maskq, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * SCALE
        valid = (kj[None, :] <= rq[:, None]) & (kj[None, :] > (rq[:, None] - WIN)) \
            & (rj[None, :] < k_w) & maskq
        s = tl.where(valid, s, float("-inf"))
        p = tl.exp(s - mx[:, None])                                   # [BM,BW]
        dot = tl.sum(dout * out, axis=1)                              # [BM]
        dp = tl.dot(dout.to(tl.bfloat16), tl.trans(k)) - dot[:, None]  # dout·(v_j − out)
        ds = p / den[:, None] * dp                                    # [BM,BW]
        dq = tl.dot(ds.to(tl.bfloat16), k) * SCALE
        tl.store(DQ + ((b * M + rq[:, None]) * H + h) * D + rd[None, :], dq, mask=maskq)
        # K 与 V 是同一张量：dk 要加**两条路径** —— score 路径 (scale·dSᵀq) + value 路径 (P/D)
        pn = (p / den[:, None]).to(tl.bfloat16)
        dk = tl.dot(tl.trans(ds).to(tl.bfloat16), q) * SCALE + tl.dot(tl.trans(pn), dout.to(tl.bfloat16))
        # 写 per-tile 缓冲（[b*nb, BW, D] fp32），host 侧再 scatter-add 到 [b,n,d]：
        # 比 2D atomic_add 更稳（实测 tl.atomic_add 的 2D 掩码版本在这里没有写入）
        tl.store(DKT + (pid_t * H + h) * BW * D + rj[:, None] * D + rd[None, :], dk)
        sink = tl.load(SINK + h).to(tl.float32)
        T = tl.exp(sink - mx)                                         # [BM]
        dsink = -tl.sum(dot * T / den, axis=0)
        tl.atomic_add(BSINKF + h, dsink)

    def _grid(b, m):
        return (b * ((m + BM - 1) // BM), )

    def win_fwd(q, kv, sink, m, win, scale):
        b, m, h, d = q.shape
        nb = (m + BM - 1) // BM
        num = torch.empty(b, m, h, d, dtype=torch.float32, device=q.device)
        den = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        mx = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        _win_fwd[(b * nb, h)](q, kv, sink, num, den, mx, b, nb, m, kv.size(1), win,
                              SCALE=scale, BM=BM, BW=BW, D=d, H=h,
                              num_warps=4, num_stages=2)
        return num, den, mx

    def win_bwd(q, kv, sink, mx, dout, out, den, m, win, scale, dkv_f32):
        """返回 (dq, dsink)；dk 通过 scatter-add 累加进传入的 dkv_f32 [b,n,d]。"""
        b, m, h, d = q.shape
        n = kv.size(1)
        nb = (m + BM - 1) // BM
        dq = torch.empty(b, m, h, d, dtype=torch.bfloat16, device=q.device)
        dsink = torch.zeros(h, dtype=torch.float32, device=q.device)
        dkt = torch.zeros(b * nb * h, BW, d, dtype=torch.float32, device=q.device)
        _win_bwd[(b * nb, h)](dout, out, den, q, kv, mx, sink, dq, dkt, dsink,
                              b, nb, m, n, win, SCALE=scale, BM=BM, BW=BW,
                              D=d, H=h, num_warps=4, num_stages=2)
        # per-tile 的 key 行号：lo = max(0, t*BM - win + 1)，kj = lo + [0,BW)
        t = torch.arange(nb, device=q.device)
        lo = (t * BM - win + 1).clamp_min(0)
        kj = (lo.view(1, nb, 1) + torch.arange(BW, device=q.device).view(1, 1, BW))
        valid = kj < m
        idx = (torch.arange(b, device=q.device).view(b, 1, 1) * n + kj) \
            .expand(b, nb, BW).reshape(-1)
        src = dkt.view(b * nb, h, BW, d).sum(1).reshape(-1, d)   # 各 head 的 dk 相加
        dkv_f32.view(b * n, d).index_add_(0, idx[valid.expand(b, nb, BW).reshape(-1)],
                                          src[valid.expand(b, nb, BW).reshape(-1)])
        return dq, dsink

    @triton.jit
    def _win_bwd_nd(DNUM, DDEN, Q, KV, MX, DQ, DKT, B, NB, M, N, WIN,
                    SCALE: tl.constexpr, BM: tl.constexpr, BW: tl.constexpr,
                    D: tl.constexpr, H: tl.constexpr):
        """num/den 接口的反向：P = exp(s - mx)（mx detach），dS = P ⊙ (dnum·v + dden)
        （P 未归一化 ⇒ 无减法项）；dq = scale·dS@k，dk = scale·dSᵀ@q + Pᵀ·dnum（K=V）。"""
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        b = pid_t // NB
        it = pid_t % NB
        i0 = it * BM
        rd = tl.arange(0, D)
        rq = i0 + tl.arange(0, BM)
        lo = tl.maximum(0, i0 - WIN + 1)
        k_w = tl.minimum(M, i0 + BM) - lo
        rj = tl.arange(0, BW)
        kj = lo + rj
        maskq = rq[:, None] < M
        q = tl.load(Q + ((b * M + rq[:, None]) * H + h) * D + rd[None, :], mask=maskq, other=0.0)
        k = tl.load(KV + (b * N + kj[:, None]) * D + rd[None, :],
                    mask=(rj[:, None] < k_w) & (kj[:, None] < M), other=0.0)
        mx = tl.load(MX + (b * M + rq) * H + h, mask=(rq < M), other=0.0)
        dnum = tl.load(DNUM + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                       mask=maskq, other=0.0).to(tl.float32)
        dden = tl.load(DDEN + (b * M + rq) * H + h, mask=(rq < M), other=0.0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        valid = (kj[None, :] <= rq[:, None]) & (kj[None, :] > (rq[:, None] - WIN)) \
            & (rj[None, :] < k_w) & maskq
        s = tl.where(valid, s, float("-inf"))
        p = tl.exp(s - mx[:, None])
        dp = tl.dot(dnum.to(tl.bfloat16), tl.trans(k)) + dden[:, None]
        ds = p * dp
        dq = tl.dot(ds.to(tl.bfloat16), k) * SCALE
        tl.store(DQ + ((b * M + rq[:, None]) * H + h) * D + rd[None, :], dq, mask=maskq)
        pn = p.to(tl.bfloat16)
        dk = (tl.dot(tl.trans(ds).to(tl.bfloat16), q) * SCALE
              + tl.dot(tl.trans(pn), dnum.to(tl.bfloat16)))
        tl.store(DKT + (pid_t * H + h) * BW * D + rj[:, None] * D + rd[None, :], dk)

else:  # pragma: no cover
    def _win_fwd(*a, **k):
        raise RuntimeError("triton 不可用")

    def _win_bwd(*a, **k):
        raise RuntimeError("triton 不可用")

    def _win_bwd_nd(*a, **k):
        raise RuntimeError("triton 不可用")


# --------------------------------------------------------------------------- host
def _grid(b, m):
    return (b * ((m + BM - 1) // BM),)


def win_fwd(q, kv, sink, m, win, scale):
    b, m, h, d = q.shape
    nb = (m + BM - 1) // BM
    num = torch.empty(b, m, h, d, dtype=torch.float32, device=q.device)
    den = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
    mx = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
    _win_fwd[(b * nb, h)](q, kv, sink, num, den, mx, b, nb, m, kv.size(1), win,
                          SCALE=scale, BM=BM, BW=BW, D=d, H=h, num_warps=4, num_stages=2)
    return num, den, mx


def _scatter_dk(dkt, b, nb, h, n, m, win, d):
    t = torch.arange(nb, device=dkt.device)
    lo = (t * BM - win + 1).clamp_min(0)
    kj = (lo.view(1, nb, 1) + torch.arange(BW, device=dkt.device).view(1, 1, BW))
    valid = (kj < m).expand(b, nb, BW)
    idx = (torch.arange(b, device=dkt.device).view(b, 1, 1) * n + kj).expand(b, nb, BW).reshape(-1)
    src = dkt.view(b * nb, h, BW, d).sum(1).reshape(-1, d)
    mflat = valid.reshape(-1)
    dkv = torch.zeros(b * n, d, dtype=torch.float32, device=dkt.device)
    dkv.index_add_(0, idx[mflat], src[mflat])
    return dkv.view(b, n, d)


def win_bwd_nd(q, kv, mx, dnum, dden, m, win, scale):
    b, _, h, d = q.shape
    n = kv.size(1)
    nb = (m + BM - 1) // BM
    dq = torch.empty(b, m, h, d, dtype=torch.bfloat16, device=q.device)
    dkt = torch.zeros(b * nb * h, BW, d, dtype=torch.float32, device=q.device)
    _win_bwd_nd[(b * nb, h)](dnum, dden, q, kv, mx, dq, dkt, b, nb, m, n, win,
                             SCALE=scale, BM=BM, BW=BW, D=d, H=h,
                             num_warps=4, num_stages=2)
    return dq, _scatter_dk(dkt, b, nb, h, n, m, win, d)


class _WinFn(torch.autograd.Function):
    """窗口 + sink 的 num/den 接口（mx 视为常数；softmax 平移不变性由外层合并保证）。"""

    @staticmethod
    def forward(ctx, q, kv, sink, m, win, scale):
        num, den, mx = win_fwd(q, kv, sink, m, win, scale)
        ctx.save_for_backward(q, kv, mx)
        ctx.m, ctx.win, ctx.scale = m, win, scale
        return num, den, mx

    @staticmethod
    def backward(ctx, dnum, dden, dmx):
        q, kv, mx = ctx.saved_tensors
        dq, dkv = win_bwd_nd(q, kv, mx, dnum.contiguous(), dden.contiguous(),
                             ctx.m, ctx.win, ctx.scale)
        return dq, dkv, None, None, None, None


def sparse_attn_fused(q, kv, attn_sink, topk_idxs, softmax_scale, win, ncomp):
    """window 用 Triton 融合 kernel（num/den 接口），compress 与合并用 torch（可求导）。"""
    b, m, h, d = q.shape
    n = kv.size(1)
    dev = q.device
    q = q.contiguous()
    num_w, den_w, mx_w = _WinFn.apply(q, kv, attn_sink, m, win, softmax_scale)
    mx_w = mx_w.detach()
    if ncomp:
        comp = topk_idxs[:, :, win:].clamp_min(0).to(torch.int64)
        valid = topk_idxs[:, :, win:] >= 0
        tok = (torch.arange(b, device=dev).view(b, 1, 1) * n + comp).reshape(-1)
        kvc = kv.reshape(b * n, d).index_select(0, tok).reshape(b, m, ncomp, d)
        s_c = torch.einsum("bmhd,bmcd->bmhc", q, kvc) * softmax_scale
        s_c = s_c.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx_c = s_c.amax(-1)
        p_c = (s_c - mx_c.unsqueeze(-1)).exp()
        den_c = p_c.sum(-1)
        num_c = torch.einsum("bmhc,bmcd->bmhd", p_c.to(kvc.dtype), kvc)
    else:
        mx_c = torch.full((b, m, h), float("-inf"), device=dev)
        den_c = torch.zeros(b, m, h, device=dev)
        num_c = torch.zeros(b, m, h, d, device=dev)
    sink = attn_sink.to(torch.float32).view(1, 1, h)
    with torch.no_grad():
        mx = torch.maximum(torch.maximum(mx_w, mx_c), sink)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    sw = (mx_w - mx).exp()
    sc = (mx_c - mx).exp()
    den = den_w * sw + den_c * sc + (sink - mx).exp()
    num = num_w * sw.unsqueeze(-1) + num_c * sc.unsqueeze(-1)
    den_safe = torch.where(den > 0, den, torch.ones_like(den))
    out = num / den_safe.unsqueeze(-1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out.to(q.dtype)


def available():
    return HAS_TRITON and torch.cuda.is_available()
