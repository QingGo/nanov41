"""整段融合的 sparse attention（forward）：window + compress + sink 全在一个 Triton kernel 内。

## 为什么（S16 §34/§37）
attention 在 5090/b=16 占 35.5 ms（29%），但它只有 5.8% FLOPs。三次"轻量改造"都失败：
整层融合（compress 留 torch）更慢、lse+softmax 变体 0.64×、torch 级没有空间。
结论：唯一可行路线是**真正融合**——把 window band、compress gather、sink 以及 softmax 全部放进一个 kernel，
让 `[b,h,m,w]` 的 score/prob 只存在于寄存器/共享内存，不落 HBM。

## 本文件（第一步：forward）
- grid = (b * n_tiles, h)，每 program 处理 BM=64 个 query 行、BW=128 个窗口 key、NC=32 个 compress key；
- 窗口：`s_w = q·kᵀ * scale`（fp16 点积、fp32 累加），band 掩码；
- compress：按 NC 逐列 gather（`idx [BM,NC]`，-1 表示无效），`s_c[:,c] = Σ_d q·kc`；
- sink：作为第 NC+1 个"value=0 的 key"参与同一个 softmax；
- 输出 `out [b,m,h,d] bf16` 与 `lse [b,m,h] fp32`（backward 用）。

数值口径与 `fast_attn` 一致：scores/probs 用 fp16 存储/点积，softmax 归约在 fp32。
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

BM = 64
BW = 128
NC = 32


if HAS_TRITON:

    @triton.jit
    def _attn_fwd_full(Q, KV, IDX, SINK, OUT, LSE, B, NB, M, N, WIN,
                       SCALE: tl.constexpr, BM: tl.constexpr, BW: tl.constexpr,
                       NC: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                       HAS_COMP: tl.constexpr):
        pid_t = tl.program_id(0)          # b * NB + tile
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
                    mask=maskq, other=0.0)                                   # bf16 [BM,D]
        k = tl.load(KV + (b * N + kj[:, None]) * D + rd[None, :],
                    mask=(rj[:, None] < k_w) & (kj[:, None] < M), other=0.0)  # bf16 [BW,D]
        qf = q.to(tl.float32)
        qh = q.to(tl.float16)
        kh = k.to(tl.float16)
        s_w = tl.dot(qh, tl.trans(kh)) * SCALE                               # [BM,BW] fp32
        valid_w = (kj[None, :] <= rq[:, None]) & (kj[None, :] > (rq[:, None] - WIN)) \
            & (rj[None, :] < k_w) & maskq
        s_w = tl.where(valid_w, s_w, float("-inf"))

        rc = tl.arange(0, NC)
        s_c = tl.full((BM, NC), float("-inf"), dtype=tl.float32)
        if HAS_COMP:
            # 第一遍：只算 compress 的 score（32 列，逐列 gather + 行内点积）
            for c in tl.static_range(NC):
                ic = tl.load(IDX + (b * M + rq) * NC + c, mask=(rq < M), other=-1).to(tl.int64)
                kc = tl.load(KV + (b * N + ic[:, None]) * D + rd[None, :],
                             mask=(ic[:, None] >= 0), other=0.0)              # [BM,D] bf16
                sc = tl.sum(qf * kc.to(tl.float32), axis=1) * SCALE
                sc = tl.where(ic >= 0, sc, float("-inf"))
                s_c = tl.where(rc[None, :] == c, sc[:, None], s_c)

        sink = tl.load(SINK + h).to(tl.float32)
        m_w = tl.max(s_w, axis=1)
        m_c = tl.max(s_c, axis=1) if HAS_COMP else tl.full((BM,), float("-inf"), dtype=tl.float32)
        mx = tl.maximum(tl.maximum(m_w, m_c), sink)
        mx_safe = tl.where(mx == float("-inf"), 0.0, mx)
        p_w = tl.exp(s_w - mx_safe[:, None])                                  # [BM,BW]
        num = tl.dot(p_w.to(tl.float16), kh)                                  # [BM,D] fp32
        den = tl.sum(p_w, axis=1)
        if HAS_COMP:
            p_c = tl.exp(s_c - mx_safe[:, None])                              # [BM,NC]
            den = den + tl.sum(p_c, axis=1)
            # 第二遍：accumulate compress 的 PV（逐列，kc 命中 L2）
            for c in tl.static_range(NC):
                ic = tl.load(IDX + (b * M + rq) * NC + c, mask=(rq < M), other=-1).to(tl.int64)
                kc = tl.load(KV + (b * N + ic[:, None]) * D + rd[None, :],
                             mask=(ic[:, None] >= 0), other=0.0)
                pc = tl.sum(tl.where(rc[None, :] == c, p_c, 0.0), axis=1)     # [BM]
                num = num + pc[:, None] * kc.to(tl.float32)
        den = den + tl.exp(sink - mx_safe)
        den_safe = tl.where(den > 0, den, 1.0)
        out = num / den_safe[:, None]
        tl.store(OUT + ((b * M + rq[:, None]) * H + h) * D + rd[None, :],
                 out.to(OUT.dtype.element_ty), mask=maskq)
        tl.store(LSE + (b * M + rq) * H + h, mx_safe + tl.log(den_safe), mask=(rq < M))

    def attn_fwd(q, kv, sink, idxs, m, win, ncomp, scale):
        """q [b,m,h,d] bf16 / kv [b,n,d] bf16 / idxs [b,m,win+ncomp] int32 → (out, lse)。"""
        b, m_, h, d = q.shape
        n = kv.size(1)
        nb = (m + BM - 1) // BM
        comp = torch.full((b, m, NC), -1, dtype=torch.int64, device=q.device)
        has = ncomp > 0
        if has:
            comp[:, :, :ncomp] = idxs[:, :, win:win + ncomp].to(torch.int64).clamp_min(-1)
        out = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)
        lse = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        _attn_fwd_full[(b * nb, h)](q.contiguous(), kv.contiguous(), comp, sink, out, lse,
                                    b, nb, m, n, win, SCALE=scale, BM=BM, BW=BW, NC=NC,
                                    D=d, H=h, HAS_COMP=has, num_warps=4, num_stages=2)
        return out, lse
else:  # pragma: no cover
    def attn_fwd(*a, **k):
        raise RuntimeError("triton 不可用")


def available():
    return HAS_TRITON and torch.cuda.is_available()
