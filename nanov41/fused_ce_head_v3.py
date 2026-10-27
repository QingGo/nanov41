"""FP8 head + CE（v3）：3 个 kernel 级阶段，CE 的 elementwise 全部搬进 GEMM 的 epilogue/prologue。

设计依据：内部 CE kernel 规格（研究记录，不随本仓库发布）。
现状 `fp8_head_v2` = 28.3 ms：3 次 fp8 GEMM（16.6 ms @70% MFU）+ 22.5 GB 搬运（14 ms），
因为 CE 的 `ce_fwd` / `ce_bwd_fp8` 是**独立 kernel**，把全部 logits 流量暴露在 GEMM 之外。
v3 的三段：

    K1     logits[t,v] = (X·Wᵀ)  fp8 GEMM, fp32 累加 → epilogue 直接出
             P[TP,V] e4m3 = 448·exp(l − tilemax)   每 (row, V-tile) 的 shift+scale
             TM/TS[TP,NVT] fp32                    partial max / partial Σexp
             TL[TP] fp32                           target logit（寄存器直出，无损）
    reduce TM/TS → lse、amax = max(1−p_t, 1/s)（v2 同款闭式）、a[t,vt]、loss
    K2     dX[t,:] = Σ_v dsc[t,v]·Wq_col[v,:]·swc·rs[t]，dsc = clamp(P̃·a − 1_hit·inv, ±448)
    K3     dW[v,:] = Σ_t dsc[t,v]·Xq_col[t,:]·sxc·(grad/count)
    （K3 转置的是 X（6.3 MB），不是 dlogits（2.12 GB），见 S16 §53.5 第 7 条）

## 为什么 logits 存 `P̃ = 448·exp(l − tilemax)` 而不是 fp8 logits

e4m3 只有 3 位尾数 ⇒ **相对**精度 2⁻⁴ ≈ 6%，与 scale 无关。|logit| ~ 20 时 ulp = 2，
直接存 logit（即使 per-row 归一到 448）在 softmax 顶部就有 δl ≈ 0.0625·|l| ≈ 1.25，
exp 后 p 差 3.5×。改成存**概率域** exp(l − tilemax)：
  * tile 内最大值 = 448（精确可表示），越靠近 top 网格越密（e4m3 近零步长 2⁻⁹）；
  * 误差自归一：δp/p ≈ 6%（且 p 越小绝对误差越小），不随 |logit| 放大；
  * 与 v2 **同一套 dlogits 量化栅格**（per-row amax = max(1−p_t, 1/s)、同一 e4m3 舍入）
    ⇒ 两次量化误差几乎完全抵消。

## 实测（RTX 5090 sm_120，torch 2.14.1+cu130，triton 3.8.0；b=16, T=16384, V=129280, d=384）

    fwd+bwd  15.75 ms eager / 15.77 ms graph（基线 fp8_head_v2 同机 32.5/32.0 ms；任务基线 28.3 ms ⇒ **1.80×**）
    分段： prep_w 0.33 + K1 5.79 + reduce 0.12 + K2 4.37(+0.10) + prep_xt 0.07 + K3 5.04 = 15.87 ms
    精度： vs v2  loss rel 7.8e-8 / dX 2.4e-3 / dW 3.8e-2（gate: 1e-4 / 5e-2 / 5e-2）
           vs fp32 torch oracle  dW 3.9e-4（v2 自己是 3.8e-2 ⇒ 那 3.8e-2 全是 v2 的 bf16 partial 误差）
    graph： eager 与 graph replay **逐位一致**（loss/dX/dW）
    GEMM：  Triton fp8 441 TF ≈ cuBLASLt 444 TF（同形状）⇒ epilogue 可以真"免费"长在 GEMM 里
    smem：  5090 每 CTA 只有 99 KB（不是 227 KB）⇒ stages*(A+B) ≤ 96 KB；num_warps=16 + 196 KB 累加器会
            spill 到 289 ms。K1 取 BM=64（acc 32 KB ⇒ 2 CTA/SM）比 BM=128 快 1.1 ms。

## 数值口径（本机 CPU 仿真 /tmp/sim_v2v3.py，T=256 V=129280 d=384 BN=128）

两条路都跑在**同一组 fp8 量化操作数**上，差别只有"logits 怎么存 / d 怎么算"：
v2 = bf16 logits + 一次 per-row-amax 量化；v3 = P̃ 概率域存储 + 闭式 amax 再量化。

| 场景 | abs(logit)max | v2 dX vs exact | v3 dX vs exact | **v3 vs v2** | dW v3 vs v2 | loss rel |
|---|---|---|---|---|---|---|
| σ=1 | 6.0 | 1.35e-3 | 1.31e-3 | **3.1e-4** | 3.0e-4 | 0 |
| σ=3 | 17.8 | 3.22e-3 | 3.23e-3 | **3.6e-4** | 3.5e-4 | 0 |
| σ=3+boost | 17.8 | 2.61e-3 | 2.62e-3 | **2.9e-4** | 3.6e-4 | 0 |
| σ=10+boost | 59.4 | 2.13e-2 | 2.13e-2 | **2.3e-5** | 2.5e-5 | 0 |

⇒ v3 的 dX/dW 精度与 v2 **同一档**（不是"劣化"，个别场景更好），且两者互差 ~3e-4
（gate 5e-2 的 1/160）。loss 与 v2 **逐位同档**（前提：K1 epilogue 也把 logits 过一遍 bf16，
见 `_k1_kernel` 注释）。

反向对照（不可行路线）：raw-logit + per-row scale 存 e4m3 → lse 误差 0.19/0.50/0.54/1.67 nat、
dX rel 3e-4/**1.9e-2**/**2.0e-2**/**1.8e-1** ⇒ "省掉 exp 直接存 logits" 这条路不可用。
"""
from __future__ import annotations

import os
import sys

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001  (Mac / 无 triton 也能 import 本模块)
    HAS_TRITON = False
    triton = None
    tl = None

E4M3 = 448.0
LOG2E = 1.4426950408889634



def _envi(name, dflt):
    return int(os.environ.get(name, dflt))


# ---- tile 选择（env 可覆盖；sweep 见 tools/test_fused_ce_head_v3.py）----
K1_BM = _envi("NANOV41_V3_K1_BM", 64)
K1_BN = _envi("NANOV41_V3_K1_BN", 128)     # V 的 tile，2 的幂且整除 V
# 5090 sm_120 的 smem 上限只有 **99 KB/CTA**（100 KB/SM），所以 tile 必须按
# stages*(A_tile + B_tile) <= 96 KB 选；K1(128,128,64,S=4)=64KB，K2/K3 的 B 是 3 个
# [128,BN]/[128,BT] 片 ⇒ k-step 只能取 64（3*(4+24)=84KB）
K1_BK = _envi("NANOV41_V3_K1_BK", 64)
K1_W = _envi("NANOV41_V3_K1_W", 8)
K1_S = _envi("NANOV41_V3_K1_S", 4)
K2_BM = _envi("NANOV41_V3_K2_BM", 64)
K2_BN = _envi("NANOV41_V3_K2_BN", 64)     # k-step（v 方向），须整除 K1_BN
K2_SPLIT = _envi("NANOV41_V3_K2_SPLIT", 5)
K2_W = _envi("NANOV41_V3_K2_W", 8)
K2_S = _envi("NANOV41_V3_K2_S", 3)
K3_BV = _envi("NANOV41_V3_K3_BV", 64)      # v 方向 tile（= 输出行）
K3_BT = _envi("NANOV41_V3_K3_BT", 64)     # t 方向 k-step
K3_W = _envi("NANOV41_V3_K3_W", 8)
K3_S = _envi("NANOV41_V3_K3_S", 3)
RED_BR = _envi("NANOV41_V3_RED_BR", 4)
WP_BV = 64
WP_BD = 128
QR_BM = 64
QR_BK = 128


if HAS_TRITON:

    # ==================================================================== K1
    @triton.jit
    def _k1_kernel(P, TM, TS, TL, XQ, SX, WQ, SW, TGT,
                   V: tl.constexpr, D: tl.constexpr, NVT: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """fp8 GEMM + 在线 softmax epilogue（partial max/sum + 概率域 P̃ + target logit）。"""
        pid_m = tl.program_id(0)
        pid_v = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_v * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = XQ + rm[:, None] * D + rk[None, :]
        b_ptrs = WQ + rn[:, None] * D + rk[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(D // BK):
            a = tl.load(a_ptrs + k * BK)                 # [BM,BK] fp8, k(=D) 连续
            b = tl.load(b_ptrs + k * BK)                 # [BN,BK] fp8, k(=D) 连续
            acc = tl.dot(a, tl.trans(b), acc)            # 原生 TN（两个操作数都 k-major）
        sx = tl.load(SX + rm)
        sw = tl.load(SW + rn)
        lg = acc * sx[:, None] * sw[None, :]
        # 与 v2 同口径：v2 的 `_scaled_mm(out_dtype=bf16)` 把 logits 过一遍 bf16，
        # 之后 lse/amax/target-logit 全部基于这份 bf16。这里显式复刻，
        # 否则 loss 与 v2 的差 ~1e-3（bf16 ulp 在 |logit|~20 时 0.06），过不了 1e-4 门槛；
        # 对 P̃ 无影响（e4m3 的 6% 远粗于 bf16 的 0.4%）。
        lg = lg.to(tl.bfloat16).to(tl.float32)
        tmax = tl.max(lg, axis=1)
        e = tl.exp2((lg - tmax[:, None]) * 1.4426950408889634)
        tsum = tl.sum(e, axis=1)
        tl.store(TM + rm * NVT + pid_v, tmax)
        tl.store(TS + rm * NVT + pid_v, tsum)
        pq = tl.minimum(e * 448.0, 448.0).to(tl.float8e4nv)
        tl.store(P + rm[:, None] * V + rn[None, :], pq)
        tgt = tl.load(TGT + rm).to(tl.int32)
        hit = rn[None, :] == tgt[:, None]
        tval = tl.sum(tl.where(hit, lg, 0.0), axis=1)
        tl.store(TL + rm, tval, mask=tl.sum(hit.to(tl.int32), axis=1) > 0)

    # ================================================================ reduce
    @triton.jit
    def _reduce_kernel(TM, TS, TL, TGT, A, LSE, INV, AMAX, LOSS,
                       NVT: tl.constexpr, NVTP: tl.constexpr, BR: tl.constexpr,
                       IGNORE: tl.constexpr):
        """partial max/sum → lse、amax（闭式）、每 (row,tile) 的 dlogits 量化系数 a。"""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        j = tl.arange(0, NVTP)
        mk = j < NVT
        tm = tl.load(TM + r[:, None] * NVT + j[None, :], mask=mk[None, :],
                     other=-float("inf"))
        ts = tl.load(TS + r[:, None] * NVT + j[None, :], mask=mk[None, :], other=0.0)
        rowm = tl.max(tm, axis=1)
        s = tl.sum(ts * tl.exp2((tm - rowm[:, None]) * 1.4426950408889634), axis=1)
        lse = rowm + tl.log(s)
        tgt = tl.load(TGT + r).to(tl.int32)
        valid = tgt != IGNORE
        tv = tl.load(TL + r)
        pt = tl.exp2((tv - lse) * 1.4426950408889634)
        amax = tl.maximum(1.0 - pt, 1.0 / s)
        inv = tl.where(valid, 448.0 / amax, 0.0)
        cf = tl.exp2((tm - lse[:, None]) * 1.4426950408889634)      # exp(tilemax − lse)
        a = cf * (inv[:, None] / 448.0)                      # d = P̃·a − 1_hit·inv
        tl.store(A + r[:, None] * NVT + j[None, :], a, mask=mk[None, :])
        tl.store(LSE + r, lse)
        tl.store(INV + r, inv)
        tl.store(AMAX + r, amax)
        tl.store(LOSS + r, tl.where(valid, lse - tv, 0.0))

    # ==================================================================== K2
    @triton.jit
    def _k2_kernel(DX, P, A, INV, TGT, WQT, SWC, RS,
                   V: tl.constexpr, D: tl.constexpr, NVT: tl.constexpr,
                   TP: tl.constexpr, KBN: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, NSPL: tl.constexpr, OUTFP32: tl.constexpr):
        """dX[t,:] = Σ_v dsc[t,v]·Wq_col[v,:]·swc[:]·rs[t]；dsc 在寄存器里现算（不落 dlogits）。

        D = 384 手展开成 3 个 [BM,128] 累加器（tl.arange 要求 2 的幂）；
        WQT 是 [D,V]（v 连续）⇒ 两个操作数都 k(=v)-major，原生 TN。
        """
        pid_m = tl.program_id(0)
        pid_s = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        tgt = tl.load(TGT + rm).to(tl.int32)
        inv = tl.load(INV + rm)
        rs = tl.load(RS + rm)
        rd = tl.arange(0, 128)
        acc0 = tl.zeros((BM, 128), dtype=tl.float32)
        acc1 = tl.zeros((BM, 128), dtype=tl.float32)
        acc2 = tl.zeros((BM, 128), dtype=tl.float32)
        for j in range(NSPL):
            v0 = (pid_s * NSPL + j) * BN
            vt = v0 // KBN
            rn = v0 + tl.arange(0, BN)
            av = tl.load(A + rm * NVT + vt)
            pq = tl.load(P + rm[:, None] * V + rn[None, :]).to(tl.float32)
            d = pq * av[:, None] - tl.where(rn[None, :] == tgt[:, None], inv[:, None], 0.0)
            dq = tl.maximum(tl.minimum(d, 448.0), -448.0).to(tl.float8e4nv)
            b0 = tl.load(WQT + rd[:, None] * V + rn[None, :])
            acc0 = tl.dot(dq, tl.trans(b0), acc0)
            b1 = tl.load(WQT + (rd + 128)[:, None] * V + rn[None, :])
            acc1 = tl.dot(dq, tl.trans(b1), acc1)
            b2 = tl.load(WQT + (rd + 256)[:, None] * V + rn[None, :])
            acc2 = tl.dot(dq, tl.trans(b2), acc2)
        swc0 = tl.load(SWC + rd)
        swc1 = tl.load(SWC + rd + 128)
        swc2 = tl.load(SWC + rd + 256)
        o = rm[:, None] * D
        base = DX + pid_s * (TP * D)
        if OUTFP32:
            tl.store(base + o + rd[None, :], acc0 * swc0[None, :] * rs[:, None])
            tl.store(base + o + (rd + 128)[None, :], acc1 * swc1[None, :] * rs[:, None])
            tl.store(base + o + (rd + 256)[None, :], acc2 * swc2[None, :] * rs[:, None])
        else:
            tl.store(base + o + rd[None, :],
                     (acc0 * swc0[None, :] * rs[:, None]).to(tl.bfloat16))
            tl.store(base + o + (rd + 128)[None, :],
                     (acc1 * swc1[None, :] * rs[:, None]).to(tl.bfloat16))
            tl.store(base + o + (rd + 256)[None, :],
                     (acc2 * swc2[None, :] * rs[:, None]).to(tl.bfloat16))

    # ==================================================================== K3
    @triton.jit
    def _k3_kernel(DW, P, A, INV, TGT, XT, SXC, GSC,
                   V: tl.constexpr, D: tl.constexpr, NVT: tl.constexpr,
                   TP: tl.constexpr, KBN: tl.constexpr, BV: tl.constexpr,
                   BT: tl.constexpr):
        """dW[v,:] = Σ_t dsc[t,v]·Xq_col[t,:]·sxc[:]·gsc（A 操作数是 dscᵀ，寄存器内即算即转）。

        P 按 [BV,BT] 取（v 连续，每行 BT 字节）；P 每个元素只读一次（CTA 独占 v 区间）。
        """
        pid_v = tl.program_id(0)
        vb = pid_v * BV + tl.arange(0, BV)
        vt = (pid_v * BV) // KBN
        rd = tl.arange(0, 128)
        acc0 = tl.zeros((BV, 128), dtype=tl.float32)
        acc1 = tl.zeros((BV, 128), dtype=tl.float32)
        acc2 = tl.zeros((BV, 128), dtype=tl.float32)
        for k in range(TP // BT):
            rt = k * BT + tl.arange(0, BT)
            pq = tl.load(P + rt[None, :] * V + vb[:, None]).to(tl.float32)
            at = tl.load(A + rt * NVT + vt)
            it = tl.load(INV + rt)
            tgt = tl.load(TGT + rt).to(tl.int32)
            d = pq * at[None, :] - tl.where(vb[:, None] == tgt[None, :], it[None, :], 0.0)
            dq = tl.maximum(tl.minimum(d, 448.0), -448.0).to(tl.float8e4nv)
            x0 = tl.load(XT + rd[:, None] * TP + rt[None, :])
            acc0 = tl.dot(dq, tl.trans(x0), acc0)
            x1 = tl.load(XT + (rd + 128)[:, None] * TP + rt[None, :])
            acc1 = tl.dot(dq, tl.trans(x1), acc1)
            x2 = tl.load(XT + (rd + 256)[:, None] * TP + rt[None, :])
            acc2 = tl.dot(dq, tl.trans(x2), acc2)
        # dsc = d·448/amax 里那个 448 在 K3 无处可折（a 是 per-t、在 t-sum 内），
        # 故在此除回；swc 之外的 1/448 与 sxc 一起构成标量输出 scale（graph-safe 标量张量）
        # sxc / gsc 都是**标量**（[1] 张量）：xt 用单标量量化（与 v2 的 TensorWise 同档）
        g = tl.load(GSC) * tl.load(SXC) * (1.0 / 448.0)
        sc0 = g
        sc1 = g
        sc2 = g
        o = vb[:, None] * D
        tl.store(DW + o + rd[None, :], (acc0 * sc0[None, :]).to(tl.bfloat16))
        tl.store(DW + o + (rd + 128)[None, :], (acc1 * sc1[None, :]).to(tl.bfloat16))
        tl.store(DW + o + (rd + 256)[None, :], (acc2 * sc2[None, :]).to(tl.bfloat16))

    # ========================================================== 小工具 kernel
    @triton.jit
    def _wprep_kernel(W, AR, AC, WQ, WQT, V: tl.constexpr, D: tl.constexpr,
                      BV: tl.constexpr, BD: tl.constexpr):
        """读一次 bf16 权重 → fwd 用的行缩放 WQ[V,D] + dX 用的列缩放 WQT[D,V]。"""
        rv = tl.program_id(0) * BV + tl.arange(0, BV)
        rd = tl.program_id(1) * BD + tl.arange(0, BD)
        w = tl.load(W + rv[:, None] * D + rd[None, :]).to(tl.float32)
        ar = tl.load(AR + rv)
        ac = tl.load(AC + rd)
        wq = tl.maximum(tl.minimum(w * (1.0 / ar)[:, None], 448.0), -448.0)
        tl.store(WQ + rv[:, None] * D + rd[None, :], wq.to(tl.float8e4nv))
        wt = tl.maximum(tl.minimum(w * (1.0 / ac)[None, :], 448.0), -448.0)
        tl.store(WQT + rd[:, None] * V + rv[None, :], tl.trans(wt).to(tl.float8e4nv))

    @triton.jit
    def _qrow_kernel(X, S, XQ, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        """行缩放 fp8（fwd 的 X [TP,D]）。"""
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rk = tl.program_id(1) * BK + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * K + rk[None, :]).to(tl.float32)
        s = tl.load(S + rm)
        q = tl.maximum(tl.minimum(x * (1.0 / s)[:, None], 448.0), -448.0)
        tl.store(XQ + rm[:, None] * K + rk[None, :], q.to(tl.float8e4nv))

    @triton.jit
    def _qcolt_kernel(X, AM, S, XT, K: tl.constexpr, BT: tl.constexpr,
                      BD: tl.constexpr):
        """X·amax[t] 按列（单标量）量化并转置成 XT[D,TP]（K3 的 B 操作数）。"""
        rt = tl.program_id(0) * BT + tl.arange(0, BT)
        rd = tl.program_id(1) * BD + tl.arange(0, BD)
        x = tl.load(X + rt[:, None] * K + rd[None, :]).to(tl.float32)
        am = tl.load(AM + rt)
        sv = tl.load(S)                       # 量化 scale = amax(X·amax)/448
        q = tl.maximum(tl.minimum(x * (am * (1.0 / sv))[:, None], 448.0), -448.0)
        tl.store(XT + rd[:, None] * tl.num_programs(0) * BT + rt[None, :],
                 tl.trans(q).to(tl.float8e4nv))

    # ================================================================== host
    def _quant_w(weight):
        """[V,D] bf16 → WQ[V,D](行缩放) + WQT[D,V](列缩放)。"""
        V, D = weight.shape
        aw = weight.abs()                                       # bf16 临时（99 MB）
        sw = (aw.amax(dim=1).float() / E4M3).clamp_min(1e-12)   # [V] 量化 scale（fwd 行缩放）
        swc = (aw.amax(dim=0).float() / E4M3).clamp_min(1e-12)  # [D] 量化 scale（dX 列缩放）
        del aw
        wq = torch.empty(V, D, dtype=torch.float8_e4m3fn, device=weight.device)
        wqt = torch.empty(D, V, dtype=torch.float8_e4m3fn, device=weight.device)
        _wprep_kernel[(V // WP_BV, D // WP_BD)](weight, sw, swc, wq, wqt, V, D,
                                                WP_BV, WP_BD, num_warps=8)
        return wq, sw, wqt, swc

    def _quant_x(x):
        M, D = x.shape
        sx = (x.float().abs().amax(dim=1) / E4M3).clamp_min(1e-12)
        xq = torch.empty(M, D, dtype=torch.float8_e4m3fn, device=x.device)
        _qrow_kernel[(M // QR_BM, D // QR_BK)](x, sx, xq, D, QR_BM, QR_BK, num_warps=4)
        return xq, sx

    def _quant_xt(x, amax):
        """K3 的 B 操作数：xp[t,:] = X[t,:]·amax[t]，再**单标量**量化（与 v2 的 TensorWise 同档）。"""
        M, D = x.shape
        rowmax = x.float().abs().amax(dim=1)                    # [M]
        s = ((rowmax * amax).amax() / E4M3).clamp_min(1e-12).reshape(1)
        xt = torch.empty(D, M, dtype=torch.float8_e4m3fn, device=x.device)
        _qcolt_kernel[(M // QR_BM, D // QR_BK)](x, amax, s, xt, D, QR_BM, QR_BK, num_warps=4)
        return xt, s

    def _check_shapes(V, D, TP):
        assert D == 384, "v3 只支持 d=384（D 方向 3×128 展开）"
        assert V % K1_BN == 0, f"V={V} 不能被 K1_BN={K1_BN} 整除"
        nvt = V // K1_BN
        nsteps = V // K2_BN
        assert nsteps % K2_SPLIT == 0, "V/K2_BN 必须能被 K2_SPLIT 整除"
        assert (K1_BN % K2_BN == 0) and (K1_BN % K3_BV == 0 or K3_BV % K1_BN == 0), \
            "K1_BN / K2_BN / K3_BV 必须对齐（同一 vtile 内）"
        assert TP * V < 2 ** 31, "P 的 int32 偏移会溢出"
        return nvt, nsteps

    def _forward_state(hidden, weight, ids, ignore_index):
        b, s, d = hidden.shape
        n = b * s
        V = weight.shape[0]
        dev = hidden.device
        BM = K1_BM
        pad = max(BM, K3_BT, RED_BR, 128)
        TP = (n + pad - 1) // pad * pad
        NVT, nsteps = _check_shapes(V, d, TP)
        NVP = triton.next_power_of_2(NVT)
        tgt = torch.full((TP,), ignore_index, dtype=torch.long, device=dev)
        if s > 1:
            tgt[:n].view(b, s)[:, :-1] = ids[:, 1:]
        x = hidden.reshape(n, d)
        if TP != n:
            x = torch.cat([x, x.new_zeros(TP - n, d)])
        xq, sx = _quant_x(x)
        wq, sw, wqt, swc = _quant_w(weight)
        P = torch.empty(TP, V, dtype=torch.float8_e4m3fn, device=dev)
        TM = torch.empty(TP, NVT, dtype=torch.float32, device=dev)
        TS = torch.empty(TP, NVT, dtype=torch.float32, device=dev)
        TL = torch.zeros(TP, dtype=torch.float32, device=dev)
        _k1_kernel[(TP // BM, NVT)](P, TM, TS, TL, xq, sx, wq, sw, tgt, V, d, NVT,
                                    BM, K1_BN, K1_BK, num_warps=K1_W, num_stages=K1_S)
        A = torch.empty(TP, NVT, dtype=torch.float32, device=dev)
        LSE = torch.empty(TP, dtype=torch.float32, device=dev)
        INV = torch.empty(TP, dtype=torch.float32, device=dev)
        AMAX = torch.empty(TP, dtype=torch.float32, device=dev)
        LOSS = torch.empty(TP, dtype=torch.float32, device=dev)
        _reduce_kernel[(TP // RED_BR,)](TM, TS, TL, tgt, A, LSE, INV, AMAX, LOSS,
                                        NVT, NVP, RED_BR, ignore_index, num_warps=8)
        return dict(P=P, A=A, LSE=LSE, INV=INV, AMAX=AMAX, LOSS=LOSS, xq=xq, sx=sx,
                    wq=wq, sw=sw, wqt=wqt, swc=swc, tgt=tgt, TP=TP, n=n, d=d, V=V,
                    NVT=NVT, nsteps=nsteps, dev=dev, x=x, weight=weight,
                    hidden_shape=hidden.shape)

    class _V3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, ids, ignore_index):
            st = _forward_state(hidden, weight, ids, ignore_index)
            cnt = (st["tgt"][: st["n"]] != ignore_index).sum().clamp_min(1)
            loss = st["LOSS"][: st["n"]].sum() / cnt
            ctx.st = st
            ctx.ignore_index = ignore_index
            return loss

        @staticmethod
        def backward(ctx, grad_out):
            st = ctx.st
            P, A, INV, AMAX = st["P"], st["A"], st["INV"], st["AMAX"]
            TP, d, V, NVT, nsteps = st["TP"], st["d"], st["V"], st["NVT"], st["nsteps"]
            tgt, dev = st["tgt"], st["dev"]
            gsc = (grad_out.float() / (tgt[: st["n"]] != ctx.ignore_index)
                   .sum().clamp_min(1)).detach().reshape(1)
            # rs[t] = amax/448 · grad/count = (1/inv)·gsc（invalid 行 inv=0 → 0）
            rs = torch.where(INV > 0, gsc.reshape(()) / INV, torch.zeros_like(INV))
            # ---- K2: dX
            SPL = K2_SPLIT
            NSPL = nsteps // SPL
            if SPL > 1:
                parts = torch.empty(SPL, TP, d, dtype=torch.float32, device=dev)
                _k2_kernel[(TP // K2_BM, SPL)](parts, P, A, INV, tgt, st["wqt"], st["swc"], rs,
                                               V, d, NVT, TP, K1_BN, K2_BM, K2_BN, NSPL, True,
                                               num_warps=K2_W, num_stages=K2_S)
                dxf = parts[0]
                for i in range(1, SPL):
                    dxf = dxf.add_(parts[i])
                dX = dxf.to(st["x"].dtype).reshape(st["hidden_shape"])
            else:
                dX = torch.empty(TP, d, dtype=st["x"].dtype, device=dev)
                _k2_kernel[(TP // K2_BM, 1)](dX, P, A, INV, tgt, st["wqt"], st["swc"], rs,
                                             V, d, NVT, TP, K1_BN, K2_BM, K2_BN, NSPL, False,
                                             num_warps=K2_W, num_stages=K2_S)
                dX = dX.reshape(st["hidden_shape"])
            # ---- K3: dW
            xt, sxc = _quant_xt(st["x"], AMAX)
            dW = torch.empty(V, d, dtype=torch.bfloat16, device=dev)
            _k3_kernel[(V // K3_BV,)](dW, P, A, INV, tgt, xt, sxc, gsc,
                                      V, d, NVT, TP, K1_BN, K3_BV, K3_BT,
                                      num_warps=K3_W, num_stages=K3_S)
            return dX, dW.to(st["weight"].dtype), None, None
else:  # pragma: no cover

    def _forward_state(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")


def fused_head_ce_v3(hidden, weight, ids, ignore_index=-100):
    """hidden [b,s,d]、weight [V,d]、ids [b,s] → 标量 loss（autograd 可用）。"""
    if not HAS_TRITON:
        raise RuntimeError("triton 不可用")
    return _V3.apply(hidden, weight, ids, ignore_index)


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()


def install() -> None:
    """把 `train_utils.fused_head_ce` 的 fp8 路径指向 v3（env `NANOV41_HEAD_FP8=v3`）。

    不改任何现存文件：只在运行时 monkeypatch（含已经 `from .train_utils import
    fused_head_ce` 绑定的模块属性）。
    """
    from . import train_utils as _tu

    cur = _tu.fused_head_ce
    if getattr(cur, "_nanov41_v3", False):
        return

    def _patched(hidden, weight, ids, chunk=None, ignore_index=-100, z_loss=0.0):
        if (os.environ.get("NANOV41_HEAD_FP8", "").lower() in ("3", "v3") and z_loss == 0
                and available()):
            try:
                return fused_head_ce_v3(hidden, weight, ids, ignore_index)
            except Exception:  # noqa: BLE001
                if os.environ.get("NANOV41_V3_STRICT"):
                    raise
        return cur(hidden, weight, ids, chunk, ignore_index, z_loss)

    _patched._nanov41_v3 = True
    _patched._nanov41_orig = cur
    _tu.fused_head_ce = _patched
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("nanov41"):
            continue
        if getattr(mod, "fused_head_ce", None) is cur:
            setattr(mod, "fused_head_ce", _patched)
