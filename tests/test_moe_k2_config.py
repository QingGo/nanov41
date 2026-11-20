# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""K2: MoE tile/env 解析 + `check_tiles` 形状校验（**CPU-only**，不需要 CUDA/triton）。

背景：`triton_grouped_moe` 原来在模块级对**硬编码的 384**（D8/D12 的 dim=inter=384）做整除
断言，于是 aligned d12（dim=512 / inter=256）上的合法组合被误拒：
`NANOV41_MOE_TILE=64,256,64,8,3`（384 % 256 != 0）和 `NANOV41_MOE_DW_TILE=128,128,128,8,3`
都会在 import 时炸。K2 把整除性搬到 `check_tiles(dim, inter, p_max)`（按真实形状，首次调用
检查一次），把与形状无关的合法性（2 的幂、warps ∈ {1,2,4,8,16}）留在解析时。

这些测试全部在 CPU 上跑（`HAS_TRITON=False` 时模块仍会解析 env，所以能测）。
"""
from __future__ import annotations

import contextlib
import importlib
import os

import pytest
import torch

import nanov41.triton_grouped_moe as tgm

_ENV_KEYS = (
    "NANOV41_MOE_TILE", "NANOV41_MOE_DW_TILE", "NANOV41_MOE_BWD_TILE",
    "NANOV41_MOE_DW_KERNEL", "NANOV41_MOE_DW13_TILE", "NANOV41_MOE_DW2_TILE",
    "NANOV41_MOE_DW13_EB", "NANOV41_MOE_DW2_EB", "NANOV41_MOE_DX_TILE",
    "NANOV41_MOE_FUSED_DX2", "NANOV41_MOE_DW2_TIGHT",
    "NANOV41_MOE_NOPAD", "NANOV41_MOE_FUSED_TAIL", "NANOV41_MOE_LAYOUT",
)

# aligned d12（K2 目标形状）与 D8/D12（老基准形状）
ALIGNED = (512, 256, 24576)
OLD = (384, 384, 33216)


@contextlib.contextmanager
def reload_with(**env):
    """在给定 env 下重新 import 模块；退出时恢复原 env 并重新 import 回默认状态。"""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in env.items():
        if v is not None:
            os.environ[k] = str(v)
    try:
        yield importlib.reload(tgm)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(tgm)


def test_defaults_unchanged():
    """默认值必须与 K2 之前一致（生产不带这些 env 时行为不变）。"""
    assert (tgm.BM, tgm.BN, tgm.BK, tgm.WARPS, tgm.STAGES) == (64, 64, 128, 4, 2)
    assert (tgm.DW_BM, tgm.DW_BK, tgm.DW_BN, tgm.DW_WARPS, tgm.DW_STAGES) == (128, 32, 64, 4, 3)
    assert tgm.DW_KERNEL == "legacy"
    assert tgm.FUSED_DX2 is False
    assert tgm.DW2_TIGHT is False
    assert tgm.DWK13_TILE == (128, 64, 32, 4, 3) and tgm.DWK13_EB == 4
    assert tgm.DWK2_TILE == (128, 128, 32, 4, 3) and tgm.DWK2_EB == 4


def test_production_env_parses_on_both_shapes():
    """生产 env（m1_ab_run.sh 的 overrides）在两个形状上都要通过 check_tiles。"""
    with reload_with(NANOV41_MOE_TILE="64,64,64,4,3",
                     NANOV41_MOE_DW_TILE="128,64,64,4,3") as m:
        assert (m.BM, m.BN, m.BK, m.WARPS, m.STAGES) == (64, 64, 64, 4, 3)
        assert (m.DW_BM, m.DW_BK, m.DW_BN) == (128, 64, 64)
        m.check_tiles(*ALIGNED)
        m.check_tiles(*OLD)


def test_wide_bn_legal_for_aligned_but_not_for_384():
    """K2 的主要解锁：BN=256 在 aligned（inter=256）合法，在老 384 形状仍被拒。"""
    with reload_with(NANOV41_MOE_TILE="64,256,64,8,3") as m:
        assert m.BN == 256 and m.WARPS == 8
        m.check_tiles(*ALIGNED)
        with pytest.raises(AssertionError, match="BN"):
            m.check_tiles(*OLD)


def test_legacy_dw_retile_128_128_legal_for_aligned():
    """K2 给 legacy dW kernel 的「低流量」重分块（DW_BK=128 / DW_BN=128）在 aligned 合法。"""
    with reload_with(NANOV41_MOE_DW_TILE="128,128,128,8,3") as m:
        assert (m.DW_BM, m.DW_BK, m.DW_BN, m.DW_WARPS) == (128, 128, 128, 8)
        m.check_tiles(*ALIGNED)


def test_non_power_of_two_tile_rejected_at_parse():
    """BM/BN/BK 必须是 2 的幂（tl.arange 的硬约束）—— 在解析时就报错，别等到 JIT。"""
    with pytest.raises(AssertionError, match="power of 2"):
        with reload_with(NANOV41_MOE_TILE="64,96,64,4,3"):
            pass


def test_bad_warps_rejected():
    with pytest.raises(AssertionError, match="WARPS"):
        with reload_with(NANOV41_MOE_TILE="64,64,64,3,3"):
            pass


def test_bwd_bm_must_equal_fwd_bm():
    """`tile_e` 是按**前向 BM** 生成的（长度 P_max // BM），反向 kernel 用 `TILE_E[pid_m]` 索引
    自己单位的 M tile ⇒ BWD_BM 必须严格等于 BM。

    K2 之前的断言是 `BM % BWD_BM == 0`，允许 BWD_BM < BM；那会把子 tile 映射到错的 expert
    （pid_m 超过 tile_e 长度还会读越界），是**静默算错**。这里两个方向都要被拒。
    """
    for bwd_bm in (128, 32):
        with pytest.raises(AssertionError, match="BWD_BM"):
            with reload_with(NANOV41_MOE_TILE="64,64,64,4,3",
                             NANOV41_MOE_BWD_TILE=f"{bwd_bm},64,64,4,3"):
                pass
    with reload_with(NANOV41_MOE_TILE="64,64,64,4,3",
                     NANOV41_MOE_BWD_TILE="64,256,64,8,3") as m:
        assert (m.BWD_BM, m.BWD_BN) == (64, 256)
        m.check_tiles(*ALIGNED)


def test_grouped_dw_env_and_divisibility():
    """`NANOV41_MOE_DW_KERNEL=grouped` + 独立 tile（dW13 两个累加器 / dW2 一个）。"""
    with reload_with(NANOV41_MOE_DW_KERNEL="grouped",
                     NANOV41_MOE_DW13_TILE="128,64,32,4,3",
                     NANOV41_MOE_DW2_TILE="128,128,32,4,3",
                     NANOV41_MOE_DW13_EB="2", NANOV41_MOE_DW2_EB="8") as m:
        assert m.DW_KERNEL == "grouped"
        assert m.DWK13_TILE == (128, 64, 32, 4, 3) and m.DWK13_EB == 2
        assert m.DWK2_TILE == (128, 128, 32, 4, 3) and m.DWK2_EB == 8
        m.check_tiles(*ALIGNED)
        m.check_tiles(*OLD)


def test_grouped_dw_rejects_bad_td_tn():
    """dW2 的 M 维 = inter、N 维 = dim；TD 不整除 inter 必须报错（否则静默算错范围）。"""
    with reload_with(NANOV41_MOE_DW_KERNEL="grouped",
                     NANOV41_MOE_DW2_TILE="512,128,32,4,3") as m:
        with pytest.raises(AssertionError, match="DWK2_TD"):
            m.check_tiles(*ALIGNED)


def test_grouped_dw_tr_must_be_pow2_multiple_of_16():
    """TR（行方向分块）必须是 2 的幂且是 16 的倍数 —— 这是「与旧 kernel 逐位相同」的前提：
    bf16 MMA 每 16 行累加一段 fp32，TR=16 的倍数时累加顺序与旧 kernel 完全一致。"""
    with pytest.raises(AssertionError, match="power of 2"):
        with reload_with(NANOV41_MOE_DW_KERNEL="grouped",
                         NANOV41_MOE_DW13_TILE="128,64,24,4,3"):
            pass
    for tr in (16, 32, 64, 128):
        with reload_with(NANOV41_MOE_DW_KERNEL="grouped",
                         NANOV41_MOE_DW13_TILE=f"128,64,{tr},4,3") as m:
            assert m.DWK13_TILE[2] == tr and tr % 16 == 0
            m.check_tiles(*ALIGNED)


def test_fused_dx2_env_divisibility():
    with reload_with(NANOV41_MOE_FUSED_DX2="1", NANOV41_MOE_DX_TILE="64,128,64,4,3") as m:
        assert m.FUSED_DX2 is True and m.DX2_TILE == (64, 128, 64, 4, 3)
        m.check_tiles(*ALIGNED)
    with reload_with(NANOV41_MOE_FUSED_DX2="1", NANOV41_MOE_DX_TILE="64,1024,64,4,3") as m:
        with pytest.raises(AssertionError, match="DX2_BN"):
            m.check_tiles(*ALIGNED)
    # DX2_BM 必须等于前向 BM（tile_e 的索引单位）
    with reload_with(NANOV41_MOE_TILE="64,64,64,4,3",
                     NANOV41_MOE_FUSED_DX2="1", NANOV41_MOE_DX_TILE="128,128,64,4,3") as m:
        with pytest.raises(AssertionError, match="DX2_BM"):
            m.check_tiles(*ALIGNED)


def test_dw2_tight_flag():
    with reload_with(NANOV41_MOE_DW2_TIGHT="1") as m:
        assert m.DW2_TIGHT is True
    with reload_with(NANOV41_MOE_DW2_TIGHT="off") as m:
        assert m.DW2_TIGHT is False


def test_check_tiles_is_cached_per_shape():
    """第一次检查后按 (dim, inter, p_max) 缓存 —— 每层 forward 调一次，不能有可测开销。"""
    with reload_with() as m:
        m.check_tiles(*ALIGNED)
        assert ALIGNED in m._TILE_CHECKS
        m.check_tiles(*ALIGNED)          # 第二次走缓存（不重复 assert）
        assert m._TILE_CHECKS == {ALIGNED}


def test_moe_stacked_imports_without_triton():
    """`moe_stacked`/`moe_triton` 在无 triton 的 CPU 环境里必须能 import（生产 import 链）。"""
    import nanov41.moe_stacked as ms
    import nanov41.moe_triton as mt

    assert ms.tgm is tgm
    assert mt.tgm is tgm
    assert hasattr(ms, "patch_model") and hasattr(mt, "install")


def test_grouped_paths_raise_without_triton():
    """无 triton 时新 kernel 入口必须显式报错（而不是 AttributeError）。"""
    if tgm.HAS_TRITON:  # pragma: no cover - GPU 机器
        pytest.skip("triton available")
    for fn in (tgm.dw13_grouped, tgm.dw2_grouped, tgm.agg_forward2_tb):
        with pytest.raises(RuntimeError, match="triton"):
            fn(None, None, None, None, None)


# =========================================================================== kernel 语义（CPU 镜像）
# 下面两个测试用一个纯 torch 的「逐 program 镜像」复现 `_dw_grouped_kernel` 的**取行/分块/累加**
# 逻辑（GPU 上由 tests/test_moe_k2_kernels.py 对拍真正的 Triton kernel）。它能抓到设计层面的错误：
# tile 覆盖漏块/重复、EB 分组漏 expert、padded 行映射、TR 掩码。
def _fake_layout(counts, bm=64, ks=2, token_of_flat=None):
    """按 production 的 padding 布局手搓 (seg_start, count, flat_of_row, P_max)。

    不依赖 Triton（`MoELayout` 的 fast 路径是 Triton kernel，CPU 上跑不了）。
    """
    counts = torch.as_tensor(counts, dtype=torch.int64)
    e = counts.numel()
    padded = ((counts + bm - 1) // bm) * bm
    seg_start = torch.cat([torch.zeros(1, dtype=torch.int64), padded.cumsum(0)[:-1]]).to(torch.int32)
    cum = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)[:-1]])
    p = int(padded.sum())
    fof = torch.zeros(p, dtype=torch.int32)
    for i in range(e):
        for j in range(int(counts[i])):
            fof[int(seg_start[i]) + j] = int(cum[i] + j)
    return seg_start, counts.to(torch.int32), fof, p


def _dw_grouped_ref(x, dc1, dc3, flat, seg_start, count, e, d, n, ks,
                    td, tn, tr, eb, a_pad, dual):
    """`_dw_grouped_kernel` 的逐 program CPU 镜像（fp32 累加）。"""
    dw1 = torch.zeros(e, d, n, dtype=torch.float32)
    dw3 = torch.zeros(e, d, n, dtype=torch.float32) if dual else None
    xf, d1 = x.float(), dc1.float()
    d3 = dc3.float() if dual else None
    seg, cnts = seg_start.tolist(), count.tolist()
    for pb in range(e // eb):
        for pm in range(d // td):
            rm = torch.arange(pm * td, (pm + 1) * td)
            for pn in range(n // tn):
                rn = torch.arange(pn * tn, (pn + 1) * tn)
                for ei in range(eb):
                    ee = pb * eb + ei
                    start, cnt = seg[ee], int(cnts[ee])
                    acc1 = torch.zeros(td, tn)
                    acc3 = torch.zeros(td, tn)
                    for r0 in range(0, cnt, tr):
                        loc = torch.arange(r0, r0 + tr)
                        m = loc < cnt
                        padded = start + loc
                        src = padded if a_pad else (flat[padded].to(torch.int64) // ks)
                        a = torch.zeros(td, tr)
                        a[:, m] = xf[src[m]][:, rm].T
                        dc = torch.zeros(tr, tn)
                        dc[m] = d1[padded[m]][:, rn]
                        acc1 += a @ dc
                        if dual:
                            dc3b = torch.zeros(tr, tn)
                            dc3b[m] = d3[padded[m]][:, rn]
                            acc3 += a @ dc3b
                    dw1[ee, rm.unsqueeze(1), rn.unsqueeze(0)] = acc1
                    if dual:
                        dw3[ee, rm.unsqueeze(1), rn.unsqueeze(0)] = acc3
    return dw1, dw3


def _ref_dw(x, dc, seg_start, count, e, a_pad, flat, ks, d, n):
    """数学参考：逐 expert 用**真实行**做 Aᵀ @ DC（fp32）。"""
    out = torch.zeros(e, d, n, dtype=torch.float32)
    seg, cnts = seg_start.tolist(), count.tolist()
    xf, df = x.float(), dc.float()
    for ee in range(e):
        start, cnt = seg[ee], int(cnts[ee])
        rows = torch.arange(start, start + cnt)
        if cnt == 0:
            continue
        src = rows if a_pad else (flat[rows].to(torch.int64) // ks)
        out[ee] = xf[src].T @ df[rows]
    return out


@pytest.mark.parametrize("tile,eb", [
    ((32, 16, 16, 4, 3), 1), ((32, 16, 16, 4, 3), 2), ((16, 16, 32, 4, 3), 4),
    ((64, 8, 16, 4, 3), 8), ((32, 8, 64, 4, 3), 1),
])
def test_dw_grouped_kernel_semantics_match_reference(tile, eb):
    """任意 (TD,TN,TR,EB) 都必须覆盖且只覆盖一遍所有 (expert, d-tile, n-tile)。"""
    td, tn, tr = tile[0], tile[1], tile[2]
    e, dim, inter, t, ks = 8, 64, 32, 96, 2
    torch.manual_seed(0)
    counts = torch.tensor([13, 25, 64, 65, 7, 1, 12, 5])   # 混合：< BM、= BM、跨 block
    assert int(counts.sum()) == t * ks
    seg_start, count, fof, p = _fake_layout(counts, bm=64, ks=ks)
    x = torch.randn(t, dim)
    dgate = torch.randn(p, inter)
    dup = torch.randn(p, inter)
    got1, got3 = _dw_grouped_ref(x, dgate, dup, fof, seg_start, count, e, dim, inter, ks,
                                 td, tn, tr, eb, a_pad=False, dual=True)
    ref1 = _ref_dw(x, dgate, seg_start, count, e, False, fof, ks, dim, inter)
    ref3 = _ref_dw(x, dup, seg_start, count, e, False, fof, ks, dim, inter)
    for a, b, name in ((got1, ref1, "dW1"), (got3, ref3, "dW3")):
        rel = float((a - b).abs().max() / (b.abs().max() + 1e-30))
        assert rel < 1e-5, f"{name} rel {rel:.2e} at tile={tile} eb={eb}"

    # dW2：A = padded 缓冲（act），行号 = padded 行；输出 [E, inter, dim] ⇒ M=inter、N=dim
    if inter % td or dim % tn:
        # 非法 tile 由 wrapper 的 assert 挡住（`_dw_grouped_launch`: d % td / n % tn），这里跳过
        return
    act = torch.randn(p, inter)
    dy = torch.randn(p, dim)
    got2, _ = _dw_grouped_ref(act, dy, dy, fof, seg_start, count, e, inter, dim, ks,
                              td, tn, tr, eb, a_pad=True, dual=False)
    ref2 = _ref_dw(act, dy, seg_start, count, e, True, fof, ks, inter, dim)
    rel = float((got2 - ref2).abs().max() / (ref2.abs().max() + 1e-30))
    assert rel < 1e-5, f"dW2 rel {rel:.2e} at tile={tile} eb={eb}"


def test_dw_grouped_padding_rows_contribute_zero():
    """`DW2_TIGHT` 的依据：padding 行的 act 恒为 0 ⇒ 全行归约与只归约真实行**数值相同**。

    （GPU 上这一条升级为 torch.equal：见 tests/test_moe_k2_kernels.py。）
    """
    e, dim, inter, t, ks = 4, 32, 16, 40, 2
    counts = torch.tensor([10, 20, 3, 47])
    seg_start, count, fof, p = _fake_layout(counts, bm=64, ks=ks)
    act = torch.zeros(p, inter)
    valid = torch.zeros(p, dtype=torch.bool)
    for i in range(e):
        s, c = int(seg_start[i]), int(count[i])
        act[s:s + c] = torch.randn(c, inter)
        valid[s:s + c] = True
    dy = torch.randn(p, dim)
    full = _ref_dw(act, dy, seg_start, count + 0, e, True, fof, ks, inter, dim)
    # 用 padded 长度（seg_end）归约：多算的行 act == 0 ⇒ 结果不变
    seg_end = torch.cat([seg_start[1:], torch.tensor([p], dtype=torch.int32)])
    padded_len = (seg_end - seg_start).to(torch.int32)
    loose = _ref_dw(act, dy, seg_start, padded_len, e, True, fof, ks, inter, dim)
    assert torch.allclose(full, loose, atol=1e-6), "padding rows must contribute exactly 0"
    assert int(padded_len.sum()) > int(count.sum())
