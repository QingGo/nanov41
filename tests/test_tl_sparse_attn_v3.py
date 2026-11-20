# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""`tl_sparse_attn_v3` — 切分逻辑（CPU）+ 数值契约（GPU）。

* CPU 部分不依赖 tilelang/CUDA：验证 block 对齐的切分方案与 h padding 规则，
  以及没有 CUDA 时 `sparse_attn_v3` 会回退（不改变默认路径）。
* GPU 部分（`pytest.mark.skipif`）钉住 v3 的数值契约：
  - `rows=total` 的 `dq`/`dkv`/`dsink` 与 fp64 参考相差 ≤ 5e-3（= dq 的 bf16 输出台阶）；
  - `dkv=all, rows=block` 与 v1 的 `dq`/`dsink` 逐位一致（证明 kernel 重构没有改数学）；
  - v1 的 `dq` 与 fp64 参考相差 > 1e-2（记录 v1 的逐 block rowsum bug，避免被"修回去"）；
  - v3 的 fwd/bwd 可以被 CUDA graph 捕获并 replay。
"""
from __future__ import annotations

import pytest
import torch

from nanov41 import tl_sparse_attn_v3 as v3


# --------------------------------------------------------------------------- CPU
def test_hp_padding_rule():
    assert v3._hp_of(8) == 16          # aligned d12：h=8 补到 16（T.gemm 的 M 必须是 16 的倍数）
    assert v3._hp_of(16) == 16
    assert v3._hp_of(32) == 32
    assert v3._hp_of(4) == 16


def test_split_plan_aligned():
    p = v3._split_plan(128, 64, 8192, 8, 64)
    assert p == {"topk": 192, "nb": 3, "nwin": 2, "ncblk": 1, "hp": 16}


@pytest.mark.parametrize("win,nc,m,h,bk", [
    (128, 0, 8192, 8, 64),      # swa 层（没有 compress 段）→ 回退 v1
    (64, 32, 1024, 8, 64),      # nano d12：nc=32 不是 64 的倍数 → 回退
    (128, 64, 8192, 8, 48),     # window kernel 的 bk 不整除 m → 回退
    (96, 64, 8192, 8, 64),      # win 不是 64 的倍数 → 回退
    (128, 64, 8192, 24, 64),    # h=24 需要 padding 但 >16 → 回退
])
def test_split_plan_fallbacks(win, nc, m, h, bk):
    assert v3._split_plan(win, nc, m, h, bk) is None


def test_config_knobs_roundtrip():
    base = v3.get_config()
    try:
        v3.set_config(dkv="all", rows="block")
        assert v3.get_config()["dkv"] == "all"
        assert v3.get_config()["rows"] == "block"
        with pytest.raises(ValueError):
            v3.set_config(dkv="nope")
        with pytest.raises(ValueError):
            v3.set_config(rows="nope")
    finally:
        v3.set_config(**{k: base[k] for k in ("dkv", "rows")})


def test_cpu_fallback_matches_reference():
    """没有 CUDA 时 `sparse_attn_v3` 必须回退到参考实现（同数值）。"""
    if v3.available():
        pytest.skip("CUDA 可用：这条只验证回退路径")
    from nanov41.patched_env import sparse_attn as ref

    torch.manual_seed(0)
    b, m, h, d, win, nc, ratio = 1, 32, 8, 64, 8, 2, 2
    n = m + m // ratio
    q = torch.randn(b, m, h, d) * 0.2
    kv = torch.randn(b, n, d) * 0.2
    sink = torch.randn(h) * 0.1
    end = torch.arange(m).unsqueeze(1)
    widx = (end - win + 1).clamp_min(0) + torch.arange(win)
    widx = torch.where(widx > end, torch.full_like(widx, -1), widx)
    cidx = torch.randint(0, m // ratio, (b, m, nc)) + m
    idxs = torch.cat([widx.int().unsqueeze(0).expand(b, -1, -1), cidx.int()], -1).contiguous()
    got = v3.sparse_attn_v3(q, kv, sink, idxs, d ** -0.5)
    exp = ref(q, kv, sink, idxs, d ** -0.5)
    assert got.shape == exp.shape
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- GPU
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")


def _make(b, m, h=8, d=64, win=128, nc=64, ratio=2, seed=0):
    torch.manual_seed(seed)
    n = m + m // ratio
    q = torch.randn(b, m, h, d, device="cuda") * 0.2
    kv = torch.randn(b, n, d, device="cuda") * 0.2
    sink = torch.randn(h, device="cuda") * 0.1
    do = torch.randn(b, m, h, d, device="cuda") * 0.2
    end = torch.arange(m, device="cuda").unsqueeze(1)
    widx = (end - win + 1).clamp_min(0) + torch.arange(win, device="cuda")
    widx = torch.where(widx > end, torch.full_like(widx, -1), widx)
    reach = (torch.arange(1, m + 1, device="cuda") // ratio).unsqueeze(1)
    cidx = torch.argsort(torch.rand(b, m, n - m, device="cuda"), dim=-1)[..., :nc]
    cidx = torch.where(cidx < reach.unsqueeze(0), cidx + m, torch.full_like(cidx, -1))
    cidx, _ = cidx.sort(dim=-1)
    idxs = torch.cat([widx.int().unsqueeze(0).expand(b, -1, -1), cidx.int()], -1).contiguous()
    return q.bfloat16(), kv.bfloat16(), sink.float(), do.bfloat16(), idxs


def _fp64_ref(q, kv, sink, idxs, do, scale):
    b, m, h, d = q.shape
    n = kv.size(1)
    qq = q.double().requires_grad_(True)
    kk = kv.double().requires_grad_(True)
    ss = sink.double().requires_grad_(True)
    valid = idxs >= 0
    flat = kk.reshape(b * n, d)
    fi = (torch.arange(b, device=q.device).view(b, 1, 1) * n + idxs.clamp_min(0).long()).reshape(-1)
    kvg = flat.index_select(0, fi).reshape(b, m, idxs.size(-1), d)
    s = torch.einsum("bmhd,bmtd->bmht", qq, kvg) * scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = s.amax(dim=-1)
    p = (s - mx.unsqueeze(-1)).exp()
    den = p.sum(dim=-1) + torch.exp(ss.view(1, 1, h) - mx)
    out = torch.einsum("bmht,bmtd->bmhd", p, kvg) / den.unsqueeze(-1)
    (out * do.double()).sum().backward()
    return qq.grad, kk.grad, ss.grad


def _grads(fn, q, kv, sink, idxs, do, scale):
    qq = q.detach().clone().requires_grad_(True)
    kk = kv.detach().clone().requires_grad_(True)
    ss = sink.detach().clone().requires_grad_(True)
    y = fn(qq, kk, ss, idxs, scale)
    torch.autograd.backward(y, do)
    return y.detach(), qq.grad, kk.grad, ss.grad


def _rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


@CUDA
def test_gpu_v3_matches_fp64_reference():
    pytest.importorskip("tilelang")
    if not v3.available():
        pytest.skip("tilelang/cuda unavailable")
    q, kv, sink, do, idxs = _make(1, 512)
    scale = 64 ** -0.5
    dq_r, dkv_r, ds_r = _fp64_ref(q, kv, sink, idxs, do, scale)
    base = v3.get_config()
    try:
        v3.set_config(dkv="split", rows="total")
        p = v3.plan(q, kv, idxs)
        assert p is not None and p["nwin"] == 2 and p["ncblk"] == 1
        _, dq, dkv, ds = _grads(v3.sparse_attn_v3, q, kv, sink, idxs, do, scale)
        assert _rel(dq, dq_r) < 5e-3
        assert _rel(dkv, dkv_r) < 5e-3
        assert _rel(ds, ds_r) < 5e-3
        # v1 的逐 block rowsum：dq 明显偏（记录下来，禁止"修回去"）
        from nanov41 import tl_sparse_attn as v1
        if v1.available():
            _, dq1, _, _ = _grads(v1.sparse_attn_tl, q, kv, sink, idxs, do, scale)
            assert _rel(dq1, dq_r) > 1e-2
    finally:
        v3.set_config(**base)


@CUDA
def test_gpu_v3_all_block_is_bitwise_v1():
    pytest.importorskip("tilelang")
    if not v3.available():
        pytest.skip("tilelang/cuda unavailable")
    from nanov41 import tl_sparse_attn as v1

    if not v1.available():
        pytest.skip("tilelang v1 unavailable")
    q, kv, sink, do, idxs = _make(1, 512)
    scale = 64 ** -0.5
    base = v3.get_config()
    try:
        v3.set_config(dkv="all", rows="block")
        _, dq_v3, _, ds_v3 = _grads(v3.sparse_attn_v3, q, kv, sink, idxs, do, scale)
        _, dq_v1, _, ds_v1 = _grads(v1.sparse_attn_tl, q, kv, sink, idxs, do, scale)
        assert _rel(dq_v3, dq_v1) < 1e-6
        assert _rel(ds_v3, ds_v1) < 1e-6
    finally:
        v3.set_config(**base)


@CUDA
def test_gpu_v3_graph_capture():
    pytest.importorskip("tilelang")
    if not v3.available():
        pytest.skip("tilelang/cuda unavailable")
    q, kv, sink, do, idxs = _make(1, 512)
    scale = 64 ** -0.5
    base = v3.get_config()
    try:
        v3.set_config(dkv="split", rows="total")

        def build():
            qq = q.detach().clone().requires_grad_(True)
            kk = kv.detach().clone().requires_grad_(True)
            ss = sink.detach().clone().requires_grad_(True)
            torch.autograd.backward(v3.sparse_attn_v3(qq, kk, ss, idxs, scale), do)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                build()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            build()
        g.replay()
        torch.cuda.synchronize()
    finally:
        v3.set_config(**base)
