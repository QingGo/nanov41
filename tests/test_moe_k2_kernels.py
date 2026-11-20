# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""K2: 新做的两个 kernel 必须与生产路径**逐位相同**（CUDA + Triton）。

覆盖：
1. `_dw_grouped_kernel`（`dw13_grouped` / `dw2_grouped`）vs 生产 `_dw13_nopad_kernel` /
   `_dw_kernel`：**同一 tile** 时任意 `(TD,TN,TR,EB)` 都必须 `torch.equal`（fp32/bf16 的 MMA
   累加顺序完全一致）。
2. `NANOV41_MOE_DW2_TIGHT=1`：只归约真实行（跳过 padding 行）——padding 行对 dW 的贡献恰好 0。
3. `NANOV41_MOE_FUSED_DX2=1`：`_aggemm2_tb_kernel` 两个 fp32 累加器各自 round 到 bf16 再相加，
   与「两次 agg_forward_tb + 一次 bf16 add」逐位一致。
4. 端到端：`DW_KERNEL=grouped + FUSED_DX2=1` 的整层 fwd+bwd（y 与 5 个梯度）全部 `torch.equal`。

关于**跨 tile** 的比较（2026-11-11 修正）：`_dw_grouped_kernel` 按 `TR` 行分块累加
（`for r0 in range(0, cnt, TR): acc += tl.dot(a, dc)`，2 参数 dot ⇒ 每个 TR 行块一次 fp32 加），
而 legacy kernel 按 `DW_BM` 行分块。只有 `TR == DW_BM` 时两次 kernel 的「MMA 块序列」才完全一致
（`cnt <= min(TR, DW_BM)` 的单块情形也一致）；其它 TR 只是 fp32 加法结合顺序不同（数学相同，
bf16 输出最多差 1 ulp）。所以：
* 同一 tile（含生产 tile）→ `torch.equal`；
* 跨 tile → 按项目口径 `max|d|/max|ref| < 2e-2`（bf16 噪声级）断言，并打印 bitwise 标记。
另外，`num_stages x (A + DC[_1+_3])` 超过 sm_120 的 101 KB 共享内存预算的 tile 会在编译期报
Triton `OutOfResources`（不是数值错误）→ 测试跳过它们并保留生产 tile 必跑。

跑法（GPU）：
    /root/t141/bin/python -m pytest tests/test_moe_k2_kernels.py -q
"""
from __future__ import annotations

import pytest
import torch

import nanov41.triton_grouped_moe as tgm

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and tgm.HAS_TRITON),
    reason="needs CUDA + Triton for the grouped MoE kernels",
)

# aligned d12 的比例（dim != inter，是 K1 发现的非方阵陷阱）；T 小一点让测试几秒内跑完
E, DIM, INTER, T, K = 128, 512, 256, 512, 2

# sanity: 生产路径必须逐位相同的 bf16 噪声门（项目「worst parameter」口径 max|d|/max|ref|）
BF16_NOISE_REL = 2e-2


def _case(seed: int = 0, e: int = E, dim: int = DIM, inter: int = INTER, t: int = T, k: int = K):
    dev = "cuda"
    torch.manual_seed(seed)
    idx = torch.randint(0, e, (t * k,), device=dev).to(torch.int64)
    ws = (torch.rand(t * k, device=dev) * 0.5 + 0.5).to(torch.bfloat16)
    x = (torch.randn(t, dim, device=dev) * 0.05).to(torch.bfloat16)
    w1 = (torch.randn(e, dim, inter, device=dev) * 0.05).to(torch.bfloat16)
    w3 = (torch.randn(e, dim, inter, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(e, inter, dim, device=dev) * 0.05).to(torch.bfloat16)
    g = torch.randn(t, dim, device=dev, dtype=torch.float32) * 0.001
    return dict(e=e, dim=dim, inter=inter, t=t, k=k, idx=idx, ws=ws, x=x, w1=w1, w3=w3,
                w2=w2, g=g)


def _run_e2e(c: dict, dw_kernel: str = "legacy", fused_dx2: bool = False,
             dw2_tight: bool = False):
    """跑一次完整 fwd+bwd（FAST layout + NOPAD + FUSED_TAIL，即生产路径）。"""
    old = (tgm._LAYOUT, tgm.FAST_LAYOUT, tgm.NOPAD, tgm.FUSED_TAIL, tgm.FUSED_DY,
           tgm.DWS_INLINE, tgm.TB, tgm.DW_KERNEL, tgm.FUSED_DX2, tgm.DW2_TIGHT)
    tgm._LAYOUT, tgm.FAST_LAYOUT = "fast", True
    tgm.NOPAD = tgm.FUSED_TAIL = tgm.TB = True
    tgm.FUSED_DY, tgm.DWS_INLINE = False, True
    tgm.DW_KERNEL, tgm.FUSED_DX2, tgm.DW2_TIGHT = dw_kernel, fused_dx2, dw2_tight
    try:
        layout = tgm.MoELayout(c["idx"], c["e"], c["k"])   # flag 设定之后再建 layout
        tgm.check_tiles(c["dim"], c["inter"], layout.P_max)
        xs = [c["x"].clone().requires_grad_(True), c["ws"].clone().requires_grad_(True),
              c["w1"].clone().requires_grad_(True), c["w3"].clone().requires_grad_(True),
              c["w2"].clone().requires_grad_(True)]
        out = tgm.grouped_moe_experts_v2(xs[0], xs[2], xs[4], xs[3], layout, xs[1], 10.0)
        (out.float() * c["g"]).sum().backward()
    finally:
        (tgm._LAYOUT, tgm.FAST_LAYOUT, tgm.NOPAD, tgm.FUSED_TAIL, tgm.FUSED_DY,
         tgm.DWS_INLINE, tgm.TB, tgm.DW_KERNEL, tgm.FUSED_DX2, tgm.DW2_TIGHT) = old
    names = ["dx", "dws", "dW1", "dW3", "dW2"]
    return out.detach(), dict(zip(names, [t.grad.detach().clone() for t in xs], strict=True))


def _dw_delta(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """(bitwise, max|a-b|/max|b|) -- the project's worst-parameter convention."""
    eq = bool(torch.equal(a, b))
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return eq, (d / s if s else d)


def _is_resource_error(exc: BaseException) -> bool:
    """Triton 'tile does not fit this GPU' compile/launch failure (not a numeric mismatch)."""
    msg = str(exc).lower()
    return any(k in msg for k in ("out of resource", "shared memory", "too many resources",
                                  "outofresources", "ptxas", "resource_exhausted",
                                  "register"))


#: 生产可用的 grouped-dW tile 集合：(DWK13_TILE, DWK2_TILE, DWK13_EB, DWK2_EB)
#: 约束：TD 整除 dim(512)/inter(256)，TN 整除另一个，TR % 16 == 0，EB 整除 E(128)；
#: num_stages 选 2/3 让 `stages*(A+DC[_1+_3])` 留在 sm_120 的 101 KB 以内。
TILE_CASES = [
    # (DWK13: TD,TN,TR,W,S)  (DWK2: TD,TN,TR,W,S)   EB13 EB2
    ((128, 64, 32, 4, 3), (128, 128, 32, 4, 3), 4, 4),   # K2 默认 tile（生产候选）
    ((128, 32, 32, 4, 3), (64, 128, 32, 4, 3), 2, 2),    # TD/TN/EB 变化，TR 与默认相同
    ((64, 128, 32, 4, 3), (128, 64, 32, 4, 3), 8, 1),    # TD/TN 对调，EB 取两个极端
    ((128, 64, 64, 4, 2), (128, 128, 64, 4, 2), 4, 4),   # TR = 2 x 默认（多块累加）
    ((128, 64, 16, 8, 2), (128, 128, 16, 8, 2), 4, 4),   # TR = 默认/2
    ((64, 32, 128, 4, 2), (64, 64, 128, 4, 2), 4, 4),    # TR == legacy DW_BM（128）→ 必逐位
]


class _Tiles:
    """Temporarily swap the grouped-dW tile globals (restored even on exceptions)."""

    def __init__(self, t13, t2, eb13, eb2):
        self.new = (t13, t2, eb13, eb2)
        self.saved = None

    def __enter__(self):
        self.saved = (tgm.DWK13_TILE, tgm.DWK2_TILE, tgm.DWK13_EB, tgm.DWK2_EB)
        tgm.DWK13_TILE, tgm.DWK2_TILE, tgm.DWK13_EB, tgm.DWK2_EB = self.new
        return self

    def __exit__(self, *exc):
        (tgm.DWK13_TILE, tgm.DWK2_TILE,
         tgm.DWK13_EB, tgm.DWK2_EB) = self.saved
        return False


def test_grouped_dw_is_bitwise_identical_to_legacy():
    """同一 tile（K2 默认）下，grouped 与生产 legacy kernel 的 y + 5 个梯度逐位相同。"""
    c = _case()
    o_ref, g_ref = _run_e2e(c, dw_kernel="legacy")
    o_new, g_new = _run_e2e(c, dw_kernel="grouped")
    assert torch.equal(o_ref, o_new), "forward changed"
    for name in ("dx", "dws", "dW1", "dW3", "dW2"):
        rel = float((g_ref[name] - g_new[name]).norm() / (g_ref[name].norm() + 1e-30))
        assert torch.equal(g_ref[name], g_new[name]), f"{name} differs (rel {rel:.3e})"


def test_grouped_dw_tile_invariance():
    """跨 tile：同一 tile 逐位相同（生产 tile 必跑）；其它 tile 只允许 bf16 级 fp32 重结合差。

    小 T（~8 行/expert）下每次 kernel 都只有一个行块，因此**所有可行 tile** 都应逐位相同；
    这里对 `TR == DW_BM` 的 tile 断言 `torch.equal`，其余断言 `max|d|/max|ref| < 2e-2` 并打印
    bitwise 证据。超 smem 预算的 tile 跳过（Triton OutOfResources，不是数值错误）。
    """
    c = _case()
    _, g_ref = _run_e2e(c, dw_kernel="legacy")
    dw_bm = int(tgm.DW_BM)
    ran, skipped, report = 0, [], {}
    for t13, t2, eb13, eb2 in TILE_CASES:
        with _Tiles(t13, t2, eb13, eb2):
            try:
                _, g_new = _run_e2e(c, dw_kernel="grouped")
            except Exception as exc:                     # noqa: BLE001
                if _is_resource_error(exc):
                    skipped.append((t13, t2, str(exc).splitlines()[0][:110]))
                    continue
                raise
        ran += 1
        same_row_block = (t13[2] == dw_bm and t2[2] == dw_bm)
        for name in ("dW1", "dW3", "dW2"):
            eq, rel = _dw_delta(g_new[name], g_ref[name])
            report[f"{name}@{t13[2]}/{t2[2]}/EB{eb13}"] = (eq, round(rel, 9))
            assert rel < BF16_NOISE_REL, (
                f"{name} rel={rel:.3e} at T13={t13} T2={t2}: real numeric mismatch")
            if same_row_block:
                assert eq, f"{name} not bitwise at TR == DW_BM={dw_bm} ({t13}/{t2})"
    assert ran >= 3, f"only {ran} tile sets ran; skipped={skipped}"
    print(f"[k2] tile invariance: {ran} feasible tile sets, skipped={len(skipped)}; "
          f"bitwise flags={report}")
    if skipped:
        print(f"[k2] skipped (smem budget): {skipped}")


def test_grouped_dw_production_rows():
    """生产行数（每 expert ≈ 2 x DW_BM 行）下，换 tile 只允许 bf16 级 fp32 重结合差。

    这是 R75/R76 in-graph 跑法的真实 regime：legacy kernel 按 `DW_BM` 行分块、grouped 按 `TR`
    行分块，两者只在 `TR == DW_BM` 时逐位相同；其它 TR 是合法 fp32 重结合。用 2e-2 门保证
    「数学一致、没有真 bug」，并打印每个 tile 的 bitwise 标记。
    """
    c = _case(t=16384)                                  # 16384*2/128 = 256 行/expert
    _, g_ref = _run_e2e(c, dw_kernel="legacy")
    report, ran = {}, 0
    for t13, t2, eb13, eb2 in TILE_CASES:
        with _Tiles(t13, t2, eb13, eb2):
            try:
                _, g_new = _run_e2e(c, dw_kernel="grouped")
            except Exception as exc:                     # noqa: BLE001
                if _is_resource_error(exc):
                    continue
                raise
        ran += 1
        for name in ("dW1", "dW3", "dW2"):
            eq, rel = _dw_delta(g_new[name], g_ref[name])
            report[f"{name}@{t13[2]}/{t2[2]}/EB{eb13}"] = (eq, round(rel, 9))
            assert rel < BF16_NOISE_REL, (
                f"{name} rel={rel:.3e} at production rows, tile T13={t13} T2={t2}")
    assert ran >= 3, "no feasible tile set ran at production rows"
    print(f"[k2] production-rows grouped-vs-legacy: {ran} tile sets; {report}")


def test_dw2_tight_is_bitwise_identical():
    """跳过 padding 行（seg_start+count）不改任何 dW 的位。"""
    c = _case()
    _, g_ref = _run_e2e(c, dw_kernel="legacy", dw2_tight=False)
    _, g_new = _run_e2e(c, dw_kernel="legacy", dw2_tight=True)
    for name in ("dW1", "dW3", "dW2", "dx", "dws"):
        assert torch.equal(g_ref[name], g_new[name]), name


def test_fused_dx2_is_bitwise_identical():
    """单 kernel dx（两个 fp32 累加器各自 round 到 bf16 后相加）与两次 GEMM + add 逐位一致。"""
    c = _case()
    _, g_ref = _run_e2e(c)
    _, g_new = _run_e2e(c, fused_dx2=True)
    assert torch.equal(g_ref["dx"], g_new["dx"]), (
        f"dx rel {float((g_ref['dx'] - g_new['dx']).norm() / (g_ref['dx'].norm() + 1e-30))}")
    for name in ("dW1", "dW3", "dW2", "dws"):
        assert torch.equal(g_ref[name], g_new[name]), name


def test_full_k2_stack_bitwise_identical():
    """`DW_KERNEL=grouped + FUSED_DX2=1 + DW2_TIGHT=1` 与生产路径完全一致（含 y）。"""
    c = _case()
    o_ref, g_ref = _run_e2e(c)
    o_new, g_new = _run_e2e(c, dw_kernel="grouped", fused_dx2=True, dw2_tight=True)
    assert torch.equal(o_ref, o_new)
    for name in ("dx", "dws", "dW1", "dW3", "dW2"):
        assert torch.equal(g_ref[name], g_new[name]), name


def test_grouped_dw_requires_experts_divisible_by_eb():
    """E 不被 EB 整除时必须显式报错（否则 grid 会漏掉尾部 expert，静默算错）。"""
    c = _case(e=4, t=64, k=1)          # E=4 是 2 的幂（fast layout 要求）
    saved = tgm.DWK13_EB
    tgm.DWK13_EB = 8                   # 4 % 8 != 0
    try:
        with pytest.raises(AssertionError, match="EB"):
            _run_e2e(c, dw_kernel="grouped")
    finally:
        tgm.DWK13_EB = saved


def test_k2_stack_runs_with_k1():
    """k=1（每 token 只选 1 个 expert）也要跑通并保持逐位一致。"""
    c = _case(t=256, k=1)
    o_ref, g_ref = _run_e2e(c)
    o_new, g_new = _run_e2e(c, dw_kernel="grouped", fused_dx2=True, dw2_tight=True)
    assert torch.equal(o_ref, o_new)
    for name in ("dx", "dws", "dW1", "dW3", "dW2"):
        assert torch.equal(g_ref[name], g_new[name]), name
