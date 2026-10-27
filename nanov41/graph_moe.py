"""Graph-safe MoE: 循环版 MoE 的"静态分组大小"变体（CUDA graph 用）。

## 为什么需要它

`torch._grouped_mm` 内部要把 offs 拷到 host 拿每组大小，所以在 CUDA graph 捕获时会触发
"CPU<->CUDA copy during capture" 错误（实测 torch 2.12.1，sm89/sm120 都一样）。而 ref/loop 版
MoE 的 `torch.bincount(...).tolist()` 同样是 host 同步，无法捕获。

真实训练要 graph + MoE，正解是"device 侧 offsets 的 grouped GEMM"（DeepGEMM 那类）；这里先给出
一个**基准测试可用**的折中：Python 侧的每 expert token 数（几个 int）在捕获前烤死，所有张量
计算（gate/argsort/gather/GEMM/scatter）仍在图内每步重算。只要**权重和输入不变**，重算出的
分组与烤死的一致，结果逐位正确。

⚠️ 限制（必须写进报告）：
- 图内不含 optimizer step（否则权重变化 → 路由漂移 → 烤死的大小就不对了）。opt 时间单独量。
- 真实训练（每步新数据）需要 device-offsets grouped GEMM；见 S16 文档"下一步"。

## 用法

    from nanov41.graph_moe import record_static_routing, make_static_forward
    rec = record_static_routing(model, lambda: model(ids))   # 一次 eager dry-run，记录每层分组
    install(model, rec)                                      # 把每层 MoE 换成静态大小版
    # ... 然后 capture ...
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------- 记录一次路由
def _group_sizes(gate_out, n_experts: int, k: int) -> list[int]:
    _, idx = gate_out
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=n_experts)[:n_experts]
    return [int(c) for c in counts.tolist()]


def record_static_routing(model, forward_fn):
    """跑一次 eager forward，记录每个 MoE 模块本 batch 的 per-expert token 数。

    返回 {module_id: sizes}。dry-run 的输出丢弃（只在 capture 前调用一次）。
    """
    from nanov41.train_utils import moe_grouped_forward  # 复用现有分组逻辑

    recorded: dict[int, list[int]] = {}
    orig = {}

    def make_recorder(mod):
        def fwd(self, x, image_mask=None):
            shape = x.size()
            xf = x.view(-1, self.dim)
            w, idx = self.gate(xf, None if image_mask is None else image_mask.flatten())
            recorded[id(self)] = _group_sizes((w, idx), self.n_routed_experts,
                                              self.n_activated_experts)
            return moe_grouped_forward(self, x, image_mask)
        return fwd

    for mod in model.modules():
        if mod.__class__.__name__ == "MoE":
            orig[id(mod)] = mod.forward
            mod.forward = make_recorder(mod).__get__(mod, type(mod))
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for mod in model.modules():
            if id(mod) in orig:
                mod.forward = orig[id(mod)]
    return recorded


# ---------------------------------------------------------------- 静态大小前向
def make_static_forward(sizes: list[int]):
    """返回一个 MoE.forward 绑定版：每 expert 的 token 数来自 `sizes`（烤死），其余全在图内算。"""
    sizes = list(sizes)

    def fast_moe_static(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        w, idx = self.gate(x, None if image_mask is None else image_mask.flatten())
        T, k = idx.shape
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.div(order, k, rounding_mode="floor")
        xs = x[tok]
        ws = w.reshape(-1)[order]
        outs = []
        pos = 0
        for e, c in zip(range(self.experts_start_idx, self.experts_end_idx), sizes):
            if c == 0:
                continue
            outs.append(self.experts[e](xs[pos:pos + c], ws[pos:pos + c, None]))
            pos += c
        y_sorted = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
        flat_out = torch.empty(T * k, self.dim, dtype=y_sorted.dtype, device=y_sorted.device)
        flat_out[order] = y_sorted
        y = flat_out.view(T, k, self.dim).sum(1).to(x.dtype)
        return (y + self.shared_experts(x)).view(shape)

    return fast_moe_static


def install(model, recorded: dict[int, list[int]]) -> None:
    """把 model 里每个 MoE 换成静态路由前向（按 module id 匹配 dry-run 记录）。"""
    n = 0
    for mod in model.modules():
        if mod.__class__.__name__ == "MoE":
            sizes = recorded.get(id(mod))
            if sizes is None:
                raise RuntimeError(f"MoE 模块 {mod} 没有 dry-run 记录，先 record_static_routing()")
            mod.forward = make_static_forward(sizes).__get__(mod, type(mod))
            n += 1
    assert n > 0, "模型里没有 MoE 模块"
