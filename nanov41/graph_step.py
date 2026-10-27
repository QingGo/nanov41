"""CUDA-graph capture of one d12 training step (P2-e).

为什么：quiet 5090 上 b=8/seq1024 一步 209.4 ms 里 device-busy 只有 170.3 ms，gap(CPU/launch)
39.1 ms；吵一点的实例更夸张（4090 实测 gap 61 ms，13844 个 kernel/step，CPU 侧
cudaLaunchKernel 63 ms）。CUDA graph 把整个 fwd+CE+bwd（可选 optimizer）录成一张图，
replay 时 CPU 只发一次 `cudaGraphLaunch`，gap 归零。

设计要点（踩过的坑都写在这里）：
1. **没有任何 CPU 同步**：捕获期间 `.item()/.tolist()/.cpu()` 会直接报错（PyTorch 保护），
   所以 MoE 必须用 grouped/张量版计数（`moe_grouped_forward_gmm` 或去 tolist 的 loop 版）。
2. **静态输入**：ids 是固定的 device tensor；每步"换数据"要么在捕获前固定，要么用
   graph 的 copy_ 把新数据写进静态 buffer（本 harness 提供 `set_inputs`，bench 用后者）。
3. **梯度缓冲要在捕获前就存在**：先 warmup 若干步，让 `.grad` 分配好；捕获时开头用
   `torch._foreach_zero_` 原地清零（不能 `set_to_none=True`，那会在捕获中分配新张量）。
4. **AdamW 默认放在图外**（`--graph-opt` 可放进图内）：fused AdamW 的 capturable 语义
   依赖版本，图外跑 optimizer 的 CPU 开销只有几 ms，而图内没有任何收益。
5. 形状必须固定：本 harness 要求 b/seq 固定（训练本来就是这样）。
6. 内存：捕获期间的新分配来自 graph 私有 pool，replay 复用同一批地址。

用法：
    gs = GraphStep(model, loss_fn, warmup=3)
    gs.capture()                      # 录图
    for _ in range(10): gs.replay()   # 计时用
    gs.set_ids(new_ids)               # 可选：把新 batch 拷进静态 buffer
"""
from __future__ import annotations

import gc

import torch


def _foreach_zero_grads(params) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch._foreach_zero_(grads)


class GraphStep:
    """把 (zero_grad + loss + backward [+ optimizer]) 捕获成一张 CUDA graph。"""

    def __init__(self, model, loss_fn, opt=None, graph_opt: bool = False, warmup: int = 3,
                 grad_mode: str = "backward"):
        """grad_mode:
          * "backward"    —— loss.backward()，梯度累加进 Parameter.grad（torch 的 AccumulateGrad）。
             实测在 torch 2.12 上会让整模型 fwd+bwd 捕获报 cudaErrorStreamCaptureImplicit。
          * "grad"        —— torch.autograd.grad(loss, params) 拿返回的梯度再赋给 p.grad：
             **没有 AccumulateGrad 节点**，绕开跨 stream 判定；捕获期新分配梯度张量（走 graph pool），
             replay 时同一批地址被重写，因此 p.grad 始终有效。
        """
        assert grad_mode in ("backward", "grad")
        self.model = model
        self.loss_fn = loss_fn
        self.opt = opt
        self.graph_opt = graph_opt
        self.warmup = warmup
        self.grad_mode = grad_mode
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.graph: torch.cuda.CUDAGraph | None = None
        self.loss = None
        self.device = next(model.parameters()).device
        self.capture_stats: dict = {}

    # ------------------------------------------------------------------ helpers
    def _zero_grads(self) -> None:
        _foreach_zero_grads(self.params)

    def _step_body(self):
        self.loss = self.loss_fn()
        if self.grad_mode == "grad":
            grads = torch.autograd.grad(self.loss, self.params, allow_unused=True)
            for p_, g_ in zip(self.params, grads):
                p_.grad = g_          # 纯 Python 赋值（捕获一次）；replay 重写同一批张量
        else:
            self._zero_grads()
            self.loss.backward()
        if self.graph_opt:
            assert self.opt is not None
            self.opt.step()

    def sync(self) -> None:
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ capture
    def capture(self) -> torch.cuda.CUDAGraph:
        """先 warmup（分配 grad / 编译 kernel / 建 optimizer state），再录图。

        两个必须做的卫生工作（否则捕获必失败，实测）：
        1. warmup 放在 side stream 上（PyTorch 官方 recipe）——让 grad 缓冲、cuBLAS workspace
           等都在同一个非默认 stream 上创建；
        2. 捕获前把 warmup 的 autograd 图引用清干净（`self.loss = None` + gc.collect()），
           否则 AccumulateGrad 节点带着上一次迭代的 stream 信息，触发
           "stream does not match ... break CUDA graph capture" 而失败。
        """
        # 之前如果跑过 eager step，grad 缓冲是在默认 stream 上创建的；直接进捕获会触发
        # AccumulateGrad stream mismatch（实测）。清成 None，让下面的 side-stream warmup 重建。
        for p_ in self.params:
            p_.grad = None
        gc.collect()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.loss = self.loss_fn()
                if self.grad_mode == "grad":
                    grads = torch.autograd.grad(self.loss, self.params, allow_unused=True)
                    for p_, g_ in zip(self.params, grads):
                        p_.grad = g_
                else:
                    self._zero_grads()
                    self.loss.backward()
                if self.opt is not None:
                    self.opt.step()
        torch.cuda.current_stream().wait_stream(side)
        self.loss = None
        gc.collect()
        torch.cuda.synchronize()
        # 有些 expert 在这个 batch 里一个 token 都没分到 → grad is None，这是正常的：
        # 捕获/重放里它保持 None（routing 固定），optimizer 图外也会跳过它。
        missing = [i for i, p in enumerate(self.params) if p.grad is None]
        self.capture_stats["no_grad_params"] = len(missing)
        self.capture_stats["grad_mode"] = self.grad_mode

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_body()
        self.graph = g
        self.capture_stats.update({"params": len(self.params), "graph_opt": self.graph_opt})
        return g

    # ------------------------------------------------------------------ replay
    def replay(self, n: int = 1) -> None:
        assert self.graph is not None, "先 capture()"
        for _ in range(n):
            self.graph.replay()

    def set_ids(self, ids: torch.Tensor) -> None:
        """把新 batch 写进捕获时用到的静态 ids（要求是同一个 tensor 对象）。

        用法：capture 前把 ids 存成 model 侧闭包捕获的 tensor，这里原地 copy_。
        """
        assert hasattr(self, "_static_ids"), "请在子类/闭包里把 static ids 交给 bind_static()"
        self._static_ids.copy_(ids, non_blocking=False)

    def bind_static(self, ids: torch.Tensor) -> None:
        self._static_ids = ids

    # ------------------------------------------------------------------ timing
    def time_replays(self, n: int = 10, warmup: int = 3) -> float:
        """返回每次 replay 的墙钟秒数（同步后计时；图内无 gap，墙钟≈device-busy）。"""
        import time
        self.replay(warmup)
        self.sync()
        t0 = time.perf_counter()
        self.replay(n)
        self.sync()
        return (time.perf_counter() - t0) / n


def capture_probe(fn, device: str = "cuda") -> tuple[bool, str]:
    """试捕获一个小闭包，返回 (ok, err)。用于探测某个 kernel/模块是否 graph-safe。"""
    try:
        fn()  # warmup
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - 探测用，任何失败都要报告
        return False, f"{type(exc).__name__}: {exc}"
