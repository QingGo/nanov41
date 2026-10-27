#!/usr/bin/env python3
"""同进程内交错 A/B：对比 head 模式 (linear/fused) × MoE 模式 (loop/gmm)。

为什么要进程内交错：共享 GPU 主机的 CPU 争用/时钟漂移会让"分开跑两个进程"的 step 时间差
10–20%，必须交错测 + 取中位数。需要 CUDA。用法：
    python tools/ab_bench.py --b 8 --seq 1024 --configs linear:loop fused:loop fused:gmm
"""
import argparse
import statistics
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from nanov41.configs import D12                            # noqa: E402
from nanov41.patched_env import import_patched              # noqa: E402
M = import_patched()
from nanov41.model import NanoV41, init_random_weights      # noqa: E402
from nanov41.train_utils import (                           # noqa: E402
    moe_grouped_forward, moe_grouped_forward_gmm, head_bf16_forward, fused_head_ce,
)
from nanov41.tl_sinkhorn import hc_split_sinkhorn_tl        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--configs", nargs="+", default=["linear:loop", "fused:loop"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--compile", action="store_true",
                    help="对 step 用 torch.compile(mode=reduce-overhead)（P1-e：CUDA graph）")
    args = ap.parse_args()

    M.hc_split_sinkhorn = hc_split_sinkhorn_tl
    torch.manual_seed(0)
    cfg = dict(D12); cfg["max_batch_size"] = args.b
    model = NanoV41(cfg, param_dtype="bf16").cuda()
    init_random_weights(model, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    head_w = model.inner.head.weight
    ids = torch.randint(0, cfg["vocab_size"], (args.b, args.seq), device="cuda")

    def set_config(spec):
        head, moe = spec.split(":")
        M.ParallelHead.forward = (head_bf16_forward if head == "linear"
                                   else (lambda self, x, full_logits=False: x))
        M.MoE.forward = moe_grouped_forward_gmm if moe == "gmm" else moe_grouped_forward

    def step():
        if M.ParallelHead.forward is head_bf16_forward:
            loss = None
        # 统一按当前 patch 计算：linear 时 model 返回 logits，需要 CE
        out = model(ids)
        if getattr(out, "shape", (0,))[-1] == cfg["dim"]:
            loss = fused_head_ce(out, head_w, ids)
        else:
            from nanov41.train_utils import chunked_cross_entropy_shift
            loss = chunked_cross_entropy_shift(out, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def timed(n, warmup):
        fn = step_c if args.compile else step
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    if args.compile:
        torch._dynamo.config.suppress_errors = True
        global step_impl
        step_impl = torch.compile(step, mode="reduce-overhead", fullgraph=False)

    def step_c() -> None:
        step_impl()

    res = {c: [] for c in args.configs}
    for r in range(args.rounds):
        for spec in args.configs:
            set_config(spec)
            t = timed(args.steps, args.warmup)
            res[spec].append(t)
            print(f"  round{r} {spec}: {t:.1f} ms", flush=True)
    print()
    for spec in args.configs:
        med = statistics.median(res[spec])
        tok_s = args.b * args.seq / (med / 1000)
        print(f"{spec:16s} median {med:7.1f} ms  {tok_s:8.0f} tok/s  all={[round(x,1) for x in res[spec]]}")


if __name__ == "__main__":
    main()
