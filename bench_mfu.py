"""Per-component MFU / time breakdown for d12 training (P0 harness).

给每个顶层模块类别挂 record_function scope（前向 + 完整反向），profiler 跑几步，输出：
  category | FWD/BWD 设备时间 | 占比 | 理论 FLOPs | 有效 TF | precision-MFU | CPU 时间
外加整步：tokens/s、achieved TF、bf16-MFU、precision-MFU，以及 device-busy%（判断是否 CPU-bound）。

d12 的 FLOPs 拆分（FlopCounter 实测）：
  fwd 141.3 MFLOP/token，其中 head 70.3% / ffn(expert) 22.7% / attn 5.8% / other 1.2%；
  训练 = 3x forward。

正确性说明（v2 修的两个 bug）：
  1. 旧版在 exit 时新建 record_function 对象再 __exit__()，旧对象 self.record=None，scope 根本不会关闭；
     这里进入时保存 record_function 实例，退出时对同一实例 __exit__。
  2. 旧版对嵌套模块（ffn / ffn.gate / ffn.shared_experts）都挂 hook，同名 scope 在 profiler 里
     会重复累加；这里只给"每类最外层"模块挂 hook。

用法：
  python bench_mfu.py --b 8 --seq 1024                 # 默认 5090 bf16 峰值
  python bench_mfu.py --b 16 --seq 1024 --json /tmp/p0_b16.json
  python bench_mfu.py --b 8 --seq 1024 --peak fp8_5090 # 同一组数按 FP8 峰值算 MFU
"""
import argparse
import json
import re
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nanov41.configs import D12                            # noqa: E402
from nanov41.patched_env import import_patched             # noqa: E402
M = import_patched()
from nanov41.model import NanoV41, init_random_weights     # noqa: E402
from nanov41.train_utils import (                          # noqa: E402
    moe_grouped_forward, moe_grouped_forward_gmm, head_bf16_forward,
    chunked_cross_entropy, chunked_cross_entropy_shift, fused_head_ce,
)
from nanov41.tl_sinkhorn import hc_split_sinkhorn_tl       # noqa: E402

PEAK = {
    "bf16_5090": 209.5e12, "bf16_4090": 165.2e12,
    "fp8_5090": 419.0e12, "fp4_5090": 838.0e12,
    "bf16_h800": 989.0e12, "fp8_h800": 1979.0e12,
}
FWD_FLOPS_PER_TOKEN = 141.3e6
# 每类占 fwd FLOPs 的比例（其余 = other，含 CE、mHC、norm、hc mix 等）
SPLIT = {"head": 0.703, "moe": 0.227, "attn": 0.058, "embed": 0.0}
EXTRA_SCOPES = ("ce_fwd", "opt")   # 非模块 scope：CE 前向 / AdamW step


def categorize(name: str):
    n = name.lower()
    if n.endswith("head") or ".head" in n:
        return "head"
    if "ffn" in n or "moe" in n or "experts" in n:
        return "moe"
    if n.endswith("attn") or "attention" in n:
        return "attn"
    if n.endswith("embed") or "embedding" in n:
        return "embed"
    return None


class ScopeRecorder:
    """管理 record_function 的进入/退出（同一实例成对调用）。"""

    def __init__(self):
        self.stacks: dict[int, list] = {}

    def enter(self, key, kind, cat):
        rec = torch.autograd.profiler.record_function(f"{kind}:{cat}")
        rec.__enter__()
        self.stacks.setdefault(key, []).append(rec)

    def exit(self, key):
        lst = self.stacks.get(key)
        if lst:
            lst.pop().__exit__(None, None, None)


def instrument(model):
    """只给每类最外层模块挂 hook（避免嵌套同名 scope 重复计时）。返回 (hooks, 覆盖的模块名)。"""
    rec = ScopeRecorder()
    hooks, covered = [], set()
    for name, mod in model.named_modules():
        cat = categorize(name)
        if cat is None or name in covered:
            continue
        parts = name.split(".")
        if any(".".join(parts[:i]) in covered for i in range(1, len(parts))):
            continue                     # 任意祖先已覆盖同类，跳过嵌套子模块（否则同名 scope 重复计时）
        covered.add(name)
        key = id(mod)
        hooks.append(mod.register_forward_pre_hook(
            lambda m, a, c=cat, k=key: rec.enter(k, "FWD", c)))
        hooks.append(mod.register_forward_hook(
            lambda m, a, o, k=key: rec.exit(k)))
        hooks.append(mod.register_full_backward_pre_hook(
            lambda m, go, c=cat, k=key: rec.enter(k, "BWD", c)))
        hooks.append(mod.register_full_backward_hook(
            lambda m, gi, go, k=key: rec.exit(k)))
    return rec, hooks, covered


def parse_trace(path: str):
    """从 chrome trace 统计真实 device 时间（kernel/memcpy/memset）、kernel 次数与 top kernel。"""
    try:
        with open(path, encoding="utf-8") as f:
            tr = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0.0, 0, []
    dev_us, n_kernel = 0.0, 0
    by_name: dict[str, float] = {}
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(e.get("dur", 0.0))
        dev_us += dur
        if e.get("cat") == "kernel":
            n_kernel += 1
            nm = (e.get("name") or "?").split("(")[0][:90]
            by_name[nm] = by_name.get(nm, 0.0) + dur
    top = sorted(((n, round(d / 1000.0, 2)) for n, d in by_name.items()), key=lambda x: -x[1])
    return dev_us, n_kernel, top


def profiled_events(prof, steps):
    """把 key_averages 整理成 {name: (device_ms/step, cpu_ms/step, count)} 和 {name: is_cuda}。"""
    from torch.autograd import DeviceType
    out, is_cuda = {}, {}
    for ev in prof.key_averages():
        ms_dev = (getattr(ev, "self_device_time_total", None)
                  or getattr(ev, "cuda_time_total", 0)) / 1000.0 / steps
        ms_cpu = getattr(ev, "self_cpu_time_total", 0) / 1000.0 / steps
        out[ev.key] = (ms_dev, ms_cpu, ev.count)
        is_cuda[ev.key] = getattr(ev, "device_type", None) == DeviceType.CUDA
    return out, is_cuda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3, help="profiler 内的步数")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--peak", default="bf16_5090", choices=sorted(PEAK))
    ap.add_argument("--json", default=None, help="把结果写到这个 json 文件")
    ap.add_argument("--ce-mode", default="shift", choices=["shift", "slice"],
                    help="shift=零拷贝 CE（默认）；slice=旧版 logits[:, :-1].reshape")
    ap.add_argument("--head-mode", default="fused", choices=["fused", "linear", "check"],
                    help="fused=分块 head+CE（默认，不落全量 logits）；linear=旧路径；check=只对比 loss")
    ap.add_argument("--moe-mode", default="loop", choices=["loop", "gmm", "fast", "triton", "check"],
                    help="loop=逐 expert 循环；gmm=torch._grouped_mm；fast=fast_moe（w13 融合）；check=对比 loss")
    ap.add_argument("--attn-mode", default="ref", choices=["ref", "fast"],
                    help="ref=patched_env.sparse_attn；fast=fast_attn（band 分块，无 gather）")
    ap.add_argument("--mhc-mode", default="ref", choices=["ref", "fast", "fast-nx"],
                    help="ref=原 hc_pre/post/mixes；fast=fast_mhc exact；fast-nx=fast_mhc 激进档")
    ap.add_argument("--mhc-level", type=int, default=None,
                    help="fast_mhc 混合档 0=exact 1=+hc_post bf16 2=+hc_mixes bf16 3=全 bf16")
    # --- E2E harness 非侵入参数（默认行为与旧版完全一致）-------------------
    ap.add_argument("--no-profiler", action="store_true",
                    help="跳过 torch.profiler / chrome trace 段，只做 step 计时并输出摘要 JSON")
    ap.add_argument("--timed-steps", type=int, default=3,
                    help="计时步数（默认 3，与旧行为一致）；warmup 后取均值")
    args = ap.parse_args()
    if args.timed_steps < 1:
        raise SystemExit("--timed-steps 必须 >= 1")

    if not torch.cuda.is_available():
        print("bench_mfu.py requires CUDA (benchmarked on an RTX 5090); aborting.", file=sys.stderr)
        return 2
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"{props.total_memory / 2**30:.0f} GiB | b={args.b} seq={args.seq}", flush=True)

    if args.head_mode == "linear":
        M.ParallelHead.forward = head_bf16_forward
    else:
        # head 只返回 hidden，投影+CE 交给 fused_head_ce（分块，不落全量 logits）
        M.ParallelHead.forward = lambda self, x, full_logits=False: x
    if args.moe_mode == "gmm":
        M.MoE.forward = moe_grouped_forward_gmm
    elif args.moe_mode == "triton":
        import nanov41.moe_triton as moe_triton
        moe_triton.install(M)
    elif args.moe_mode == "fast":
        import nanov41.fast_moe as fast_moe
        fast_moe.install(M)
        print(f"fast_moe installed: {fast_moe.describe() if hasattr(fast_moe, 'describe') else 'on'}")
    elif args.moe_mode == "loop":
        M.MoE.forward = moe_grouped_forward
    M.hc_split_sinkhorn = hc_split_sinkhorn_tl
    if args.attn_mode == "fast":
        import nanov41.fast_attn as fast_attn
        fast_attn.reset_stats()
        fast_attn.install(M)

    cfg = dict(D12)
    cfg["max_batch_size"] = args.b
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="bf16").to(dev)
    init_random_weights(model, seed=0)
    head_weight = model.inner.head.weight
    if args.mhc_mode != "ref":
        import nanov41.fast_mhc as fast_mhc
        if args.mhc_level is not None:
            fast_mhc.set_mixed_level(args.mhc_level)
        # install 要传 patched 模块（M，含 Block），不是模型实例
        fast_mhc.install(M, exact=(args.mhc_mode == "fast"))
        print(f"fast_mhc installed exact={fast_mhc.get_mode()}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), fused=True)
    ids = torch.randint(0, cfg["vocab_size"], (args.b, args.seq), device=dev)
    rec, hooks, covered = instrument(model)
    print(f"instrumented: {sorted(covered)}", flush=True)

    def compute_loss(profile_mode=False):
        """fused: model 返回 hidden，fused_head_ce 里分块投影+CE；linear: 旧路径。"""
        if profile_mode:
            with torch.autograd.profiler.record_function("TOT:fwd"):
                out = model(ids)                     # 模块级 head/moe/attn scope 在其内部
            with torch.autograd.profiler.record_function("FWD:ce_fwd"):
                if args.head_mode in ("fused", "check"):
                    loss = fused_head_ce(out, head_weight, ids)
                elif args.ce_mode == "shift":
                    loss = chunked_cross_entropy_shift(out, ids)
                else:
                    loss = chunked_cross_entropy(out[:, :-1], ids[:, 1:])
            return loss
        out = model(ids)
        if args.head_mode in ("fused", "check"):
            return fused_head_ce(out, head_weight, ids)
        if args.ce_mode == "shift":
            return chunked_cross_entropy_shift(out, ids)
        return chunked_cross_entropy(out[:, :-1], ids[:, 1:])

    def step(profile_mode=False):
        if profile_mode:
            loss = compute_loss(True)
            opt.zero_grad(set_to_none=True)
            with torch.autograd.profiler.record_function("TOT:bwd"):
                loss.backward()
            with torch.autograd.profiler.record_function("X:opt"):
                opt.step()
            return loss
        loss = compute_loss(False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    if args.moe_mode == "check":
        # 同权重下对比 loop MoE 与 grouped_mm MoE 的 loss（应逐位一致）
        l_loop = compute_loss(False).item()
        M.MoE.forward = moe_grouped_forward_gmm
        l_gmm = compute_loss(False).item()
        print(f"CHECK moe loss: loop={l_loop:.6f} gmm={l_gmm:.6f} "
              f"diff={abs(l_loop - l_gmm):.3e}", flush=True)
        return {"check": True, "loop": l_loop, "gmm": l_gmm}

    if args.head_mode == "check":
        # 同权重下对比 fused 与 linear 两条路径的 loss（数值等价性）
        l_fused = compute_loss(False).item()
        M.ParallelHead.forward = head_bf16_forward      # 切回旧 head
        with torch.no_grad():
            logits = model(ids)
            l_linear = chunked_cross_entropy_shift(logits, ids).item()
        print(f"CHECK loss: fused={l_fused:.6f} linear={l_linear:.6f} "
              f"diff={abs(l_fused - l_linear):.3e}", flush=True)
        return {"check": True, "fused": l_fused, "linear": l_linear}

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        step()
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t0) / args.timed_steps

    tokens = args.b * args.seq
    flops_per_step = 3 * tokens * FWD_FLOPS_PER_TOKEN
    achieved = flops_per_step / step_s
    bf16_mfu = achieved / PEAK["bf16_5090"]
    used_peak = PEAK[args.peak]
    prec_mfu = achieved / used_peak

    if args.no_profiler:
        # E2E harness 路径：跳过 profiler / chrome trace，只输出计时摘要（step_ms 与旧版同口径）
        summary = {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "b": args.b, "seq": args.seq, "tokens_per_step": tokens,
            "step_ms": round(step_s * 1000, 2),
            "tokens_per_s": round(tokens / step_s, 1),
            "flops_per_step": round(flops_per_step / 1e9, 2),
            "achieved_tflops": round(achieved / 1e12, 2),
            "bf16_mfu_pct": round(bf16_mfu * 100, 2),
            "precision_mfu_pct": round(prec_mfu * 100, 2),
            "peak_used": args.peak,
            "head_mode": args.head_mode,
            "ce_mode": args.ce_mode,
            "moe_mode": args.moe_mode,
            "attn_mode": args.attn_mode,
            "mhc_mode": args.mhc_mode,
            "warmup": args.warmup,
            "timed_steps": args.timed_steps,
            "profiler": False,
        }
        print(f"\nstep {summary['step_ms']:.1f} ms | tokens/s {summary['tokens_per_s']:.0f} | "
              f"achieved {summary['achieved_tflops']:.1f} TF | bf16-MFU {summary['bf16_mfu_pct']:.1f}% "
              f"(no-profiler, warmup={args.warmup} timed={args.timed_steps})", flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nwritten {args.json}")
        for h in hooks:
            h.remove()
        return summary

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step(profile_mode=True)
    torch.cuda.synchronize()
    ev, ev_is_cuda = profiled_events(prof, args.steps)

    # device busy：从 chrome trace 精确统计真实 kernel/memcpy/memset 时长（op 事件会嵌套重复）
    trace_path = "/tmp/mfu_trace.json"
    prof.export_chrome_trace(trace_path)
    dev_us_total, kernel_counts, top_kernels = parse_trace(trace_path)
    top_kernels = [(n, round(ms / args.steps, 2)) for n, ms in top_kernels]  # trace 是 steps 步合计

    # 模块 + 额外 scope 的明细
    rows = []
    scoped_dev = 0.0
    for cat in ("head", "moe", "attn", "embed"):
        for kind in ("FWD", "BWD"):
            name = f"{kind}:{cat}"
            if name not in ev:
                continue
            ms, cpu_ms, cnt = ev[name]
            if ms <= 0 and cpu_ms <= 0:
                continue
            scoped_dev += ms
            fl = (3 if kind == "BWD" else 1) * tokens * FWD_FLOPS_PER_TOKEN * SPLIT.get(cat, 0.0)
            tf = fl / (ms / 1000) if ms > 0 else 0.0
            rows.append({"scope": name, "category": cat, "kind": kind,
                         "ms": round(ms, 2), "share": round(ms / (step_s * 1000) * 100, 1),
                         "tflops": round(tf / 1e12, 1),
                         "prec_mfu": round(tf / used_peak * 100, 1),
                         "cpu_ms": round(cpu_ms, 2), "count": cnt})
    for extra in EXTRA_SCOPES:
        for kind in ("FWD", "X"):
            name = f"{kind}:{extra}"
            if name in ev:
                ms, cpu_ms, cnt = ev[name]
                scoped_dev += ms
                rows.append({"scope": name, "category": extra, "kind": kind,
                             "ms": round(ms, 2), "share": round(ms / (step_s * 1000) * 100, 1),
                             "tflops": None, "prec_mfu": None,
                             "cpu_ms": round(cpu_ms, 2), "count": cnt})
    fwd_tot = ev.get("TOT:fwd", (0, 0, 0))[0]
    fwd_mod = sum(ev.get(f"FWD:{c}", (0, 0, 0))[0] for c in ("head", "moe", "attn", "embed"))
    bwd_mod = sum(ev.get(f"BWD:{c}", (0, 0, 0))[0] for c in ("head", "moe", "attn", "embed"))
    ce_ms = ev.get("FWD:ce_fwd", (0, 0, 0))[0]
    opt_ms = ev.get("X:opt", (0, 0, 0))[0]
    dev_ms_total = dev_us_total / 1000.0 / args.steps
    # 反向设备时间 = device busy - 前向 - CE 前向 - 优化器（含 CE 反向 / hc·norm 反向 / gradclip）
    bwd_total = dev_ms_total - fwd_tot - ce_ms - opt_ms
    fwd_extra = fwd_tot - fwd_mod
    bwd_extra = bwd_total - bwd_mod
    # 墙钟里既非设备时间也非 scope 计时 = CPU/启动空洞
    other_ms = step_s * 1000 - dev_ms_total

    if args.attn_mode == "fast":
        import nanov41.fast_attn as fast_attn
        summary_extra_attn = fast_attn.get_stats()
        print(f"\nfast_attn stats: {summary_extra_attn}", flush=True)
    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "b": args.b, "seq": args.seq, "tokens_per_step": tokens,
        "step_ms": round(step_s * 1000, 2),
        "tokens_per_s": round(tokens / step_s, 1),
        "flops_per_step": round(flops_per_step / 1e9, 2),
        "achieved_tflops": round(achieved / 1e12, 2),
        "bf16_mfu_pct": round(bf16_mfu * 100, 2),
        "precision_mfu_pct": round(prec_mfu * 100, 2),
        "peak_used": args.peak,
        "ce_mode": args.ce_mode,
        "head_mode": args.head_mode,
        "moe_mode": args.moe_mode,
        "device_busy_ms": round(dev_ms_total, 2),
        "device_busy_pct": round(dev_ms_total / (step_s * 1000) * 100, 1),
        "gap_ms": round(other_ms, 2),
        "fwd_total_ms": round(fwd_tot, 2),
        "bwd_total_ms": round(bwd_total, 2),
        "ce_fwd_ms": round(ce_ms, 2),
        "opt_ms": round(opt_ms, 2),
        "fwd_extra_ms": round(fwd_extra, 2),
        "bwd_extra_ms": round(bwd_extra, 2),
        "kernel_launches_per_step": int(round(kernel_counts / args.steps)),
        "top_kernels": [[n, ms] for n, ms in top_kernels[:15]],
        "rows": rows,
        "fwd_extra_share_pct": round(fwd_extra / (step_s * 1000) * 100, 1),
        "bwd_extra_share_pct": round(bwd_extra / (step_s * 1000) * 100, 1),
        "gap_share_pct": round(other_ms / (step_s * 1000) * 100, 1),
        "instrumented": sorted(covered),
    }

    print(f"\nstep {summary['step_ms']:.1f} ms | tokens/s {summary['tokens_per_s']:.0f} | "
          f"achieved {summary['achieved_tflops']:.1f} TF | bf16-MFU {summary['bf16_mfu_pct']:.1f}% | "
          f"precision-MFU({args.peak}) {summary['precision_mfu_pct']:.1f}%", flush=True)
    print(f"device-busy {summary['device_busy_pct']:.1f}% ({summary['device_busy_ms']:.1f} ms) of wall | "
          f"gap(CPU/launch) {summary['gap_ms']:.1f} ms | "
          f"kernel launches/step ~{summary['kernel_launches_per_step']}", flush=True)
    print("top kernels (device time):")
    for kname, kms in top_kernels[:6]:
        print(f"  {kms:8.2f} ms  {kname[:88]}")
    print(f"{'scope':12s} {'ms':>8s} {'share':>7s} {'TF':>7s} {'prec-MFU':>9s} {'CPU ms':>8s}")
    for r in rows:
        tf = f"{r['tflops']:.1f}" if r["tflops"] is not None else "-"
        mfu = f"{r['prec_mfu']:.1f}%" if r["prec_mfu"] is not None else "-"
        print(f"{r['scope']:12s} {r['ms']:8.2f} {r['share']:6.1f}% {tf:>7s} {mfu:>9s} "
              f"{r['cpu_ms']:8.2f}")
    print(f"{'fwd_extra':12s} {summary['fwd_extra_ms']:8.2f} "
          f"{summary['fwd_extra_share_pct']:6.1f}%   # hc mix/norm/其他（TOT:fwd - 模块 fwd）")
    print(f"{'bwd_extra':12s} {summary['bwd_extra_ms']:8.2f} "
          f"{summary['bwd_extra_share_pct']:6.1f}%   # CE 反向 + hc/norm 反向（TOT:bwd - 模块 bwd）")
    print(f"{'gap/cpu':12s} {summary['gap_ms']:8.2f} {summary['gap_share_pct']:6.1f}%"
          f"   # 墙钟 - device busy（CPU/启动空洞）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nwritten {args.json}")
    for h in hooks:
        h.remove()
    return summary


if __name__ == "__main__":
    main()
