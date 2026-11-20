#!/usr/bin/env python3
"""P2 / V4.1-aligned trainer: graph-captured nano-v41 on uint32 token shards.

One process per arm.  The R69 assignment-based CUDA-graph recipe is used (capture once, then
replay fwd+loss+bwd; the optimizers run outside the graph), with three additions that align the
training recipe with ``ref/`` (DeepSeek-V4.1-Flash report §2.5 / §4.2.2):

* **three-optimizer suite** (``--optimizer v41``, default): Muon (head-wise Q, update RMS 0.18,
  wd 0.1) + Sinkhorn momentum for Engram tables / token embedding / head (Algorithm 1:
  K=11, tau=1e-3, eps=1e-20, gamma=0.18, Engram LR x5) + AdamW for norms/biases
  (``nanov41/muon.py``).  ``--optimizer adamw`` keeps the old whole-model AdamW + sparse
  SparseRowAdam Engram baseline for A/B.
* **loss-free load balancing**: sequence-level balance loss (weight 1e-4) and the
  auxiliary-loss-free gate-bias update (speed 1e-3) from ``nanov41/v41_balance.py``.
* **gradient accumulation** to the aligned 2^19-token optimizer step (``--grad-accum``), because
  the official batch is 100.6M tokens while one graph replay must stay small on 32 GB.
* optimizer-state checkpoint/resume (``--save-optimizer`` + ``--resume``) and discounted-MFU
  accounting (``tools/mfu_accounting.py`` convention; logged here as raw bf16 MFU).

Usage (on the t141/R69 environment):
    source tools/final_env.sh
    /root/t141/bin/python tools/p2_train.py --arm D12-aligned-v41 \
        --config-json docs/aligned/d12_aligned.json \
        --tokens 500000000 --data-dir /root/autodl-tmp/climbmix_p2 \
        --batch 8 --seq 8192 --grad-accum 8 --peak-lr 8.22e-4 \
        --lr-schedule constant --warmup 100 --out runs/aligned/A-v41
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_bgt():
    return _load_module("p2_bgt", ROOT / "tools" / "bench_graph_train.py")


class TokenMixLoader:
    def __init__(self, data_dir: pathlib.Path, seq: int, batch: int, mix: dict[str, float],
                 seed: int):
        manifest = json.loads((data_dir / "manifest.json").read_text())
        self.arrays = {name: np.memmap(data_dir / f"{name}.u32.bin", dtype="<u4", mode="r")
                       for name in mix}
        self.counts = {name: int(manifest["domains"][name]["tokens"]) for name in mix}
        self.seq, self.batch = int(seq), int(batch)
        names = list(mix)
        probs = np.asarray([mix[n] for n in names], dtype=np.float64)
        self.names = names
        self.probs = probs / probs.sum()
        self.rng = np.random.default_rng(seed)
        self.tokens_per_step = self.seq * self.batch

    def next_batch(self) -> np.ndarray:
        out = np.empty((self.batch, self.seq), dtype=np.int64)
        domains = self.rng.choice(len(self.names), size=self.batch, p=self.probs)
        for i, d in enumerate(domains):
            arr = self.arrays[self.names[d]]
            n = int(arr.shape[0])
            off = int(self.rng.integers(0, max(1, n - self.seq)))
            out[i] = arr[off:off + self.seq]
        return out

    # ---------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if state and "rng" in state:
            self.rng.bit_generator.state = state["rng"]


def _state_to_cpu_dict(obj):
    """torch optimizer state_dict -> CPU tensors (portable checkpoint)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_state_to_cpu_dict(v) for v in obj)
    return obj


def _zero_grads(params) -> None:
    """Zero captured gradient buffers in place, dtype-grouped (mixed bf16/fp32 params)."""
    by_dtype: dict = {}
    for p in params:
        if p.grad is not None:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    for tensors in by_dtype.values():
        torch._foreach_zero_(tensors)


# ---------------------------------------------------------------- R79 memory-probe hooks
# All of these are env-gated and default OFF: with the env vars unset the training path, the math
# and the memory behaviour are bit-identical to before they existed.  They exist to (a) name the
# capture-time live-allocation anomaly that blocks 2^20-token optimizer steps and (b) test cheap
# fixes (warm-up length, allocator flush before/after capture, optimizer-state pre-allocation).
def _r79_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


def _r79_mem(tag: str) -> dict:
    """Print one machine-readable memory line ([r79] MEM {...}) and return it."""
    torch.cuda.synchronize()
    d = {
        "tag": tag,
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 4),
    }
    try:
        free, total = torch.cuda.mem_get_info()
        d["free_gib"] = round(free / 2**30, 4)
        d["total_gib"] = round(total / 2**30, 4)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never kill the run
        d["mem_get_info_error"] = repr(exc)[:160]
    print(f"[r79] MEM {json.dumps(d, sort_keys=True)}", flush=True)
    return d


def _r79_prealloc_optimizer_states(bundle, tables) -> dict:
    """Allocate the optimizers' lazily-created state tensors BEFORE graph capture.

    Env: ``NANOV41_R79_PREALLOC_STATES=1``.  Nothing here changes training math: every buffer it
    creates is the exact zero-initialised tensor the optimizer would create itself on its first
    ``step()``; the point is to move that ~5 GiB spike *before* the CUDA-graph pool exists so the
    post-capture first optimizer step cannot OOM on it.
    """
    info: dict = {}
    if bundle is None:
        return info
    # Preferred: the OptimizerBundle pre-allocation helper added in R79 (muon.py, o-opt's file).
    bundle_helper = getattr(bundle, "ensure_states_", None)
    if bundle_helper is not None:
        try:
            bytes_by_opt = bundle_helper()
            info["bundle_ensure_states"] = {k: int(v) for k, v in bytes_by_opt.items()}
            state_bytes = getattr(bundle, "state_bytes", None)
            if state_bytes is not None:
                info["state_bytes"] = {k: int(v) for k, v in state_bytes().items()}
            info["allocated_after_gib"] = round(torch.cuda.memory_allocated() / 2**30, 4)
            print(f"[r79] PREALLOC {json.dumps(info, sort_keys=True)}", flush=True)
            return info
        except Exception as exc:  # noqa: BLE001 - fall through to the generic path
            info["bundle_helper_error"] = f"{type(exc).__name__}: {exc}"[:200]

    def _alloc(tag: str, opt, keys=("momentum",), allow_generic: bool = True) -> None:
        if opt is None:
            return
        helper = (getattr(opt, "preallocate_states", None)
                  or getattr(opt, "_ensure_states_", None))
        if helper is not None:
            helper()
            info[tag] = "helper"
            return
        if not allow_generic:
            info[tag] = "skipped(no helper)"
            return
        n_alloc = 0
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state[p]
                for key in keys:
                    if key not in st:
                        st[key] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                        n_alloc += 1
        info[tag] = f"generic({n_alloc})"

    # Muon: fp32 momentum 2.5 GiB (bf16 variant -> 1.25 GiB; only the muon.py helper knows the
    # dtype rule, so the generic fallback is skipped when that env flag is on).
    _alloc("muon", getattr(bundle, "muon", None),
           allow_generic=not _r79_flag("NANOV41_MUON_MOMENTUM_BF16"))
    # Dense Sinkhorn (token embedding + head): fp32 momentum, 1.06 GiB total.
    _alloc("sinkhorn", getattr(bundle, "sinkhorn", None))
    # SinkhornEngram: dense fp32 (momentum, grad) buffers per table, created by _ensure(table).
    eng = getattr(bundle, "engram", None)
    if eng is not None and tables:
        if hasattr(eng, "_ensure"):
            for t in tables:
                eng._ensure(t)
            info["engram"] = "ensure"
        else:
            info["engram"] = "skipped(no _ensure)"
    info["allocated_after_gib"] = round(torch.cuda.memory_allocated() / 2**30, 4)
    print(f"[r79] PREALLOC {json.dumps(info, sort_keys=True)}", flush=True)
    return info


def parse_token_marks(spec: str) -> list[int]:
    """Parse a token-mark list for ``--save-at-tokens``.

    Accepts ``50m,100m,402653184`` (case-insensitive ``k/m/b`` suffixes, ``_`` group
    separators, ``;`` as an alternative separator) and returns sorted unique token counts.

    Why: WSD (warmup-stable-decay) screens want several *budget prefixes* out of ONE run, e.g.
    train to 100M tokens but evaluate at 25M/50M/100M to measure rank-stability between short
    and longer horizons (docs/r91-search-protocol.md).  The checkpoint is written at the first
    optimizer step whose cumulative token count reaches the mark.
    """
    marks: set[int] = set()
    for raw in (spec or "").replace(";", ",").split(","):
        part = raw.strip().lower().replace("_", "")
        if not part:
            continue
        mult = 1
        if part.endswith("m"):
            mult, part = 1_000_000, part[:-1]
        elif part.endswith("b"):
            mult, part = 1_000_000_000, part[:-1]
        elif part.endswith("k"):
            mult, part = 1_000, part[:-1]
        marks.add(int(round(float(part) * mult)))
    return sorted(marks)


def lr_at(step: int, total: int, peak: float, warmup: int = 100, *,
          schedule: str = "cosine", decay_start: float = 0.8, min_ratio: float = 0.1) -> float:
    """LR schedules.  ``cosine`` is the original (P2/5B) one.  ``wsd`` keeps the peak LR for the
    first ``decay_start`` fraction of the run, then cosine-decays to ``min_ratio``; ``constant``
    holds the peak (used for the intermediate 5B stages and for the aligned 0.5B A/B, which sits
    inside the official constant-LR phase)."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if schedule == "constant":
        return peak
    if schedule == "wsd":
        start = int(max(0.0, min(1.0, decay_start)) * total)
        if step < start:
            return peak
        progress = (step - start) / max(1, total - start)
        return peak * (min_ratio + (1.0 - min_ratio) * 0.5 *
                       (1.0 + math.cos(math.pi * min(1.0, progress))))
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--config", default=None, help="config name (default d12; with --config-json "
                                                  "the registered aligned name)")
    ap.add_argument("--config-json", default="",
                    help="docs/aligned/d12_aligned.json: register engine_config + engram preset")
    ap.add_argument("--engram", default=None,
                    help="none | d12 | d12_thin | d12_thick | <registered aligned name> "
                         "(default: auto = aligned name when --config-json is given, else none)")
    ap.add_argument("--tokens", type=int, default=600_000_000)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/p2_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step "
                         "(aligned target: batch*seq*grad-accum = 2^19 tokens)")
    ap.add_argument("--optimizer", default="v41", choices=["v41", "adamw"],
                    help="v41 = Muon+Sinkhorn+AdamW suite (report §2.5); adamw = baseline arm")
    ap.add_argument("--peak-lr", type=float, default=3e-3)
    ap.add_argument("--engram-lr", type=float, default=0.0,
                    help="absolute Engram LR (adamw arm only); default peak-lr * engram-lr-mult")
    ap.add_argument("--engram-lr-mult", type=float, default=5.0)
    ap.add_argument("--muon-wd", type=float, default=0.1)
    ap.add_argument("--no-headwise", action="store_true", help="disable head-wise Muon for Q")
    ap.add_argument("--adamw-weight-decay", type=float, default=0.1,
                    help="whole-model AdamW baseline arm weight decay")
    ap.add_argument("--adamw-eps", type=float, default=1e-8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd", "constant"])
    ap.add_argument("--decay-start", type=float, default=0.8)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--balance-loss", type=float, default=1e-4,
                    help="sequence-level balance loss weight (report: 1e-4; 0 disables)")
    ap.add_argument("--bias-speed", type=float, default=1e-3,
                    help="auxiliary-loss-free gate bias update speed (report: 1e-3; 0 disables)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--save-at-tokens", default="",
                    help="comma list of token marks (e.g. 50m,100m or raw ints): save a "
                         "model checkpoint at the first step whose cumulative tokens reach each "
                         "mark, named <out>.t<mark>.pt, and print a machine-readable "
                         "'[save-mark] mark=<int> step=<int> path=<path>' line. Used by the "
                         "WSD multi-point search screens (docs/r91-search-protocol.md); "
                         "optimizer state is not included unless --save-optimizer is also set.")
    ap.add_argument("--save-optimizer", action="store_true",
                    help="include dense optimizer state (+ Engram momentum) in checkpoints "
                         "(large: ~4-5 GB for the aligned d12; enables exact resume)")
    ap.add_argument("--resume", default="", help="resume model/optimizer/step from a checkpoint")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--init-from", default="", help="weights-only warm start")
    ap.add_argument("--abort-on-nan", action="store_true")
    ap.add_argument("--z-loss", type=float, default=0.0)
    ap.add_argument("--ema-decay", default="",
                    help="comma list of EMA decays for eval-time weight averaging, e.g. '0.99' or "
                         "'0.99,0.995' (empty = off). Each decay saves an extra checkpoint "
                         "'<out>.ema<decay>.pt' (model weights only) at the end of the run.")
    ap.add_argument("--ema-every", type=int, default=1,
                    help="update the EMA shadows every N optimizer steps (default 1; use 4 to cut "
                         "the ~10%%/step fp32 shadow cost to ~2.5%%)")

    ap.add_argument("--nan-check-every", type=int, default=1)
    ap.add_argument("--no-series", action="store_true",
                    help="disable the default {out}.series.json diagnostics writer "
                         "(per-log-point loss/lr/tok-s/MFU/gn/imb + final per-gate load summary); "
                         "logging only, never changes training numbers")
    ap.add_argument("--profile-loop", type=int, default=0, metavar="N",
                    help="after graph capture, run N timing steps that decompose the step into "
                         "per-segment GPU/host times, write <out>.loop_profile.json and exit "
                         "(0 disables; no checkpoint is written)")
    ap.add_argument("--mix", default="",
                    help="domain mix as name=weight,... (default: Marin phase-1 "
                         "web=0.926,code=0.061,math=0.013)")
    ap.add_argument("--peak-tflops", type=float, default=209.5,
                    help="dense bf16 peak TFLOPs for the MFU denominator (5090: 209.5, 4090: 165.2)")
    ap.add_argument("--fwd-flops-per-token", type=float, default=0.0,
                    help="override forward FLOPs/token (default: aligned JSON accounting, else "
                         "the analytic helper in tools/bench_graph_train.py)")
    args = ap.parse_args()

    accum = max(1, int(args.grad_accum))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    bgt = _load_bgt()

    # ---------------------------------------------------------------- aligned config
    aligned_accounting = None
    config_name, engram_name = args.config, args.engram
    if args.config_json:
        ab = _load_module("p2_aligned_build", ROOT / "tools" / "aligned_build.py")
        with_engram = args.engram != "none"
        registered = ab.install_aligned(args.config_json, with_engram=with_engram)
        if config_name is None:
            config_name = registered
        if engram_name is None:
            engram_name = registered if with_engram else "none"
        aligned_accounting = ab.aligned_accounting(args.config_json)
    else:
        config_name = config_name or "d12"
        engram_name = engram_name or "none"
    if engram_name == "none":
        engram_name = None

    model, cfg = bgt.build(args.batch, args.seq, mhc_level=3, post_triton=True,
                           mhc_split_triton=True, mhc_core_triton=True,
                           config_name=config_name, engram=engram_name)
    import nanov41.nano_engram as ne
    tables = ne.engram_tables(model) if engram_name else []

    # ---------------------------------------------------------------- loss-free balancing
    balance = None
    if args.balance_loss > 0 or args.bias_speed > 0:
        balance = _load_module("p2_v41_balance", ROOT / "nanov41" / "v41_balance.py")
        balance.install(model, bgt.M, weight=args.balance_loss)

    # ------------------------------------------------------- fused CE plan-A v2 (opt-in)
    # `NANOV41_CE_TRITON_V2=1` swaps the bf16-head CE backward from `torch.softmax` to
    # `triton_ce.ce_bwd` (dlogits from lse, no [chunk,V] softmax buffer). Default path untouched;
    # the installer also falls back automatically when the fp8 head is requested or z_loss != 0.
    if os.environ.get("NANOV41_CE_TRITON_V2", "0").strip().lower() not in ("", "0", "false", "off"):
        try:
            from nanov41 import fused_ce_triton as _ce_v2

            _ce_v2.maybe_install()
        except Exception as exc:  # noqa: BLE001 - optional fast path, never kill the run
            print(f"[ce] fused CE v2 install skipped: {exc}", flush=True)

    hw = model.inner.head.weight
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    param_names = [n for n, _ in named_params]
    dtype_set = {p.dtype for p in params}
    unused_grads: list[str] = []

    # ---------------------------------------------------------------- EMA shadows (eval-time)
    # Marin's WSD+EMA report: evaluating the EMA weights instead of the final weights is worth
    # ~0.015-0.017 bpb at 8B for free.  fp32 shadows (one per requested decay) are updated after the
    # optimizer step; the shadow is only used to write '<out>.ema<decay>.pt' at the end, so the
    # training path itself is unchanged.  Cost: ~10% of a step at --ema-every 1 (2.2 GiB fp32
    # shadow, mul+add over all params), ~2.5% at --ema-every 4.
    ema_decays: list[float] = [float(x) for x in str(args.ema_decay).replace(";", ",").split(",")
                               if x.strip()]
    # EMA covers every trainable parameter (incl. the Engram table rows: sparse row updates are
    # exactly what a decayed average handles; untouched rows keep their init value in both copies).
    ema_params = list(named_params)
    ema_shadows: list[dict[str, torch.Tensor]] = []
    if ema_decays:
        n_numel = sum(p.numel() for _, p in ema_params)
        for d in ema_decays:
            ema_shadows.append({n: p.detach().float().clone() for n, p in ema_params})
        print(f"[ema] decays={ema_decays} every={args.ema_every} shadow_params={n_numel/1e6:.1f}M "
              f"shadow_gib={n_numel*4*len(ema_decays)/2**30:.2f}", flush=True)

    # ---------------------------------------------------------------- optimizers
    import nanov41.muon as muon
    # R77: env-gated fast out-of-graph optimizer paths (harvest/Sinkhorn/clip/AdamW); default OFF,
    # enabled with NANOV41_OPT_FAST=1.  Guarded so a broken/missing module can never kill a run.
    try:
        from nanov41 import opt_fast as _opt_fast
    except Exception as _exc:  # noqa: BLE001
        _opt_fast = None
        print(f"[opt] opt_fast unavailable: {type(_exc).__name__}: {_exc}", flush=True)
    bundle = None
    opt = sparse = None
    if args.optimizer == "v41":
        bundle = muon.build_optimizers(model, cfg, base_lr=args.peak_lr,
                                       engram_tables=tables,
                                       engram_lr_mult=args.engram_lr_mult,
                                       headwise=not args.no_headwise,
                                       muon_wd=args.muon_wd, verbose=True)
        bundle.set_lr(args.peak_lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95),
                                weight_decay=args.adamw_weight_decay,
                                eps=args.adamw_eps,
                                fused=len(dtype_set) == 1)
        if tables:
            engram_lr = (args.engram_lr if args.engram_lr > 0
                         else args.peak_lr * args.engram_lr_mult)
            sparse = ne.SparseRowAdam(tables, lr=engram_lr)
            print(f"[opt] adamw baseline + sparse Engram Adam (engram_lr={engram_lr:.3e})",
                  flush=True)
        else:
            print("[opt] adamw baseline", flush=True)

    # ---------------------------------------------------------------- data
    default_mix = {"web": 0.926, "code": 0.061, "math": 0.013}
    if args.mix:
        mix = {}
        for item in args.mix.split(","):
            name, val = item.split("=")
            mix[name.strip()] = float(val)
        if abs(sum(mix.values()) - 1.0) > 1e-6:
            raise SystemExit(f"--mix weights must sum to 1.0, got {sum(mix.values()):.4f}")
    else:
        mix = default_mix
    loader = TokenMixLoader(pathlib.Path(args.data_dir), args.seq, args.batch, mix, args.seed)
    print(f"[mix] {mix}", flush=True)

    tokens_per_micro = args.batch * args.seq
    tokens_per_step = tokens_per_micro * accum
    total_steps = args.tokens // tokens_per_step
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # WSD multi-point screens: map each requested token mark to the 1-based optimizer step at
    # which the cumulative token count first reaches it (see docs/r91-search-protocol.md).
    mark_steps: dict[int, int] = {}
    if args.save_at_tokens:
        budget_tokens = total_steps * tokens_per_step
        try:
            _marks = parse_token_marks(args.save_at_tokens)
        except ValueError as exc:
            print(f"error: --save-at-tokens {args.save_at_tokens!r}: {exc}", file=sys.stderr)
            return 2
        for mark in _marks:
            if mark <= 0 or mark > budget_tokens:
                print(f"[save-mark] skip mark={mark} (run budget is {budget_tokens} tokens)",
                      flush=True)
                continue
            step_1b = (mark + tokens_per_step - 1) // tokens_per_step  # ceil(mark / tps)
            step_1b = max(1, min(step_1b, total_steps))
            if step_1b in mark_steps:
                print(f"[save-mark] skip mark={mark} (step {step_1b} already holds mark "
                      f"{mark_steps[step_1b]})", flush=True)
                continue
            mark_steps[step_1b] = mark
        if mark_steps:
            print(f"[save-mark] plan tokens_per_step={tokens_per_step} "
                  f"marks={{step: tokens}}={dict(sorted(mark_steps.items()))}", flush=True)

    # ---------------------------------------------------------------- FLOP accounting / resume
    if args.fwd_flops_per_token > 0:
        fwd_flops = float(args.fwd_flops_per_token)
    elif aligned_accounting:
        fwd_flops = float(aligned_accounting["fwd_flops_per_token_est"])
    else:
        fwd_flops = float(bgt._fwd_flops_per_token_model(model, cfg))
    peak_flops = float(args.peak_tflops) * 1e12

    start_step = 0
    loader_resume_state = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        sd = ck.get("state_dict", ck)
        res = model.load_state_dict(sd, strict=False)
        start_step = int(ck.get("step", 0))
        loader_resume_state = ck.get("loader_state")
        saved_kind = ck.get("optimizer_kind", "")
        saved_opt = ck.get("optimizer")
        if saved_opt is None:
            print("[resume] WARNING no optimizer state in the checkpoint: continuing with a FRESH "
                  "optimizer (save with --save-optimizer for an exact resume)", flush=True)
        elif saved_kind != args.optimizer:
            print(f"[resume] WARNING optimizer kind mismatch saved={saved_kind!r} "
                  f"now={args.optimizer!r}: fresh optimizer", flush=True)
        elif bundle is not None:
            bundle.load_state_dict(saved_opt)
            print(f"[resume] optimizer state restored ({saved_kind})", flush=True)
        elif opt is not None:
            opt.load_state_dict(saved_opt["adamw"])
            if sparse is not None and saved_opt.get("sparse") is not None:
                sparse.load_state_dict(saved_opt["sparse"])
            print("[resume] adamw + sparse optimizer state restored", flush=True)
        print(f"[resume] {args.resume}: step={start_step} missing={len(res.missing_keys)} "
              f"unexpected={len(res.unexpected_keys)}", flush=True)
    elif args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        sd = ck.get("state_dict", ck)
        try:
            res = model.load_state_dict(sd, strict=False)
            print(f"[init] loaded {args.init_from} (missing={len(res.missing_keys)}, "
                  f"unexpected={len(res.unexpected_keys)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[init] WARNING could not load {args.init_from}: {exc}", flush=True)

    if start_step and mark_steps:
        dropped = {s: m for s, m in mark_steps.items() if s <= start_step}
        if dropped:
            print(f"[save-mark] WARNING resuming from step {start_step}: marks {dropped} are "
                  f"already behind the resume point and will NOT be saved", flush=True)
            for s in dropped:
                mark_steps.pop(s, None)

    static_ids = torch.zeros((args.batch, args.seq), dtype=torch.int64, device="cuda")
    # loss accumulator updated by the captured graph so grad-accum loss is the micro-batch mean
    loss_acc = torch.zeros((), dtype=torch.float32, device="cuda")

    def loss_fn():
        loss = bgt.fused_head_ce(model(static_ids), hw, static_ids, z_loss=args.z_loss)
        if balance is not None:
            aux = balance.take_aux_losses(model)
            if aux:
                loss = loss + sum(aux)
        return loss

    _FOREACH_GRAD = os.environ.get("NANOV41_FOREACH_GRAD", "0").strip().lower() not in (
        "", "0", "false", "no", "off")
    # R77: fp8 head-weight quantisation cache refresh (per optimizer step; env-gated, no-op off).
    try:
        from nanov41 import head_fwdgrad as _head_fwdgrad
    except Exception:  # noqa: BLE001
        _head_fwdgrad = None

    def _hc_maybe_refresh():
        if _head_fwdgrad is None:
            return
        try:
            _head_fwdgrad.maybe_refresh_head_weight_cache()
        except Exception as _exc:  # noqa: BLE001
            print(f"[head] wcache refresh failed: {type(_exc).__name__}: {_exc}", flush=True)

    # R82: fused FP8 linear weight cache refresh (NANOV41_LINEAR_FP8_V2=1; no-op otherwise).
    try:
        from nanov41 import fused_fp8_linear as _ff8v2
    except Exception:  # noqa: BLE001
        _ff8v2 = None

    _ff8v2_seen = [False]

    def _ff8v2_maybe_refresh():
        if _ff8v2 is None:
            return
        try:
            n_ = _ff8v2.maybe_refresh()
            if n_ and not _ff8v2_seen[0]:
                _ff8v2_seen[0] = True
                try:
                    from nanov41 import fp8_cast_registry as _cr
                    _h, _m = _cr.hits()
                    _st = _ff8v2.stats()
                    print(f"[fp8v2] refreshed {n_} weight slots (producer-cast hits={_h} miss={_m}; "
                          f"wgrad fp8={_st.get('wgrad_fp8_calls', 0)} "
                          f"bf16={_st.get('wgrad_bf16_calls', 0)})", flush=True)
                except Exception:  # noqa: BLE001
                    print(f"[fp8v2] refreshed {n_} weight slots", flush=True)
        except Exception as _exc:  # noqa: BLE001
            print(f"[fp8v2] refresh failed: {type(_exc).__name__}: {_exc}", flush=True)

    def grad_body():
        raw = loss_fn()
        leaves = [v for t in tables for (_, v) in t._records]
        grads = torch.autograd.grad(raw * (1.0 / accum), params + leaves, allow_unused=True)
        if not torch.is_grad_enabled():  # won't happen; cheap guard against autograd misuse
            raise RuntimeError("grad_body must run with grad enabled")
        for nm, g_ in zip(param_names, grads[:len(params)]):
            if g_ is None and nm not in unused_grads:
                unused_grads.append(nm)
        if accum == 1:
            for p_, g_ in zip(params, grads[:len(params)]):
                if g_ is not None:
                    p_.grad = g_.to(p_.dtype)
        elif _FOREACH_GRAD:
            # R77: one fused foreach add per (dtype, device) group instead of one CUDA kernel
            # per parameter.  Bitwise identical math (each in-place add is a+b in the grad
            # dtype); the win is launch count: 303 add_ kernels/replay -> ~2.  On the 5090
            # this is 6.37 ms/replay of graph time (3.18 ms/8k, ~204 ms/optimizer step).
            _groups: dict = {}
            for p_, g_ in zip(params, grads[:len(params)]):
                if g_ is None:
                    continue
                if p_.grad is None:  # only at capture time
                    p_.grad = torch.zeros_like(g_)
                g2 = g_.to(p_.dtype)
                _groups.setdefault((p_.dtype, p_.device), ([], []))
                _groups[(p_.dtype, p_.device)][0].append(p_.grad)
                _groups[(p_.dtype, p_.device)][1].append(g2)
            for _dsts, _srcs in _groups.values():
                torch._foreach_add_(_dsts, _srcs)
        else:
            for p_, g_ in zip(params, grads[:len(params)]):
                if g_ is not None:
                    if p_.grad is None:  # only at capture time
                        p_.grad = torch.zeros_like(g_)
                    p_.grad.add_(g_.to(p_.dtype))
        for leaf, g_ in zip(leaves, grads[len(params):]):
            if g_ is not None:
                leaf.grad = g_
        loss_acc.add_(raw.detach().float())  # captured: mean over accumulated micro-batches
        return raw

    for p in params:
        p.grad = None
    if accum > 1:
        for p in params:
            p.grad = torch.zeros_like(p)
    for t in tables:
        t.clear_records()
        t.set_static_records(False)
    gc_mod = __import__("gc")
    gc_mod.collect()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    # R79: the pre-capture warm-up length is env-overridable (default = the historical 2*accum
    # eager micro-batches, i.e. two full grad-accum cycles).  NANOV41_R79_WARMUP_MICROS=N runs N
    # eager micro-batches total (one is enough to JIT/autotune every fixed-shape kernel; the R77
    # accum64 anomaly is a *live-allocation* spike that scales with this loop, so testing 1-2 is
    # the point).  NANOV41_R79_TRACE_WARMUP=1 prints allocated/reserved per iteration.
    _warm_iters = 2 * accum
    _warm_env = os.environ.get("NANOV41_R79_WARMUP_MICROS", "").strip()
    if _warm_env:
        _warm_iters = max(1, int(_warm_env))
    _trace_warm = _r79_flag("NANOV41_R79_TRACE_WARMUP")
    if _warm_iters != 2 * accum or _trace_warm:
        print(f"[r79] warmup micro-batches = {_warm_iters} (default {2 * accum}) "
              f"trace={int(_trace_warm)}", flush=True)
    with torch.cuda.stream(side):
        for _wi in range(_warm_iters):
            static_ids.copy_(torch.from_numpy(loader.next_batch()).cuda(non_blocking=True))
            grad_body()
            for t in tables:
                t.clear_records()
            if _trace_warm:
                torch.cuda.synchronize()
                print(f"[r79] WARMUP_ITER {_wi + 1}/{_warm_iters} "
                      f"alloc_gib={torch.cuda.memory_allocated() / 2**30:.4f} "
                      f"reserved_gib={torch.cuda.memory_reserved() / 2**30:.4f} "
                      f"peak_alloc_gib={torch.cuda.max_memory_allocated() / 2**30:.4f}",
                      flush=True)
    torch.cuda.current_stream().wait_stream(side)
    gc_mod.collect()
    torch.cuda.synchronize()
    for t in tables:
        t.clear_records()
        t.set_static_records(False)
    if accum > 1:
        _zero_grads(params)
    loss_acc.zero_()
    # R79 pre-capture hooks (default OFF): move the optimizer-state spike before the graph pool
    # exists, and/or flush the allocator's cached (fragmented) blocks so the capture can start from
    # a clean slate.  Neither changes the training math.
    if _r79_flag("NANOV41_R79_PREALLOC_STATES"):
        torch.cuda.synchronize()
        _r79_mem("prealloc_before")
        _r79_prealloc_optimizer_states(bundle, tables)
        _r79_mem("prealloc_after")
    if _r79_flag("NANOV41_R79_PRE_CAPTURE_CACHE"):
        torch.cuda.synchronize()
        _r79_mem("pre_capture_cache_flush_before")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _r79_mem("pre_capture_cache_flush_after")
    elif _r79_flag("NANOV41_R79_MEM_TRACE"):
        _r79_mem("pre_capture")
    g = torch.cuda.CUDAGraph()
    override = override_fn = None
    if os.environ.get("NANOV41_CAPTURE_TRY_OVERRIDE", "0").lower() not in ("0", "false", "no", "off"):
        override_fn = getattr(getattr(torch.autograd, "graph", None),
                              "set_override_stale_capture_stream", None)
        if override_fn is not None:
            try:
                override = override_fn(True)
                print("[capture] set_override_stale_capture_stream(True)", flush=True)
            except Exception as exc:  # noqa: BLE001 - best-effort escape hatch
                print(f"[capture] override unavailable: {exc}", flush=True)
                override_fn = None
    try:
        with torch.cuda.graph(g, stream=side):
            g_loss = grad_body()
    finally:
        if override_fn is not None:
            try:
                if override is not None and hasattr(override, "__exit__"):
                    override.__exit__(None, None, None)
                else:
                    override_fn(False)
            except Exception:  # noqa: BLE001
                pass
    for t in tables:
        t.set_static_records(True, zero_grads=False)  # assignment-captured leaf grads
    if balance is not None:
        balance.take_aux_losses(model)  # drop capture-time list entries
    # R79 post-capture allocator flush (default OFF): the first optimizer step creates ~5 GiB of
    # state *after* the graph pool exists.  Warm-up leaves a heavily fragmented cache behind, so
    # flush the cached (unused) blocks now -- graph pools are held by CUDAGraph itself, so replay
    # is unaffected (verified by re-running the profile steps in the same process).
    if _r79_flag("NANOV41_R79_POST_CAPTURE_CACHE"):
        _r79_mem("post_capture_before_cache_flush")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _r79_mem("post_capture_after_cache_flush")
        try:  # smoke-test that the captured graph still replays after empty_cache()
            g.replay()
            torch.cuda.synchronize()
            print("[r79] POST_FLUSH_REPLAY_OK", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[r79] POST_FLUSH_REPLAY_FAILED {type(exc).__name__}: {exc}", flush=True)
    elif _r79_flag("NANOV41_R79_MEM_TRACE"):
        _r79_mem("post_capture")
    head_env = os.environ.get("NANOV41_HEAD_FP8", "0").strip().lower()
    ce_chunk = os.environ.get("NANOV41_CE_CHUNK", "default")
    v3_ok = cfg.get("dim") == 384
    # v2 is the generic-dim fp8 path (`torch._scaled_mm`); v2/v3 silently fall back to the bf16
    # `_FusedHeadCE` when unavailable, so print the *resolved* implementation for the log record.
    fp8_off = head_env in ("", "0", "false", "off")
    if head_env in ("3", "v3"):
        head_impl = "fp8-v3" if v3_ok else "bf16 chunked (v3 needs dim=384)"
    elif head_env in ("2", "v2"):
        try:
            from nanov41 import fp8_head_v2 as _f8v2

            head_impl = "fp8-v2 (generic dim, _scaled_mm)" if _f8v2.available() else \
                "bf16 chunked (fp8-v2 unavailable on this host)"
        except Exception:  # noqa: BLE001
            head_impl = "bf16 chunked (fp8-v2 import failed)"
    else:
        head_impl = "bf16 chunked"
    print(f"[head] NANOV41_HEAD_FP8={head_env} dim={cfg.get('dim')} "
          f"CE_CHUNK={ce_chunk} -> {head_impl}", flush=True)
    # Only the bf16 `_FusedHeadCE` needs chunk % seq == 0 (graph-safe CE backward); the fp8 v2/v3
    # paths have no such constraint, so the warning only applies to the bf16 path.
    if (fp8_off and isinstance(ce_chunk, str) and ce_chunk.isdigit()
            and int(ce_chunk) % args.seq != 0):
        print(f"[warn] NANOV41_CE_CHUNK={ce_chunk} is not a multiple of seq={args.seq}: "
              f"the bf16 CE backward takes a value-dependent mask path (not CUDA-graph safe)",
              flush=True)
    print(f"[capture] PASS arm={args.arm} cfg={config_name} engram={engram_name} "
          f"accum={accum} tokens/step={tokens_per_step} "
          f"table_params={(ne.engram_param_count(model)/1e6) if tables else 0.0:.2f}M "
          f"fwd_flops/token={fwd_flops/1e6:.2f}M", flush=True)

    if _r79_flag("NANOV41_PROFILE_KERNELS"):
        # R86: per-kernel breakdown of one warm-graph replay (device time).  Exits the process
        # right after capture, so it composes with the normal production env flags.
        _n_prof = int(os.environ.get("NANOV41_PROFILE_REPLAYS", "3"))
        torch.cuda.synchronize()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        from torch.profiler import profile as _tprof, ProfilerActivity as _PA
        with _tprof(activities=[_PA.CUDA]) as _prof:
            for _ in range(_n_prof):
                g.replay()
            torch.cuda.synchronize()
        _rows = list(_prof.key_averages())
        _rows.sort(key=lambda e: e.self_device_time_total, reverse=True)
        _tot = sum(e.self_device_time_total for e in _rows) / max(_n_prof, 1)
        print(f"[kern] per-replay CUDA total {_tot:.1f}us over {_n_prof} replays", flush=True)
        _ksum = sum(e.count for e in _rows)
        for _e in _rows[:80]:
            _per = _e.self_device_time_total / max(_n_prof, 1)
            if _per <= 0:
                continue
            print(f"[kern] {100.0 * _per / _tot:6.2f}% {_per:9.1f}us "
                  f"n={_e.count / max(_n_prof, 1):6.1f} {_e.key[:100]}", flush=True)
        print(f"[kern] DONE total_kernel_calls={_ksum}", flush=True)
        raise SystemExit(0)

    if unused_grads:
        print(f"[grad] WARNING {len(unused_grads)} params got no gradient during capture "
              f"(first 3: {unused_grads[:3]})", flush=True)
        # Known case: the CSA2 indexer has no differentiable path in the released reference
        # code (Top-K indices are discrete), so those params only get the decoupled weight
        # decay when `accum>1` pre-zeroes every grad.  Drop the buffers so the optimizers skip
        # them (they stay at init; the official indexer training objective is an M2 item).
        named_map = dict(named_params)
        dropped = 0
        for nm in unused_grads:
            q = named_map.get(nm)
            if q is not None and q.grad is not None:
                q.grad = None
                dropped += 1
        if dropped:
            print(f"[grad] dropped pre-zeroed grad buffers for {dropped} no-gradient params "
                  f"(frozen, no decoupled weight decay)", flush=True)
    else:
        print(f"[grad] all {len(params)} trainable params received gradients", flush=True)
    if loader_resume_state:
        loader.load_state_dict(loader_resume_state)

    # ---------------------------------------------------------------- series diagnostics
    # {out}.series.json: one point per log line (loss/lr/tok-s/MFU/gn/imb/elapsed) plus a final
    # per-gate load summary (max/mean/min + top-5 experts per layer).  Default ON; --no-series
    # disables.  Diagnostics only: nothing here touches the training math, and every write is
    # guarded so a filesystem hiccup can never kill an A/B run.
    series = None
    gate_load_sums: dict[str, torch.Tensor] = {}
    gate_load_n = 0
    series_path = None if args.no_series else out.with_suffix(".series.json")
    if series_path is not None:
        _env_keys = ("NANOV41_HEAD_FP8", "NANOV41_CE_CHUNK", "NANOV41_ATTN_V2", "FAST_ATTN_TL",
                     "NANOV41_MOE_STACKED", "NANOV41_MOE_DW_TILE", "NANOV41_MOE_BWD_TILE",
                     "NANOV41_MUON_RMS", "NANOV41_SINKHORN_GAMMA", "NANOV41_ENGRAM_LR_MULT",
                     "NANOV41_MOE_LAYOUT", "PYTORCH_CUDA_ALLOC_CONF")
        series = {"arm": args.arm, "out": str(out), "config": config_name,
                  "engram": engram_name, "optimizer": args.optimizer,
                  "resume": bool(args.resume), "start_step": start_step,
                  "steps_total": total_steps, "tokens_per_step": tokens_per_step,
                  "log_every": max(1, args.log_every),
                  "env": {k: os.environ[k] for k in _env_keys if k in os.environ},
                  "points": [], "gate_load_summary": {}, "summary": {}}

    def _write_series() -> None:
        if series is None:
            return
        try:
            series_path.write_text(json.dumps(series, indent=1, default=float))
        except Exception as exc:  # noqa: BLE001 - diagnostics must never kill the run
            print(f"[series] WARNING: could not write {series_path}: {exc}", flush=True)

    def _accumulate_gate_loads() -> None:
        """Add the current per-gate expert-selection counts to the running sums (log points)."""
        nonlocal gate_load_n
        if series is None or balance is None:
            return
        attr = getattr(balance, "_LOAD_ATTR", "_v41_load")
        n = 0
        for name, mod in model.named_modules():
            if not getattr(mod, "_v41_is_gate", False):
                continue
            buf = mod._buffers.get(attr)
            if buf is None or buf.numel() == 0:
                continue
            counts = buf.detach().float().cpu()
            if float(counts.sum()) <= 0:
                continue
            key = name if name.endswith(".gate") else (f"{name}.gate" if name else "gate")
            if key in gate_load_sums:
                gate_load_sums[key] += counts
            else:
                gate_load_sums[key] = counts.clone()
            n += 1
        if n:
            gate_load_n += 1

    def _gate_load_summary() -> dict:
        if not gate_load_sums or gate_load_n <= 0:
            return {}
        out_map: dict = {}
        for key, sums in sorted(gate_load_sums.items()):
            mean_load = sums / float(gate_load_n)
            mean = float(mean_load.mean())
            if mean <= 0:
                continue
            k = min(5, mean_load.numel())
            top = torch.topk(mean_load, k)
            out_map[key] = {
                "experts": int(mean_load.numel()),
                "mean": round(mean, 3),
                "max": round(float(mean_load.max()), 3),
                "min": round(float(mean_load.min()), 3),
                "max_over_mean": round(float(mean_load.max() / mean), 4),
                "top5_experts": [int(i) for i in top.indices.tolist()],
                "top5_counts": [round(float(v), 2) for v in top.values.tolist()],
                "sampled_steps": gate_load_n,
            }
        return out_map

    # ---------------------------------------------------------------- loop profile (opt-in)
    # M6-P1: decompose the training step into (a) GPU-exact segment times, measured with a
    # synchronize at every segment boundary, and (b) the real pipelined wall (same kernel set, no
    # per-segment sync).  `pipelined_non_gpu_ms` = pipelined wall - sum(pure GPU) is the
    # loop/optimizer/data tax the throughput work has to remove.  Steps advance the optimizer
    # exactly like the real loop, then the process exits (no checkpoint).
    if args.profile_loop > 0:
        n_steps = max(1, int(args.profile_loop))
        n_sync = max(1, min(2, n_steps // 2))
        n_pip = max(1, n_steps - n_sync)
        gpu_keys = ("h2d", "replay", "accumulate", "clip", "opt_adamw", "opt_muon",
                    "opt_sinkhorn", "opt_engram", "bias", "log_sync")
        host_keys = ("setup", "data") + gpu_keys
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated() / 2**30
        base_reserved = torch.cuda.memory_reserved() / 2**30
        recs: list[dict] = []

        def _run_profile_step(step: int, sync_seg: bool) -> dict:
            lr = lr_at(step, total_steps, args.peak_lr, args.warmup,
                       schedule=args.lr_schedule, decay_start=args.decay_start,
                       min_ratio=args.min_lr_ratio)
            evs = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                   for k in gpu_keys}
            host = {k: 0.0 for k in host_keys}
            seg_wall = {k: 0.0 for k in gpu_keys}

            def _seg(key, fn):
                # sync mode: wall around (fn + device synchronize) is the clean way to measure a
                # segment's device time -- CUDA-event pairs around a captured-graph replay read 0
                # here (the replay is launched on the capture stream), so events are only used in
                # the pipelined mode as a rough in-stream timeline.
                t0 = time.perf_counter()
                if not sync_seg:
                    evs[key][0].record()
                fn()
                if sync_seg:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                if not sync_seg:
                    evs[key][1].record()
                dt = t1 - t0
                host[key] += dt
                seg_wall[key] += dt

            def _setup():
                if bundle is not None:
                    bundle.set_lr(lr)
                else:
                    for grp in opt.param_groups:
                        grp["lr"] = lr
                    if sparse is not None:
                        sparse.lr = lr * args.engram_lr_mult
                loss_acc.zero_()
                if balance is not None:
                    balance.zero_load_buffers(model)
                if accum > 1:
                    _zero_grads(params)

            torch.cuda.synchronize()
            t_step0 = time.perf_counter()
            t0 = time.perf_counter()
            _setup()
            host["setup"] += time.perf_counter() - t0
            for _m in range(accum):
                t0 = time.perf_counter()
                batch_np = loader.next_batch()
                host["data"] += time.perf_counter() - t0
                _seg("h2d", lambda: static_ids.copy_(
                    torch.from_numpy(batch_np).cuda(non_blocking=True)))
                _seg("replay", g.replay)
                if bundle is not None:
                    _seg("accumulate", bundle.accumulate)
                else:
                    def _harvest():
                        for t in tables:
                            t.collect_row_grads()
                    _seg("accumulate", _harvest)
            gn_t = None

            def _clip():
                nonlocal gn_t
                if args.grad_clip and args.grad_clip > 0:
                    if _opt_fast is not None and getattr(_opt_fast, "CLIP", False) \
                            and bundle is not None:
                        gn_t = _opt_fast.clip_grad_norm_(bundle, params, args.grad_clip)
                    else:
                        gn_t = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            _seg("clip", _clip)
            if bundle is not None:
                for key, sub in (("opt_adamw", getattr(bundle, "adamw", None)),
                                 ("opt_muon", getattr(bundle, "muon", None)),
                                 ("opt_sinkhorn", getattr(bundle, "sinkhorn", None)),
                                 ("opt_engram", getattr(bundle, "engram", None))):
                    if sub is not None:
                        _seg(key, sub.step)
            else:
                def _plain_opt():
                    opt.step()
                    if sparse is not None:
                        sparse.step(harvest=False)

                _seg("opt_adamw", _plain_opt)
            # R77: refresh the fp8 head-weight quantisation cache once per optimizer step
            # (NANOV41_HEAD_WCACHE=1; cheap no-op otherwise).  Must run OUTSIDE the graph.
            _hc_maybe_refresh()
            _ff8v2_maybe_refresh()
            if balance is not None and args.bias_speed > 0:
                _seg("bias", lambda: balance.update_gate_bias(model, speed=args.bias_speed))

            def _log_sync():
                _ = float(loss_acc.item()) / accum
                if gn_t is not None:
                    _ = float(gn_t)
                if balance is not None:
                    try:
                        balance.collect_load_stats(model)
                    except Exception:  # noqa: BLE001
                        pass

            _seg("log_sync", _log_sync)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t_step0) * 1e3
            if sync_seg:
                gpu_ms = {k: seg_wall[k] * 1e3 for k in gpu_keys}
            else:
                gpu_ms = {k: evs[k][0].elapsed_time(evs[k][1]) for k in gpu_keys}
            rec = {"step": step, "sync": bool(sync_seg), "wall_ms": round(wall_ms, 3),
                   "gpu_ms": {k: round(v, 3) for k, v in gpu_ms.items()},
                   "host_ms": {k: round(v * 1e3, 3) for k, v in host.items()},
                   "gpu_total_ms": round(sum(gpu_ms.values()), 3),
                   "host_total_ms": round(sum(host.values()) * 1e3, 3)}
            recs.append(rec)
            return rec

        for i in range(n_pip):
            _run_profile_step(i, sync_seg=False)
            print(f"[profile] pip step {i + 1}/{n_pip} wall={recs[-1]['wall_ms']:.1f}ms "
                  f"gpu={recs[-1]['gpu_total_ms']:.1f}ms", flush=True)
        for i in range(n_sync):
            _run_profile_step(n_pip + i, sync_seg=True)
            print(f"[profile] sync step {i + 1}/{n_sync} wall={recs[-1]['wall_ms']:.1f}ms "
                  f"gpu={recs[-1]['gpu_total_ms']:.1f}ms", flush=True)

        def _mean(rows, key, sub):
            vals = [r[sub][key] for r in rows if key in r[sub]]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        def _mean_scalar(rows, key):
            vals = [r[key] for r in rows if key in r]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        pip = [r for r in recs if not r["sync"]]
        syn = [r for r in recs if r["sync"]]
        pip_gpu = {k: _mean(pip, k, "gpu_ms") for k in gpu_keys}
        syn_gpu = {k: _mean(syn, k, "gpu_ms") for k in gpu_keys}
        syn_host = {k: _mean(syn, k, "host_ms") for k in host_keys}
        pip_host = {k: _mean(pip, k, "host_ms") for k in host_keys}
        pure_gpu = round(sum(syn_gpu.values()), 3)
        wall_pip = _mean_scalar(pip, "wall_ms")
        wall_syn = _mean_scalar(syn, "wall_ms")
        peak = {"allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
        report = {
            "config": {
                "arm": args.arm, "batch": args.batch, "seq": args.seq, "accum": accum,
                "tokens_per_step": tokens_per_step, "optimizer": args.optimizer,
                "head_fp8": os.environ.get("NANOV41_HEAD_FP8", "0"),
                "head_fwdgrad": os.environ.get("NANOV41_HEAD_FWDGRAD", "0"),
                "ce_chunk": os.environ.get("NANOV41_CE_CHUNK", "default"),
                "attn_v2": os.environ.get("NANOV41_ATTN_V2", "0"),
                "fast_attn_tl": os.environ.get("FAST_ATTN_TL", "0"),
                "moe_stacked": os.environ.get("NANOV41_MOE_STACKED", "0"),
                "moe_dw_tile": os.environ.get("NANOV41_MOE_DW_TILE", ""),
                "moe_tile": os.environ.get("NANOV41_MOE_TILE", ""),
                "profile_steps": n_steps, "pipelined_steps": len(pip), "sync_steps": len(syn),
                "gpu": torch.cuda.get_device_name(0),
            },
            "baseline_mem_gib": {"allocated": round(base_alloc, 3),
                                 "reserved": round(base_reserved, 3)},
            "peak_mem_gib": peak,
            "pipelined": {"wall_ms": wall_pip, "gpu_ms": pip_gpu,
                          "gpu_total_ms": round(sum(pip_gpu.values()), 3),
                          "host_ms": pip_host,
                          "host_total_ms": round(sum(pip_host.values()), 3)},
            "sync": {"gpu_ms": syn_gpu, "gpu_total_ms": pure_gpu,
                     "host_ms": syn_host, "host_total_ms": round(sum(syn_host.values()), 3),
                     "wall_ms": wall_syn},
            "derived": {
                "replay_ms_per_micro": round(syn_gpu["replay"] / max(accum, 1), 3),
                "accumulate_ms_per_micro": round(syn_gpu["accumulate"] / max(accum, 1), 3),
                "replay_gpu_total_ms": syn_gpu["replay"],
                "pipelined_non_gpu_ms": round(wall_pip - pure_gpu, 3),
                "pipelined_non_gpu_pct": round(100.0 * (wall_pip - pure_gpu)
                                               / max(wall_pip, 1e-9), 2),
                "sync_step_ms": wall_syn,
                "tokens_per_sec_pipelined": round(tokens_per_step / max(wall_pip / 1e3, 1e-9), 1),
                "raw_mfu_pipelined": round(3.0 * fwd_flops
                                           * (tokens_per_step / max(wall_pip / 1e3, 1e-9))
                                           / peak_flops, 5),
            },
            "steps": recs,
        }
        prof_path = out.with_suffix(".loop_profile.json")
        prof_path.parent.mkdir(parents=True, exist_ok=True)
        prof_path.write_text(json.dumps(report, indent=2))
        print(f"[profile] wrote {prof_path}", flush=True)
        print(f"[profile] PIPELINED wall={wall_pip:.1f}ms tok/s="
              f"{report['derived']['tokens_per_sec_pipelined']:.0f} "
              f"rawMFU={report['derived']['raw_mfu_pipelined'] * 100:.2f}%", flush=True)
        print(f"[profile] PURE GPU total={pure_gpu:.1f}ms  gaps={wall_pip - pure_gpu:.1f}ms "
              f"({report['derived']['pipelined_non_gpu_pct']:.1f}% of wall)", flush=True)
        for k in gpu_keys:
            print(f"[profile]   {k:12s} gpu={syn_gpu[k]:8.2f}ms  "
                  f"host={syn_host[k]:8.2f}ms  per_micro="
                  f"{syn_gpu[k] / max(accum, 1):7.3f}ms", flush=True)
        return 0

    # ---------------------------------------------------------------- training loop
    print(f"[train] steps={start_step}..{total_steps} tokens/step={tokens_per_step} "
          f"total_tokens={(total_steps - start_step)*tokens_per_step/1e6:.1f}M "
          f"optimizer={args.optimizer} balance={args.balance_loss} bias_speed={args.bias_speed}",
          flush=True)
    t0 = time.time()
    done0 = start_step * tokens_per_step
    loss_acc_sum, loss_n = 0.0, 0
    last_good = True
    last_ckpt = ""
    log_every = max(1, args.log_every)

    def save_ckpt(path: pathlib.Path, step: int) -> None:
        payload = {"cfg": cfg, "state_dict": model.state_dict(), "step": step,
                   "tokens": step * tokens_per_step, "arm": args.arm,
                   "engram": engram_name, "optimizer_kind": args.optimizer,
                   "loader_state": loader.state_dict(), "config_name": config_name}
        if args.save_optimizer:
            if bundle is not None:
                payload["optimizer"] = bundle.state_dict()
            elif opt is not None:
                payload["optimizer"] = {
                    "adamw": _state_to_cpu_dict(opt.state_dict()),
                    "sparse": sparse.state_dict() if sparse is not None else None,
                }
        # atomic: a preemption during the multi-GB write must not leave a truncated file that the
        # arm runner would accept as "already trained" (R91 review M11)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, args.peak_lr, args.warmup,
                   schedule=args.lr_schedule, decay_start=args.decay_start,
                   min_ratio=args.min_lr_ratio)
        if bundle is not None:
            bundle.set_lr(lr)
        else:
            for grp in opt.param_groups:
                grp["lr"] = lr
            if sparse is not None:
                sparse.lr = lr * args.engram_lr_mult
        if accum > 1:
            _zero_grads(params)
        loss_acc.zero_()
        if balance is not None:
            balance.zero_load_buffers(model)
        for _m in range(accum):
            batch_np = loader.next_batch()
            static_ids.copy_(torch.from_numpy(batch_np).cuda(non_blocking=True))
            g.replay()
            # harvest the sparse Engram gradients of *this* replay before the next one overwrites
            if bundle is not None:
                bundle.accumulate()
            else:
                for t in tables:
                    t.collect_row_grads()
        check_now = (step % max(1, args.nan_check_every) == 0) or \
                    (step % log_every == 0) or (step == total_steps - 1)
        loss = None
        if check_now:
            loss = float(loss_acc.item()) / accum
            last_good = math.isfinite(loss)
            if not last_good:
                print(f"[nan] non-finite loss at step {step+1}/{total_steps} "
                      f"(last ckpt: {last_ckpt or 'none'}); aborting", flush=True)
                if args.abort_on_nan:
                    _write_series()
                    return 3
        grad_norm = None
        if args.grad_clip and args.grad_clip > 0:
            if _opt_fast is not None and getattr(_opt_fast, "CLIP", False) \
                    and bundle is not None:
                gn_t = _opt_fast.clip_grad_norm_(bundle, params, args.grad_clip)
            else:
                gn_t = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            grad_norm = float(gn_t) if check_now else None
        if bundle is not None:
            bundle.step()
        else:
            opt.step()
            if sparse is not None:
                sparse.step(harvest=False)
        # EMA shadows: read the post-step weights (fp32 shadows; outside the graph). Uses the foreach
        # kernels when the bf16-live/fp32-shadow mix is supported, else a per-tensor loop.
        if ema_shadows and ((step + 1) % max(1, args.ema_every) == 0):
            with torch.no_grad():
                names = [n for n, _ in ema_params]
                live = [p.detach() for _, p in ema_params]
                for j, decay in enumerate(ema_decays):
                    tensors = [ema_shadows[j][n] for n in names]
                    torch._foreach_mul_(tensors, decay)
                    try:
                        torch._foreach_add_(tensors, live, alpha=1.0 - decay)
                    except (RuntimeError, TypeError):
                        for sh, pl in zip(tensors, live):
                            sh.add_(pl, alpha=1.0 - decay)
        # R77: refresh the fp8 head-weight quantisation cache (NANOV41_HEAD_WCACHE=1; no-op otherwise)
        _hc_maybe_refresh()
        _ff8v2_maybe_refresh()
        if balance is not None and args.bias_speed > 0:
            balance.update_gate_bias(model, speed=args.bias_speed)
        if step % log_every == 0 or step == total_steps - 1:
            if loss is None:
                loss = float(loss_acc.item()) / accum
            loss_acc_sum += loss
            loss_n += 1
            dt = time.time() - t0
            done = (step + 1) * tokens_per_step
            tps = (done - done0) / max(dt, 1e-9)
            mfu = 3.0 * fwd_flops * tps / peak_flops
            gn = f" gn={grad_norm:.2f}" if grad_norm is not None else ""
            extra = ""
            imb_val = None
            if balance is not None:
                try:
                    stats = balance.collect_load_stats(model)
                    if stats:
                        worst = max(stats.values())
                        imb_val = worst
                        extra = f" imb={worst:.2f}"
                except Exception:  # noqa: BLE001 - logging must never kill the run
                    pass
            print(f"[step {step+1}/{total_steps}] loss={loss:.5f} lr={lr:.2e} "
                  f"tok/s={tps/1e3:.1f}k mfu={mfu*100:.1f}%{gn}{extra}", flush=True)
            if series is not None:
                _accumulate_gate_loads()
                series["points"].append({
                    "step": step + 1,
                    "loss": round(loss, 6),
                    "lr": lr,
                    "tok_s": round(tps, 1),
                    "mfu_pct": round(mfu * 100, 3),
                    "gn": (round(grad_norm, 6) if grad_norm is not None else None),
                    "imb": (round(imb_val, 4) if imb_val is not None else None),
                    "elapsed_s": round(dt, 2),
                })
                _write_series()
        if args.save_every and (step + 1) % args.save_every == 0 and last_good:
            last_ckpt = str(out.with_suffix(f".step{step + 1}.pt"))
            save_ckpt(pathlib.Path(last_ckpt), step + 1)
        # WSD multi-point screen: one checkpoint per requested token budget prefix (the final
        # budget is covered by the normal <out>.pt checkpoint, so skip it here to save disk).
        if (step + 1) in mark_steps and last_good and (step + 1) < total_steps:
            mark = mark_steps[step + 1]
            mark_path = out.with_suffix(f".t{mark}.pt")
            save_ckpt(mark_path, step + 1)
            print(f"[save-mark] mark={mark} step={step + 1} tokens={(step + 1) * tokens_per_step} "
                  f"path={mark_path}", flush=True)
    torch.cuda.synchronize()
    if not last_good:
        print(f"[nan] not saving final checkpoint (non-finite loss); keeping last good ckpt: "
              f"{last_ckpt or 'none'}", flush=True)
        _write_series()
        return 4
    train_seconds = time.time() - t0          # stop the clock before the final checkpoint write
    save_ckpt(out.with_suffix(".pt"), total_steps)
    if ema_shadows and total_steps > start_step:
        base_sd = model.state_dict()
        for j, decay in enumerate(ema_decays):
            tag = ("%g" % decay).replace(".", "p")
            sh = ema_shadows[j]
            sd = {k: (sh[k].to(v.dtype) if k in sh else v) for k, v in base_sd.items()}
            payload = {"cfg": cfg, "state_dict": sd, "step": total_steps,
                       "tokens": total_steps * tokens_per_step, "arm": args.arm,
                       "engram": engram_name, "optimizer_kind": args.optimizer,
                       "config_name": config_name, "ema_decay": decay}
            ema_path = out.with_suffix(f".ema{tag}.pt")
            tmp = ema_path.with_name(ema_path.name + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, ema_path)
            print(f"[ema] saved {ema_path} (decay={decay})", flush=True)
    if series is not None:
        series["gate_load_summary"] = _gate_load_summary()
        series["summary"] = {
            "steps": total_steps - start_step,
            "tokens": (total_steps - start_step) * tokens_per_step,
            "train_seconds": round(train_seconds, 1),
            "tokens_per_sec": round((total_steps - start_step) * tokens_per_step
                                    / max(train_seconds, 1e-9), 1),
            "avg_logged_loss": loss_acc_sum / max(1, loss_n),
            "raw_bf16_mfu": 3.0 * fwd_flops * ((total_steps - start_step) * tokens_per_step
                                               / max(train_seconds, 1e-9)) / peak_flops,
            "peak_lr": args.peak_lr,
        }
        if series["points"]:
            last = series["points"][-1]
            series["summary"].update({"last_loss": last["loss"], "last_gn": last["gn"],
                                      "last_imb": last["imb"]})
        _write_series()
    summary = {"arm": args.arm, "config": config_name, "engram": engram_name,
               "optimizer": args.optimizer, "grad_accum": accum,
               "tokens": (total_steps - start_step) * tokens_per_step,
               "steps": total_steps - start_step, "tokens_per_step": tokens_per_step,
               "peak_lr": args.peak_lr, "lr_schedule": args.lr_schedule,
               "decay_start": args.decay_start, "min_lr_ratio": args.min_lr_ratio,
               "save_at_tokens": args.save_at_tokens,
               "balance_loss": args.balance_loss, "bias_speed": args.bias_speed,
               "z_loss": args.z_loss,
               "train_seconds": round(train_seconds, 1),
               "tokens_per_sec": round((total_steps - start_step) * tokens_per_step
                                       / max(train_seconds, 1e-9), 1),
               "avg_logged_loss": loss_acc_sum / max(1, loss_n),
               "raw_bf16_mfu": 3.0 * fwd_flops * ((total_steps - start_step) * tokens_per_step
                                                  / max(train_seconds, 1e-9)) / peak_flops,
               "fwd_flops_per_token": fwd_flops, "peak_tflops": args.peak_tflops,
               "table_params": ne.engram_param_count(model) if tables else 0,
               "ckpt": str(out.with_suffix(".pt")),
               "optimizer_state_saved": bool(args.save_optimizer)}
    out.with_suffix(".train.json").write_text(json.dumps(summary, indent=2))
    if series is not None:
        print(f"[series] wrote {series_path} ({len(series['points'])} points, "
              f"{len(series['gate_load_summary'])} gates)", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
