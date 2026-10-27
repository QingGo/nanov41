#!/usr/bin/env python3
"""W1 parity harness: prove that the training port matches the official V4.1 reference.

Three checks, all on a tiny config (CPU is enough):
  T1  full-sequence logits  ==  incremental (prefix-by-prefix) last-token logits
      -> validates the sliding-window mask, the compressed-KV causality and the
         indexer top-k causality of the full-sequence training path.
  T2  patched (trainable) model  ==  pristine official `ref/model.py`  on the same
      weights and prefixes -> validates that our five patches changed gradients, not math.
  T3  backward smoke: loss finite, gradients finite and non-zero -> the training path is
      actually differentiable end to end.

Exit code 0 iff all checks pass at the printed tolerances.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.configs import CONFIGS  # noqa: E402
from nanov41.model import (  # noqa: E402
    NanoV41, build_pristine, causal_lm_loss, init_random_weights, inner_forward,
)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def grad_stats(model: torch.nn.Module) -> dict:
    total, nonzero, nan = 0, 0, 0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += p.grad.numel()
        nonzero += int((p.grad != 0).sum())
        nan += int(torch.isnan(p.grad).sum())
    return {"grad_elems": total, "nonzero": nonzero, "nan": nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny_all_topk", choices=sorted(CONFIGS))
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float32"])
    ap.add_argument("--t1-tol", type=float, default=None)
    ap.add_argument("--t2-tol", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    tie_cfg = CONFIGS["tiny"] if args.config != "tiny" else None
    torch.manual_seed(args.seed)
    t0 = time.time()
    model = NanoV41(cfg, param_dtype=args.dtype).to(args.device)
    init_random_weights(model, seed=args.seed)
    print(
        f"[build] config={args.config} params={model.num_params()/1e6:.2f}M "
        f"(non-embedding {model.num_params_non_embedding()/1e6:.2f}M) "
        f"layers={cfg['n_layers']} dim={cfg['dim']} experts={cfg['n_routed_experts']} "
        f"window={cfg['window_size']} ratios={cfg['compress_ratios']} "
        f"dtype={args.dtype} in {time.time() - t0:.1f}s",
        flush=True,
    )
    vocab = cfg["vocab_size"]
    x = torch.randint(0, vocab, (args.batch, args.seq_len), device=args.device)

    # ---------------------------------------------------------------- T1
    banner("T1  full-sequence logits vs incremental last-token logits (patched model)")
    logits_full = model(x)  # [b, s, vocab]
    t1 = []
    for i in range(args.seq_len):
        _ids, logits_last, _hidden = inner_forward(model.inner, x[:, : i + 1])
        t1.append(max_abs(logits_full[:, i], logits_last))
    worst = max(t1)
    where = t1.index(worst)
    print(f"  prefixes={args.seq_len}  max|diff|={worst:.3e} at L={where + 1}")
    print(f"  first10={[round(v, 5) for v in t1[:10]]}")
    t1_tol = args.t1_tol if args.t1_tol is not None else (1e-4 if args.dtype == "float32" else 5e-2)
    t1_ok = worst <= t1_tol
    print(f"  -> {'PASS' if t1_ok else 'FAIL'} (tol {t1_tol:g}, dtype {args.dtype})", flush=True)

    # ---------------------------------------------------------------- T2
    banner("T2  patched (trainable) vs pristine official reference, same weights")
    pristine = build_pristine(cfg, param_dtype=args.dtype).to(args.device)
    missing, unexpected = pristine.load_state_dict(model.inner.state_dict(), strict=False)
    if missing or unexpected:
        print(f"  state_dict mismatches: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    t2 = []
    for i in range(args.seq_len):
        _o1, l1, _h1 = inner_forward(model.inner, x[:, : i + 1])
        _o2, l2, _h2 = inner_forward(pristine, x[:, : i + 1])
        t2.append(max_abs(l1, l2))
    worst2 = max(t2)
    print(f"  prefixes={args.seq_len}  max|diff|={worst2:.3e}  values={[round(v, 6) for v in t2[:6]]}")
    t2_tol = args.t2_tol if args.dtype == "bf16" else 1e-6
    t2_ok = worst2 <= t2_tol and not (missing or unexpected)
    print(f"  -> {'PASS' if t2_ok else 'FAIL'} (tol {t2_tol:g})", flush=True)

    # ---------------------------------------------------------------- T3
    banner("T3  backward smoke (causal-LM loss -> gradients)")
    model.zero_grad(set_to_none=True)
    loss = model(x, labels=x)
    loss.backward()
    stats = grad_stats(model)
    finite = torch.isfinite(loss).item() and stats["nan"] == 0
    covered = stats["nonzero"] > 0 and stats["grad_elems"] > 0
    print(f"  loss={loss.item():.6f}  grad_elems={stats['grad_elems']}  "
          f"nonzero={stats['nonzero']} ({100 * stats['nonzero'] / max(stats['grad_elems'], 1):.1f}%)  nan={stats['nan']}")
    t3_ok = bool(finite and covered)
    print(f"  -> {'PASS' if t3_ok else 'FAIL'}", flush=True)

    # ---------------------------------------------------------------- T1b (informational)
    banner("T1b indexer tie-breaking probe (random init; informational)")
    if tie_cfg is not None:
        torch.manual_seed(args.seed + 1)
        m2 = NanoV41(tie_cfg, param_dtype=args.dtype).to(args.device)
        init_random_weights(m2, seed=args.seed + 1)
        x2 = torch.randint(0, tie_cfg["vocab_size"], (args.batch, args.seq_len), device=args.device)
        lf2 = m2(x2)
        t1b = []
        for i in range(args.seq_len):
            _ids, ll, _h = inner_forward(m2.inner, x2[:, : i + 1])
            t1b.append(max_abs(lf2[:, i], ll))
        nonzero = [i + 1 for i, v in enumerate(t1b) if v > (1e-4 if args.dtype == "float32" else 2e-2)]
        print(f"  index_topk={tie_cfg['index_topk']}  max|diff|={max(t1b):.3e}  "
              f"positions above tol: {nonzero}")
        print("  (expected: a few positions; the indexer ReLU maps many scores to exactly 0 and")
        print("   topk tie-breaking depends on the candidate count, i.e. on prefix length.")
        print("   A canonical tie-break / batch-invariant indexer is a W2 item.)")
    else:
        print("  skipped (strict config already uses topk >= #positions)")

    banner("VERDICT")
    print(f"  T1 {t1_ok} | T2 {t2_ok} | T3 {t3_ok}")
    ok = t1_ok and t2_ok and t3_ok
    print("  ALL PASS" if ok else "  SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
