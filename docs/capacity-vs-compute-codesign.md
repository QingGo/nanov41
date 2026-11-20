# Capacity vs compute co-design: what is actually "free"

Status note (2026-11-06). The question: if we increase parameter count while training-throughput
(tokens/s) barely moves, is that a free capability win? Partly yes — but only on the **capacity**
axis, and the price is paid in optimizer/memory traffic, not FLOPs. This note makes the accounting
explicit and turns it into two cheap experiments.

## 1. The three knobs (do not mix them)

Let `F` = training FLOPs/token, `P` = total params, `A` = active params.
`tokens/s ≈ MFU · peak / F`; the optimizer cost and activation memory scale with `P` and `b·seq`.

| knob | FLOPs/token | tokens/s | memory | optimizer | capability |
|---|---|---|---|---|---|
| **inactive capacity** (more experts, Engram tables, bigger embedding/lookup) | unchanged | ≈unchanged (−3% for +44% P in d12e32 estimate) | ↑ params + grads + Adam states + activation pool | ↑ ∝ P (unless sparse/selective) | ↑ (capacity/quality, unproven at this scale) |
| **active compute** (wider dim, more layers, more top-k) | ↑ ∝ A | ↓ ≈ ∝ F unless MFU rises as much | ↑ activations + pool | ↑ | ↑ (loss/FLOP gains) |
| **tokens/step** (bigger batch×seq) | per-token unchanged | ↑ (amortizes fixed launch/tail) | ↑ activations/pool (often the wall) | unchanged | same loss, faster wall-clock |

"Free" is only true for the first row (and partially for the third). The second row is the
trade-off nanochat already made at compute-optimal allocation.

## 2. What the numbers say (d12 anchors measured; rest are model estimates)

Measured d12 anchor: graph step 57.5 ms (single step), 293.5k tok/s with 4× accumulation
(55.8 ms/16K), raw MFU 59.4%, discounted MFU ≈38.5%, peak 14.46 GiB. Device-time model uses the
measured 61% device utilization and the measured optimizer cost (4.39 ms at 193.4M params, linear
in P); no MFU gain from larger GEMMs is assumed (conservative — the sweep will measure it).

| config | total / non-emb params | active params | fwd FLOPs/token | train FLOPs/token | est. tokens/s | est. raw MFU | est. discounted MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| d12 | 193.4M / 94.1M | 19.8M | 141.3M (measured) | 423.9M | 293.5k (measured) | 59.4% (measured) | 38.5% (measured) |
| **d12e32** (32 experts, top-2) | **278.4M / 179.1M** | **19.9M** | ~139M | ~417M | ≈−3.3% (~284k) | ≈−1.8pp | ≈−1.1pp (optimizer cost) |
| d16 | 353.9M / 221.5M | 45.4M | 223.1M | 669.4M | ≈−38% (~182k) | ≈−0.6pp | ≈+3.0pp |
| d20 | 596.4M / 430.9M | 86.8M | 339.2M | 1.017G | ≈−60% (~117k) | ≈−1.1pp | ≈+5.6pp |

- **Inactive capacity is nearly free in compute**: d12e32 adds 85M params (+44%) for ~3% tokens/s,
  because only optimizer/memory traffic grows; FLOPs/token is identical. It does **not** raise
  discounted MFU (the extra optimizer time slightly lowers it) — it is free *capability*, not free
  *efficiency*.
- **Active growth is not free**: d16/d20 buy +3.0/+5.6 pp discounted MFU by replacing FP8-head-
  discounted work with bf16 backbone work, but tokens/s falls by ~38%/60%. The discounted gain
  comes exactly from spending more wall-clock per token; if the larger model also raises device
  utilization (bigger GEMMs, amortized tail), the drop shrinks — that is the measurement the sweep
  below will make.
- **The real no-cost levers** are (a) more tokens per step/shape at the same model (raises MFU and
  tokens/s; memory is the limit — the current graph pool is 14.5 GiB of a 32 GiB card), (b) better
  data/tokenizer (fewer tokens to the same quality), (c) multi-GPU.

## 3. Why this matters for the Engram plan

> **Status 2026-11-06:** Engram is now **mandatory** in the OPD main training path (policy
> change in [`engram-integration-plan.md`](engram-integration-plan.md)); sizing is fixed in
> [`engram-design.md`](engram-design.md): d12 +128 M table (24 expert-equivalents, +0.70 %
> forward FLOPs), d16 +256 M, d20 +512 M.

Engram is the extreme version of the "capacity" row: table parameters are looked up, not multiplied,
so they add **zero per-token FLOPs**; the only training cost is the optimizer/memory traffic on the
table. That cost disappears if the table uses **sparse updates** (only touched rows): this is exactly
what `engram-peft`'s sparse Adam does, and what the nano-Engram implementation must do to keep the
"free capacity" property. At inference, Engram adds memory reads (rows/token) but not FLOPs, which
fits the low-inference-cost positioning.

## 4. Experiments (both cheap; ~2 h GPU total ≈ ¥6)

1. **Scaling sweep** (`tools/r80_scale_sweep.sh`, ready): d12 / d12e32 / d16 / d20 × 2 rounds;
   measures actual step ms, tokens/s, raw MFU, peak memory and whether larger GEMMs raise device
   utilization. d20 runs at b=8 (b=16 will likely OOM).
2. **Free-capacity quality A/B**: d12 vs d12e32 at identical data/seed/steps (0.6B tokens each,
   ~36 min/run at 280k tok/s, ≈¥3/run): does +44% total params at −3% tokens/s improve held-out
   bpb? If yes, capacity expansion is a genuine free win and should be combined with Engram tables
   before the OPD student.
3. **Optimizer variants microbench** (part of the kernel work): fused vs foreach vs capturable vs
   bf16 optimizer states; the optimizer is 7.6% of the step and the main price of capacity, so
   cutting it raises both tokens/s and the value of capacity.

## 4b. If the objective is capability per unit cost, the answer changes

"Tokens/s must not drop" is a throughput objective. If instead the objective is **capability per
unit cost** (training ¥, or inference ¥), a bigger model is often the *better* choice — this is
exactly what compute-optimal scaling is about.

Definitions:

- training cost per token ∝ FLOPs/token, so cost per 1B tokens at the same ¥/FLOP efficiency:
  d12 ≈ **¥2.63/1B**, d16 ≈ **¥4.2/1B** (1.60×), d20 ≈ **¥6.3/1B** (2.40×);
- capability per training ¥ improves iff the loss/quality gain exceeds the FLOPs/token increase
  (e.g. d16 must cut the reducible loss by >1.6×, or reach the target in <62% of d12's tokens);
- inference cost per token ∝ decode FLOPs: d12 ≈141M, d16 ≈230M, d20 ≈320M — still 5.6–12.5×
  cheaper than nanochat d24 (~1.8G), so if the capability gain is large the inference side stays
  acceptable.

Why bigger is plausibly better per ¥: our current d12@100B is far on the **data-heavy side** of the
compute-optimal frontier for its compute budget (Chinchilla optimum at C=4.2e19 is ~600M params /
~12B tokens; d12 has 70.65M FLOPs-active params at 100B tokens). At a *fixed training budget*
C, moving toward the compute-optimal point (fewer tokens, more parameters) reduces loss; that is the
same trade nanochat d24 makes. The catch: the exact crossover is empirical, and we currently have
**no loss data at all** — only throughput/MFU.

### Iso-FLOPs capability experiment (the decisive one)

Hold training FLOPs constant and compare models (same data/seed, measure held-out bpb):

| C (FLOPs) | d12 tokens | d16 tokens | d20 tokens | GPU time / cost on 1×5090 (each) |
|---:|---:|---:|---:|---:|
| 2.0e18 | 4.7B | 3.0B | 2.0B | ≈4.5 h / ¥12.5 |
| 5.0e18 | 11.8B | 7.5B | 4.9B | ≈11 h / ¥31 |

Full grid (3 models × 2 C levels) ≈ **46 h / ¥130**, and it produces the capability-vs-cost curves
for both training and (with decode-bench) inference. A single C level (≈¥38) already tells us
whether d16/d20 beats d12 at the same training dollars.

**Prerequisite (today's real blocker):** the tokenizer/data pipeline and an eval harness
(held-out bpb plus CORE/task proxies). The repo has no tokenizer/data pipeline; the 129k vocab + OPD
plan already forces building it. Without it we can measure throughput (ready) but not capability.

Decision gate: pick the model that maximizes capability per training ¥ while keeping decode FLOPs
≤ ~1/3 of nanochat d24; if d12 wins per ¥ at our budget, keep d12 and re-invest the savings in more
tokens/OPD; if d16/d20 win, adopt them and re-run the Engram/capacity sizing on the larger backbone.

## 4c. Budget check: can the iso-FLOPs d12/d16/d20 comparison fit in ¥20?

At the measured d12 efficiency (≈59.4% raw MFU ⇒ 124 TF effective on 1×5090, ¥2.78/h), a model sees
`C` FLOPs in `C / 124e12` seconds. One model at C=1.0e18 costs ≈2.24 h ≈ **¥6.2**:

| C (FLOPs) | tokens per model (d12 / d16 / d20) | time per model | 3-model total |
|---:|---|---:|---:|
| 1.0e18 | 2.36B / 1.49B / 0.98B | 2.24 h | **≈6.7 h ≈ ¥18.7** |
| 1.5e18 | 3.54B / 2.24B / 1.47B | 3.36 h | ≈10.1 h ≈ ¥28 |
| 2.0e18 | 4.72B / 2.98B / 1.96B | 4.48 h | ≈13.4 h ≈ ¥37.4 |

So **¥20 fits exactly the C≈1.0e18 three-model screening** (+ a few ¥ of eval), using multi-domain
held-out BPB (T0). What it does **not** buy is a capability (CORE) decision: at ~1–2B tokens the
CORE/task scores are near random, so ¥20 answers *"which model has the best bpb per FLOP"*, not
*"which has the best capability per ¥"*. For capability/¥:

- 3 models at C=2.0e18: **≈¥37** (+ T0/CORE-lite minutes) — the clean single-shot answer;
- 2 models (d12 vs the best bigger one) at C=2.0e18: **≈¥25**;
- two-stage: ¥19 screening (C=1e18) → extend only the top-2 to C=2e18 (their runs are 4.48 h each;
  adding the 2.24 h remainder needs checkpoint resume, otherwise re-run) ⇒ ~¥40.

Recommendation: spend ¥0 now on the data/eval harness, then **¥19 for the screening** (3 models,
C=1e18, T0 multi-domain BPB), then decide whether the slope justifies the ≈¥25–40 confirmation with
CORE-lite. Do not skip the harness to save GPU money — the bottleneck is data/eval, not the ¥20.

## 5. Bottom line

- **Capacity (inactive params, Engram) ≈ free in compute, costs optimizer/memory** — a real
  co-design lever, especially after sparse-optimizer work.
- **Active params are not free**: +3–6 pp discounted MFU costs 38–60% tokens/s in the current
  estimate; only actual measurement (device utilization of bigger GEMMs) can shrink that.
- The genuinely free throughput lever is **more tokens/step and less per-token overhead**, not a
  bigger model.
