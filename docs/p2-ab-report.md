# P2 A/B: Engram table size vs MoE capacity at 0.6 B tokens

Status: **complete 2026-11-07**.  Five arms, each trained on 600 M tokens of a Marin-8B
phase-1-like mix (DCLM-Baseline 92.6 % / codeparrot-clean 6.1 % / ProofPile-2 1.3 %), then scored
on the T0 seven-domain held-out BPB pilot (15.0 M tokens).  This is the P2 "capacity gate" from
[`engram-design.md`](engram-design.md) and [`engram-integration-plan.md`](engram-integration-plan.md).

## 1. Protocol

| item | value |
|---|---|
| model | d12 (193.4 M), 16 experts top-2, mHC-4, seq 1024, batch 16 |
| training tokens | 600 M per arm (36,621 steps; exactly 599,998,464) |
| optimizer | AdamW peak 3e-3 (cosine to 10 %, 100 warmup), Engram sparse Adam lr 1e-2 |
| stack | R69-tuned graph trainer (`torch 2.14.1+cu130`, tilelang 0.1.15, R69 accepted flags); fwd+loss+bwd captured once, dense + sparse optimizers outside the graph |
| data | `tools/p2_make_data.py`: DCLM-Baseline shards 01-09 (shard 10 is the T0 holdout), codeparrot-clean **train**, ProofPile-2 **train**; tokenizer pinned DeepSeek-V4.1-Flash (129,280), EOS between docs |
| eval | `tools/eval_t0.py` on `runs/eval/data_pilot`: 7 text domains, 15.0 M scored tokens, window 1024, batch 16 |
| teacher-match | **not run**: no OPD teacher logits / DeepSeek teacher checkpoint is cached in this workspace; T0 is the primary signal for this gate |

Arms (one seed each; `d12+E` in the design notes is the `E-default` arm):

| arm | config | Engram preset | table rows | table params |
|---|---|---|---:|---:|
| E-none | d12 | none | 0 | 0 |
| E-thin | d12 | `d12_thin` (B/2, vocab 250 k) | 2 × 1.0 M | 64.0 M |
| E-default | d12 | `d12` (B, vocab 500 k) | 2 × 2.0 M | 128.0 M |
| E-thick | d12 | `d12_thick` (2B, vocab 1 M) | 2 × 4.0 M | 256.0 M |
| d12e32 | d12e32 (32 experts) | none | 0 | 0 |

## 2. Results

| arm | train loss | tok/s | T0 macro BPB | byte-weighted BPB | Marin phase-1 mix BPB | jellyfish mix BPB | table params |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-none | 4.8767 | 253.9 k | 1.9758 | 2.0673 | 1.4749 | 1.6118 | 0 |
| E-thin | 4.8175 | 243.2 k | 1.9709 (−0.25 %) | 2.0610 (−0.31 %) | 1.4610 (−0.95 %) | 1.6029 (−0.55 %) | 64.0 M |
| **E-default** | **4.7461** | 243.7 k | **1.9595 (−0.83 %)** | **2.0509 (−0.79 %)** | **1.4474 (−1.87 %)** | **1.5887 (−1.43 %)** | 128.0 M |
| E-thick | 4.7660 | 243.8 k | 2.0045 (+1.45 %) | 2.0973 (+1.45 %) | 1.4690 (−0.41 %) | 1.6213 (+0.59 %) | 256.0 M |
| d12e32 | 4.8883 | 250.9 k | 1.9819 (+0.31 %) | 2.0734 (+0.29 %) | 1.4783 (+0.23 %) | 1.6164 (+0.29 %) | 0 (MoE +85 M) |

Per-domain for the best arm vs E-none (BPB, lower is better):

| domain | E-none | E-default | delta |
|---|---:|---:|---:|
| web | 1.4744 | 1.4455 | −0.0289 |
| wikipedia | 1.7001 | 1.6718 | −0.0282 |
| zh | 3.5561 | 3.5259 | −0.0301 |
| academic | 2.1816 | 2.1672 | −0.0144 |
| code | 1.3959 | 1.3844 | −0.0115 |
| stackexchange | 1.5888 | 1.5813 | −0.0075 |
| math | 1.9338 | 1.9405 | +0.0066 |

## 3. Reading

- **Engram at the default size is the best capacity lever per unit cost here**: vs matched d12 it
  improves held-out web/encyclopedic/code BPB and the Marin-phase1 token mix by **1.87 %** (−0.0275
  BPB), at a **~4 % training-throughput cost** (243.7 k vs 253.9 k tok/s) and +128 M table params
  (non-active memory; forward FLOPs +0.98 M/token = +0.7 %).
- **Thin (B/2) is directionally positive but weak**; **thick (2B) regresses** at this token budget
  (256 M table rows with 0.6 B tokens: under-fit/over-parameterised at the same sparse lr), so the
  sweet spot is around B = 500 k rows / 128 M table params for d12 at 0.6 B tokens.
- **MoE capacity (+16 experts, +85 M params) gives no gain at 0.6 B tokens** — slightly worse than
  d12 — while Engram does help.  This supports spending the P2 capacity budget on Engram rather than
  more experts at this scale.
- Math BPB is the one domain where Engram is neutral/slightly worse; the table lookup does not help
  reasoning-style content at this scale.

## 4. Eval cost / MFU

T0 eval is forward-only with the full 129,280-vocab logits materialised, so its MFU is low:
31.9 k tok/s at batch 4 (**~2.1 %** of 209.5 TF bf16 peak) and 78.5 k tok/s at batch 16
(**~5.2 %**, 2.46× faster).  The 15 M-token pilot takes **~192 s/checkpoint** at batch 16; the five
arms cost ~16 min total versus ~3.4 h of training, so deeper eval optimization is not on the P2
critical path.  If eval later scales to many checkpoints, the right fix is a fused-head NLL path
(avoid the 258 KB/token logits tensor), not a bigger CUDA graph.  `max_batch_size` in the training
cfg caps eval batch at 16.

## 5. Decision

- **Keep Engram in the OPD main path at the default geometry** (`d12`: layers [1,4], ngram 3,
  2 heads × 32, 500 k bucket vocab / 2.0 M rows, 128 M table params).
- **Do not enlarge the table to 2B at 0.6 B tokens**; re-test thick only at larger token budgets
  (≥2 B) or with a table-specific lr/schedule.
- **d12e32 is not a substitute for Engram** at this scale; keep 16 experts for the base model.
- Next: P3 OPD data + teacher-match (top-1 agreement / KL) on the student rollouts; keep T0 as the
  regression metric.

## 6. Caveats

- One seed per arm; differences of ~0.005 BPB are at/near run-to-run noise for a 0.6 B-token budget.
  The E-none→E-default gap (−0.0275 on the phase-1 mix) is larger than the observed arm noise, but
  the thin/thick ordering should be re-confirmed with more seeds before locking geometry.
- Eval is on 15 M held-out tokens (T0 pilot), not the 20–50 M-token final slices; treat as a screen.
- Teacher-match was not available (no OPD teacher data in the workspace).
- `d12+E` in the plan's shorthand = `E-default`; `d12e32` is the parameter-matched MoE control.

Raw data: `runs/p2/summary.json`, `runs/p2/<arm>.t0.json`, `runs/p2/<arm>.train.json`; corpus
recipe in `tools/p2_make_data.py`, trainer in `tools/p2_train.py`, eval/aggregation in
`tools/p2_run_arms.sh` / `tools/p2_report.py`.

## 7. P2b seed replication (2026-11-07)

The P2 result was one seed per arm.  P2b re-ran the primary pair (`E-none` vs `E-default`) with two
additional data/model seeds (1235, 1236, same 600 M-token corpus and protocol; paired order).  These
two seeds ran on the fallback bj-B1 instance `e865` with the base stack (torch 2.12.1+cu130,
tilelang 0.1.15, `final_env` + R69 flags) because the t141 host was GPU-busy; paired deltas are
within-stack valid, but absolute BPB is not directly comparable to the P2 seed-1234 run on t141.

| seed | arm | train loss | tok/s | macro BPB | byte-weighted | phase-1 mix | jellyfish mix |
|---|---|---|---:|---:|---:|---:|---:|
| 1235 | E-none | 4.8636 | 238.5 k | 2.0109 | 2.1061 | 1.4797 | 1.6276 |
| 1235 | E-default | 4.6622 | 231.3 k | 1.9226 | 2.0142 | 1.4249 | 1.5581 |
| 1236 | E-none | 4.8481 | 241.7 k | 1.9996 | 2.0950 | 1.4751 | 1.6200 |
| 1236 | E-default | 4.7208 | 230.6 k | 1.9649 | 2.0604 | 1.4419 | 1.5857 |

Paired deltas (E-default − E-none):

| seed | macro BPB | byte-weighted | phase-1 mix | jellyfish mix |
|---|---:|---:|---:|---:|
| 1235 | **−4.39 %** | −4.37 % | **−3.71 %** | −4.27 % |
| 1236 | **−1.74 %** | −1.65 % | **−2.26 %** | −2.12 % |
| P2 seed 1234 (t141) | −0.83 % | −0.79 % | −1.87 % | −1.43 % |

- **All three seeds agree in direction**: E-default is better on every reported aggregate; no seed
  flips the sign.  On the two same-stack seeds the phase-1 mix gain is **−2.98 % ± 1.03 %** (range
  −3.71 … −2.26 %), macro **−3.06 % ± 1.88 %**.
- The original single-seed P2 delta (−1.87 % phase-1 / −0.83 % macro) was therefore **not an
  outlier in sign but was smaller than the two new seeds**; the honest range to quote for the
  default-vs-none effect is roughly **1–4 % BPB on the Marin-phase1 mix**, always positive.
- Caveat: the two new seeds ran on a different stack (torch 2.12.1 vs 2.14.1, fallback host); the
  absolute BPB shift between stacks (~0.01–0.03) is larger than some deltas, so the *paired* deltas
  are the primary evidence.  The thin/thick ranking from P2 remains single-seed and should be
  re-confirmed before locking the table size; the E-default-vs-none decision is now robust.

## 8. Loss curves and a continuation smoke test (2026-11-07)

- **Loss curves** for the P2b seeds: `runs/p2b/loss_curves.png` (raw logged batch loss, every 1000
  steps, 4 runs).  Shape is the expected pretraining curve: ~11.8 at step 1, ~5.6-5.9 by step 1 k,
  then a slow noisy decline to ~4.1-4.4.  E-default sits below E-none for almost the entire run; the
  final logged losses are E-none **4.864 / 4.848**, E-default **4.662 / 4.721** (seeds 1235/1236),
  matching the T0 BPB ordering.
- **Continuation** (greedy, no KV cache, CPU float32, 20 new tokens; tool
  `tools/p2_continue.py`, samples `runs/p2b/continuation_samples.json`):
  - "The capital of France is" -> " a part of the world's economy. The country is a part of the world's economy. The"
  - code prompt -> plausible Python opener (`\n\n def _get_ops(): return _get_ops()`) then degrades
  - Chinese prompt -> collapses to a repeated `0.0.0.0...` token spam
  - **KV-cache parity**: the new `nanov41/generate.py` incremental path (official per-layer KV
    caches) produces **bit-identical greedy ids** to full-prefix decoding on all three prompts;
    see [`generation-kv.md`](generation-kv.md).  CPU speed (2.5–9 tok/s) is not representative.
  Functional conclusion: the checkpoint loads and produces finite next-token distributions; the
  model can teacher-forced-continue (T0 scores 15 M held-out tokens) but free-running quality after
  0.6 B tokens is limited (repetitive, language-mixed, no SFT/chat template), and the repo still has
  no KV-cache generation engine.  This is a sanity check, not a capability claim; a proper generate
  path / vLLM conversion is future work.
