# Positioning: a small model that beats nanochat on inference cost

Status: strategy note (2026-11-06). Companion numbers:
[`compare-nanochat.md`](compare-nanochat.md), [`benchmarks.md`](benchmarks.md) §3/R69,
feasibility report §3x.

## 1. What "beat nanochat" should mean

nanochat's scoreboard is *time/cost to GPT-2 capability* on an 8×H100 node: d24 reaches
CORE 0.25851 (> GPT-2 0.256525) with 4.33e19 FLOPs, 8.758B tokens, 3.04 h, ≈$40–73 list / $15–25
spot; the leaderboard best is 1.65 h. It also ships the whole pipeline (tokenizer, SFT, RL, eval,
inference CLI) and a 2,048-token context.

For a single-consumer-GPU project the honest scoreboard is three-dimensional, not one:

| axis | nanochat d24 (reference) | our target |
|---|---|---|
| Training FLOPs to GPT-2 CORE | 4.33e19 | **≤ 4.5e19** (same budget) |
| Training cost | $40–73 list / $15–25 spot (3.04 h, 8×H100) | **≤ ¥300 (~$40)** (1×5090, ~95 h) or 8×5090 ~12 h |
| CORE (DCLM 22-task) | 0.2585 | **≥ 0.2565** |
| Total / active params | 1.38B / 1.38B (dense) | **~1.2–1.4B total / ≤150M active** (MoE) |
| Decode FLOPs / token | ≈1.8 GFLOP (2×730M + attention) | **≤ 300 MFLOP (≥6× lower)** |
| Context | 2,048 | **1M (differentiator)** |
| Post-training | SFT + RL + chat/CORE evals + inference engine | must be built (M3) |
| Wall-clock | 3 h on 8×H100 | 95 h on 1×5090, or ~12 h on 8×5090 (¥22/h) |

The key insight: **nanochat is compute-optimal-ish, so it pays full dense compute per token at
inference too. A MoE decouples capacity from active compute — if the same total capacity (and enough
tokens) reaches the same CORE, inference becomes several times cheaper at the same training FLOPs.**
Matching nanochat's training cost while cutting decode cost 6–10× is a real, defensible win; beating
it on wall-clock and on training dollars at the same time is not realistic on AutoDL 5090s.

## 2. Where we are today

- d12: 193.41M total / 94.12M non-embedding / 19.8M backbone-active / 70.65M FLOPs-active (head
  included). Decode = 141.3 MFLOP/token — **12.5× cheaper than nanochat d24's ~1.8 GFLOP/token** —
  and the stack runs at 59.4% raw bf16 MFU, 293.5k tok/s on one 5090 (GFX/MMLU-class quality is
  unmeasured).
- The 100B-token plan spends 4.24e19 FLOPs at 19.8M active ⇒ **compute-suboptimal for capability**
  (Chinchilla at 4.3e19 would be ~600M dense × ~12B tokens); it optimizes inference economics and
  long context, not CORE.
- Missing: tokenizer, data pipeline, SFT/RL, eval harness (CORE/BPB + task evals), multi-GPU DDP,
  FP8 linears (head only today). The V4.1-port track also uses the 129k DeepSeek vocab, whose head is
  ~70% of d12 forward FLOPs — a poor fit for a "low inference cost" product.

## 3. Levers (ordered by expected payoff)

| lever | effect | status / risk |
|---|---|---|
| **L1. MoE capacity/activation decoupling** | same total capacity 1.2–1.4B but ≤150M active ⇒ same training FLOPs, decode 6–10× cheaper | needs the ladder; MoE sample efficiency at this sparsity is the key unknown |
| ~~L2. Product tokenizer (32k–64k BPE)~~ **superseded** | the 129k vocab is required for API top-20 OPD (teacher token ids); see [`opd-strategy.md`](opd-strategy.md). Head FLOPs are now a fixed floor — control it with dim/FP8/low-rank instead |
| **L3. FP8 training** | 5090 has ~2× bf16 peak (419 TF); head already fp8; linears/MoE fp8 at nano shapes are latency-bound (R51/R54: not a clear win at d=384) | measure per-GEMM; realistic 1.2–1.7× not 2× |
| **L4. 1M context** | nanochat cannot do it; long-doc QA/agents are a differentiator | chunked BPTT graph trainer already bitwise-consistent (262K/1M); need RULER/needle evals |
| **L5. Data quality / curriculum** | fewer tokens to the same CORE; our planned web/code/math/zh mix + cooldown is differentiated vs ClimbMix | needs the data pipeline; sample-efficiency effect unverified at this scale |
| **L6. Multi-GPU for wall-clock** | 8×5090 ≈ ¥22/h; 4.3e19 → ~12 h (vs 95 h single) at the same dollars | DDP engineering; small model allreduce is cheap even over PCIe |

## 4. Experiment plan (cheap → decisive)

| phase | what | cost | gate |
|---|---|---|---|
| P0 | Eval harness: port nanochat's MIT `core_eval.py`/BPB logic (or DCLM CORE reference) into our repo; reproduce the published d24 reference on our eval (or trust their 0.2585) | CPU, ¥0 | numbers line up |
| P1 | Tokenizer + data pipeline (1–2 weeks CPU): train 32k/64k BPE on the planned mix; build tokenized shards + CORE/BPB/MMLU/ARC/GSM8K + RULER/needle eval | ¥0 GPU | data+ecc ready |
| P2 | **Scaling ladder at fixed C ≈ 4.3e19** (this is the decisive experiment): {d12 193M/19.8M active @ ~4B tok; d16 ~350M/35M @ ~8B; d20 ~600M/60M @ ~15B; d22 ~1.0B/100M @ ~25B; d24 ~1.4B/150M @ ~37B} — measure val_bpb and CORE | 5 runs × 1–3 h ≈ **¥30–50** (single 5090) | **CORE ≥ 0.2565 at ≤4.5e19 with active ≤150M?** |
| P3 | If P2 passes: FP8/optimization pass on the winner (measure GEMM-level), plus DDP on 8×5090 for wall-clock | ~1 week eng, ¥20–50 | 1.2–1.7× training speed |
| P4 | Final pretrain of the winning point (4.3e19) + SFT/RL (SmolTalk/GSM8K, nanochat-style) + evals + inference bench | **≈¥260–300** (1×5090 ~95 h; ~¥260 on 8×5090 ~12 h) | CORE + chat evals |
| P5 | Publish the 3-D comparison table (train FLOPs/cost, CORE, decode FLOPs/latency, context) | — | claim |

Total to a defensible claim: **≈¥300–400 GPU**, dominated by P4; P2 costs about one dollar per
hour of curiosity. If P2 shows our architecture cannot reach 0.2565 at 4.3e19 (quite possible: d12
today is 7× smaller than nanochat), the fallback is the differentiated position (1M-context small
model + research port), not a nanochat clone.

## 5. Decision gates

- **G1 (capability)**: P2 finds a point with CORE ≥ 0.2565 at ≤4.5e19 FLOPs and active ≤150M.
  If the best point needs >6e19, training cost stops being competitive (≥¥400 on 1×5090) and the
  "low training cost" claim should be dropped in favour of inference/long-context claims.
- **G2 (inference)**: decode ≤300 MFLOP/token at that point (≥6× below nanochat d24).
- **G3 (pipeline)**: SFT/RL + evals reproduce nanochat-style chat numbers; otherwise only compare
  base CORE/BPB.
- **G4 (honesty)**: wall-clock on one 5090 is ~30× nanochat's 8×H100 node; do not claim wall-clock
  superiority without an 8×5090 run (≈12 h at the same dollar cost).

## 6. Two-track structure

1. **V4.1-port track (current repo)**: 129k DeepSeek vocab, Engram/mHC/1M context architecture
   research; stays the place for R69 throughput work and the Engram plan.
2. **Product track (this note)**: 32k–64k tokenizer, MoE capacity/active decoupling, CORE parity,
   long-context optional, SFT/RL pipeline; the thing that can be compared head-to-head with
   nanochat.

The two share the training stack (kernels, graph trainer, chunked BPTT, MFU work) but should not
share the tokenizer/vocab decision.

## 7. Risks

- **MoE sample efficiency at 150M active**: no public evidence at exactly this sparsity/scale; P2 is
  the answer. Dense fallback: a 600M dense model at 4.3e19 has nanochat-like training cost but
  ~5× higher decode too, i.e. no inference win.
- **CORE is a broad-knowledge benchmark**: a 1.4B-total/150M-active model trained on ~37B tokens
  (the same FLOPs as nanochat's 8.76B dense tokens) may lose to a dense 1.38B at CORE even if bpb is
  close; the ladder must measure CORE, not just loss.
- **Tokenizer switch invalidates the V4.1-port numbers**; keep the two tracks separate.
- **Pipeline is the real moat**: nanochat's SFT/RL/eval/inference stack is finished; ours is not.
  Without P1/P4 the comparison stays at base-model level.
