# nanov41 vs nanochat — parameters, data, compute and cost

[Karpathy's nanochat](https://github.com/karpathy/nanochat) (Nov 2025) is the successor line of
nanoGPT: a single-node, end-to-end harness that trains a **GPT-2-capability chat model** — tokenizer,
pretraining, SFT, RL, evaluation and inference CLI — for roughly **$15–48** on one 8×H100 node.
nanov41 (d12) is a downscaled DeepSeek-V4.1-Flash training port: sparse MoE, mHC residual and
1M-token context, pretraining only. The two projects share a "small budget, single node, from
scratch" philosophy but spend it very differently. nanochat numbers below are from the upstream
README, `dev/LEADERBOARD.md` and `runs/speedrun.sh` (fetched 2026-11-05); nanov41 numbers are
measured in this repo ([`benchmarks.md`](benchmarks.md) §3 / R69, report §3x plan).

## 1. Model

| | **nanochat d24** (documented Run 1) | **nanov41 d12** |
|---|---|---|
| Architecture | dense GPT-2-style decoder: RoPE, QK-norm, relu² MLP, sliding-window `SSSL`, untied embeddings, per-layer value embeddings | sparse MoE (16 experts, top-2) + mHC 4-way residual + window/compressed sparse attention |
| Depth / width | 24 layers / dim 1536 (12 heads ×128), seq 2048, vocab 32,768 | 12 layers / dim 384 (mHC mult 4), context up to 1M, vocab 129,280 |
| Params (total) | **≈1.38B** (≈730M non-embedding "scaling" params; derived from `GPTConfig`, consistent with 8.76B tokens ÷ ratio 12) | **193.4M** |
| Params active / token | 1.38B (dense) | **19.8M (10.2%)** |
| Context | 2,048 tokens default (FA3, sliding window) | **up to 1,048,576** (chunked BPTT, bitwise-consistent) |
| Post-training | SFT (SmolTalk) + RL (GSM8K) + CORE/chat evals + inference engine | S4 SFT is a *planned optional* 0.2–0.5B-token stage; no SFT/RL/eval pipeline in the repo yet |

nanochat also trains d26 (≈1.7B total / ≈920M non-embedding) for the GPT-2-capability target;
GPT-2 itself is 1.6B params, CORE 0.256525.

## 2. Data and training budget

| | nanochat d24 (Run 1, 2026-01-29) | **nanov41 d12** |
|---|---|---|
| Pretraining data | NVIDIA **ClimbMix-400B** (150 shards ≈ 37.5B chars for GPT-2 capability; 6542 shards available) | self-built mix: **web 68–72%, code 10–12%, math 5–6%, zh 10–12%, wiki+books 2–3%**; S2 long-context 12–15B tokens (32K/128K/1M = 60/30/10); S3 cooldown 5% |
| Tokenizer | BPE 32,768 trained on ~2B chars | DeepSeek 129,280 (needs pre-tokenization pipeline) |
| Tokens trained (pretrain) | **8.758B** (16,704 steps × 524,288 tokens) | **100B recommended** (20B pilot; 300B stretch) |
| Tokens / param | 12 per non-embedding param, 6.3 per total param | **517 per total param**, 5,051 per active param |
| Total training FLOPs | **4.33 × 10¹⁹** (≈ "4e19 capability model") | **4.24 × 10¹⁹** for 100B tokens |
| FLOPs / token | ≈4.94 GFLOP (6N + attention) | **423.9 MFLOP** (3× forward FlopCounter; ~70% is the 129k-vocab head, backbone ≈ 6·N_active ≈ 119 MFLOP) |
| Post-training data | SFT tasks (SmolTalk), RL on GSM8K, CORE (22 evals) | planned S4: 0.2–0.5B tokens instruction mix |

The total-compute budgets are almost identical (4.33e19 vs 4.24e19 FLOPs) — but nanochat spends it
on a **1.38B dense model × 8.8B tokens** (near compute-optimal, capability-proven), while nanov41
spends it on **19.8M active params × 100B tokens** (heavy overtraining, long-context and MoE
research). That is the core architectural bet; only a real pilot run can say which allocation wins
at a given quality bar.

## 3. Throughput and MFU

| | nanochat d24 (Run 1) | **nanov41 d12** |
|---|---|---|
| Hardware | 8×H100 80GB | **1×RTX 5090 32GB** |
| Wall time (pretrain only) | **3.04 h** (16,704 steps); current leaderboard best **1.65 h**; speedrun script ~1.5 h pipeline | **94.7 h ≈ 3.94 days** (100B tokens); 18.9 h (20B pilot) |
| Tokens/s | 800k/s node = **100k/s per H100** | **293.5k/s on one 5090** |
| Model FLOPs utilization | ≈**50%** bf16 (d26 fp8 run ≈29% of fp8 peak) | **59.4%** bf16 (124.4 TFLOP/s of 209.5 TF) |
| Cost per 1B pretrain tokens | $8.3 at $3/H100-h (≈¥60); ≈$1.7 spot (≈¥12) | **¥2.63** |

Per GPU nanov41 produces ~2.9× more tokens/s than an H100 in that run, because its per-token
compute is ~11.7× lower. Normalized for peak FLOPs the MFU is in the same class (59% vs 50%);
nanochat's wall-clock advantage comes from having 8 H100s (≈38× the FLOP/s of one 5090) and from
not needing 100B tokens.

## 4. Cost (pretraining only unless noted)

Assumptions: nanochat uses ~$3/H100-hr ⇒ 8×H100 node ≈ $24/h (README quotes ~$100 tier for
2 hours, ~$15 on spot); nanov41 uses AutoDL 5090 at ¥2.78/h, ¥7.2/$.

| Run | GPU-hours | Cost |
|---|---:|---:|
| nanochat d24, 8.76B tokens (3.04 h, 8×H100) | 24.3 | **≈$73** (spot ≈$15–25) |
| nanochat current speedrun (1.65 h, 8×H100, pretrain only) | 13.2 | **≈$40** (spot ≈$10) |
| nanochat full pipeline (tokenizer→pretrain→SFT→RL→eval, README) | ~16 | **≈$48** (spot **≈$15**) |
| **nanov41 S0 mix ablations** (1.8B tokens) | 1.7 | ≈¥5–7 |
| **nanov41 20B pilot** | 18.9 | **≈¥53** (≈¥60 with overhead) |
| **nanov41 100B recommended** | 94.7 | **≈¥263** (≈¥303 with overhead) |
| nanov41 300B stretch | 284 | ≈¥790 (≈¥910) |

So: **nanochat buys GPT-2-level capability (CORE 0.2585 > GPT-2 0.2565) for ≈$40–50 in 1.5–3
hours**; nanov41 spends a comparable amount of money (¥263–303 ≈ $37–42) on 100B-token
pretraining of a much smaller MoE on a single consumer GPU, over ~4 days. The two are not buying the
same thing.

## 4b. FAQ: why 193.4M params / 19.8M active, and why 100B tokens?

**Where nanov41's 193.41M params sit** (`docs/v41-scaling-check.md`):

| group | params | note |
|---|---:|---|
| embedding + head (129,280 × 384 × 2) | 99.29M | **51.3% of total**; the vocab is *not* scaled down from V4.1-Flash |
| routed experts, all 16/layer × 12 | 84.9M | MoE total; 74.3M are *inactive* for any given token |
| non-expert backbone (attention, shared expert, mHC, gate) | 9.2M | |
| **total** | **193.41M** | non-embedding = 94.12M |
| **active / token (backbone)** | **19.8M** | = 2/16 active experts (10.6M) + non-expert backbone (9.2M) |
| **FLOPs-equivalent active (incl. head)** | **70.65M** | forward = 141.3 MFLOP/token = 2 × 70.65M |

nanochat d24 for comparison: ~680M transformer matrices + **604M per-layer value embeddings** +
50M token embedding + 50M head ⇒ ≈1.38B total, of which ≈730M are dense *compute* params. So the
"total params" axis is misleading in both directions: we are inflated by an unscaled 129k vocab and
by inactive experts, nanochat is inflated by lookup-only value embeddings. The fair axes are
**FLOPs/token (423.9 MFLOP vs 4.94 GFLOP, 11.7×)** and **training FLOPs for the run (4.24e19 vs
4.33e19, ~1×)**.

**Why 100B tokens then?** Because with a fixed compute budget (both runs are ~4.2–4.3e19 FLOPs),
the token count is set by `T = C / FLOPs_per_token`: 4.24e19 / 423.9M ≈ 100B for our MoE, while
4.33e19 / 4.94G ≈ 8.76B for nanochat d24. nanochat spends the budget near the **Chinchilla-optimal
point** (their default target ratio is 10.5, Run 1 used 12: 730M non-embedding × 12 ≈ 8.8B tokens;
the Chinchilla optimum at C ≈ 4.3e19 is ~600M params × ~12B tokens). We deliberately spend it on
the **small-model / inference-optimal corner**: 193M total / 19.8M active parameters × 100B tokens
= 517 tok per total param, 1,415 per FLOPs-active param, i.e. ~70× beyond Chinchilla for our compute
size. That trade is standard for small deployable models (SmolLM2-1.7B: 11T tokens ≈ 6,470
tok/param; Qwen2.5-0.5B: 18T tokens ≈ 36,000 tok/param), where inference cost matters more than
training cost, and it is what makes the 100B-token plan a ~¥300 / ~4-day exercise on one 5090.

The honest cost of that choice: at equal training FLOPs, a larger model trained on fewer tokens
(the nanochat allocation) should reach a *lower loss*; our allocation instead optimizes
quality-per-inference-FLOP, 1M-context capability and architecture research. That is exactly why
the next gate is the 20B pilot (~18.9 h, ≈¥60): it measures the marginal value of the 100B plan,
and a same-compute control point (larger model × ~10B tokens) would settle the allocation question.

## 5. What the comparison does and does not say

- **Does say**: at ~4.2e19 FLOPs, nanochat yields a measured GPT-2-beating chat model. nanov41's
  planned 100B run uses the same compute budget but targets a 193M-total-parameter MoE with 1M
  context; its cost per token is ~2.9× cheaper than nanochat's run and it runs on one consumer GPU,
  but it has **no quality result yet**.
- **Does not say** nanov41 is "better" or "worse" in capability: we have not run the 20B/100B
  pretraining, so there is no CORE/BPB/loss curve to compare. The 20B pilot (~18.9 h, ≈¥60) is the
  next gate; SFT/RL and an eval harness are still missing (nanochat ships all of them).
- nanochat also validates the full stack (tokenizer, dataloader, SFT/RL, inference engine); nanov41
  currently validates only the pretraining stack (numerics, kernels, graph replay, long context).
  Building the data/eval pipeline is a known ~1–2 week CPU-side task (report §3x risk 6).
- If the goal is "cheapest GPT-2-level chat model", nanochat wins outright ($15–48, 1.5–3 h).
  If the goal is "train a DeepSeek-V4.1-Flash-style MoE with 1M context on one GPU", nanov41 is the
  relevant stack and its 100B-token run is a ~¥300 / ~4-day exercise.

## 6. Sources

- nanochat README (time-to-GPT-2 leaderboard, `$48` / `$15` spot, 4e19 FLOPs): https://github.com/karpathy/nanochat
- `dev/LEADERBOARD.md` (Run 1: d24, 16,704 steps, 4.33e19 FLOPs, 10,949 s, CORE 0.25851; Run 2: d26+fp8, 2.91 h, CORE 0.2578): https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md
- `runs/speedrun.sh`, `nanochat/gpt.py`, `scripts/base_train.py` (architecture, ratio 12→tokens, batch 524,288)
- nanov41: [`benchmarks.md`](benchmarks.md) §3 / R69, `runs/r69_mfu_5090.json`, feasibility report §3x

## 6. H100 价格核验（2026-11-13，修正 §4 的 spot 假设）

§4 里 "spot $10-15/24.3 H100-h"（=$0.62–1.03/H100-h）来自 nanochat README 的自我报告，**不是普适市场价**。
按 grabgpu.com 的实时快照（17 家厂商，2026-11-13 检查）：

| 项目 | 核验值 |
|---|---|
| H100 报价区间 | **$1.39–5.99 / GPU-h**（中位 $2.50；同卡 4.3× 价差） |
| 最便宜在售 | $1.39（Lium，仅 1x 配置）；RunPod $1.99；Vast.ai $2.20；Hyperstack $2.50（1x–8x）；Lambda $3.99 |
| **spot 最低** | **$0.66 / GPU-h**（仅 1 家、单卡有货） |
| **8 卡整机当前在售最低** | **≈$2.50/H100-h（Hyperstack）→ 8×H100 节点 ≈$20/h ≈ ¥142/h** |
| AutoDL 国内 | H100/A100/A800/H800 查询时 **空闲 0 张**（无货） |

按 nanochat 实测 50% MFU（494 TFLOP/s/H100）重算 **¥/FLOPs**（5090 实测 38% MFU = 1.03×10¹⁷ FLOPs/¥）：

| H100 单价 | ¥/H100-h | FLOPs/¥ | vs 5090 |
|---|---:|---:|---:|
| $0.66（spot 底价，单卡） | 4.7 | 3.8×10¹⁷ | **3.7×** |
| $1.00 | 7.1 | 2.5×10¹⁷ | 2.4× |
| $1.39（最便宜在售） | 9.9 | 1.8×10¹⁷ | 1.75× |
| $2.50（**8 卡在售最低**） | 17.8 | 1.0×10¹⁷ | **0.97×（持平）** |
| $3.99（Lambda） | 28.3 | 6.3×10¹⁶ | 0.61×（更差） |

结论修正：**"8×H100 spot 单位算力明显更便宜"只在能锁定 ≤$1.4/H100-h 的 8 卡 spot 套餐时成立**
（1.7–3.7×）；按当前 8 卡在售价（$2.5/H100-h）与我们 5090 的 ¥/FLOP **基本持平**，按 list/on-demand 反而更差。
nanochat 的 $48/$15 数字需要低价 spot 才能复现。8×H100 的**确定**收益是：显存（可训 1.3B dense / d16 d20）
和墙钟时间（同 FLOPs 3h vs 单卡 5090 151h），不是必然的 ¥/FLOP 优势。

来源：https://grabgpu.com/gpus/nvidia-h100/?price=spot（2026-11-13 快照）
