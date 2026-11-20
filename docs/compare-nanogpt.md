# nanov41 vs nanoGPT — parameters, data, compute and cost

A side-by-side of this project (d12, a downscaled DeepSeek-V4.1-Flash training port) and
[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), which is the natural reference point for
"train a GPT from scratch on one to eight GPUs". The two are **not the same kind of model**, so the
table separates *scale*, *data*, *compute* and *cost* and states the assumptions. All nanoGPT
numbers below are from the upstream README/config files (fetched 2026-11-05); all nanov41 numbers
are measured in this repository on one RTX 5090 (see [`benchmarks.md`](benchmarks.md) §3 / R69) and
the 100B-token plan in the feasibility report §3x.

## 1. Model

| | nanoGPT — Shakespeare demo | nanoGPT — GPT-2 124M repro | **nanov41 d12** |
|---|---|---|---|
| Architecture | dense GPT-2 decoder | dense GPT-2 decoder | sparse MoE (16 experts, top-2) + mHC 4-way residual + sparse/compressed attention |
| Parameters (total) | ~10.7M (6 layers, 6 heads, 384 dim) | 124M | **193.4M** |
| Parameters (active / token) | 10.7M (dense) | 124M (dense) | **19.8M (10.2% of total)** |
| Vocab | 65 (char) | 50,257 (GPT-2 BPE) | 129,280 (multilingual) |
| Context length | 256 chars | 1,024 tokens | **up to 1M** (chunked BPTT; 32K/128K/1M stages) |
| Positions / attention | learned pos emb, full causal | learned pos emb, full causal | RoPE, window + compressed-KV sparse attention |
| Downstream goal | learn char statistics | reproduce GPT-2 on OWT | pretrain a V4.1-Flash-family MoE backbone (research port) |

## 2. Data

| | Shakespeare demo | GPT-2 124M repro | **nanov41 d12** |
|---|---|---|---|
| Dataset | Tiny Shakespeare, 1,115,394 chars (~1 MB) | OpenWebText, ≈ 9B GPT-2-BPE tokens (~40 GB) | web + code + math + zh + wiki/books mix (see §3x) |
| Tokens trained on | ~82M char tokens (64×256×5000 iters ≈ 82 epochs) | **300B tokens** (491,520 tok/iter × 600k iters ≈ 33 epochs of OWT) | **100B tokens recommended** (20B pilot; 300B optional stretch) |
| Tokens / total param | ~7,700 | 2,419 | **517** (5051 per *active* param) |
| Stage plan | single run | single run | S0 mix ablations (~¥7) → S1 short-context → S2 long-context (32K/128K/1M) → S3 cooldown 5% |

Notes: the GPT-2 original (OpenAI, 2019) saw WebText ≈ 8–10B tokens (≈ 73 tok/param); nanoGPT's
reproduction intentionally overtrains to ~300B tokens. nanov41's 517 tok/total-param sits in the
OLMo2 / DCLM-7B 371–528 band; per **active** parameter it is a heavily overtrained 5051 tok/param,
which is the explicit trade-off of the MoE design (cheap inference, expensive-looking data budget).

## 3. Compute per token and total run

| | GPT-2 124M (dense) | **nanov41 d12** |
|---|---:|---:|
| Training FLOPs / token | ≈ 744 MFLOP (6N; ~810–850 incl. 1K attention) | **423.9 MFLOP** (3× forward FlopCounter, incl. the 129k-vocab head) |
| Tokens in the reference run | 300B | 100B (recommended) |
| Total training FLOPs | ≈ 2.3 × 10²⁰ (234 EFLOPs) | **4.24 × 10¹⁹ (42.4 EFLOPs)** |
| Measured throughput (per GPU) | ≈ 108.5k tok/s/GPU (300B / 4 days / 8×A100) | **293.5k tok/s** on 1×5090 |
| Effective model MFU | ≈ 26% of A100 bf16 peak (derived) | **59.4% of 5090 bf16 peak** (124.4 TFLOP/s) |
| Hardware of the reference run | 8×A100 40GB DDP | 1×RTX 5090 (32 GB) |
| Wall time | ~4 days (config comment: ~5 days) | **94.7 h ≈ 3.94 days** for 100B |

Per-token compute is ~1.8× lower for nanov41 because only 19.8M parameters are active and the
backbone is small. Note that the 129,280-vocab head is ~70% of nanov41's remaining 423.9 MFLOP/token
(≈ 2·V·d = 99 MFLOP forward ⇒ ≈ 298 MFLOP with fwd+bwd); the **backbone** is only ≈ 6·N_active ≈
119 MFLOP/token vs GPT-2's ≈ 512 MFLOP/token, i.e. ~4.3× cheaper before the vocabulary term. Per-GPU token
throughput is ~2.7× higher, from both the lower FLOPs/token and the much higher MFU (fused
Triton/TileLang kernels, CUDA-graph replay, FP8 head; nanoGPT deliberately stays plain PyTorch+DDP).

## 4. Cost

Assumptions: AutoDL RTX 5090 = **¥2.78/GPU-h** (2026-11 rate); A100-40GB = **¥3/GPU-h** for the
"same cloud price" column, and 2023 Lambda Labs list ≈ $1.29–1.99/A100-h for the "as-run 2023"
column; ¥7.2/$. Overheads (restarts/eval) excluded here (+15% in the report).

| Run | GPU-hours | Cost @ same cloud price | Cost as-run (2023 list) |
|---|---:|---:|---:|
| nanoGPT Shakespeare char (1×A100, ~3 min) | 0.05 | ¥0.15 | ~$0.09 |
| nanoGPT GPT-2 124M, 300B tok (8×A100, ~4 d) | 768 | **≈ ¥2,300** | **≈ $990–1,530 ≈ ¥7,100–11,000** |
| **nanov41 20B pilot** (1×5090) | 18.9 | **≈ ¥53** (≈¥60 with 15% overhead) | — |
| **nanov41 100B recommended** (1×5090) | 94.7 | **≈ ¥263** (≈¥303 with overhead) | — |
| nanov41 300B stretch (1×5090) | 284 | ≈ ¥790 (≈¥910) | — |
| Cost per 1B tokens | 0.95 vs 2.56 | **¥2.63** vs **¥7.7** | — |

The GPT-2 numbers are a *reference implementation* cost, not a tuned run; the same OWT/GPT-2 target
has been done much faster by the "speedrun" forks (e.g. modded-nanogpt), which is a fairer
efficiency baseline than the original nanoGPT script.

## 5. What this comparison does and does not say

- **Does say**: at the same cloud price, nanov41's training tokens are ~2.9× cheaper and its
  per-GPU throughput ~2.7× higher, and it targets a much longer context (1M vs 1K) and a
  modern sparse-MoE architecture. The 100B-token pretraining run is a ~¥300 / ~4-day exercise on a
  single consumer GPU.
- **Does not say**: it says nothing about model *quality* yet. nanoGPT ships concrete loss numbers
  (GPT-2 124M ≈ 2.85 on OWT after the repro recipe; Shakespeare char ≈ 1.47); nanov41 has validated
  numerics/parity/kernels/throughput but has not yet run its 20B/100B pretraining, so no
  loss-per-token or downstream comparison exists. A 20B pilot (~18.9 h, ≈¥60) is the next gate.
- The two projects also have different goals: nanoGPT is a minimal, readable GPT-2 reproduction
  (now deprecated upstream in favour of nanochat); nanov41 is a fused-kernel, CUDA-graph,
  sparse-MoE training stack that downscales DeepSeek-V4.1-Flash by 226.5× in activated parameters
  (see [`v41-scaling-check.md`](v41-scaling-check.md)).

## 6. Sources

- nanoGPT README, `config/train_gpt2.py`, `config/train_shakespeare_char.py`,
  `data/shakespeare_char/prepare.py` — https://github.com/karpathy/nanoGPT
- nanov41 R69 measurements: [`benchmarks.md`](benchmarks.md) §3, `runs/r69_mfu_5090.json`
- nanov41 training plan (100B / mix / stages): feasibility report §3x
