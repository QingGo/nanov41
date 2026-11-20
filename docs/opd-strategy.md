# The 129k vocabulary is a hard constraint: OPD-centric strategy

Status: strategy note (2026-11-06). Supersedes the tokenizer lever in
[`product-vs-nanochat.md`](product-vs-nanochat.md) §3/L2.

The plan is to later run **on-policy distillation (OPD) from the official
DeepSeek-V4.1-Flash API using its top-20 logits**. The student must speak the **teacher's exact
129,280-token vocabulary** (token ids in the API's top-20 are only meaningful for that tokenizer),
so the vocab cannot be shrunk to 32k/64k. That changes the economics in a useful way: it kills the
"small tokenizer" lever but turns the project into a **distillation-based small model**, which is a
better fit for "high capability at low inference cost" than training a scratch GPT-2 clone.

## 1. What fixing the 129k vocab costs

The LM head is a `d → V` matrix, so its forward cost is `2·V·d` per token (FP8 in our stack, but it
is the floor of per-token compute):

| model dim `d` | head forward FLOPs/token (V=129,280) |
|---:|---:|
| 256 | 66 MFLOP |
| 384 (d12) | **99 MFLOP** |
| 512 | 132 MFLOP |
| 640 | 165 MFLOP |
| 768 | 199 MFLOP |
| 1024 | 265 MFLOP |
| 1536 | 397 MFLOP |

- At d12 the head is **~70% of the 141.3 MFLOP/token** decode cost and ~70% of forward training
  FLOPs. A larger model makes the head a smaller share, but the absolute head cost never goes away.
- Implications: keep `d` moderate (512–768) if inference cost is the product; do **not** assume the
  tokenizer lever can cut inference cost. Mitigations that keep the vocab: FP8 head (already on),
  FP4 head (R61 found no win at d12 on the 5090 — revisit at larger d), low-rank head (2–3× cheaper
  but may hurt distillation fidelity — needs an ablation), shared embedding/head (saves params, not
  FLOPs).
- Student/dense baselines for reference (rough, decode forward = head + 2·active + attention):

| student | total / active | dim | decode FLOPs/token (est.) | vs nanochat d24 (≈1.8 GFLOP) |
|---|---|---:|---:|---:|
| d12 today | 193M / 19.8M | 384 | 141 M | **12.5× cheaper** |
| d16 | ~350M / 35M | 512 | ~230 M | ~7.6× cheaper |
| d20 | ~600M / 60M | 640 | ~320 M | ~5.5× cheaper |
| d24 | ~1.4B / 150M | 768 | ~550 M | ~3.2× cheaper |

## 2. Why OPD changes the training plan

- Scratch training to GPT-2 CORE at 4.3e19 FLOPs (the nanochat allocation) is compute-optimal-ish and
  expensive per token; **OPD replaces most of the web-token budget with teacher feedback**: the
  student trains on its own rollouts scored by the teacher's top-20 logits, typically over
  **0.5–10B** tokens rather than 100B.
- Student training cost therefore drops: at ~1 GFLOP/token (d20-class), 2B OPD tokens ≈ 2e18 FLOPs
  → ~4.5 h on one 5090 (≈¥12); 10B tokens ≈ 22 h (≈¥62). A 5–10B-token web pretrain to give the
  student a reasonable starting policy costs only ~¥20–40 (100B-token runs are no longer the plan).
- **Student architecture includes Engram (mandatory).** The student is a V4.1-Flash port, and the
  teacher was trained with Engram; the table is a cheap capacity axis (d12: +128 M params,
  +0.70 % forward FLOPs, sparse updates) that fits the "quality per inference FLOP" goal.
  Sizing/placement: [`engram-design.md`](engram-design.md). Treat it as part of the student
  definition for OPD, not as an optional post-hoc add-on.
- The **dominant cost becomes the teacher API + on-policy data generation**, and the **storage of
  top-20 logits** (20 ids + 20 logprobs ≈ 120 B/token ⇒ 1B teacher-scored tokens ≈ 120 GB, 10B ≈
  1.2 TB; must subsample or compress). Both must be sized before committing.
- The scoreboard against nanochat therefore becomes: *quality per inference FLOP and per API dollar*
  — nanochat is the scratch-trained baseline, our model is the distilled student of a much stronger
  teacher. The 1M-context capability (teacher is native 1M) is a second differentiator.

## 3. What must be built (missing pieces)

1. **Teacher API layer**: top-20 logit endpoint semantics, pricing, rate limits, retries, async
   batching, version pinning (teacher drift invalidates data), long-context support.
2. **On-policy rollout student inference**: our repo has chunked prefill but **no generate/KV-cache
   engine**; OPD needs fast student sampling at 2K–128K context. This is the largest engineering
   piece (can borrow nanochat's `engine.py`/`infer_bench.py` patterns, MIT).
3. **OPD objective**: teacher top-20 support is not the full distribution; standard options are
   renormalized top-k KL / on-policy GKD-style losses (optionally mixed with teacher-forced NLL or
   hard-label data). Implement masked, chunked loss on top-20 ids; keep the FP8 head.
4. **Data pipeline (tokenizer reused)**: the V4.1-Flash tokenizer is pinned and verified
   (`docs/tokenizer-contract.md`); tokenize prompts with it, store (prompt, student rollout,
   teacher top-20) shards, dedupe/filter, long-context curriculum. No tokenizer training is needed.
5. **Eval harness**: teacher-match metrics (top-1 agreement, KL on held-out prompts), CORE/MMLU/
   ARC/GSM8K, long-context RULER/needle, chat format; plus inference FLOPs/latency/tokens/s.

## 4. Phased plan and gates

| phase | what | cost |
|---|---|---|
| P0 | API spec + OPD objective design + cost model (price/token, storage per token, target volume) + eval harness (borrow nanochat CORE/BPB, MIT) | **¥0** |
| P1 | Short web pretrain to seed the student: 5–10B tokens with the DeepSeek tokenizer on the planned mix (1×5090) | **¥20–40** |
| P2 | OPD pilot: 0.5–2B on-policy tokens, d12 student; measure teacher-match vs base student | GPU **¥5–20** + API |
| P3 | Student-size ladder (d12 → d16 → d20) at a fixed OPD token budget; pick the quality-vs-decode-FLOP Pareto point | GPU **¥30–60** + API |
| P4 | Final student: 2–10B OPD tokens + long-context OPD (32K→1M) + SFT/RL + evals + inference bench | GPU **¥100–200** + API |
| P5 | Head-to-head table vs nanochat: CORE/MMLU/ARC/GSM8K, decode FLOPs/latency, long context, total cost (GPU+API) | — |

Decision gates:

- **G1**: OPD pilot improves teacher-match/benchmarks over the same student trained on hard labels
  at equal token count. If not, OPD is not paying for itself at this scale.
- **G2**: at matched quality (e.g. CORE/MMLU), the student's decode FLOPs ≤1/3 of nanochat d24 and
  total cost (GPU+API) ≤ nanochat's training cost. Otherwise fall back to the inference/long-context
  positioning without the "beats nanochat" claim.
- **G3**: teacher API cost model fits budget (drive volume from the price of top-20 scoring, not the
  other way round); if API cost >¥1–2k, shrink the token budget or use a cheaper/open teacher.
- **G4**: long-context OPD works through the API (or via a local open teacher) before promising 1M.

## 5. Risks

- **API economics**: on-policy scoring means the teacher forward pass for the rollout prefixes; the
  bill can exceed the GPU bill by 10–100×. This is the single biggest unknown — size P2 only after
  getting the real price.
- **top-20 is a truncated distribution**: KL/top-k objectives are approximations; teacher-confidence
  bias can inflate harmless errors. Mitigate with renormalized KL + hard-label/teacher-forced mix.
- **Teacher drift/versioning**: any API model update changes the target; pin the version and cache.
- **Capacity**: a 19.8M-active student may be too small to absorb a 16B-active teacher; the d12→d20
  ladder is exactly the capacity test. The 129k head means capacity spent on the head, not the
  backbone.
- **Inference engine + rollout pipeline are missing** — this is the main schedule risk, not the GPU
  cost.

## 6. What this means for the old plan

- The 100B-token scratch pretraining in report §3x is **no longer the primary plan**; keep a short
  5–10B web pretrain as the student seed and spend the rest on OPD data quality/coverage.
- The V4.1-port track (Engram/mHC/1M, 129k vocab) and the OPD student track share the tokenizer and
  kernels — that is now an advantage, not a conflict.
- nanochat is still useful as the **scratch baseline** (same-tokenizer comparisons are impossible,
  but CORE/decode-FLOP comparisons are).

## 7. 实测 probe（2026-11-14）

官方 API 的 `deepseek-flash` = DeepSeek-V4.1-Flash-1017 已实测可用（FIM completion 端点 raw-text 打分，
`logprobs=20`，非思考模式）：2000 个 ClimbMix-val 位置上 **H_t≈1.47 bits、top-20 覆盖 0.96、
真实 token 在 top-20 内 92%、教师 CE 1.33 nats/token（bpb≈0.40，含 miss ~0.45–0.55）**；
学生 bpb 1.2006 → **可蒸馏 gap ≈0.8 bpb**。成本：强制逐 token 打分 ¥167/1M tokens（= 我们训练成本的
~31,000×），教师自生成 ¥4–8/1M tokens（~750–1,500×）。结论：**信息量远大于我们原先估计（之前假设
教师 bpb 0.6–0.7），但"同样 ¥"仍不可能做全量 KD**；可行的最小验证是"教师生成 + 软标签"的
10–100M token KD pilot（API ¥40–700 + GPU ¥1–3），在 `docs/scaling-learnings.md` §8.2 有完整分析。
原始数据：`runs/r91/teacher_probe/`；脚本：`tools/r91_teacher_probe.py`。
