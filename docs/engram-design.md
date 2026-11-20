# Engram in nanov41 — sizing, layer placement and relation to MoE (v0 proposal)

Status: **design proposal 2026-11-06; P0 + P1-CPU implemented 2026-11-06**
(`tools/engram_hash_parity.py`, `nanov41/nano_engram.py`, `train.py --engram`, 20 tests; GPU
graph/chunk integration still pending). Policy: Engram is **mandatory** in the OPD main training
path (decision recorded in [`engram-integration-plan.md`](engram-integration-plan.md); the earlier
"opt-in" decision is withdrawn). This document proposes the geometry we implement in P1 and the
arms we test in P2. Numbers are computed from `ref/config.json`, `nanov41/configs.py` and the
actual prime-bucket layout in `ref/engram.py`, not guessed.

## 0. TL;DR

| model | Engram layers | geometry | table params | bf16 table | table / MoE-total | expert-equivalents | added FLOPs/token | forward-FLOPs share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d12 (dim 384, 12L, 16 experts) | `[1, 4]` | ngram 3, 2 heads × 32, B=0.5 M | **128.0 M** | 0.256 GB | 1.42× (MoE 90.2 M) | 24.1 | +0.98 MFLOP | **0.70 %** of 141.3 |
| d16 (dim 512, 16L, 16 experts) | `[1, 6]` | ngram 3, 2 heads × 32, B=1 M | **256.0 M** | 0.512 GB | 1.20× (213.9 M) | 20.3 | +1.31 MFLOP | **0.59 %** of 223.1 |
| d20 (dim 640, 20L, 16 experts) | `[1, 7]` | ngram 3, 2 heads × 32, B=2 M | **512.0 M** | 1.024 GB | 1.23× (417.8 M) | 20.8 | +1.64 MFLOP | **0.48 %** of 339.2 |

Rule of thumb: **table ≈ 1.2–1.4 × MoE total ≈ 20–24 routed-expert-equivalents**, while the
Engram-active `wkv` is only **1–3 % of active-MoE params** and adds **< 1 %** forward FLOPs.
Table capacity (bucket base `B`) costs *memory only*; `head_dim`/`heads`/layers cost FLOPs.

## 1. Knobs and cost formulas

`EngramLayout` (`ref/engram.py`) turns the config into one prime-bucket range per
`(layer, n-gram size, head)`. With

- `L_E` Engram layers, `max_ngram_size = g`, `n_heads = h`, `head_dim = H`, bucket base `B`,
- `n_c = (g − 1) · h` hash ids per token (lookups/token),
- rows/layer `R ≈ n_c · B` (exact sum of primes; e.g. B=500 k, h=2, g=3 ⇒ R = 2,000,136),

the costs are:

```
table params        P_E   = L_E · R · H
wkv params          P_wkv = L_E · n_c · H · d · (hc_mult + 1)
forward FLOPs       ≈ 2 · P_wkv + O(n_c · H) dequant/gather     (independent of B!)
table memory (bf16)       = 2 · P_E bytes
sparse-Adam memory ≈ (2 + 2·moments_bytes) · P_E · touched_fraction
hash cache         = batch · max_seq_len · 8 B      (int64, non-persistent buffer)
```

Two consequences drive the whole design:

1. **Rows are free in FLOPs.** Growing `B` only grows table memory / optimizer bytes / checkpoint
   bytes; the per-token compute is set by `n_c · H` through `wkv`. This is what makes Engram a
   capacity axis rather than a compute axis.
2. **Heads/dim/layers are not free.** `P_wkv` and FLOPs scale with `n_c · H · d` and `L_E`. Keeping
   `H = 32` (the minimum fp8 block multiple in `ref/model.py`) is the cheapest useful choice at
   nano width; the V4.1 reference uses `H = 256` with `dim = 5120`.

## 2. The V4.1-Flash reference point (why not copy its ratios)

| quantity | V4.1-Flash |
|---|---:|
| backbone params (MoE total at 40 layers) | 545 B |
| Engram table | 2 layers `[1, 14]` / 40, g=4, h=8, H=256, B=16 M → rows 384,006,168 / 384,016,682, **196.6 B** params (~202.8 GB fp8) |
| table / MoE-total | **0.36×** |
| `wkv` params (2 layers) | 314.6 M = **3.2 %** of active-MoE params (9.91 B) |
| added FLOPs / token | 2 · 314.6 M ≈ 0.63 GFLOP (prefill-active ≈ 8 B params ⇒ ~3.9 % of the 16 GFLOP forward) |

Copying `table/MoE = 0.36` at d12 would give a **32.5 M-param table**, i.e. `B ≈ 127 k` buckets
per (n-gram, head) — below the compressed vocabulary (99,092) and ~8× more collision pressure than
our proposal, with only 0.5 M rows/layer. That is the wrong trade at nano scale: our `H/dim` is
1/12 (vs 1/20), so a row is cheap (32 floats), while collision pressure is expensive. We therefore
deliberately run a **larger table relative to MoE** (1.2–1.4×) while keeping the active `wkv` at a
similar small fraction of active MoE (1–3 % vs the reference's 3.2 %).

## 3. Proposed geometry (v0)

Fixed across sizes: `engram_max_ngram_size = 3`, `engram_n_heads = 2`, `engram_head_dim = 32`,
`engram_compressed_vocab_size = 99,092`, `engram_pad_id = 2`, `L_E = 2` layers.

| model | layers | `engram_layer_ids` | `engram_vocab_size` (B) | `engram_num_embeddings` (exact) | table params |
|---|---|---|---:|---|---:|
| d12 | 12 | `[1, 4]` | 500,000 | `[2,000,136, 2,000,370]` | 128.02 M |
| d16 | 16 | `[1, 6]` | 1,000,000 | `[4,000,112, 4,000,418]` | 256.02 M |
| d20 | 20 | `[1, 7]` | 2,000,000 | `[8,000,152, 8,000,396]` | 512.02 M |

The second row number differs slightly from the first because the reference draws primes from one
global `seen` set in layer order (that is why the official config is `[384006168, 384016682]`); this
is reproduced here, not approximated.

### Why 2 layers at these positions

- Relative depth matches the reference: V4.1 puts Engram at layers `[1, 14]` of 40 = 2.5 % and
  35 % depth. Scaled to our depths: d12 `[1, 4]`, d16 `[1, 6]`, d20 `[1, 7]` (0-based).
- Engram writes into the residual **before** the block, so the early layer shapes the representation
  used by everything above it; the mid layer adds phrase-level memory closer to where the model
  forms predictions.
- Layer-count is an ablation axis: 1 layer (half table, half wkv) or 4 layers with `B/2` each (same
  total table, denser injection but more collision pressure) are both testable in P2.

### Config snippet (d12; d16/d20 analogous)

```python
dict(
    ...
    engram_layer_ids=[1, 4],
    engram_max_ngram_size=3,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_vocab_size=500_000,
    engram_num_embeddings=[2_000_136, 2_000_370],
    engram_compressed_vocab_size=99_092,
    engram_pad_id=2,
)
```

`engram_num_embeddings[i]` must be ≥ the largest hash id of layer `i` (sum of its bucket primes);
using the exact sums above makes every row reachable and no id masked off.

## 4. Relation to MoE parameters

| quantity (d12) | MoE | Engram v0 |
|---|---:|---:|
| total params | 90.2 M (47 % of the 193.4 M backbone) | 128.0 M table + 0.49 M `wkv` (≈40 % of the 321.9 M total) |
| active params / token | 15.93 M (81 % of 19.8 M active backbone) | 0.49 M `wkv` (3.1 % of active MoE) |
| forward FLOPs / token | 2 · 15.93 M = 31.9 MFLOP | 0.98 MFLOP (3.1 % of MoE, 0.70 % of the whole forward) |
| optimizer | dense over every expert (all 16, incl. inactive) | **sparse**: only hash-touched rows |
| data dependence | router / token distribution | n-gram hashing over the pinned 129k tokenizer |
| what it adds | conditional computation (function approximators) | n-gram-keyed memory (knowledge/phrase recall) |

The useful exchange rate is the **routed-expert equivalent**: one expert costs
`L · 3 · dim · moe_inter_dim` params (5.31 M at d12, 12.58 M at d16, 24.58 M at d20).

- **d12 + Engram v0 = +24 expert-equivalents** at +128 M table params and +0.7 % FLOPs.
- The existing capacity lever **d12e32** adds +16 experts = +85.2 M params, also at ~0 extra active
  FLOPs, but needs **dense** gradients/Adam state for all 85 M params and a router that balances 32
  experts. Engram v0 gives ~1.5× that capacity with sparse updates at 0.7 % extra FLOPs — but it is
  **not a substitute**: MoE is conditional computation, Engram is memory. They are complementary;
  the A/B must include a parameter/FLOPs-matched MoE control.
- Because rows are FLOP-free, the `B` (capacity) decision is a **memory/optimizer budget** decision,
  while `n_c`, `H`, `L_E` are **compute** decisions. That is the main design lever vs MoE, where
  every added expert costs dense optimizer memory and routing load.

## 5. Memory / step-cost budget (d12 v0)

| item | estimate |
|---|---|
| table weights (bf16) | 0.256 GB (fp8 storage halves it; training stays bf16 first) |
| sparse Adam moments (bf16 m,v, worst case all rows touched) | 0.512 GB; a 10 % touched set ≈ 51 MB |
| dense grads for `wkv`/gate | < 2 MB |
| checkpoint growth per save | +0.256 GB |
| hash-id cache (`max_batch_size × max_seq_len` int64) | 64 KB at the current b=8/seq=1024; 8 MB at b=1/1M; 128 MB at b=16/1M |
| step-time target | ≤ +5 % (gathers + wkv + sparse update; to be measured, not assumed) |

The 1M-context chunked path needs the hash cache carried across chunk replays; the reference
`NgramHashState.cache` already writes by absolute `start_pos`, so this is a matter of keeping one
cache buffer per batch and replaying it, not recomputing hashes inside the graph.

## 6. Validation plan (P0 → P2)

| step | what | gate |
|---|---|---|
| P0 (¥0, CPU) | ✅ **done 2026-11-06**: `tools/engram_hash_parity.py` + `tests/test_engram_hash.py`; official rows 384,006,168 / 384,016,682 reproduced exactly, preset rows, hash ids deterministic / in-bucket / chunk-continuous | ids bitwise equal |
| P1 (CPU part ✅) | ✅ **done 2026-11-06**: `nanov41/nano_engram.py` (dtype-following table, deterministic sorted segment-sum row grads, sparse Adam, `install()`) + `train.py --engram {d12,d16,d20,tiny} --tokenizer`; `tests/test_engram_train.py` covers dense-grad equivalence, touched-row-only updates, state round-trip, model forward/backward, train.py smoke and checkpoint→eval-provider loading | no unexplained grad deltas |
| P1 (GPU part, pending 5090) | static hash-id graph input + chunked-replay cache + Triton sparse update/segment kernel; measure step-time overhead | ≤ +5 % step |
| **status** | **P0 + P1-CPU done 2026-11-06**; **P1-GPU eager validated 2026-11-07** on a rented 5090: d12 b=8×seq=1024 baseline 431.43 ms vs Engram 435.39 ms (**+0.92 %** fwd/bwd), Engram+sparse Adam 430.58 ms (within noise), peak 20.26 GiB, 128.02 M table params; evidence `runs/engram_gpu_5090_2026-11-07.json`; 20 engram tests pass locally and remotely | — |
| P1 (GPU graph part) | ✅ **done 2026-11-07** on the R68/R69 graph stack: R68 chunked graph 64K/16K **+4.86 %** with bitwise eager-vs-graph parity; R69 full-step graph d12 b16×1024 **+2.32 %** (default stack) and **+4.55 %** on the R69-tuned torch 2.14/tilelang stack (baseline 66.31 ms, same-host rounds); **gate kernels fused** — official DeepSeek TileKernels TileLang gate fwd/bwd for d16 (dim 512, **+1.60 %**), plus a vendored 128-tile / 1-warp-per-head adaptation (`nanov41/engram_kernels/`, MIT) that runs **d12 dim 384 / d20 dim 640 natively without padding** (validated vs `engram_gate_ref`: grads ≤3.6e-4; tuned d12 graph **+2.56 %** after fusing the sparse table Adam / removing dead zeroing (before that +4.5 %), tied with Inductor and ~8 % faster than the padded path); evidence `runs/graph_engram_tl_nano_5090_2026-11-07.json`, `runs/graph_engram_tlpad_5090_2026-11-07.json`; evidence `runs/graph_engram_ab_5090_2026-11-07.json`, `runs/graph_engram_r69tuned_5090_2026-11-07.json`, `runs/g47_engram_chunk_65536_5090.json` | met (≤ +5 %) |
| P2 (≈¥8–15) | ✅ **done 2026-11-07**: 5 arms × 0.6 B tokens at 0.6 B-token budget (E-none/E-thin/E-default/E-thick/d12e32), T0 multi-domain BPB; see [`p2-ab-report.md`](p2-ab-report.md) and `runs/p2/summary.json` | **E-default (B, 128 M table) wins**: −0.0275 BPB (−1.87 %) on Marin-phase1 mix vs d12; thin −0.014; thick **regresses**; d12e32 **no gain**. Keep default geometry; **3-seed replication (P2b) confirms the sign: phase-1 mix −1.87/−3.71/−2.26 %, macro −0.83/−4.39/−1.74 %**, no flip; teacher-match deferred (no OPD teacher data) |
| P2b | If the S0 gain is absent, shrink `B` or change layers — **keep the module** (policy) | — |

## 7. Open questions / risks

- **Collision pressure**: B=0.5 M buckets/lookup vs compressed vocab 99,092 (bigram space ≈ 9.8 B).
  Load ≈ tokens/B: 1.2 k accesses/bucket at 0.6 B tokens, 40 k at 20 B tokens, 2 M in V4.1 (B=16 M,
  33 T tokens). Our buckets are much less trained than V4.1's; that is acceptable for nano token
  budgets but the capacity curve must be measured (P2 thin/default/thick arms).
- **Benefit at nano scale is unproven**: Engram may only help knowledge-like probes and not
  teacher-match/reasoning; P2 is the gate that decides capacity, not whether to keep the module.
- **Init/gate stability**: reference table init is fp8-inference oriented; `init_random_weights`
  currently fails on fp8 params. P1 needs a bf16 table and a documented init (small table norm,
  gate near zero) so early training is not dominated by random memory.
- **Determinism**: mandatory path ⇒ deterministic sorted-unique segment-sum, no parity exception.
- **OPD interplay**: teacher top-20 logits supervise outputs, not Engram rows directly; table
  updates during OPD should use a smaller LR / sparse schedule — an P2b/P3 question.
- **Tokenizer lock**: all hash multipliers derive from the compressed vocab size; changing the
  tokenizer invalidates the table (`docs/tokenizer-contract.md`).
- **1M context**: cache size scales with `batch × max_seq_len`; must be explicit in the long-context
  training config and the chunked-graph trainer.

## 9. Engram allocation & size scaling（2026-11-13 调研 + 实验设计）

**问题**：不同大小的模型（dense h=768 backbone 157.7M vs MoE backbone 752M）用同一个 production
Engram 220.05M 是否合理？

### 9.1 文献结论

- **Engram 论文**（DeepSeek, *Conditional Memory via Scalable Lookup*, arXiv:2601.07372, §3）：
  1. **有限参数预算下存在 U 型分配律**。定义 ρ = MoE 分到的 inactive 参数预算比例
     （P_MoE=ρ·P_sparse，P_Engram=(1−ρ)·P_sparse），在 iso-param + iso-FLOPs（P_tot/P_act≈10）下，
     把 MoE 预算压到 **ρ≈40%** 仍与纯 MoE 持平，最优在中间 → **Engram 最优大小取决于总预算，不是固定
     绝对值**；他们据此把 72→55 个 routed experts 的参数挪给 5.7B 的 Engram（26.7B total / 3.8B active）。
  2. **无限记忆区**：加密 memory slots 的验证 loss 呈 **严格 power law（log-linear）改善**，且不增加
     每 token FLOPs（Engram-40B：18.5B Engram / 39.5B total 继续优于 27B）。
- **Memory Layers at Scale**（Meta, arXiv:2412.09764）：记忆层容量可扩到 **128B 参数**，在 iso-compute
  和 iso-param+FLOPs 两种口径下都优于 dense/MoE；收益在 factual 任务上最大 → 记忆是独立的容量轴。

### 9.2 我们的现状与"同 220M"的问题

| 配置 | backbone | Engram | total（含 wkv） | P_tot/P_act | Engram 占 sparse 预算 | Engram/非嵌入 backbone |
|---|---:|---:|---:|---:|---:|---:|
| MoE G4（production） | 752.2M | 220.05M | 976.2M | **10.2**（≈论文的 10） | 27%（ρ_MoE=73%） | 0.355 |
| MoE G8 | 749.9M | 220.05M | 973.9M | 10.4 | 27% | 0.356 |
| dense h=768 | 157.7M | 220.05M | 381.7M | **4.0** | 100%（无 inactive 专家） | **8.7×** |
| dense h=33024（容量等） | 752.2M | 220.05M | 976.2M | 1.42 | 100% | 0.355 |

→ 对 MoE 臂：220M 与生产/论文口径一致（ρ_MoE≈73%，接近论文 U 型中"Engram 偏少"一侧），合理。
→ 对 **dense h=768**：Engram 占总参数 **58%**、是 backbone 的 **8.7×**，远超生产 0.355×、也超过论文
Engram-40B 的 47%——这是**过度分配**，在 200M token 屏幕上很可能不划算（P2 实测：旧 d12 在 0.6B token
时 256M 表比 128M 表回退，说明表过大在低 token 预算下会 under-fit）。
**结论：同一 220M 只适合作为"控制变量组"（隔离 FFN），不能当作分配最优；主对照应改成 iso-param 分配律。**

### 9.3 几何约束（为什么不能简单按比例缩小）

生产几何 = 2 层 × 8 heads × ngram4 × head_dim 32 ⇒ **每 bucket 1,536 个 table 参数**；bucket 数 B 必须
≥ 压缩词表 99,092（否则哈希碰撞爆炸）。于是：

| 目标 | B | 表参数 | 备注 |
|---|---:|---:|---|
| 生产 | 142,935 | 220.05M | ✓ 与 d12_aligned 一致 |
| **几何最小可行** | 99,092 | **152.63M** | 再小必须缩几何 |
| 2× 生产 | 285,870 | 439.69M | |
| 4× 生产 | 571,740 | 878.62M | 32GB 显存风险 |
| 小几何（ngram3, 2 heads） | 99,092 | 25.38M | 每 token 注入量也变小（另一变量） |

按官方 0.355×非嵌入 backbone 规则，dense h=768 应配 ~9M Engram — **在生产几何下不可实现**（下限 152.6M）。
因此 "按比例缩小" 只有两条路：用几何最小可行 152.6M，或缩几何（同时改变注入强度）。

### 9.4 修订后的实验设计（配置已生成）

**Block A（主实验，复现论文 U 型分配律）**：iso-param≈976.2M、iso-FLOPs 183.0M/token、几何固定，
只改变 ρ（expert 参数 ↔ Engram slots）：

| arm | routed experts | Engram 表 | Engram 占 sparse | 说明 |
|---|---:|---:|---:|---|
| A0-moe-e128 | 128 top2 | 220.0M | 27% | production |
| A1-moe-e96 | 96 top2 | 371.0M | 45% | |
| A2-moe-e64 | 64 top2 | 522.0M | 64% | |
| A3-moe-e32 | 32 top2 | 673.0M | 82% | |
| **A4-dense-eng815** | dense FFN h=768 | **814.6M** | **100%（ρ→0）** | 全部 sparse 预算变 memory |

**Block B（表大小最优值）**：固定 dense FFN（FLOPs-equal），Engram 表 152.6M / 220.0M / 439.7M，
看 200M token 屏幕上表大小与 bpb 的关系（对应 P2 在旧 d12 上的 64/128/256M 曲线）。

成本：8 臂 × 200M ≈ **3h ≈ ¥8.5**（单卡 5090）+ 冒烟 ~¥0.7；之后对最优 2–3 个点在 2B token 上确认
（≈¥20）以测"最优点是否随 token 预算移动"。

### 9.5 实验结果（2026-11-13，单卡 5090，v41 优化器，200M tokens，P0 5M 子样本）

协议：production 栈（v41 优化器、Engram ON、129k vocab、seq 8192、b2×8192×accum64）、warmup 30、
每臂 200M tokens、P0 5M-token held-out 子样本、LR 8.22e-4。失败经验：adamw baseline + Engram 在
190 步长跑下会 OOM（行状态/合并缓冲随步数增长），**这批实验全部用 v41 优化器**（生产路径）。

**A. ρ 分配扫描（iso-param 976.2M、iso-FLOPs 183.0M fwd/token）**

| arm | FFN | Engram | Engram 占 sparse 预算 | P0 bpb | tok/s | MFU | ¥ |
|---|---|---|---:|---:|---:|---:|---:|
| A0-moe-e128 | MoE 128 top2（production） | 220.05M | 27% | 1.8035 | 145.7k | 38.2% | 1.13 |
| A1-moe-e16 | MoE 16 top2 | 748.5M | 91% | 1.8004 | 151.3k | 39.6% | 1.09 |
| A2-moe-e64 | MoE 64 top2 | 522.0M | 64% | **1.7980** | 151.8k | 39.8% | 1.09 |
| A3-moe-e32 | MoE 32 top2 | 673.0M | 82% | 1.8022 | 153.4k | 40.2% | 1.08 |
| **A4-dense-eng815** | **dense h=768** | **814.6M** | **100%（ρ→0）** | **1.7926** | **163.3k** | **42.8%** | 1.01 |
| A5-moe-none | MoE 128 top2 | 0 | 0% | 1.8905 | 150.7k | 39.5% | 1.09 |

（A5 不是 iso-param：总参 752.2M，因为 fast layout 要求 2 的幂专家数，E=128 是该预算下最大。）

**B. dense 表大小扫描（固定 dense h=768 FFN，FLOPs-equal）**

| arm | Engram 表 | B | P0 bpb | tok/s |
|---|---|---:|---:|---:|
| B1 | 152.6M | 99,092 | 1.8065 | 164.3k |
| B2 | 220.0M | 142,935 | 1.8020 | 164.1k |
| B3 | 439.7M | 285,870 | 1.7975 | 163.9k |
| A4 | 814.6M | 530,025 | **1.7926** | 163.3k |

**结论**

1. **Engram 是当前尺度下最强的容量杠杆**：纯 MoE（A5，1.8905）→ 任意 Engram 臂（1.7926–1.8035），
   **−0.087~−0.098 bpb（约 −4.6%~−5.2%）**，远超旧 P2 家族测到的 −1.9%。
2. **同 220.05M 确实偏低（你的质疑成立）**：在 iso-param/iso-FLOPs 下把专家预算换成 Engram 持续改善：
   27% → 64% 份额 −0.0055 bpb；**→ 100%（dense FFN + 814.6M 表）−0.0109 bpb（−0.60%），且吞吐还快 12%**
   （163.3k vs 145.7k，MFU 42.8% vs 38.2%）。MoE 一侧的 ρ 曲线在 64–91% 之间很平（±0.002 噪声内），
   **当前尺度最优点在 ρ→0（全 memory）一侧**。
   - 与 Engram 论文的 U 型最优（ρ≈40–60%）不矛盾：他们的实验在 **2e20–6e20 FLOPs、568–993M active**
     下做，比我们大 ~500×；我们的 200M-token/91.5M-active 区间是"专家训不动、memory 行学得快"的区域。
3. **dense 表大小在 200M tokens 下仍单调收益**：每翻倍表 ≈ **−0.0058 bpb**（log-linear），到 814.6M 仍未饱和，
   与 Engram 论文"无限记忆区 power law"一致；P2 旧家族 256M 回退的现象没有复现（v41 + 新几何 + 大 token 预算不同）。
4. **候选产线改动**：把 FFN 从 128 专家 top-2 改为 **dense h=768 + 814.6M Engram**（iso-param 972–976M、
   iso-FLOPs 183M），在 200M 屏幕上更优且更快；需要在 **1–2B tokens** 上复核（大 token 预算可能把最优点
   拉回专家一侧），再决定是否改 7.5B 产线。

> **后续（2026-11-13 已脚本化）**：用 R91 四层级联在更大预算上复核本节的分配结论 ——
> B 层把 Engram 表大小扫描搬到 503.3M tokens，D1 在 1.007B tokens 上确认 top 候选 + 产线对照，
> 详见 `docs/r91-search-protocol.md`（脚本 `tools/r91_queue.sh`）。

> **B 层实跑结果（2026-11-14，302M tokens/臂，v41，WSD）**：dense h768 表 220M→1.3110、440M→1.3044（−0.0066）、
> 815M→1.3028（−0.0016，开始饱和）；MoE E128+eng220 1.3446、E16+eng748 1.3230（往 memory 挪 −0.0216）。
> 即：**200M 屏幕的"表越大越好"在 302M 上仍成立但已接近平台；dense+memory 在两个预算都赢 MoE。**
> 详见 `docs/scaling-learnings.md` §7 与 `runs/r91/b/`。
