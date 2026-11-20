# Engram in nanov41 — feasibility and integration plan

Question: can the training port "hook up Engram", and do
[EngramDB](https://github.com/QingGo/EngramDB) and
[engram-peft](https://github.com/QingGo/engram-peft) help? Short answer: the V4.1 Engram code is
already in this repository (`ref/engram.py`, wired through `ref/model.py`); what is missing is the
**tokenizer/data pipeline plus a training-capable table implementation**. `engram-peft` is directly
reusable (it solves the hash precompute / sparse-optimizer / gate problems). `EngramDB` is an
**inference-side disk store for tables larger than HBM** — not needed at nano scale, relevant only
when the table outgrows the GPU or we serve a full V4.1 model.

> **Update 2026-11-06 (policy change: Engram is mandatory):** Engram is **required** and is part of
> the **OPD main training path** — the earlier "opt-in / not in the OPD main path" decision is
> withdrawn. Reasons: (a) the student is a V4.1-Flash port and the teacher was trained with Engram,
> so an Engram-less student is a structurally different hypothesis class; (b) the OPD objective is
> teacher-match on the student's own rollouts, and a capacity axis that improves held-out
> loss/teacher-KL for ≲1 % extra forward FLOPs is exactly what a 200–600 M student needs;
> (c) the tokenizer prerequisite is solved (129k vocab + compressed map **99,092** pinned);
> (d) the capacity-per-¥ analysis ([`capacity-vs-compute-codesign.md`](capacity-vs-compute-codesign.md))
> favors the sparse table over adding dense experts.
> **Sizing / layer placement / geometry: [`engram-design.md`](engram-design.md).**
> Consequence: P1 (training-capable Engram) is on the critical path, not a later milestone, and the
> deterministic sparse-gradient requirement below is mandatory (no opt-in parity exception).
> See also [`opd-strategy.md`](opd-strategy.md).

## 1. What already exists here

- `ref/engram.py`: full V4.1 Engram — compressed-token map (NFKC/lowercase, 129,280 → 99,092 ids),
  per-(layer, n-gram, head) prime buckets, int64 hash multipliers, `NgramHashState` (2..max_ngram
  hashes per position, cache across prefill/decode), `ParallelEngramEmbedding` (sharded table,
  fp8 rows + block scales) and `Engram` (hash → `wkv` → gate against the residual stream → additive
  write).
- `ref/model.py` / `nanov41/_ref_patched_model.py`: `EngramLayout` built from
  `engram_layer_ids`, `Block.engram` at the selected layers, so the training port already contains
  the code path (P1 of the patch script removes `inference_mode`, so gradients reach `wkv` and the
  gate).
- `nanov41/nano_engram.py` (**new, P1-CPU**): training-capable table (dtype-following buffer,
  deterministic sorted segment-sum row gradients, sparse Adam), `install()` replacement for
  `ParallelEngramEmbedding` + the inference-mode hash unwrap, d12/d16/d20/tiny presets, tokenizer
  adapter; `train.py --engram ... --tokenizer ...` wires it into the training loop.
- `tools/engram_hash_parity.py` (**new, P0**): checks official/preset rows against prime-bucket sums
  and hash-id determinism/range/chunk continuity.
- `nanov41/configs.py`: the knobs exist (`engram_layer_ids`, `engram_num_embeddings`,
  `engram_max_ngram_size`, `engram_vocab_size`, `engram_n_heads`, `engram_head_dim`,
  `engram_compressed_vocab_size`) but nano configs set them to zero with the comment
  *"Engram tables need the tokenizer; enabled in a later milestone"*.
- Full V4.1 geometry (for reference): 2 layers `[1,14]`, max n-gram 4, 8 heads × 256 dim,
  384 M rows/layer ⇒ ~196 B fp8 rows (~202.8 GB on disk — the number EngramDB targets).

## 2. What it would take for nanov41 (training from scratch)

| item | status | work |
|---|---|---|
| DeepSeek-V4.1-Flash tokenizer + compressed-token map (`99092`) | ✅ **verified** — reuse the pinned V4.1-Flash `tokenizer.json`; V3 tokenizer gives 128,815/98,627 and must not be used (`docs/tokenizer-contract.md`) | only the data pipeline (tokenize/dedupe/pack) remains |
| training-capable table | ✅ **implemented** in `nanov41/nano_engram.py` (P1-CPU) | dtype-following buffer table, sparse row grads, sparse Adam; fp8 training table later |
| deterministic sparse gradient | ✅ **implemented (CPU/reference)** — `segment_sum_rows` uses stable argsort + fp32 prefix sums, no atomics; CUDA kernel + graph parity still pending | keep the bitwise eager-vs-graph gate; no opt-in parity exception |
| CUDA-graph / chunked BPTT | ✅ **done 2026-11-07** on the R68/R69 graph stack (`nanov41/chunk_graph.py` + internal `tools/bench_graph_train.py`/`g47_engram_chunk_graph.py`): hash ids are recomputed inside the captured graph, the table gather's sparse leaf grads are captured as pool buffers and reduced outside by `SparseRowAdam`; R68 2-step eager-vs-graph is bitwise identical (weights + losses), R69 full-step capture PASS | measured below |
| nano geometry (v0, see [`engram-design.md`](engram-design.md)) | full V4.1 geometry is too wide for d12 | d12: 2 layers `[1,4]`, max-ngram 3, 2 heads × 32 dim, B=0.5 M buckets ⇒ table **128 M params / 0.26 GB bf16**, `wkv` 2×(4×32)→1920 ⇒ **+0.98 MFLOP/token (0.70 %** of the current 141 MFLOP/token), 4 hash ids/token; d16/d20 scale rows ×2/×4 (256 M / 512 M params) |

So the architecture port is ~1–2 days; the training adaptation (sparse + deterministic + graph) is
~3–5 days and is now on the OPD critical path; the tokenizer/data prerequisite is already met
(pinned tokenizer + compressed map 99,092).

## 3. How the two QingGo repositories help

### engram-peft — directly reusable (best fit)

Paper-aligned PEFT implementation of the same DeepSeek Engram (arXiv 2601.07372): forward-hook
injection into HF Transformers, `EngramConfig(target_layers=...)`, `train_mode` (`engram_only` /
`preserve_trainable` / `full_finetune`), LoRA+Engram stacking, CPU hash precomputation
(`EngramDataCollator`, `num_workers>0`, "zero hashing overhead on GPU"), sparse Adam and sparse
gradient clipping, DDP/DeepSpeed, and an EngramDB Store-I integration. Reported TinyLlama-1.1B
benchmark: +545 M Engram params, ~1% of them updated per step, +1.3 GB memory, +8% step time;
LoRA+Engram eval loss 0.9656 vs LoRA 0.9890 and Engram-only 1.0165.

Two ways to use it:
1. **Now (¥0 GPU, CPU/1 small GPU):** use it on a small HF model (TinyLlama / Qwen) to reproduce the
   "knowledge injection" gain before we port anything — a cheap go/no-go signal for Engram at all.
2. **Later (our stack):** borrow the pieces we would otherwise rebuild — `hashing.py` + `collator.py`
   (tokenizer compression + multi-head hash ids), `layer.py` (gate/hash → residual), `trainer.py`
   (sparse Adam/clipping) — into `nanov41/enano_engram.py`, or wrap our model in an HF-style module
   and let engram-peft inject through hooks. Also a natural fit for **S4 SFT / knowledge PEFT** after
   the base checkpoint exists (`train_mode="engram_only"`), which is cheaper than pretraining Engram
   from scratch.

### EngramDB — inference-scale storage, later

Disk-first engine for huge **static read-only** Engram/PLE tables (`build → index → warm → serve`,
Store-I raw rows vs Store-P materialised view, NVMe prefetch, 28.6× Store-P fold gain, Python/Rust/C
ABI, vLLM/SGLang/engram-peft integrations). It explicitly does **not** do model forward compute or
the training write path. At d12 scale the table (0.5–2 GB) fits in HBM, so EngramDB is not on the
critical path. It becomes relevant when:
- the Engram table (or a future scaled-up nanov41) no longer fits in GPU/CPU memory — then EngramDB
  is the read path; or
- we serve a full V4.1-class model (202.8 GB table) — the intended use case.
It also provides `engramdb-keygen` deterministic V4.1 rowids/`PleSpec`, useful to **cross-check** the
hashing in `ref/engram.py` and engram-peft.

## 4. Phased plan (cheap first)

| phase | what | cost / effort |
|---|---|---|
| P0 | ✅ Hash parity done 2026-11-06 (`tools/engram_hash_parity.py`, `tests/test_engram_hash.py`): official rows exact, preset rows, hash ids deterministic/in-bucket/chunk-continuous | ¥0, done |
| P1 | Nano-Engram **required-path** support: geometry from [`engram-design.md`](engram-design.md); ✅ CPU path done 2026-11-06; ✅ GPU eager validated 2026-11-07; ✅ **R68/R69 CUDA-graph integration done 2026-11-07** (d12: R69 full-step graph +2.32 %, R68 chunked graph +4.86 %, both within the ≤5 % target; R68 parity bitwise; evidence `runs/graph_engram_ab_5090_2026-11-07.json`, `runs/g47_engram_chunk_65536_5090.json`) | done (Triton kernel optimisation is optional follow-up) |
| P2 | ✅ **done 2026-11-07** (S0-scale A/B, capacity gate): 0.6 B tokens × {E-none, E-thin, E-default, E-thick} + d12e32 control; T0 multi-domain BPB. Result: **E-default wins** (phase-1 mix −1.87 % vs d12; thin −0.95 %; thick regresses; d12e32 +0.23 % worse). Teacher-match deferred: no OPD teacher logits cached. **P2b seed replication (seeds 1235/1236): E-default beats E-none on phase-1 mix in all seeds (−3.71 %, −2.26 %; P2 seed1234 −1.87 %) → sign robust, magnitude 1–4 %.** Report: [`p2-ab-report.md`](p2-ab-report.md), raw `runs/p2b/summary.json` | done (~¥18 GPU total) |
| P2b | Capacity decision: pick the table size that maximizes capability/¥; if Engram shows no gain at S0, shrink it or change layer placement — do **not** remove it from the architecture | — |
| P3 | If the gain is real, include Engram in the 20B pilot; else record a negative result in the playbook | — |
| P4 | S4 SFT / knowledge PEFT via engram-peft on top of the 20B/100B checkpoint (`engram_only` / LoRA+Engram) | ~¥5–15 |
| P5 | EngramDB only if the table outgrows HBM / full V4.1 serving | later |

## 5. Graph training integration (2026-11-07)

Mechanics of the R68/R69 integration (all in `nanov41/nano_engram.py`, `nanov41/chunk_graph.py` and
the internal graph drivers):

- **Hash ids inside the graph.** `NgramHashState` runs inside the captured forward (it has no host
  syncs); the static `ids` buffer is copied before each replay and the n-gram cache is a persistent
  buffer, so chunked lookbacks replay correctly.
- **Table gather as a captured leaf.** `NanoEngramTable.forward` detaches into a leaf; during capture
  `torch.autograd.grad(loss, params + leaves)` bakes the leaf-grad output buffers into the graph
  (R69) or the captured backward accumulates into them (R68).  `set_static_records(True)` keeps the
  capture-time `(ids, leaf)` records, and `zero_leaf_grads()` zeroes the captured grad buffers in
  place before every replay (never replaces them).
- **Sparse update outside the graphs.** After replay, `SparseRowAdam.step()` reduces leaf grads with
  the deterministic sorted segment-sum and updates only touched rows; this stays outside both the
  per-chunk graphs and the full-step graph.
- **Mixed dtypes.** Engram adds fp32 gate params to a bf16 backbone; `torch.autograd.grad` can return
  fp32 grads for bf16 params, so the graph driver casts grads to the param dtype and auto-disables
  fused AdamW when param dtypes are mixed (both A/B arms use the same optimizer setting).
- **Eager-vs-graph parity** is checked by R68 (`g47_engram_chunk_graph.py`, 2 steps, weights and
  losses bitwise identical); R69 has `--engram` + `--graph-only` for pure step/MFU measurement.

Measured on a 5090 (default graph stack, torch 2.12.1+cu130): R69 full-step graph b16×1024
221.49 → 226.62 ms (**+2.32 %**), R68 chunked graph 64K/16K 427.8 → 448.6 ms (**+4.86 %**), both
within the ≤5 % target; evidence `runs/graph_engram_ab_5090_2026-11-07.json`.

On the R69-tuned stack (`/root/t141`, torch 2.14.1+cu130, triton 3.8, tilelang 0.1.15,
`tools/final_env.sh` + R69 accepted flags) the same-host median is 66.31 ms baseline vs 69.60 ms
Engram (**+4.55 %**, 4 alternating rounds) for d12; d16 (dim 512) with the **official DeepSeek
TileKernels TileLang gate fwd/bwd** is 138.74 → 140.96 ms (**+1.60 %**). Evidence
`runs/graph_engram_r69tuned_5090_2026-11-07.json`.  Note the historical R69 58 ms was a west-D
torch-2.12 measurement; this bj-B1 host gives 66 ms with the same env, so only the paired relative
overhead is meaningful.

**Gate kernel fusion (2026-11-07).** The training forward now offers three gate paths, selected
automatically by `nano_engram.install`:
- **official TileLang kernels** (`TileKernels/tile_kernels/engram`: `engram_gate_fwd`/`engram_gate_bwd`,
  MIT) with `TILE_KERNELS_PATH`; used when `hidden_size % 256 == 0` (d16 dim 512; production
  4096/7168 also qualify).  `nanov41/engram_tl.py` wraps them in a custom autograd Function.
- **Inductor/Triton-fused bf16 gate** (`torch.compile` on the torch forward) for the other nano dims
  (d12 384, d20 640); this cut the tuned d12 graph overhead from +6.1 % (eager torch gate) to +4.55 %.
- plain torch bf16 gate fallback when neither is available.
**Nano 128-tile kernel fork (2026-11-07).**  Instead of relying on zero-padding, the official gate
kernels were vendored into `nanov41/engram_kernels/` (MIT, DeepSeek copyright retained) and adapted
for the nano hidden sizes:
- forward: add a `blk_d=128` tile (`threads=32 × vec_size=4` bf16) to the valid-block list;
- backward: `warps_per_head = 1` (128 threads) when `hidden % 256 != 0`, a dynamic `v_copy_vec_size`
  for 384/640, and a single-warp `dldg` reduction (upstream hardcodes two warps);
- `nanov41/engram_tl.py` selects these nano kernels for 384/640 (no padding, `c=1`) and the
  untouched upstream kernels for native `hidden%256==0` (d16), with the power-of-two padded official
  path as a fallback when the nano package is absent.
Validated vs `engram_gate_ref`: d12 (384) grads ≤3.6e-4, d20 (640) ≤1.9e-4, d16 native ≤1.6e-4;
R69-tuned d12 graph: 64.65 ms baseline → 67.55 ms with nano TileLang (**+4.5 %**), 67.14 ms with the
Inductor gate (+3.9 %) — statistically tied and ~8 % faster than the 4x-padded official path
(73.2 ms).  d20 full graph (b=8) captures and runs.  Evidence
`runs/graph_engram_tl_nano_5090_2026-11-07.json`; the earlier padded-path study remains in
`runs/graph_engram_tlpad_5090_2026-11-07.json`.

**Step-2 tuning (same day).**  Profiling showed the gate kernels are only ~0.65 ms/graph (fwd+bwd,
2 layers, b=16×1024) and memory-bound; most of the +2.9 ms was outside the graph.  Four changes cut
the tuned d12 graph-step overhead to **+2.56 %** (63.69 → 65.32 ms median, 3 interleaved rounds):
1. reuse the kernel's own `x + gate*value` output on the native path (no extra addcmul kernel);
2. fused sparse table Adam into one Triton kernel per table (`nanov41/sparse_adam_triton.py`);
3. fused the deterministic segment-sum + Adam for the single-record graph path into one program-per-row
   kernel (no intermediate `[rows,32]` grad tensor);
4. skip param/leaf grad zeroing when `accum==1`: the R69 assignment-based capture overwrites the baked
   grad buffers on replay (R68 chunked accumulation keeps zeroing).
Validated: sparse kernels vs torch (weight ≤4e-6, moments ≤1e-5), 5-step graph run stable, capture
PASS.  Evidence `runs/graph_engram_tl_nano_opt_5090_2026-11-07.json`.

## 5. Risks

- **Benefit is unproven at nano scale**: V4.1's 196 B-param Engram is a knowledge-capacity axis; at
  193 M total params a 0.1–0.5 B-param table may or may not move held-out bpb/teacher-match. P2 is
  the capacity gate; the module itself stays in the architecture regardless.
- **Tokenizer dependency**: the whole data pipeline already needs the DeepSeek tokenizer; changing
  the tokenizer later re-derives the compressed map and every hash multiplier.
- **Determinism/parity**: sparse embedding gradients are atomic by default; because Engram is
  mandatory, the repo's 0.0 eager-vs-graph parity gate requires the deterministic segment-sum
  implementation — an opt-in parity exception is no longer acceptable.
- **Graph/chunk complexity**: hash ids must be static graph inputs; the 1M chunked path needs the
  hash cache carried across chunk replays.
- **Training recipe for fp8 tables**: `ref/engram.py` is an inference implementation (fp8 rows +
  scales); start training experiments with bf16 tables, revisit fp8 only after the sparse optimizer
  is stable.
