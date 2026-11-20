# Decode 200 tok/s: bottleneck analysis, measured ceiling, and the fast path there

Status: analysis complete 2026-11-08.  **M1** (graph-safe runtime, ~310 tok/s), **M2**
(attention/MoE/mHC/head/elementwise GPU-busy work, 487.5 tok/s verified), **M2d** (fused MoE +
RMSNorm + head, 790 tok/s verified / 812 ceiling) and **M3** (continuous batching, one graph per
batch size, **8.8 k tok/s aggregate at B=16** with bit-identical per-sequence output) are done.
Goal: **200 tok/s single-stream decode** on RTX 4090, d12 (193.9 M total, 19.8 M active) — met
with 3.9× headroom single-stream and 44× aggregate at B=16.

## 1. Where we are

Real P2b checkpoint, 4090, batch 1, bf16, ctx 512, native stack (FP8 Engram + small-M MoE +
fused mHC + decode attention fast path):

| metric | value |
|---|---:|
| end-to-end eager decode | **23.28 tok/s** (42.95 ms/step) |
| GPU busy per step (kernel sum) | **3.10 ms** |
| CUDA kernel launches per step | **1019** |
| CPU dispatch/dispatch-gap per step | **≈ 49 ms (94 % of the step!)** |

The step is not compute bound and not even GPU bound: it is **CPU launch-bound**. That is why
kernel-level work stopped moving the end-to-end number much: the fused kernels reduced launches
1730 → 1019 and wall time 48.7 → 42.6 ms, but every remaining op still pays Python/ATen dispatch.

## 2. Theoretical bounds

**Bandwidth roofline (batch 1).** Per token the decode reads:
- head weight: 129,280 × 384 fp32 = **198.5 MB** (the `ParallelHead` keeps fp32 by design);
- active non-embed bf16 weights: 19.8 M × 2 B = **39.4 MB** (routed top-2 experts, shared expert,
  attention/compressor/mHC);
- Engram tables are not streamed: 2 layers × 4 rows × 32 values ≈ 1 KB.

Total ≈ **238 MB/token**. 4090 HBM ≈ 1008 GB/s peak, ~85–90 % achievable → **0.26–0.28 ms/token**
→ **~3600 tok/s** bandwidth floor. With a bf16 head it would be 139 MB → ~6500 tok/s floor.
**Compute is irrelevant**: 138.9 MFLOP/token at ~160 TFLOP/s ≈ 0.9 µs.

**Kernel-count floor.** A decode step is a chain of ~1 k tiny kernels; even inside a CUDA graph
each costs ~1–3 µs of execution/launch-gap on this GPU. So with the *current* op count the
practical floor is ~1–3 ms/step. Measured below: 3.08 ms.

**Conclusion:** the 200 tok/s target needs **≤ 5 ms/step**. That is 60 % *above* the measured
graph ceiling with the current kernels. The target is reachable without any new kernel; the work
is removing the CPU dispatch cost.

## 3. Measured ceiling: CUDA graph replay of the current step

`tools/graph_ceiling_test.py` captures one decode step at a *fixed* `start_pos` (timing-only
experiment; position-dependent Python is frozen at capture) and times graph replay. Raw:
`runs/p2b/graph_ceiling*.json`.

| config | eager | graph replay | graph tok/s | speedup |
|---|---:|---:|---:|---:|
| ctx 64, native+attn | 53.21 ms | **3.083 ms** | **324.3** | 17.3× |
| ctx 256, native+attn | 56.17 ms | **3.022 ms** | **330.9** | 18.6× |
| ctx 512, native+attn | 58.54 ms | **3.084 ms** | **324.3** | 19.0× |
| ctx 1008, native+attn | 52.73 ms | **3.047 ms** | **328.2** | 17.3× |
| ctx 512, no attn fast path | 76.84 ms | 3.215 ms | 311.0 | 23.9× |

Graph replay ≈ kernel busy time (3.10 ms): CUDA graphs remove the CPU dispatch and nearly all
inter-kernel gaps. **Ceiling with today's kernels: ~324–331 tok/s**, flat across 64–1008 context
(the fused decode attention more than pays for its kernels: −0.13 ms GPU vs the reference path).

## 4. GPU busy breakdown (3.10 ms/step, graph replay)

| category | µs/step | share | notes |
|---|---:|---:|---|
| elementwise + copies | 944 | 30 % | indexer/compressor, MoE gathers, mHC residual, attn out-proj |
| gemm (gemvx) | 653 | 21 % | **head fp32 364 µs** + attention/MoE projections |
| mHC fusions | 650 | 21 % | `_pm_pre_fwd` 438 µs, `_split_sinkhorn_fwd` 124 µs, `_mixes_pre_fwd` 88 µs |
| decode attention kernels | 622 | 20 % | `sparse_attn_decode` 565 µs (only 8 programs — under-parallel) |
| reductions | 89 | 3 % | |
| MoE small-M | 75 | 2 % | |
| other | 72 | 2 % | |

After the graph removes the CPU cost, *this* becomes the only thing that matters: every µs of GPU
busy is ~0.32 tok/s at 3.1 ms. 200 tok/s only requires 5 ms; the path to 400–600 tok/s is
shrinking this 3.1 ms.

## 5. Fast path to ≥ 200 tok/s (and beyond)

### Step A — graph-safe decode runtime (the fast move; no new kernel needed)

What has to change for a real (dynamic `start_pos`) decode graph:

1. **Position as a tensor.** Replace every Python-int `start_pos` in the decode path with a static
   `pos` buffer + tensor ops: RoPE row via `freqs[pos]`, window-ring slot via `index_copy_`
   (tensor index), compressor state slot / group index via `index_copy_`, Engram hash cache
   gathers with position tensor.
2. **Static shapes.** Pad the compressed-position count to the fixed `index_topk` (32) with -1 and
   mask; read the full compressed/index caches (512 or 1024 rows) and mask scores by
   `arange(max) < compress_len`; `topk`/`sort` on fixed sizes; no dynamic `unflatten`/slicing.
3. **Unconditional compressor writes.** The reference writes the in-progress group only on group
   completion. Writing every step is safe: an in-progress index is never read
   (`compress_len = (pos+1)//ratio` excludes it) and is overwritten when the group completes.
4. **Cross-layer shared state as buffers.** `topk_idxs` / compressed KV / index key currently get
   *rebound* to new tensors by the source layer; for graph capture they must be persistent buffers
   written with `copy_`/`index_copy_`.
5. **Capture once per sequence.** Prefill eagerly → capture the step graph → reset caches →
   pre-fill again → loop `token.copy_(); pos.fill_(); replay(); argmax(logits)`. Argmax/sampling
   stays outside the graph (~10–20 µs/step); one graph per sequence (paged KV / batching later
   needs per-step graphs, standard practice).

Expected: **300+ tok/s immediately** (measured ceiling 324–331), 200 target exceeded with margin.
Risk is concentrated in (1)/(2): every data-dependent shape or Python branch must be padded or
made tensor-driven; the repo already has a pure-tensor decode fast path for attention
(`nanov41/attn_decode.py`) to build on. Validation: logit parity vs eager over ~100 steps at ctx
64/512/1008 + greedy token agreement.

### Step B — shrink GPU busy for 400–600 tok/s

Ordered by measured payoff (µs/step saved per hour of work):

| change | now | target | saves |
|---|---:|---:|---:|
| parallelize/vectorize `sparse_attn_decode` (grid is 8 programs; use 2-D tile scores) | 565 | ~80 | ~0.5 ms |
| T≤8 mHC kernel (`_pm_pre` + sinkhorn + mixes as one/few kernels) | 650 | ~250 | ~0.4 ms |
| fuse elementwise/copies (attn out-proj, indexer/compressor, MoE gathers) | 944 | ~400 | ~0.5 ms |
| bf16 (or FP8) head copy for inference | 364 | 190 / 100 | ~0.2–0.25 ms |
| MoE small-M single kernel | 75 | ~50 | small |

That is ~3.1 ms → ~1.5 ms → **~650 tok/s** ceiling; expect 400–500 tok/s with sampling/loop
overheads. The bf16 head needs an eval check (T0 BPB) because it rounds the logits.

### Step C — serving

Paged KV + continuous batching + per-step padded graphs turn the single-stream number into
aggregate throughput; not needed for the 200 tok/s single-stream goal.

## 6. Milestones

| milestone | work | expected |
|---|---|---|
| M1 (fast) | graph-safe decode runtime + validation | **300+ tok/s**, 200 target met |
| M2 | sparse-attn vectorization, T≤8 mHC, elementwise fusion, bf16 head | 400–600 tok/s |
| M3 | continuous batching / one graph per batch size (slot rows instead of page tables) | **done**: 8.8 k tok/s @ B=16, bit-identical tokens |

Also fixed on the way: `enable_moe_decode` routed prefill through the small-M path because it
checked `x.shape[0]` (batch) instead of the token count — a 1008-token prefill allocated a
568 MB expert gather and OOM'd at ctx 1008. Now checks `numel()//dim`.

## 7. M1 result: graph-safe decode runtime — done (2026-11-08)

`nanov41/decode_graph.py` (`enable_graph_decode` → `GraphDecoder`) implements the plan above:
tensor `pos` buffer, padded static shapes (index-topk fixed 32 with -1 masking, full compressed/
index caches masked by position, fixed `topk`/`sort`), persistent group buffers for the shared
compressed KV / top-k idxs, unconditional compressor writes, and the `attn_decode` Triton kernels
(with the sparse-attention kernel now reading the position from a device tensor so replay is
position-generic).  Prefill stays eager; the step graph is captured once after prefill, then the
caches are re-prefilled and the replay loop serves generation (`argmax` outside the graph).

Validation on the real P2b checkpoint (`tools/verify_decode_graph.py`,
`runs/p2b/decode_graph_verify.json`, RTX 4090, bf16, 64 greedy tokens unless the context cap
applies):

| check | result |
|---|---|
| Engram hash ids (tensor-pos rewrite vs `NgramHashState.forward`) | **exact, 0 mismatches** |
| eager native stack (int `start_pos`) vs tensor-pos graph forward | **logit max abs diff 0, argmax 100 %** |
| captured graph replay vs the same forward run eagerly | **logit max abs diff 0** |
| greedy token agreement (graph vs eager, all contexts) | **100 %** |

Steady-state generation loop (prepare/capture excluded, argmax sync included):

| context | graph tok/s | ms/token | eager tok/s | speedup |
|---|---:|---:|---:|---:|
| 64 | **308.1** | 3.25 | 22.3 | 13.8× |
| 256 | **314.1** | 3.18 | 22.2 | 14.1× |
| 512 | **305.9** | 3.27 | 23.2 | 13.2× |
| 1008 | **311.5** | 3.21 | 21.4 | 14.6× |

So the 200 tok/s target is met with ~50 % headroom, and the parity route is exact (the graph is
bit-identical to the eager native stack on these runs, not just bf16-close).  Constraints of this
first runtime: single sequence, `prompt + generated <= max_seq_len` (1024, no paged KV yet), one
graph capture per prompt (~0.3 s, amortized), and prefill stays eager (no graph for prefill).

GPU-regression test: `tests/test_decode_graph.py` (gated by `NANOV41_SLOW_TESTS=1` on CUDA).

### Next: M2 (400–600 tok/s) and M3 (serving)

M2 is now purely GPU-busy reduction (graph replay ≈ kernel busy 3.1 ms): vectorized sparse-attn
(565 → ~80 µs), T≤8 mHC fused kernel (650 → ~250 µs), elementwise/copy fusion (944 → ~400 µs),
bf16/FP8 head (364 → 190/100 µs).  M3 is paged KV + continuous batching + per-step graphs.

## 8. M2 progress: 433 tok/s ceiling / ~397 tok/s verified (2026-11-08)

M2 is GPU-busy reduction only (graph replay ≈ kernel busy).  Two changes landed:

1. **Vectorised decode sparse-attention kernel** (`attn_decode.sparse_attn_decode_kernel` v2):
   the old kernel walked the 64 window + 32 compressed slots one at a time with a scalar dot
   (only 8 programs, ~96 serial iterations); v2 loads `BT=16` KV rows as 2-D tiles and reduces
   over D, carrying online-softmax state over the (static) window/compressed chunk loops.  Output
   is identical to the torch mirror at every tested position (bf16-exact), captures fine, and
   cuts the attention kernel group **622 → 118 µs/step**.
2. **bf16 inference head** (`nanov41/head_decode.py`): a bf16 copy of the fp32 198.5 MB
   `ParallelHead.weight` halves the largest streamed tensor.  Quality: greedy generation
   **100 % agreement** over 128 tokens (ctx 512) and T0 BPB **1.922626 vs 1.922626** (delta 0,
   byte-weighted 2.014134 identical).  Head GEMV **365 → 228 µs**, total gemm group 654 → 398 µs,
   graph loop 381 → 422 tok/s in the same measurement.

Combined effect (real P2b ckpt, 4090, graph replay + generation loop):

| stage | ms/step | graph tok/s | verified greedy loop |
|---|---:|---:|---:|
| M1 (graph runtime, pre-M2 kernels) | 3.08 | 324 | 306–314 |
| **M2a + M2b (attn v2 + bf16 head)** | **2.31** | **433** | **396–397** (ctx 64/512) |
| M2 target (mHC + elementwise fusion) | ~1.45 | ~690 | ~550–600 |

Parity is unchanged: eager-native vs graph-forward logit max abs diff **0**, capture parity **0**,
greedy agreement **100 %** at ctx 64/512 (`runs/p2b/decode_graph_verify_m2.json`).

Remaining GPU busy after M2a/b (2.35 ms/step): elementwise+copies **954 µs** (41 %), mHC
**648 µs** (28 %), gemm **398 µs** (17 %, head already bf16), attention kernels **118 µs** (5 %),
reductions 89 µs, small-M MoE 75 µs, other 72 µs.  An mHC launch-config sweep (`W_PMPF` 1/2, BK
variants, the v2 kernel path) gave no gain — the 437 µs `_pm_pre_fwd` is structure-bound at M=1
(BM=16 minimum for `tl.dot`), so the next item is a dedicated T≤8 mHC kernel (target ~250 µs),
then an elementwise/copy audit (indexer/compressor/attention-output/MoE gathers) targeting ~400 µs.

## 9. M2c: decode-specialised mHC kernels — 648 → 226 us (2026-11-08)

The fused-v3 mHC kernels are tiled for training (mixes / sinkhorn / post as separate launches, 16
rows minimum for `tl.dot`), so at T=1 they cost ~648 us/step.  `nanov41/mhc_decode_fast.py` replaces
the per-block chain for a single decode token with three single-program kernels:

  A. mixes + split-sinkhorn (attn cell) + hc_pre                    -> attn input
  B. hc_post(attn) + mixes + split-sinkhorn (ffn cell) + hc_pre     -> residual y, ffn sinkhorn, ffn input
  C. hc_post(ffn)                                                   -> block output

They mirror fused-v3 arithmetic (bf16 `fn` dot, fp32 statistics, same softmax/sinkhorn iteration
order and epsilon placement) and are inference-only: `enable_mhc_decode_fast` wraps
`Block.forward`, taking the fast path only for a single token (prefill keeps the validated fused
path).  `enable_graph_decode(..., mhc_fast=True)` is the default.

Validation (`tools/verify_mhc_decode.py`, `runs/p2b/mhc_decode_fast_verify.json`):

| check | result |
|---|---|
| block output y (v3 vs fast, real weights, layers 0/6, 16 steps) | max abs 1.0 on bf16 values (rounding), ffn pre_mix max abs 1.5e-4 |
| per-step logits, same forced prefixes, 16 steps | max 0.180, mean 0.0119 — the same bf16 class as fused-v3-vs-torch (0.0126) |
| greedy token agreement | **100 %** |
| graph decode parity (eager-native vs graph-forward, ctx 64/512) | logit max abs **0**, capture parity **0**, greedy **100 %** |

Effect on the graph step (`runs/p2b/graph_ceiling_m2c.json`, ctx 512):

| | M2a+b | **M2c** |
|---|---:|---:|
| graph replay | 2.31 ms | **1.86 ms** |
| graph ceiling | 433 tok/s | **537.6 tok/s** |
| verified greedy loop | 396–397 tok/s | **487.5 / 487.6 tok/s** (ctx 64/512) |
| mHC GPU time | 648 us | **226 us** |

Cumulative: eager 23 → **488 tok/s verified (21×)**, 538 ceiling; vs the 200 tok/s target 2.4×.

Remaining GPU busy (1.86 ms): elementwise+copies **897 us (48 %)**, gemm 383 us, mHC 226,
attention kernels 110, reductions 85, small-M MoE 72, other 58.  The next M2 step (M2d) is the
elementwise/copy audit, which turned into "replace whole module chains with fused kernels" after
the attribution below — see §10.

## 10. M2d: fused MoE / RMSNorm / head — 1.86 → 1.23 ms (812 tok/s ceiling, 790 verified)

**Attribution, done properly.**  `tools/profile_ranges_decode.py` was rewritten: every CUDA kernel
event's `external_id` equals the `external_id` of its launching CPU op (verified on this torch), and
that op has a `cpu_parent` chain containing the `record_function` module ranges — so kernels can be
attributed exactly instead of by name guessing.  On the eager step (same kernels as the graph) the
biggest owner was not "elementwise copies" but the **small-M MoE at ~870 us/step (38 %)**, then mHC
279, attn 430 (inflated ~16 % by CUPTI), block 295; the "block" bucket was almost all block
RMSNorms and the top launches were `aten::copy_` 362 us / 256 calls, `aten::bmm` 108 / 50,
`aten::topk` 97 / 26, `aten::index_select` 94 / 37, `aten::clamp` 58 / 48.

Three fusions:

| module | before | after | implementation |
|---|---:|---:|---|
| small-M MoE (12 layers) | ~45 torch kernels/layer (gate casts+GEMV+softplus+`topk(16)`+gather, 3×`index_select`+3×`bmm` routed, 3×`bmm` shared, clamp/silu/cast, `index_add_`) | **4 Triton kernels/layer** | `nanov41/moe_decode_fast.py`: `moe_gate_kernel` (16 fp32 logits → sqrt-softplus → bias → top-2 → normalise/route-scale), `moe_gu_kernel` (per expert slot × inter chunk; expert offset resolved in-kernel, no weight gather materialisation; same bf16 rounding chain as the torch path), `moe_down_partial_kernel` (K-split) + `moe_down_combine_kernel` (reproduces `bf16(bf16(d0+d1)+shared)`) |
| block RMSNorm (24 calls/step) + final/compressor norms | ~7 ATen kernels each (~150–200 us) | **1 kernel each** | `nanov41/rmsnorm_decode.py`: one program per row, fp32 statistics, one rounding; installed on the `RMSNorm` class for decode-sized tensors (prefill takes the reference path) |
| bf16 output head (fp32→bf16 copy, M2b) | cuBLAS GEMV 228 us (99 MB at 0.44 TB/s) | **`head_gemv_kernel` 109 us** (0.91 TB/s) | `nanov41/head_decode.py`: vocab-tiled GEMV, `F.linear`'s bf16 output rounding kept; decode path only |

Validation (`tools/verify_moe_decode_fast.py`, `runs/p2b/moe_decode_fast_verify.json`): routing
indices identical for all 12 layers × 1/2/4 rows, fp32 routing-weight diff ≤ 1.2e-7, MoE output
diff is exactly bf16-rounding class (max 0.5–1.0 absolute on values up to ~100).  The full graph
verify (`runs/p2b/decode_graph_verify_m2d.json`) gives **logit diff 0**, capture parity **0**,
greedy **100 %**, and the generated text is **identical to the M2c run** at ctx 64/512.

| | M2c | **M2d** |
|---|---:|---:|
| graph replay | 1.86 ms | **1.231 ms** |
| graph ceiling | 537.6 tok/s | **812.4 tok/s** |
| verified greedy loop | 487.5 tok/s | **781.9 / 790.0 tok/s** (ctx 64/512) |
| kernel busy / replay gap | 1.83 ms / 0.03 ms | **1.169 ms / 0.06 ms** |

Kernel budget after M2d (1.169 ms): mHC 269, MoE 253 (gate 84 / gu 108 / down 37+combine),
gemm 230 (head 109 + attention projections 115), elementwise+copies 215, attention kernels 149,
other 38, reduce 16.  One profiling rounding caveat: after the M3 batch-invariant patch (§11) the
same single-stream step measures **1.315 ms / 760 tok/s** ceiling and 715–780 tok/s verified at
ctx 512/64 — the wo_a contraction moved from a GEMM to an explicit fp32 multiply+reduce (elementwise
+77 us, reduce +19 us) and the run-to-run GPU clock spread is ~5–7 %.

## 11. M3: continuous batching + one graph per batch size — 8.8 k tok/s aggregate

**Why slot rows instead of paged KV.**  Paging exists to make the KV working set fit or to share
prefixes.  Here the model context is bounded at `max_seq_len = 1024` and the checkpoint's caches are
already allocated `[max_batch_size = 16, max_seq_len, ...]` by the reference implementation, so a
request can own one batch row for its whole life: a "slot" is a page of `max_seq_len` tokens, and
there is nothing to evict.  The batch runtime therefore uses **slot rows + one captured graph per
batch size**; the graph is replayed for the exact number of live rows (1..16).

What M3 adds:

* **batch-safe graph step** (`nanov41/decode_graph.py`): `token` [B,1] / `pos` [B], per-row cache
  scatter for window/compressed/indexer state, per-row top-k buffer, batch RoPE kernel, batched
  sparse-attention kernel (one program per (sequence, head)), and batched mHC kernels (one program
  per row, with the two block RMSNorms fused into kernels A/B).  The MoE and head kernels were
  already token-batched.
* **`nanov41/batch_decode.py`**: `BatchDecoder` captures one graph per batch size at startup on
  static buffers, prefills each new request with the reference eager forward on a **scratch cache
  row** (the reference prefill always writes batch row 0, so the live cache tensors are swapped for
  batch-1 scratch rows and the row is copied into its slot afterwards), advances the batch with
  GPU-only ops when lengths are known, and **compacts slots** (copy the last live row into a freed
  row) as requests finish.
* **batch-invariant numerics** (the subtle part).  cuBLAS dispatches a GEMV for M=1 and a GEMM for
  M≥2, and `torch.einsum` picks different bmm algorithms for batch 1 vs batch >1; the difference is
  ~1 fp32 ulp before the bf16 rounding, but 12 layers of bf16 rounding amplify it — batched decode
  drifted from the single-sequence runtime and eventually flipped tokens (measured: first logit
  diff at decode step 4, token divergence at step 9, at B as low as 2).  Fixed by making every
  decode projection batch-invariant:
  - `nanov41/linear_decode.py`: `linear_decode_kernel`, a `BLOCK_M=16`-tiled projection that
    reduces over K in fixed `BLOCK_K` chunks, so a row's result does not depend on how many rows
    are in the launch; installed on the reference module's `linear()` helper (all `Linear`s) and
    used by the Engram decode gate.  Prefill (>16 rows) keeps cuBLAS.
  - `attn_decode.blockdiag_wo_a`: the block-diagonal `wo_a` projection as an explicit fp32
    multiply+reduce (also replaces the einsum in the eager decode attention path so eager and graph
    stay bit-identical).
  With those, **batched decode is bit-identical to the single-sequence runtime** at every batch size
  tested (1..16), not merely "close".

Results (real P2b checkpoint, 4090, ctx 512, 32 greedy tokens per request;
`runs/p2b/batch_decode_m3.json`):

| batch | ms/step | aggregate tok/s | per-seq tok/s | greedy agreement vs single |
|---:|---:|---:|---:|---:|
| 1 | 1.409 | 732.9 | 732.9 | 100 % |
| 2 | 1.439 | 1434.6 | 717.3 | 100 % |
| 4 | 1.458 | 2832.9 | 708.2 | 100 % |
| 8 | 1.592 | 5188.5 | 648.6 | 100 % |
| **16** | **1.882** | **8775.8** | 548.5 | **100 %** |

Continuous serving with arrivals/departures and slot compaction (24 requests, prompt lengths
48–320, `max_new` 8–24, dynamic prefill into freed slots;
`tools/bench_continuous_decode.py` → `runs/p2b/continuous_decode_m3.json`): 31 steps, active/step
avg 10.3, **decode 5.85 k tok/s, all 24/24 requests token-identical to single-sequence greedy**.
The wall time of that run is prefill-dominated (1.87 s of eager prefill vs 0.06 s of graph decode):
batched/graphed prefill is the obvious next serving item.

Single-stream after M3: `tools/verify_decode_graph.py` → eager vs graph-forward logit diff **0**,
capture parity **0**, greedy **100 %**, generated text **identical to M2d**, **780.2 / 715.7 tok/s**
at ctx 64/512 (`runs/p2b/decode_graph_verify_m3.json`).  B=1 through `BatchDecoder` (1.409 ms) is
the same step plus the on-device argmax/copy.

Tools added for M2d/M3: `tools/profile_ranges_decode.py` (fixed attribution),
`tools/verify_moe_decode_fast.py`, `tools/bench_batch_decode.py`,
`tools/bench_continuous_decode.py`, plus the `dbg_batch_*` debug helpers used to localise the
batch-vs-single drift (per-module diff, per-operand diff, cache-row diff).
