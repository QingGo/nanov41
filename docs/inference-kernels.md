# Native inference kernels

Project goal: **efficient inference**, not only training.  This is the native path (no vLLM
production kernels); the fallback vLLM model in `nanov41/vllm_plugin.py` exists only as a
correctness/reference path.

## First kernel: Engram decode (2026-11-07)

Implemented in [`nanov41/engram_decode.py`](../nanov41/engram_decode.py):

- `EngramDecodeCache` / `decode_hash_ids`: incremental n-gram hash for the current token(s),
  same formula as `ref/engram.py::NgramHashState` (start/dead masking, lookback from a tiny
  compressed-id cache);
- `quantize_table_ue8m0`: bf16 table -> e4m3 rows + one E8M0 byte per 32-wide block (official/vLLM
  layout);
- `gather_dequant`: one Triton program per 64 rows — FP8 row gather + E8M0 dequant to bf16;
- `engram_gate_decode`: gather -> wkv -> q/k gate -> residual add (gate math still PyTorch; wkv/gate
  fusion is the next step).

Measured on RTX 4090 (torch 2.12.1 / triton 3.7.1):

| ids/step | Triton gather+dequant | torch fallback | parity |
|---:|---:|---:|---|
| 8 | 30.1 µs | 99.8 µs | maxdiff 0 |
| 64 | 28.8 µs | 99.7 µs | maxdiff 0 |
| 512 | 29.1 µs | 100.4 µs | maxdiff 0 |

Hash decode matches both an independent torch reference and `NgramHashState.forward`
(`tests/test_engram_decode.py`).  Raw: `runs/p2b/engram_decode_benchmark.json`.

## Roadmap / priorities

1. **Engram decode** — done at kernel level (hash + FP8 gather/dequant); next fold wkv + gate into
   the kernel or a fused epilogue.
2. **MoE small-M decode** — training grouped GEMM is tuned for prefill; add M=1–8 paths
   (routing/permutation, shared expert, combine).
3. **mHC decode** — small-M fused pre/post + norm (training kernels assume larger token counts).
4. **Decode attention** — last; paged KV + split-K; can borrow ideas from vLLM design docs but not
   its production kernels.
5. **Fused-head NLL/eval** — already available in the eval harness (`chunked` exact, `fused` FP8).

Guardrails: every kernel needs a reference parity test, a microbench entry, and must not regress the
training graph; keep decode kernels separate from `*_train` implementations.

## Second path: MoE small-M decode (2026-11-07)

`nanov41/moe_decode.py` adds a decode-oriented MoE forward for M=1--8:

- stack routed expert weights once (`enable_moe_decode`);
- for each (token, slot): gather the selected W1/W3/W2, run three small batched GEMMs
  (`bmm`), apply SwiGLU clamps + routing weight, `index_add_` into the token output, add shared
  expert;
- fallback to the original MoE forward above `max_tokens`.

RTX 5090, d12 MoE layer (dim=384, inter=384, E=16, top-2):

| tokens | reference loop | small-M | speedup | parity |
|---:|---:|---:|---:|---:|
| 1 | 852.9 µs | 455.8 µs | 1.87× | 9.77e-4 (bf16) |
| 2 | 1092.1 µs | 450.8 µs | 2.42× | 9.77e-4 |
| 4 | 1952.2 µs | 460.2 µs | 4.24× | 9.77e-4 |
| 8 | 2911.2 µs | 463.8 µs | 6.28× | 9.77e-4 |

Full-model CPU float32 integration test vs the reference MoE forward passes with maxdiff 4.8e-6
(`tests/test_decode_integration.py`).  Raw: `runs/p2b/moe_decode_benchmark.json`.

This is still PyTorch batched GEMM, not a hand-written Triton kernel; it removes the per-expert
launch storm and is the first native decode MoE path.  A dedicated Triton small-M kernel can
replace the `bmm` calls behind the same interface later.

One-call integration: `nanov41.native_decode.enable_native_decode(model)` patches both the FP8
Engram path and the small-M MoE path on an inference copy; `nanov41.generate.generate_kv` then runs
the native decode forward.  What is still missing for production decode: mHC small-M, decode
attention (paged KV/split-K), paged/continuous batching, CUDA graph and an end-to-end tokens/s
benchmark.

Next: mHC decode; then decode attention (paged KV/split-K) and the paged/continuous-batching
runtime.

## End-to-end native decode benchmark (2026-11-07)

Random d12 model with the same shapes (12 layers, 2 Engram layers, 16 experts top-2), RTX 4090,
batch 1, 16 greedy decode tokens; reference = torch path, native = FP8 Engram + small-M MoE,
native+mHC = additionally the R69 fused mHC kernels:

| path | tok/s | ms/token | speedup |
|---|---:|---:|---:|
| reference | 12.81 | 78.05 | 1.0× |
| native Engram+MoE | 14.09 | 70.99 | 1.10× |
| + fused mHC | **20.88** | **47.88** | **1.63×** (1.48× over native) |

Caveat: fused mHC changes accumulation order; on the random model logits differ max 0.27
(mean 0.0093), so it is opt-in (`enable_native_decode(..., mhc=True)`) until validated against a
real checkpoint.  This is a shapes-representative benchmark, not a real-capability tok/s claim;
paged KV/continuous batching and decode CUDA graph are still missing.  Raw:
`runs/p2b/native_decode_e2e_benchmark.json`.

## Real-checkpoint decode benchmark + native path decisions (2026-11-07)

Checkpoint: P2b `E-default_s1235` (`/tmp/p2b_default.pt`, 745 MB, 193.91 M params, 12 layers,
Engram layers [1,4], 16 experts top-2) loaded in bf16; RTX 4090 24 GB, torch 2.12.1+cu130 /
triton 3.7.1.  Batch 1, 5 prompts (English / code / Chinese), 32 greedy tokens, official KV-cache
path.  Tools: `tools/bench_real_decode.py` (throughput + parity + T0 BPB + component profile),
`tools/bench_real_decode_extra.py` (mHC / FP8-table isolation), `tools/profile_real_decode.py`
(kernel-level profile).  Raw: `runs/p2b/real_decode_benchmark.json`,
`runs/p2b/real_decode_mhc_extra.json`, `runs/p2b/real_decode_profile.json`.

### Decode throughput (real weights)

| path | tok/s | ms/token | speedup vs reference | prefill (ms) |
|---|---:|---:|---:|---:|
| reference (torch fallbacks, torch mHC) | 12.64 | 79.10 | 1.00× | 95.9 |
| native Engram+MoE (FP8 tables + small-M MoE) | 13.26 | 75.40 | 1.05× | 79.1 |
| reference + fused mHC | 18.12 | 55.20 | 1.43× | 61.6 |
| **native + fused mHC** | **19.34** | **51.69** | **1.53×** (1.46× over native) | **53.2** |

A second process measured ref 11.49 and native+mHC 17.3–17.5 tok/s, i.e. ~5–10 % run-to-run spread
on a shared 4090.  These are single-request, real-weight numbers, not serving throughput; paged KV,
continuous batching and a decode CUDA graph are still missing.

### What the native path changes numerically (quality)

| comparison | token agreement (160 greedy tokens) | prefix-aligned logit delta | T0 macro BPB |
|---|---:|---:|---:|
| native vs reference | 65.0 % | mean 0.0175, max 1.65 | — |
| reference + FP8 Engram tables (torch path only) | 81.9 % | mean 0.0178, max 1.64 | 1.922632 (−6e-6) |
| fused mHC v3 vs reference | **100 %** | mean 0.0126, max 1.08 | 1.922632 (−6e-6) |
| fused mHC v4 vs v3 | 100 % | **0.0 (bit-identical)** | — |
| native + fused mHC vs native | 100 % | mean 0.0109, max 0.13 | — |

- **FP8 Engram tables are quality-free here**: e4m3 rows + one E8M0 byte per 32-wide block
  (mean relative table error 2.25 %, p99 5.88 %); T0 macro BPB 1.922638 → 1.922632 and byte-weighted
  2.014150 → 2.014145.  The 65 % greedy-token agreement is small margin flipping in a 0.6 B-token
  model (isolating the tables on the torch path gives 81.9 %; the rest is the small-M MoE bf16
  accumulation order) — it is **not** a BPB regression.
- **Fused mHC is numerically free at the metric level**: 100 % token agreement on all 160 tokens,
  ΔBPB −6e-6, and the training block chain (v4) is bit-identical to v3 at T=1.  It is now the
  default in `enable_native_decode` (`mhc=True`); pass `mhc=False` for the pristine torch path.
- The reference eval reproduced the published T0 exactly (1.922638 vs 1.922638), so throughput and
  BPB numbers above are on the same harness as the training results.

### Where a decode step goes, and why the bottleneck is *not* FLOPs

Component profile (`torch.cuda.Event` module hooks) at context 512, native+fused mHC:

| component | ms/step | share |
|---|---:|---:|
| attention (torch fallback: window + compressor/indexer/sparse attention) | 22.50 | 41.6 % |
| mHC + norms + residual (fused v3) | 16.51 | 30.6 % |
| MoE small-M (batched `bmm` path) | 11.44 | 21.2 % |
| Engram FP8 gather + gate | 1.25 | 2.3 % |
| head | 0.42 | 0.8 % |
| Python / other | 1.74 | 3.2 % |

Kernel-level profile of the same path (`runs/p2b/real_decode_profile.json`): **1730 CUDA kernel
launches per decode step but only 3.86 ms/step of GPU busy time** (wall ≈ 52–54 ms).  The decode
step is **CPU launch/dispatch-bound**; the mHC kernels are ~0.7 ms/step of GPU-busy time
(`_pm_pre_fwd_kernel` 478 µs, `_split_sinkhorn_fwd_kernel` 135 µs, `_mixes_pre_fwd_kernel` 96 µs),
so the 16.5 ms mHC figure is mostly per-op overhead, not mHC math.

Consequences for the roadmap:

1. **Launch-count reduction is the lever.**  By measured CPU share: decode attention (~42 %) →
   mHC T≤8 fusion (~31 %) → MoE small-M fusion (~21 %) → Engram (~2 %).
2. A dedicated T≤8 mHC kernel is still justified (it would cut ~5 launches/block plus the norms
   around it), but it is not accuracy-driven: this is now a pure launch-overhead play.
3. A decode CUDA graph would collapse all 1730 launches, but `start_pos` is currently a Python int
   feeding per-position `arange`/topk/index tables inside the reference attention and hash code.
   It must first become a graph-safe (tensor / precomputed-table) input — a runtime refactor, not a
   kernel; do it after the kernels or as its own workstream.
4. paged KV / continuous batching remain after that; they change the runtime, not the numerics.

### Decisions (2026-11-07)

| decision | outcome | evidence |
|---|---|---|
| Can fused mHC be the default decode path? | **yes** (`mhc=True` in `enable_native_decode`) | 100 % token agreement, ΔT0 BPB −6e-6, 1.43–1.53× tok/s |
| Is a T≤8 mHC kernel needed for accuracy? | **no** — accuracy is already fine | v3 100 % agreement, v4 ≡ v3 |
| Is it needed for throughput? | **yes, but second**: ~31 % of CPU-bound step time, 5 training-tuned launches/block | component + kernel profile |
| Next kernel | decode attention (largest share, launch-heavy torch fallback) | 41.6 % of step |
| Then | MoE small-M fusion, T≤8 mHC fusion | 21 % / 31 % |
| Then | graph-safe position/runtime refactor + decode CUDA graph | 1730 launches → 1 |

## Decode attention fast path (2026-11-07) — roadmap step ①

The kernel-level profile above showed the decode step is CPU launch-bound and attention is the
largest component (~42 % of a native step).  `nanov41/attn_decode.py` keeps the reference math but
collapses the elementwise/reduction tail of `Attention.forward` into three Triton kernels:

| kernel | replaces | notes |
|---|---|---|
| `rmsnorm_kernel` | `RMSNorm` (q_norm / kv_norm): ~4 torch ops each | float32 stats, dtype cast on store |
| `rope_kernel` | `apply_rotary_emb` (~8 ops), forward and inverse | strided last-dim slice, in place; bit-exact fp32 complex multiply |
| `sparse_attn_decode_kernel` | window ring gather + sparse softmax + sink + weighted sum + window-index generation (~30 ops) | one program per head; window slots derived arithmetically (`oldest = p % win + 1`, valid iff `slot <= p`), compressed slots come from the existing indexer with the reference `+win` offset; 2-pass max/denominator inside one kernel |

The `Linear` projections, the `Compressor` and the `Indexer` are reused unchanged, so all GEMMs
(the numerically dominant part) stay on the exact reference cublas path.  `enable_attn_decode`
patches `Attention.forward` on an inference copy; prefill, `start_pos == 0` and multi-token calls
fall back to the reference forward.  Torch mirrors of all three kernels are provided and tested
on CPU (`tests/test_attn_decode.py`), and `NANOV41_ATTN_DECODE=torch` forces the mirror path.

### Parity

- CPU torch mirror vs reference attention: sparse-attention output exact; whole TINY model decode
  logits max|d| 2e-5 over 15 positions (torch mirror path is bit-identical to the reference).
- GPU Triton path vs reference on the real P2b checkpoint (`runs/p2b/attn_decode_verify.json`,
  24 decode steps, prefix-aligned): **logit max|d| 0.071, mean|d| 0.0086, argmax agreement 100 %,
  greedy token agreement 100 %** (reference text reproduced exactly).
- Debug note: the first version double-negated sin in the inverse-RoPE epilogue (torch mirror
  conjugated *and* the kernel negated); fixed — the GPU stage debug now shows 0.0 diff for
  q / kv / sparse-attn stages.

### End-to-end (real P2b checkpoint, RTX 4090, batch 1, 24 greedy tokens x 3 prompts)

`tools/bench_e2e_incremental.py`, raw `runs/p2b/e2e_incremental.json`:

| config | tok/s | ms/token | speedup |
|---|---:|---:|---:|
| reference | 11.26 | 88.78 | 1.00× |
| native Engram+MoE+fused mHC | 19.46 | 51.39 | 1.73× |
| **native + decode attention** | **23.28** | **42.95** | **2.07×** (1.20× over native) |

CPU-wall component profile at ctx 512 (`tools/profile_cpu_decode.py`,
`runs/p2b/decode_cpu_profile.json`):

| config | wall | attn | small-M MoE | mHC+norm+resid | Engram |
|---|---:|---:|---:|---:|---:|
| reference | 78.61 | 20.62 | 15.49 | 39.44 | 1.16 |
| native | 48.75 | 20.80 | 10.26 | 14.54 | 1.16 |
| native + decode attention | **42.61** | **14.95 (−28 %)** | 10.37 | 14.07 | 1.17 |

CUDA launches per decode step drop 1730 → **1019** (−41 %; `runs/p2b/real_decode_profile_attn.json`).
The fused attention kernel itself is 0.59 ms/step of GPU busy time (49 µs/call, only 8 programs
per layer) — it is launch-count, not GPU-time, that this path is buying; the kernel can be
vectorized further if GPU time ever matters.

Remaining attention work (not done yet): the indexer/compressor of the two index-source layers
(~100 launches/step), fusing the q/kv projection + norm + RoPE chain into fewer launches, and the
output projection (`einsum` + `wo_b`); together these are why attention is still 15 ms of the
42.6 ms step.

Next roadmap step: **② MoE small-M fusion** (10.4 ms/step, ~21 %) then ③ T≤8 mHC fusion
(14.1 ms/step, ~33 %).

## Graph-safe decode runtime (M1, 2026-11-08)

The ceiling experiment above showed the decode step is CPU launch-bound (≈49 of 53 ms) and that a
CUDA graph of the *current* kernels replays in ~3.1 ms (324–331 tok/s).  `nanov41/decode_graph.py`
turns that timing-only experiment into a real runtime:

- **tensor position**: a persistent ``pos`` device tensor replaces the Python-int ``start_pos``
  everywhere in decode (RoPE row via ``freqs[pos]``, window-ring slot / compressor state slot /
  group index / index-key row via ``index_copy_``, Engram hash cache via tensor gathers);
- **static shapes**: compressed top-k padded to ``index_topk`` (-1 masking), full compressed/index
  caches masked by ``arange(max) < compress_len``, fixed ``topk``/``sort``, no dynamic slicing;
- **persistent cross-layer buffers**: each compression group's compressed KV / top-k idxs are
  written in place (the reference *rebinds* `shared_attn` tensors, which a graph cannot capture);
- **unconditional compressor writes** (safe: an in-progress group index is never read before the
  group completes and overwrites it);
- **one graph per sequence**: eager prefill → capture step graph → re-prefill → replay loop with
  ``argmax`` outside the graph; prefill and multi-token calls stay eager/reference.

Validation (`tools/verify_decode_graph.py`, `runs/p2b/decode_graph_verify.json`):

| check | result |
|---|---|
| hash ids (tensor rewrite vs reference `NgramHashState`) | exact |
| eager native stack vs tensor-pos graph forward | logit max abs diff **0**, argmax 100 % |
| captured replay vs same forward eager | logit max abs diff **0** |
| greedy agreement at ctx 64 / 256 / 512 / 1008 | **100 %** |

| context | graph tok/s | ms/token | eager tok/s | speedup |
|---|---:|---:|---:|---:|
| 64 | **308.1** | 3.25 | 22.3 | 13.8× |
| 256 | **314.1** | 3.18 | 22.2 | 14.1× |
| 512 | **305.9** | 3.27 | 23.2 | 13.2× |
| 1008 | **311.5** | 3.21 | 21.4 | 14.6× |

Constraints: single sequence, context ≤ ``max_seq_len`` (1024; no paged KV yet), capture per
prompt (~0.3 s), eager prefill.  Regression test: `tests/test_decode_graph.py`
(`NANOV41_SLOW_TESTS=1`, CUDA).  Next per the plan: M2 GPU-busy reductions (vectorized sparse-attn,
T≤8 mHC kernel, elementwise fusion, bf16 head) → 400–600 tok/s; M3 paged KV/continuous batching.

## M2 GPU-busy reduction (2026-11-08): 324 → 433 tok/s ceiling

With the graph runtime in place the step time *is* GPU busy (3.1 ms), so M2 works on kernels and
streamed bytes only:

| change | before | after | quality |
|---|---:|---:|---|
| vectorised decode attention kernel (2-D `BT=16` KV tiles + online softmax over the static window/compressed chunk loops; replaces the scalar 96-iteration loop) | 622 µs/step | **118 µs/step** | identical to the torch mirror at every tested position (bf16-exact), capture parity 0 |
| bf16 inference output head (bf16 copy of the 198.5 MB fp32 head; halves the largest streamed tensor) | head GEMV 365 µs | **228 µs** | greedy 100 % agreement over 128 tokens; T0 BPB 1.922626 → 1.922626 (Δ 0) |

Graph replay 3.08 → **2.31 ms/step (433 tok/s ceiling)**; verified greedy loop **396–397 tok/s**
at ctx 64/512 (`runs/p2b/decode_graph_verify_m2.json`), eager-native vs graph-forward logit diff
still **0** and greedy agreement **100 %**.

M2c adds a **decode-specialised mHC chain** (`nanov41/mhc_decode_fast.py`): three single-program
kernels per block replace the training-tiled mixes / sinkhorn / post launches (T=1 kept the 16-row
`tl.dot` padding).  mHC GPU time **648 → 226 µs**, graph replay **2.31 → 1.86 ms (537.6 tok/s
ceiling)**, verified greedy loop **487.5 tok/s** at ctx 64/512, graph parity still logit diff 0 /
greedy 100 %; block-level vs fused v3: ffn pre_mix max 1.5e-4, y bf16-rounding, logit mean 0.0119
(the same class as fused-v3-vs-torch).  Tools: `tools/verify_mhc_decode.py`,
`runs/p2b/mhc_decode_fast_verify.json`, `runs/p2b/graph_ceiling_m2c.json`.

Remaining 1.86 ms: elementwise+copies 897 µs, gemm 383 µs, mHC 226 µs, attention kernels 110 µs,
reductions 85 µs, small-M MoE 72 µs, other 58 µs (M2d: fused block RMSNorms, indexer/compressor,
MoE gathers/gate, attention out-proj).

Old (pre-M2c) breakdown: elementwise+copies 954 µs, mHC 648 µs, gemm 398 µs, attention kernels 118 µs,
reductions 89 µs, small-M MoE 75 µs.  mHC launch-config sweeps gave no gain (`BM=16` is the
`tl.dot` minimum at M=1), so the next M2 item is a dedicated T≤8 mHC kernel, then the
elementwise/copy audit.  Tool: `tools/bench_head_bf16.py`; raw: `runs/p2b/head_bf16_m2.json`,
`runs/p2b/graph_ceiling_m2.json`.


## M2d (2026-11-08): fused decode MoE / RMSNorm / head, and exact kernel attribution

`tools/profile_ranges_decode.py` now attributes kernels exactly: each CUDA kernel event's
`external_id` matches its launching CPU op's `external_id`, and that op's `cpu_parent` chain
contains the `record_function` module ranges.  On the eager decode step (same kernels as the graph)
the dominant owner was the small-M MoE (~870 us/step, 38 %), then mHC 279, attention 430 (CUPTI
inflated ~16 %), block RMSNorm 295.

- [`nanov41/moe_decode_fast.py`](../nanov41/moe_decode_fast.py): four Triton kernels per layer
  replace the small-M torch chain (gate casts + fp32 GEMV + sqrt-softplus + `topk(16)` + gather,
  3 × `index_select` + 3 × routed `bmm`, 3 × shared `bmm`, clamp/silu/cast, bf16 `index_add_`):
  `moe_gate_kernel`, `moe_gu_kernel` (expert offset resolved in-kernel; identical bf16 rounding
  chain), `moe_down_partial_kernel` (K-split) + `moe_down_combine_kernel`.  Validation
  (`tools/verify_moe_decode_fast.py`): routing indices identical on all layers/rows, fp32 routing
  weights ≤1.2e-7, output diff = bf16 rounding.
- [`nanov41/rmsnorm_decode.py`](../nanov41/rmsnorm_decode.py): one-program-per-row RMSNorm (fp32
  statistics, one store rounding) for the 24 block norms + final/compressor norms; prefill keeps
  the reference path.
- [`nanov41/head_decode.py`](../nanov41/head_decode.py): `head_gemv_kernel` streams the 99 MB bf16
  head at ~0.91 TB/s (109 us vs cuBLAS GEMV 228 us), keeping `F.linear`'s bf16 output rounding.

Effect: graph replay **1.86 → 1.231 ms (537.6 → 812.4 tok/s ceiling)**, verified greedy loop
**487.5 → 790 tok/s**, parity unchanged (logit diff 0, capture parity 0, greedy 100 %, generated
text identical to M2c).  Raw: `runs/p2b/graph_ceiling_m2d.json`,
`runs/p2b/decode_graph_verify_m2d.json`, `runs/p2b/moe_decode_fast_verify.json`,
`runs/p2b/decode_attribution.json`.

## M3 (2026-11-08): batched decode kernels + batch-invariant projections

Continuous batching needs the decode step to run B sequences at once.  The model context (1024) is
the same size as one cache row, so "paging" degenerates to slot rows and the runtime replays one
captured graph per batch size 1..16 (`nanov41/batch_decode.py`).  Kernel/runtime changes:

- batched sparse attention (one program per (sequence, head), per-row window/compressed top-k),
  batch RoPE (per-sequence freqs rows), batched mHC kernels (one program per row; block RMSNorms
  fused in), per-row cache scatter and top-k buffers (`nanov41/decode_graph.py`);
- [`nanov41/linear_decode.py`](../nanov41/linear_decode.py): `linear_decode_kernel`, a
  `BLOCK_M=16`-tiled projection with a fixed K-chunk reduction order, installed on the reference
  `linear()` helper and used by the Engram decode gate.  This removes cuBLAS' M=1-GEMV vs M≥2-GEMM
  rounding difference, which otherwise makes batched decode drift from the single-sequence runtime
  (first logit diff at step 4, token divergence at step 9, even at B=2);
- block-diagonal `wo_a` as an explicit fp32 contraction in both the eager and graph decode paths
  (`attn_decode.blockdiag_wo_a`), for the same batch-invariance reason.

With those, batched decode is **bit-identical** to the single-sequence runtime at B=1..16:
`runs/p2b/batch_decode_m3.json` (B=16: 1.882 ms/step, **8775.8 tok/s aggregate**, 100 % token
agreement every sequence), `runs/p2b/continuous_decode_m3.json` (24 mixed requests with arrivals,
departures and slot compaction: 5.85 k decode tok/s, 24/24 token-identical), and
`runs/p2b/decode_graph_verify_m3.json` (single-stream: logit diff 0, capture parity 0, greedy
100 %, text identical to M2d, 780/716 tok/s at ctx 64/512).
