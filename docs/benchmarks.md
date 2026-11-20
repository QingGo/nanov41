# Benchmarks

All numbers below were produced by this repository's own scripts. Raw JSON evidence for the
GPU runs is checked in under [`runs/`](../runs/). Absolute throughput is sensitive to
torch/CUDA/driver versions and host CPU contention — compare within a run, not across machines.

> **Target update (2026-10-31):** the project goal is **60% precision-discounted MFU**, not 60%
> raw bf16-MFU. R50 best is **35.58% discounted**; 60% discounted at the R50 mix is **35.83 ms**,
> so the gap is **1.69×**. The 55.25 ms raw-60% wall below is a diagnostic, not the target. R58’s
> component floor is **42–46 ms (~48–53% discounted)**; 60% on sm_120 needs a different platform
> or an algorithmic change. See [`optimization-playbook.md`](optimization-playbook.md).

## 1. CPU / Apple MPS parity and smoke tests

Measured 2026-10-27 on macOS arm64: Python 3.13.2, torch 2.9.0, 4 CPU threads, MPS available,
no CUDA.

| Command | Result | Wall time |
|---|---|---|
| `parity.py --dtype float32` | ALL PASS (T1 2.123e-07, T2 0.0, T3 680040 grads / 95.6% non-zero / 0 NaN) | ~6 s |
| `parity.py --dtype bf16` | ALL PASS (T1/T2 = 0.0) | ~16 s |
| `parity.py --dtype float32 --device mps` | ALL PASS (T1 2.384e-07) | ~117 s (first run) |
| `train.py --config tiny_all_topk --steps 100 ...` | loss 6.2230 → 5.3106 | 4.8 s (~2,650 tok/s) |
| `train.py --config d8 --steps 0` | 81.62M params (15.43M non-embedding) | ~6 s |
| `train.py --config d12 --steps 0` | 193.41M params (94.12M non-embedding) | ~7 s |

The CPU training smoke test is deterministic: two runs produce bit-identical loss curves.

## 2. RTX 5090 training breakdown (d12, bf16, torch 2.12.1+cu130)

`bench_mfu.py` with the default fused head + shift CE, per-component timing and MFU against
bf16 peak. Raw results: [`runs/p0_b8.json`](../runs/p0_b8.json),
[`runs/p1_b16_s512.json`](../runs/p1_b16_s512.json).

| Config | b × seq | tokens/step | step time | tokens/s | achieved TFLOPS | bf16 MFU | device-busy |
|---|---|---|---|---|---|---|---|
| d12, attention ref / MoE loop | 8 × 1024 | 8,192 | 236.93 ms | 34,576 | 14.66 | 7.0% | 85.1% |
| d12, attention ref / MoE loop | 16 × 512 | 8,192 | 199.65 ms | 41,031 | 17.39 | 8.3% | 84.3% |

Each step launches ~8.5k–9.9k kernels, so these runs are partly **CPU-launch bound**, not tensor-core
bound. That is the motivation for the fused/chunked head, grouped MoE GEMM and CUDA-graph work
tracked for W2; the graph-training driver is internal tooling, but the W2 results and their JSON
evidence are summarized in §3 below (W1/W2 kernel-parity checks that are CPU-runnable are in
[`tests/`](../tests/)).

Absolute peak used for the MFU denominator: **bf16, 5090 vendor peak (209.5 TFLOPS)**. Throughput
figures should be compared between variants measured in the same interleaved run
(`tools/ab_bench.py`), not across sessions.

**Host-CPU sensitivity (re-run 2026-10-28).** A fresh run of the same command on a different 5090
host gave **275.6 ms/step, 29,720 tok/s, device-busy 54.6%** (vs 236.93 ms / 34,576 tok/s /
85.1% above): the step is CPU-launch bound, so the host CPU (and its contention) changes the result
by ~15%. Both runs are kept as evidence — [`runs/p0_b8.json`](../runs/p0_b8.json) (original) and
[`runs/p0_b8_5090_rerun_2026-10-28.json`](../runs/p0_b8_5090_rerun_2026-10-28.json) (re-run) —
and numbers should only be compared within one session.

## 3. W2 training-step results (best measured, R42 → R50)

W2 optimized the training path (fused kernels, grouped MoE, CUDA-graph capture) using an internal
graph-training driver that is **not part of this repository**. Curated JSON evidence is checked in
under [`runs/`](../runs/); this section states the measurement protocol and the results.

**Protocol.** d12, bf16 on RTX 5090, torch 2.14.1+cu130. The step includes the optimizer
(graph replay + optimizer); before timing, 3 training steps are compared against eager and the
parameter parity is reported as `parity_param_rel` (all runs below: **0.0**). Tokens/step =
16,384 at b=16×seq=1024 (8,192 at b=8). FLOPs/token = 423.9 MFLOP (3× forward FlopCounter).
Conventions: **raw bf16-MFU = TFLOP/s ÷ 209.5 TF** (cross-round throughput, precision-inflated);
**discounted MFU = Σ α·FLOPs ÷ (t · 209.5 TF)** with α = 1 / 0.5 / 0.25 for bf16 / fp8 / fp4
(the DeepSeek-V4.1-Flash report's Figure 2 convention) and a **single fixed denominator**.

| Run (b=16, seq=1024) | eager step (ms) | graph step (ms) | graph TFLOP/s | raw bf16-MFU |
|---|---:|---:|---:|---:|
| R42 — first graph-trained best | 122.90 | 92.08 | 75.43 | 36.0% |
| R48 — best | 110.38 | 60.92 | 114.01 | 54.42% |
| R49 — best (re-measured as R50 base) | 102.34 | 60.94 | 113.96 | 54.40% |
| **R50 — best (+ MoE NOPAD)** | — | **60.43–60.48** | **114.83–114.93** | **54.81–54.86%** |
| R50 — MoE fp8 v1 | — | 60.88–60.94 | 114.07–113.96 | 54.40–54.45% |
| R50 — best at b=8, seq=1024 | 103.13 | 34.87 | 99.58 | 47.53% |

- **Raw-60% diagnostic (not the project target): bf16 wall at b=16 is 55.25 ms**
  (125.7 TFLOP/s); R50 best is **5.2 ms** away. The project target is 60% **discounted** (35.83 ms).
- **Where the 5.18 ms is (raw-60 diagnostic, not the project target):** the FP8 head+CE is
  already at ~**349 TF ≈ 83% of the 419 TF FP8 peak** (≤2.35 ms of theoretical room); the non-head
  work is **2.063 TFLOP bf16 in 46.4 ms = 44.4 TF = 21.2% of bf16 peak**, so the *raw* 60% wall
  only needs the non-head to improve **~13% (to 50 TF)**. The slack is in the non-GEMM parts:
  MoE ~9.0 ms (78–84% of bf16 peak, room only via FP8 wgrad), attention ~6–10 ms (memory-bound,
  ~19% peak), mHC ~8.4 ms (bandwidth-bound, ~1% peak) and a ~21 ms residual tail (optimizer 2.7 ms,
  attention projections/rope/indexer, embedding/norm/gate/copies, plus the 2.1 ms host/launch gap).
  A linear fit of the measured b=8 (34.87 ms) and b=16 (60.43 ms) steps gives
  `step ≈ 9.31 + 3.195·b`; extrapolated raw values are **b=24 ≈ 57.8%, b=32 ≈ 59.4%,
  b=48 ≈ 61.1%** if the 32 GB fits, but these are raw-MFU projections, not the discounted target.
  The **project target is 60% discounted (35.83 ms)**, which requires ~1.69× overall (and ~1.9×
  non-head) and is not reachable on sm_120 with tile/fusion alone.
- **Discounted MFU (precision-fair):** weight operations by 1 / 0.5 / 0.25 for bf16 / fp8 / fp4
  and divide by the fixed 209.5 TF bf16 peak — no per-run peak swapping. R50 best (fp8 head +
  bf16 rest) = **35.58%** discounted MFU; its 60% wall is **35.83 ms**. What-if the MoE also ran
  fp8 at the same 60.43 ms: discounted MFU 29.35%, 60% wall 29.56 ms — i.e. precision only helps
  when it is actually faster at equal effective work. Tools: `tools/mfu_accounting.py`.
- **Low-precision expert GEMMs are a full-scale win, not a nano-scale one (R51 microbench,
  2026-10-29):** with MXFP4-QAT expert weights + FP8 compute, per-expert GEMMs at the **nano
  d12 shapes** are *not* faster than bf16 (graph-replay wgrad 384×384×2048: FP8 0.0103 ms vs
  bf16 0.0083 ms; eager loops 3–4× slower because the GEMMs are launch/tile dominated). At
  **full V4.1-Flash expert shapes** (dim 5120, moe_inter 2304, 2k–16k tokens/expert) FP8 is
  **1.76–2.43× faster** (374–551 vs 213–227 TFLOPS). A 30-step MXFP4-QAT training A/B at nano
  scale decreased loss 0.2499→0.1486 vs bf16 →0.1412 (+5.2% final loss), i.e. QAT trains but
  the throughput win only appears at target scale. Evidence:
  [`runs/r51_fp4_expert_microbench_5090.json`](../runs/r51_fp4_expert_microbench_5090.json).
- **Grouped/CUDA-graph execution is the bigger nano-scale lever (R52 Step-0, 2026-10-29):** a
  one-layer, 16-expert bf16 microbench runs at 36–45 TFLOPS as an eager per-expert loop, and at
  **83–105 TFLOPS (2.0–2.3×)** once the same work is CUDA-graph-captured or batched; at full
  V4.1 expert shapes the loop is already at 206 TFLOPS and grouping adds nothing. The official
  grouped FP8 path (`torch._scaled_grouped_mm`) and DeepGEMM's grouped/Mega MoE kernels are
  gated to SM90/SM100, so a 5090 grouped FP8/FP4 kernel must be written in TileLang/Triton;
  DeepGEMM's Mega MoE (FP8 activations × FP4 weights, UE8M0/32 SF, SM100 tcgen05) serves as
  the layout/fusion reference. Evidence:
  [`runs/r52_grouped_probe_5090.json`](../runs/r52_grouped_probe_5090.json).
- **Projection fusion (measured 2026-10-30, b=16):** fusing the attention `wq_a`+`wkv` projections
  and the shared-expert `w1`+`w3` projections (same input → one GEMM; cat-backward also merges the
  dX/dW GEMMs) gives **59.38 → 59.00 ms (−0.39 ms, −0.66%)**, i.e. 55.86→56.20% raw and
  36.20→36.44% discounted MFU. The fused path is not bitwise identical (step-1 loss rel diff
  7.1e-5 because cuBLAS tiling changes the fp32 accumulation order), so it stays an **opt-in**
  env-gated path (`NANOV41_ATTN_PROJ_FUSED`, `NANOV41_SHARED_EXPERT_FUSED`) and the default remains
  parity-exact. Evidence:
  [`runs/r56_projection_fusion_5090.json`](../runs/r56_projection_fusion_5090.json).
  A native-Parameter variant (one `wqkv`/`w13`, no per-forward `cat`) removes another 24 kernels:
  in a second same-session sweep 61.46 → 60.99 (cat) → **60.92 ms (native)**, i.e. −0.54 ms
  (−0.88%) vs baseline and −0.07 ms vs the cat variant, with native loss bitwise equal to cat
  (still 7.1e-5 off the default path). Evidence:
  [`runs/r57_native_param_fusion_5090.json`](../runs/r57_native_param_fusion_5090.json).
- **Tile-knob sweep (measured 2026-10-30, b=16, native-fusion config):** 15 variants across the
  head-v3 K1/K2/K3 and mHC PMPF/MHC knobs — **none beats the defaults** (baseline 60.39–60.42 ms;
  best alternatives within 0.02 ms, worst 62.78 ms). Same session, native fusion is 60.85/60.91 →
  **60.41/60.41 ms (−0.76%)**. The tile/fusion space is now exhausted; the 35.83 ms discounted-60%
  wall is the project target, 1.69× away, and needs a different platform (grouped FP8/FP4 kernels
  exist on H800/B200, not sm_120) or an algorithmic change. Evidence:
  [`runs/r58_tile_sweep_5090.json`](../runs/r58_tile_sweep_5090.json).
- **Head+CE cost decomposition (measured 2026-10-30):** isolated b=16 head+CE is **40.65 ms** on the
  bf16 fused path (chunk 8192; ideal 3-GEMM 23.29 ms at 209.5 TF) vs **14.95 ms** for FP8 v3. The
  bf16 profile is GEMMs 28.5 ms (dX-side GEMM 15.0 vs 7.77 ideal) + bf16 softmax 6.58 ms + CE
  kernels 4.6 ms, which is why the all-bf16 step takes 86.32 ms — keeping the FP8 head is the
  **throughput** choice (60.41 ms). For the current 60% discounted target, all-bf16 is **not** a cheap hedge: it would need
  55.25 ms at 60% discounted vs 86.32 ms measured. The slower all-bf16 step scores a higher
  discounted MFU (38.41% vs 35.59%) but is still far from 60%; raw throughput is a diagnostic.
  Evidence: [`runs/r60_head_isolation_5090.json`](../runs/r60_head_isolation_5090.json).
- **Residual-tail census (measured 2026-10-30, b=16, R50-best graph):** the 57.63 ms device
  time splits into head 13.93, MoE 9.88, mHC 9.48, attention v2 kernel 5.70, small cutlass
  GEMMs 6.47, elementwise/copy/sort 5.80, RMS/rope 1.52 and ~4.85 ms unattributed (plus 3.29 ms
  optimizer/host gap). Candidate replacements were tested: the official TileKernels mHC is
  **slower** than our Triton path on 5090 (2.54 vs 1.02 ms mixes+pre fwd), the official
  sparse-attention kernel is forward-only (no training backward) and only beats the old
  fast_attn helper rather than the production v2 kernel, optimizer-in-graph is neutral, and a
  head-tile sweep confirms the defaults. Evidence:
  [`runs/r55_tail_census_5090.json`](../runs/r55_tail_census_5090.json).
- **Precision-mix / batch decision gate (measured 2026-10-30, b=16, same session):** the FP8
  head mix gives **61.12 ms / 54.24% raw bf16-MFU / 35.18% discounted MFU**; reverting the head
  to bf16 gives **86.32 ms / 38.41%** — only +3.2pp discounted, because the bf16 head path runs
  at ~60% of bf16 peak, so "go all-bf16" is not a cheap hedge. Batch scaling cannot reach 60%
  either: the FP8 head v3 has an int32 offset limit (`TP·V < 2³¹` cap; ≤16,610 tokens at
  V=129,280), so b≥24 silently fell back to the bf16 head, and the all-bf16 b=16→48 points fit
  ~5.0 ms per 1k tokens ⇒ ~41% discounted asymptote. Reaching the project target of discounted 60% (35.83 ms) needs component-level work
  (head, MoE, attention, mHC, tail) **plus** either a platform with grouped FP8/FP4 or an
  algorithmic change; R58’s component floor is 42–46 ms (48–53%). Evidence:
  [`runs/r54_precision_mix_5090.json`](../runs/r54_precision_mix_5090.json).
- **Scaling check:** d12 matches V4.1-Flash **226.5×** down-scaling in both activated parameters
  (16B vs 70.65M) and raw decode FLOPs (32 GFLOP vs 141.3 MFLOP); total backbone params (552B,
  2854×) are **not** a compute-scaling metric. See
  [`v41-scaling-check.md`](v41-scaling-check.md).
- **Negative result — MoE fp8 v1**: kernel-level wins (`_aggemm_tb` −25%, `_swiglu_fwd` −5.4%)
  were cancelled by the added per-layer weight/activation quantization (~0.45 ms) and the
  latency-bound K=384 grouped GEMMs. End-to-end step was unchanged (60.88–60.94 vs 60.94 ms).
  Next planned work (not measured): fp8 on the large-K dW segment with pre-quantized
  activations/weights → projected ≈57.5 ms / ≈57.6% bf16-MFU.
- **Reproducing**: the driver is internal, so these numbers cannot be reproduced from this
  checkout; the JSON files under `runs/` are the primary evidence. Absolute timings are single
  session and host-CPU sensitive (see the re-run caveat in §2).

### R69 — MFU push (measured 2026-11-05; torch 2.12.1 → 2.14.1+cu130, tilelang 0.1.15)

Five opt-in flags were accepted after same-session interleaved A/B; every run keeps the 3-step
eager-vs-graph parameter parity at **0.0**. Numbers below are this session's; the host moved ~3%
between rounds, so same-round deltas are the authoritative metric (evidence:
[`runs/r69_mfu_5090.json`](../runs/r69_mfu_5090.json)).

| Config (b=16, seq=1024) | graph step (ms) | tok/s | raw bf16-MFU | Δ same-round |
|---|---:|---:|---:|---:|
| R50-env baseline (this session) | 59.90 / 61.64 | 273.5k / 265.8k | 55.3% / 53.8% | — |
| + mHC V4 split (`NANOV41_MHC_V4=1 V4_MODE=split`) | 59.03 | 277.5k | 56.2% | **−1.44%** |
| + MoE fast layout (`MOE_LAYOUT=fast`) | 59.61 | 274.9k | 55.6% | −0.49% |
| + attention bf16 compress-dkv (`ATTN_DKV_BF16=1`) | 58.17 | 281.7k | 57.0% | −0.77% on top of the two above |
| + MoE DWS inline (`MOE_DWS_INLINE=1`) | 57.19 | 286.5k | 58.0% | **−1.63%** |
| + MoE fwd/bwd BK=64 (`MOE_TILE=64,64,64,4,3`) | 57.61 | 284.4k | 57.5% | −0.91% |
| **accepted R69 env** (all five, measured on a slower-host round) | **58.78** | **278.7k** | **56.4%** | **−4.64% vs same-round base** |
| R69 env + 4× gradient accumulation | 229.19 / 4 micro-batches | **285.9k** | **57.9%** | −2.41%/token |

- **Estimates on the faster host state** (component-stacked, from same-round deltas): R69 env
  ≈ **56.7 ms / 289.2k tok/s / 58.5% raw MFU**; with 4× accumulation ≈ **55.3 ms / 296.2k tok/s /
  59.9% raw MFU**. Measured absolute best in the session: **58.14 ms / 281.8k tok/s / 57.0%**.
- **Rejected**: the earlier MoE R65 bundle (+2.40%; its FUSED_DY part is +0.89% on W2), projection
  fusion (+0.45%; R56/R57's earlier −0.7% does not reproduce on this base), attention v2 `G=4`
  (+0.13%), `V3_PMPFB_BK=64` (+0.10%), optimizer-inside-graph (−0.39% but 3-step parity 1.62),
  CE chunk 4096/16384 (no change).
- **Optimizer is 4.39 ms/step (7.5%)**: `--skip-opt` measures 53.75 ms vs 58.14 ms. 4× gradient
  accumulation amortizes it (parity 0.0 vs eager accumulation; effective batch becomes 4×16K).
- **Kernel census** (device 56.27 ms/replay, 2857 kernels): head K1/K2/K3 13.77 ms (already ≈83% of
  the fp8 peak), attention TileLang 5.31 ms (47 launches), MoE ~9.5 ms, mHC ~6.8 ms, small cutlass
  GEMMs ~6.6 ms, elementwise/copy/sort ~4.4 ms (incl. 100 vectorized `add` kernels 2.13 ms of
  gradient accumulation and 0.83 ms CUB radix sorts). The remaining slack is bandwidth-bound.
- **torch 2.14.1 upgrade (same host/GPU, follow-up 2026-11-05):** the image's torch was 2.12.1;
  installing the R50/R63 known-good stack (torch 2.14.1+cu130 + triton 3.8.0, tilelang 0.1.15 reused
  from the base env via `venv --system-site-packages`, `tools/setup_torch214.sh`) is a **device-side
  win**: same-host 2×2 with 2 interleaved rounds — base config 61.46 → 61.13 ms (**−0.54%**), R69
  config 58.67 → **57.54 ms (−1.93%)**, eager host cost unchanged (113.4 vs 113.2 ms). The R69 flag
  effect grows from −4.53% (2.12) to −5.87% (2.14). With 4× accumulation the new best is
  **55.83 ms per 16K tokens = 293.5k tok/s / 59.4% raw bf16-MFU** (vs 57.06 ms on 2.12.1, −2.15%);
  3-step parity stays **0.0** and the GPU kernel tests (`test_fast_attn_v3`, `test_moe_fast_layout`)
  pass under the new stack. torch 2.14 also has `set_override_stale_capture_stream` (the harness
  prints `override API: True`), so future harness versions can capture after the eager baseline
  again; the current capture-first grad-mode recipe keeps working unchanged. Evidence:
  [`runs/r69_mfu_5090.json`](../runs/r69_mfu_5090.json) `torch214_upgrade`.
- **Shape sweep at 16K tokens/step**: b64×256 59.10 (best, −1.43%), b32×512 59.52 (−0.73%),
  b16×1024 59.96, b8×2048 60.75, b4×4096 61.25, b32×1024 OOM (graph pool + 32K activations > 32 GB).
- **Capture recipe on torch 2.12**: the graph must be captured **before any eager backward**, with
  `torch.autograd.grad` ("grad mode") warm-up+capture on one side stream — 2.12 lacks
  `set_override_stale_capture_stream`, and a prior `.backward()` makes the full-model capture fail
  with `cudaErrorStreamCaptureImplicit`. The harness (`tools/bench_graph_train.py`) now does this and
  also supports `--accum K`.

## 4. sm_120 NVFP4 / FP8 experiments (optional, experimental)

**Grouped FP8 GEMM prototype (measured 2026-10-30, RTX 5090, torch 2.12.1+cu130):** a single
Triton grouped kernel (`C[e] = A[e] @ B[e]`, one program per expert/tile) was benchmarked on the
nano MoE shapes (E=16, K=384 for fwd, K=2304 for dW). It is **2.8–6.5× faster than `torch.bmm`**
on these batched shapes, and the e4m3 variant is **1.27–1.70× faster than the bf16 variant**
(best dW13 tile 128×128×64: **358.8 TFLOPS** vs 210.6 TFLOPS bf16). The catch is accuracy: with
per-expert scales the FP8 result is ~3.7e-2 max-relative from the bf16 reference, so a
per-token/block scale strategy (or a training-tolerance test) is needed before integration.
This makes the production MoE dW segment (4.93 ms/step) the next target, starting with a bf16
grouped rewrite. Evidence:
[`runs/r59_grouped_fp8_proto_5090.json`](../runs/r59_grouped_fp8_proto_5090.json).

[`experiments/tl_kernels/`](../experiments/tl_kernels/) contains optional TileLang kernels for
NVFP4 block-scaled GEMM and fused FP8 casts on sm_120 (RTX 5090). They are **not** used by
`parity.py` / `train.py` and are not covered by the performance CI. Correctness tests for the
pure-torch host quantizer run on CPU (`pytest experiments/tl_kernels/tests`); GPU tests skip
without sm_120 + TileLang.

**Re-validated on 2026-10-28** on a fresh RTX 5090 (torch 2.12.1+cu130, tilelang 0.1.15):
all 13 GPU tests passed, and `python experiments/tl_kernels/bench.py` produced the numbers below
(raw: [`runs/tl_kernels_5090_2026-10-28.json`](../runs/tl_kernels_5090_2026-10-28.json)). Absolute
timings vary by a few percent between runs.

| Item | Re-validated result (2026-10-28) |
|---|---|
| NVFP4 GEMM kernel-only (head shape 8192×129280×384→pad 512) | **1.66–1.72 ms, 632–655 TFLOPS**, ≈2.2–2.3× vs bf16 (`a @ b.T`, 3.73 ms) |
| Fused TileLang NVFP4 quantizer ([129280, 512] weight) | **0.53–0.54 ms** vs 9.42 ms for the PyTorch quantizer (**17.3–17.7×**) |
| Quantizer agreement vs PyTorch | scale words **100% identical**, packed bytes **93.97%** identical (E2M1 rounding ties), dequant max abs diff **1.37e-2** |
| NVFP4 GEMM vs dequantized reference (small shape) | max abs diff **0.0** (bit-identical to the dequantized-operand matmul) |
| Fused FP8 cast ([8192, 129280]) | **2.10 ms** vs 4.84 ms for the bf16-math PyTorch path (**2.3×**; strict fp32 math path: 13.2 ms, 6.3×); bytes **100% equal** |
| FP4 forward + FP8 backward (head) | ≈2.2× vs bf16 — this end-to-end number is from the internal log and was **not** re-measured here |

CUDA parity was also re-checked on the same instance: `parity.py --dtype float32` and
`--dtype bf16` both print **ALL PASS** (T1 max diff 2.384e-07 / 1.788e-07; T2 0.0). The
informational T1b indexer tie probe shows 6.6e-02 on CUDA (device-dependent tie order; see the
README limitations). Logs: [`runs/parity_cuda_fp32_5090.log`](../runs/parity_cuda_fp32_5090.log),
[`runs/parity_cuda_bf16_5090.log`](../runs/parity_cuda_bf16_5090.log).

## 5. Reproducing

```bash
# CPU/Mac parity + smoke numbers
python3 parity.py --dtype float32
python3 train.py --config tiny_all_topk --steps 100 --batch-size 4 --seq-len 32 --lr 1e-3
python3 train.py --config d8 --steps 0

# CPU kernel-parity regressions (fast subset; slow subset ~133 s)
python3 -m pytest -q -m "not slow"
python3 -m pytest -q -m slow

# optional sm_120 experiments (needs CUDA + TileLang; GPU tests skip otherwise)
python3 -m pytest experiments/tl_kernels/tests -q

# GPU breakdown (needs CUDA; e.g. RTX 5090)
python3 bench_mfu.py --b 8 --seq 1024 --json /tmp/p0_b8.json
python3 bench_mfu.py --b 16 --seq 512 --json /tmp/p1_b16_s512.json
```
