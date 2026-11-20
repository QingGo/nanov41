# R93 / P0b — dense d24_c1 stage-time & MFU decomposition

Status: **final for handoff, partially measured.** b1×8192 graph replay is measured with the
production kernel stack (numbers below); b2×8192 and the `p2_train --profile-loop` segment
attribution were **not measured** — the instance was shut down with 0 free cards available
(Lead's call), so those parts are analytic/static only and are labelled as such.

Method = R45/R6x/R7x standard: `tools/bench_graph_train.py` capture recipe (fwd+CE+bwd inside a
CUDA graph, optimizer outside), 20 timed medians; kernel attribution via `torch.profiler` (3
replays). R93 uses a thin wrapper copy `tools/r93_mfu_bench.py` that adds `--config-json` (aligned
dense config + the K=40k Engram C40=32,410 patch) and an authoritative `--fwd-flops 575,953,504`
(the inherited analytic estimator double-counts the head for dense configs: `2*(head+non_emb)` =
`2*body+4*head`, +14.7% at d24_c1 — do not trust raw numbers from the unpatched bench).
Config: **d24_c1 dense** dim 1024 / 24 layers / h=2304 / V=41,280 / Engram ON (2 layers, 220.05M
table) / seq 8192 / micro b1; env `tools/r90_env5090.sh` + dense downgrade
`tools/r90_env_dense5090.sh` (attn v3 TL, mHC v4 split level 3 + post/split/core triton, fp8-v2
head+CE, fp8 linear v2 scope=wo_b, EW fusion, fused grad, v41 optimizer outside graph).
Accounting fwd = 575.95 MFLOP/token; real + Engram wkv = 591.7 MFLOP/token (+2.7%, excluded by the
repo convention → true hardware MFU is ~2.7% higher than the convention number).

## 1. Measured: b1×8192 graph replay vs the P0b training loop

| measurement | value |
|---|---|
| **graph replay only** (b1, CE_CHUNK=8192, `--skip-opt`, 20 medians) | **181.76 ms/micro** |
| … tokens/s | **45,070 tok/s** |
| … achieved TF / raw MFU @209.5 TF | 77.87 TF / **37.17%** |
| kernels per replay | 6,439 |
| memory after capture | reserved 26.13 GiB, peak_allocated 25.16 GiB, **peak_reserved 27.4 GiB** |
| **P0b train loop** (same shape, 256 micro/step) | 46.7 s/step ÷ 256 = **182.4 ms/micro** = 44,334 tok/s = **36.56%** raw MFU |
| **out-of-graph gap** (optimizer Muon+Adam / Sinkhorn / Engram table / clip / harvest / data loader) | **0.6 ms/micro ≈ 0.3%** |
| train vs replay MFU | 36.56% vs 37.17% → **0.6 pp** |

**Verdict:** at `accum=256` there is **no meaningful out-of-graph headroom** (<1%). The whole MFU
problem is inside the graph. Real-fwd-adjusted (Engram wkv included) the hardware MFU is 37.56%
(train) / 38.17% (replay) — still ~62% of peak left inside the graph.

## 2. Measured: kernel buckets (b1 replay = 181.76 ms = 100%)

Per-replay totals from the profiler (`self_device_time_total / replays`; bucket noise ±1-2%):

| bucket | ms/replay | share | composition |
|---|---:|---:|---|
| **bf16 GEMMs** | **68.7** | **37.8%** | 19 distinct cutlass/nvjet kernels: 128x128_32x4 nt 14.19×72, 256x128_32x3 tn 10.96×51, 128x64_64x3 nn 9.45×48, sm10 gemm 8.04×50, 64x128 tn 5.31×50, 128x64 tn/nn 4.77+4.76×24, wmma small 2.23, … (q/kv/wo low-rank GEMMs + dense FFN w1/w3/w2 at M=8192) |
| **elementwise / copy / norm** | **~68** | **~37.5%** | `at::native::BinaryFunctor<float,float,float> Mul` **17.40 ×476**, direct-copy 5.66×414, bf16-copy 4.49×481, `elementwise 128,2` 4.54×48, `clamp_scalar` 4.40×100, `compare_scalar<float>` 4.22×72, silu_bwd 3.23×24, silu 1.86×24, adds 2.30×101 + 1.39×149, reductions 1.67×174 + 0.90×148 + 0.51×101, memcpy128 2.06×204, memsets 0.58×174, pow_scalar 0.77×101, … |
| **mHC kernels (v3/v4 Triton)** | **~21.5** | **~11.8%** | `_post_bwd_dual` 8.56×45, `_mixes_pre_bwd` 5.51×48, `_post_fwd` 4.11×48, `_mixes_pre_fwd` 2.41×48, `_split_sinkhorn_bwd` 0.93×48 |
| **attention v3 (TileLang)** | **~9.8** | **~5.4%** | `kernel_kernel` 7.84×48 (fwd+bwd fused TL kernel), `_dkv_win_tri` 1.94×23, casts 0.37+0.33 |
| **head / CE (fp8-v2)** | **~2.5** | **~1.4%** | `_ce_bwd_fp8` 0.63, `_ce_fwd` 0.43, nvjet head GEMM 1.20 — V=41k+fp8 head is nearly free |
| long tail (>80 kernel names) | 1.25 | 0.7% | |

**Key inversion vs the MoE stack:** the bf16 GEMMs are already **~87% of peak** if you attribute
~12.5 of the 14.15 TFLOP/micro to them (12.5 TFLOP / 68.7 ms = 182 TF) — the GEMMs are *not* the
problem. **~50% of the step is non-GEMM** (elementwise/copy/norm 37.5% + mHC 11.8%), and that is
where all remaining MFU lives. This is the structural dense-vs-MoE difference: R90/R83 production
MoE (fwd 183M/token) ran 54.2% raw MFU because grouped-GEMM work dominated; d24_c1 does 575.95M
fwd/token but with *only* 37.8% of its time in GEMMs.

## 2b. Static analysis of the biggest non-GEMM items (CPU-only, ranked; no shapes available for graph-replay kernels)

The graph profiler cannot record tensor shapes (replay has no python/aten dispatch), so call-site
attribution below is from the source + kernel names + call counts. Ranked candidates for the
**17.40 ms × 476-call fp32 `MulFunctor` kernel** — the single biggest non-GEMM device item:

1. **Unfused `RMSNorm` (very likely dominant).** `nanov41/_ref_patched_model.py:314-322`
   `RMSNorm.forward`: `x=x.float(); var=x.square().mean(-1); x=x*rsqrt(var+eps); return
   (self.weight*x).to(dtype)` → **3 fp32 muls in fwd** (`square`, `x*rstd`, `weight*x`) plus
   fp32 muls in the bwd (`dx=(dy*weight)`, `dw=(dy*x).sum` etc.), all on `[8192, dim]` fp32
   buffers. The d24 dense stack calls RMSNorm **~4-5×/layer** (attn_norm + ffn_norm at
   `_ref_patched_model.py:986,993`, q_norm/kv_norm in `Attention`/`_window_kv` around lines
   735/752, plus compressor/indexer norms on the source layers) ⇒ **~96-121 calls/step × ~3-5
   fp32 muls ≈ 290-600 Mul kernels** — matches the 476 count and the 17.4 ms (≈36 µs avg for a
   [8192,1024] fp32 mul ≈ 100 MB traffic at ~3 TB/s). **A fused fwd+bwd RMSNorm already exists**
   in the repo — `nanov41/fused_rmsnorm.py` (R45: "6-7 aten kernels → 1 fwd + 2 bwd kernels",
   fp32 internals, dw deterministic, replay-bitwise parity) — but **`NANOV41_RMS_FUSED` is NOT
   set in `tools/r90_env5090.sh`**, so the production stack torches the fp32 path.
   *Safety:* numerics identical to official (fp32 internals, bitwise-validated in R45); the only
   question is speed at d24 shapes (R83's +0.66%-slower result was for the **fp8-cast producer
   fusion variants**, not the plain bf16 fused kernels; R45's plain kernels were a win at d12, and
   d24 doubles tensor size per call, which amortises launch overhead better).
2. **Dense SwiGLU fp32 segment.** `Expert.forward` `_ref_patched_model.py:862-878` (P6 `DenseFFN`
   wraps it): `gate=w1(x).float(); up=w3(x).float(); clamp×2; x=F.silu(gate)*up; x.to(dtype)` →
   1 fp32 mul fwd + 2 fp32 muls bwd (`grad_gate=dy*up`, `grad_up=dy*silu(gate)`) ×24 layers = 72
   Mul kernels, plus clamp (4.40 ms ×100 counter suggests ~4/layer), the `.float()`/`.to(dtype)`
   copies (part of the 5.7+4.5 ms copy buckets) and `silu`/`silu_bwd` kernels 5.1 ms. `nanov41/
   fused_swiglu.py` (R45) fuses clamp+silu+mul+cast into one kernel and **patches `M.Expert.forward`
   for the `weights is None` path — which is exactly what DenseFFN uses**; also not enabled in
   `r90_env5090.sh`. Expected save ~4-7 ms (clamp + mul + copies + silu kernels).
3. **mHC `hc_split_sinkhorn` wrapper fp32 scaling.** `nanov41/tl_sinkhorn.py:57-67`: the TileLang
   Sinkhorn kernel is fused, but the *wrapper* still does `m=mixes.float()`, 3 fp32
   `m*scale+base` lines (pre/post/comb) and the eps adds ×2/layer ⇒ ~6 small fp32 muls/layer
   (tensors [8192,48] ≈ 1.5 MB, cheap ≈µs each; explain a chunk of the 476 *count* but only
   ~0.5-1 ms of time). Could be folded into the TL kernel, low priority.
4. **RoPE torch path.** `apply_rotary_emb` (official) is not fused (`NANOV41_ROPE_FUSED` not set);
   ~3 rope calls/layer (q, window kv, o) + compressor/indexer ⇒ ~80-100 small kernels; R77/R85
   measured this knob at **~0 (±0.07 ms)** at d12, expected ~0.1-0.3 ms here — negligible.
5. **Indexer fp32 muls** (`_ref_patched_model.py:585-587`: `weights_proj(x)*(softmax_scale*
   n_heads**-0.5)` and `(relu(index_score)*weights).sum(2)`) — runs only on the 2 index-source
   layers at d24, larger tensors ([8192, heads, ncomp]) ⇒ possibly 1-2 ms; `nanov41/fused_indexer.py`
   exists (R47) but is not enabled in the production env (R77 noted it was neutral-to-negative at
   d12).
- **~5.7 ms copy ×414 + 4.5 ms bf16-copy ×481 sources (identifiable classes):** the
  `.float()` / `.to(dtype)` casts of items 1+2 (norm x→fp32, weight·x→bf16, gate/up→fp32,
  silu·up→bf16) account for a large share — **enabling `fused_rmsnorm` + `fused_swiglu` removes
  those casts by construction**; the rest are the Engram table gather (`NanoEngramTable` scatters
  bf16 rows; 2 layers), `static_ids.copy_` (1/replay), `wo_a` einsum prep/pad copies and
  optimizer-state copies (out of graph). No separate copy-fusion project is suggested before 1+2
  are tested.
- `compare_scalar<float>` 4.22 ms ×72 (exactly 3/layer) is **not yet attributed** (candidates:
  mHC v3 clamp/`where` mixes at `fused_mhc_v3` splits, indexer masked_fill, sinkhorn eps
  guards); worth one `--profile-eager` run (with python stack + shapes) as the decisive step.

## 3. b2 × 8192 (16k-token micro) — NOT measured; analytic verdict + measured shortfall

- b2 was attempted in the **training** path during P0b and OOM'd in the capture warm-up **mid-FFN
  forward** (`_ref_patched_model.py:864 gate=self.w1(x).float()`), with **~30.5 GiB allocated and a
  144-256 MiB allocation pending against the 31.36 GiB capacity** (the 144 MiB request is exactly a
  `[16384, 2304]` fp32 FFN buffer). CE_CHUNK reductions (2048/512) did not change the outcome.
- Why CE_CHUNK cannot be the fix (static): at the OOM site the CE head/`fused_head_ce` has not run
  yet in that iteration; the alive memory is params(grad)-states + the per-layer activation set +
  the partially built autograd graph. CE workspace at chunk 8192 with V=41,280 and fp8-v2 is only
  ~0.7-1.3 GiB (and the fp8 path does not materialise an fp32 full-vocab logits tensor), so
  shrinking it to 128 frees <1.5 GiB — an order of magnitude less than the b2 gap.
- Estimated b2 demand: b1 graph-only peak allocated 25.16 GiB (of which ~21.5 GiB is activations:
  params 1.1 + grads 1.1 + init clone 1.5). Doubling the micro to 16k tokens roughly doubles the
  activation set (~43 GiB) ⇒ **b2 needs ~40+ GiB vs 31.36 available; the gap is ~8-10 GiB**, not
  ~1 GiB.
- Memory levers for b2 (next experiment, ordered by value/effort):
  1. **FFN activation recompute** (store only the layer input; recompute `w1/w3+sigmoid` in bwd):
     removes the fp32 gate/up/silu saves — the single largest activation block
     (2·M·h·4B + M·h·2B per layer ≈ 0.75 GiB at b2 × 24 ≈ 18 GiB at b2). Expected to bring b2
     under 31 GiB by itself. Cost: ~1-2 days (patch `Expert` with `torch.utils.checkpoint`-like
     recompute or a custom Function; must keep graph capture + fp8 wgrad path working).
  2. **bf16 FFN intermediates** (gate/up kept bf16, only clamp/silu math in fp32): saves
     ~M·h·4B/layer (~9 GiB at b2 if done instead of recompute; less combined). Simpler (~0.5 day),
     but numerics change slightly → needs loss A/B.
  3. **Muon momentum bf16 / optimizer state reduction** (env `NANOV41_MUON_MOMENTUM_BF16` exists;
     check current default) — remember optimizer states are allocated when the step runs, so this
     also shifts the training peak, not just the replay.
  4. **CE_CHUNK 128-512** (frees ≤1.5 GiB; worth setting at the same time but not sufficient).
  5. Graph-pool trim: the bench's replay pool reserved 26.13 GiB; the training capture recipe adds
     warmup + state buffers. `torch.cuda.graph` private pool cannot be shrunk, but capturing fewer
     buffers (bf16 intermediates) shrinks it proportionally.
- Expected gain if b2 lands: **+3-5 pp train MFU** (production MoE R79 measured b1→b2 = 31.0%→36.8%
  = +5.8 pp; for dense, GEMMs are already ~87% efficient at b1, so the b2 win is mostly halved
  per-token launch/ew overhead and better small-GEMM tiling — hence the lower 3-5 pp). Confidence
  medium (extrapolated, not measured for this config).

## 4. Bottom line: is 45% train MFU plausible?

Arithmetic on the measured b1 replay (181.76 ms, 14.15 TFLOP/micro):
- 42% MFU ⇔ step ≥ 14.15/0.42/209.5 = 160.8 ms → **save ≥21 ms** of the 103 ms non-GEMM time.
- 45% MFU ⇔ step ≥ 150.1 ms → **save ≥31.7 ms** (~31% of the non-GEMM overhead).
- 50% MFU ⇔ step 135.1 ms → save ≥46.7 ms (~45% of non-GEMM).

My assessment (marked as estimates; every item needs a rerun to confirm):
- **42% = believable 1-2 day target**, driven by **b2 unlock** (+3-5 pp) plus the two existing
  R45 fused kernels (`NANOV41_RMS_FUSED` + `NANOV41_SWIGLU_FUSED`; combined expected 8-16 ms ≈
  +2-4 pp if they are neutral-positive at d24 — they remove the fp32 muls/casts that dominate the
  17.4 ms Mul + copy buckets). Confidence: medium-high that 40-42% is reachable; medium that
  42% is.
- **45% = believable 3-5 day target, not 1-2 days.** It requires *both* b2 and a ~2× reduction of
  the elementwise/copy bucket (via RMS/SwiGLU fusion **and** further fusion of the remaining
  clamp/compare/sinkhorn-wrapper/tail kernels, or dropping the fp32 norm semantics with an
  A/B-validated bf16 variant), plus a small GEMM-side gain. Each of those is individually plausible
  (they are all already implemented kernels in the repo), but none has been measured at d24
  shapes, and R83 showed that eager-fusion wins at d12 can turn into +0.66% losses when the
  producer-fusion adds amax traffic. **Confidence that 45% is reachable at all: medium (~50-60%);
  confidence it is a 1-2 day job: low.** The one structural risk is that the ~103 ms non-GEMM
  cost is dominated by *required* fp32 norm/mHC semantics (bitwise-parity constraint), in which
  case the ceiling is ~42-44% rather than 45%+.

## 5. Ranked next experiments (GPU, cheapest first; all need A/B B A interleaved because GPU
   clocks cannot be locked in the container — R84 methodology note)

| # | experiment | expected | effort | numerics |
|---|---|---|---|---|
| 1 | `NANOV41_RMS_FUSED=1` A/B on the b1 graph-only bench (+`--profile 3`, compare buckets) | −6-10 ms → +1.5-2.5 pp | 10 min GPU | fp32-internal, R45 replay-bitwise parity |
| 2 | `NANOV41_SWIGLU_FUSED=1` A/B (same bench) | −3-7 ms → +1-2 pp | 10 min GPU | same clamp math, fp32 internal |
| 3 | both together | −9-17 ms → +2.5-4 pp | 20 min GPU | re-check loss few-step |
| 4 | b2 memory: FFN activation recompute (or bf16 intermediates first), then b1→b2 graph-only replay + 3-step p2_train probe at 2^21 tok/step | +3-5 pp | 1-2 days | loss A/B 50 steps |
| 5 | decisive attribution: bench `--profile-eager 3` (aten by shape **with python stack**) to name the 476 muls / 414 copies / 72 compares exactly before any more fusion work | —  | 10 min GPU | none |
| 6 | sinkhorn-wrapper muls into TL kernel; remaining `compare_scalar`/clamp fusions | ≤1 ms | 1-2 days | fp32 |
| 7 | body fp8 linear at d24 GEMMs: extend `NANOV41_LINEAR_FP8_V2_SCOPE` from `wo_b` to dense-FFN GEMMs (R82 measured 1.40-1.41× on full-V4.1 shapes; wgrad stays bf16) | GEMM bucket −10-20 ms if it holds → +2.5-4 pp | 2-5 days | R82 had loss rel 8e-7 |
| 8 | GEMM tile tuning for the small-N kernels (q_lora/o_lora/wq_a) | +1-2 pp | 2-3 days | bitwise |

**Fastest credible path to 42%:** #1+#2 (0.5 h GPU) → measure; if +2-4 pp holds, #4 (1-2 days) →
40-42%. To 45%: add #7 and #6, and accept it is a 3-5 day kernel campaign with medium confidence.

## 6. Exact artifacts / reproducibility
- Bench command actually run (b1 baseline):
  `CE_CHUNK=8192 /root/r93_bench.sh --b 1 --seq 8192 --steps 3 --bench 20 --graph-only --skip-opt --profile 3 --json runs/r93/p0b/mfu_b1_chunk8192.json`
  where `/root/r93_bench.sh` sources `tools/r90_env5090.sh` + dense unset + K40k tokenizer and
  calls `tools/r93_mfu_bench.py --config-json docs/r93/configs/d24_c1_dense40k.json --mhc-level 3
  --mhc-post-triton --mhc-split-triton --mhc-core-triton --peak bf16_5090`.
- NOTE: the naive `source tools/r90_env_dense5090.sh` from a `bash -c` shell silently fails
  (`dirname "$0"` = "." → `./r90_env5090.sh` missing), which silently drops the whole production
  kernel stack. Two failed bench attempts were caused by this (one exact-mHC attempt at 25.16 GiB
  b1 peak, one with the v2 tl attention backward needing 136 KiB dynamic shared memory). Always
  source the env from a script whose `$0` is in `tools/`.
- Raw profiler logs and `mfu_b1_chunk8192.json` were written on the instance
  (`/root/r93_mfu_b1_8192.log`, `/root/r93_mfu_chain*`); numbers above are transcribed from that
  log before shutdown. Re-create with the command above; expect ±1-2% kernel-level noise.
- Unmeasured items: b2×8192 replay/memory at CE_CHUNK 2048/1024/512; `p2_train --profile-loop 5`
  segment split (expected to confirm the 0.3% out-of-graph gap); b1 CE_CHUNK 8192 vs 2048 replay
  A/B; `--profile-eager` stack attribution.
