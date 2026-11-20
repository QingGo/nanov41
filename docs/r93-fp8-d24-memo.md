# R93 — FP8 for d24_c1 dense body: paths, measurements, enablement, gates, expected band
Status 2026-11-15, CPU-only repo read (no GPU, all instances down). Target **d24_c1 dense** (dim 1024,
L24, dense SwiGLU h=2304, seq 8192, V=41,280, Engram 220.05M). Baseline b1×8192 CUDA-graph replay
**181.76 ms / 14.15 TFLOP / raw MFU 37.17 %** (`runs/r93/p0b/mfu_profile.md`; buckets: bf16 GEMM 68.7 ms
=37.8 %, head/CE fp8 2.5 ms, ~50 % non-GEMM). FLOP shares: dense FFN **8.35 TFLOP (59 %)**,
attn+other body 3.72 (26 %), head 2.08 (14.7 %).
**GO to validate FFN+wo_b+engram FP8 (~2 h ≈ ¥6).** It is the only lever on the 68.7 ms GEMM bucket:
central **−15.8 ms/replay → 165.9 ms, raw MFU 40.7 % (+3.55 pp)**, band +2.7…+4.1 pp. It is *not* the
biggest lever (RMS/SwiGLU fusion = +2.5–4 pp for 20 GPU-min, do first), and the FFN speedup below is
**interpolated, not measured** — E1 gates it.

## 1. Available FP8 paths, exact knobs — and what is ON today
| path | what | env / API |
|---|---|---|
| **`nanov41/fused_fp8_linear.py` (v2, production)** | training-oriented "cached fp8 W + cuBLASLt `_scaled_mm`": per-output-channel W8 cache (`FP8WeightSlot`, static graph-safe buffers), Triton rowwise act quant (torch fallback), fwd+dgrad+wgrad | `NANOV41_LINEAR_FP8_V2`; `_SCOPE` = comma **substring** list on full param name (empty=all eligible); `_MIN_FLOPS` = `2·M·N·K` per-call cutoff; `_QUANT=triton\|torch`; `_DGRAD=1`; `_WGRAD=0\|1(Triton split-K)\|2(cuBLASLt vendor)`; `register_weights(model)`→`install(M)`→`maybe_refresh()` |
| `nanov41/linear_fp8.py` (O1, default-OFF) | official blockwise **128×128** semantics: `tl`=`ref/kernel.py::act_quant`+`fp8_gemm`, `sm`=`_scaled_mm`; graph-capturable; d12 NO-GO | `NANOV41_LINEAR_FP8`, `_BACKEND=tl\|sm`, `_BLOCK=128`, `_ROUND_SCALE`, `_DGRAD`, `_WGRAD`, `_PER_ROW_A`, `_QSHARE`, `_SCOPE`, `_MIN_FLOPS` |
| `nanov41/fp8_head_v2.py`+`head_fwdgrad.py` | fp8 head/CE (`_q_row/_q_col/_q_per_tensor`), fwd-precomputed grads, per-step W cache | `NANOV41_HEAD_FP8=v2`, `_HEAD_FWDGRAD=1`, `_HEAD_FWDGRAD_DW=bf16`, `_HEAD_WCACHE=1` |
| `nanov41/fp8_cast_registry.py` | producer→consumer fp8 handoff (fused cast); **negative, keep OFF** | `NANOV41_FP8_CAST_FUSE` (unset) |
**ON in `tools/r90_env5090.sh`:** `LINEAR_FP8_V2=1`, `_SCOPE=wo_b` (only `attn.wo_b`), `_WGRAD=2`, dgrad/quant
defaults; `HEAD_FP8=v2`+`HEAD_FWDGRAD=1`+`HEAD_WCACHE=1`. **OFF:** `NANOV41_LINEAR_FP8` (O1), `FP8_CAST_FUSE`.
⇒ today FP8 = **wo_b + head only**; dense FFN, engram wkv, wq/kv are bf16 = the opportunity.
## 2. Measured bf16-vs-fp8 per-shape data (graph-timed; R82 §6–§9, `runs/r82/fp8_v2_bench*.json`)
"total" = bf16(fwd+dgrad+wgrad) ÷ (fp8 fwd + fp8 dgrad + **bf16 wgrad**), quant incl.:
| anchor (M,K,N) | bf16 fwd/dgrad/wgrad ms | fp8 fwd× / dgrad× | total |
|---|---|---|---|
| wo_b 16384,1024,512 | .0984/.0959/.0884 | 1.39 / 1.47 | 1.26 |
| engram 16384,768,2560 | .3112/.3079/.2942 | 1.63 / 1.35 | 1.28 |
| dense twin 16384,512,512 | .0500/.0505/.0489 | 1.12 / 1.14 | 1.08 |
| wq_a 16384,512,128 · wq_b 16384,128,512 | .0220/.0262/.0217 · .0300/.0214/.0206 | 0.89/0.96 · 1.10/0.86 | 0.95 · 0.99 |
| wkv/compressor 16384,512,64 | .0165/.0221/.0202 | 0.64 / 0.80 | 0.80 |
| full-V4.1 up 8192,5120,2304 | .826/.856/.857 | 1.69 / 1.84 | **1.40** |
| full-V4.1 down 8192,2304,5120 | .861/.849/.864 | 1.83 / 1.73 | **1.41** |
v2 numerics: per-tensor rel vs bf16 **fwd 3.6–3.7e-2, dgrad 3.4–3.9e-2** (e4m3 grade).
**d24_c1 target shapes (NO measurement except as noted — interpolate, don't invent):**
| target | status / estimate |
|---|---|
| FFN w1,w3 (8192,1024,2304) / w2 (8192,2304,1024) | **NO measurement.** GEMM-only fwd/dgrad ≈1.75–1.9× (N=2304 as full-V4.1, K=1024–2304); +quant (≈0.01–0.02 ms/GEMM) ⇒ **total 1.30–1.45, central 1.35** |
| attn wo_b (8192,2048,1024) | NO direct; d12 1.26 at smaller K/N ⇒ **1.30–1.40** |
| engram wkv (8192,768,5120) | NO direct; d12 1.28 @N2560, N doubles ⇒ **1.30–1.40** |
| wq_a (8192,1024,256), wq_b (8192,256,1024) | NO direct; d12 analogs 0.95/0.99 ⇒ ~1.0 (skip phase 1) |
| wkv / compressor-ratio1 (8192,1024,64) | NO direct; d12 analog 0.80 ⇒ **negative (skip)** |
| head V=41,280 | already FP8: head/CE bucket **2.5 ms (1.4 %)**, no upside (≤1 pp) |

## 3. Concrete enablement (names, wgrad, capture safety)
- `install()` matches full param name: `eligible_name()` drops `indexer|gate|head|norm|embed|sinkhorn|.experts.`,
  then requires a `_SCOPE` substring. Dense FFN = `inner.layers.{i}.ffn.w.w1.weight` (+`w2`,`w3`; `DenseFFN.w=Expert`,
  `_ref_patched_model.py:875–891`); attn `…attn.{wq_a,wq_b,wkv,wo_b}.weight`; engram `…engram.wkv.weight`.
- **Phase-1 scope:** `_SCOPE=wo_b,ffn.w.w1,ffn.w.w3,ffn.w.w2,engram.wkv` (or `_MIN_FLOPS=3e10`: keeps FFN
  38.7 (w1/w2/w3 alike), wo_b 34.4, engram 64.4 GFLOP; drops wq_a 4.3, wkv 1.07). `attn.wo_a` is an `einsum` on the raw
  weight (not `M.linear`) → **cannot** be converted; leave bf16.
- **wgrad stays bf16 in the first gate.** R84 Triton split-K 0.50–0.54× ⇒ rejected (+2.52 % replay, −2.9 % A/B).
  Viable: R85 vendor cuBLASLt (`_WGRAD=2`, prod default): 1.22× wo_b, 1.37× engram, 1.03× dense twin,
  0.86× shared_w1, 2.14× full-V4.1-up; gate −0.32 % replay/+0.23 % tok, loss rel 1.7e-6. Cost = fused
  rescale+transpose prep (22.6 µs wo_b / 71.9 µs full-V4.1) + first-use `csrc/wgrad_lt.cu` build ~38 s
  (cached) — **build before capture** or it falls back to bf16 mid-run. d24 FFN est ≥1.2× ⇒ +0.2–0.5 pp, gate separately.
- **Graph safety:** `register_weights()` allocates slots before capture; `maybe_refresh()` = only per-step work
  (outside graph, ~0.2–0.5 ms/step); Triton/`_scaled_mm` capture-safe (R82/R85 `capturable:true`; O1 S7 too).
  Master stays **bf16** (W8 re-quantised per step ⇒ implicit STE) — deviation vs official "FP32 master +
  128×128 blockwise"; cuBLASLt has no block-scaled variant, record as alignment deviation (O1 blockwise d12 NO-GO).

## 4. Numerics/parity gates before default — and negatives not to repeat
- **G1:** 20-step loss token-matched A/B, gate **loss rel ≤1e-4** (v2-flip ref 8.4e-6, vendor wgrad 1.7e-6;
  24-layer e4m3 noise is larger → ≤1e-4 honest), no NaN; per-GEMM rel 3–6e-2 expected — **never claim bitwise**.
- **G2:** in-graph replay **A B B A** interleaved (+token-matched 10-step A/B). R83 §7.3: sequential arms drift
  ±0.5–0.8 %, same config can flip sign; gate phase-1 **replay Δ ≤ −8 ms**. **G3:** 50-step A/B same
  box/data/seed: ≥+2 % tok/s, raw MFU +≥1.5 pp, loss rel ≤1e-4, peak mem ≤31 GiB.
- **Do not repeat:** producer-fused cast (R83 A B B A **+0.66 % slower** — producers must write bf16+fp8+amax,
  only ~1/3 casts fusable; keep `FP8_CAST_FUSE=0`); d12 small-shape NO-GO (eager −6.5…−26 % was **CPU-launch
  bound** — fp8 must be graph-timed only); d12 full-body NO-GO (≤+0.2…1 %); fp8 wgrad without split-K (5–7×
  slow), Triton split-K (R84 reject), TileLang wgrad only engram 1.21× (R84); don't chase head (1.4 % of replay).
## 5. Expected GEMM-bucket saving (cons. / central / opt.)
Bucket 68.7 ms ≈ 12.5 TFLOP attributed (182 TF/s ≈ 87 % bf16 peak). FFN 8.35 TFLOP = **66.8 % of bucket
≈45.9 ms** (fwd 15.3 / dgrad 15.3 / wgrad 15.3); attn+engram ≈31 % ≈21.5 ms (wo_b ≈13.6 ms = 2.47 TFLOP,
engram ≈2.1 ms = 0.39 TFLOP, ~5.8–7.1 ms small-N skipped). Assumed totals incl bf16 wgrad:
Saving: FFN 45.9 ms @1.25/**1.35**/1.45 ⇒ 9.2/**11.9**/13.6 ms; wo_b 13.6 ms @1.25/1.32/1.40 ⇒
2.7/**3.3**/3.9 ms; engram 2.1 ms @1.28/1.35/1.45 ⇒ 0.46/**0.55**/0.65 ms; **total 12.4/15.8/18.2 ms**.
⇒ replay **169.4 / 165.9 / 163.6 ms**, raw MFU **39.9 / 40.7 / 41.3 %** (+2.7 /**+3.6**/+4.1 pp), 48.4k/49.4k/50.1k
tok/s. Adding `_WGRAD=2` on FFN (+~1.5–2 ms) → ≈+4 pp (and extend scope to `wq_b`/check b2 headroom only if positive). Risk: d24 FFN speedup is interpolated — if it lands at
dense-twin 1.08× the whole thing is only **+0.5–1 pp** ⇒ E1 gates. If b2×8192 lands, M doubles and the M=16384
anchors apply better (upside, not counted).
## 6. Engram FP8 storage (O4) — alignment/memory item, not an MFU lever
Table currently `NanoEngramTable` bf16 (220.05 M = 440 MB), no fp8 scales; `quantize_table_ue8m0` exists only in
`tools/engram_fp8_export.py` (inference/export). FP8 e4m3 + E8M0 per-32 saves **~213 MB**. On-GPU gain
**~0.0–0.2 pp MFU**: gather traffic only 2 layers × 24 cols × 32 dim × 8192 tok bf16−fp8 ≈12.6 MB/replay
(≈0.004 ms); real engram cost is Sinkhorn/optimizer outside graph (0.05 ms/micro amortised) which fp8 storage
does not change (bf16 master + 0.25 ms/step refresh). Effort 2–4 days (fp8 table + fused dequant gather +
refresh + ckpt/export compat). **Defer** — ~2 % of the b2 capture deficit (8–10 GiB), no MFU impact.

## 7. Ranked GPU-costed validation list for next window (all A B B A; ~2 h ≈ ¥6)
| # | experiment | env / command | GPU-min | gate |
|---|---|---|---|---|
| 0 | cheaper prereq: `NANOV41_RMS_FUSED=1`+`NANOV41_SWIGLU_FUSED=1` A/B | b1 graph-only bench | 20 | −9–17 ms → +2.5–4 pp; land first |
| 1 | **E1 shape truth**: add d24 targets to `tools/bench_fp8_linear_v2.py` (8192,1024,2304 / 8192,2304,1024 / 8192,2048,1024 / 8192,768,5120), graph-timed | `python tools/bench_fp8_linear_v2.py --shapes d24 --json runs/r93/fp8_ffn_bench.json` | 10 | FFN total ≥1.25× and fwd+dgrad ≥1.5× else **stop** |
| 2 | E2 in-graph replay A B B A, phase-1 scope, `_WGRAD=0` | source env from a `tools/*.sh` runner, then b1 graph-only bench (`/root/r93_bench.sh --b 1 --seq 8192 --graph-only --skip-opt`) | 12 | paired replay ≤ −8 ms vs base |
| 3 | E3 G1 loss probe, 20 steps token-matched | `tools/r93_p0b_train.py --tokenizer runs/r92/vocab/K40000_s1280/tokenizer.json -- --config-json docs/r93/configs/d24_c1_dense40k.json --tokens … --grad-accum 256` | 25 | loss rel ≤1e-4, no NaN |
| 4 | E4 `_WGRAD=2` add-on (4 paired profiles) | 同上 + `NANOV41_LINEAR_FP8_V2_WGRAD=2` | 12 | further ≤ −2 ms replay |
| 5 | E5 G3 50-step A/B, final scope | p2_train two arms | 40 | ≥+2 % tok/s, +≥1.5 pp MFU, loss rel ≤1e-4 |
**Env gotcha:** `tools/r90_env_dense5090.sh` silently fails when sourced from `bash -c` (`dirname "$0"`="."
→ cannot source `r90_env5090.sh`), silently dropping the whole production kernel stack (caused 2 failed R93 bench
attempts). Always source it from a script file whose `$0` is in `tools/` (e.g. new `tools/r93_fp8_ffn_run.sh`).
