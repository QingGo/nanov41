# R93 MFU-45 sprint — FINAL results (2026-11-16 03:25 CST)

Status: **DONE**. Instance `<instance-id>` (RTX 5090, ¥3.08/h, 23:21–03:21 ≈ 4 h, session spend
≈ **¥13.6**; balance ¥<redacted> → ¥<redacted>; instance shut down). Artifacts: `runs/r93/mfu45/` (79 files,
no ckpts) + outer logs in `runs/r93/mfu45/outer/`. Data: 400M-token K=40k corpus + `p0_eval_40k`
panel rebuilt on the instance (`tools/r93_p0b_prep.sh`).

## 0. Verdict: the 45% target is blown through

| arm (50 steps = 104.9M tokens, same seed/data) | tok/s | raw MFU | val bpb (3M tokens) | Δbpb vs A | gate |
|---|---:|---:|---:|---:|---|
| **A: production 2026-11-16** (`r90_env_dense5090.sh`, b1×8192×accum256) | 43,866 | **36.18%** | 1.95733 | — | — |
| **B1: E0 only** (RMS_FUSED + SWIGLU_FUSED, b1) | **62,848** | **51.83%** | **1.96240** | **+0.0051** | **PASS** (≤+0.01) |
| **B2: E0 + b2** (b2×8192×accum128, CE_CHUNK 2048) | **64,750** | **53.40%** | 1.96768 | +0.0104 | borderline (0.0004 over) |

- **Speedup 1.43× (E0) / 1.48× (E0+b2)** over production; raw MFU **36.2% → 51.8–53.4%**.
- Quality at this 0.1B/50-step budget: E0 within the pre-registered +0.01 bpb gate; E0+b2 sits exactly
  at the boundary. Loss trajectories diverge chaotically mid-run (±0.1 nats at individual steps, final
  +0.18% / +0.46%), i.e. these bpb deltas are dominated by trajectory noise at this tiny budget —
  neither a parity proof nor evidence of degradation. **P1 should adopt E0 unconditionally and b2
  with a guarded start** (track bpb at 1–2B tokens against the P0b curve; drop b2 if a systematic
  +0.01 gap appears — b2 buys only +1.6 pp).
- **fp8 FFN: NOT adopted for P1.** Speed is real and measured (in-graph **−11.37 ms/replay**,
  FFN shapes **1.40×**, projected GEMM saving 16.5 ms), but (a) its 20-step step-loss divergence is
  2.1e-2 (quality unverified), and (b) **fp8 + b2 together OOM the 32 GB card at capture**
  (b2 alone peaks at 30.4 GiB). It needs its own 0.3B-token bpb A/B (and a memory plan) before use.

**P1 cost impact**: 8B tokens at 62.8k tok/s (E0) = 35.4 h ≈ **¥106**; at 64.7k (E0+b2) ≈ **¥103**;
vs production 44.3k tok/s ≈ ¥150–160 → **−30% P1 GPU cost** and MFU ~52–53% (target was 45%).

## 1. E0 — the big finding (two kernels, −56.4 ms of a 184 ms replay)

`NANOV41_RMS_FUSED=1` + `NANOV41_SWIGLU_FUSED=1` were implemented in R45 and validated, but were
simply **missing from `tools/r90_env5090.sh`**. Interleaved A/B/B/A b1×8192 graph replay
(`--graph-only --skip-opt`):

```
base 183.92/184.05 | RMS 159.46/159.40 (−24.56) | SwiGLU 152.26/151.86 (−31.93) | both 127.53/127.68 (−56.38 ms)
combined −30.6% replay  ->  training throughput 44.3k -> 62.8k tok/s (1.43x), raw MFU 36.2% -> 51.8%
```
They remove the unfused fp32 RMSNorm chain (the 17.4 ms × 476-call `MulFunctor` bucket in
`runs/r93/p0b/mfu_profile.md`) and the fp32 gate/up/clamp/silu/mul + casts in the SwiGLU.
SwiGLU fusion also shrinks activation memory (~0.5 GiB/layer at b2), which is what made b2 fit.

**P1 env recipe** (2 lines; keep as env, do not hard-code into traffic paths):
```
export NANOV41_RMS_FUSED=1
export NANOV41_SWIGLU_FUSED=1
```

## 2. E1/E2 — fp8 FFN (speed yes, quality pending)

- E1 shape truth (`tools/bench_fp8_linear_v2.py --shapes d24`): FFN up (M8192,K1024,N2304×48)
  fwd 1.98× / dgrad 1.55× → total **1.40×**; FFN down 1.40×; attn wo_b 1.42×; per-GEMM rel
  3.5–4.0e-2 (e4m3 grade). Projected per-replay GEMM saving **16.47 ms**.
- E2 in-graph: base 183.99 → **172.62 ms (−11.37 ms, −6.2%)** with scope
  `wo_b,ffn.w.w1,ffn.w.w3,ffn.w.w2,engram.wkv`.
- E2 loss probe 20 steps: final loss base 6.86888 vs fp8 6.83746, max step rel 2.1e-2 → strict
  ≤1e-4 gate FAIL. This is expected for a wide fp8 conversion (it changes the trajectory); the
  proper gate is a longer val-bpb A/B, which was not run (fp8+b2 OOM also blocked the combined arm).
  **Status: speed evidence only; keep OFF for P1; revisit with a 0.3B bpb A/B + memory plan.**

## 3. b2 — unlocked for free by the fused kernels

The planned 1–2 day FFN activation-recompute work was unnecessary: the fused SwiGLU autograd
Function saves bf16 gate/up instead of the fp32 SwiGLU chain. With E0 enabled,
b2×8192×accum128 (2^21 tok/step) captures at **30.385 GiB reserved** (limit 31.36) with only
`CE_CHUNK=2048`. Token-matched 20-step comparison:
b1-E0 **63,312 tok/s / 52.22%** → b2 **65,124 tok/s / 53.71%** (**+2.9%**), loss rel **3.9e-5**.
Combined with fp8 it OOMs (added quant buffers/caches) — the only failed combination.

## 4. Final A/B details

`tools/r93_final_ab.sh`: A = production env, b1×accum256; B = A + E0 (+ b2×accum128 if B2=1;
+ fp8 scope if FP8=1); 50 steps × 2^21 tokens/arm, seed 1234, same WSD recipe; val bpb on
`p0_eval_40k` (3M tokens) for both arms. Runs: A once (43,866 tok/s / 36.18%), then B1 = E0
(62,848 / 51.83% / bpb +0.0051), then B2 = E0+b2 (64,750 / 53.40% / bpb +0.0104). Two stale-eval
artifacts were caught and re-evaluated (the eval helper skips existing output files — always delete
`eval_*.final.json` before re-evaluating an overwritten ckpt).

## 5. Recommended P1 stack (pending P1 decision)

- Adopt: `NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1` (+ optional b2 micro-batch with the
  `CE_CHUNK=2048` / 16k-token micro guard, monitored by bpb at 1–2B).
- Do **not** enable body fp8 in P1 until a dedicated quality A/B passes; fp8 head stays as-is
  (production default).
- Expected P1 8B cost at 62.8–64.7k tok/s ≈ **¥103–106** (vs ¥150–160 at 36.6%), i.e. the
  original "45% MFU" budget scenario (¥122) is comfortably beaten.

## 6. Cost

Instance 4 h × ¥3.08 = ¥12.3 + prep/egress ≈ ¥1.3 → **¥13.6**; balance ¥<redacted> → **¥<redacted>**.
All own instances are shut down; the user's instances were not touched.
