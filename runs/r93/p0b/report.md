# R93 / P0b — dense compute-optimal calibration (d24_c1, K=40k)

_generated 2026-11-15T08:57:29Z by `tools/r93_p0b_analyze.py`_

## 0. Verdict

- **PASS** — primary gate: final bpb 1.3141166612143378 vs 1.578 ±0.05
- steady tok/s **44.3k**, raw MFU 36.56% (real-fwd-adjusted 37.56%; +2.7% for Engram wkv, which the repo accounting convention excludes), peak reserved 27.777 GiB
- steps 286 x 2,097,152 tok/step = 0.600B tokens; session spend ~¥11.41 / ¥19.0
- abort checks: NaN events 0, non-finite loss False, over budget False

## 1. Model / accounting

- d24_c1: dim 1024, 24 layers, dense SwiGLU h=2304, seq 8192, V=41,280, Engram ON (C40 = 32410)
- fwd FLOPs/token (repo accounting) **576.0M**; real + Engram wkv = 591.7M; N_act 287.98M; bpb gate line 0.748*(C/4.33e19)^-0.20

## 2. Throughput / memory

| item | value |
|---|---|
| smoke 5090 (b1/accum256) | 44460.0 tok/s, MFU 36.67%, peak 27.777 GiB, rc=0 |
| smoke 5090_b1a256.series (bNone/accumNone) | None tok/s, MFU None%, peak None GiB, rc=None |
| smoke 5090_b1a256.train (bNone/accum256) | None tok/s, MFU None%, peak None GiB, rc=None |
| full run (5090) | 44.3k tok/s steady, MFU 36.56% (adj 37.56%), peak reserved 27.777 GiB, rc=0, 226 min |
| train.json | tokens/s 44334.3, raw MFU 36.56%, 13528.7s |

## 3. val bpb vs the gate

| ckpt | tokens | val bpb (p0_eval_40k) | gate = line(D) | delta | note |
|---|---:|---:|---:|---:|---|
| p0b_d24_c1.final | 0.600B | 1.3141166612143378 | 1.5779682217095496 | -0.2639 | PRIMARY (WSD cooldown complete) |
| p0b_d24_c1.t200000000 | 0.200B | 1.5580958129311666 | 1.9655832478045647 | -0.4075 | mid-run, no cooldown yet -> expected above line |
| p0b_d24_c1.t400000000 | 0.400B | 1.3913311914489235 | 1.711139603581689 | -0.3198 | mid-run, no cooldown yet -> expected above line |

### 3-point bpb law fit  bpb = a + b·D^(−c)

- fit: a=1.2934, b=191905753862428.4688, c=1.789 (SSE 7.11e-04) on ['0.20B:1.558', '0.40B:1.391', '0.60B:1.314']
- **extrapolated bpb at D=6.0B: 1.294**  [1.261, 1.329] (10-90% MC, per-point noise ±0.03)
- comparison: §8.1 line value at that budget = 0.996; prior-anchored long-horizon estimate ≈0.996 (D=6.0B). Use the fit as the defensible P0b-based extrapolation; model-form uncertainty (3 points, 10x extrapolation) is the dominant error, see the fixed-exponent robustness rows below.

| exponent c (fixed) | a | b | bpb @ target |
|---:|---:|---:|---:|
| c=0.2 | 0.3127 | 56.89 | 0.943 |
| c=0.35 | 0.7917 | 616 | 1.025 |
| c=0.5 | 0.9825 | 8146 | 1.088 |
| c=0.6 | 1.0563 | 4.808e+04 | 1.122 |
| c=0.8 | 1.1479 | 1.802e+06 | 1.175 |
| c=1 | 1.2021 | 7.171e+07 | 1.214 |

- MC band (±0.03 per-point noise): [1.261, 1.329] (10-90% MC, per-point noise ±0.03)

Train-CE cross-check: CE 3.975 nats/token at 600M tokens ⇒ bpb 1.291 using canonical 4.4426 B/token (train loss, not val bpb). P0b is far over-parameterised vs the 6B-token frontier (2.1 tokens/param here vs ~20.8 at P1), so P0b over-performance does NOT transfer 1:1 to P1 — the 3-point fit extrapolation is the defensible estimate.

Loss curve (dense model, WSD):

| tokens B | loss | tok/s | lr |
|---|---:|---:|---:|
| 0.002 | 10.834836 | 44578.8 | 9.666666666666665e-05 |
| 0.044 | 7.649954 | 44533.9 | 0.0020299999999999997 |
| 0.086 | 6.58212 | 44537.9 | 0.0029 |
| 0.128 | 5.567746 | 44558.4 | 0.0029 |
| 0.170 | 5.015753 | 44558.0 | 0.0029 |
| 0.212 | 4.743159 | 44552.3 | 0.0029 |
| 0.254 | 4.547266 | 44558.9 | 0.0029 |
| 0.296 | 4.427553 | 44323.5 | 0.0029 |
| 0.338 | 4.325643 | 44355.6 | 0.0029 |
| 0.380 | 4.257987 | 44383.2 | 0.0029 |
| 0.422 | 4.214708 | 44237.9 | 0.0029 |
| 0.463 | 4.134942 | 44268.2 | 0.0029 |
| 0.505 | 4.070924 | 44290.7 | 0.0029 |
| 0.547 | 4.035083 | 44314.8 | 0.002831687388393557 |
| 0.589 | 3.965096 | 44330.8 | 0.0005560985492541349 |

## 4. Artifacts

- dir `/root/nanov41/runs/r93/p0b`: `PREP_DONE.json`, `smoke_<profile>.json`, `full_<profile>.json`, `p0b_*.train.json`, `p0b_*.series.json`, `p0b_*.log`, `eval_*.json`, `summary.json`, `report.md`
- checkpoints: `/root/nanov41/runs/r93/p0b/p0b_d24_c1.t200000000.pt`, `/root/nanov41/runs/r93/p0b/p0b_d24_c1.t400000000.pt`
- final ckpt eval: `/root/nanov41/runs/r93/p0b/eval_p0b_d24_c1.final.json`

## 5. Notes / caveats

- Gate line is the R90/R91 R90-era interpolation `0.748*(C/4.33e19)^-0.20` with C=3*fwd*D; it carries ±2x FLOPs uncertainty (gamma≈0.20 fitted from 2 anchors), so ±0.05 bpb is a health band, not a precise prediction.
- The 0.2/0.4B checkpoints sit inside the WSD stable phase (no cooldown), so they should read a few 0.01-0.06 bpb ABOVE the line; only the final (cooldown) point is the gate.
- Engram table params (220.0M) are memory-only (not in fwd FLOPs); the repo accounting convention excludes the Engram wkv too, so p2_train's raw MFU is ~2.7% optimistic — both numbers reported above.

