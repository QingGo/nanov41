# nanov41 optimization playbook

**Audience:** an autonomous coding/perf agent working on `nanov41`.
**Scope:** training-step efficiency on the RTX 5090, `d12`, `b=16 × seq=1024` (the W2 target),
with `b=8` as the secondary shape. **Primary goal: 60% precision-discounted MFU** (V4.1
convention, fixed 209.5 TF bf16 denominator). Raw bf16-MFU is a diagnostic only. This is **not**
a guide to model quality, inference serving, or “recursive self-improvement”.

**Status of this document:** operational playbook derived from the measured W1/W2 rounds and the
external `signal_bank` study. It is deliberately conservative: it tells the agent what to measure,
what not to redo, and when to stop.

**One-sentence rule:**

> Change one thing, measure the same harness interleaved against the current best, require parity,
> and only then keep it.

---

## 0. Current best and hard guardrails

### Current best measured

| Item | Value |
|---|---|
| target | `d12`, bf16, `b=16 × seq=1024`, graph replay, AdamW outside graph |
| best step | **60.43–60.48 ms** (R50 + MoE NOPAD) |
| raw bf16 MFU (diagnostic) | **54.81–54.86%** (`TFLOP/s ÷ 209.5`) |
| **discounted MFU (primary)** | **35.6%** (fp8 head, bf16 rest; V4.1 convention) |
| **project target** | **60% discounted MFU** |
| **60% discounted wall** | **35.83 ms** (current 60.43 ms; **1.69× away**) |
| 60% raw wall (not the target) | 55.25 ms |
| 3-step parameter parity | **0.0** |
| b=8 | 34.87 ms, 99.58 TFLOP/s, 47.53% raw MFU |
| device time / gap | 57.63 ms device, **3.29 ms gap** (graph replay) |
| kernels/replay | ~3,235 |

Evidence: `runs/r50_best_b16.json`, `runs/r50_best_b8.json`, `runs/r55_tail_census_5090.json`,
`runs/r58_tile_sweep_5090.json`, `docs/benchmarks.md`.

### R69 update (2026-11-05)

`bench_graph_train.py` can no longer capture the full step on torch 2.12 with the old protocol
(a prior `.backward()` leaves autograd stream state that trips `cudaErrorStreamCaptureImplicit`);
it now captures **before** the eager baseline using the R65 GraphStep "grad mode" recipe. With that
fixed harness, the W2 step was pushed **59.90 → 58.14–58.78 ms (−2.9 to −4.6%, same-round)** by five
opt-in flags — `MHC_V4=1 V4_MODE=split`, `MOE_LAYOUT=fast`, `ATTN_DKV_BF16=1`, `MOE_DWS_INLINE=1`,
`MOE_TILE=64,64,64,4,3` — and by another **−2.4%/token** with 4× gradient accumulation (the
optimizer alone is 4.39 ms = 7.5% of the step). 3-step parity stays 0.0. A follow-up same-host 2×2
shows **torch 2.14.1+cu130** (R50/R63 known-good stack, `tools/setup_torch214.sh`) is another
**−1.9%** on the R69 config (device-side; eager cost unchanged), giving a best of **55.83 ms per
16K tokens = 293.5k tok/s / 59.4% raw MFU** with accumulation. Accepted/rejected detail, kernel
census, the torch-version matrix and the remaining ~2–5% to the 300k tok/s ceiling are in
`docs/benchmarks.md` §3 ("R69") and `runs/r69_mfu_5090.json`; the R69 flags are in
`tools/final_env.sh`.

### Hard guardrails

1. **Never optimize without a baseline.** Reproduce the current harness first; record the exact
   command, torch version, GPU, host, and commit.
2. **Parity is a gate, not a metric.** Any change that alters the training step must pass
   `parity.py` and the 3-step parameter-parity check used by the graph harness. A faster step with
   broken parity is a regression.
3. **Do not edit `ref/` or `nanov41/_ref_patched_model.py` by hand.** Use
   `tools/make_patched_ref.py`; run `--check`.
4. **One variable per experiment.** No “fusion + fp8 + tile change” bundles. If it wins, you will
   not know why; if it loses, you will not know what to revert.
5. **Interleaved A/B, same session.** Host CPU contention changes step time by ~15%; cross-process
   or cross-host comparisons are invalid. Use `tools/ab_bench.py` or an interleaved in-process
   harness; median of ≥3 repetitions.
6. **New code is opt-in and fallback-safe.** Default OFF; env flag or explicit `install()`; keep a
   pure-torch/reference path. Do not silently change the default training path.
7. **Primary goal metric is 60% discounted MFU.** `step_ms` is the operational metric; every
   candidate must report both raw and discounted MFU. Discounted MFU uses α = bf16 1 / fp8 0.5 /
   fp4 0.25 and the fixed 209.5 TF denominator (`tools/mfu_accounting.py`). Do not substitute
   raw bf16-MFU or top-1 for the target.
8. **No LLM aggregation, no diagnosis prose, no RSI framing.** See §6.

---

## 1. Where the time actually goes

R55 tail census at R50-best (b=16, seq=1024, graph replay; `runs/r55_tail_census_5090.json`):

| segment | ms | share of device |
|---|---:|---:|
| head (K1/K2/K3) | 13.93 | 24% |
| MoE | 9.88 | 17% |
| mHC | 9.48 | 16% |
| attention v2 kernel | 5.70 | 10% |
| small cutlass GEMMs | 6.47 | 11% |
| elementwise/copy/sort/fill | 5.80 | 10% |
| rms/rope | 1.52 | 3% |
| misc/unattributed | 4.85 | 8% |
| **device total** | **57.63** | |
| gap (CPU/launch) | 3.29 | |

Implications:

- After CUDA-graph capture, the step is **device-bound**, not launch-bound. Do not spend the next
  round on “reduce Python overhead” unless a fresh profile shows `gap_ms` > 5% of step.
- The largest *remaining* opportunities are **small GEMMs + elementwise** (12.27 ms combined),
  **mHC** (9.48), **MoE dW** (~4.2), **attention backward** (5.70), and the **head epilogue**
  (13.93 vs ideal ~11.65).
- **Target math (discounted).** R50 is **35.58%** discounted at 60.43 ms; 60% discounted at the
  same mix is **35.83 ms**. That is **24.6 ms / 40.7% / 1.69×** away — not 5.2 ms. R58’s
  component-floor estimate on sm_120 is **42–46 ms (~48–53% discounted)**, so tile/fusion work
  cannot reach 60% on this platform. 60% needs a different platform (grouped FP8/FP4 on
  H800/B200), an algorithmic FLOP reduction, or a precision change that measurably improves
  discounted MFU. The 55.25 ms raw-60% wall is **not** the goal; raw MFU is diagnostic.

---

## 2. Mandatory optimization loop

Use this loop for every candidate. Do not skip steps.

### Step 1 — Reproduce the baseline

```bash
cd /path/to/nanov41
git status --short
python3 parity.py --dtype bf16
python3 tools/make_patched_ref.py --check
python3 -m pytest -q -m "not slow"
python3 tools/mfu_accounting.py --preset r50_b16   # discounted MFU = 35.58%, 60% wall = 35.83 ms
```

Record the exact harness and the discounted-MFU baseline. **Important:** the published 60.43 ms uses the internal graph-training
driver, which is not versioned. The public `bench_mfu.py` measures the eager path (R50 eager was
106.88 ms). If the graph driver is unavailable, use `bench_mfu.py` or
`tools/ab_bench.py --compile` as the A/B harness and label results as eager/compile-proxy — never
compare them directly to 60.43 ms.

### Step 2 — Decompose before optimizing

Run the current best with profiling and write down:

```text
step_ms = device_ms + gap_ms
device breakdown by segment
kernel count/replay
```

- If `gap_ms / step_ms > 5%`: fix host/launch first (graph already captured; check graph safety,
  optimizer placement, Python work outside graph).
- If `device_ms` dominates: pick a segment from §1, not “the model” in general.

### Step 3 — Pick one ranked target (§3)

Write the hypothesis in one sentence and the expected gain. If you cannot state a mechanism, do not
start.

### Step 4 — Implement opt-in, with fallback

- Env-gated (`NANOV41_*`) or an `install()` function; default path untouched.
- Keep the old path available for A/B.
- If the change is numerical, state the tolerance and parity expectation up front.

### Step 5 — Correctness gate

```bash
python3 parity.py --dtype bf16
python3 -m pytest -q -m "not slow"
# plus the graph harness 3-step parity if available
```

- Any parity failure → revert; do not “fix” parity by loosening the test unless the change is
  explicitly allowed to be non-bitwise and the tolerance is documented.
- The projection-fusion path is a precedent: non-bitwise (7.1e-5 relative loss), explicitly opt-in,
  default OFF.

### Step 6 — Interleaved A/B

- Same process, same session, same host; alternate baseline/variant; warmup; 3 rounds × 3 steps;
  median.
- Include `device_ms`, `gap_ms`, kernels/replay if available.
- Acceptance threshold: **median step improvement > 1%** and larger than run-to-run noise; parity
  passes. A 0.2–0.5 ms change on a 60 ms step is within host noise unless repeated.

### Step 7 — Record evidence

Write a JSON under `runs/` (or local if not yet public) with:

```json
{
  "id": "rXX_short_name",
  "date": "YYYY-MM-DD",
  "commit": "git sha",
  "device": "RTX 5090",
  "torch": "2.14.1+cu130",
  "shape": "b=16 seq=1024",
  "baseline_ms": 60.43,
  "variant_ms": 0.0,
  "delta_ms": 0.0,
  "delta_pct": 0.0,
  "raw_mfu_pct": 54.86,
  "discounted_mfu_pct": 35.58,
  "mix": {"fp8": 4.882, "bf16": 2.063},
  "parity_param_rel": 0.0,
  "device_ms": 0.0,
  "gap_ms": 0.0,
  "kernels_per_step": 0,
  "env": {},
  "verdict": "accept|reject|negative",
  "notes": "mechanism, caveats"
}
```

For an accepted result, also update `docs/benchmarks.md` and (if public evidence) whitelist the
JSON in `.gitignore` + add a row to `runs/README.md`.

### Step 8 — Stop rules

- 3 independent attempts on the same target with no >1% gain → write a negative result and move on.
- Any parity/correctness failure → revert immediately.
- Host noise > 10% → do not trust single-process timings; use interleaved A/B or a quieter instance.
- You are about to tune a tile knob → stop; R58 exhausted the tile space.

---

## 3. Ranked next bets for 60% discounted MFU

The old plan was ranked for **raw** 60% wall (55.25 ms). That is not the target. For **60%
discounted MFU** the wall target is **35.83 ms** at the R50 precision mix, and the gap is **1.69×**.
The ranking below is therefore different: first decide whether 60% is reachable at all on the
target hardware, then do the work that actually moves *discounted* MFU.

### P0 — Feasibility gate (do this before optimizing kernels)

Answer these before writing code:

1. **Hardware and denominator.** Is the target 60% discounted MFU on the RTX 5090 with the fixed
   209.5 TF bf16 denominator, or is a different platform allowed? If a different platform is
   allowed, re-baseline there first; do not mix 5090 and H800/B200 numbers.
2. **Target time.** At the R50 mix, 60% discounted = **35.83 ms**. Current = 60.43 ms. Required
   reduction = **24.6 ms (40.7%, 1.69×)**.
3. **Floor.** R58’s component-floor estimate on sm_120 is **42–46 ms**, i.e. **~48–53%
   discounted**. That is below 60%. If the answer to (1) is “5090 only”, then 60% cannot be reached
   by tile/fusion/component tuning alone; pick one of:
   - **platform route**: grouped FP8/FP4 kernels on H800/B200 (re-baseline, re-define denominator);
   - **algorithmic route**: reduce FLOPs/token or change the computation while preserving model
     quality (this is a model/algorithm change, not just a kernel optimization);
   - **accept and report** that the 5090 path is short of 60%, with the measured floor.
4. **Do not start micro-optimization until this gate is answered.** A 1 ms kernel win is useful but
   cannot close 24.6 ms; pretending otherwise wastes the agent's budget.

### P0 — Discounted-positive precision changes

Lower precision is **not automatically** a discounted-MFU win. The metric weights FP8/FP4 FLOPs by
0.5/0.25 while using the fixed bf16 denominator; a precision change must reduce wall time enough to
compensate. **The only valid test is measured discounted MFU**, not raw MFU.

Current evidence:

- FP8 head mix: **61.12 ms, 35.18% discounted** vs all-bf16: **86.32 ms, 38.41% discounted**.
  The FP8 head is the throughput choice, but it **lowers** discounted MFU; it is not a 60% route
  as-is.
- FP8 MoE v1 (per-tile quant): no step gain; discounted MFU is worse (60.88–60.94 ms vs 60.94).
- All-FP8 MoE what-if at the same 60.43 ms: **29.35% discounted**; 60% wall **29.56 ms** — the
  lower discount ratio makes the target harder, not easier.
- Grouped FP8 dW prototype (R59): 1.27–1.70× vs Triton bf16, but per-expert accuracy 3.7e-2 needs
  a scale strategy. Only pursue if it wins **discounted MFU** end-to-end.

**Do:**

- For every precision variant, report `raw_mfu_pct`, `discounted_mfu_pct`, wall time, and the mix.
- Reject a variant if discounted MFU does not improve, even if raw MFU does.
- Prefer precision changes where the kernel speedup is large enough that the 0.5/0.25 discount is
  overcome end-to-end; verify empirically, do not assume.
- Keep a bf16 fallback and parity.

**Expected:** uncertain; this is a research route, not a guaranteed 60% path.

### P1 — Component wall-time work (necessary, not sufficient)

These are the same R55 targets, but they are **not** enough for 60% by themselves. Each 1 ms saved
at ~60 ms is only ~0.6 pp discounted MFU; 24.6 ms is needed. Use them to improve the baseline and
to build toward the floor, not as a claim of 60%.

| priority | target | measured tail | realistic gain | note |
|---|---|---:|---:|---|
| P1 | small GEMMs + elementwise/copy/sort | 12.27 ms | 1–3 ms | projection fusion R56/R57 only −0.4–0.5 ms; remaining pointwise chains |
| P1 | mHC (`_pm_pre` + `_post`) | 9.48 ms | 1–3 ms | bandwidth ~3×; official mHC slower; numerics sensitive |
| P1 | MoE dW (large-K, pre-quantized FP8) | ~4.2 ms | 1–2 ms | FP8 v1 failed; only large-K dW; accuracy gate |
| P1 | attention v2 backward (`fast_attn_v3` dkv bf16) | 5.70 ms | 0.9–1.5 ms | opt-in; official sparse attn has no bwd |
| P2 | head epilogue (K1) | 13.93 ms | 1–2 ms | FP8 head is throughput choice; tile knobs exhausted |

If even the optimistic sum leaves >10 ms of the 24.6 ms gap, stop and report that component
tuning cannot reach 60%; go back to the P0 feasibility gate.

### P2 — Algorithmic FLOP reduction

This is the only route that can change the discounted numerator itself:

- reduce FLOPs/token (e.g. fewer activated experts, sparser attention, lower-rank paths) **while
  preserving loss/quality**;
- measure both quality and discounted MFU; a faster but worse model is not a win;
- treat this as a model change, not a kernel optimization; write a separate experiment plan and
  parity/quality gate.

### P3 — Raw 60% wall (not the goal)

Do not chase the 55.25 ms raw-60% wall. It is a useful diagnostic for throughput, but the project
target is 60% **discounted**. Report raw MFU only alongside discounted MFU.

## 4. Known negative results — do not redo

| Direction | Evidence | Verdict |
|---|---|---|
| Head/mHC tile-knob sweep (15 variants) | `runs/r58_tile_sweep_5090.json` | none beats defaults; space exhausted |
| MoE FP8 v1 (per-tile activation quant) | `runs/r50_fp8.json` | no gain: K=384 latency-bound; quant eats savings |
| Official TileLang mHC | `runs/r55_tail_census_5090.json` | slower on 5090 than our Triton path |
| Official sparse attention | `runs/r55_tail_census_5090.json` | fwd-only, no training backward; not a drop-in |
| AdamW inside CUDA graph | `runs/r55_tail_census_5090.json` | loses ~0.1 ms and breaks parity; keep outside |
| Projection fusion as a default | `runs/r56/r57` | only −0.4–0.5 ms, non-bitwise; keep opt-in |
| FP8 head as a 60% discounted route | `runs/r54_precision_mix_5090.json` | 61.12 ms / 35.18% discounted; lower than all-bf16 38.41% and far from 60% |
| All-bf16 as a cheap 60% hedge | `runs/r54_precision_mix_5090.json` | 86.32 ms / 38.41%; 60% discounted needs 55.25 ms, still far |
| Tile/fusion-only path to 60% discounted | `runs/r58` | not reachable on sm_120; component floor 42–46 ms = 48–53% |

Also: `runs/r51` shows FP8 expert GEMM is **not** faster for the nano expert shapes but is
1.76–2.43× faster at full V4.1 shapes — do not generalize a full-scale result to nano, or vice
versa.

---

## 5. Metrics and acceptance

### Primary (project goal)

- **Discounted MFU at b=16, seq=1024**, V4.1 convention, fixed 209.5 TF denominator.
  Target: **60%**. R50: **35.58%**; 60% wall: **35.83 ms**; gap: **1.69×**.
- Operational metric: median `step_ms`, same harness, interleaved, ≥3 rounds × 3 steps.
- For every candidate, report both `raw_mfu_pct` and `discounted_mfu_pct`. A step-time win that
  lowers discounted MFU is not progress toward the goal.
- `b=8` separately; do not mix shapes in one ranking.
- Acceptance for an experiment: >1% median step improvement (or a measured discounted-MFU win)
  and parity passes. Acceptance for the **goal** is separate: measured discounted MFU must move
  toward 60%; no single micro-opt is assumed to close the 1.69× gap.

### Secondary

- `device_ms`, `gap_ms`, `device_busy_pct`, `kernels_per_step`.
- Raw bf16-MFU: `TFLOP/s ÷ 209.5`; label as precision-inflated; **diagnostic only**, not the goal.
- Discounted MFU: `Σ α·FLOPs ÷ (t · 209.5)`, α = bf16 1 / fp8 0.5 / fp4 0.25. Recompute with
  `python3 tools/mfu_accounting.py --preset r50_b16`.
- No top-1, no answer-key metrics, no “accuracy” on a synthetic batch as a proxy for speed.

### Noise discipline

- Host CPU changes a launch-bound step by ~15% (see `docs/benchmarks.md` p0 rerun).
- One process = one session; compare only within it.
- If the host is noisy, take more rounds, not more one-off runs.
- Record the GPU model, torch version, driver, host, and commit in every JSON.

---

## 6. LLM-assisted optimization policy

The external `signal_bank` study (96 states, 4 actions, 9 context arms, 2 flash models) is useful
only for **how to use an LLM**, not as a claim that an LLM beats a measured rule. The actionable
findings:

1. **Action labels are the active ingredient.** `search_action` (labels, no gains) ≈ full search;
   `search_cfg` (config/time, no best action) collapses; shuffled labels collapse.
2. **Measured gains are optional; labels are not.** Do not spend tokens on long gain prose if the
   action label is already present.
3. **Diagnosis/profiler prose is weak.** `prof` and `diagn_fix` are small/nulled; a diagnosis
   without an actionable knob is not a signal. Do not build a diagnosis pipeline for nanov41.
4. **LLM selection is not better than simple aggregation.** In-sample majority+hand beat the LLM;
   held-out with the *same* hand rule was non-significant (no reversal); LLM q was high, fallback p
   weak.
5. **Top-1 can be inflated by answer-bearing context.** Use step time/regret, not top-1.
6. **Coverage matters.** Hamming-1 retrieval covered no mHC-best states; if the action space does
   not contain the best action, a better model cannot help.
7. **Different models may have different ceilings**, but do not assume a signal transfers; test a
   capability ladder and held-out configs.

### Allowed use of an LLM in nanov41

- **Proposal, not aggregation.** LLM proposes one action or one code change; deterministic
  vote/rule selects; fallback to the hand rule on invalid/low-confidence output.
- **Action prompt:** give the 3 nearest measured states, each with state summary, action label,
  and measured gain; allowed action IDs only; require JSON:
  `{"action_id": "...", "reason": "...", "confidence": 0..1}`.
- **Code prompt:** reference kernel + shape + tests + current ms; require code only; run tests;
  benchmark; best-of-N with a correctness gate.
- **Held-out validation:** fit retrieval/rule on train configs; evaluate on held-out configs; do
  not tune on the evaluation table.
- **No human labels**, no answer-key row, no top-1 target.

### When not to use an LLM

- If the action space is 24 configs and a measured table exists: exhaustive/BO/random is cheaper
  and more reliable.
- If you cannot measure action labels/gains: a text-only LLM has no signal.
- If the LLM does not beat majority/rule on held-out: stop; do not add complexity.

---

## 7. Evidence and reproducibility checklist

Every accepted change must leave:

- [ ] exact command(s) and env flags;
- [ ] commit sha;
- [ ] GPU, torch, driver, host;
- [ ] baseline and variant medians, all rounds;
- [ ] `parity.py` result and 3-step param parity;
- [ ] `device_ms`/`gap_ms` if profiled;
- [ ] JSON under `runs/` (or local until public);
- [ ] one-row update to `docs/benchmarks.md` if it is a new best;
- [ ] negative result written down if rejected.

Do not put private paths, API keys, or host credentials in the repo.

---

## 8. First five tasks (do in order)

1. **Confirm the target and denominator.** Is 60% discounted MFU on the 5090 with the fixed
   209.5 TF denominator, or is another platform allowed? If another platform, re-baseline there.
   Write the answer down before touching code.
2. **Reproduce the baseline and compute its discounted MFU.** Run the harness; record step time,
   raw MFU, discounted MFU, and the precision mix. `tools/mfu_accounting.py --preset r50_b16`
   should reproduce 35.58% and 35.83 ms at 60%.
3. **Establish the floor / route.** Confirm the R58 component-floor estimate (42–46 ms = ~48–53%
   discounted) on this host. Decide: platform route, algorithmic route, or report the 5090 shortfall.
4. **Run the precision gate.** Measure FP8 head / FP8 MoE / all-bf16 variants with **discounted**
   MFU, not raw. Do not assume lower precision helps; current FP8 head lowers discounted MFU.
5. **Only then attack one component** (small GEMM/pointwise or mHC) for wall-time relief. Report
   both metrics; if the candidate cannot plausibly contribute to the 24.6 ms gap, stop and return
   to the feasibility gate.

## 9. Escalation / stop conditions

Stop and report if:

- parity/correctness fails after a change and cannot be fixed without loosening a documented
  tolerance;
- 3 attempts on the same target produce no >1% gain;
- the measured host noise is larger than the candidate effect;
- the only path to the target is a platform/algorithm change (e.g. 60% **discounted** on sm_120; R58 floor is 48–53%);
- an LLM-based policy does not beat majority/rule on held-out.

**Do not** turn this into an RSI project. The model is frozen, the action is a config/kernel flag,
and there is no retention, capability gain, counterfactual, or compounding. The value here is
measurement discipline, not self-improvement.

---

*Last updated: 2026-10-31. Based on R42–R60 evidence in `runs/`, `docs/benchmarks.md`, and the
external `signal_bank` study. Update this file when a measured result changes the ranking.*
