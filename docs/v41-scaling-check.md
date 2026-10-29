# Scaling check: nano-v41 vs DeepSeek-V4.1-Flash decode FLOPs

Why this exists: the W2 efficiency numbers are measured on a shrunken V4.1-Flash proxy. If the
parameter down-scaling ratio differed from the decode-FLOPs down-scaling ratio, MFU measured on the
proxy would not transfer to the full model. This note records the check and the accounting
convention used for precision.

Companion tools: [`tools/v41_scaling_check.py`](../tools/v41_scaling_check.py) (this note's
numbers) and [`tools/mfu_accounting.py`](../tools/mfu_accounting.py) (precision-discounted MFU).

## TL;DR

| Scaling definition | d12 vs V4.1-Flash | d8 vs V4.1-Flash | Verdict |
|---|---:|---:|---|
| **Activated params (decode)** — 16B vs 70.65M / 39.50M | **226.5×** | **405.1×** | ✅ |
| **Raw decode FLOPs** — 32 GFLOP/token vs 141.3 / 79.0 MFLOP | **226.5×** | **405.1×** | ✅ matches activated params |
| **Discounted decode FLOPs** (report α = 1/0.5/0.25) — 11.05 GFLOP vs 91.7 MFLOP | **120.5×** | 240.6× | ✅ internally consistent, different from raw because the precision mixes differ |
| Total backbone params — 552B vs 193.4M / 81.6M | 2854.1× | 6763.0× | ❌ **not** a compute-scaling metric (see below) |
| Non-embedding params — 550.7B vs 94.1M / 15.4M | 5850.7× | 35689.6× | ❌ even further off (includes inactive experts) |
| Raw prefill FLOPs — 16 GFLOP/token (CED, 8B) vs 141.3 / 79.0 MFLOP | 113.2× | 202.5× | ⚠️ exactly half the decode ratio: the nano port does not implement CED prefill halving |

**Conclusion:** the check passes when "parameter scaling" means **activated (decode) parameters**:
d12 is a **226.5×** down-scale of V4.1-Flash in both activated parameters and decode FLOPs
(405.1× for d8). Total backbone parameters (552B / 193.4M = 2854×) must **not** be used for
compute-efficiency comparisons; the two definitions differ by 12.6× for d12.

## 1. Report facts used here

From the DeepSeek-V4.1-Flash technical report (§2.1, Figure 2) and the released
[`ref/config.json`](../ref/config.json):

| Fact | Value |
|---|---|
| Backbone parameters | **552B** (+196B Engram parameters) |
| Layers | 40 (20 causal-encoder + 20 decoder, CED) |
| Vocab / dim / moe_inter_dim | 129,280 / 5,120 / 2,304 |
| Routed experts | 384, 6 activated + 1 shared |
| Activated params per token | **8B during prefill, 16B during decode** |
| Precision accounting | BF16 / FP8 / FP4 weighted by **1 / 0.5 / 0.25** (Figure 2 caption) |
| Released precisions | linear layers `dtype=fp8`, experts `expert_dtype=fp4`, KV cache FP4 |

## 2. Derived breakdown

**V4.1-Flash** (from the released config; `tools/v41_scaling_check.py` prints this table):

| Component | Params | Per-token at decode |
|---|---:|---|
| Routed experts (all 384) | 543.582B | 8.493B activate (6/384) @ FP4 |
| Shared expert(s) | 1.416B | 1.416B @ FP4 |
| Attention + indexer + head + embedding | ~6.0B | ~6.091B @ FP8 (non-expert) |
| — of which embedding+head | 1.324B | head is computed; embedding lookup ≈ 0 FLOPs |
| **Activated total** | — | **16.000B ⇒ 32.00 GFLOP raw/token** |
| **Discounted (bf16-eq)** | — | **5.523B effective ⇒ 11.05 GFLOP/token** (experts 0.25×, rest 0.5×) |

**nano-v41** (params regression-tested in `tests/test_configs.py`; forward FLOPs from the
FlopCounter used for the W1/W2 numbers):

| Config | Total / non-emb params | Forward FLOPs/token | Activated params (= FWD/2) | Head params (vocab × dim) |
|---|---:|---:|---:|---:|
| d8 | 81.62M / 15.43M | 79.0 MFLOP | 39.50M | 33.10M |
| d12 | 193.41M / 94.12M | 141.3 MFLOP | 70.65M | 49.64M |

## 3. Why total parameters do not match

The nano configs are **compute-scaled** (activated-parameter-scaled), not total-parameter-scaled:

1. **The vocabulary is not scaled.** Both models use the same 129,280-token vocabulary. For d12,
   embedding+head = 99.29M = **51.3%** of all parameters; for V4.1-Flash they are 1.324B =
   **0.24%** of the backbone. The head *is* activated compute (70.3% of d12 forward FLOPs), so it
   counts in decode FLOPs, but it inflates total parameters disproportionately in the small model.
2. **Different expert activation rate.** d12 activates 2/16 routed experts (**12.5%**); V4.1-Flash
   activates 6/384 (**1.56%**). Inactive experts still count as total parameters (74.3M of d12's
   193.4M) and 535B of the full backbone, but they cost nothing per token.
3. **Different scope.** The nano port omits MTP/DSpark, Engram (196B params, sparsely activated) and
   the vision encoder, and does not implement the CED prefill shortcut.
4. Therefore `total_params_ratio ≈ (inactive-expert ratio) × (activation-fraction ratio) × …`,
   which is unrelated to per-token compute. The report itself compares models with **both** total
   and activated parameters precisely because they answer different questions.

## 4. Precision discounting (and why the two ratios can differ)

The report's convention weights operations by precision (BF16 1, FP8 0.5, FP4 0.25). Under that
rule:

- V4.1-Flash: experts FP4, everything else FP8 → effective activated params 5.523B → **11.05 GFLOP
  bf16-equivalent per decode token**.
- nano-v41 R50 best: only the vocabulary head GEMM runs in FP8 (α = 0.5), everything else BF16 →
  effective 45.83M params → **91.7 MFLOP bf16-equivalent per token**.

Ratio: 5.523B / 45.83M = **120.5×** = 11.05 GFLOP / 91.7 MFLOP. It matches, but it is not the
226.5× raw ratio because the full model uses more low-precision compute. This is exactly why raw
FLOPs and discounted FLOPs must not be compared across precision mixes — the same rule must be
applied on both sides.

## 5. MFU accounting convention

From now on the repository reports two numbers:

```
raw FLOPs (precision-inflated)  MFU_raw = Σ F_p / (t · P_bf16)
discounted (V4.1 convention)    MFU_eq  = Σ α_p F_p / (t · P_bf16),  α = {bf16:1, fp8:0.5, fp4:0.25}
```

Both use the **fixed** RTX 5090 bf16 peak (209.5 TFLOP/s). No per-run mixed-precision peak is used,
so switching precision only helps when the wall time actually drops. For R50 best (60.43 ms/step):

| Metric | Value |
|---|---:|
| Raw FLOPs / step | 6.945 TFLOP → raw MFU **54.86%** |
| Discounted FLOPs / step | 4.504 TFLOP (×0.649) → discounted MFU **35.58%** |
| 60% wall, current mix | 35.83 ms |
| 60% wall if MoE also FP8 (what-if) | 29.56 ms (at the same 60.43 ms the discounted MFU would be 29.35%) |

Raw MFU is still useful as the **cross-round throughput** metric (it matches how the W2 rounds
were tracked), but it is precision-inflated and must be labelled as such; precision decisions use
the discounted number. Recompute with:

```bash
python3 tools/mfu_accounting.py --preset r50_b16
python3 tools/mfu_accounting.py --preset r50_b16_all_fp8_moe
python3 tools/mfu_accounting.py --time-ms 60.43 --flop fp8:4.882 --flop bf16:2.063
python3 tools/v41_scaling_check.py           # tables above
```

## 6. Assumptions and limitations

- The 16B decode activation is split into routed-active / shared / non-expert using the released
  config (6 of 384 experts; 1 shared expert). The report does not publish the per-component split,
  so the FP4/FP8 assignment is an inference from `dtype=fp8` + `expert_dtype=fp4`; the effective
  number is therefore an estimate (±a few %).
- Attention FLOPs over the context are not added: the report shows decode FLOPs are nearly constant
  with context (sparse CSA2), and at short context `2 × activated params` dominates. At long context
  an attention term applies to both models; the ratio stays approximately unchanged because both
  use sparse attention, but it is not included here.
- nano forward FLOPs come from the W1/W2 FlopCounter: 141.3 MFLOP/token (d12), 79.0 MFLOP/token
  (d8). Their ratio to 2 × activated params is 1.00–1.02 (the small gap is indexing/HC elementwise
  work not counted as params).
- Prefill: V4.1-Flash activates 8B/token (CED halves prefill); the nano port computes every layer
  for every token, so its prefill and decode per-token costs are identical (141.3 MFLOP). A
  prefill-scaling comparison needs the CED shortcut first.
