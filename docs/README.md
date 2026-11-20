# docs/

Version-controlled documentation for nanov41. The curated set covers architecture, benchmarks,
design notes and the R90–R95 single-GPU sprint reports (see the last section). Everything else
under `docs/` is internal development material and is **not** part of the public repository
(see `.gitignore`): working notes, experiment logs and ops runbooks stay local.

| Document | Contents |
|---|---|
| [`architecture.md`](architecture.md) | Model architecture as expressed by the official code, config invariants, and the five patches used by the training port |
| [`benchmarks.md`](benchmarks.md) | Curated W1 benchmark numbers (CPU/Mac parity + RTX 5090 throughput) with pointers to raw JSON evidence |
| [`v41-scaling-check.md`](v41-scaling-check.md) | Scaling sanity check against DeepSeek-V4.1-Flash decode FLOPs, plus the precision-discount convention used for MFU |
| [`optimization-playbook.md`](optimization-playbook.md) | Agent-facing optimization loop: current best, ranked tail targets, negative results, parity/A/B rules, LLM-use policy |
| [`compare-nanogpt.md`](compare-nanogpt.md) | nanov41 vs Karpathy's nanoGPT: parameters, tokens, FLOPs/token, throughput and cost, with assumptions |
| [`compare-nanochat.md`](compare-nanochat.md) | nanov41 vs Karpathy's nanochat (GPT-2-capability full pipeline): model/data/compute/cost comparison |
| [`engram-integration-plan.md`](engram-integration-plan.md) | Engram feasibility **and policy** (mandatory in the OPD main path): current code, tokenizer/training work, EngramDB / engram-peft |
| [`status-2026-11-07.md`](status-2026-11-07.md) | Project status snapshot: completed phases, trained data, measured training/eval/inference performance, remaining training/inference/eval work |
| [`decode-200tps-plan.md`](decode-200tps-plan.md) | Decode bottleneck analysis → **M1** graph-safe runtime (306–314 tok/s, bit-identical), **M2** GPU-busy kernels (537.6 tok/s ceiling / 487.5 verified), **M2d** fused MoE/RMSNorm/head (**812 tok/s ceiling, 790 verified**), **M3** continuous batching + one graph per batch size (**8.8 k tok/s @ B=16**, bit-identical per row) |
| [`inference-kernels.md`](inference-kernels.md) | Native inference kernels: Engram decode hash + FP8/E8M0 gather, MoE small-M and fused 4-kernel decode MoE, decode attention fast path, batched decode kernels + batch-invariant projections, real-checkpoint quality, launch-bound profile + roadmap |
| [`vllm-conversion.md`](vllm-conversion.md) | vLLM DeepSeek-V4.1 checkpoint conversion prototype: Engram FP8/E8M0 table export, key/config mapping, GPU load-test plan |
| [`generation-kv.md`](generation-kv.md) | KV-cache generate (parity-checked), 0.6B-token/parameter ratios, expected scaling effect, vLLM assessment |
| [`p2-ab-report.md`](p2-ab-report.md) | P2 capacity gate: 0.6 B-token A/B of Engram table sizes and d12e32 vs d12; T0 BPB result and decision |
| [`engram-design.md`](engram-design.md) | Engram sizing, layer placement and MoE relationship (P0/P1-CPU implemented): knob/cost formulas, d12/d16/d20 geometry, V4.1 ratios, memory/optimizer budget, P0–P2 validation plan |
| [`product-vs-nanochat.md`](product-vs-nanochat.md) | Positioning as a low-inference-cost small model: scoreboard, levers, scaling ladder and decision gates vs nanochat |
| [`opd-strategy.md`](opd-strategy.md) | 129k vocab is fixed for API top-20 on-policy distillation: head-FLOP economics, OPD pipeline, staged plan and gates |
| [`eval-proxies.md`](eval-proxies.md) | Is pretraining loss a capability proxy? Cheap eval ladder (bpb → CORE-lite → CORE → task suites → teacher-match) |
| [`eval-harness.md`](eval-harness.md) | The T0 eval harness: shard/manifest contract, exact BPB byte accounting, agent-trajectory token masks, Marin-referenced data sources/mixes, CORE-lite reserved interface, usage and cost |
| [`tokenizer-contract.md`](tokenizer-contract.md) | The pinned DeepSeek-V4.1-Flash tokenizer: artifacts, verified vocab/pad/Engram-map contract, storage encoding |
| [`capacity-vs-compute-codesign.md`](capacity-vs-compute-codesign.md) | What parameter growth is "free": capacity vs active compute accounting, d12e32/d16/d20 estimates and experiments |
| [`kernel-optimization-status.md`](kernel-optimization-status.md) | R69 training-kernel breakdown, ceilings, overlap/launch analysis, remaining wins and the decision to stop |

## Sprint reports (R90–R95)

Single-RTX-5090 sprint reports: 5090 bring-up and the ¥70 nanochat comparison (R90–R91),
the K=40k vocabulary/data/eval migration and pre-P1 calibration (R93 P0a/P0b), the MFU-45
sprint, the P1 8B run (final raw bpb **1.06235**, EMA **1.05808**), its complete evaluation,
and the follow-up research/P2 plan (R94–R95). Raw evidence for the reports is curated under
`runs/` where applicable.

### R90 — 5090 bring-up and the ¥70 nanochat comparison
| Document | Contents |
|---|---|
| [`r90-final-report.md`](r90-final-report.md) | Final R90 report: the final run, 5090 throughput/MFU, bpb, cost accounting and the nanochat cost-efficiency comparison |
| [`r90-search-log.md`](r90-search-log.md) | Chronological training-search log: arms, budget, interruptions and decisions |
| [`r90-data-stage3-plan.md`](r90-data-stage3-plan.md) | Stage-3 data-recipe plan (ClimbMix vs DCLM-Baseline vs FineWeb-Edu) |
| [`r90-vocab-32k-plan.md`](r90-vocab-32k-plan.md) | Stage-2 32k-vocab / head ablation plan and accounting |

### R91 — search protocol
| Document | Contents |
|---|---|
| [`r91-search-protocol.md`](r91-search-protocol.md) | Four-stage cascaded hyperparameter-search protocol for spending fewer tokens per search |

### R93 — K=40k migration, MFU-45, P1 8B
| Document | Contents |
|---|---|
| [`r93-p0a-p0b-plan.md`](r93-p0a-p0b-plan.md) | P0a/P0b plan: K=40k vocab/data/eval migration + KD-data prep + dense calibration run |
| [`r93-p0a-p0b-results.md`](r93-p0a-p0b-results.md) | P0a/P0b results: tokenizer corpus, eval panel, KD dataset status, d24 calibration |
| [`r93-dense-plan.md`](r93-dense-plan.md) | P1 dense compute-optimal pivot: plan, FLOP/token accounting, P0b protocol |
| [`r93-hp-search-plan.md`](r93-hp-search-plan.md) | Pre-P1 hyperparameter-search state and plan (LR grid, measured cost model) |
| [`r93-hp-search-results.md`](r93-hp-search-results.md) | Pre-P1 hyperparameter-search results report |
| [`r93-p1-launch-recipe.md`](r93-p1-launch-recipe.md) | P1 8B launch recipe after the HP search (data, LR schedule, EMA, z-loss) |
| [`r93-p1-run-log.md`](r93-p1-run-log.md) | P1 8B run log: marks, LR probe, interruptions, final cost |
| [`r93-p1-final-report.md`](r93-p1-final-report.md) | P1 8B final report: loss curve, 3M-panel marks, LR-probe outcomes, cost, lessons |
| [`r93-ab-modes.md`](r93-ab-modes.md) | CED+CSA2 official-faithful layer-mode A/B vs d24_c1 (rejected: +0.0054 bpb, +7% wall) |
| [`r93-decode-estimate.md`](r93-decode-estimate.md) | d24_c1 inference-speed estimate (single-stream / batched) against measured anchors |
| [`r93-fp8-d24-memo.md`](r93-fp8-d24-memo.md) | FP8 for the d24_c1 dense body: paths, measurements, enablement, gates, expected band |
| [`r93-mfu-decomposition.md`](r93-mfu-decomposition.md) | MFU stage decomposition of the d24_c1 production kernel stack and remaining headroom |
| [`r93-mfu45-fp8-plan.md`](r93-mfu45-fp8-plan.md) | MFU-45 target: measurement conventions, FP8 alignment, route and gates |
| [`r93-mfu45-sprint.md`](r93-mfu45-sprint.md) | MFU-45 sprint final results (aim exceeded; step-time breakdown) |
| [`r93/configs/`](r93/configs/) | Configs used by the R93 experiments (d12w/d16w/d24_c1/d24_c1_ced/d32n, 40k vocab) |

### R94 — research reports
| Document | Contents |
|---|---|
| [`r94-shape-scaling.md`](r94-shape-scaling.md) | V4 depth/width scaling relation and the shape choice at our parameter scale |
| [`r94-small-model-methods-survey.md`](r94-small-model-methods-survey.md) | First-pass survey of small-language-model training methods |
| [`r94-research/arch.md`](r94-research/arch.md) | Architecture-side methods (2025–2026) |
| [`r94-research/data-side.md`](r94-research/data-side.md) | Data-side methods: filtering, synthetic data, mixtures, curriculum |
| [`r94-research/recipe.md`](r94-research/recipe.md) | Recipe/schedule/optimizer literature survey |
| [`r94-research/kd-capability.md`](r94-research/kd-capability.md) | Knowledge distillation, small-model capability and evaluation |
| [`r94-research/zgcm-1.md`](r94-research/zgcm-1.md) | ZGCM-1 notes: what transfers to our setting (FP8+TWEO, Muon, curriculum) |

### R95 — evaluation and P2 plan
| Document | Contents |
|---|---|
| [`r95-p1-eval-report.md`](r95-p1-eval-report.md) | Complete P1 8B evaluation: marks, full-val bpb (raw/EMA, CIs), sampling error, capability snapshot, multi-domain bpb, generation samples, nanochat/CORE comparison |
| [`r95-small-model-research-synthesis.md`](r95-small-model-research-synthesis.md) | Cross-synthesis of the R94 research reports: levers, ¥100 lever ledger, conflict table, execution order |
| [`r95-p2-plan.md`](r95-p2-plan.md) | P2 plan under the ¥100–250 budget: tiers, phases, gates, do-not-do list |
