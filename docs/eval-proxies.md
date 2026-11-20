# Is pretraining loss a good capability proxy? And what is cheap but more accurate?

Status note (2026-11-06). Short answer: **loss/bpb is a good proxy for training progress and for
ranking within one model family/data/tokenizer, but a weak proxy for "capability" across
architectures, data mixes and task types.** The cheap upgrade is a small multi-domain bpb panel +
a likelihood-based benchmark subset (CORE-lite), plus — for the OPD plan — teacher-match metrics.

## 1. When loss is good, and when it lies

| question | loss/BPB as proxy |
|---|---|
| Is this run healthy / which checkpoint is further along? | ✅ excellent (smooth, cheap, low variance) |
| Does a scaling change help at fixed model/data/eval code? | ✅ good first-order signal |
| Does data mix A beat B? | ⚠️ unreliable — loss follows the mix it is trained on; need per-domain held-out bpb |
| Does a MoE beat a dense model at equal loss? | ⚠️ loss does not measure sparse/expert quality or routing behavior; could tie while capabilities differ |
| Does it beat nanochat / GPT-2? | ❌ not comparable: different tokenizer (129k vs 32k), data mix, and eval distribution |
| Did OPD improve the student? | ❌ the objective is teacher match, not web-token likelihood |

Why it lies: loss is dominated by high-entropy tokens and data format; tokenizer changes the
token-level loss (bpb fixes tokenization but not data mix); different architectures can have
equal bpb and different benchmark accuracy; benchmarks are usually fit by a **task-specific
sigmoid/logistic in loss** (a small loss change can move an accuracy score by many points, and
near-chance tasks are pure noise at 200–600M scale).

Practical rule: use loss/bpb to **rank candidates cheaply within the same family**, then confirm the
top 1–2 with a benchmark panel before spending on long runs.

## 2. Cheap eval ladder (cost on 1×5090, d12-class forward = 141 MFLOP/token)

| tier | what | cost (est.) | when |
|---|---|---|---|
| T0 | multi-domain **held-out bpb** (web/code/math/zh/books, fixed 20–50M tokens per slice) | ~1–5 min per slice | every run; catches data-mix overfitting |
| T1 | **CORE-lite**: DCLM CORE tasks subset (~8–10 likelihood-based tasks, 200–500 examples, 0/5-shot) | ~5–15 min | every candidate that survives T0 |
| T2 | **full CORE** (22 tasks; the metric nanochat uses for GPT-2 capability, CORE 0.2565 baseline) | ~20–60 min | top 1–2 models only |
| T3 | task suite: ARC-E/ARC-C, HellaSwag, PIQA, WinoGrande, BoolQ, MMLU subset (likelihood, 0/5-shot) | ~10–30 min | before committing to a long run |
| T4 | generation evals: GSM8K (CoT), HumanEval pass@1, chat SFT evals | minutes-hours + API | only after SFT/RL pipeline exists |
| T5 | long-context: RULER / needle-in-a-haystack subsets (2K/32K/128K/1M) | ~0.5–2 h per context | for the 1M-context claim |
| OPD | **teacher-match**: top-1 agreement, top-20 KL/reverse-KL on held-out prompts; plus teacher-score of student samples | minutes on cached OPD data / small API bill | primary OPD metric |

Notes: all T0–T3 are forward-only (no generation), deterministic, and cheap; CORE is the standard
"capability aggregate" already used by nanochat and DCLM, so it makes our numbers comparable to the
nanochat leaderboard at the *task* level (tokenizer differences only affect efficiency, not the
definition). At 200–600M scale many tasks are near chance — aggregate (CORE) is more stable than
individual scores; report bootstrap CIs.

## 2b. Agent trajectories as a held-out domain — yes, but carefully

Add agent/tool-use data as **separate held-out domains**, not mixed into web BPB:

| domain | what | why |
|---|---|---|
| agent trajectories | multi-turn user/tool-call/tool-output sequences rendered with the pinned `chat_template.jinja` (thinking + chat modes, tool markers) | the target use case; catches template/tool-token modeling |
| assistant-turn tokens | BPB on the assistant tokens only (exclude tool outputs/user text) | tool outputs are high-entropy logs and swamp a mixed BPB |
| tool-call structure | exact-match / validity rate of emitted tool-call JSON or markers on held-out prefixes | closer to agent ability than BPB, still cheap (teacher-forced) |
| long tool outputs / logs / terminal sessions | 8K–128K–1M-context slices | exercises the 1M-context claim and the real agent context shape |

Caveats: (a) BPB on trajectories is still a weak proxy for *task success* — for the final claim add a
small agentic success eval (50–200 tasks, tool execution) after the rollout engine exists; (b) the
chat template must match `chat_template.jinja` exactly, otherwise BPB is not comparable; (c) **the OPD
training data will itself be teacher/student trajectories** — split prompts/tasks so eval trajectories
are disjoint from training ones (contamination is trivial to introduce here); (d) report the
per-token-type breakdown, never one mixed number.

## 3. "More accurate per ¥" recipe for our decision

1. **Data/eval harness first** ✅ T0 built — see [`eval-harness.md`](eval-harness.md): pinned
   tokenizer, exact UTF-8 byte accounting, multi-domain held-out BPB with assistant/tool token
   masks, and a reserved CORE-lite scoring interface. Remaining: hold out the real 20–50 M-token
   slices and port the CORE task files/format (nanochat's `core_eval.py` is MIT; DCLM data from HF).
   Pilot slices from Marin-referenced sources are already built (7 domains, 15.0 M tokens,
   phase-1 + jellyfish mix coverage 1.0) — see [`eval-harness.md`](eval-harness.md) §2b.
2. **Model-size decision (d12 vs d16 vs d20 at iso-FLOPs)**: run T0 + CORE-lite on each; pick the
   winner on **capability per training ¥**, not on a single bpb; confirm the top-2 with full CORE.
   This costs minutes of eval per run — negligible next to the training runs.
3. **For OPD**: use teacher match (top-1 agreement / KL) as the primary signal, T0/T1 as regression
   guards; full CORE only at milestones. Teacher-match is far more sensitive than web bpb to whether
   distillation is working.
4. **Don't** use MMLU/GSM8K/HumanEval generation at 200M scale as a primary decision metric — they
   are near-chance/noisy and generation-eval cost/API noise dominates the signal; use them only for
   external "does it work at all" reporting after SFT.

## 4. Bottom line

- Loss/bpb: keep it as the **cheap screening metric**, not the capability metric.
- Cheap + more accurate: **multi-domain bpb (T0) + CORE-lite (T1)**, escalating to full CORE (T2)
  for finalists; for OPD, **teacher-match** is the right primary metric.
- For external comparability with nanochat, CORE is the boundary object: ours is a different
  tokenizer/architecture, so compare **task scores**, not loss/token or tokens/s.
