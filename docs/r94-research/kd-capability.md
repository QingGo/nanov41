# R94 调研：知识蒸馏 · 小模型能力 · 评测（2025–2026）

> 范围：预训练蒸馏、小模型后训练（SFT/DPO/RLVR）、小模型评测实践、2025–26 公开强小模型配方、合成数据。
> 目的：为「单卡 5090、288M active / 8B tokens、held-out bpb 1.066（EMA 1.062）、¥100–300/轮」的 nanov41 P1
> 找出**可落地、成本可核算**的能力提升路线，并给出评测与预期分数带。
> 检索日期 2026-11-18；所有数字均来自论文/官方卡片，未核实者显式标注。
>
> **证据分级**：`A` = 论文正文数字，可复核；`B` = 官方博客/模型卡，未见完整论文；`C` = 本文基于锚点的外推/估算（非文献数字）；`D` = 未证实。

---

## 0. 结论速览（TL;DR）

1. **在固定算力下，预训练阶段的 logit 蒸馏基本不赚**。MiniPLM（ICLR'25）在**等训练 FLOPs** 下实测：200M/500M/1.2B 学生上，Vanilla KD、SeqKD、MiniLLM 全部 ≈ 或无 KD（200M：39.9 vs 39.9/39.7/39.0）；唯一稳定为正的是**离线数据精炼** MiniPLM（+1.4/+1.6/+1.7 分，9 任务均值）`A`。在线 KD 的算力花在教师前向上，等于用同样 FLOPs 换了更少的 token。
2. **但"教师分布"本身很值钱——前提是它几乎不额外花算力**。Gemma 2 用 7B 教师蒸馏 2B 学生、同样 500B tokens：3 benchmark 均值 **60.3 → 67.7（+7.4）**`A`。差别在于 Gemma 有现成教师、且把 KD 当作"超过数据墙的额外信号"，不是拿它替换 token。
3. **我们真正的瓶颈不是"缺蒸馏"，是"缺 token + 缺后训练"**。288M × 8B tokens = 28 tokens/参数；同尺寸公开强模型是 300–1000 tokens/参数（H2O-Danube2-1.8B 用 3T，Qwen3 系列用 36T）`A`。bpb 1.066 → 0.718 的差距约需 **10–30× 有效算力**（按我们实测 β≈0.06–0.08 外推）`C`，¥100–300 买不到，KD 也买不到。
4. **(a) 若目标是"能力"而非 bpb，¥100–300 内性价比排序是：SFT/SeqKD 合成数据（¥15–30）≫ 1B-token KD cooldown（¥50–70）> 末段 1–2B 数据精炼退火（¥60–115）**。理由：前两者直接改"生成行为"（重复循环、不会收尾、不服从格式），单位 ¥ 的可见能力增益最大；后者只能买到 −0.01~−0.03 bpb 和 +1~2 分下游。
5. **(b) 我们 288M/8B 的合理预期（估计，非实测）**：MMLU **25–27**（25 是随机基线）、ARC-E **45–52**、ARC-C **24–28**、HellaSwag **30–35**、PIQA **60–64**、WinoGrande **50–53**、GSM8K **0–1%**、CORE **0.10–0.15**。锚点：GPT-2 small 六任务均值 0.3954、L20-Edu-135M（13B tokens，单卡 L20）0.4150、SmolLM2-135M 0.4917`A`。**任何"288M/8B 能到 MMLU 40+"的说法都不成立。**
6. **(c) 最低成本评测：¥1–2/轮、<10 分钟**。lm-evaluation-harness 跑 ARC-E/ARC-C/HellaSwag/PIQA/WinoGrande/OBQA/LAMBADA + MMLU（或 tinyBenchmarks 100 题）+ GSM8K 200 题，288M 模型前向约 4,500 万 token，5090 上约 17 分钟，成本 ¥1–2；CORE 22 任务占其中 ~10 分钟。**先建评测，再花钱**。

---

## 1. 概览表

| # | 方法 | 年份 / 类型 | 做法（一句） | 关键证据（规模 → 数字） | 我们的可借鉴性（工时 / GPU / 单卡） | 预期 bpb / 能力 |
|---|---|---|---|---|---|---|
| 1 | **MiniPLM** [2410.17215](https://arxiv.org/abs/2410.17215) | 2024-10 / ICLR'25，高引用 | 教师−参考模型 log-prob 差做 reward，离线重采样语料；学生只读精炼数据 | 1.8B 教师 → 200M/500M/1.2B 学生，Pile 50B tokens，**等 FLOPs**：9 任务均值 39.9→41.3 / 43.2→44.8 / 44.9→46.6；数据需求 −2.4× `A` | 中：需教师跑语料；1.7B 教师 1B tokens ≈ 15h/¥42，全量 8.5B ≈ ¥364（超预算）；只能做末段 1–2B | bpb −0.005~−0.015（C）；下游 +0.5~1.5 分 |
| 2 | **SeqKD / textbook**（Phi-1/Phi-3）[2306.11644](https://arxiv.org/abs/2306.11644) [2404.14219](https://arxiv.org/abs/2404.14219) | 2023–24，经典 | 教师生成"教科书"文本，学生按自己的 tokenizer 重tokenize 做 SFT/继续预训练 | Phi-1：6B web + **1B 合成** → 1.3B 模型 HumanEval **50.6%**；同 pipeline 350M 也有 45% `A`；Phi-3：3.8B/3.3T，MMLU 69% `A` | **高**：纯文本级，绕开 tokenizer 不匹配；生成 100M tokens ≈ 3h/¥8；SFT 0.3B ≈ ¥4 | 生成可用性↑↑；GSM8K 0→2–6%；bpb ≈ 0 |
| 3 | **Gemma 2 全量 logit KD** [2408.00118](https://arxiv.org/abs/2408.00118) | 2024-07，高引用 | 2B/9B 不做 NTP，直接对 27B 教师的 next-token 分布做 forward KL | 2B/500B tokens：from scratch 60.3 vs 蒸馏自 7B **67.7（+7.4）**，3 benchmark 均值 `A`；2B/9B 比同尺寸强 2–3× | 低（对我们）：教师前向是纯增量算力；1B tokens 已是 ¥40–50 | bpb −0.01~−0.02（1B 规模，C） |
| 4 | **Gemma 3** [2503.19786](https://arxiv.org/abs/2503.19786) | 2025-03 | 1B–27B 全系"trained with distillation"；4B-IT 对标 Gemma2-27B-IT | 官方报告，未给 KD 消融数字 `A/B`；Gemma-3-1B base MMLU **26.26**（Qwen3 报告 Table）`A` | 同上 | 说明 1B 级 KD 是业界默认，但 1B 也才 26 MMLU |
| 5 | **Llama 3.2 1B/3B**（剪枝+蒸馏） | 2024-09，官方博客/模型卡 `B` | 从 Llama 3.1 8B 剪枝，再用 8B/70B 的 logits 蒸馏；9T tokens 预训练 | 1B base（SmolLM2 评测口径）：ARC 49.2 / HellaSwag 61.2 / MMLU-Pro 11.7 / GSM8K 7.6 `A` | 低：剪枝需要同架构大模型，我们没有 | 仅作"剪枝后必须蒸馏回血"的证据 |
| 6 | **GKD / MiniLLM / OPD** [2306.13649](https://arxiv.org/abs/2306.13649) [2306.08543](https://arxiv.org/abs/2306.08543)；[Thinking Machines 2025-10](https://thinkingmachines.ai/blog/on-policy-distillation/) | 2023 经典 + 2025 综述级博客 | 学生在自己 rollout 上向教师分布对齐（reverse KL / 教师打分），缓解 exposure bias | Thinking Machines：Qwen3-8B 上 OPD 复现 RL 策略，**梯度步少 7–10×，总算力效率 50–100×** `B`；GKD 在 T5 上稳定超 SeqKD `A` | 中：需要可打分教师 + rollout；我们教师 API 贵、本地 1.7B 教师弱 | 后训练阶段有效；**预训练阶段无效**（见 #1） |
| 7 | **Distillation Scaling Laws** [2502.08606](https://arxiv.org/abs/2502.08606) | 2025-02 / ICML'25，高引用 | 用 scaling law 分配 teacher/student 算力 | 学生 300M–10B；结论：**有现成教师时蒸馏在"随学生规模增长的算力上限"内优于监督学习**；教师也要训时通常不如 SL `A` | 高（决策用）：我们属于"教师已存在"，落在蒸馏占优区 | 支持做 KD，但只支持"教师免费/已存在"的前提 |
| 8 | **Effective Teacher Support** [2611.14850](https://arxiv.org/abs/2611.14850) | **2026-11 新** | 只保留教师最高概率的极小词表子集并 renormalize（100%→0.01%） | Qwen3 4B→1.7B：保留 **0.01%** 词表（按其词表规模 ≈13–15 个 token）MMLU-Pro **不降**；步时 −38%、显存 −24%；Nemotron 需 0.1% 才不崩 `A` | **很高**：直接决定我们 top-k logits 的 k；0.01% ≈ top-15 | 让"存 top-20 logits"从 120GB/1B tokens 变为可行 |
| 9 | **BPM 跨 tokenizer OPD** [2607.22334](https://arxiv.org/abs/2607.22334) | **2026-07 新** | 把教师 next-token 分布投影到学生词表的字节前缀空间，质量守恒 + 残差类 | >99% 训练位置可精确恢复 teacher 的 byte-prefix marginal `A` | **必需**：我们的 41,280 词表与任何教师都不同 | 跨 tokenizer logit KD 的唯一严谨做法 |
| 10 | **CSD / CAST 跨 tokenizer** [2610.28895](https://arxiv.org/abs/2610.28895) [2610.13257](https://arxiv.org/abs/2610.13257) | **2026-10 新** | 构造共同支撑集 / credit alignment，单一目标匹配 | 声称统一多种 tokenizer 对、免手工启发式 `A`（摘要级） | 中：BPM 的替代实现，选一个即可 | 同上 |
| 11 | **Mid-training KD（Switch Distillation）** [2609.01532](https://arxiv.org/abs/2609.01532) | **2026-09 新** | 中期训练阶段做 forward KL；发现它会**拖慢事实记忆**但继续提升推理，改为"只在教师高置信 token 上蒸馏" | 受控实验（规模未在摘要披露，`D` 规模） | 高：正好对应"8B 之后再加 0.5–1B 中训练"的位置 | 推理/格式 ↑，事实类（MMLU）持平或略降 |
| 12 | **LIMO / s1（极少量 SFT）** [2502.03387](https://arxiv.org/abs/2502.03387) [2501.19393](https://arxiv.org/abs/2501.19393) | 2025-02，高引用 | 817 / 1,000 条高质量带 CoT 的样本做 SFT，靠 test-time compute 放大 | 32B 上 AIME24 **63.3%**、MATH500 95.6% `A`；但**同一配方在 3.8B 上反而降**（见 #13） | 低：我们的底座没有可"eliciting"的知识 | 0.3B 上预期负收益（见 §5） |
| 13 | **Phi-4-Mini-Reasoning** [2504.21233](https://arxiv.org/abs/2504.21233) | 2025-04 | 四段式：长 CoT 中训练 → 高质量 CoT SFT → Rollout DPO → RLVR | 3.8B：base AIME 10.0 / MATH 71.8；**+LIMO 6.7 / 57.8；+S1K 3.0 / 47.0；全配方 57.5 / 94.6 / GPQA 52.0** `A` | 中：中训练数据量按 B tokens 计，我们只能做 1/50 规模 | 说明"少量 SFT 会伤小模型"，完整配方才有用 |
| 14 | **DeepSeek-R1 蒸馏** [2501.12948](https://arxiv.org/abs/2501.12948) | 2025-01，超高引用 | 671B 推理模型的 800k 条长 CoT 直接 SFT 到 1.5B–70B | R1-Distill-Qwen-1.5B：AIME24 **28.9**、MATH-500 **83.9**、GPQA 33.8、LiveCodeBench 16.9 `A` | 中：需要 1.5B 底座 + 大量长 CoT；我们 288M 容量不够 | **能力阈值证据**：1.5B 是"推理可被蒸馏出来"的下限附近 |
| 15 | **RLVR @1B（OLMo-2-1B）** [2611.02574](https://arxiv.org/abs/2611.02574) | **2026-11 新** | 追踪 13 个 checkpoint，测 GSM8K 首答与 256 次覆盖 | GSM8K **58.0→61.3**；覆盖率 94.5→96.9；错误探针 AUROC 不变 → **只是 sharpen，不是学会** `A` | 高（预期管理）：我们做 RLVR 最多 +2–4 分（C），且需 verifier | GSM8K +2~4（若底座已有 50%+）；我们底座 0%，无效 |
| 16 | **RLVR 低数据/低算力** [2604.18381](https://arxiv.org/abs/2604.18381) | **2026-04 新** | 系统测 SLM 在低数据、程序化任务上的 RLVR 缩放 | 低复杂度任务可泛化到高复杂度；混合复杂度最好 `A`（摘要级，无绝对分数） | 中：若做 RLVR，从程序化任务（计数/图/空间）起步比 GSM8K 现实 | 0.3B 上"程序化任务"可能 +5–15 分（C，未证实） |
| 17 | **RLVR 不产生新推理** [2504.13837](https://arxiv.org/abs/2504.13837) | 2025-04，高引用 | 大 k 的 pass@k 探边界 | RLVR 模型 k=1 更好，**k 大时 base 更高** → 没学到新解 `A` | 高（预期管理）：RLVR 不是能力来源 | 不投 |
| 18 | **SmolLM2** [2502.02737](https://arxiv.org/abs/2502.02737) | 2025-02，高引用 | 1.7B / **11T tokens** 多阶段；FineMath/Stack-Edu/SmolTalk | 1.7B：ARC 51.7 / HS 66.1 / MMLU-Pro 19.3 / GSM8K 48.8（instruct）`A`；135M 六任务均值 0.4917 `A` | 中：数据配方可抄（末段混 math/code/instruct），token 量抄不了 | bpb 可能不降、能力升 |
| 19 | **Qwen3-0.6B/1.7B** [2505.09388](https://arxiv.org/abs/2505.09388) | 2025-05，超高引用 | 0.6B–235B；**用小模型从旗舰蒸馏**降低小模型训练算力 | 0.6B base：MMLU **52.81**、MMLU-Pro 24.74、GSM8K **59.59**；1.7B：62.63 / 36.76 / 75.44 `A` | 低（token 量 36T 量级）；但证明"小模型 + 蒸馏 + 海量 token"是唯一路径 | 我们差 3 个数量级 token |
| 20 | **MiniCPM / MiniCPM4** [2404.06395](https://arxiv.org/abs/2404.06395) [2506.07900](https://arxiv.org/abs/2506.07900) | 2024-04 / 2025-06 | WSD 调度、model wind tunnel、UltraClean 数据、8T tokens | MiniCPM-1.2B：MMLU **49.63**、GSM8K 31.77；2.4B：53.46 / 53.83 `A` | 中：WSD 我们已在用；wind tunnel（小代理定超参）值得保留 | 已借鉴；剩余空间在数据 |
| 21 | **H2O-Danube2-1.8B** [2401.16818](https://arxiv.org/abs/2401.16818) | 2024-01 | 1T+2T=3T tokens，Llama/Mistral 式配方 | MMLU **40.20** / HS 72.95 / ARC 43.34 / GSM8K 29.80 `A` | 低：3T tokens | **最强"参数≠能力"锚点**：1.8B 也要 3T token 才 40 MMLU |
| 22 | **Phi-4-mini** [2503.01743](https://arxiv.org/abs/2503.01743) | 2025-03 | 3.8B，200k 词表，重合成数据（math/code） | 摘要：同尺寸显著领先、数学/代码对标 2× 模型 `A`（正文表格未含纯文本 MMLU，略） | 低 | 合成数据配方可抄 |
| 23 | **Tokenizer Myths** [2611.04215](https://arxiv.org/abs/2611.04215) | **2026-11 新** | 用未训练模型 + 同数据训练模型检验 bpb 的三个"神话" | **BPB 并非 tokenizer 无关**；未训练模型上 BPB 就偏向某些词表；真实模型上 **BPB 与下游仅弱相关** `A` | **极高（评测方法论）**：我们只有 bpb，这条直接判死刑 | 必须补下游评测，bpb 不能当能力代理 |
| 24 | **tinyBenchmarks** [2402.14992](https://arxiv.org/abs/2402.14992) | 2024-02，高引用 | 用 IRT 挑 100 条题估计全量分数 | MMLU 14k 题 → **100 题**即可可靠复现 `A` | **极高**：廉价高频评测 | 评测成本 ↓ 100× |
| 25 | **CORE / DCLM** [2406.11794](https://arxiv.org/abs/2406.11794) | 2024-06，高引用 | 22 个任务的 ensemble 指标（中心化准确率） | GPT-2 CORE **0.256525**；nanochat d24 **0.2585** @ bpb 0.748 `A` | 高：与 nanochat 唯一可比口径 | 我们预计 0.10–0.15（C） |

---

## 2. 逐方法

### 2.1 预训练蒸馏

#### 2.1.1 MiniPLM：等算力下唯一为正的预训练 KD 形态（ICLR'25，高引用）

**做法**：不让学生看教师的 soft label，而是用教师与一个弱参考模型（104M）的 log-prob 差作为"难度/信息量" reward，对语料做重要性重采样（Difference Sampling），学生只在精炼后的子集上做普通 NTP。教师推理完全离线，每条样本只存一个 float（50B tokens 仅 200MB）。

**证据**（1.8B Qwen 教师，Pile 50B，**严格控制总训练 FLOPs 相同**）：

| 学生 | 无 KD | Vanilla KD | MiniLLM | SeqKD | **MiniPLM** |
|---|---:|---:|---:|---:|---:|
| 200M | 39.9 | 39.9 | 39.0 | 39.7 | **41.3** |
| 500M | 43.2 | 43.6 | 41.5 | 42.3 | **44.8** |
| 1.2B | 44.9 | 45.4 | 43.6 | 44.3 | **46.6** |

（9 个零样本下游任务均值；`A`）论文原话：等算力下 "all KD methods perform similar or worse than Pre-Train w/o KD"。外推数据需求降低 2.4×。

**对我们的可借鉴性**：这是**最重要的负面+正面证据**。负面：不要指望"把教师 logits 接进来一起预训"能提 bpb——在线 KD 的教师前向会吃掉等量算力。正面：**离线数据精炼是等算力正收益**，但打分成本随语料线性增长：1.7B 教师 1B tokens ≈ 15h/¥42，8.5B ≈ 131h/¥364——**我们只能精炼末段 1–2B（¥42–86）**，不能全量。
**工时**：3–5 天（打分脚本 + 采样 + 重训配置）；**GPU**：1×5090 可（教师 1.7B bf16 ≈ 3.4GB）；**预期**：bpb −0.01~−0.03，下游 +1~2 分（按 MiniPLM 的 +1.4~1.7 迁移，打折）。

#### 2.1.2 SeqKD / textbook：唯一绕过 tokenizer 的"蒸馏"

**做法**：教师生成文本（讲解、解题、对话），学生用自己的 tokenizer 重 tokenize 后做 SFT 或继续预训练。信息量低于 logits，但没有词表对齐问题。

**证据**：
- Phi-1：6B 过滤 web + **1B GPT-3.5 合成教科书** → 1.3B 参数 HumanEval **50.6%**；同 pipeline 的 350M 也有 45% `A`。这是"合成数据 = 能力"的最强早期证据。
- Phi-3：3.8B / 3.3T tokens（重过滤 web + 合成），MMLU 69% `A`。
- SmolLM2：FineMath / Stack-Edu / SmolTalk 三个自建数据集在多阶段中混入，1.7B 超过 Qwen2.5-1.5B `A`。

**可借鉴性**：**最高**。我们与任何教师都不同词表（41,280），SeqKD 是零成本的规避方案；生成 100M tokens 用本地 Qwen3-4B ≈ 3h/¥8。缺点：只学"输出长什么样"，不学"教师认为别的词也该得多少概率"。

#### 2.1.3 Gemma 2 / Gemma 3：全量 logit KD 的工业级证据

**做法**：2B/9B 完全用 KD 目标（forward KL，教师的 next-token 分布）替代 NTP，训练超过 compute-optimal 50× 的 token 量；Table 6 的消融是 2B 从 7B 教师蒸馏。

**证据**：Table 6 —— 2B、500B tokens，from scratch 3-benchmark 均值 **60.3**，从 7B 蒸馏 **67.7（+7.4）**`A`。注意：这是**同 token 数**对比，教师前向算力是额外投入；论文没有给"等总算力"对比。

**可借鉴性**：低。我们的教师前向无法免费：1B tokens 的 1.7B 教师前向 ≈ 15h/¥42，8B ≈ ¥340 且存储 top-20 logits ≈ 1TB。**结论：KD 只能作为"末段 1B token 的 cooldown"，不能覆盖全程。**

#### 2.1.4 Llama 3.2：剪枝 + 蒸馏回血（官方博客级证据 `B`）

1B/3B 由 Llama 3.1 8B **结构化剪枝**得到，再用 Llama 3.1 8B/70B 的 logits 蒸馏恢复，预训练 9T tokens。公开卡片数字（SmolLM2 统一评测口径）：1B base ARC 49.2 / HellaSwag 61.2 / MMLU-Pro 11.7 / GSM8K 7.6 `A`。
**可借鉴性**：低（我们没有同架构大模型可剪）。它只证明一件事：**剪枝掉的容量必须靠蒸馏补回**，否则小模型会明显退化。

#### 2.1.5 2025–2026 新方法（这一节是本调研的增量）

**(a) On-policy distillation（OPD）从"后训练技巧"变成主流。**
GKD（2306.13649）与 MiniLLM（2306.08543）是经典：学生在自己生成的序列上接受教师反馈，用 reverse KL 缓解 exposure bias。2025-10 Thinking Machines 的博客把它讲成了"后训练第三条路"：在 Qwen3-8B 上，OPD 复现 RL 训练出的策略只需 **7–10× 更少梯度步，总算力效率 50–100×**`B`。2026 年 arXiv 上 OPD 相关论文爆发（11 月单月 >20 篇），说明它已是后训练标配。
**关键区分**：OPD 在**后训练**有效，是因为学生已具备该能力、只需"锐化"；在**预训练**阶段（MiniPLM 的等算力实验）无效。二者不矛盾。

**(b) 教师 logits 压缩：top-k 可以小到 0.01%。**
Effective Teacher Support（2611.14850，2026-11）：Qwen3 4B→1.7B，只保留教师最高概率的 **0.01%** 词表（按其词表规模 ≈13–15 个 token）并 renormalize，MMLU-Pro **无退化**；步时 −38%、显存 −24%；Nemotron 12B→9B 需 0.1%。**这条直接决定我们的存储/带宽预算**：top-13~20 足够，1B tokens × 20 × 8B ≈ 160GB，可接受；top-1000 则 8TB，不可行。

**(c) 跨 tokenizer KD：从 ULD 到 BPM/CSD/CAST。**
- ULD（2402.12030，2024）：用最优传输在词表间做软对齐，开创性但只是 loss 层面的近似。
- **BPM（2607.22334，2026-07）**：把教师分布投影到"学生词表的字节前缀"上，质量守恒 + 显式残差类；>99% 位置可精确恢复。这是目前最干净的方案。
- CSD（2610.28895）/ CAST（2610.13257）/ Breaking the Tokenizer Barrier（2606.09456）：都是 2026 年的同期工作，思路分别是共同支撑集、credit 对齐、token 映射；2606.09456 明确报告"跨 tokenizer OPD 比 SFT baseline 显著更省算力"。
**对我们的意义**：我们的 41,280 词表是自建的，与 DeepSeek/Qwen 都不同，**做 logit KD 必须选一个跨 tokenizer 方案**；否则只能退回 SeqKD。

**(d) 中训练 KD（mid-training）：推理涨、事实不涨。**
Switch Distillation（2609.01532，2026-09）：forward KL 在预训练阶段同时提升推理与事实回忆，但在**中训练**阶段会**拖慢事实记忆**（教师对程序性数据更自信、学生对低熵事实学得更早），于是改为"只在教师高置信 token 上蒸馏"，推理收益保留、事实不掉。**这正好是我们"8B 之后再补 0.5–1B 中训练"的场景**，也预告了 MMLU 不会因 KD 明显上涨。

**(e) 蒸馏的 scaling law 给了决策边界。**
Distillation Scaling Laws（2502.08606，ICML'25）：学生 300M–10B、token 1B–10T 的扫描。核心结论：**教师已存在时，蒸馏在"随学生规模增长的算力上限"内优于监督学习；教师也要训练时通常不如 SL**；固定 token 预算下，学生越大，教师 cross-entropy 应越小（教师要更强）。我们属于"教师已存在（API 或本地小教师）"，落在蒸馏占优区——但占优幅度随学生增大而缩小。

#### 2.1.6 算账：教师前向到底多少钱（我们自己的数）

| 动作 | 吞吐假设 | 1B tokens 耗时 | 成本（¥2.78/h） |
|---|---|---:|---:|
| 学生训练（288M，62.5k tok/s，P1 实测） | 62.5k tok/s | 4.4 h | **¥12** |
| 教师前向 1.7B bf16（~30% MFU，18k tok/s） | 18k tok/s | 15 h | **¥42** |
| 教师前向 4B bf16（~40% MFU，10k tok/s） | 10k tok/s | 28 h | **¥78** |
| 教师生成 100M tokens（4B，10k tok/s） | 10k tok/s | 2.8 h | **¥8** |
| 学生 SFT 0.3B tokens | 62.5k tok/s | 1.3 h | **¥4** |

→ **KD 覆盖 8B 全程的教师成本 ≈ ¥340（1.7B 教师）**，已超单轮预算；覆盖末段 1B ≈ ¥42–54，可行。这是所有路线排序的物理基础。

---

### 2.2 小模型后训练：SFT / DPO / RLVR 在 0.1–1B 的证据

#### 2.2.1 SFT 在 ~1B 有效，在 ~0.3B 主要买"可用性"

- SmolLM2-1.7B：base → instruct，GSM8K **31.1 → 48.8**、MMLU-Pro 19.4、IFEval 56.7（table 4）`A`。SFT+DPO 带来 +17.7 的 GSM8K，这是"后训练能把 latent 能力调出来"的清晰证据。
- Qwen3 系列：官方明确"利用旗舰模型知识降低小模型训练算力"，0.6B base 即有 MMLU 52.8 / GSM8K 59.6，后训练再叠加 thinking 模式 `A`。
- MiniCPM-1.2B（SFT 后）：MMLU 49.63 / GSM8K 31.77 `A`。
- **反例（很重要）**：L20-Edu-135M 的 SFT+权重插值把六任务均值从 0.4150 变成 0.41497——**SFT 不改变多选题能力**，它改变的是生成行为 `A`。

**结论**：对我们 288M，SFT 的收益主要在"不再贪心复读、会收尾、能按格式答"，而不是 MMLU/ARC 分数。这恰好是当前最痛的点，所以性价比极高。

#### 2.2.2 拒绝采样 / RFT / 少量高质量 SFT：有门槛

- LIMO（2502.03387）：817 条样本，在 **32B** 上 AIME24 63.3 / MATH500 95.6 `A`。作者提出 "LIMO Hypothesis"：推理能力的门槛不是数据量，而是**预训练已编码的领域知识**。
- s1（2501.19393）：1,000 题 + test-time scaling。
- **Phi-4-Mini-Reasoning 的反证**（2504.21233）：3.8B 上，base AIME 10.0 / MATH 71.8；加 LIMO → **6.7 / 57.8**；加 S1K → **3.0 / 47.0**；只有走完"长 CoT 中训练（B tokens 级）→ SFT → Rollout DPO → RLVR"才到 57.5 / 94.6 `A`。
  → **少量 SFT 对能力不足的模型是负收益**（分布外长 CoT 会把模型带偏）。这条直接否掉"拿 1k 条 R1 轨迹 SFT 我们的 288M"的想法。

#### 2.2.3 RLVR 在 1B 级：只能 sharpen，且需要底座已会

- OLMo-2-1B（2611.02574，2026-11）：13 个 checkpoint 全程追踪，GSM8K **58.0 → 61.3**，256 次采样覆盖率 94.5 → 96.9，错误检测探针 AUROC 不变 → **知识早就在，RLVR 只是提高命中率** `A`。
- 1.5B GRPO（2503.16219）：7,000 样本、4×A40、24h、$42，AMC23 **63 → 80**、AIME24 46.7 `A`。但底座是 R1-Distill-Qwen-1.5B（已被蒸馏过），不是从零小模型。
- 2504.13837：大 k 的 pass@k 下 base 反而更高 → RLVR 不产生新推理路径 `A`。
- 2604.18381（2026-04）：SLM 在低数据/低算力下，**程序化任务（计数、图、空间）** 的 RLVR 可从小复杂度泛化到大复杂度，混合复杂度最好 `A`。

**拒绝采样 / RFT（rejection sampling fine-tuning）**：采样 k 条解，用 verifier 留下正确的做 SFT（STaR [2203.14465] 与自提升/拒绝采样 [2210.11610] 的经典做法；LIMO 与 Phi-4-Mini-Reasoning 的 Rollout DPO 都是它的近亲）。
**它和 RLVR 受同一个约束**：底座必须已经能以非零概率答对，否则一条正样本都采不到（`C`，逻辑推论）。对 288M/8B、GSM8K≈0 的底座，拒绝采样在数学上无效；但它对**程序化任务**（个位数加减、格式转换）可能有效，因为底座有非零成功率——这是唯一值得试的 RL 形态。

**结论**：RLVR 对我们是"最后 5%"的工具，不是"从 0 到 1"的工具。底座 GSM8K 为 0 时，GRPO 没有正样本可学（全 0 奖励 → 零梯度），这是硬约束。

#### 2.2.4 "小模型能不能有可用能力"的实际阈值（综合判断 `C`）

| 规模 | 预训练 token | 能做什么 | 证据 |
|---|---|---|---|
| 135M | 2T（SmolLM2-135M） | 流畅续写、简单常识；六任务 0.49 | `A` |
| 135M | 13B（L20-Edu） | 六任务 0.415，GSM8K≈0 | `A` |
| 288M | 8B（我们） | 约等于 GPT-2 small 档；生成需 SFT 才可用 | `C` |
| 0.6B | 36T 量级（Qwen3 系列） | MMLU 52.8、GSM8K 59.6 | `A` |
| 1.5B | 蒸馏 + 长 CoT | AIME 28.9、MATH 83.9 | `A` |
| 1.7B | 11T（SmolLM2） | GSM8K 48.8、ARC 51.7 | `A` |
| 3.8B | 3.3T + 四段后训练 | AIME 57.5、MATH 94.6 | `A` |

→ **"可用能力"的分水岭在 1B 左右，且要求 ≥1T 预训练 token**。288M/8B 处于"能说话、不能推理"的区间；后训练能改善"说话"，不能创造"推理"。

---

### 2.3 小模型评测实践

#### 2.3.1 lm-evaluation-harness 常用任务（我们该跑的子集）

| 任务 | 题数 | few-shot | 度量 | 我们是否要 |
|---|---:|---|---|---|
| arc_easy / arc_challenge | 2,376 / 1,172 | 0/25 | acc, acc_norm | 要（主） |
| hellaswag | 10,042 | 0/10 | acc, acc_norm | 要（主） |
| piqa | 1,838 | 0 | acc | 要 |
| winogrande | 1,267 | 5 | acc | 要 |
| openbookqa | 500 | 0 | acc | 要 |
| lambada_openai | 5,153 | 0 | acc, ppl | 要（语言建模健康度） |
| mmlu | 14,042 | 5 | acc | 要（但用子集，见下） |
| gsm8k | 1,319 | 5/0-shot CoT | exact_match | 要（生成，256 token） |
| ifeval | 541 | 0 | prompt/instruction acc | 选做（需 chat 模板） |
| boolq / rte / anli | 3k/2.7k/1.6k | 0 | acc | 选做 |

**参考组合**：SmolLM2 的评测组合是 ARC / HellaSwag / MMLU-Pro / PIQA / GSM8K / MATH / HumanEval / IFEval / MT-Bench / OpenRewrite-Eval（表 4，`A`）；nanochat 用 CORE（22 任务）+ bpb + 少量生成样例。我们建议 **L1（常识）+ MMLU + GSM8K + IFEval + 自建生成健康度**，覆盖"知识/推理/指令/可用性"四轴，成本已在 §4.1 核算。

#### 2.3.2 CORE（DCLM 22 任务）：与 nanochat 唯一的可比口径

DCLM（2406.11794）的 CORE = 22 个任务中心化准确率的平均；GPT-2（1.6B）基线 **0.256525**；nanochat d24 在 bpb 0.748 时 CORE **0.2585**，换 ClimbMix 后 bpb 0.7185、CORE 0.2571 `A`。nanochat 自己实现了单文件 `core_eval.py`，可以直接借（MIT）。
**用途**：我们与 nanochat 的差距用 CORE 表达，比 bpb 更可信；但 CORE 噪声约 ±0.005（nanochat 榜上同 bpb 的两次 run 差 0.0064）。

#### 2.3.3 tinyBenchmarks：100 题估 MMLU

2402.14992：IRT 挑 100 条题即可可靠复现 MMLU 全量分数 `A`。适合"每次改动后 2 分钟看一眼"，但**不要用它做最终结论**（100 题的二项标准误在 p=0.3 时约 ±4.6 分）。

#### 2.3.4 bpb 与下游能力：弱相关，且跨 tokenizer 不可比

这是本次调研**对项目方法论最重要的一条**：
- Tokenizer Myths（2611.04215，2026-11）：① BPB 并非 tokenizer 无关（未训练模型上 BPB 就系统性偏向某些词表）；② **真实模型上 BPB 与下游仅弱相关**；③ "unigram-normalized loss" 也不 tokenizer-agnostic `A`。
- nanochat 自己的数据是最干净的自然实验：**同一模型、同一 CORE 口径，只换验证集**，bpb 从 0.74833 → 0.71854（−0.03），CORE 从 0.2585 → 0.2571（−0.0014，噪声内）`A`。**0.03 的 bpb 差异可以完全不带来能力差异。**
- 我们的 1.066 与 nanochat 的 0.718 都是 ClimbMix 系验证集，但 tokenizer 不同（41,280 vs 32,768），按 ① 这条，**差距里有一部分是词表伪影**（我们的词表更大、压缩率更高，通常 bpb 会更低而非更高，所以真实差距只会更大，不会更小）。

**结论**：bpb 只能做"同一 tokenizer、同一验证集、同一模型族"内的相对比较；跨项目、跨词表、跨语料一律不可比。必须补下游评测。

#### 2.3.5 噪声与显著性

- CORE：±0.005（nanochat 榜）。
- MMLU 全量 14k：二项 SE ≈ 0.4 分；100 题子集 ≈ 4.6 分。
- HellaSwag 10k：SE ≈ 0.5 分；ARC-E 2.4k：SE ≈ 1.0 分。
- 我们的模型更小，方差更大。**决策规则：<1.5 分的差异不决策；关键结论用全量集 + 2 个种子。**

---

### 2.4 2025–26 公开强小模型配方（数字均为公开值）

| 模型 | 参数 | 预训练 tokens | 数据/训练要点 | 蒸馏/后训练 | 公开数字（base 除注明） | 我们能抄什么 |
|---|---:|---:|---|---|---|---|
| **SmolLM2** (2502.02737) | 135M/360M/1.7B | **11T**（1.7B） | 4 阶段；FineMath / Stack-Edu / SmolTalk 自建；每阶段按上一阶段表现调 mix | SFT + DPO（SmolTalk） | 1.7B：ARC 51.7 / HS 66.1 / MMLU-Pro 19.3 / GSM8K 48.8（instruct） | 末段 10–20% 混 math/code/instruct；数据 mix 要迭代 |
| **SmolLM3** (HF blog, 2025-07) `B` | 3B | 11.2T | 多语言、长上下文、hybrid reasoning（think/no_think） | SFT + APO/DPO | 官方博客给 MMLU-Pro/GSM8K 等，本文未逐项核实 `D` | 双模式（快答/慢想）的模板设计 |
| **Qwen3-0.6B/1.7B** (2505.09388) | 0.6B / 1.7B | 36T 量级 | 强数据 + 强教师；thinking/non-thinking 统一 | 旗舰蒸馏 + SFT + RL | 0.6B：MMLU 52.81 / MMLU-Pro 24.74 / GSM8K 59.59 / BBH 41.47；1.7B：62.63 / 36.76 / 75.44 / 54.47 | "小模型靠旗舰蒸馏"的路线确认；0.6B 是 288M 的 2× |
| **MiniCPM-1.2B/2.4B** (2404.06395) | 1.2B/2.4B | 1T+ | WSD 调度；model wind tunnel；数据/模型双轴 scaling | SFT + DPO | 1.2B：MMLU 49.63 / GSM8K 31.77；2.4B：53.46 / 53.83 | WSD + 小代理调超参（我们已用） |
| **MiniCPM4** (2506.07900) | 0.5B/8B | **8T** | UltraClean 过滤 + UltraChat v2；InfLLM v2 稀疏注意力 | chunk-wise rollout RL；BitCPM 三值 | 官方称端侧效率最优 `A`（摘要） | 数据清洗管线；稀疏注意力 |
| **H2O-Danube2-1.8B** (2401.16818) | 1.8B | **3T** | Llama2/Mistral 式；1T+2T 增量训练 | SFT + DPO | MMLU **40.20** / HS 72.95 / ARC 43.34 / GSM8K 29.80 / Wino 68.03 | **"1.8B 也要 3T token 才 40 MMLU"** 的最硬锚点 |
| **Phi-4-mini** (2503.01743) | 3.8B | 未披露 | 200k 词表；重合成数据（math/code 为主） | SFT + DPO + RL | 摘要：同尺寸领先、数学/代码对标 2× 模型 | 合成数据配比；词表扩到 200k 的取舍 |
| **Gemma3-1B** (2503.19786) | 1B | 未披露 | 蒸馏训练；local:global 注意力 | 蒸馏 + 后训练 | base MMLU **26.26**（Qwen3 报告口径） | 1B 级 base 的 MMLU 现实值 |
| **Llama3.2-1B** | 1B | 9T + 剪枝蒸馏 | 从 8B 剪枝；8B/70B logits 蒸馏 | SFT + DPO | ARC 49.2 / HS 61.2 / MMLU-Pro 11.7 / GSM8K 7.6 | 剪枝后蒸馏回血 |

**横向读法**：同样是 1B 级，MMLU 从 26（Gemma3-1B）到 52.8（Qwen3-0.6B），差 2×；差异几乎全部来自 **token 量（2T vs 36T 量级）+ 教师质量 + 数据**，不是架构。我们 8B tokens 在这个坐标系里处于最左下角。

---

### 2.5 用大模型合成数据提升小模型：成本/收益证据

| 证据 | 规模 | 收益 | 成本 | 对我们的含义 |
|---|---|---|---|---|
| **phi-1**（2306.11644） | 1.3B，6B web + **1B 合成** | HumanEval 50.6%（同 pipeline 350M 45%） | GPT-3.5 生成 1B tokens（2023 年价，未披露） | 合成数据在**窄域**（代码）能顶掉大量 web 数据 |
| **Phi-3**（2404.14219） | 3.8B / 3.3T | MMLU 69% | 未披露 | "重过滤 web + 合成"是 3B 级标配 |
| **Cosmopedia**（HF, 2024）`B` | ~30B tokens 合成教科书 | 小模型在知识/教材风格上明显优于同量 web | 生成成本按 2024 年 API 价 | 公开可下载，可直接混入 |
| **LIMO**（2502.03387） | 817 条 / 32B | AIME 63.3 | 近乎零 | **只在大底座上成立**；3.8B 上反降 |
| **Phi-4-Mini-Reasoning**（2504.21233） | 3.8B + 长 CoT 中训练 | AIME 10.0 → 57.5 | 中训练数据 B tokens 级 | 合成数据的价值在"中训练规模"，不在"1k 条" |
| **SmolTalk / FineMath / Stack-Edu**（2502.02737） | 数十–数百 B tokens 级 | 1.7B 全面超 Qwen2.5-1.5B | 公开 | 可直接取用做 SFT/退火 |

**成本侧（我们自己的数）**：本地 Qwen3-4B 生成 100M tokens ≈ ¥8；生成 1B ≈ ¥78。**收益侧的经验法则**：合成数据对"格式/技能/窄域"有效（phi-1 的代码、SmolTalk 的对话），对"广谱知识"无效（知识要靠预训练 token）。所以合成数据应该投在**后训练/退火**，不是替代预训练。

---

## 3. 我们 setting 的推荐路线（含预算）

### 3.0 先立三条硬约束

1. **词表不匹配是硬约束**：学生 41,280，任何教师都不同 → logit KD 必须用 BPM/CSD，或者退回 SeqKD（文本级）。
2. **教师前向是纯增量成本**：1B tokens ≈ ¥42（1.7B 教师）。8B 全程 ≈ ¥340，超预算。
3. **bpb 不再是好目标**：Tokenizer Myths（2026-11）证明它跨词表弱相关；我们已有 1.066 这个数，继续优化它的边际信息量很低。**应把预算转向"能力可测"的方向。**

### 3.1 (a) 三条路线排序（¥100–300 预算内）

#### R1（最推荐）：SeqKD + SFT 后训练 —— ¥15–30，1 周

**做什么**
1. 用本地 Qwen3-4B（或 API）生成 3 类数据，共 **100–200M tokens**：
   - 指令/对话 60%：SmolTalk + MagPie 风格，覆盖"收尾、格式、拒答"；
   - 数学 CoT 25%：GSM8K/MATH 题面 + 教师分步解（**答案可验证**，只保留最终答案正确的轨迹）；
   - 代码/结构化 15%：短函数 + 单测。
2. 与学生预训练数据 **1:4 混入**（防遗忘），SFT 0.2–0.4B tokens，LR 2e-5~1e-4，1 epoch。
3. 可选：在 5k 条"教师优选 vs 学生采样"对上做 1 轮 DPO（¥2）。

**为什么划算**：直接修"贪心复读、不收尾、不服从格式"——这是当前唯一"用户能感知"的能力缺口；文献侧有 SmolLM2（GSM8K +17.7）与 Phi-4-Mini-Reasoning（SFT 是四段式的一环）支撑；失败模式清楚（灾难性遗忘，用混料+低 LR 控制）。
**预期**（`C`）：bpb ±0.005（不动）；IFEval 0→10–20%；GSM8K 0→2–6%（仅限训练分布内题型）；MMLU +0.5；**复读率显著下降**。
**风险**：<1B 模型 SFT 过量会掉 bpb/常识（Phi-4-mini 的 LIMO 反例），所以**混料比例和小 LR 是硬要求**。

#### R2（值得做一轮）：1B-token KD cooldown —— ¥50–70，1 周

**做什么**：在 R1 之前或之后，取最后 1B tokens 做 KD 退火：教师 Qwen3-1.7B，**top-13~20 logits**（2611.14850 证明 0.01% 够），BPM（2607.22334）投影到我们词表；离线存 logits（≈160GB）后学生训练 1B tokens。
**成本**：教师 15h（¥42）+ 学生 4.4h（¥12）+ 存储/IO ≈ ¥5 → **¥60**。
**预期**（`C`）：bpb −0.008~−0.018；ARC-E/HS +1~2；MMLU +0.5~1（mid-training KD 对事实类可能不涨，2609.01532）。
**为什么值得**：这是唯一能在"等算力"框架下站得住的预训练 KD 形态（Gemma2 的 +7.4 是 token-matched，不是 iso-FLOPs；我们只做末段 1B，占比 12.5%，即使打折也有正期望）；且工程量可控（BPM 有参考实现思路，1–2 天）。**诚实提醒**：同样 ¥60 直接多训 5B token 大约能换 −0.04 bpb，比 R2 的 bpb 收益大 2–5 倍；R2 的价值假设在"能力/推理风格"而非 bpb，所以必须用 §4 的下游评测来判定，bpb 不作为它的 KPI。
**止损线**：若 0.3B 试点 bpb 不降 >0.005，放弃。

#### R3（可选，排在 R1/R2 之后）：MiniPLM 式数据精炼，只做末段 1–2B —— ¥60–115，1–2 周

**做什么**：用 1.7B 教师 + 288M 自身（作 reference）对**末段 1–2B tokens** 打 log-prob 差，取 top 50–70% 后做退火训练（不是从头重训 8B）。
**成本**：打分 1B ≈ ¥42、2B ≈ ¥86；退火 1–2B ≈ ¥12–24；评测 ¥3 → **¥57（1B）~¥113（2B）**。若想"精炼后重训 8B"，需要先打分 16B+（keep 50%）≈ ¥670，**超预算，不可行**。
**预期**：bpb −0.005~−0.015（只作用于末段，按 MiniPLM 在 200M 的 +1.4 分迁移并打折）；下游 +0.5~1.5。
**为什么不优先**：单位 ¥ 增益最小；我们的 ClimbMix 已是过滤语料，精炼空间可能小于 Pile；且它和 R2 争同一段"末段退火"的预算，两者只能选一个（R2 的证据更强）。

#### 对照臂（必须留）：8B → 20B tokens 继续预训 —— ¥165

如果目标是"绝对质量/bpb"，按我们实测 β≈0.06–0.08，8B→20B 约 −0.05~−0.08 bpb（1.066→~0.99），比任何 KD 路线都多。**但它不解决"能力"问题**（MMLU 仍 ~26）。建议在 R1/R2 之后，若还有预算再考虑。

### 3.2 两轮预算方案（推荐执行顺序）

| 轮次 | 内容 | GPU 机时 | 成本 | 产出 |
|---|---|---:|---:|---|
| **Round A（¥40）** | 建评测（§4）+ 100M 合成数据 + SFT 0.3B + DPO | ~4 h | **¥15–20** | 第一份下游分数；生成质量修复 |
| **Round B（¥70）** | 1B KD cooldown（top-20 + BPM）+ 复评 | ~20 h | **¥60–70** | bpb −0.01 量级；ARC/HS +1–2 |
| **Round C（可选）** | 末段 2B 数据精炼 + 退火（替代 R2，不叠加） | ~35 h | **¥110** | bpb −0.01；下游 +0.5–1.5 |
| 合计（A+B） | | 24 h | **¥75–90** | 留下 ¥10–200 余量做 2–3 次小消融 |

**工时估算**：评测适配 2–3 天（一个人）；R1 数据管线 2 天 + 训练 4h；R2 BPM 实现 2 天 + 训练 20h；R3 3 天 + 45h。**单卡 24GB 全部可行**（教师 1.7B bf16 3.4GB、学生 288M bf16 0.6GB、seq 8192 激活 <8GB）。

### 3.3 明确不做什么

| 不做 | 原因 | 证据 |
|---|---|---|
| 全量 8B logit KD | 教师成本 ¥340+，超预算；等算力下无收益 | 2.1.1 / 2.1.6 |
| 用 1k 条 R1 轨迹 SFT | 3.8B 上已证明反降，288M 更糟 | 2504.21233 |
| 288M 上做 RLVR/GRPO | 底座 GSM8K≈0，无正样本；1B 上也只 +3 分 | 2611.02574 / 2504.13837 |
| 追 bpb 0.718 | 需 10–30× 算力；且跨词表不可比 | §2.3.4 |
| 自训大教师 | 教师训练成本远高于调用/本地小教师 | 2502.08606 |

---

## 4. 评测方案

### 4.1 (c) 最低成本方案：任务集 + 时长 + ¥

**一次性投入**：给 lm-evaluation-harness 写一个 `HFLM` 风格的 adapter（读我们的 checkpoint + tokenizer，暴露 `loglikelihood` / `generate_until`）。**2–3 人天**，之后边际成本近零。

| 层级 | 任务 | 样本量 | 前向 token（估） | 5090 耗时 | 成本 |
|---|---|---:|---:|---:|---:|
| L0 冒烟 | tinyMMLU(100) + ARC-E(500) + 20 条生成 | 620 条 | ~0.2M | <1 min | ¥0.05 |
| L1 主评测 | ARC-E/ARC-C/HS/PIQA/Wino/OBQA/LAMBADA | ~30k 条 | ~8M | ~2.5 min | ¥0.2 |
| L2 知识 | MMLU 全量 5-shot | 14,042 题 | 6.7M | ~2 min | ¥0.15 |
| L3 生成 | GSM8K 200 题（256 tok）+ IFEval 100 条 | 300 条 | ~0.3M 生成 | ~2 min | ¥0.15 |
| L4 对标 | CORE 22 任务（借 nanochat `core_eval.py`） | 22 任务 | ~30M | ~10 min | ¥0.6 |
| **合计** | | — | **~45M** | **~17 min** | **¥1.2** |

**结论：一次完整评测 ≈ ¥1–2、20 分钟以内**；L0+L1 的日常回归 ≈ ¥0.3、3 分钟。**这是整个项目里性价比最高的投入，应先做。**

**统计规则**：
- 决策用 L1+L2 全量集；L0 只用于"跑通/回归"。
- 单次差异 <1.5 分不决策；关键节点跑 2 个种子（成本 ×2，仍 <¥5）。
- 报告 `acc` 与 `acc_norm` 两个口径（小模型对长度归一化敏感）。
- 所有任务固定 few-shot 数与模板，写进 config，避免"模板漂移"导致的假涨跌。

### 4.2 (b) 我们的 benchmark 预期分数带

**方法**：用同规模、单卡、相近 token 量的公开锚点插值，不外推（我们自己的 P1 教训：短跑外推长跑会偏乐观）。锚点：
- GPT-2 small（124M，~9B tokens，L20 harness 零样本）：ARC-C 22.6 / ARC-E 39.7 / HS 31.4 / PIQA 62.1 / Wino 50.7 / LAMBADA 30.8，mean6 **0.3954** `A`
- L20-Edu-135M（134.5M，13B tokens，单卡 L20）：ARC-C 28.7 / ARC-E 49.6 / HS 32.4 / PIQA 61.5 / Wino 50.8，mean6 **0.4150** `A`
- SmolLM2-135M（2T tokens）：ARC-E 58.5 / HS 43.0，mean6 **0.4917** `A`
- Pythia-1.4B（300B tokens）：MMLU 25.56 / HS 54.96 / ARC 32.68 / GSM8K 0.83 `A`
- H2O-Danube2-1.8B（3T tokens）：MMLU 40.20 / HS 72.95 / ARC 43.34 / GSM8K 29.80 `A`

**估计带（`C`，非实测；误差 ±3–5 分）**：

| 指标 | 估计带 | 主要依据 |
|---|---:|---|
| MMLU (5-shot) | **25–27** | 25 是 4 选 1 随机；Pythia-1.4B/300B 才 25.6；我们 token 少 37× |
| ARC-E | **45–52** | GPT-2 small 39.7，L20-135M 49.6；我们参数 2× 但 token 少 |
| ARC-C | **24–28** | GPT-2 small 22.6，L20 28.7 |
| HellaSwag | **30–35** | GPT-2 small 31.4，L20 32.4；这类任务靠表层统计，token 量决定 |
| PIQA | **60–64** | 几乎所有 ≥100M 模型都在 60±3 |
| WinoGrande | **50–53** | 同上，接近随机 50 |
| GSM8K | **0–1%** | Pythia-1.4B/300B 为 0.83；我们无 CoT 训练 |
| CORE | **0.10–0.15** | 由 mean6 与 nanochat 曲线插值；CORE 是中心化准确率，不可与 mean6 直接比 |
| mean6 | **0.39–0.43** | 介于 GPT-2 small 与 L20-135M 之间 |

**读法**：
- 如果实测落在带内 → 模型"正常"，无需怀疑训练 bug；差距是算力/数据，不是技巧。
- 如果 MMLU <25 → 有 bug（评测模板/分词/位置编码），先修评测。
- 如果 ARC-E >55 或 HS >40 → 大概率数据泄漏（ClimbMix 与评测集重叠），要查。
- **SFT 后**：MMLU/ARC 基本不动（±0.5），GSM8K 可能 +2–6，IFEval 从 0 到 10–20。**不要用 MMLU 判断 SFT 成败**，用 IFEval/复读率/收尾率。

### 4.3 生成健康度（自建，¥0.2/次，直接对应"能力很弱"）

固定 30 条 prompt（事实问答、算术、代码、闲聊各 7–8 条），贪心 + temperature 0.7 各跑一次，自动统计：
1. **复读率**：4-gram 重复占比 >30% 的样本比例；
2. **收尾率**：是否在 256 token 内产出 EOS；
3. **格式遵循率**：要求"只输出数字/只输出代码块"的遵守比例；
4. **算术正确率**：20 道一位数/两位数加减（这是 288M 唯一可能学会的"推理"）。
这 4 个数比 MMLU 更能反映"能不能用"，且完全自动化。

---

## 5. 风险与反例（写给自己看的）

1. **等算力 KD 无用**（MiniPLM，2.1.1）：任何"接教师 logits 一起预训"的方案，先问"教师前向的 FLOPs 从哪来"。
2. **少量 SFT 会伤小模型**（Phi-4-Mini + LIMO：AIME 10.0→6.7，MATH 71.8→57.8）：数据分布必须匹配底座能力；宁可少而"格式对"，不要多而"长 CoT"。
3. **RLVR 只锐化不创造**（2504.13837、2611.02574）：底座没有的能力，RL 变不出来；0 奖励 → 零梯度。
4. **bpb 跨词表/跨语料不可比**（2611.04215；nanochat 换数据集 bpb −0.03 而 CORE 不动）：不要再用 bpb 0.718 作为唯一北极星。
5. **灾难性遗忘**：288M 容量小，SFT/KD 都会覆盖预训练分布；混料 ≥50% 预训练数据 + 低 LR + 早停。
6. **教师不是越强越好**：DSL（2502.08606）指出固定 token 预算下学生越大、教师 cross-entropy 应越小；对 288M，1.7B–4B 教师够用，671B 教师的长 CoT 反而分布不匹配。
7. **数据泄漏**：ClimbMix 与 ARC/HS/MMLU 可能重叠；跑评测前做 13-gram 去重检查，否则分数虚高。
8. **API 成本/限流**：DeepSeek V4.1-Flash 的 top-20 logits 若按 token 计费，1B tokens 可能 ¥1000+；本地教师是唯一可控路径。
9. **存储**：1B tokens × top-20 × 8B = 160GB；top-1000 就 8TB。必须按 2611.14850 的 0.01% 结论压到 top-13~20。
10. **单卡 24GB 的隐藏成本**：教师+学生同时在线（OPD）时，seq 8192 的激活会挤爆；离线 logits 更稳。

---

## 6. 参考文献

### 6.1 预训练蒸馏 / KD 理论

1. **MiniPLM: Knowledge Distillation for Pre-Training Language Models** — 2024-10（ICLR'25）. https://arxiv.org/abs/2410.17215 · 高引用/经典
2. **Gemma 2: Improving Open Language Models at a Practical Size** — 2024-07. https://arxiv.org/abs/2408.00118 · 高引用（Table 6：+7.4）
3. **Gemma 3 Technical Report** — 2025-03. https://arxiv.org/abs/2503.19786 · 官方报告
4. **Phi-3 Technical Report** — 2024-04. https://arxiv.org/abs/2404.14219 · 高引用
5. **Textbooks Are All You Need (phi-1)** — 2023-06. https://arxiv.org/abs/2306.11644 · 经典
6. **Distillation Scaling Laws** — 2025-02（ICML'25）. https://arxiv.org/abs/2502.08606 · 高引用/经典
7. **On-Policy Distillation of Language Models (GKD)** — 2023-06. https://arxiv.org/abs/2306.13649 · 经典
8. **MiniLLM: On-Policy Distillation of Large Language Models** — 2023-06. https://arxiv.org/abs/2306.08543 · 经典
9. **Towards Cross-Tokenizer Distillation: ULD Loss** — 2024-02. https://arxiv.org/abs/2402.12030 · 经典
10. **On-Policy Distillation（博客）** — Thinking Machines, 2025-10. https://thinkingmachines.ai/blog/on-policy-distillation/ · 2025 综述级 `B`

### 6.2 2026 新方法（KD / 跨 tokenizer / 压缩）

11. **Effective Teacher Support for Reasoning Distillation** — 2026-11. https://arxiv.org/abs/2611.14850 · **2026 新**（0.01% 词表）
12. **Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization (BPM)** — 2026-07. https://arxiv.org/abs/2607.22334 · **2026 新**
13. **CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer** — 2026-10. https://arxiv.org/abs/2610.28895 · **2026 新**
14. **CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation** — 2026-10. https://arxiv.org/abs/2610.13257 · **2026 新**
15. **Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families** — 2026-06. https://arxiv.org/abs/2606.09456 · **2026 新**
16. **Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall（Switch Distillation）** — 2026-09. https://arxiv.org/abs/2609.01532 · **2026 新**
17. **Rethinking Supervision Scaling in On-Policy Distillation** — 2026-11. https://arxiv.org/abs/2611.20687 · **2026 新**（背景）
18. **Memorization Dynamics in Knowledge Distillation for Language Models** — 2026-01. https://arxiv.org/abs/2601.15394 · 2026

### 6.3 小模型后训练 / 推理

19. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** — 2025-01. https://arxiv.org/abs/2501.12948 · 超高引用（1.5B 蒸馏 28.9 AIME）
20. **LIMO: Less is More for Reasoning** — 2025-02. https://arxiv.org/abs/2502.03387 · 高引用
21. **s1: Simple test-time scaling** — 2025-01. https://arxiv.org/abs/2501.19393 · 高引用
22. **Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning LMs in Math** — 2025-04. https://arxiv.org/abs/2504.21233 · 关键反例
23. **RLVR Sharpens the Pencil, Not the Mind** — 2026-11. https://arxiv.org/abs/2611.02574 · **2026 新**（OLMo-2-1B）
24. **Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't** — 2025-03. https://arxiv.org/abs/2503.16219 · 1.5B/$42
25. **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?** — 2025-04. https://arxiv.org/abs/2504.13837 · 高引用（pass@k）
26. **Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes** — 2026-04. https://arxiv.org/abs/2604.18381 · **2026 新**
27. **Phi-4-Mini Technical Report** — 2025-03. https://arxiv.org/abs/2503.01743 · 官方

### 6.4 小模型配方与评测

28. **SmolLM2: When Smol Goes Big** — 2025-02. https://arxiv.org/abs/2502.02737 · 高引用
29. **SmolLM3（HF blog）** — 2025-07. https://huggingface.co/blog/smollm3 · 官方博客 `B`（数字未逐项核实）
30. **Qwen3 Technical Report** — 2025-05. https://arxiv.org/abs/2505.09388 · 超高引用
31. **MiniCPM: Unveiling the Potential of Small Language Models** — 2024-04. https://arxiv.org/abs/2404.06395 · 高引用
32. **MiniCPM4: Ultra-Efficient LLMs on End Devices** — 2025-06. https://arxiv.org/abs/2506.07900 · 2025
33. **H2O-Danube-1.8B Technical Report** — 2024-01. https://arxiv.org/abs/2401.16818 · 高引用
34. **MobileLLM: Optimizing Sub-billion Parameter LMs** — 2024-02. https://arxiv.org/abs/2402.14905 · 经典
35. **L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small LM** — 2026-06. https://arxiv.org/abs/2606.22189 · **2026 新**（最贴近我们的锚点）
36. **tinyBenchmarks: evaluating LLMs with fewer examples** — 2024-02. https://arxiv.org/abs/2402.14992 · 高引用
37. **DataComp-LM (DCLM/CORE)** — 2024-06. https://arxiv.org/abs/2406.11794 · 高引用
38. **Tokenizer Myths and Where To Find Them** — 2026-11. https://arxiv.org/abs/2611.04215 · **2026 新**（bpb 批判）
39. **nanochat（代码/README/leaderboard）** — 2026. https://github.com/karpathy/nanochat · 工程基线 `B`

### 6.5 合成数据

40. **Cosmopedia（HF blog）** — 2024-02. https://huggingface.co/blog/cosmopedia · 官方博客 `B`
41. **FineMath / Stack-Edu / SmolTalk** — 随 SmolLM2 发布（见 28）
42. **STaR: Bootstrapping Reasoning With Reasoning** — 2022-03. https://arxiv.org/abs/2203.14465 · 经典（拒绝采样 SFT 的源头）
43. **Large Language Models Can Self-Improve** — 2022-10. https://arxiv.org/abs/2210.11610 · 经典（自生成 + 过滤）

> 未采用/未证实：Phi-4-mini 的纯文本 MMLU 具体值（正文表格未含，未引用）；SmolLM3 的各项分数（仅博客，未逐项核）；Llama 3.2 的蒸馏细节（仅官方博客/模型卡，`B` 级）；2026 年 OPD 论文的绝对收益数字（多数只给相对结论，未引用具体值）。
