# 小模型（SLM）训练方法调研 —— 能借鉴什么（2026-11-18）

> 目的：在 P1 8B 完赛（raw 1.0662 / EMA 1.0620，全量 val 复评 raw 1.06235）之后，
> 系统梳理"训练小模型"这一方向的高引用/热门方法，筛出我们（288M active、单卡 5090、
> 每轮预算 ¥100–300）真正用得上、且成本可承受的杠杆。

## 0. 一句话结论

**现代"小模型"的 token 预算在 1T–11T 量级（每参数 100–1000 tokens），而我们是 8B tokens /
288M active ≈ 28 tokens/参数——整整差了 1.5–3 个数量级算力。** 所以：
- 几乎全部 SLM 论文的成功，首要原因是**海量 token + 严格数据筛选**，而不是某个训练 trick；
- 在我们预算内，**数据侧（质量筛选/难度重采样/合成数据混入）是唯一可能拿到 0.02–0.06 bpb 的杠杆**；
- 架构类（deep-thin、GQA、权重共享）可以在 0.1–0.3B 锚点上便宜验证，预期 0.01–0.03 bpb；
- 蒸馏（KD）对"能力"提升明显、对 bpb 提升有限，且**在线 KD 在固定算力下不划算**（见 MiniPLM）。

## 1. 代表性工作与对其训练配方的事实梳理

| 工作 | 规模 / token 数 | 核心做法 | 关键结果（引用自 abstract） |
|---|---|---|---|
| **SmolLM2** (2025, [2502.02737](https://arxiv.org/abs/2502.02737)) | 1.7B / **~11T tokens** | 多阶段训练：web + math/code/instruction 混合（FineMath、Stack-Edu、SmolTalk），小规模消融定配比，逐阶段调 mix | 超过 Qwen2.5-1.5B / Llama3.2-1B；证明"小模型要**过度训练**" |
| **MiniCPM** (2024, [2404.06395](https://arxiv.org/abs/2404.06395)) | 1.2B / 2.4B（非嵌入） | **WSD LR 调度**（warmup-stable-decay）；model wind tunnel 小模型试超参；数据/模型双轴 scaling law | 2.4B 接近 7B–13B；**compute-optimal 数据/模型比远高于 Chinchilla**（即更该多训 token） |
| **Phi-3** (2024, [2404.14219](https://arxiv.org/abs/2404.14219)) | 3.8B / **3.3T tokens** | "重过滤 web + 合成数据"混训（textbook 路线） | 69% MMLU，手机可部署 |
| **TinyLlama** (2024, [2401.02385](https://arxiv.org/abs/2401.02385)) | 1.1B / ~1T tokens（≈3 epochs） | 开源全流程（FlashAttention/Lit-GPT 高效训练） | 同尺寸 SOTA；证明**重复 ~3 epoch 无害** |
| **MobileLLM** (ICML'24, [2402.14905](https://arxiv.org/abs/2402.14905)) | 125M / 350M | **深而窄**架构（30–42 层）+ embedding sharing + GQA；block-wise 权重共享（LS 变体） | 比同尺寸 SOTA +2.7%/4.3%；LS 再 +0.7%/0.8% → **亚十亿参数下架构很重要** |
| **Gemma 2** (2024, [2408.00118](https://arxiv.org/abs/2408.00118)) | 2B / 9B | **2B/9B 直接用 KD 训练**（不从零做 next-token）；interleaved local-global attention + GQA | 同尺寸最强，可对标 2–3× 大模型 |
| **DCLM** (NeurIPS'24, [2406.11794](https://arxiv.org/abs/2406.11794)) | 412M–7B / 2.6T tokens | 240T Common Crawl 上的**数据筛选基准**；结论：**model-based filtering 是质量关键** | DCLM-Baseline 7B：64% MMLU 5-shot；比 MAP-Neo 少 40% 算力；比 Llama3-8B 少 6.6× 算力 |
| **Scaling Data-Constrained** (NeurIPS'23, [2305.16264](https://arxiv.org/abs/2305.16264)) | 最多 9B 模型 / 900B tokens | 数据受限下重复训练的影响；缓解手段：加入 code 数据、去掉过强过滤 | **≤4 epoch 重复几乎无损**；再多则价值趋零；给出考虑重复价值的 scaling law |
| **MiniPLM** (ICLR'25, [2410.17215](https://arxiv.org/abs/2410.17215)) | students 200M/500M/1.2B；teacher 1.8B | 离线 **Difference Sampling**：reward = log p_teacher − log p_ref（104M 参考模型），Pile 100B 取 top 50% → 50B 精炼 | 9 下游任务提升；**数据需求降低 2.4×**；跨模型家族可行；在线 KD 不划算 |
| **TinyStories / Cosmopedia**（合成数据路线） | 1M–30B | 用强模型生成"教科书/故事"式干净语料 | 小模型在窄域上表现远超同尺寸；Cosmopedia 是公开的 30B 合成教科书语料 |

## 2. 方法分类与"我们能不能用"

### 2.1 数据侧（最高优先级）
1. **质量分类器过滤**（DCLM/FineWeb-Edu 路线）：训练一个 quality/fastText 分类器或用现成 FineWeb-Edu 分数，
   去掉低教育价值网页。对我们的意义：我们 8.5B 是 ClimbMix 原始 web，**大概率含大量低质量/重复内容**。
   成本：分类器推理可 CPU/GPU 做，8.5B tokens 的分数 ≈ 1–2h/¥5；重采样后 8B 重训 ¥110。
   风险：ClimbMix 相对 FineWeb 已是"清洗过"的，收益可能小于 DCLM 的绝对值。
2. **难度/信息量重采样**（MiniPLM 路线）：用 teacher 和 reference 的 log-prob 差做 reward。我们可用
   **Qwen3-1.7B 作 teacher + 我们自己的 288M 作 reference**（避免再训 104M），对 2–3B tokens 打分。
   成本：teacher fwd ≈ 4–6h/¥15–20（bf16 24GB 可放），只对子集打分再按比例采样。
3. **多阶段混合 + 合成数据**（SmolLM2/Phi 路线）：末段 10–20% 混入高质量 math/code/instruction 合成语料
   （FineMath/Stack-Edu/Cosmopedia 现成，下载即用）。对我们的 bpb（web val）可能**不降甚至略升**，
   但对下游能力/生成质量提升明显——如果目标从"bpb<0.9"转向"可用的小模型"，这是最便宜的路。
4. **数据重复**：我们目前 0.93 epoch（无重复），按 data-constrained 结论，**加到 4 epoch 内几乎无损**
   → 如果只是想要更多训练 token，先确认是否值得（价值 ≈ 新数据的一半以上，值得利用现有 8.5B 语料）。

### 2.2 训练策略
- **WSD 调度**：我们已在用（MiniCPM 同款）；可借鉴"stable 阶段 ckpt 可反复续训/消融，只需重跑短 decay"
  ——以后所有锚点实验都可以用这一招省时间。
- **超参迁移/wind tunnel**：MiniCPM 用 0.1B 级代理模型定超参再放大。我们已经在做（S1 LR 扫描），
  但应该把"锚点预算"从 0.2–0.6B 提高到 ~1–2B 以贴近长跑行为（P1 的教训：0.2–0.6B 外推到 8B 偏乐观）。
- **过度训练**：MiniCPM/SmolLM2/TinyLlama 都是"每参数 100–1000 tokens"。我们 28 tokens/参数远低于
  典型 SLM 配方——这不是"我们配方错"，而是**预算决定**。追赶的边际收益：按我们实测曲线 β≈0.06–0.08，
  8B→20B tokens 约 −0.05 bpb（¥165），20B→40B 再约 −0.04（¥330）——**到 0.89 要 100B+ tokens 或更大模型**。

### 2.3 架构侧（亚十亿参数才重要）
- **深而窄 + GQA + embedding sharing**（MobileLLM）：我们 d24_c1 是 dim 1024 / 24 层，深度不算深；
  可做 0.1–0.3B 锚点对比"24 层 vs 36/48 层（同参数）"，成本 ¥10–20/组。
- **Gemma 2 用 KD 训 2B**：说明业界把 KD 当"小模型首选"，但也说明"到那个量级"才值得投入。
- 我们的 Engram（稀疏 220M 参数）+ mHC 是差异化资产，但在 iso-FLOPs 对比里没有体现出优势（P1 结果）；
  要验"Engram 是否值得"，最好的实验是**同 token 预算、开/关 Engram 的 A/B**（0.4B 锚点，¥15–20）。

## 3. 对 R93/P2 的落地清单（按性价比排序）

| # | 实验 | 成本 | 预期收益 | 决策标准 |
|---|---|---|---|---|
| 1 | **E1 数据难度筛选**（用现有 P1 模型对 2B tokens 打分，去"太易/太噪"，0.4–0.6B 锚点 A/B） | ¥20–30 | −0.01~−0.03 bpb（等效 1.2–1.5× 数据） | 0.4B 锚点上 ≥0.01 优势才扩大 |
| 2 | **架构锚点**：deep-thin（同参数 36/48 层）、Engram on/off | ¥30–40 | ±0.01~0.03 bpb（可能发现我们架构不是最优） | 同 token 预算下比较 |
| 3 | **合成数据混入**：末段 10–20% FineMath/Stack-Edu/Cosmopedia | ¥15–25（需先下载训练） | bpb 可能持平，能力明显提升 | 以能力 eval 为主 |
| 4 | **MiniPLM-lite**（Qwen3-1.7B teacher + 我们的 288M reference，2–3B 语料打分，top 50–70% 重采样） | ¥40–60 | −0.02~−0.04 bpb（若论文迁移） | 0.6B 锚点 ≥0.015 优势 |
| 5 | **多 token 预算**：8B→20B（重复或扩语料） | ¥140–180 | −0.04~−0.06 bpb | 只在 1–4 验证通过后 |
| 6 | **更大模型 + KD**：600–730M active × 10–20B tokens（Gemma 2 路线） | ¥300–600+ | 可能到 0.85–0.95 | 需先有锚点校准 + 预算 |

**明确不推荐**：在线 KD（MiniPLM 图 2 证明固定算力下不划算）；盲目放大模型（8B 数据下模型越大越饿）；
>4 epoch 重复（data-constrained 结论价值趋零）。

## 4. 关键认知修正（本轮调研最重要的收获）

1. **我们离"现代小模型"的算力差距是数量级级的**：nanochat d24（730M, ~21.5B tokens, 4.33e19 FLOPs）
   是 30× 我们的算力；SmolLM2-1.7B（11T tokens）是 1000× 量级。bpb 1.06 vs 0.718 的差距首先是算力差距。
2. **在数字上追 bpb 到 0.9 以下，靠 trick 组合是做不到的**：数据筛选+合成数据+KD 加一起乐观 0.05–0.10 bpb
   （1.06 → ~0.96–1.01），仍到不了 0.9；要 0.89 需 20B+ tokens 或 600M+ 参数。
3. **相对目标应该改成"同预算下的效率"**：我们每 ¥100 能买到多少 bpb？P1 实测 8B 长跑 ¥128 ≈ 1.06 bpb
   → ~¥74/0.01 bpb（后段）。这是可对外叙述、也可内部优化的指标；数据筛选若成功，可以直接换算成
   "同 bpb 省 X% 成本"。

## 5. 参考链接

- MiniPLM: <https://arxiv.org/abs/2410.17215> · <https://github.com/thu-coai/MiniPLM>
- SmolLM2: <https://arxiv.org/abs/2502.02737>
- MiniCPM: <https://arxiv.org/abs/2404.06395>
- Phi-3: <https://arxiv.org/abs/2404.14219>
- TinyLlama: <https://arxiv.org/abs/2401.02385>
- MobileLLM: <https://arxiv.org/abs/2402.14905>
- Gemma 2: <https://arxiv.org/abs/2408.00118>
- DCLM: <https://arxiv.org/abs/2406.11794>
- Scaling Data-Constrained LMs: <https://arxiv.org/abs/2305.16264>
