# 数据侧调研：288M active / 8.57B tokens 单卡小模型（2026-11-18）

> 范围：数据筛选/质量分类、合成数据、数据混合与 curriculum、数据重复、离线数据蒸馏，以及 2025–2026 新工作。
> 结论先行：**在我们的规模上，数据侧不是"再刷一个 SOTA 过滤器的收益"，而是"用有限 token 换更高效的数据使用方式"**。最值得先做的是 ①重复标定后跑 2 epoch（置信度最高）②末段 curriculum/domain upsampling（最便宜）③MiniPLM-lite 的 teacher–reference 差异采样（上限最高）。DCLM/FineWeb-Edu 这类"从 raw CC 里筛高质量"的方法，因为 ClimbMix 上游已经 curated，直接套用的增量收益明显变小。

---

## 0. 结论摘要（TL;DR）

1. **我们的 bpb 曲线在 8B 处仍下降，但局部 β 只有 0.06–0.08**（P1：4B 1.1176 → 6B 1.0905 → 8B 1.0662）。按这个斜率，**最有把握的"数据侧"杠杆是把 8.57B 用第二遍**：2 epoch ≈ 17.1B tokens，预期 −0.02～−0.04 bpb，成本 ≈¥230/76h，置信度高。它也是所有筛选/curriculum 实验的控制基线。
2. **ClimbMix 是 NVIDIA 的 curated 400B mix，不是 raw Common Crawl**。DCLM/FineWeb-Edu 的收益是"raw CC → 过滤后"的收益；在 ClimbMix 8.57B 子集上再做一次全量过滤，预期只剩 −0.005～−0.015 bpb，而 8.57B 的 LLM 打分成本可能接近甚至超过一次 8B 训练。**不要默认"再过滤一遍就能复现 DCLM 的 +6.6pp"**。
3. **最值得做的 3 个数据侧改动**：
   - **R1（先做）**：重复标定 → 2 epoch。0.5B 锚点 4 臂 ≈¥70–100；8B 确认 ≈¥230。预期 −0.02～−0.04 bpb，置信度高。
   - **R2（最便宜）**：末段 10–20% 放高质量数据（quality-ascending curriculum），配合更温和的 WSD decay/EMA。0.5B 锚点 4 臂 ≈¥40–60；8B 确认 ≈¥115–145。预期 −0.005～−0.02 bpb，置信度中。
   - **R3（上限最高）**：MiniPLM-lite（1.7B teacher + 50–100M reference + 0.1B proxy 回归），选 top 25% 数据跑 4 epoch。0.5B 锚点 ≈¥85–145；8B 确认 ≈¥170–230。预期 −0.02～−0.05 bpb，置信度低–中。
4. **在"288M 参数、8B tokens"这个规模大概率无效/不划算**：DoReMi（proxy 280M ≈ 目标模型）、RegMix（512×1B proxy ≈ 512B tokens）、全量 SemDeDup（ClimbMix 上游已 curated）、在线 KD/logit 蒸馏（固定算力下不划算）、7B judge 全量打分（成本 > 训练）、用 7B 生成 8.57B 合成数据（成本 > 训练）、>4 epoch 或"中等子集×中等重复"（2026 新证据显示这是最差区间）。
5. **一个必须记住的数字**：在我们这个点上，按我们的外推，**固定 8B token 预算内数据侧方法的合理上限约 −0.02～−0.04 bpb；把 2 epoch 算进来可到 −0.05～−0.07**（约 1.066 → 1.00–1.01）。到 0.89 需要 20B+ tokens 或 600M+ active，而不是某个数据 trick。

---

## 1. 我们的 setting、成本模型与判断标准

### 1.1 基线

| 项 | 值 |
|---|---|
| 模型 | d24_c1 dense，288M active + Engram 220M 稀疏表；dim 1024 / 24 层 / V=41,280 / seq 8192 |
| 语料 | `p2_climbmix_40k`，8.568B tokens；本轮实际训练 7.998B tokens（0.934 epoch，无重复） |
| 训练 | 7.998B tokens / 3,814 步 / 35.5h / 62,569 tok/s / MFU 51.6% |
| 结果 | raw bpb **1.0662**（bits/byte；val loss ≈3.295 nats/token），EMA(0.995) **1.0620** |
| 曲线 | 1B 1.2771 / 2B 1.1897 / 4B 1.1176 / 6B 1.0905 / 8B 1.0662；局部 β 0.102→0.090→0.060→0.078 |
| 已有技巧 | WSD、z-loss、EMA、fp8 head、融合 kernel；LR 8e-4 已探针确认最优 |
| 预算 | 单次实验 ¥100–300；本轮 8B 直接硬件 ≈¥128 |

> 注意：这里的 bpb 是 **bits per byte**，不是 bits per token；3.295 nats/token ÷ ln2 ÷ ~4.46 bytes/token ≈ 1.066。

### 1.2 成本模型（估算，±2×）

按 P1 的 62.5k tok/s、¥3.1/h 全包价估算：

| 操作 | 吞吐假设 | 8.57B tokens | 1B tokens | 0.5B tokens | 0.1B tokens |
|---|---|---:|---:|---:|---:|
| 学生训练 | 62.5k tok/s | 38h / ¥115 | 4.4h / ¥14 | 2.2h / ¥7 | 0.45h / ¥1.5 |
| 1.7–1.8B teacher 推理 | 8–12k tok/s | 200–300h / ¥620–930 | 23–35h / ¥70–110 | 12–17h / ¥35–55 | 2.3–3.5h / ¥7–11 |
| 7B teacher 推理 | 2.5–4k tok/s | 600–950h / ¥1,900–3,000 | 70–110h / ¥220–350 | 35–55h / ¥110–170 | 7–11h / ¥22–35 |
| 50–100M reference 打分 | 100–200k tok/s | 12–24h / ¥40–100 | 1.4–2.8h / ¥5–12 | 0.7–1.4h / ¥2–6 | 0.1–0.2h / ¥0.5–1 |
| 0.5B 分类器打分 | 20–40k tok/s | 60–120h / ¥190–380 | 7–14h / ¥22–45 | 3.5–7h / ¥11–22 | 0.7–1.4h / ¥2–5 |
| fastText/CPU 打分 | — | 5–15 CPU·h / ¥5–20 | — | — | — |

**判断原则**：数据处理成本 > 训练成本的方案，直接淘汰；能在 0.1B proxy 上做、再用线性头外推的，不要全量跑 teacher。

### 1.3 收益标定：−0.01 bpb 到底有多大

P1 实测：4B→6B（1.5×）−0.027 bpb；6B→8B（1.33×）−0.024 bpb。按局部 β≈0.07：

- **唯一数据翻倍**（8.57B→17.1B 新数据）：约 −0.03～−0.05 bpb；
- **重复第二遍**（同样 17.1B，但第二遍是重复）：约 −0.02～−0.04 bpb（重复折扣 20–40%）；
- **有效数据 ×1.2–1.5**（筛选/curriculum/蒸馏）：约 −0.005～−0.02 bpb；
- **MiniPLM 声称的 2.4× 数据效率**：理论 −0.04～−0.06；在我们的规模/重复约束下打折到 **−0.02～−0.05**。

> 这些是外推区间，不是承诺。0.5B 锚点的 β 比 8B 大（P1 教训：短跑 β=0.154，长跑 0.06–0.08），所以锚点只能筛方向，最终必须 1B+ 或 8B 确认。

### 1.4 ClimbMix 的特殊性（决定了很多方法的优先级）

- `nvidia/Nemotron-ClimbMix` 是 NVIDIA 的 **400B token curated mix**（nanochat 使用；本地记录其 bpb 0.748 来源），不是 raw CC。
- 上游至少做了配比/筛选；去重与过滤的具体细节以其 dataset card 为准（未逐项核实）。我们的 8.57B 是其中约 2.1% 的子集（具体抽样方式未证实）。
- **直接影响**：
  - DCLM/FineWeb-Edu 的"raw CC → 高质量"收益不能直接套用；我们做的是 **secondary filtering**，收益更小。
  - ClimbMix 的配比已经均衡；DoReMi/RegMix 的"找更好的 domain mix"空间变小，且 proxy 成本高。
  - 如果子集保留了来源/质量标签，curriculum 和 top-k 筛选会便宜很多；**第 0 步应该先审计**（见 §5.3）。

### 1.5 证据分级

- **A**：与我们同量级（≤1B 模型、≤30B tokens）或有 400M/8B 直接锚点；
- **B**：1B–7B / 100B–1T，需外推；
- **C**：7B+ / 1T+ 或工业报告，外推风险大；
- **D**：预印本/博客/未证实。

---

## 2. 概览表

> 引用热度是定性判断（"经典/高引用"= 领域内广泛引用；"2026 新"= 尚无引用量积累），未逐篇抓取 citation count。2026 条目多为预印本/会议论文，尚未经过独立复现，按方向信号使用。成本按 §1.2 估算；收益是 8B 规模的外推区间。

| 方法 | 年份/热度 | 做法一句话 | 关键证据（规模→收益） | 我们可借鉴性 | 估算成本 | 预期 bpb/能力收益（置信度） |
|---|---|---|---|---|---|---|
| **DCLM** | 2024，经典/高引用 | 在 240T CC 池上做 model-based filtering + 去重 + 配比，产出 DCLM-Baseline | 412M/8.2B 锚点；7B/2.6T 达 64% MMLU，比 MAP-Neo +6.6pp 且少 40% 算力 | **需改造**：ClimbMix 已 curated；可做轻量 secondary filter | 分类器 8.57B 打分 ¥190–380 + 训练 ¥110 | −0.005～−0.015（中低） |
| **FineWeb-Edu** | 2024，经典/高引用 | 用 Llama-3-70B 标注训练教育质量分类器，从 FineWeb 取 1.3T 教育文本 | 350B 采样训练即优于 FineWeb/其他开放集；MMLU/ARC 明显提升 | **需改造**：只取"教育"会窄化；可作 quality 信号之一 | 分类器打分 ¥190–380；或复用现成分数 ¥0–20 | −0.005～−0.015（中低） |
| **QuRating** | 2024，高引用 | GPT-4 成对比较训练 QuRater，给 260B 语料打 4 类质量分，选 30B 训练 1.3B | 最佳（教育价值）≈ uniform 多训 50% steps；quality-as-logits 采样更好；curriculum 不改数据也有增益 | **可直接用/改造**：用小型 judge 或现成打分做排序 | 打分 ¥50–200；训练 ¥110 | −0.005～−0.02（中） |
| **DoReMi** | 2023，经典 | 280M proxy 用 Group DRO 学 domain weights，再重采样训练 8B | Pile 上 +6.5pp 平均 few-shot；达到 baseline 少 2.6× steps；GLaM 匹配下游调权 | **不适用/低优先**：proxy 280M ≈ 我们目标 288M，省不了算力 | proxy 8B ≈¥110 + 主跑 ¥110 | 不确定；不建议（低） |
| **DSIR** | 2023，经典 | 用 hashed n-gram 重要性重采样，让选出的数据接近目标分布 | 100M docs/4.5h；8 个目标域接近专家 curation；通用预训练 GLUE +2–2.5% | **需改造**：要有明确 target；ClimbMix 已均衡，收益有限 | CPU 4.5h/100M docs；全量 ¥5–20 | −0.003～−0.01（低） |
| **SemDeDup** | 2023，经典 | 用预训练 embedding 做语义去重，删"语义重复"而非精确重复 | LAION 删 50% 数据性能几乎不降，训练时间减半；C4 上优于先前方法 | **不适用**：ClimbMix 上游已 curated；8.57B 的 embedding+kNN 很贵 | embedding+kNN ¥100–300+ | 不确定；不建议（低） |
| **Nemotron-CC** | 2024/2025，高关注 | classifier ensemble + 合成改写 + 少用 heuristic，保留更多 token | 8B/1T 高质量子集比 DCLM +5.6 MMLU；6.3T 全量 4× unique real tokens；8B/15T 超 Llama 3.1 8B +5 MMLU | **可借鉴思路**：不要过度过滤；质量+数量平衡 | 全量管线 ¥1,000+；子集复用 ¥0–50 | 参考价值 > 直接复现 |
| **WebGraphMix** | 2026 新 | 用 Common Crawl host-level web graph 的 centrality 调 central/peripheral 比例 | 400M/8B：1:1 混合 41.4% vs uniform 39.8%；再叠 quality 43.8% | **可直接用/改造**：若有 host/URL 元数据，几乎零成本 | 图计算 CPU ¥5–20 + 训练 ¥110 | +1–4pp 下游；bpb −0.005～−0.015（中） |
| **SPOKES** | 2026 新 | 直接优化 diversity（G-Vendi）+ quality 联合选数据 | DCLM/FineWeb 上 quality+diversity +1.5/+1.4 points，优于 semantic dedup/纯 quality | **可改造**：用 embedding 多样性约束，防止过滤窄化 | embedding + 选择 ¥50–150 | −0.005～−0.015（中低） |
| **CuraWeb** | 2026 新 | quality/redundancy/diversity 联合优化，2T 语料 | 3B 规模平均 +1.8%，知识/推理任务更明显 | **思路可借鉴**：多目标而非单一 quality | 全量管线 ¥1,000+；子集 ¥0–50 | 参考价值 > 直接复现 |
| **DataOrchestra** | 2026 新 | 逐 example 决定 drop/clean/rewrite，LLM 选操作 | 0.5B–7B from scratch，11 benchmarks 稳定优于单一处理；减少无效处理算力 | **需改造**：需要一套小工具模型；成本高 | 处理 ¥300–1,000+ | 能力 +1–3pp；bpb 不确定（低） |
| **Cosmopedia** | 2024，高引用 | Mixtral-8x7B 生成教科书/博客/故事，25B tokens | 30M files/25B tokens；cosmo-1b 在 ARC/MMLU 超 TinyLlama-1.1B，接近 Qwen-1.5-1B（部分） | **需改造**：通用 web bpb 未必降；适合能力/领域 | 现成下载 ¥0–20；自生成 ¥500+ | bpb ±0.01；能力 +2–5pp（低） |
| **TinyStories** | 2023，经典 | GPT-3.5/4 生成 3–4 岁词汇的短故事 | <10M 参数模型可生成连贯多段故事；窄域合成数据有效性证明 | **不适用**：域不匹配（故事 vs web） | ¥0–10（现成） | 对我们的 web bpb 基本无效 |
| **Phi-1 textbook** | 2023，经典 | 6B "textbook quality" web + 1B GPT-3.5 合成教科书/练习 | 1.3B 模型 HumanEval 50.6%、MBPP 55.5%；350M 版 45% | **不适用/需改造**：代码域；通用 web 不适用 | 自生成 ¥500+ | 代码能力显著；bpb 不适用 |
| **SmolLM2（FineMath/Stack-Edu/SmolTalk）** | 2025，高关注 | 1.7B/11T 多阶段：web→math/code→decay 期 math 14%；自建 FineMath 54B、Stack-Edu | 超过 Qwen2.5-1.5B/Llama3.2-1B；证明小模型要"多阶段+高质量末段" | **可借鉴**：末段混 10–15% math/code；数据可下载 | 下载+tokenize ¥10–30；训练 ¥110 | bpb 持平/略降；GSM8K/MMLU +2–5pp（中） |
| **BeyondWeb** | 2025 新 | 系统研究合成数据生成：改写策略、generator、源数据 | 3B/180B 超 8B/180B on Cosmopedia；比 Cosmopedia +5.1pp、比 Nemotron-Synth +2.6pp；比 open web 快 7.7× | **需改造**：生成成本高；但"结构化格式"结论可用 | 生成 8.57B ¥2,000+；子集 ¥100–300 | 能力 +3–5pp；bpb −0.01～−0.03（低） |
| **FinePhrase** | 2026 新 | 把 web 改写成表格/数学题/FAQ/教程等结构化格式，产 486B 开放数据 | 生成 >1T tokens 做控制实验；generator >1B 无额外收益；比现有合成 baseline 好且生成成本低 30× | **可直接用/改造**：下载 486B 子集，末段混 5–15% | 下载 ¥0–20；训练 ¥110 | bpb −0.005～−0.02；能力 +2–5pp（中低） |
| **Data Mixing Laws** | 2024/ICLR25 | 拟合 mixture→loss 的函数，再嵌套 scaling law 外推 | 1B/100B RedPajama：优化 mix ≈ 多训 48% steps | **需改造**：需要 domain 标签和多次 proxy 训练 | proxy 10–20 跑 × 1B ≈¥150–300 | −0.005～−0.02（中低） |
| **RegMix** | 2024/ICLR25 | 512 个 1M 小模型 × 1B tokens 拟合回归，选 mix 再训 1B/25B | 1B/25B 在 64 个候选 mix 中最好；7B/100B 仍优于人工；匹配 DoReMi 只用 10% 算力 | **不适用**：512B proxy tokens ≈ 2,000+ GPU·h，远超预算 | ¥2,000+ | 不建议 |
| **Domain upsampling at end** | 2024 | 末段 10–20% 上采样领域数据 | 7B/1T：MMLU +6.90pp、GSM8K +8.26pp、HumanEval +6.17pp；10–20% 最优 | **可直接用**：把高质量/目标域放 decay 期 | 数据 ¥0–20 + 训练 ¥110 | 能力 +2–6pp；bpb −0.005～−0.02（中） |
| **OLMo 2 late-stage curriculum** | 2025 | 预训练后期引入 Dolmino Mix 1124 专用 mix（annealing） | 7B/13B/32B 全开放；late-stage curriculum 显著提升多 benchmark | **可直接用**：WSD decay 期换 mix | ¥110–130 | −0.005～−0.02（中） |
| **Curriculum learning (systematic)** | 2025 | 6 种难度指标、200+ 模型、最多 100B tokens；easy→hard / pacing / interleaved | 18–45% fewer steps 到 baseline；warmup curriculum 持续 +3.5%；compression/MTLD/readability 最有效 | **可直接用**：用压缩率/可读性排序，几乎零成本 | 排序 CPU ¥0–10 + 训练 ¥110 | −0.005～−0.02（中） |
| **LR decay wastes best data** | 2025 | 指出 ascending-quality curriculum 与标准 LR decay 不兼容；用温和 decay + model averaging | 1.5B/30B：+1.64% 平均（vs random），无需改数据 | **直接相关**：我们已在 WSD+EMA，需调 decay/min LR | 仅训练 ¥110–130 | 决定 R2 成败（中） |
| **Muennighoff data repetition** | 2023，经典 | 系统研究重复数据与 compute 的 scaling | 最多 900B tokens/9B 模型；≤4 epoch 重复与 unique 几乎无差；更多重复价值趋零 | **可直接用**：2 epoch 有依据；>4 epoch 不建议 | 2 epoch ¥230 | −0.02～−0.04（高） |
| **Internal Data Repetition（2026）** | 2026 新 | 用 no-repetition scaling law 量化"内部重复"的 compute-equivalent loss | 344M Qwen3-style；重复占 10% FLOPs 时，最坏等价于只用 67% FLOPs；中等子集×中等重复最差 | **风险警示**：MiniPLM 的 25%×4 必须加 random-25% 控制 | 仅分析；实验 ¥50–100 | 决定 R3 设计（中） |
| **MiniPLM** | 2024/ICLR25 | teacher–reference log-prob 差做 Difference Sampling，离线选数据；proxy 回归外推 | 200M/500M/1.2B student，1.8B teacher；数据需求 ↓2.4×；25% 数据×4 epoch 最佳；0.1B proxy 只需 0.2% 算力 | **可直接用/改造**：我们规模正好命中 200M student；teacher 可用 Qwen3-1.7B | 全量 ¥170–230 | −0.02～−0.05（低–中） |
| **SeqKD** | 2016，经典 | 用 teacher beam search 序列做学生训练数据（离线） | 经典 sequence-level KD；在 MT/摘要有效；pretraining-from-scratch 证据弱 | **需改造**：只适合有明确 teacher 的域；全量生成贵 | 生成 0.1B ¥10–30；全量 ¥2,000+ | 能力 +1–3pp；bpb 不确定（低） |

---

## 3. 逐方法细节

### 3.1 数据筛选与质量分类

#### 3.1.1 DCLM（DataComp-LM，2024，经典/高引用，证据 A）

- **做法**：240T Common Crawl 池；用 fastText/分类器做 model-based quality filtering + 去重 + 配比，产出 DCLM-Baseline。
- **关键证据**：DCLM 的 **400M-1x 锚点 = 412M 参数 / 8.2B tokens**，和我们的 288M/8B 几乎同一算力档；7B/2.6T 达 64% 5-shot MMLU，比 MAP-Neo +6.6pp 且少 40% 算力；与 Mistral-7B-v0.3（63%）、Llama 3 8B（66%）可比。
- **可借鉴性**：**需改造**。DCLM 的收益来自"raw CC → 高质量"；ClimbMix 已经 curated，我们做的是二次过滤。可借鉴的是：
  - 用**简单分类器**（DCLM 发现 bigram/fastText 往往不输复杂模型）；
  - 在 400M/8B 锚点上先验证"过滤后 1 epoch" vs "不过滤 1 epoch"；
  - 不要把过滤率拉到 90%（Nemotron-CC 的教训：过度过滤伤长 token horizon）。
- **成本**：fastText/CPU 全量打分 ¥5–20；0.5B 分类器全量 ¥190–380；训练 8B ¥110。
- **预期**：−0.005～−0.015 bpb；如果子集确实保留了大量低质尾，可能到 −0.02。置信度中低。
- **坑**：DCLM 的 412M/8.2B 锚点用的是它自己的 pool 和 tokenizer；跨数据集外推会高估。

#### 3.1.2 FineWeb-Edu（2024，经典/高引用，证据 B）

- **做法**：用 Llama-3-70B 给网页打"教育价值"0–5 分，训练分类器，从 15T FineWeb 中取 1.3T 教育文本。
- **关键证据**：350B-token 采样训练即优于 FineWeb 和其他开放数据集；MMLU/ARC 等知识推理 benchmark 明显提升；1.3T 可用。
- **可借鉴性**：**需改造**。教育过滤会窄化分布，适合"末段/能力"而不是全程；如果 ClimbMix 已含 FineWeb-Edu 类数据，增量更小。
- **成本**：分类器全量打分 ¥190–380；或复用现成 FineWeb-Edu 分数（若子集有 URL 对齐）¥0–20。
- **预期**：−0.005～−0.015 bpb；能力（MMLU/ARC）+1–3pp。置信度中低。
- **坑**：我们的 val 是 web held-out，教育过滤可能让 val bpb 不降反升（分布变窄）。

#### 3.1.3 QuRating（2024，高引用，证据 B）

- **做法**：用 GPT-4 成对比较训练 QuRater，学 4 类质量（写作风格、所需专业知识、事实/冷知识、教育价值）；给 260B 语料打分，选 30B 训练 1.3B。
- **关键证据**：最佳模型（教育价值）≈ uniform sampling **多训 50% steps**；用 quality rating 做 logits 采样比 top-k 更好（质量+多样性平衡）；rating 还能构造 curriculum，不改数据也有增益。
- **可借鉴性**：**可直接用/改造**。我们不需要 4 类全做；用一个小型 reward model 或现成 scorer 给 8.57B 排序即可。关键是"soft sampling + curriculum"，不是硬 top-k。
- **成本**：小模型打分 ¥50–200；训练 ¥110。
- **预期**：−0.005～−0.02 bpb；置信度中。
- **坑**：QuRating 的 1.3B/30B 比我们大；它的"50% more steps"是 1.3B 上的，外推到 288M 会缩水。

#### 3.1.4 DoReMi（2023，经典，证据 B）

- **做法**：280M proxy 用 Group DRO 学 domain weights，再按权重重采样训练 8B（30×）。
- **关键证据**：Pile 上平均 few-shot +6.5pp；达到 baseline 少 2.6× steps；GLaM 上匹配下游调权。
- **可借鉴性**：**不适用/低优先**。DoReMi 的卖点是"280M proxy 指导 8B"；我们的目标模型就是 288M，proxy 省不了算力。而且 ClimbMix 已经均衡，domain 重加权空间小。
- **成本**：proxy 8B ≈¥110 + 主跑 ¥110 = ¥220+；不划算。
- **预期**：不确定；不如把同样的钱花在 2 epoch 或 curriculum。

#### 3.1.5 DSIR（2023，经典，证据 B）

- **做法**：用 hashed n-gram 特征估计重要性权重，从大池中重采样出接近 target 分布的子集；无需训练 neural model。
- **关键证据**：100M docs/4.5h；8 个目标域接近专家 curation；通用预训练（target=Wikipedia+books）GLUE +2–2.5%。
- **可借鉴性**：**需改造**。DSIR 需要明确 target；我们如果 target 是"高质量通用 web"，容易把分布拉向 Wikipedia/书籍，伤 web bpb 的多样性。可作为"末段 target"工具，而不是全程筛选。
- **成本**：CPU ¥5–20；非常便宜。
- **预期**：−0.003～−0.01 bpb；置信度低。
- **坑**：target 选错会把长尾知识丢掉；ClimbMix 已均衡时收益小。

#### 3.1.6 SemDeDup（2023，经典，证据 B）

- **做法**：用预训练 embedding 找语义近邻，删"语义重复"（非精确重复）。
- **关键证据**：LAION 删 50% 数据性能几乎不降，训练时间减半；C4 上优于先前方法。
- **可借鉴性**：**不适用**。ClimbMix 上游已 curated（去重细节未核实）；8.57B 的 embedding + kNN 成本高；删 50% 对我们 8B token 预算太激进。
- **成本**：embedding+kNN ¥100–300+；训练 ¥110。
- **预期**：不确定；不建议。

#### 3.1.7 Nemotron-CC（2024/2025，高关注，证据 C）

- **做法**：classifier ensemble + 合成数据改写 + 减少 heuristic filter，保留更多 token；产 6.3T。
- **关键证据**：8B/1T 上高质量子集比 DCLM +5.6 MMLU；6.3T 全量有 4× unique real tokens；8B/15T（其中 7.2T 来自该数据）超 Llama 3.1 8B：MMLU +5、ARC-C +3.1。
- **可借鉴性**：**思路可借鉴**，不是直接复现。核心结论：**不要为了质量把 90% 数据扔掉**；在 8B token 预算下，质量与数量要平衡，尤其我们还想跑 2 epoch。
- **成本**：全量管线 ¥1,000+；子集/现成数据 ¥0–50。
- **预期**：参考价值 > 直接收益。

#### 3.1.8 2026 新进展：WebGraphMix / SPOKES / CuraWeb / DataOrchestra

这四篇代表 2026 年数据侧的四条新方向；都还很新，**引用热度未形成，按"方向信号"看，不要当定论**。

- **WebGraphMix（2026，证据 A）**：用 Common Crawl host-level web graph 的 centrality 区分"中心"与"边缘"网页，按 1:1 混合。**400M/8B：41.4% vs uniform 39.8%；再叠 quality classifier 43.8%**。这是与我们 setting 最接近的 2026 结果——如果 ClimbMix 子集保留 URL/host 元数据，这个改动几乎零成本。预期 bpb −0.005～−0.015，置信度中。
- **SPOKES（2026，证据 B）**：直接优化 diversity（G-Vendi）+ quality 联合选择；DCLM/FineWeb 上 +1.5/+1.4 points，优于 semantic dedup 和纯 quality。**启示**：纯 quality top-k 会窄化，quality+diversity 才稳。
- **CuraWeb（2026，证据 C）**：quality/redundancy/diversity 三目标联合，2T 语料，3B 规模 +1.8%。工业级管线，不适合我们全量复现，但"多目标"思路可借用。
- **DataOrchestra（2026，证据 B）**：逐 example 决定 drop/clean/rewrite，用 LLM 选操作；0.5B–7B from scratch，11 benchmarks 稳定优于单一处理。**启示**：2026 的方向是"per-example orchestration"，但需要一套工具模型，成本高；我们最多做"规则 + 小模型"的简化版。

### 3.2 合成数据

#### 3.2.1 Cosmopedia（2024，高引用，证据 B/C）

- **做法**：Mixtral-8x7B 基于 curated topics + web seeds 生成教科书/博客/故事/WikiHow；25B tokens、30M files。
- **关键证据**：cosmo-1b 在 ARC-easy/ARC-challenge/OpenBookQA/MMLU 超 TinyLlama-1.1B，部分接近 Qwen-1.5-1B；但和 Phi-1.5 仍有差距。
- **可借鉴性**：**需改造**。通用 web bpb 未必降；适合末段混 5–15% 提升知识与推理。现成数据可直接下载。
- **成本**：下载+tokenize ¥0–20；训练 ¥110。
- **预期**：bpb ±0.01；能力 +2–5pp；置信度低。
- **坑**：合成数据的"风格"与 web val 不同，bpb 可能变差；用能力 eval 判断，不要只看 bpb。

#### 3.2.2 TinyStories（2023，经典，证据 C）

- **做法**：GPT-3.5/4 生成 3–4 岁词汇的短故事。
- **关键证据**：<10M 参数模型可生成连贯多段故事；证明窄域合成数据能让小模型"会说话"。
- **可借鉴性**：**不适用**。域不匹配（故事 vs web）；但它的方法论（用合成数据控制难度/词汇）可借鉴到 curriculum。
- **成本**：现成 ¥0–10。
- **预期**：对 web bpb 基本无效。

#### 3.2.3 Phi-1 textbook（2023，经典，证据 C）

- **做法**：6B "textbook quality" web + 1B GPT-3.5 合成教科书/练习。
- **关键证据**：1.3B 模型 HumanEval 50.6%、MBPP 55.5%；350M 版 45%。
- **可借鉴性**：**不适用/需改造**。代码域专用；通用 web 不适用。但"textbook-style + 练习"对 math/code 末段混入有效。
- **成本**：自生成 ¥500+；现成 FineMath/Stack-Edu 更便宜。

#### 3.2.4 SmolLM2：FineMath / Stack-Edu / SmolTalk（2025，高关注，证据 C）

- **做法**：1.7B/11T，四阶段：web → 加 math/code → 加 instruction → decay 期 math 14%；自建 FineMath（54B）、Stack-Edu（从 450B 代码子集过滤）、SmolTalk。
- **关键证据**：超过 Qwen2.5-1.5B / Llama3.2-1B；证明小模型要"多阶段 + 高质量末段"；FineMath/Stack-Edu 公开可下载。
- **可借鉴性**：**可借鉴**。我们不需要 11T；在 8.57B 的末段 10–15% 混 FineMath/Stack-Edu（按 tokenizer 重训或对齐）即可。注意我们的 tokenizer 是 40k，FineMath 是 Llama tokenizer，需要重新 tokenize。
- **成本**：下载+tokenize ¥10–30；训练 ¥110。
- **预期**：bpb 持平/略降；GSM8K/MMLU +2–5pp；置信度中。
- **坑**：math/code 过多会伤 web bpb；控制在 10–15%。

**SmolLM3（2025，3B 级 / 约 11.2T tokens，证据 C）**：SmolLM2 数据配方的扩展，加入多语言、长上下文和推理数据（[HF blog](https://huggingface.co/blog/smollm3)）。对我们的直接增量有限（3B/11T 规模，远大于我们），但再次验证"多阶段 + 高质量末段"的路线；如果后续要做多语言/长上下文，可把它当数据配方参考。

#### 3.2.5 BeyondWeb（2025 新，证据 C）

- **做法**：系统研究合成数据生成：改写策略、generator 模型、源数据选择；产 BeyondWeb。
- **关键证据**：3B/180B 超 8B/180B on Cosmopedia；比 Cosmopedia +5.1pp、比 Nemotron-Synth +2.6pp；比 open web 快 7.7×；2.7× faster than Nemotron-Synth。
- **可借鉴性**：**需改造**。生成 8.57B 合成数据成本 ¥2,000+，远超预算；但结论"结构化格式（表格/数学题/FAQ/教程）优于普通改写"可直接用于小规模末段数据。
- **成本**：子集 0.1–0.5B ¥100–300。
- **预期**：能力 +3–5pp；bpb −0.01～−0.03（低置信度）。

#### 3.2.6 FinePhrase（2026 新，证据 B）

- **做法**：把 web 文本改写成表格、数学题、FAQ、教程等结构化格式；生成 >1T tokens 做控制实验，发布 486B 开放数据。
- **关键证据**：结构化格式一致优于 curated web 和先前合成方法；**generator >1B 参数没有额外收益**；生成成本降低 30×。
- **可借鉴性**：**可直接用/改造**。下载 486B 中的子集，末段混 5–15%；或者用 1B 级 generator 自己改写 0.1–0.2B tokens（成本可控）。
- **成本**：下载 ¥0–20；自改写 0.1B ≈¥30–60；训练 ¥110。
- **预期**：bpb −0.005～−0.02；能力 +2–5pp；置信度中低。
- **坑**：结构化格式会改变 val 分布；用能力 eval 为主。

#### 3.2.7 MGA reformulation（2025，证据 B）

- **做法**：Massive Genre-Audience 改写，把现有语料改写成多种体裁/受众，缓解重复数据的负面效应；发布 770B MGACorpus。
- **关键证据**：在 up to 13B 参数上优于 data repetition/upsampling。
- **可借鉴性**：**需改造**。如果 R1 发现 2 epoch 重复有损伤，MGA 是"把重复变成多样"的替代方案；但生成成本高。
- **成本**：0.1–0.5B 改写 ¥100–300。
- **预期**：−0.01～−0.03 bpb（低–中）。

### 3.3 数据混合与 curriculum

#### 3.3.1 Data Mixing Laws（2024/ICLR25，证据 B）

- **做法**：拟合 mixture→loss 的函数，再嵌套 training-step/model-size scaling law，用小规模训练预测大模型在未见 mix 上的表现。
- **关键证据**：1B/100B RedPajama 上优化 mix ≈ 多训 48% steps。
- **可借鉴性**：**需改造**。需要 domain 标签和多次 proxy 训练；我们 8.57B 的 domain 标签可能不完整。若只做 2–3 个 mix 的线性扫描，不如直接用 curriculum。
- **成本**：proxy 10–20 跑 × 1B ≈¥150–300；不便宜。
- **预期**：−0.005～−0.02；置信度中低。

#### 3.3.2 RegMix（2024/ICLR25，证据 B）

- **做法**：512 个 1M 参数小模型 × 1B tokens 拟合回归，选最佳 mix，再训 1B/25B。
- **关键证据**：1B/25B 在 64 个候选 mix 中最好；7B/100B 仍优于人工选择；匹配 DoReMi 只用 10% 算力。
- **可借鉴性**：**不适用**。512×1B = 512B proxy tokens，约 2,000+ GPU·h；我们单卡预算不可能。
- **成本**：¥2,000+；直接淘汰。

#### 3.3.3 Domain upsampling at the end（2024，证据 C）

- **做法**：末段 10–20% 上采样领域数据（相对 CC）。
- **关键证据**：7B/1T：MMLU +6.90pp、GSM8K +8.26pp、HumanEval +6.17pp；10–20% 最优。
- **可借鉴性**：**可直接用**。这是我们 R2 的核心证据；把高质量 web/math/code 放 decay 期。
- **成本**：数据 ¥0–20 + 训练 ¥110。
- **预期**：能力 +2–6pp；bpb −0.005～−0.02；置信度中。
- **坑**：7B/1T 的外推；我们的 8.57B 只有 1T 的 0.86%，末段比例要更保守（10–15%）。

#### 3.3.4 OLMo 2 late-stage curriculum（2025，证据 C）

- **做法**：预训练后期引入 Dolmino Mix 1124 专用 mix（annealing）。
- **关键证据**：7B/13B/32B 全开放；late-stage curriculum 显著提升多 benchmark；模型在 Pareto frontier。
- **可借鉴性**：**可直接用**。和 R2 相同；OLMo 2 证明了"后期换 mix"在大模型有效，我们的 WSD decay 期正好对应。
- **成本**：¥110–130。
- **预期**：−0.005～−0.02；置信度中。

#### 3.3.5 Curriculum learning：系统研究（2025，证据 B）

- **做法**：6 种难度指标（压缩率、MTLD、Flesch 等），200+ 模型、最多 100B tokens；vanilla curriculum / pacing / interleaved。
- **关键证据**：18–45% fewer steps 到 baseline；warmup curriculum 持续 +3.5%；compression ratio、MTLD、readability 最有效。
- **可借鉴性**：**可直接用**。用压缩率/可读性给 8.57B 排序，成本几乎为零；在 0.5B 锚点上先验证 easy→hard 是否适合我们。
- **成本**：排序 CPU ¥0–10 + 训练 ¥110。
- **预期**：−0.005～−0.02；置信度中。
- **坑**：P1 的 β 在后期变平，curriculum 的"加速收敛"不一定转化为最终 bpb；要看终点而不是中间 loss。

#### 3.3.6 LR decay wastes your best data（2025，证据 B）

- **做法**：指出 ascending-quality curriculum 与标准 LR decay 不兼容；用温和 decay + model averaging（最后几个 checkpoint 加权平均）。
- **关键证据**：1.5B/30B，+1.64% 平均（vs random），无需改数据；constant LR 下 curriculum 优势更大。
- **可借鉴性**：**直接相关**。我们已经在用 WSD + EMA；但 EMA 不是 checkpoint averaging。R2 要把 decay 调温和（min LR 0.2 而非 0.1，或 decay-start 0.85）并试 model averaging，否则 curriculum 的收益会被 decay 吃掉。
- **成本**：仅训练 ¥110–130。
- **预期**：决定 R2 成败；单独看不增加 bpb，但能让 curriculum 的 −0.01 变成现实。

#### 3.3.7 2026 curriculum 分析（证据 B）

- **做法**：14M–1B、300B tokens，三种语言学校准 curriculum（Age-of-Acquisition、词频、Verb Variation），分析 latent phases/GNS/output-head。
- **关键证据**：curriculum 主要改变"在每个 phase 停留的时间"，不创造新 phase；reverse-order 会丢掉大部分优势；更大模型上差异变小。
- **可借鉴性**：**风险提示**：我们的 288M 属于"中小"；curriculum 可能只加速早期/中期，对最终 bpb 影响有限。R2 的判据要看**终点 bpb**，不要被中间曲线迷惑。

### 3.4 数据重复

#### 3.4.1 Muennighoff et al.（2023，经典，证据 A/B）

- **做法**：系统研究 data-constrained regime 的重复训练，提出考虑重复价值的 scaling law。
- **关键证据**：最多 900B tokens/9B 模型；**≤4 epoch 重复与 unique 几乎无差**；更多重复价值趋零；缓解手段包括加 code、去掉过强过滤。
- **可借鉴性**：**可直接用**。我们 8.57B 单 epoch，2 epoch 有明确依据；但 4 epoch 是上限，不是目标。
- **成本**：2 epoch ¥230。
- **预期**：−0.02～−0.04；置信度高。

#### 3.4.2 Internal Data Repetition Destroys LMs（2026 新，证据 A）

- **做法**：用 no-repetition scaling law 量化"内部重复"的 compute-equivalent gain/loss；研究重复子集大小 × 重复次数的交互。
- **关键证据**：344M Qwen3-style，FineWeb-Edu-Dedup；**中等大小子集 × 中等重复次数最差**；重复占 10% FLOPs 时，最坏等价于只用 67% FLOPs；重复峰值随模型大小幂律移动。
- **可借鉴性**：**关键风险警示**。MiniPLM 的 "25% 数据 × 4 epoch" 正好落在"中等子集 × 中等重复"区间；必须加 **random-25% × 4 epoch** 控制，区分"选得准"与"重复本身有害"。
- **成本**：分析便宜；实验 ¥50–100。
- **预期**：可能把 R3 的预期从 −0.05 压到 −0.02；置信度中。

#### 3.4.3 Scaling Domain Data Repetition（2026 新，证据 B）

- **做法**：在固定 TPP 下研究 domain 数据的最优重复次数随模型规模的变化。
- **关键证据**：最优重复次数随模型规模**轻微上升**；与 domain 的最终 val loss 强负相关（loss 越低越可重复）；与 unique data 量弱相关；小 proxy 可估计大模型。
- **可借鉴性**：**可改造**。如果我们要重复某个 domain（math/code），先用 0.5B proxy 扫 1/2/3/4 epoch，找最优；不要全局统一 4 epoch。
- **成本**：proxy 3 臂 × 0.5B ≈¥25。
- **预期**：避免过度重复；bpb −0.005～−0.015。
- **相关证据（Finetuner's Fallacy, 2026）**：把领域数据从预训练早期就重复混入（SPT），可比标准预训练少 1.75× tokens 达到同领域性能；在远离 web 的领域，1B SPT 可超 3B 标准预训练。这支持 R2 的 domain upsampling，但它是领域特化，不等价于通用 web bpb 提升。

#### 3.4.4 Repetition Mismatch（2026 新，证据 B）

- **做法**：指出小规模 mixture 实验外推失败的一个主因是"重复率不匹配"；用 subsampling 匹配目标重复率。
- **关键证据**：1.17B 模型，只用 1/16 target tokens 的 repetition-controlled 实验，mix 误差 0.10 vs 不控制 0.85；三源场景 757M 两次 horizon 即可。
- **可借鉴性**：**直接用于我们的锚点设计**。0.5B 锚点如果只用 0.5B unique，而 8B 目标是 2 epoch，锚点的重复率不匹配；要按目标 TPP/epoch 设计锚点（例如 0.5B 锚点也用 2 epoch，或 1B 锚点 1 epoch）。
- **成本**：设计层面，零额外成本。
- **预期**：提高锚点外推可靠性。

#### 3.4.5 稀疏模型重复警告（2026 新，证据 B）

- **做法**：研究 MoE 在数据重复下的过拟合；80M–1B active、8.5B total。
- **关键证据**：80M dense 可重复 >8× 而几乎不退化；MoE 在 4× 就开始受损；稀疏度越高越严重；强 masking 可缓解但不能完全恢复。
- **可借鉴性**：**风险提示**。我们是 dense + Engram 稀疏表，不是 MoE；但 Engram 的稀疏记忆可能放大重复数据的 memorization。R1 的 2 epoch 要监控 train/val gap 和 Engram 命中率，必要时加 dropout/masking。
- **成本**：监控零成本；正则化实验 ¥20–50。

### 3.5 数据侧蒸馏

#### 3.5.1 MiniPLM（2024/ICLR25，证据 A）

- **做法**：离线 Difference Sampling：reward = log p_teacher(x) − log p_ref(x)，从 corpus 中选 teacher 偏好但 reference 认为难的数据；学生从零训练。proxy 变体：只在 0.1B tokens 上算 teacher/ref 分数，再训一个线性头在 reference hidden states 上外推到全 corpus。
- **关键证据**：
  - 200M/500M/1.2B student，1.8B Qwen teacher；9 个下游任务提升；**数据需求 ↓2.4×**（等效多 68B tokens）；
  - **25% 数据 × 4 epoch 最佳**（α=0.25），和 Muennighoff 的 4-epoch 结论一致；
  - proxy 变体：0.1B/50B = 0.2% 的 teacher 计算，平均 40.9 vs full 41.3 vs Vanilla KD 39.9；
  - 跨模型家族有效（Llama3.1、Mamba）。
- **可借鉴性**：**可直接用/改造**。这是我们规模最匹配的蒸馏工作（200M student）。我们可以用 Qwen3-1.7B/SmolLM2-1.7B 作 teacher，50–100M reference，0.1B proxy；选 top 25%，跑 4 epoch。关键控制：random-25%×4，排除"重复本身"的影响。
- **成本**：teacher proxy 0.1B ¥7–11；reference 训练 0.5B ¥7；全量打分（50–100M ref）¥40–100；训练 8.57B ¥115；合计 ¥170–230。
- **预期**：−0.02～−0.05 bpb；置信度低–中（2026 重复警告是主要风险）。
- **坑**：
  - **tokenizer 不匹配（工程难点）**：MiniPLM 是 corpus-level，跨模型家族可行；但我们的 40k tokenizer 与 Qwen（151k）/Llama（128k）不同。需要"同一文本、两套 tokenization"分别算 teacher/student logp，或用一个与 40k 兼容的 teacher。**R3 前先用 10M tokens pilot 验证 reward 与难度判断的相关性**；如果对齐做不好，R3 直接放弃。
  - 2026 内部重复警告。
  - reference 太弱/太强都会影响 reward 质量。

> **重要修正**：MiniPLM 的跨模型家族实验用的是 Qwen teacher + Llama/Mamba student，tokenizer 不同，但它是 **corpus-level**：teacher 在自己的 tokenization 上算 logp，student 在自己的 tokenization 上训练；严格说样本级 reward 需要文本对齐，论文通过"同一文本、不同 tokenization"处理。工程上可行，但需要小心。我们可以在 R3 前先做一个 10M token 的 pilot 验证 reward 与人工/难度判断的相关性。

#### 3.5.2 SeqKD（2016，经典，证据 C）

- **做法**：teacher beam search 生成序列，student 在这些序列上做 MLE。
- **关键证据**：经典 sequence-level KD；在 MT/摘要有效；pretraining-from-scratch 证据弱。
- **可借鉴性**：**需改造**。只适合有明确 teacher 的域；全量生成 8.57B 成本 ¥2,000+。可做 0.1B 子集的 domain-specific 蒸馏。
- **成本**：0.1B 生成 ¥10–30；训练 ¥110。
- **预期**：能力 +1–3pp；bpb 不确定；置信度低。

#### 3.5.3 工业界证据：Gemma 2 / Qwen3（2024/2025，证据 C）

- Gemma 2 的 2B/9B 直接用 27B teacher 蒸馏训练（不是 from-scratch next-token）；Qwen3 的 0.6B–8B 用 strong-to-weak distillation。它们说明"小模型 + 强 teacher"是工业主路径，但都是 **post-training/继续训练**，不是我们这种 from-scratch 8B token 预算。可借鉴的是"离线、语料级"而不是"在线 logit"。

---

## 4. 我们 setting 的 3 项推荐实验

### 4.0 共同设置与判据

- **模型/优化**：d24_c1 288M active + Engram 220M；seq 8192；LR 8e-4；WSD warmup 30 / decay-start 0.9 / min 0.1；EMA 0.995；其余同 P1。
- **锚点规模**：0.5B tokens/arm（2.2h，¥7）；finalist 加 2 seeds；1B 确认（4.4h，¥14）。
- **评测**：held-out 3M panel，raw + EMA bpb；同时记录 train/val gap（防重复过拟合）。
- **判据**：0.5B 锚点的 seed noise 约 ±0.003–0.006 bpb；要求 **Δbpb ≥0.008–0.01** 才进入下一轮；终点 bpb 比中间 loss 重要。
- **锚点外推**：P1 教训是短跑 β 偏大；所以锚点只做 go/no-go，最终要 1B 或 8B 确认。若目标含 2 epoch，锚点也要匹配重复率（见 2606.07597）。

### 4.1 R1（先做）：重复标定 → 2 epoch

**假设**：在 288M/8.57B 上，第二遍数据仍值 60–80% 的 unique 数据；2 epoch 的收益大于任何同成本的筛选/curriculum。

**0.5B 锚点设计**（4 臂）：

| 臂 | 数据 | tokens | 目的 |
|---|---|---|---|
| A | 0.5B unique × 1 epoch | 0.5B | 1 epoch 基线 |
| B | 0.5B unique × 2 epoch | 1.0B | 重复 2 遍 |
| C | 1.0B unique × 1 epoch | 1.0B | 同算力 unique 对照 |
| D（可选） | 0.5B unique × 4 epoch | 2.0B | 重复上限 |

**成本**：A+B+C = 2.5B tokens ≈ 11h ≈ ¥35；加 D 2B ≈ 9h ≈ ¥28；B/C 各加 1 seed ≈ ¥35；合计 **¥70–100**。

**判据**：
- **B − C ≤ 0.008 bpb** → 重复惩罚可接受；
- **B − A ≥ 0.015 bpb** → 第二遍值得；
- 若 B − C > 0.015 → 重复有害，放弃 2 epoch，转 R2/R3；
- 若 B − A < 0.01 → 2 epoch 不值 ¥230，停止。

**8B 确认**：17.1B tokens，76h，**¥230**。预期 raw 1.066 → **1.030–1.045**；EMA 1.026–1.040；置信度 **高（~70%）**。

**监控**：12B/14B/17B 的 val bpb；若 14B 后 val 转升，改用 1.5 epoch（12.8B，¥165）。Engram 命中率/训练记忆率异常时加 dropout。

### 4.2 R2（最便宜）：末段 curriculum / domain upsampling

**假设**：把高质量 10–15% 放在 WSD decay 期，配合温和 decay/EMA，能在不增加 token 的情况下拿到 −0.005～−0.02 bpb；如果 order 无效，top-50% 过滤本身也可能有效。

**打分**：优先用现成 FineWeb-Edu 分数/URL 对齐；否则用 fastText/压缩率/可读性（CPU，¥5–20）。不要用 7B judge 全量。

**0.5B 锚点设计**（4 臂）：

| 臂 | 数据顺序 | 目的 |
|---|---|---|
| A | random（control） | 基线 |
| B | 前 85% 低质 → 后 15% 高质（ascending） | curriculum 主臂 |
| C | 前 15% 高质 → 后 85% 低质（descending） | 顺序对照 |
| D | 1B 池的 top 50%（0.5B unique），1 epoch | 纯过滤对照 |

> 注：D 臂不要用 0.5B 池的 top 50%（那只有 0.25B unique，跑 0.5B tokens 会变成 2 epoch，和重复混淆）；从 1B unique 池取 top 50%，保持 0.5B unique × 1 epoch，才和 A 同算力。

**成本**：4×0.5B = 2B ≈ 9h ≈ ¥28；打分 ¥10–30；合计 **¥40–60**。

**判据**：
- **B − A ≥ 0.008** 且 **B − C ≥ 0.005** → 顺序有效；
- **D − A ≥ 0.005** → 过滤本身有效；
- 若 B − A < 0.005 → 停止（WSD decay 可能吃掉收益）；
- 若 D 最好而 B/C 无效 → 改做 top-k 过滤，不做 curriculum。

**8B 确认**：8.57B，末段 15% 高质（**从同一 8.57B 中按质量分选，只改顺序**；若混外部 FineMath/Stack-Edu，另做一臂，不要和顺序混淆），decay-start 0.85 / min LR 0.2（或 model averaging），**¥115–145**。预期 **−0.005～−0.02 bpb**；置信度 **中（~50%）**。

**混淆控制（重要）**：如果 8B 确认改了 decay/min LR，必须加一个**同 schedule 的 random 对照**（+¥115），否则不能把差异归因于数据顺序；预算不够时，先在 0.5B 做 **2×2**（4 个数据臂 × 2 种 decay：P1 的 0.9/0.1 vs 温和的 0.85/0.2），8 臂 × 0.5B = 4B ≈ ¥56；这能直接检验 2511.18903 的 curriculum×decay 交互，再决定 8B 是否改 schedule。

**注意**：R2 与 R1 可以组合：第二 epoch 的末段 15% 用高质数据；但先单独验证 R2，避免混淆。

### 4.3 R3（上限最高）：MiniPLM-lite top-k 选择

**假设**：teacher–reference 差异采样选出的 25% 数据，跑 4 epoch，能在同算力下超过 random 25%×4 和 1 epoch baseline。

**实现**（pilot 先做 10M tokens 验证 reward 相关性）：
1. teacher：Qwen3-1.7B 或 SmolLM2-1.7B（文本级 logp；tokenizer 不同需做文本对齐）；
2. reference：50–100M 模型，0.5B tokens 训练（¥7）；
3. proxy：0.1B tokens，算 r = log p_teacher − log p_ref；在 reference hidden states 上训线性头，外推到 8.57B；
4. 选 top 25%（2.14B unique），跑 4 epoch（8.56B tokens）。

**0.5B 锚点设计**（4 臂）：

| 臂 | 数据 | tokens | 目的 |
|---|---|---|---|
| A | random 0.5B × 1 epoch | 0.5B | baseline |
| B | top 25% × 4 epoch | 0.5B | 选择+重复 |
| C | random 25% × 4 epoch | 0.5B | 重复控制（关键） |
| D | top 50% × 2 epoch | 0.5B | 温和版 |

**成本**：训练 4×0.5B = 2B ≈ ¥28；teacher proxy 0.1B ¥7–11；reference 0.5B ¥7；全量打分（50–100M ref）¥40–100；合计 **¥85–145**。

**判据**：
- **B − C ≥ 0.01** → 选择信号有效；
- **B − A ≥ 0.015** → 总体值得；
- 若 B − C < 0.005 → 选择信号弱，试 D；
- 若 C − A < −0.01 → 重复有害，放弃 4 epoch，改 D 或 top 70%×1.5。

**8B 确认**：8.57B total；top 25%×4 或 top 50%×2；**¥170–230**。预期 **−0.02～−0.05 bpb**；置信度 **低–中（~35–45%）**。

**最大风险**：2026 "Internal Data Repetition" 的"中等子集×中等重复最差"；C 臂是生死判据。若 C 明显差于 A，R3 不成立。

### 4.4 组合路线图与预算

| 阶段 | 内容 | 成本 | 产出 |
|---|---|---:|---|
| 0 | 审计 ClimbMix 子集（重复率/来源/质量分布/污染） | ¥0–10 | 决定 R2/R3 是否值得 |
| 1 | R1 锚点（4 臂 + 2 seeds） | ¥70–100 | 2 epoch 是否可行 |
| 2 | R2 锚点（4 臂） | ¥40–60 | curriculum/过滤是否有效 |
| 3 | R3 锚点（4 臂 + pilot） | ¥85–145 | 选择信号是否有效 |
| 4 | 最优方案 1B 确认 | ¥30 | 锚点外推校准 |
| 5 | 最优方案 8B 确认 | ¥110–230 | 最终结论 |

- **严格 ¥300 预算**：R1 三臂锚点 ¥35 + R1 8B ¥230 = ¥265；R2 锚点（¥40）视余额，若做约 ¥<redacted>。
- **¥400–500 预算**：R1/R2/R3 锚点 + 最优 8B ≈ ¥400–500。

### 4.5 如果只能做一件事

**做 R1**：把 8.57B 跑 2 epoch（17.1B tokens，¥230，76h）。理由：
- 我们的曲线在 8B 仍下降，β≈0.07；第二遍数据是唯一有 A 级证据（Muennighoff ≤4 epoch）的杠杆；
- 它不需要任何新数据/打分，工程风险最低；
- 它给出 R2/R3 的对照基线；没有这个基线，筛选/curriculum 的 −0.01 可能只是噪声。

如果目标不是 bpb 而是"能力"（MMLU/GSM8K/生成质量），则优先 R2 + 10–15% FineMath/Stack-Edu 末段混入（¥115–145），bpb 可能持平，但能力收益更明显。

---

## 5. 风险与"别做"清单

### 5.1 在这个规模大概率无效的方法

| 方法 | 为什么无效/不划算 |
|---|---|
| **DoReMi** | proxy 280M ≈ 目标 288M；省不了算力，还要多跑一次 8B。 |
| **RegMix** | 512×1B = 512B proxy tokens ≈ 2,000+ GPU·h；预算差一个数量级。 |
| **SemDeDup** | ClimbMix 上游已 curated；8.57B 的 embedding+kNN 贵；删 50% 对我们 8B token 预算太激进。 |
| **全量 DCLM/FineWeb-Edu 再过滤** | 上游已 curated；收益是 secondary filtering；分类器全量打分 ¥190–380，超过一次训练；过度过滤伤多样性（Nemotron-CC 证据）。 |
| **在线 KD / logit 蒸馏** | 固定算力下不如直接多训（MiniPLM Fig.2）；50B tokens 的 logits 需 30PB 存储；teacher 在线推理吃算力。 |
| **7B judge 全量打分** | 8.57B tokens × 7B teacher ≈ 600–950h / ¥1,900–3,000；超过预算 10×。 |
| **7B 生成 8.57B 合成数据** | 同上；且合成数据在 288M 上未必优于 curated web。 |
| **>4 epoch / 中等子集×中等重复** | Muennighoff：>4 epoch 价值趋零；2026 Internal Repetition：中等子集×中等重复最差；Engram 可能放大记忆。 |
| **多阶段复杂 curriculum / mixture 搜索** | SmolLM2 是 11T token 规模；我们 8.57B 只有 0.08%，stage 边界和 mix 的噪声会吃掉收益。 |
| **纯 quality top-k 过滤到很小** | 分布窄化，长尾/知识任务受损；SPOKES/CuraWeb 都强调 quality+diversity。 |
| **指望数据 trick 到 0.89** | 固定 8B 预算约 −0.02～−0.04 bpb；含 2 epoch 到 −0.05～−0.07；0.89 需要 20B+ tokens 或 600M+ active。 |

### 5.2 风险

1. **分布窄化**：quality-only 过滤会伤多样性；用 quality+diversity（SPOKES/CuraWeb 思路）或保留 20–30% 的"低分但多样"数据。
2. **重复过拟合**：2 epoch 要监控 train/val gap；Engram 稀疏表可能 memorization；必要时 dropout/masking。
3. **Curriculum × LR decay 冲突**：标准 WSD decay 会吃掉 ascending curriculum 的收益（2511.18903）；用温和 decay/min LR 0.2 + model averaging/EMA。
4. **锚点外推失败**：0.5B 的 β 比 8B 大；必须 1B 或 8B 确认；用 repetition-controlled 锚点（2606.07597）。
5. **Teacher tokenizer 不匹配**：MiniPLM 跨 tokenizer 需要文本级对齐；先做 10M token pilot 验证 reward 相关性。
6. **数据污染**：合成数据/改写可能带入 benchmark 污染；13-gram 去污染。
7. **成本失控**：数据处理先用 0.1B proxy + 线性头；不要一上来全量 7B judge。
8. **License**：ClimbMix 是 NVIDIA 条款；FineMath/Stack-Edu 是 ODC-BY；Cosmopedia 需确认；商用前逐个核对。

### 5.3 第 0 步审计清单（¥0–10，1–2h CPU）

- **重复率**：MinHash 13-gram，抽 10M docs；目标 near-dup <5%。
- **来源/domain 分布**：URL host/路径、语言、长度；和 ClimbMix 预期配比对比。
- **质量分布**：FineWeb-Edu classifier 抽 1M docs，看分数分布的长尾。
- **污染**：13-gram 对 MMLU/ARC/GSM8K/HellaSwag 等 eval 集；命中即删。
- **长度/截断**：seq 8192 下，多少 docs 被截断；长文档比例。
- **tokenizer fertility**：各 domain tokens/byte；确保 math/code 不被过度切碎。
- **Engram 命中率**：重复数据是否让稀疏表记忆化；监控 train/val gap。

---

## 6. 参考文献

**数据筛选/质量分类**

- DCLM: [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) (2024; NeurIPS 2024)
- FineWeb / FineWeb-Edu: [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) (2024)
- QuRating: [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) (2024; ICML 2024)
- DoReMi: [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) (2023; NeurIPS 2023)
- DSIR: [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) (2023; NeurIPS 2023)
- SemDeDup: [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540) (2023)
- Nemotron-CC: [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) (2024/2025; ACL 2025)
- WebGraphMix: [Hubs or Fringes: Pretraining Data Selection via Web Graph Centrality](https://arxiv.org/abs/2606.11499) (2026 新)
- SPOKES: [Spokes: Optimizing for Diverse Pretraining Data Selection](https://arxiv.org/abs/2606.15216) (2026 新)
- CuraWeb: [CuraWeb: Joint Optimization of Quality, Redundancy, and Diversity for Web-Scale Pretraining Data](https://arxiv.org/abs/2607.22662) (2026 新)
- DataOrchestra: [DataOrchestra: Learning to Orchestrate Per-Example Curation of Pretraining Data](https://arxiv.org/abs/2607.24717) (2026 新)
- Data mixing survey: [Data Mixing for Large Language Models Pretraining: A Survey and Outlook](https://arxiv.org/abs/2604.16380) (2026 新)

**合成数据**

- Cosmopedia: [Cosmopedia: how to create large-scale synthetic data for pre-training](https://huggingface.co/blog/cosmopedia) (2024; HF blog) / [dataset](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- TinyStories: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (2023)
- Phi-1: [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) (2023)
- SmolLM2: [SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737) (2025)
- BeyondWeb: [BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining](https://arxiv.org/abs/2508.10975) (2025)
- FinePhrase: [How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data](https://arxiv.org/abs/2604.13977) (2026 新; COLM 2026)
- MGA: [Reformulation for Pretraining Data Augmentation](https://arxiv.org/abs/2502.04235) (2025)

**数据混合 / curriculum**

- Data Mixing Laws: [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) (2024; ICLR 2025)
- RegMix: [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492) (2024; ICLR 2025)
- Domain upsampling: [Does your data spark joy? Performance gains from domain upsampling at the end of training](https://arxiv.org/abs/2406.03476) (2024)
- OLMo 2: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (2024/2025; COLM 2025)
- Curriculum learning: [Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://arxiv.org/abs/2506.11300) (2025)
- LR decay vs curriculum: [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](https://arxiv.org/abs/2511.18903) (2025)
- 2026 curriculum analysis: [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics](https://arxiv.org/abs/2601.21698) (2026 新)

**数据重复**

- Muennighoff: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (2023; NeurIPS 2023)
- Internal repetition: [Internal Data Repetition Destroys Language Models](https://arxiv.org/abs/2606.24998) (2026 新)
- Domain repetition: [Scaling Domain Data Repetition in LLM Pretraining](https://arxiv.org/abs/2608.14071) (2026 新)
- Repetition mismatch: [Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them](https://arxiv.org/abs/2606.07597) (2026 新; EMNLP 2026)
- Sparse/MoE repetition: [Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data](https://arxiv.org/abs/2609.11917) (2026 新)
- Specialized pretraining: [The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data](https://arxiv.org/abs/2603.16177) (2026 新)

**数据侧蒸馏**

- MiniPLM: [MiniPLM: Knowledge Distillation for Pre-Training Language Models](https://arxiv.org/abs/2410.17215) (2024; ICLR 2025) / [code](https://github.com/thu-coai/MiniPLM)
- SeqKD: [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) (2016)
- Gemma 2: [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) (2024)
- Qwen3: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (2025)

**数据/工具**

- ClimbMix: [nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
- FineMath: [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- Stack-Edu: [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
- Cosmopedia code: [huggingface/cosmopedia](https://github.com/huggingface/cosmopedia)
- MiniPLM code: [thu-coai/MiniPLM](https://github.com/thu-coai/MiniPLM)

---

### 附：一句话回答三个问题

**(a) 最值得先做的 3 个数据侧改动？**
1. 2 epoch 重复（先做重复标定，再 8B 确认）——置信度最高；
2. 末段 10–15% 高质量 curriculum/domain upsampling + 温和 WSD decay——最便宜；
3. MiniPLM-lite 的 teacher–reference 差异采样（top 25% × 4 epoch）——上限最高，但必须加 random-25% 控制。

**(b) 预算内实验设计？**
0.5B 锚点、4 臂、每臂 ¥7；R1 ¥70–100、R2 ¥40–60、R3 ¥85–145；判据用 Δbpb ≥0.008–0.015 和 order/重复控制；最优方案 1B 确认 ¥30，8B 确认 ¥110–230。

**(c) 哪些方法大概率无效？**
DoReMi、RegMix、SemDeDup、全量 DCLM/FineWeb-Edu 再过滤、在线 KD、7B judge 全量打分、7B 生成 8.57B 合成数据、>4 epoch/中等子集×中等重复、复杂多阶段 mixture 搜索；以及"指望数据 trick 到 0.89"。
