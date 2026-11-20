# ZGCM-1 调研：对我们有价值的点（2026-11-18）

- 论文：[ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search](https://arxiv.org/abs/2609.13356)（中关村学院/中关村人工智能研究院，2026-09）
- 代码：<https://github.com/zgcagi/ZGCM-1>（MIT）；权重：<https://huggingface.co/zgcagi/ZGCM-1-7B>；数据：<https://huggingface.co/datasets/zgcagi/ZGCM-1-Data>

## 1. 模型规格（7.39B dense，从零训练）

| 项 | 值 |
|---|---|
| 规模 | 7.39B dense，32 层，hidden 4096（C/L=128），SwiGLU h=11,008 |
| 注意力 | **27 层 gated SWA（128-token 窗口）+ 5 层 global，5:1**；GQA 32Q/8KV(128 dim)；QK-RMSNorm；Partial RoPE（rotary fraction 0.33）；局部输出门控 |
| 上下文 | 训练 16K，mid-training 逐步 16K→64K→256K；最大 256K |
| 预训练 | **4.19T tokens**：阶段 1 0.99T curriculum（顺序消费）+ 阶段 2 3.20T 全混合（全局 shuffle，从阶段 1 ckpt 继续） |
| 系统 | **FP8 delayed scaling + Muon + TWEO**（outlier 正则）；称 16K 预训练 time-to-loss **~4.2×**（vs AdamW/BF16 基线） |
| mid-training | 600B tokens，长度桶 B16/B64/B256 哈希互斥混合；交互轨迹重写成 MDP 状态-动作监督 |
| post-training | 执行验证轨迹的混合 think/no-think SFT；数学/code 混合 RL |
| 数据 | curated web + FinePDFs/olmOCR/arXiv LaTeX + Nemotron code + SmolLM2 系 math recipe + Nemotron 专门 reasoning；**GLM-5.1 tokenizer** |
| 架构实验 | SWA 5:1 在 10B tokens、8×H100 上与 full attn 同 tail loss（1.93）但 9566 tok/s/GPU；MLA 7645 tok/s 无 loss 改善；256K 时 KV cache 5GiB vs 32GiB（6.4×↓），吞吐 3.94× |
| 结果 | AIME 2026 75.0、MATH-500 97.1、HMMT 2025 70.4；agentic search 接近前沿大模型；7B-8B 档 14 项推理 benchmark 平均第一 |

## 2. 对我们（288M/8B tokens/单卡 5090/¥100–300）值得借鉴的

| 优先级 | 借鉴点 | 为什么值 | 成本 | 预期 |
|---|---|---|---|---|
| **P0** | **FP8 训练（delayed scaling）+ TWEO 式 outlier 正则** | 我们最大的约束是"¥→tokens"，这是唯一能改*吞吐*的杠杆；他们证明 FP8+Muon+TWEO 可把 time-to-loss 压 ~4.2×（vs AdamW/BF16）；我们已是 Muon/bf16，剩余增益估计 **1.3–1.6× 吞吐** | 3–10 人日（我们 repo 已有 body-fp8/linear-fp8 内核基础）+ 0.5B 锚点 ~¥15 | 等效 token +30–60% → 在 ¥100 档预期 bpb 再降 ~0.01–0.02；**bpb<1 概率从 ~20–35% 提到 ~40–55%**（与其他杠杆叠加时） |
| **P1** | **hybrid gated SWA(128)+global 5:1** | 简单、KV 省 6.4×、8K 下也有 1.1–1.3× 吞吐；我们现有 window+compressed KV+indexer 更重；可作为 P2 注意力替代 | 5–8 人日 + 1B 锚点 ~¥20 | 吞吐 1.1–1.3×（loss 持平）→ 同预算多 10–30% tokens |
| **P1** | **两阶段 curriculum（sequential → 全局 shuffle）** | 我们已有 8.57B 单域语料，可做"质量/难度升序顺序消费 + 最终阶段 shuffle"；与数据侧 R2 建议一致 | 1–2 人日 + 0.5B 锚点 ¥20–40 | −0.005~−0.02 bpb |
| **P2** | **数据治理管线（data-process，MIT）** | 现成 JSONL→quality/dedup/length-bucket/manifest 框架，可复用做数据筛选与审计 | 1–2 人日 | 支撑 E1/R2 实验 |
| **P2** | **开权重作为 teacher/judge** | 7.39B 权重+W&B 日志全开；生成 0.1–0.2B 数学/推理合成数据约 ¥15–25；全语料打分太贵（¥340） | ¥15–25 | 能力提升为主，bpb 中性 |
| **P3** | 评测框架（Atomic Capability Evaluation） | 与我们"建 lm-eval 再做能力评测"的计划一致 | — | 参考方法论 |

**明确不适用**：4.19T tokens/8×H100+ 集群规模、256K 长上下文 mid-training（我们 8K）、agentic RL/SFT（超出 bpb 目标）、GLM-5.1 tokenizer（换 tokenizer=重建语料，成本不可接受）。
**形状参考**：7.39B 用 32L×4096（C/L=128），与"越大越宽"趋势一致；我们的 288M 按 V4 形状律外推 C/L≈52（24×1024 已接近），ZGCM 提供不了小尺度数据点。

## 3. 对 ¥100 计划的影响（一句话）

FP8+TWEO 是唯一能同时提高"每 ¥ token 数"和"每 token 信息量"（TWEO 稳定外点）的改动；若落地 1.4×，¥100 从 7.3B 变成 ~10B tokens，叠加 MuonH/调度/课程/EMA 后，**从零训练 bpb 预期 1.00–1.03，bpb<1 进入"掷硬币"区间**，是 ¥100 预算下最值得做的单项工程投入。
