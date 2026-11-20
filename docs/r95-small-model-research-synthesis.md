# 小模型训练方法调研·综合结论（2026-11-18）

> 来源：4 份并行调研（`docs/r94-research/{arch,data-side,recipe,kd-capability}.md`，合计 ~230KB/1600+ 行、
> 约 200 条 arXiv 引用）+ ZGCM-1 专项（`r94-research/zgcm-1.md`）+ MiniPLM + 我们自己的 P1/ZGCM/V4 形状实测。
> 每条结论都标了置信度；跨报告冲突单列（§3）。

## 0. 一句话

**在我们这个点（288M active / 8B tokens / 单卡 5090 / ¥100–300 每轮），没有任何单一技巧能把 bpb 从
1.06 拉到 0.9；能拿到的是"组合拳"带来的 0.03–0.07 bpb（等效 1.3–2× token），并且必须先把钱花在
"提高每 ¥ token 数"上（FP8/系统）而不是"提高每 token 信息量"上。**

## 1. 跨报告核心结论（按重要性）

| # | 结论 | 证据来源 | 置信度 |
|---|---|---|---|
| 1 | **现代小模型 token 预算是 100–1000 tokens/参数（SmolLM2 11T/1.7B、TinyLlama 1T/1.1B、Phi-3 3.3T/3.8B）；我们 8B/288M≈28，差 1.5–3 个数量级** | data-side/recipe | 高 |
| 2 | **后期 β 变平是"幂律缺地板项"的假象**：`L=E+A·D^-β` 拟合 E≈0.887、β≈0.372（残差≤0.003），**0.89 从一开始就不可达**；bpb→1.0 需 **17–28B tokens**，→0.95 需 35–136B | recipe | 中高（单 run 拟合） |
| 3 | 数据侧：**ClimbMix 上游已 curated**，DCLM/FineWeb-Edu 的 raw→filtered 收益不能复现；二次过滤仅 **−0.005~−0.015 bpb**，成本可能≈一次 8B 训练 | data-side | 高 |
| 4 | 数据侧最稳杠杆：**2 epoch（17.1B tokens，¥230）→ −0.02~−0.04 bpb**；≤4 epoch 重复安全（Muennighoff）；末段 curriculum −0.005~−0.02；MiniPLM-lite 上限最高（−0.02~−0.05）但需 random-25% 控制 | data-side | 高 / 中 / 低 |
| 5 | 架构侧天花板 **0.005–0.015 bpb**；**mHC 在 depth 24 的收益被高估（0.004–0.008），不要加 hc_mult>4**；最有希望的是 **Block AttnRes（296M 锚点 −0.025 loss ≈0.008 bpb）**；deep-thin 先验负（Depth Delusion）与 MobileLLM（1T tokens）冲突，需实测；**32×768 非 iso-param，应测 32×896** | arch | 中 |
| 6 | 配方侧最大杠杆是 **Hyperball/MuonH**（3 行、无额外状态，≤1.2B 上 **20–30% token 等效**，Puro-2B 在 5090 上用）；其次是 **decay 0.4–0.8 线性 + min-lr 绝对值 ~1e-5**（我们的 10% decay + 0.1×peak 都偏保守） | recipe | 中 |
| 7 | **等算力下预训练 logit KD 不赚**（MiniPLM：200M 学生 39.9 vs 39.9/39.7/39.0）；Gemma2 的 +7.4 是 token-matched + 额外教师算力；**我们 8B 全程 8B 教师前向 ≈¥340，超预算** | kd | 高 |
| 8 | **先建评测再花钱**：lm-eval-harness 一轮 **¥1–2、17 分钟**；288M/8B 的合理预期：MMLU 25–27、ARC-E 45–52、HellaSwag 30–35、GSM8K 0–1%、CORE 0.10–0.15；**Phi-4-Mini 加 LIMO/S1K 反而掉分**→排除"1k 条 R1 轨迹 SFT"路线 | kd | 中高 |
| 9 | ZGCM-1 的系统收益是真的：**FP8 delayed scaling + Muon + TWEO ≈4.2× time-to-loss（vs AdamW/BF16）**；但 **SWA 5:1 的 3.94× 是 256K**（16K 估 1.1–1.2×）；对 8K 训练的启发只有 FP8+TWEO，不换注意力 | ZGCM | 中 |
| 10 | 我们自己的实测交叉验证：LR 探针 8e-4 最优（≥1.2e-3 更差）；EMA −0.00427 bpb；**CED 变体 A/B 实测 +7% wall、bpb +0.0054 → REJECT**；全量 val 与 3M 面板差 +0.004（噪声内，3M 的 std=0.0045） | P1 实测 | 高 |

## 2. 对 ¥100 从零训练的"组合拳"账本（与逐项预期）

基准：7.3B tokens（¥100 @¥2.78–3.08/h）用现配方 ≈ bpb **1.073**（带地板模型外推）。

| 杠杆 | Δbpb | 成本 | 置信度 |
|---|---:|---|---|
| MuonH/Hyperball 范数约束优化器 | −0.012~−0.017 | ¥0（代码） | 中高 |
| **FP8 body + TWEO**（来自 ZGCM） | −0.010~−0.020（吞吐 1.3–1.6×，等效更多 token） | 3–10 人日 + ¥15 锚点 | 中 |
| decay 0.4–0.8 线性 + min-lr 1e-5 | −0.003~−0.008 | ¥0 | 中 |
| 末段 curriculum + 高质量数据混合 | −0.004~−0.012 | ¥0–5 | 低–中 |
| indexer aux loss（CSA2 官方缺口） | −0.003~−0.01（选位更准） | 2–4 人日 + ¥10 锚点 | 低–中 |
| Block AttnRes | −0.005~−0.012 | 3–5 人日 + ¥30–60 | 中 |
| EMA 0.995 | −0.0043（已实测） | ¥0 | 高 |
| 合计（打折 30–40% 相关性） | **−0.045~−0.07** | | |

→ 组合后 **1.00–1.03**（中心 ~1.01）；**bpb<1 概率 ~40–55%**（若 FP8 落地到 1.5× 且组合全兑现，上界 0.97–0.99）；**loss<3（bpb<0.971）概率 <10–15%**。

## 3. 冲突与未决（必须实测裁决）

| 冲突 | 双方证据 | 裁决实验 |
|---|---|---|
| 形状：deep-thin vs wide | MobileLLM（1T tokens，+0.9–1.1%）vs Depth Delusion（30 架构拟合，深窄更差）+ V4 形状律（C/L≈52@288M，我们的 42.7 接近） | 2×2 锚点（24×1024 / 32×896 × mHC on/off，2B tokens/臂，¥110–125） |
| 数据过滤：DCLM 收益 vs ClimbMix 已 curated | DCLM（7B/2.6T，+6.6pp MMLU）vs data-side 分析（二次过滤仅 −0.005~−0.015） | 0.5B 锚点：raw vs filtered（¥40–60） |
| KD：MiniPLM 正收益 vs 等算力不赚 | MiniPLM（2.4× 数据效率 vs 等 FLOPs 的 vanilla KD 不赚） | 0.5B 锚点：MiniPLM 25% vs random-25% 控制（¥85–145） |
| 地板 E 的不确定性 | E=0.887（5 点拟合）vs 若 E 更低则 0.89 仍可能 | 1–4B 带 E 联合拟合（¥0，已有数据）→ 用 17B 锚点校准 |
| CED/CSA2 省不省 | 官方架构保留 vs 我们 A/B（8K 下 +1.7% FLOPs、+7% wall） | 32K 上下文下重测（¥20–30），8K 不做 |

## 4. 推荐执行顺序（详见 `docs/r95-p2-plan.md`）

1. **¥0 代码项**：FP8 body+TWEO 补丁、MuonH、decay/min-lr 配置、indexer aux loss、全量评测（已建）；
2. **¥35–60 锚点**：0.5B A/B 矩阵（FP8+TWEO / MuonH / decay / curriculum）→ 判据 Δbpb≥0.008；
3. **¥110–130**：从零 8B 全叠加 → 预期 1.00–1.03；不够再补 2.5B tokens（+¥35）；
4. **¥230–260**：2 epoch 17–19B 全叠加 → 0.97–1.00（bpb<1 把握大）。
5. 若目标升级到 0.9 以下：需 600M+ active / 更大 tokens / KD，预算 ¥300–600+，先锚点校准。

## 5. 参考文件

- `docs/r94-research/arch.md` · `data-side.md` · `recipe.md` · `kd-capability.md` · `zgcm-1.md`
- `docs/r94-small-model-methods-survey.md`（首轮概览）· `docs/r94-shape-scaling.md`（V4 形状律）
- `docs/r93-p1-final-report.md` / `r93-p1-eval-complete`（P1 实测）· `r95-p2-plan.md`（未来计划）
