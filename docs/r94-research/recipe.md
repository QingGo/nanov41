# R94 — 训练配方 / 调度 / 优化器 文献调研（2025–2026）

**日期**：2026-11-18 ｜ **范围**：LR 调度、超参迁移、优化器、稳定性与精度、数据顺序、长跑后期变平
**服务对象**：单卡 RTX 5090 从零训练 288M active（dim 1024 / 24L / V=41,280 / seq 8192 / Engram 220M），8B tokens，held-out bpb 1.066（EMA 1.062）
**写作约定**：论文数字一律标来源；"估"= 本报告基于换算率的推断；"未证实"= 未核对正文/图表或证据链不足。**本报告不改动任何代码或其他文档。**

---

## 0. 三条最重要结论

**结论 1（诊断）：后期 β 变平的主因是"幂律模型里少了常数项 E"，不是配方变坏。**
把 5 个 mark 用 `L = E + A·(D/1B)^(−β)` 拟合，得 E≈0.887、A≈0.390、β≈0.372，5 点残差 ≤0.003 bpb——β 在"超额损失"空间是**恒定**的。所谓 β 从 0.102 掉到 0.060/0.078，是把一条趋近地板的曲线强行拟合成**无地板**幂律的假象。
反证更硬：用 1–4B 四点、**仍固定 β=0.154**、只把 E 放进去外推，8B 预测 **1.048–1.058**（实测 1.066，误差 0.008–0.018）；而不带 E 的外推给 **0.89**（误差 0.176）。→ **0.89 从一开始就不可达**；要把 bpb 压到 1.00 需要 **≈17–28B tokens**（现在的 2–3.5 倍），压到 0.95 需要 **≈35–136B**（取决于 E 落在拟合区间的哪端）。

**结论 2（最大杠杆）：配方侧最值得改的是"范数约束优化器"（Hyperball / MuonH），不是调度。**
2026 年的两项独立工作把 Muon 系收益缩水的原因定位到"权重范数只由 weight decay 软控制"：Hyperball 把权重矩阵与其更新都约束到固定 Frobenius 范数，在 ≤1.2B 的 Qwen3 式模型上给出 **20–30% token 等效加速**，并**改善 LR 跨宽度/深度迁移**（2606.16899）。它只有 3 行改动、无额外优化器状态，且正是 Puro-2B 在 **RTX 5090** 上用的 MuonH（2608.27370）。我们已在 Muon 上，改造成本最低。

**结论 3（最便宜的确定性改进）：decay 比例 10% 偏短，min-lr 用"比例"而非"绝对值"是次优。**
2026 年两个独立来源都指向更长 decay：Puro 在 0.6B / TPP 20（50/100 抽查；我们 8B/288M = 27.8 TPP，正落在这个区间）扫 decay ratio 0.2–1.0，**最优点 0.4–1.0、近优区下沿 0.4–0.8**，并用线性 decay；annealing-ratio 定律给出 `R_opt = λ_T·T^(−0.95)`（2512.13705）。min-lr 侧，Step Law 实测"固定 1e-5"优于"lr_max/10"，因为后者在高峰值 LR 下把末期 LR 抬得过高（2503.04715）。我们当前 min-lr=0.1×8e-4=8e-5，方向上是"末期 LR 偏高"。

---

## 1. 我们的 setting 与"可借鉴性"判据

### 1.1 现状（事实，来自 `r93-p1-final-report.md`）

| 项 | 值 |
|---|---|
| 形状 | d24_c1 dense，dim 1024，24L，V=41,280（K=40k），seq 8192，Engram 220M |
| 训练量 | 7.998B tokens / 3,814 步 / 35.5 h / 62.5k tok/s / MFU 51.6% |
| 配方 | LR 8e-4、WSD warmup 30 步 / decay-start 0.9 / min-lr 0.1、z-loss 1e-4、EMA 0.995/step、grad-clip 1.0、bf16 + fp8 head |
| 优化器 | v41 bundle：Muon（update-RMS 0.18、Nesterov 0.95、wd 0.1）+ Sinkhorn（嵌入/头/Engram 表）+ AdamW（β=0.9/0.95） |
| 数据 | `p2_climbmix_40k` 8.568B，单 epoch 0.934（**无重复**） |
| 曲线 | 1B 1.2771 / 2B 1.1897 / 4B 1.1176 / 6B 1.0905 / 8B 1.0662（EMA 1.0620） |
| 成本 | ¥115.5 训练 + ¥12 探针 ≈ ¥128；**1B tokens ≈ 4.44 h ≈ ¥14.4** |

### 1.2 换算率：0.001 bpb 值多少钱（本报告自算，非论文结论）

以上面的 3 参数拟合为局部模型：`L = 0.887 + 0.390·(D/1B)^(−0.372)`。在 8B 处：

| Δbpb | 需要的额外 tokens | ≈ ¥ | 含义 |
|---:|---:|---:|---|
| 0.001 | +0.12B（+1.5%） | 1.7 | 噪声底以下 |
| 0.005 | +0.63B（+7.9%） | 9.1 | 我们 3M panel 的可辨噪声底附近 |
| 0.010 | +1.33B（+16.6%） | 19.2 | 一次 ¥19 实验的盈亏线 |
| 0.020 | +2.98B（+37%） | 43.1 | 值得为之改配方 |
| 0.050 | +11.2B（+140%） | 161.7 | 只有换数据/换容量才够得着 |

**这条表决定了实验经济学**：一次 ¥30 的锚点实验，只有当它筛掉的改动 ≥0.010 bpb（或能避免一次 ¥110 的错跑）时才划算。**配方微调的主要价值是"避免错跑"，不是"用更少 token 换更低的 bpb"。**

### 1.3 关键复核：对 5 个 mark 的两种拟合

| D (B) | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| bpb | 1.2771 | 1.1897 | 1.1176 | 1.0905 | 1.0662 |
| 纯幂律局部 β | — | 0.102 | 0.090 | 0.060 | 0.078 |

- **拟合 A（3 参数，全部 5 点）**：`L = 0.8873 + 0.3900·D^(−0.3719)`，最大残差 **0.0029 bpb**；β 逐段恒为 ~0.37–0.39。
- **拟合 B（β 固定 0.154，E/A 自由）**：`L = 0.5054 + 0.7668·D^(−0.154)`，最大残差 **0.0072 bpb**（也可接受；E 与 β 强相关，5 点定不住 E）。
- **外推检验（只用 1–4B）**：拟合 B 形式 → 8B 预测 **1.0485**；3 参数 → **1.0581**；**纯幂律（E=0）→ 1.0436，而当初用的 β=0.154 外推给 0.89**（0.89 对应的 E≈0）。

**一个必须说明的口径问题**：若 1–6B 的 mark 是 stable 段中途 eval（本报告的理解：单条 run、无分叉），而 8B 是 cooldown 之后的终值，那么这条曲线混了两种口径——river-valley（2410.05192）预测 stable 段损失偏高，所以 8B 点其实被"冷却红利"美化了。统一口径只会让后段**更平**，**因此上面的结论方向是保守的**；但它同时意味着 E 的估计会被口径污染，这是必须做跨尺寸联合拟合（E5）的第二个理由。

**结论**：错的不是 β，是 `E=0` 这个假设。今后任何"短跑外推长跑"都必须至少带一个可辨识的地板项，且地板要在**多模型尺寸联合拟合**里定（我们已有的 0.2B/0.3B/0.4B 族可直接复用，零 GPU 成本）。⚠️ 5 点 3 参数仍属弱辨识，E∈[0.51, 0.89] 都"能拟合"，**这就是为什么必须用跨尺寸联合拟合，而不是单条曲线**。

### 1.4 可借鉴性分级（本报告用）

| 级 | 含义 | 处理 |
|---|---|---|
| **A** | 同规模（≤1.2B）+ 同栈（Muon / 5090 / bf16-fp8），可直接迁移 | 直接进实验队列 |
| **B** | 同规模但不同栈/不同数据，只迁移"排序"不迁移绝对值 | 需本地 0.3–1B 复核 |
| **C** | 更大规模或不同优化器，只能作方向参考 | 不进本轮 |
| **D** | 与本项目目标冲突（如为吞吐牺牲 token 效率） | 列入"别做" |

---

## 2. 概览表

### 2.1 学习率调度

| # | 方法 | 来源（年） | 关键证据 | 可借鉴性 | 工时 / GPU | 预期 Δbpb@8B |
|---|---|---|---|---|---|---|
| S1 | WSD（warmup-stable-decay），decay≈10% | 2404.06395（2024，高引用） | 0.04–2B；10% decay 足够、2.5% 不足 | **A**（我们已在用） | 0 | 已实现 |
| S2 | WSD 的 river-valley 解释 | 2410.05192（2024，高引用） | stable 段损失"虚高"、decay 段陡降；可从主支分叉 | **A**（解释我们的曲线） | 0 | 0（诊断） |
| S3 | cooldown 形状与 β2 | 2508.01483（TMLR 2025） | 5 种形状；**1−sqrt 在 bias–variance 上最优**；cooldown 期**加大 β2 稳定有益**；形状效应 ≈ 调 AdamW 超参 | **A** | 0.5d / ¥10 | 0.002–0.010（估） |
| S4 | annealing ratio 定律 `R_opt=λ_T·T^−0.95` | 2512.13705（AAAI 2026） | 1B MoE，30k–300k 步；R_opt 随步数**下降**、随模型**上升**、跨数据集可迁移 | **B** | 0.5d / ¥30 | 0.003–0.012（估） |
| S5 | Puro 的 decay-ratio 扫描（0.2–1.0，线性） | 2608.27370（2026 新） | 0.6B / TPP 20（50/100 抽查）；最优 **0.4–1.0**、下沿 0.4–0.8；高峰值 LR 需更长 decay | **A**（同硬件同 TPP） | 0.5d / ¥35 | 0.003–0.012（估） |
| S6 | min-lr 用绝对值而非比例 | 2503.04715（2025） | 1B/80B：固定 1e-5 优于 lr_max/10；比例法使最优 LR 左偏、末期收敛变差 | **A** | 0.2d / ¥10 | 0.001–0.005（估） |
| S7 | horizon-free：constant / 1/√t + 权重平均 | 2602.03702（ICML 2026） | 150M/300M，至 32× Chinchilla；与调好的 cosine **相当** | **B** | 1d / ¥20 | 0.000–0.005（估） |
| S8 | Schedule-Free+ | 2605.19095（2026 新） | 1000 TPP 时同等损失省 **31% 时间**；**20–100 TPP 无最终损失优势** | **C**（我们在 27.8 TPP） | 2d | ≈0（本区间无优势） |
| S9 | WSqD（1/√t 基底 + 线性收尾） | 2607.10959（2026 新） | 随机凸情形 O(1/√T) 最优；SlimPajama 上优于 WSD（数值未核） | **C** | 1d | 未证实 |
| S10 | 调度形状的理论解（decay 指数 2q−1） | 2602.06797（2026 新） | 难任务 → WSD 式；**容量决定怎么 decay、难度决定何时 decay** | **C**（解释性） | 0 | 0 |
| S11 | 多幂律损失曲线预测（MPL） | 2503.12811（2025） | 少量调度拟合并预测**未见调度**的整条曲线；搜出的调度优于 cosine | **A**（规划工具） | 1d / ¥0 | 0（省错跑） |

### 2.2 超参规律与迁移

| # | 方法 | 来源（年） | 关键证据 | 可借鉴性 | 工时 / GPU | 预期 Δbpb@8B |
|---|---|---|---|---|---|---|
| H1 | muP | 2203.03466（2022，经典） | 宽度迁移 LR | **D**（我们形状固定，迁移收益小） | 3d+ | 0 |
| H2 | **WD 比 muP 更决定 LR 迁移** | 2510.19093（ICLR 2026） | muP 假设只在训练极早期成立；**是 weight decay 在稳定跨宽度更新动态**；muP ≈ 隐式 warmup | **A**（改 WD / warmup 替代 muP） | 0.5d / ¥15 | 0.000–0.005（估） |
| H3 | **norm transfer**（最优 (η,B) ⟺ 固定输出层算子范数） | 2510.03871（2025-10，v2 2026-01） | ≤1.3B / 138B tokens；最优 (η*,B*) 对应同一算子范数；可作诊断 | **A**（诊断） | 1d / ¥0 | 0（提高 LR 选择置信度） |
| H4 | Step Law：`η=1.79N^−0.713D^0.307`、`B=0.58D^0.571` | 2503.04715（2025） | 3,700 个 LLM / 100T tokens；预测最优与穷举差 0.094% | **B**（公式可作先验，但我们的 a_raw=−0.89 与 −0.27 不符） | 0.5d / ¥0 | 0（校准先验） |
| H5 | Model Wind Tunnel / ModelTunnel v2 | 2404.06395 / 2506.07900（2024/2025） | 百万级参数代理 → 外推；v2 用 μP + ScalingBench 作指标 | **A**（我们已在做 0.3–1B 锚点） | 已在用 | 0 |
| H6 | critical batch size：随**数据量**而非模型规模增长；训练中先升后平台 | 2410.21676（ICLR 2025） | 85M–1.2B / C4 | **B** | 0.5d | 0 |
| H7 | batch-size warmup（小→大） | 2505.23971（2025） | CBS 初始化≈0，随时间上升；据此训 OLMo 1B 大 batch | **D**（我们按小时付费，token 效率优先） | — | 负 |

### 2.3 优化器

| # | 方法 | 来源（年） | 关键证据 | 可借鉴性 | 工时 / GPU | 预期 Δbpb@8B |
|---|---|---|---|---|---|---|
| O1 | Muon（+WD +update-scale） | 2502.16982（2025，高引用） | 3B/16B MoE / 5.7T tokens；计算效率 ~2×；`0.2·O·√max(A,B)`、wd 0.1 | **A**（已在用，我们是 0.18） | 0 | 已实现 |
| O2 | **Hyperball / MuonH** | 2606.16899（2026 新，"Fantastic II"） | ≤1.2B Qwen3 式；**20–30% token 等效**；改善 LR 跨宽度/深度迁移；规则 3 行：`W←R·Norm(W−ηR·Norm(u))` | **A ★ 最高优先** | 0.5–1d / ¥28 | **0.005–0.015（估）** |
| O3 | 优化器公平基准（10 个，0.1–1.2B，1–8× Chinchilla） | 2509.02046（2025） | 矩阵类优化器加速从 **1.4×（0.1B）缩到 1.1×（1.2B）**；排名会随 LR decay **翻转**；Lion 最优 wd≈0.6 | **A**（方法论：必须"跑完 decay 再比"） | 0 | 0（校准预期） |
| O4 | SOAP（Shampoo+Adam 特征基） | 2409.11321（ICLR 2025） | 360M/660M；大 batch 区 **≥40% 更少迭代 / ≥35% 墙钟** | **C**（需特征分解；Muon 已占位） | 3–5d | 0.005–0.015（估，但工时不合算） |
| O5 | Lion | 2302.06675（2023，高引用） | 内存省；最优 wd≈0.6，调参敏感 | **D**（小规模无优势） | 1d | ≤0 |
| O6 | Scion（范数约束 LMO 家族） | 2502.07529 系列 / 2510.03871 | 与 Adam 同族 scaling 规则；norm transfer 同样成立 | **C**（与 Hyperball 同源思想） | 2d | 0.005–0.015（估） |
| O7 | Muon 的配对基准（数据/时间/调参预算） | 2610.21555（NeurIPS'26 WS） | 17.7M/33.6M **极小**模型；Muon 数据效率更好，但**墙钟在语言上反而差 3.4–7.0%**（未优化 NS 占 56–70% 时间） | **B**（提醒：收益依赖 kernel 成熟度；我们 NS 已融合） | 0 | 0 |
| O8 | 数据量增大后 Muon 的微调优势反转 | 2611.03633（2026-11 新） | 预训练数据预算大 → 微调后 AdamW 反超 | **C**（我们不训 SFT） | 0 | 0 |
| O9 | AdEMAMix（双 EMA，含"很旧"的动量） | 2409.03137（2024） | 语言建模上优于 AdamW；适合长跑 | **B** | 2d（改 Muon 不直接） | 未证实 |
| O10 | EMA / 权重平均 / checkpoint merging | 2511.18903（2025）、2602.03702（2026）、2507.17634（2025） | 用"末段 ckpt 加权平均"**替代** LR decay 可补回 +1.64% 平均分；WSM：merge 时长是最关键因子 | **A**（我们已有 EMA） | 0.5d / ¥2 | 0.000–0.004（估） |
| O11 | PACE（把权重拉向 EMA 的优化器包装） | 2606.25086（2026 新） | 为"返回平均模型"重新设计优化器；理论+小规模 | **C** | 2d | 未证实 |

### 2.4 稳定性与精度

| # | 方法 | 来源（年） | 关键证据 | 可借鉴性 | 工时 / GPU | 预期 Δbpb@8B |
|---|---|---|---|---|---|---|
| T1 | z-loss 1e-4 | 2204.02311（PaLM，2022，经典） | 抑制 logit 漂移；已是我们默认 | **A** | 0 | 已实现（−0.005，噪声内） |
| T2 | **OEC（输出嵌入居中）/ μ-loss** | 2601.02031（2026 新） | 论证 z-loss/softcap 只治症状；OEC 稳定性优于 z-loss、与 softcap 持平；μ-loss 对超参更鲁棒 | **B** | 1–2d / ¥5 | 0.000–0.003（估） |
| T3 | z-loss 的反向几何诊断 | 2609.16179（2026 新） | 同一 z-loss 前向值可对应不同 logit 梯度；绑定嵌入/融合 loss 会改变传输 | **C**（诊断） | 0 | 0 |
| T4 | logit soft-capping | 2408.00118（Gemma 2，2024，高引用） | Gemma 2 在 2B–27B 上采用 logit soft-capping 稳定训练（架构节，本报告未核具体系数与消融）；对**小模型**可能压制置信度（社区有"去掉 softcap"的实践，**未证实**） | **C** | 0.5d / ¥10 | −0.002–+0.003（估） |
| T5 | grad-clip 1.0（现状）/ AdaGC 自适应裁剪 | 2502.11034（2025） | AdaGC 用张量级 EMA 归一化裁剪，抑制 spike 污染一二阶矩 | **B** | 1d / ¥5 | 0.000–0.003（估） |
| T6 | Muon 族 spike 抑制（EnergyCal） | 2610.37062（2026 新） | 4B MoE：spike 负担 −99.5%、ppl −4.06% | **C**（MoE 专用） | 2d | 未证实 |
| T7 | bf16 vs fp8 训练 | 2405.18710（2024）、2411.04330（2024）、2608.27370（2026） | 2024：FP8 稳定性/超参敏感性差于 bf16；2024 精度 scaling law：降精度=降"有效参数量"；2026：5090 上 blockwise FP8 训练 1.4T tokens 可行 | **D（我们已测负收益）** | 已测 | 0（d12 步时 −6.5~−26%；修正口径后 +0.2~1%） |
| T8 | QAT（末段 40% tokens） | 2506.07900（MiniCPM4，2025） | BitCPM4：QAT 占 >40% tokens（≈2× decay 段）≈ 从零 QAT | **C**（后训练阶段） | 3d+ | 不进预训练 |
| T9 | 1.58-bit / 三值 | 2402.17764（2024，高引用） | 同规模同 token 匹配 bf16 | **D**（现阶段） | — | — |

### 2.5 数据顺序 / 长跑对策

| # | 方法 | 来源（年） | 关键证据 | 可借鉴性 | 工时 / GPU | 预期 Δbpb@8B |
|---|---|---|---|---|---|---|
| D1 | 升序质量课程 + **温和 decay / 模型平均** | 2511.18903（2025-11，v3 2026） | 1.5B/30B：常数 LR 下课程远胜随机打乱；**标准 decay 会吃掉这份收益**；改用"温和 decay + 末段 ckpt 加权平均"→ +1.64% 平均分 | **B** | 2–3d（需质量分）/ ¥0 | 0.003–0.010（估） |
| D2 | 课程作 warmup、随后随机采样 | 2506.11300（2025） | 200 个模型 / ≤100B tokens：达基线少用 **18–45% 步**；作为 warmup 前置最稳 | **B** | 2d | 0.002–0.008（估） |
| D3 | 数据受限 scaling（重复 ≤4 epoch 几乎无损） | 2305.16264（2023，高引用） | 至 900B tokens / 9B 模型 | **A**（我们 0.934 epoch，已排除重复） | 0 | 0 |
| D4 | 多阶段 continual：LR re-warm + re-decay + replay | 2403.08763（2024）、2308.04014（2023） | 405M 与 10B；**必须**配旧数据 replay 才追平重训 | **D**（无新数据时无收益证据） | 2d / ¥110 风险 | 未证实（高风险） |
| D5 | Puro 的 CMA（课程 + 模型平均） | 2608.27370（2026 新） | 1.4T tokens 配方的一等公民 | **B** | 2d | 0.003–0.010（估） |

---

## 3. 逐方法详解

### 3.1 学习率调度

#### 3.1.1 cosine vs WSD：2025–2026 的共识是"打平，但 WSD 更可用"

**做法**：WSD = 线性 warmup + 恒定 peak + 末段 decay（线性 / 1−sqrt / cosine / 指数）。

**证据**
- MiniCPM（2404.06395，2024，高引用）：0.04B–2B；**decay 占总步数 10% 足够拿到最好结果，2.5% 明显不足**；WSD 让"数据-模型 scaling law"可以在一条主支上分叉测量。
- River Valley（2410.05192，2024，高引用）：stable 段迭代在高 LR 下沿"河谷"快速前进但横向振荡，所以**损失在 stable 段偏高、在 decay 段陡降**；因此 WSD 与 cosine 最终打平，但 WSD 支持任意时刻分叉。
- Cooldown 动力学（2508.01483，TMLR 2025）：只研究 cooldown 段，发现形状选择的效果量与调 AdamW 超参**相当**；`1−sqrt` 形状在 bias–variance 折中上最优；**cooldown 期提高 β2 有稳定收益**。
- Annealing 迁移（2512.13705，AAAI 2026）：1B MoE、30k–300k 步；`R_opt = λ_T·T^(α_T)`，α_T≈−0.946；**R_opt 随总步数下降、随模型尺寸上升**，且跨验证集/数据可迁移。
- Puro（2608.27370，2026）：0.6B、TPP 20（50/100 抽查）、GBS 512、**线性** decay，TPP 20 上 5 档 decay ratio 全扫 0.2–1.0；**最优点 0.4–1.0，近优区下沿 0.4–0.8**；"更大 peak LR 要更长 decay""更长 horizon 要更长 decay"。
- 理论（2602.06797，2026）：最优 decay 指数 `2q−1`（q 为容量指数）；**难任务 → WSD 式（大部分时间停在最大可行 LR），容量决定 decay 陡度**。
- WSD 并非 Transformer 特有（2601.09000，2026：CNN 上同样的 sharpness/路径特征）。

**我们的可借鉴性（A）**：我们已在 WSD + 10% decay。真正要改的是 **decay 比例与形状**，不是"换 cosine"。注意 Puro 的"effective LR"是 Hyperball 归一化后的量（~1e-2 量级），**不能与我们的 8e-4 直接比**；可迁移的是"比例"这一维的排序。

**预期**：decay 0.1→0.25–0.3 + 1−sqrt + min-lr 绝对值，合计 0.004–0.012 bpb（估）。

#### 3.1.2 decay 比例：三个 2025–2026 来源都指向"比 10% 长"

| 来源 | 场景 | 结论 | 对我们的含义 |
|---|---|---|---|
| 2404.06395 | 0.04–2B | 10% 够、2.5% 不够 | 我们 10% 是"够用"，不是"最优" |
| 2512.13705 | 1B MoE，30k–300k 步 | R_opt 随 T **下降**，α≈−0.95；我们的 T=3.8k 远小于其拟合区间（**外推不可靠**） | 我们的 R_opt 大概率 **≥0.2** |
| 2608.27370 | 0.6B，TPP 20（50/100 抽查） | R_opt 0.4–1.0（下沿 0.4–0.8） | **最强的直接证据**：TPP 覆盖我们的 27.8 |

**可借鉴性（B/A）**：Puro 与我们同硬件、同 TPP 区间，属 A 级；但它的结论建立在**较高 effective LR**（~1e-2 量级）之上。对我们的真正风险不是"低 LR 段浪费 token"，而是**若 8e-4 本身偏低（探针只测了上侧，下侧未测），把 decay 从 10% 拉到 30% 会缩短本就不够的高 LR 期**，可能反而变差——**这正是必须先在 0.6B 上验的原因**。

**预期**：0.003–0.012 bpb（估）；若方向错，−0.005 以内（有界风险）。

#### 3.1.3 decay 形状：1−sqrt 有最好的 bias–variance 证据

2508.01483 在 5 种形状（linear / 1−sqrt / cosine / mirror-cosine / square）上给出：**1−sqrt 落在 bias–variance 最优位置**，"lowered linear"（带地板/幂次的线性）紧随其后。并且指出：**形状最优点与最差点之间的差，等于调 AdamW 超参能拿到的量级**——换句话说，形状值得调一次，但不值得反复调。

**可借鉴性（A）**：我们当前是线性（decay-start 0.9 后的线性到 min-lr）。改 1−sqrt 是**改 3 行代码、零额外成本**。

**预期**：0.001–0.006 bpb（估）。

#### 3.1.4 min-lr：该用绝对值，不是 peak 的比例

Step Law（2503.04715，3,700 个 LLM / 100T tokens）实测：`lr_min = lr_max/10` 会让最优 LR 区域**左偏**，并"阻止损失收敛到更好的局部最优"；把末期 LR 固定为 **1e-5** 更好。

**可借鉴性（A）**：我们 min-lr=0.1 → 末期 8e-5，比 1e-5 高 8×。**但**注意 Muon 的 LR 语义与 AdamW 不同（update-RMS 0.18 归一化），绝对数值不可照搬——应改成"末期 LR ≤ 2e-5 量级"或直接改成 `min-lr = 0.02–0.03×peak` 做一次 A/B。

**预期**：0.001–0.005 bpb（估），风险很低（末期 LR 更低只会让曲线更平滑）。

#### 3.1.5 warmup：我们 30 步（0.8%）不是问题

MiniCPM 明确"warmup 只要够长，对最终性能影响很小"；2505.21910（2025）给出无 warmup 训练的理论（`W_q^T W_k` 的谱能量集中 → 熵塌缩），并证明用"限制相邻步更新比"可替代 warmup。我们 8B 跑完 gn 稳定、无 spike，**warmup 不是当前瓶颈**；若将来把 LR 提到 1.2e-3 以上，再考虑 warmup 3–5%。

**可借鉴性（D，暂不动）**。

#### 3.1.6 horizon-free / anytime：长期方向，不是本轮

- 2602.03702（ICML 2026）：150M/300M，至 **32× Chinchilla**；`constant` 与 `1/√t` + 权重平均 ≈ 调好的 cosine。→ 说明"不做 decay、靠平均"在**我们的 TPP 区间之外**也成立。
- 2605.19095（2026）：Schedule-Free+ 在 **1000 TPP** 省 31% 时间，但**在 20–100 TPP 无最终损失优势**——正是我们的区间（27.8 TPP）。**不要换**。
- 2607.10959（2026）：WSqD（1/√t 基底 + 线性收尾）horizon-free 且 minimax 最优；SlimPajama 上优于 WSD（具体数值**未证实**）。

**结论**：我们的 8B 是"中等 TPP"，horizon-free 调度的收益区间在两端（很短或很长），**本轮不动**；但可把 `1/√t` 作为下一次"延长训练"时的候选。

### 3.2 超参规律与迁移

#### 3.2.1 muP 在 2025–2026 被"降级"：真正起作用的是 weight decay

- muP（2203.03466，2022，经典）：宽度方向的 LR 迁移。
- **2510.19093（ICLR 2026）**：大规模实证（LLM 训练）显示 muP 的几何假设**只在训练极早期成立**；之后是 **weight decay** 在稳定跨宽度的内部表示更新动态；**muP 主要等价于一种隐式 warmup**，可用改过的 warmup 替代；这也解释了为什么 muP 需要独立的 weight decay 变体才能迁移好。
- **2510.03871（2025-10）**：对 Adam 与 Scion，跨模型/数据尺度的最优 `(η*, B*)` 对应**同一个输出层算子范数**（"norm transfer"），≤1.3B / 138B tokens；这是"必要不充分"条件；且**分层 LR**（输出层最敏感、隐层可更低）有额外收益。

**我们的可借鉴性**
- **不要**现在切 muP（要重标定 init/LR/所有 group，3 天+，且我们形状固定、迁移收益本来就小）→ **D**。
- **可以**用 H3 的"输出层算子范数"作**诊断量**：在 0.3B 的 LR 扫描里记录 6B/8B 的 head 算子范数，若不同 LR 的最优范数收敛到同一值，就说明我们的 LR 探针有了跨规模的锚（成本 ¥0，只是加日志）。
- **可以**把 H2 的结论用于 warmup 设计：既然 muP≈隐式 warmup、且 WD 是稳定器，那么"更长 warmup + 更大 WD"是比"换参数化"更便宜的等价手段（但见 3.3.6，WD 与 Hyperball 冲突，二选一）。

#### 3.2.2 Step Law：可用的先验，但我们的实测斜率不支持外推

`η(N,D)=1.79·N^−0.713·D^0.307`，`B(D)=0.58·D^0.571`（2503.04715）。**注意**：我们 S1 实测 a_raw=−0.89（0.2B→0.4B），远陡于 −0.27/−0.35，且已判定为"随训练步数增长的不稳定门槛"而非可外推定律（`r93-hp-search-results.md` §2）。Step Law 的价值在于 **(a) 给出"bpb 对 LR 的凸性 + 宽最优"这一形状**，(b) 给出 **min-lr 用绝对值** 的实证。

**可借鉴性（B）**：只借"形状"和"min-lr"两条，不借指数。

#### 3.2.3 critical batch size 与 tokens/step：我们的 S3 结论与文献一致

- 2410.21676（ICLR 2025）：CBS 主要随**数据量**（而非模型规模）增长，85M–1.2B。
- 2505.23971（2025）：CBS 在初始化时≈0，训练中快速上升后平台；据此提出 **batch-size warmup**，用于 OLMo 1B 的大 batch 训练。
- 我们 S3：2^22 tokens/step 即使 LR×√2 也**贵 0.138 bpb** → 我们在 CBS 之上，**保持 2^21**。

**可借鉴性**
- batch-size warmup 对我们**不适用（D）**：我们按小时付费、tok/s 与 batch 基本无关，所以"用更大 batch 换更少步数"不省钱，只会损失 token 效率。
- 但 **CBS 随数据量增长** 提示：8B 时 CBS 比 0.3B 时大，**2^21 在 8B 可能偏保守**。若某天要"同样 ¥ 跑更多 tokens"，可测 2^22 + 全长 decay（S3 的 0.3B 结论不一定外推到 8B）——但这是"吞吐 vs 效率"的权衡，不是本轮重点。

#### 3.2.4 wind tunnel / 小模型代理：我们的"锚点"策略是文献主流

MiniCPM 的 Model Wind Tunnel 用百万级参数模型定超参；MiniCPM4 的 ModelTunnel v2 加了两件事：**用 μP 做跨规模校验**、**用 ScalingBench（下游任务验证集的 loss）代替开源自建语料的 loss**（2404.06395、2506.07900）。

**可借鉴性（A）**：我们的 0.3–1B 锚点 + 3M panel 已经是同一路线。**可补的一课**：MiniCPM4 强调"loss 指标要能映射到目标能力"；我们的 bpb panel 是 held-out ClimbMix，**与目标分布一致**，这点没问题；但**短跑锚点必须跑完整 decay 再比**（2509.02046 的排名翻转警告）。

#### 3.2.5 grad accum 与有效 batch

我们 b1×8192×accum256。文献侧没有"accum 与真实大 batch 不等价"的新证据（数学上等价，差异来自 BN/dropout 类，我们没有）。**唯一相关的新证据**是 2509.02046 的"排名会在 LR decay 期翻转"——这提醒我们：**任何 A/B 都必须在 decay 结束后取点**，否则 accum 数不同导致的步数不同会污染结论。

### 3.3 优化器

#### 3.3.1 Muon 的真实收益：2025 乐观，2026 收敛

| 工作 | 规模 | 结论 |
|---|---|---|
| 2502.16982（2025，高引用） | 3B/16B MoE，5.7T tokens | 计算效率 ~**2×**；关键技巧：**加 weight decay**、**per-parameter update scale**（`0.2·O·√max(A,B)`），使 Muon 可直接复用 AdamW 的 LR/wd |
| 2509.02046（2025） | 10 个优化器，0.1–1.2B，1–8× Chinchilla | 公平调参后，矩阵类优化器加速从 **1.4×（0.1B）→ 1.1×（1.2B）**；**排名会随 LR decay 翻转**；Lion 最优 wd≈0.6（vs AdamW 0.1） |
| 2610.21555（2026，NeurIPS'26 WS） | 17.7M / 33.6M（A100） | Muon 数据效率更好（AUC −3.25~22.44%），但**时间归一化后在语言上比 AdamW 差 3.4–7.0%**——因为参考实现里 Muon 调用占 56–70% 训练时间 |
| 2611.03633（2026-11） | 预训练→微调 | 预训练数据预算够大后，**AdamW 在微调后反超** |

**对我们的含义**：我们的 Muon 已融合进 CUDA graph（NS 开销被摊薄，MFU 51.6%），所以 2610.21555 的"时间更差"不适用；但 2509.02046 的"**加速随规模缩小**"是真实警告——**Muon vs AdamW 在我们规模上可能只有 1.1–1.2×，而不是 2×**。既然无法回头换 AdamW（8B 已跑完），这条只用于**校准未来预期**。

#### 3.3.2 ★ Hyperball / MuonH：本轮最值得做的优化器改动

**做法**（2606.16899 的公式，3 行）：
```
R = ||W_0||_F                      # 训练开始时记一次
u = base_optimizer_update(...)     # 即我们的 Muon 更新（含 Nesterov）
W ← R * normalize(W - η * R * normalize(u))   # 步长 η·R，投影回半径 R 的球面
```
即：**更新矩阵归一化 → 按 η·R 走一步 → 权重投影回初始范数球**。不再需要 weight decay。

**证据**
- 2606.16899（2026-06，"Fantastic Pretraining Optimizers II"）：动机理论是"weight decay 下训练收敛到只依赖超参的平衡范数，**是 weight decay 在决定角学习率**（权重方向变化多快）"；在 ≤1.2B 的 Qwen3 式模型上，**MuonH 相对 weight-decay 基线有 20–30% token 等效加速**；并且**改善 LR 跨宽度/深度迁移**（对我们的 LR 迁移之痛是直接对症）。
- 2608.27370（2026-08，Puro-2B）：**RTX 5090** 上用 **MuonH**（+ blockwise FP8 + CMA）训 1.4T tokens，<$6.9K；其"effective LR"框架正是建立在"Hyperball 使有效 LR 显式化"之上。
- 2509.02046 的"矩阵优化器收益随规模缩小"与 2606.16899 的"缩小来自 weight decay 的软范数控制"互为因果——**MuonH 就是对这个缩小的补丁**。

**我们的可借鉴性（A，最高优先）**
- 实现：在 `nanov41/muon.py` 的 Muon step 后加一次 Frobenius 归一化 + 投影；**无额外状态**（只多存每矩阵的 R，可复用 param 形状），单步开销可忽略（一次 reduce）。**0.5–1 人日**（含 shape-grouped 3-D 专家矩阵的逐切片处理）。
- **必须重标定 LR**：步长变成 `η·R`（角度量纲），与现在的 `0.18/rms` 缩放完全不同量级。做法：0.3B 上 3 臂对数网格（相邻 3×）粗扫，再用 0.6B 做 A/B。
- 与 Sinkhorn 的关系：Sinkhorn 作用于嵌入/头/Engram 表（行随机矩阵归一化），**不受影响**；Hyperball 只加在 Muon 组的矩阵上。

**预期**：0.6B 上 10–25% token 等效（估，按 1.2B 的 20–30% 打折）；折算到 8B = **0.005–0.015 bpb**（估）。**这是配方侧唯一有望 ≥0.01 的单项。**

**成本**：0.3B LR 粗扫 3×0.25B=0.75B（¥11）+ 0.6B A/B 2×0.6B=1.2B（¥17）≈ **¥28 + 1 人日**。

#### 3.3.3 SOAP / Shampoo / Scion：证据好，但工时不合算

- SOAP（2409.11321，ICLR 2025）：把 Shampoo 解释为"在预条件子特征基里跑 Adafactor"；360M/660M 上大 batch 区 **≥40% 更少迭代、≥35% 更少墙钟**（vs AdamW），比 Shampoo 再好 ~20%；代价是每 N 步一次特征分解。
- Scion（2502.07529；Disco 实现见 2510.03871）：范数约束 LMO 家族，与 Adam 同族的 (η,B) scaling 规则；"norm transfer"同样成立。
- **判断（C）**：我们在 24 层小模型上，特征分解的固定开销占比会比 660M 更高；且 Muon 已提供矩阵预条件。**ROI 不如 Hyperball（3 行 vs 3–5 天）**。列为"若 Hyperball 失败再考虑"。

#### 3.3.4 Lion：不适合我们

Lion（2302.06675，2023，高引用）内存省、但**最优 wd≈0.6**、对 LR/wd 敏感；2509.02046 的公平基准里 Lion 不是最快的矩阵类。**D 级，不做。**

#### 3.3.5 AdEMAMix / MONA：长记忆动量，值得关注但不在本轮

- AdEMAMix（2409.03137，2024）：两个 EMA（快+慢）混合，语言建模优于 AdamW；适合**长跑**（正是我们的场景）。
- MONA（2605.26842，2026）：Muon + Nesterov 式曲率加速，MoE 三尺度上优于 Muon/AdamW。
- **判断（C）**：都要改优化器内核/状态，收益未在同规模复现；**排在 Hyperball 之后**。

#### 3.3.6 weight decay / EMA / checkpoint averaging

- **weight decay**：2510.19093 说 WD 是跨宽度稳定器；但 2606.16899 说 WD 的"软范数控制"正是收益缩水的根源 → **两者冲突：wd 与 Hyperball 二选一**（MuonH 时代 wd=0）。
- **EMA**：我们实测 −0.0042 bpb（已 banked）。2026 年三篇工作把"权重平均"从技巧抬成方法：
  - 2602.03702：`1/√t` + 权重平均 ≈ cosine；
  - 2511.18903：**用"末段 ckpt 加权平均"替代 LR decay**，把课程的收益从"被 decay 吃掉"救回 +1.64%；
  - 2507.17634（WSM）：**merge 时长是最关键因子**，比 ckpt 选择/权重方案都重要。
- **可操作**：我们已有 EMA(0.995)。**再加一层"末段 N 个 ckpt 的等权/三角权平均"**（若 ckpt 粒度 ≥ 每 0.5B 一个），成本≈¥2（只 eval），预期 **0.000–0.004 bpb**（估），与 EMA 可能互补（EMA 是无限长指数窗，ckpt 平均是有限窗）。

### 3.4 稳定性与精度

#### 3.4.1 z-loss / OEC / soft-capping

- z-loss（2204.02311 PaLM，2022，经典）：我们已开 1e-4，S2 实测 −0.005（噪声内），快速路径只 +0.46% 步时。**保留**。
- **OEC（2601.02031，2026）**：指出 z-loss/softcap 只治症状；不稳定来自**输出嵌入的各向异性**；OEC（μ-centering 确定性操作 / μ-loss 正则）在稳定性上**优于 z-loss、与 softcap 持平**，且 μ-loss 对超参更鲁棒。→ 若将来出现"末期 logit 漂移"，优先试 μ-centering（确定性、零超参），而不是加 softcap。
- logit soft-capping（2408.00118 Gemma 2，2024，高引用）：Gemma 2 在 2B–27B 上采用（架构节，本报告未核具体系数/消融）；但**小模型上可能限制置信度**（社区有移除 softcap 的实践，**未证实**）。我们的 head 已是 fp8 前向，softcap 会改变 logit 动态范围，**交互不确定**。

**判断**：z-loss 不动；OEC 留作"出现末期不稳定"时的第一备选（1–2 人日）；softcap **不做**。

#### 3.4.2 grad clip / spike 抑制

- 我们 grad-clip 1.0，8B 全程 gn 稳定（S1 记录 0.09–0.31），**没有 spike 问题**。
- AdaGC（2502.11034，2025）：张量级自适应裁剪，防止异常梯度污染一/二阶矩；2610.37062（EnergyCal，2026）针对 Muon 族 MoE spike（4B 上 −99.5% spike 负担）。
- **判断（B/D）**：没有 spike 就先不加机制（增加复杂度与调参面）。若未来把 LR 提高或换 MuonH 后出现 spike，再上 AdaGC。

#### 3.4.3 bf16 vs fp8：**我们已经测过，负收益**

- 2405.18710（2024）：当时的 FP8 方案稳定性/超参敏感性劣于 bf16，**不满足"可替代"标准**。
- 2411.04330（2024，ICLR 2025）：精度感知 scaling law——降精度等价于**降低有效参数量**；训练更大模型用更低精度可能 compute-optimal（拟合 465 次 run，验证到 1.7B/26B）。
- 2608.27370（2026）：**5090 上 blockwise FP8 训 1.4T tokens 可行**，质量分数稳定——但那是它们自己的 kernel 栈 + 2B 形状。
- **我们的实测**（`docs/fp8-first-plan.md`）：d12 形状下 FP8 body/experts **反而慢 6.5–26%**（修正口径后整网 +0.2~1%），已判 NO-GO；head fp8 保留。

**判断（D）**：**不要再投入 fp8 训练**。它在 2B 形状才有吞吐红利，我们 288M/24 层是带宽/启动受限区。**唯一可考虑的是 Engram 表的 fp8 存储**（+0.3~0.8% tok/s，省 220MB），已在他们计划里。

#### 3.4.4 QAT：属于后训练，不进本轮预训练

- MiniCPM4 / BitCPM4（2506.07900，2025）：**QAT 占 >40% 总 tokens（≈2× decay 段）时，三元模型的最终 loss 接近"从零 QAT"**——即 QAT 只需末段，不必全程。
- 精度 scaling law（2411.04330）：QAT 的收益与"有效参数量损失"可预测；数据越多，量化退化越明显（对**推理**量化尤其如此）。
- 我们：head 已 fp8 前向；FP4-QAT 是 post-training 计划。**判断（C）：不在预训练里加 QAT**。

### 3.5 数据顺序 / 课程 / 多阶段

- **升序质量课程 + 温和 decay / 模型平均**（2511.18903，2025-11，v3 2026-05）：1.5B / 30B tokens；核心发现是"**课程与 LR decay 不兼容**"——常数 LR 下课程大幅领先随机，标准 decay 下优势消失；解法是 (1) 末期 LR 别降太狠，(2) **用末段 ckpt 加权平均替代 decay**；两者合用，平均分 +1.64%（相对随机打乱，无额外数据）。
- **课程作 warmup**（2506.11300，2025）：200 个模型 / ≤100B tokens；课程在早中期加速收敛，**达基线少用 18–45% 步**；最稳用法是"课程做 warmup，然后随机采样"。
- **数据受限**（2305.16264，2023，高引用）：≤4 epoch 重复几乎无损，之后收益迅速衰减。我们 0.934 epoch，**不适用**。
- **多阶段 / continual**（2403.08763，2024；2308.04014，2023）：LR re-warm + re-decay + **旧数据 replay** 才能匹配重训（405M 与 10B）。**没有新数据时，re-warmup 没有收益证据。**
- **Puro 的 CMA**（2608.27370，§3.4）：把"课程 + 模型平均"做成一等公民，是 1.4T token 配方的一部分。

**我们的可借鉴性（B）**：我们 8B 是单 epoch 随机打乱、无质量分。**可做的低成本版**：用已有质量分（若有）对**末 30% tokens 升序排列** + 更长 decay + 末段 ckpt 平均。这是 2–3 人日的数据工程 + ¥0 GPU。预期 0.003–0.010 bpb（估）。**风险**：若只改数据顺序而不改 decay，按 2511.18903 的结论**收益会被 decay 吃掉**——必须打包改。

### 3.6 "长跑后期变平"的已知对策（文献视角）

| 对策 | 来源 | 机制 | 对我们是否适用 |
|---|---|---|---|
| **更长 decay / 温和 decay** | 2512.13705、2608.27370、2511.18903 | 让模型在低 LR 下继续"沿河谷下坡" | ✅ 最便宜，先验最强 |
| **min-lr 降到绝对小值** | 2503.04715 | 末期 LR 不够低会卡在次优 | ✅ 几乎零成本 |
| **decay 形状 1−sqrt** | 2508.01483 | bias–variance 折中 | ✅ 零成本 |
| **cooldown 期加大 β2/momentum** | 2508.01483 | 降低末期方差 | ✅ 零成本（Muon momentum 0.95→0.98） |
| **权重平均 / ckpt merging 替代 decay** | 2511.18903、2602.03702、2507.17634 | 平均降低方差，等效延长有效训练 | ✅ 我们已有 EMA，可加 ckpt 平均 |
| **范数约束优化器（Hyperball）** | 2606.16899 | 让"角学习率"不随 wd 漂移，长跑后段仍在有效更新 | ✅✅ 最高优先 |
| **LR re-warmup** | 2403.08763、2308.04014 | 只在**分布漂移/新数据**时有效 | ❌ 无新数据，风险高 |
| **数据刷新 / 课程** | 2511.18903、2506.11300 | 提早期效率，末期需配温和 decay | ⚠️ 需质量分，收益被 decay 调制 |
| **换更大模型 / 加容量** | 2203.15556（经典） | 直接抬高 E 的天花板 | ⚠️ 预算外（¥300–600） |
| **更长训练** | 本报告 §1.2 | 边际 0.01 bpb/1.33B tokens | ⚠️ 性价比低但确定 |

---

## 4. 特别回答

### (a) 8B 长跑"后期 β 变平"最可能的原因，与最值得试的 2–3 个配方改动

**原因排序（按证据强度）**

1. **【主因，统计】幂律里少了常数项 E。** 5 点拟合 `E+A·D^−β` 残差 ≤0.003；把 β 固定成 0.154、仅加 E，用 1–4B 外推 8B 得 1.048–1.058（实测 1.066）。而不带 E 的幂律给 0.89。→ **"β 变平"是拟合假象；真实规律是"恒定 β + 地板"。** 目标 0.89 等于其中一个拟合的地板，从一开始就不可达。
2. **【主因，物理】地板 E 由"数据熵 + 容量"决定。** 288M active / 27.8 tokens-per-param / 单 epoch，已在 data-rich 区；E 的物理含义是"这个数据分布 + 这个 tokenizer + 这个容量的可压极限"。**优化器/调度只能在 A 上做几个百分点，动不了 E。**
3. **【次要，可改】stable 段 LR 偏高 + decay 窗口偏短。** river-valley（2410.05192）预测 stable 段损失"虚高"、靠 decay 收割；我们 decay 只有 10%，且 **LR 探针只测了 ≥8e-4 的一侧**（4B ckpt 的 4 臂是 8e-4/1.2e-3/1.8e-3/2.6e-3），**下侧未测**。S1 又显示"可承受 LR 随训练变长而下降"（a_raw=−0.89），所以 8e-4 在 8B 末期**很可能偏高**。这一项的量级是 0.005–0.02 bpb，是配方侧唯一还没排除的大项。
4. **【已排除】** 数据重复（0.934 epoch）、训练不稳定（gn 0.09–0.31、无 spike）、EMA（单独计量 −0.0042）、batch（S3 已判 2^21 最优）。

**最值得试的 3 个改动（按 ROI 排序）**

| 序 | 改动 | 为什么 | 预期（估） | 成本 |
|---|---|---|---|---|
| 1 | **MuonH（Hyperball）** + LR 重标定 | 2026 年唯一的"结构性"改进；≤1.2B 有 20–30% token 等效；改善 LR 迁移；Puro 在 5090 上用 | **0.005–0.015** | 1 人日 + ¥28 |
| 2 | **decay 0.1→0.25–0.3，形状 1−sqrt，min-lr→绝对 ~2e-5** | 两个独立 2026 来源都指向更长 decay；min-lr 比例法是已知次优 | **0.004–0.012** | 0.5 人日 + ¥35 |
| 3 | **cooldown 期提高 Muon momentum（0.95→0.98）** | cooldown 论文实测"更高 β2 稳定有益"，零成本 | **0.001–0.004** | 0 + ¥0 |

**不建议作为第一顺位**：LR re-warmup（无证据、高风险）、换 SOAP/Shampoo（工时不合算）、fp8（已测负收益）、加 logit softcap（小模型风险）。

### (b) 只花 ¥20–60，在 0.3–1B 锚点上能验证什么

**能验证（排序 / 相对差）**
- decay 比例与形状的**排序**（0.1 vs 0.3）；
- min-lr 绝对值 vs 比例的**排序**；
- MuonH vs Muon 的**排序**（需先粗扫 LR）；
- "晚期 LR 是否偏高"的**方向**（下侧探针）；
- 短跑上的 **token 等效加速**（用同 token 预算下的 Δbpb 折算）。

**不能验证（必须诚实）**
- 绝对 bpb 水平（我们自己的教训：0.2–0.6B 的 β=0.154 外推 8B 错了 0.177）；
- 地板 E 的位置（5 点定不住；需要跨尺寸联合拟合或 16B 锚点）；
- fp8 / QAT 的长期数值稳定性（短跑看不出末期漂移）；
- 数据课程的收益（2511.18903 明确说它依赖 decay 设置，短跑会失真）。

**实验设计与判据（¥20–60 套餐，合计 ≈¥42）**

| ID | 设计 | 成本 | 判据（采纳/否决） | 否决后的动作 |
|---|---|---|---|---|
| **E1** | 从 **6B mark**（decay 前）ckpt 出发，恒定 LR ∈ {3e-4, 5e-4, 8e-4}，各 0.15B tokens，同数据前缀，末值 3M panel。**要求 ckpt 含优化器状态**（`--save-optimizer`），否则丢弃前 ~50 步的瞬态 | 0.45B ≈ ¥6.5 | 若 3e-4 或 5e-4 比 8e-4 好 **≥0.005 bpb** → 判"晚期 LR 偏高"，下一轮改全程线性/更早 decay；若 8e-4 仍最好 → 判"LR 不是杠杆"，把预算转到 MuonH | 把 E2 的 decay 只做到 0.2，省下的钱给 MuonH |
| **E2** | 0.6B tokens，4 臂 2×2：`decay{0.1,0.3}` × `min-lr{0.1×peak, 绝对 2e-5}`，形状统一 **1−sqrt**，同 seed/同数据前缀，末值 eval | 2.4B ≈ ¥35 | 若 decay=0.3 两臂都赢 **≥0.008**，或 min-lr 绝对值的两臂都赢 **≥0.008** → 采纳；若只有一臂赢 → 单独采纳该项；若全在 ±0.005 内 → 判"调度已饱和" | 全部不采纳，8B 直接上 MuonH + 现调度 |
| **E3**（选做） | 现有 8B ckpt 的**末段 ckpt 平均**（若 ckpt 粒度 ≤0.5B）：等权/三角权 vs EMA vs raw，三次 eval | ¥2 | 比 EMA 再好 **≥0.002** 才进 8B | 只用 EMA |

> E1 的关键价值：它是**唯一能在 ¥10 内回答"8B 那 ¥110 是不是白花了"的实验**。它复用了已存在的 6B ckpt，不需要重训。

**若预算能到 ¥60**：把 E2 的 4 臂减为 3 臂（去掉 min-lr=0.1× 那条，因为它是最可能被否的），省下的 ¥9 加一条 **E4：MuonH 在 0.3B 的 3 臂 LR 粗扫**（0.75B ≈ ¥11），为下一轮 8B 做准备。

### (c) 哪些配方改动在 8B 规模会显著增加风险

| 改动 | 风险机制 | 证据 | 结论 |
|---|---|---|---|
| **LR re-warmup（中期抬回 peak）** | 无新数据时没有收益证据；抬 LR 会打破单调下降、可能触发 loss spike；一旦失败整轮 ¥110 报废，且**无法从曲线早期判断**（spike 常在末期） | 2403.08763、2308.04014 都只在**分布漂移**场景有效；2309.14322 显示高 LR 不稳定在小模型可复现但**需要调 LR 才显现** | **禁止**作为 8B 改动；最多在 0.3B 上做机制研究 |
| **fp8 训练（body/experts）** | 数值稳定性 + 超参敏感性 + 我们实测负吞吐；失败时难以归因（是精度还是 kernel） | 2405.18710；2411.04330；我们 `fp8-first-plan.md` d12 NO-GO | **不做**；只保留 head fp8 与 Engram 表 fp8 存储 |
| **decay 比例一步到 0.8–1.0** | Puro 的 0.8–1.0 是在**高 effective LR**（~1e-2）下成立；我们 LR 若偏低，40–90% token 会在低 LR 区空转 | 2608.27370（其 effective LR 与我们的 8e-4 不可比） | **不做**；先 0.25–0.3 |
| **logit soft-capping** | 改变 logit 动态范围，与 fp8 head 交互；小模型可能压制置信度导致 bpb 变差 | 2408.00118（2B–27B）；社区移除实践**未证实** | **不做**；若要稳，用 OEC |
| **换 SOAP / Shampoo / Scion** | 每步特征分解在 24 层小模型上固定开销占比更高；3–5 人日；收益可能与 Muon 重叠 | 2409.11321（360M/660M，大 batch 区） | **不做**（除非 Hyperball 失败） |
| **切 muP 参数化** | 需重标定 init/LR/所有 group，3 天+；且 2510.19093 说 muP≈隐式 warmup | 2510.19093 | **不做**；用 WD/warmup 替代 |
| **加大 tokens/step 到 2^22** | 我们 S3 已证伪（+0.138 bpb）；文献侧 CBS 随数据量增长只是"可能允许"，不是"更优" | 我们 S3；2410.21676 | **不做** |
| **QAT / 1.58-bit 进预训练** | 精度成本 + 与 fp8 head 叠加；MiniCPM4 也只在末段做 | 2506.07900、2402.17764 | **不做**（留后训练） |
| **单改数据顺序（不加长 decay）** | 2511.18903 明确：标准 decay 会吃掉课程收益 → 白干 | 2511.18903 | **不做**；要改就打包 |

---

## 5. 推荐实验（预算 / 判据 / 风险）

> 成本按 62.5k tok/s、¥3.25/h（1B tokens ≈ 4.44h ≈ ¥14.4）估；eval 3M panel ≈ 116s ≈ ¥0.1。

| ID | 内容 | 工时 | GPU ¥ | 判据 | 预期收益 | 风险 |
|---|---|---|---|---|---|---|
| **E1** | 6B ckpt 的**晚期 LR 下侧探针**（3e-4/5e-4/8e-4 × 0.15B） | 0.2d | **6.5** | 见 §4(b) | 决定 E2 走向；可能锁定 0.005–0.02 | 低（只是诊断） |
| **E2** | 0.6B **decay×min-lr 2×2**（1−sqrt） | 0.5d | **35** | 双赢 ≥0.008 采纳 | 0.004–0.012 | 低（有界） |
| **E3** | 8B 末段 **ckpt 平均 vs EMA** | 0.2d | **2** | ≥0.002 才采纳 | 0.000–0.004 | 极低 |
| **E4** | **MuonH 实现 + 0.3B LR 粗扫 + 0.6B A/B** | 1d | **28** | 0.6B 上 ≥0.008 才进 8B | 0.005–0.015 | 中（需重标定 LR；若 LR 扫错会误判） |
| **E5** | **带地板的跨尺寸联合拟合**（复用 0.2/0.3/0.4/8B 全部 run） | 0.5d | **0** | 给出 E 的置信区间 + 下轮 8B 的预测区间 | 0（避免 ¥110 误判） | 无 |

**建议顺序**：**E5（免费）→ E1（¥6.5）→ E2（¥35）→ E3（¥2）→ E4（¥28）**。
- 若只做一件事：**E1**（最便宜、信息量最大）。
- 若只做两件事：**E1 + E2 = ¥42**，正好落在 ¥20–60 区间。
- E4 建议单独立项（¥28 + 1 人日），因为它同时是"下一轮 8B 的必要前置"。

**E4 的细节（因为它最复杂）**
1. 在 `nanov41/muon.py` 的 step 之后加：`R` 缓存（首次 step 时记 `||W||_F`）+ `W ← R·normalize(W − η·R·normalize(Δ))`；shape-grouped 的 3-D 专家权重按最后一维切片逐个处理；
2. 0.3B 上 3 臂 LR（相对现 LR 的 ×0.3/×1/×3）粗扫，选 bpb 最低者；
3. 0.6B A/B：Muon(8e-4) vs MuonH(扫出的 η)，同 seed/同前缀/跑满 decay；
4. 判据：Δbpb ≥ 0.008 且 gn 轨迹无异常 → 进 8B。

---

## 6. 风险与"别做"清单（汇总）

**别做（有明确证据反对）**
1. **LR re-warmup / 中期抬 LR**（无新数据时无收益证据，末期 spike 风险，¥110 暴露）。
2. **fp8 训练 body/experts**（我们实测负吞吐；2024 文献说稳定性更差；Puro 的成功依赖其自有 kernel 栈）。
3. **切 muP**（3 天+，且 2510.19093 说其作用≈隐式 warmup；我们形状固定）。
4. **加大 tokens/step 到 2^22**（我们 S3 已证伪）。
5. **Lion / Shampoo / SOAP 替换 Muon**（工时 vs 收益不合算；矩阵类加速随规模缩到 1.1×）。
6. **logit soft-capping**（小模型风险 + 与 fp8 head 交互不明）。
7. **单改数据顺序**（必须与"更长 decay + 模型平均"打包，否则收益被 decay 吃掉）。
8. **QAT / 1.58-bit 进预训练**（后训练阶段的事）。
9. **Schedule-Free / WSqD 替换 WSD**（我们的 27.8 TPP 正好在"无优势"区间）。

**谨慎做（要先在 0.3–0.6B 验证排序）**
1. decay 比例一步跨到 0.5 以上（Puro 的最优区间依赖其高 effective LR）。
2. MuonH（必须重标定 LR；否则会把"LR 没调好"误判成"方法不行"）。
3. cooldown 期提高 momentum/β2（便宜，但要确认不与 NS 迭代交互出振铃）。
4. 数据课程（需要质量分；且必须打包 decay 改动）。

**可以放心做（零/低成本、方向明确）**
1. min-lr 从 0.1×peak 改成绝对小值（或 0.02–0.03×）。
2. decay 形状改 1−sqrt。
3. 末段 ckpt 平均（与 EMA 并存）。
4. 带地板的跨尺寸联合拟合（纯分析）。
5. 输出层算子范数日志（norm transfer 诊断）。

---

## 7. 参考文献

**标注**：`[经典/高引用]` = 2022–2024 高引用基础工作；`[2025]` / `[2026 新]` = 时间标签；`[未证实]` = 本报告未核正文数字。

### 7.1 学习率调度
1. Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv:2404.06395, 2024. `[经典/高引用]` https://arxiv.org/abs/2404.06395
2. Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*（WSD 分支法），arXiv:2405.18392, 2024. `[高引用]` https://arxiv.org/abs/2405.18392
3. Wen, Li, Wang, Hall, Liang, Ma, *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective*, arXiv:2410.05192, 2024. `[高引用]` https://arxiv.org/abs/2410.05192
4. Dremov, Hägele, Kosson, Jaggi, *Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler*, arXiv:2508.01483, TMLR 2025. `[2025]` https://arxiv.org/abs/2508.01483
5. Wang et al., *Scaling and Transferability of Annealing Strategies in Large Language Model Training*, arXiv:2512.13705, AAAI 2026. `[2025-12]` https://arxiv.org/abs/2512.13705
6. *Universal Dynamics of Warmup Stable Decay: understanding WSD beyond Transformers*, arXiv:2601.09000, 2026. `[2026 新]` https://arxiv.org/abs/2601.09000
7. Meterez et al., *Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging*, arXiv:2602.03702, ICML 2026. `[2026 新]` https://arxiv.org/abs/2602.03702
8. *ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models*, arXiv:2605.19095, 2026. `[2026 新]` https://arxiv.org/abs/2605.19095
9. *WSqD: A Horizon-Free Learning Rate Schedule for Large Model Training*, arXiv:2607.10959, 2026. `[2026 新，数值未证实]` https://arxiv.org/abs/2607.10959
10. *Optimal Learning Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay*, arXiv:2602.06797, 2026. `[2026 新]` https://arxiv.org/abs/2602.06797
11. *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules*, arXiv:2503.12811, 2025. `[2025]` https://arxiv.org/abs/2503.12811
12. *Taming Transformer Without Using Learning Rate Warmup*, arXiv:2505.21910, 2025. `[2025]` https://arxiv.org/abs/2505.21910

### 7.2 超参规律与迁移
13. Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*（muP），arXiv:2203.03466, 2022. `[经典/高引用]` https://arxiv.org/abs/2203.03466
14. Kosson, Welborn, Liu, Jaggi, Chen, *Weight Decay may matter more than muP for Learning Rate Transfer in Practice*, arXiv:2510.19093, ICLR 2026. `[2025-10 / 2026]` https://arxiv.org/abs/2510.19093
15. Filatov et al., *Optimal Scaling Needs Optimal Norm*, arXiv:2510.03871, 2025（v2 2026-01）. `[2025]` https://arxiv.org/abs/2510.03871
16. Li et al., *Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in LLM Pretraining (Step Law)*, arXiv:2503.04715, 2025. `[2025]` https://arxiv.org/abs/2503.04715
17. *How Does Critical Batch Size Scale in Pre-training?*, arXiv:2410.21676, ICLR 2025. `[2024/2025]` https://arxiv.org/abs/2410.21676
18. *Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training*, arXiv:2505.23971, 2025. `[2025]` https://arxiv.org/abs/2505.23971
19. *MiniCPM4: Ultra-Efficient LLMs on End Devices*（ModelTunnel v2、BitCPM4），arXiv:2506.07900, 2025. `[2025]` https://arxiv.org/abs/2506.07900
20. *Small-scale proxies for large-scale Transformer training instabilities*, arXiv:2309.14322, 2023. `[经典/高引用]` https://arxiv.org/abs/2309.14322

### 7.3 优化器
21. Liu et al., *Muon is Scalable for LLM Training*（Moonlight），arXiv:2502.16982, 2025. `[高引用]` https://arxiv.org/abs/2502.16982
22. *Fantastic Pretraining Optimizers and Where to Find Them*, arXiv:2509.02046, 2025. `[2025]` https://arxiv.org/abs/2509.02046
23. *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization*, arXiv:2606.16899, 2026. `[2026 新]` https://arxiv.org/abs/2606.16899
24. Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, arXiv:2409.11321, ICLR 2025. `[2024/2025]` https://arxiv.org/abs/2409.11321
25. Chen et al., *Symbolic Discovery of Optimization Algorithms*（Lion），arXiv:2302.06675, 2023. `[经典/高引用]` https://arxiv.org/abs/2302.06675
26. *When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark*, arXiv:2610.21555, NeurIPS 2026 Workshop. `[2026 新]` https://arxiv.org/abs/2610.21555
27. *Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap*, arXiv:2611.03633, 2026. `[2026 新]` https://arxiv.org/abs/2611.03633
28. *The AdEMAMix Optimizer: Better, Faster, Older*, arXiv:2409.03137, 2024. `[2024]` https://arxiv.org/abs/2409.03137
29. *MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training*, arXiv:2605.26842, 2026. `[2026 新]` https://arxiv.org/abs/2605.26842
30. *Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models*（PACE），arXiv:2606.25086, 2026. `[2026 新]` https://arxiv.org/abs/2606.25086
31. *WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training*, arXiv:2507.17634, 2025. `[2025]` https://arxiv.org/abs/2507.17634
32. Pethick et al., *Training Deep Learning Models with Norm-Constrained LMOs*（Scion / Disco），arXiv:2502.07529, 2025. `[2025]` https://arxiv.org/abs/2502.07529

### 7.4 稳定性与精度
33. Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*（z-loss 1e-4），arXiv:2204.02311, 2022. `[经典/高引用]` https://arxiv.org/abs/2204.02311
34. *Output Embedding Centering for Stable LLM Pretraining*（OEC / μ-loss），arXiv:2601.02031, 2026. `[2026 新]` https://arxiv.org/abs/2601.02031
35. *Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers*, arXiv:2609.16179, 2026. `[2026 新]` https://arxiv.org/abs/2609.16179
36. Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size*（logit soft-capping），arXiv:2408.00118, 2024. `[高引用]` https://arxiv.org/abs/2408.00118
37. *AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping*, arXiv:2502.11034, 2025. `[2025]` https://arxiv.org/abs/2502.11034
38. *EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training*, arXiv:2610.37062, 2026. `[2026 新]` https://arxiv.org/abs/2610.37062
39. *To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability*, arXiv:2405.18710, 2024. `[2024]` https://arxiv.org/abs/2405.18710
40. Kumar et al., *Scaling Laws for Precision*, arXiv:2411.04330, 2024（ICLR 2025）. `[2024]` https://arxiv.org/abs/2411.04330
41. Wang et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*（BitNet b1.58），arXiv:2402.17764, 2024. `[高引用]` https://arxiv.org/abs/2402.17764

### 7.5 数据顺序 / 课程 / 长跑
42. Luo et al., *How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining*, arXiv:2511.18903, 2025（v3 2026-05）. `[2025-11]` https://arxiv.org/abs/2511.18903
43. *Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning*, arXiv:2506.11300, 2025. `[2025]` https://arxiv.org/abs/2506.11300
44. Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264, 2023. `[经典/高引用]` https://arxiv.org/abs/2305.16264
45. Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, arXiv:2403.08763, 2024. `[2024]` https://arxiv.org/abs/2403.08763
46. *Continual Pre-Training of Large Language Models: How to (re)warm your model?*, arXiv:2308.04014, 2023. `[2023]` https://arxiv.org/abs/2308.04014
47. Luo et al., *Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*, arXiv:2608.27370, 2026. `[2026 新，同硬件]` https://arxiv.org/abs/2608.27370

### 7.6 背景（经典 scaling）
48. Hoffmann et al., *Training Compute-Optimal Large Language Models*（Chinchilla），arXiv:2203.15556, 2022. `[经典/高引用]` https://arxiv.org/abs/2203.15556
49. Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv:2001.08361, 2020. `[经典/高引用]` https://arxiv.org/abs/2001.08361

---

### 附：本报告未做/未核实的事
- 未逐一核对 2026 年论文的**图表级数字**：Puro 的 decay 扫描与"TPP/peak 趋势"来自正文 §3.3.2 与附录 F.2 的文字描述（图 6/17/18 未看）；WSqD 的 SlimPajama 结果只读了摘要；ScheduleFree+ 的 31% 已核对为"同等损失省 31% 训练时间"。标 `未证实` 处请勿直接引用为结论。
- 未穷举 2026 年的全部调度/优化器新工作（本报告覆盖的是被引/复现度较高、且与本项目规模相关的一批）。
- 本报告的拟合（§1.3、§1.2）是**我们用自己 5 个 mark 的二次分析**，不是文献数字；建议用 E5 的跨尺寸联合拟合替换它，再用于任何预算决策。
