# 架构侧小模型训练方法调研（2025–2026）

> 对象：单卡 RTX 5090 上从零训练 288M active / 8B tokens 的小语言模型。
> 范围：只讨论**架构侧**（残差/深度、形状、注意力、稀疏记忆、tokenizer），不重复数据/KD/超参。
> 引用数取自 Semantic Scholar（2026-11-18），只作“热度”参考；arXiv 编号均已核对。
> 结论时间：2026-11-18。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| **架构还值不值得做？** | 值得，但空间只有 **0.005–0.015 bpb**。8B/288M 下公开方法的相对 loss 增益多在 0.5–1.5%，按 bpb=1.066 换算约 0.005–0.012；β=0.06–0.08 意味着 2× tokens 才 −0.04 bpb。不要期待架构把 1.06 拉到 0.95。 |
| **(a) 最值得实测的 2–3 个** | **E1：shape×mHC 2×2**（24×1024 vs 32×896，mHC on/off，2B 锚点，约 ¥110，1–2 人日）；**E2：Block AttnRes 3 臂**（mHC / AttnRes / 两者叠加，1B，约 ¥40 + 3–5 人日）；**E3：MUDDFormer-lite**（块级 dense connections，1B，约 ¥35 + 5–8 人日，stretch）。 |
| **(b) mHC/HC 在 depth 24 是否高估？** | **是，若期待 >0.01 bpb 就高估了。** 公开证据的相对增益稳定在 ~0.85–1.2%（1B/7B/27B/296M），8B 下绝对收益大概率 **0.004–0.008 bpb**。保留 mHC 的理由是稳定性与已有 kernel，不是 bpb；**不要加 hc_mult>4**。 |
| **(c) deep-thin×mHC 是否值得测？** | **值得，但先验偏负。** 要做 **iso-param 2×2**：24×1024 vs **32×896**（不是 32×768）；32×768 少 25% 参数/FLOPs，只能做 iso-FLOPs 扩展臂（10.7B tokens）。2B 锚点 ¥110；若 32L+mHC 胜 ≥0.008 bpb 再上 8B 确认（¥212）。 |
| **(d) 哪些风险过高？** | MoE、KDA/Mamba 混合、full DAR、full MUDDFormer、hc_mult>4、vocab 64K+、weight sharing、TEMPER/Post-LN 等 2026 未证实改动。原因见 §4.4。 |

---

## 1. 我们的坐标、换算与“什么算赢”

### 1.1 事实基线

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | 288M active；dim 1024 / 24L / dense SwiGLU h=2304；V=41,280；seq 8192；mHC hc_mult=4；Engram 220M（SparseRowAdam）；MLA 风格注意力 + GQA（window+compressed KV+indexer） | 项目配置 |
| 训练 | 8B tokens，35.5 h，约 ¥128；P1 实测 raw bpb **1.0662**，EMA **1.0620** | [r93-p1-final-report](../r93-p1-final-report.md) |
| 曲线 | 局部 β：1→2B 0.102、2→4B 0.090、4→6B 0.060、6→8B 0.078 | 同上 |
| 显存 | 峰值 reserved **27.78 GiB / 32 GB**；b2×8192 OOM（差 0.3–1 GiB）；余量 **<4 GiB** | [r93-p0a-p0b-results](../r93-p0a-p0b-results.md) |
| 成本 | 62.5k tok/s、¥2.98/h：1B=4.5 h≈¥13；2B=9 h≈¥27；8B=35.5 h≈¥106；3M panel eval≈0.3 h≈¥1 | 由 P1 实测换算 |

### 1.2 单位换算（很重要）

论文里的 loss 多是 **nats/token**，我们报的是 **bits/byte**：

- 1 nat/token = 1.4427 bit/token；
- bpb = bit/token ÷ 4.4426（我们的 tokenizer 实测 bytes/token）；
- **0.01 nat/token ≈ 0.00325 bpb**；0.02 nat ≈ 0.0065 bpb；0.03 nat ≈ 0.0097 bpb；
- 反过来，bpb 1.066 上的 **1% 相对 loss 改善 ≈ 0.0107 bpb**。

所以看到论文说“loss 降 0.02–0.03”不要兴奋：那是 **0.006–0.010 bpb**。看到“5% loss 降”才是 0.05 bpb 量级，但那种结果通常出现在 0.1–0.5B/2–10B 的小规模，迁移到 8B 会缩水。

### 1.3 噪声与“赢”的阈值

| 锚点 | 3M panel 的近似噪声（1σ） | 可判定阈值 |
|---|---:|---:|
| 0.6B | 0.004–0.006 bpb | ≥0.008 |
| 1B | 0.004–0.006 | ≥0.008 |
| 2B | 0.003–0.005 | ≥0.008 |
| 8B | 0.002–0.004 | ≥0.005 |

建议：**<0.004 视为噪声；0.005–0.010 是有意义但脆弱；>0.015 才是强结论。** 单 seed 的 1B 锚点不足以判定 0.005 级差异；接近的臂要跑 2 seeds，或直接上 2B。

---

## 2. 概览表

### 2.1 残差/深度方向

| 方法（年份，引用） | 做法一句话 | 证据（规模/token/收益） | 我们的可借鉴性 | 改造成本（人日 + GPU·h/¥） | 预期 bpb / 能力 | 结论 |
|---|---|---|---|---|---|---|
| **HC**（2024，113 引，经典） | 把残差流扩成 n 条，学习 pre/post/res 三组混合矩阵 | OLMo-1B/7B + OLMoE-1.3B，500B tokens；1B V2 loss 2.811→2.781（−0.030），7B 2.581→2.559（−0.022）；早期收敛 1.8×，500B 终点增益仅 ~1% | 已被 mHC 取代；只作为对照/理解 | 0（已有 mHC）；关掉 mHC 需 1–2 人日 | 0.004–0.010 / +0.5–1.0 分 | 不单独做 |
| **mHC**（2025-12，113 引，2025 末新/高关注） | 把 HC 的混合矩阵投影到双随机流形（Sinkhorn），恢复 identity mapping，配 kernel 融合 | 3B/9B/27B；27B 262B tokens，3B 1.05T tokens；27B loss −0.021，8 项算术平均 53.1→57.6（HC 56.7）；HC 在 12k step 附近尖峰，mHC 稳定 | **已在生产**；depth 24 下增益大概率很小；不要加 stream | A/B 1–2 人日；1B×2 臂 9 GPU·h≈¥27 | 0.004–0.008 / +0.3–0.8 分 | 保留，不再投入 |
| **AttnRes**（2026-03，88 引，2026 新/高关注） | 用 softmax 对前面所有层/块输出做内容相关的深度聚合，替代固定残差 | 194M–528M active，38.7–119B tokens；296M/62.1B：1.829→1.804（−0.025）；436M：1.766→1.737；Block 版在 5.6 PFLOP/s-days 等价 1.25× compute；Kimi Linear 48B/3B/1.4T 下游全胜 | **最高优先级的残差替代**；I/O 5.5d vs mHC 34d；Block 版内存可控 | 3–5 人日 + kernel 调试 10–20 GPU·h≈¥30–60；实验 3 臂×1B≈¥40 | 0.005–0.012 / +0.5–1.0 分 | **Top-2** |
| **DAR**（2026-07，1 引，2026 新/未证实） | 两条残差流 + 跨流深度检索 + 门控写回；块级变体降开销 | 0.1/0.3/0.5B（2/6/10B tokens）、1B/50B、7B MoE/31.5B；0.5B val loss 2.796→2.653（−5.1%），1B BPB 1.029→1.008（−0.021），7B 2.814→2.739 | 小规模数字惊人，但只有 1 篇/1 引，且要重写 mHC 路径 | 6–10 人日 + 20–30 GPU·h≈¥60–90；实验 ¥40–60 | 0.008–0.020（若迁移）/ +1–2 分 | Watch，不做 Top-3 |
| **DenseFormer**（2024，72 引，经典） | 每层后对当前+所有历史层做静态加权平均（DWA） | 最多 ~100B 参数级；同 ppl 下更省内存；但 AttnRes 在 436M 上复现其无增益（1.767 vs 1.766） | 静态权重不够；已被 AttnRes 证据否定 | 2–3 人日；¥20 | 0.000–0.003 | 不做 |
| **MUDDFormer**（2025，43 引，高引用） | 每层用隐藏状态动态生成到所有前层的多路连接权重（Q/K/V/res 分流） | 405M/24L/d1024/**7B tokens** 即 1.8× compute；834M/15B 1.89×；2.8B/300B 追平 6.9B；+0.23% 参数、+0.4% 计算；但激活内存 +29%、速度 81% | **证据与我们的形状/token 几乎重合**；但 O(L) 深度状态会顶爆 32GB | 5–8 人日 + 20–30 GPU·h≈¥60–90；实验 2 臂×1B≈¥35 | 0.008–0.015 / +0.5–1.5 分 | **Top-3（stretch）** |
| **Frac-Connections**（2025-03，9 引，新） | 把 hidden state 切成分片而非扩宽，保留部分 HC 收益、降 I/O | OLMoE-1.3B DFC×4 −0.014 loss；7B MoE/3T −0.012；下游 +0.35 平均 | 若 mHC 内存吃紧可作替代；但增益太小 | 3–5 人日；2 臂×1B≈¥27 | 0.002–0.006 / +0.2–0.5 分 | 不做 |

### 2.2 形状、注意力、稀疏/记忆、tokenizer

| 方法（年份，引用） | 做法一句话 | 证据（规模/token/收益） | 我们的可借鉴性 | 改造成本（人日 + GPU·h/¥） | 预期 bpb / 能力 | 结论 |
|---|---|---|---|---|---|---|
| **MobileLLM**（2024，376 引，经典） | 亚十亿用 deep-thin（30L×512 / 32L×896）+ embedding sharing + GQA + block-wise 权重共享 | 125M/350M，**1T tokens**；deep-thin 比 12L×768/15L×1280 平均 +0.9/+1.1 分；embedding share 省 10% 参数、−0.2/−0.6 分；GQA +0.4/+0.7；LS 再 +0.7/+0.8 | 方向可借鉴，但 1T tokens vs 我们 8B；32×896 才是 iso-param 点 | 1–2 人日；2×2×2B≈¥110 | −0.005–+0.010 / ±0.5 分 | 用 E1 直接验 |
| **Depth Delusion**（2026-01，1 引，2026 新/未证实） | 用 30 个架构拟合 depth/width 条件 scaling law：D*∝C^0.12、W*∝C^0.34、Dcrit≈2.43 ln W | 17M–7B；W=512 时 24L 比 16L 差 0.033 nats；1B 最优 24L/1792，7B 最优 32L/4096；7B 64L/2816 比 32L/4096 差 +0.119 nats | **对 deep-thin 是负面先验**；W=896→Dcrit≈16.5，32L 已 2× 临界 | 阅读/校准 0.5 人日；可作为 E1 的先验 | 作为 E1 的解释框架 | 与 MobileLLM 冲突，需实测 |
| **Kimi Linear / KDA**（2025-10，245 引，高引用） | 3:1 KDA（channel-wise gated delta rule）与 MLA 混合，NoPE 全局层 | 48B total/3B active，1.4T tokens；同配方胜 full MLA，1.16× compute；KV −75%，1M context 解码 6×；scaling 653M–1.7B active，38–128B tokens | 训练 bpb 收益小；我们的 window+compressed KV 已省 attention；换 KDA 要重写 kernel | 3–6 周；不建议 | 0.000–0.005 / 推理收益大、训练 bpb 小 | 不做 |
| **Mamba-2 / Mamba**（2024/2023，经典） | SSM/状态空间对偶，线性时间序列建模 | Mamba-2 130M–1.3B，300B tokens；130M 与 Mamba 打平（42.6 vs 42.4），1.3B 56.4 略优；混合 H3 未显著超 Transformer | 8B tokens 下数据效率不占优；无 kernel | 3–6 周 | 0.000–0.005 | 不做 |
| **Hymba**（2024，126 引，经典） | attention 头 + SSM 头并行混合，meta token，跨层 KV 共享 | 125M/350M/1.5B；1.5B 比 Llama-3.2-3B 高 1.32%，cache −11.67×，吞吐 3.49× | 主要是推理/长上下文；训练 bpb 证据弱 | 3–6 周 | 训练 bpb 不明确 | 不做 |
| **混合层消融**（2026-03，新） | 对 sub-1B 混合模型逐组件删除/干预 | Qwen3.5-0.8B、Falcon-H1-0.5B；删 attention 或线性/SSM 任一路都显著掉点；似然对线性路更敏感；重要性随深度变化 | 说明“全换线性”有风险；保留 attention 是对的 | 0（只读） | — | 支持“不换注意力” |
| **Engram**（2026-01，104 引，2026 新/高关注） | N-gram 哈希查表，O(1) 取回，作为 MoE 之外的静态记忆稀疏轴 | 27B/3.8B active/262B tokens；iso-param/iso-FLOPs 胜 MoE：MMLU +3.4、BBH +5.0、HumanEval +3.0；NIAH 84.2→97.0；0.56B active +1.6B 表/100B：1.808→1.768（−0.04）；第 2 层注入最好；稀疏预算的 20–25% 给记忆最优 | **已有 220M 表**；可做表大小消融；不要盲目加表 | 1 人日；64M vs 220M×0.6B≈¥16 | −0.003–+0.005 / +0.2–0.5 分 | 可选 E4 |
| **MoE 小模型**（OLMoE 2024 / MobileMoE 2026 / MiniCPM-MoE 产品） | 细粒度专家 + 共享专家，稀疏激活 | OLMoE 1.3B active/7B total/5T tokens；MobileMoE 0.3–0.9B active/1.3–5.3B total/6T tokens，约 2× token 效率；MiniCPM-MoE 缺同等级论文（未证实） | 单卡无 EP/all-to-all；8B tokens 远不够专家收敛；Engram 已给稀疏容量 | 2–4 周；不建议 | 8B 下大概率负 | 不做 |
| **Embedding/weight sharing**（MobileLLM-LS 2024；onóna 2026；Share Your Attention 2025） | 输入输出 embedding 共享；block-wise 层共享；字典学习共享 attention 权重 | MobileLLM：省 10% 参数、−0.2/−0.6 分，LS 再 +0.7/+0.8（1T tokens）；onóna 142M/330M/1B 降 val loss（2026 新/未证实）；Share Your Attention 减 66.7% attention 参数（2025 新） | 288M/8B 下容量更稀缺；省参数可能掉点 | 2–5 人日；2 臂×1B≈¥27 | −0.005–+0.002 | 不做 |
| **Vocab scaling**（2024，144 引，经典） | 词表大小也参与 scaling；N_v∝C^0.42，N_nv∝C^0.50 | 33M–3B，500B chars；302M/43B chars 数据匹配时最优 V≈16K，数据过多时 24K；3B/1.3e21 最优 37–43K；Llama2-70B 应 216K | 我们 V=41,280 比 302M 的 16–24K 高 1.7–2.6×；但 bpb 影响小、重分词贵 | 2–3 人日；2 臂×0.6B≈¥25 | −0.005–+0.005 | 不优先 |
| **Compute Optimal Tokenization**（2026-05，4 引，2026 新） | 压缩率 T（bytes/token）进入 scaling law；最优 T 随 compute 缓降 | 988 个 latent + 320 个 subword 模型，50M–6.7B；subword 最优 T 接近 BPE 4.57 并随 compute 缓降；低算力下可高于 BPE | 我们 4.4426 与 BPE 4.57 接近，已在合理区间 | 0 | — | 不动 tokenizer |

---

## 3. 逐方法

### 3.1 残差/深度：HC、mHC、AttnRes、DAR、DenseFormer、MUDDFormer、Frac

#### 3.1.1 Hyper-Connections（HC，[2409.19606](https://arxiv.org/abs/2409.19606)，2024，ICLR 2025，113 引）

- **做法**：把 residual stream 扩成 n 条（常用 n=4），每层学习 `pre`（层输入混合）、`post`（分支输出写回）、`res`（流间混合）三组矩阵；不增加单层 FLOPs，只增加 I/O。
- **证据**：OLMo-1B/7B + OLMoE-1B-7B，全部 500B tokens。1B 的 V2 eval loss 2.811→2.781（−0.030），V3 2.544→2.515；7B 2.581→2.559（−0.022），下游 70.1→71.0。论文强调“1.8× faster convergence”，但这是**早期**；500B 终点相对增益只有 ~0.85–1.1%。HC 在 7B 上反而抑制了 baseline 的 loss spike；但 mHC 论文在 27B 上发现无约束 HC 会在 12k step 附近失稳，说明 identity mapping 在更大规模/更深时被破坏。
- **我们的可借鉴性**：我们已经是 mHC，HC 只是“更不稳定的前身”；单独做 HC 没有意义。
- **成本**：0（已有 mHC）；若要做“HC vs mHC”对照，1–2 人日 + 2 臂×1B≈¥27。
- **预期**：bpb 0.004–0.010；能力 +0.5–1.0 分。
- **结论**：不单独做；HC 的价值是解释了为什么 mHC 要加流形约束。

#### 3.1.2 mHC（[2512.24880](https://arxiv.org/abs/2512.24880)，2025-12，113 引，DeepSeek）

- **做法**：把 HC 的 `res` 混合矩阵投影到双随机矩阵流形（Sinkhorn），恢复 identity mapping；配 kernel 融合、重算和 DualPipe 通信重叠。我们的生产实现是 `hc_mult=4`、Sinkhorn γ0.18/K=11/τ1e-3。
- **证据**：3B/9B/27B；27B 主实验 262B tokens，3B 39.3B、9B 105B，另有 3B×1.05T 的 token scaling。27B 相对 baseline 最终 loss −0.021；8 项算术平均 53.1→57.6（HC 56.7），BBH 比 HC +2.1、DROP +2.3。关键不是“涨多少”，而是 **HC 在 12k step 附近尖峰、mHC 不掉**。
- **我们的可借鉴性**：已经在用，且是 DeepSeek-V4.1-Flash 的生产路径；问题不是“要不要上”，而是“在我们这个深度/ token 预算下值多少”。
- **成本**：A/B 只需关掉 mHC 或把 hc_mult 设 1；1–2 人日 + 2 臂×1B≈¥27（2B 约 ¥54）。
- **预期**：bpb **0.004–0.008**；能力 +0.3–0.8 分。理由见 §4.2。
- **结论**：保留；不要加 hc_mult=6/8（HC 论文 n=8 相对 n=4 几乎无增益，I/O 线性涨）。

#### 3.1.3 Attention Residuals（AttnRes，[2603.15031](https://arxiv.org/abs/2603.15031)，2026-03，88 引，Kimi）

- **做法**：每层不再固定 `x + f(x)`，而是用一个零初始化的 pseudo-query 对所有前层/块输出做 softmax 聚合；Block 版把层分成约 8 块，只对块级表示做 attention，降低内存/通信。
- **证据**：这是目前**与我们最接近**的公开锚点。5 个规模、8192 context、38.7–119B tokens：
  - 194M/38.7B：1.931→1.909（Block）/1.899（Full）；
  - 241M/45.4B：1.895→1.875/1.874；
  - **296M/62.1B：1.829→1.809/1.804；同表 mHC-lite=1.807**；
  - 436M/87.9B：1.766→1.746/1.737；
  - 528M/119B：1.719→1.693/1.692。
  Block 版在 5.6 PFLOP/s-days 等价 1.25× compute；Kimi Linear 48B/3B/1.4T 上下游全胜。AttnRes 的 I/O 是 5.5d/layer，mHC(m=4) 是 34d/layer；它比 mHC 更“轻”。
- **我们的可借鉴性**：**高**。296M 锚点上 AttnRes 与 mHC 打平/略好（1.804 vs 1.807），436M 上明显更好（1.737 vs 1.747），且内存 I/O 更低。它既可能替代 mHC，也可能与 mHC 叠加（未验证）。
- **成本**：3–5 人日（Block AttnRes kernel + 与 MLA/mHC 的接口 + 零初始化 + 内存 profiling）+ 10–20 GPU·h 调试（¥30–60）；正式 3 臂×1B≈13.5 GPU·h≈¥40。
- **预期**：bpb **0.005–0.012**；能力 +0.5–1.0 分。若只做“替代 mHC”，上限取 0.008；若“叠加”有协同，可能到 0.012。
- **结论**：**Top-2**。先 12L/0.2B 原型（¥5）验证 kernel，再上 24L/1B 三臂。

#### 3.1.4 Dual Attention Residuals（DAR，[2607.18730](https://arxiv.org/abs/2607.18730)，2026-07，1 引，2026 新/未证实）

- **做法**：两条残差流；每条流用另一条流的归一化状态算深度权重，再对**自己**的历史取值；块级变体把历史压到块级，控制开销。本质是“多流 + 深度检索”的交叉。
- **证据**：0.1B/2B、0.3B/6B、0.5B/10B、1B/50B、7B MoE/31.5B。0.5B val loss 2.796→2.653（−5.1%，同规模 AttnRes 2.708、mHC-4 2.764）；1B BPB 1.029→1.008（−0.021，AttnRes 1.015）；7B 2.814→2.739。数字很吸引人，但**只有 1 篇、1 引**，且 0.5B 的 5% 与 1B 的 2% 差距说明规模/配方敏感。
- **我们的可借鉴性**：中高。它需要两条流 + 跨流 attention + 门控写回；我们已有 mHC 的 4 流，理论上可复用，但 DAR 的“cross-stream addressing”与 mHC 的 Sinkhorn 是两套逻辑，改造成本高。
- **成本**：6–10 人日 + 20–30 GPU·h 调试（¥60–90）；正式 2 臂×1B≈¥27–40。
- **预期**：若迁移，bpb 0.008–0.020；能力 +1–2 分；方差大。
- **结论**：**Watch**，不进 Top-3；等独立复现或至少 2 个 seed 的 0.5B/1B 结果。

#### 3.1.5 DenseFormer（[2402.02622](https://arxiv.org/abs/2402.02622)，2024，NeurIPS，72 引）

- **做法**：每层后做 Depth-Weighted Average，对当前+所有历史层用**静态**标量权重加权。
- **证据**：同 ppl 下更省内存/推理；但 AttnRes 在 436M 上的对照显示 DenseFormer 1.767 vs baseline 1.766，**无增益**；说明“静态、输入无关”的深度混合不够。
- **我们的可借鉴性**：低；可作为 AttnRes 的负对照。
- **成本**：2–3 人日 + 1 臂×1B≈¥13。
- **预期**：0.000–0.003 bpb。
- **结论**：不做。

#### 3.1.6 MUDDFormer（[2502.12170](https://arxiv.org/abs/2502.12170)，2025，ICML，43 引）

- **做法**：MUDD 连接根据每个 token 的隐藏状态，动态生成到所有前层的多路连接权重；Q/K/V/residual 分流，权重不是静态的。可插进任意 Transformer。
- **证据**：这是**证据最贴合我们**的一篇：405M/24L/d1024/7B tokens 就有 1.8× compute 等价；834M/15B 1.89×；2.8B/300B 追平 6.9B（2.46× compute），5-shot 甚至接近 12B。只加 0.23% 参数、0.4% 计算；但激活内存 +29%、训练速度 81%。
- **我们的可借鉴性**：高，但有一个硬约束：我们峰值 27.78 GiB，余量 <4 GiB；论文 +29% 激活内存会顶到 32 GiB，**full 版大概率 OOM**。只能做块级/截断版（例如每 4 层一组、只连组内/最近 8 层），或配 activation checkpointing。
- **成本**：5–8 人日（动态连接生成器 + 块级 mask + 重算）+ 20–30 GPU·h 调试（¥60–90）；正式 2 臂×1B≈11 GPU·h≈¥35。
- **预期**：full 版 0.008–0.015 bpb；lite 版可能只有 0.004–0.010；能力 +0.5–1.5 分。
- **结论**：**Top-3（stretch）**。如果工程只能押一个“高上限”改动，选它；如果只能押一个“低风险”，选 AttnRes。

#### 3.1.7 Frac-Connections（[2503.14125](https://arxiv.org/abs/2503.14125)，2025-03，9 引）

- **做法**：不扩宽 residual stream，而是把 hidden state 切成分片，在分片间做 HC 式混合；保留部分 HC 收益、降低 I/O。
- **证据**：OLMoE-1.3B DFC×4 训练 loss −0.014；OLMoE-7B/3T −0.012，下游 +0.35 平均；DFC×2 已接近 DFC×4，说明收益饱和快。
- **我们的可借鉴性**：如果 mHC 的 4 流 I/O 成为瓶颈，Frac 是“降内存”的替代；但增益太小，且我们已经是 mHC。
- **成本**：3–5 人日 + 2 臂×1B≈¥27。
- **预期**：0.002–0.006 bpb。
- **结论**：不做；除非 mHC 显存/带宽成为硬瓶颈。

### 3.2 小模型形状：MobileLLM、宽度-深度 scaling、激活内存/深度上限

#### 3.2.1 MobileLLM（[2402.14905](https://arxiv.org/abs/2402.14905)，2024，ICML，376 引）

- **做法**：亚十亿参数下用 deep-thin（125M：30L×512；350M：32L×896），配 embedding sharing、GQA、SwiGLU；LS 变体做 block-wise 权重共享。
- **证据**：125M/350M，**1T tokens**。350M 从 15L×1280 换到 32L×896，平均 +1.1 分；125M 从 12L×768 换到 30L×512，+0.9 分。embedding sharing 省 ~10% 参数、−0.2/−0.6 分；GQA +0.4/+0.7；LS 再 +0.7/+0.8。注意：这是 1T tokens 的结论，不是 8B tokens。
- **我们的可借鉴性**：方向可借鉴，但两个坑：
  1. **32×768 不是 iso-param**。按我们的 SwiGLU 形状，32×768 只有 24×1024 的 ~75% 参数/FLOPs；要 iso-param 应取 **32×896**（h≈2016），参数只差 ~2%。
  2. 我们的 24L×1024 已经是 GPT-2-medium 形状，宽高比 42.7；MobileLLM 350M 的 32×896 宽高比 28，确实更“深窄”。但 8B tokens 下 deep-thin 的优化优势可能还没出现。
- **成本**：1–2 人日（配置/宽度适配）+ 2×2×2B≈¥110。
- **预期**：−0.005–+0.010 bpb；能力 ±0.5 分。
- **结论**：用 E1 直接验，不要凭 MobileLLM 直接改。

#### 3.2.2 宽度-深度 scaling（[2601.20994](https://arxiv.org/abs/2601.20994)，2026-01，1 引，2026 新/未证实）

- **做法**：30 个架构、17M–7B 拟合 architecture-conditioned scaling law；最优 depth D*∝C^0.12，width W*∝C^0.34；并给出 critical depth `Dcrit ≈ 2.43 ln W`，超过后加层反而变差。
- **证据**：W=512 时 16L 3.435、24L 3.468（+0.033）、32L 3.441；1B 最优 24L/1792，over-deep 80L/1024 +0.16；3B 最优 40L/2432，72L/1792 +0.16；7B 最优 32L/4096，64L/2816 差 +0.119 nats。
- **对我们的含义**：W=1024 → Dcrit≈16.9；W=896 → 16.5；W=768 → 16.1。**24L 已经超过 Dcrit，32L 更超**。但注意：这篇只有 1 引、与 MobileLLM 的 1T-token 结论冲突；Dcrit 是“拟合的 U 形拐点”，不是硬 cliff。它给 deep-thin 的是**负面先验**，不是死刑。
- **成本**：0.5 人日阅读/校准；作为 E1 的解释框架。
- **结论**：把“32L 可能更差”作为 E1 的零假设；如果 32L+mHC 仍胜，说明 mHC 确实改变了深度收益，这本身就是有价值的结论。

#### 3.2.3 激活内存/深度上限（我们的实测）

- 24L/d1024/seq8192/b1 峰值 27.78 GiB；b2 OOM 差 0.3–1 GiB。
- 32L×896 的 `L×d` 是 28,672 vs 24,576（+16.7%）；按激活池 14.5 GiB 估算，+2.4 GiB，**会顶到 ~30 GiB，需实测**。32L×768 的 `L×d` 与 24×1024 相同，显存更友好，但参数少 25%。
- 结论：E1 的 32L 臂要么用 896 + activation checkpointing（慢 20–30%），要么用 768 + iso-FLOPs 补 token；不要把“32×768 @8B”与“24×1024 @8B”直接比。

### 3.3 高效注意力/混合架构：Kimi Linear KDA、Mamba/SSM、亚十亿混合

#### 3.3.1 Kimi Linear / KDA（[2510.26692](https://arxiv.org/abs/2510.26692)，2025-10，245 引）

- **做法**：KDA 是 Gated DeltaNet 的 channel-wise 细粒度门控版本；Kimi Linear 用 3:1 的 KDA:MLA 混合，MLA 层用 NoPE。
- **证据**：48B total/3B active，1.4T tokens；同配方胜 full MLA，1.16× compute；KV cache −75%，1M context 解码 6×；scaling 实验 653M–1.7B active、38.8–128B tokens。**没有 288M/8B 的证据**。
- **我们的可借鉴性**：低。理由：
  1. 我们的 attention 已经是 window+compressed KV+indexer，训练时 attention 不是主要瓶颈；
  2. KDA 需要 chunkwise kernel、DPLR 变体、与 MLA 混合的层间调度；3–6 周工程量；
  3. 1.16× compute 主要在长上下文/大 token 预算；8B tokens 下线性注意力的样本效率优势没有证据。
- **成本**：3–6 周 + 大量 kernel 调试；不建议。
- **预期**：训练 bpb 0.000–0.005；推理 KV −75%、1M 解码 6×（但这不是当前目标）。
- **结论**：不做。

#### 3.3.2 Mamba-2 / Mamba（[2405.21060](https://arxiv.org/abs/2405.21060)，[2312.00752](https://arxiv.org/abs/2312.00752)，经典）

- **做法**：选择性状态空间/SSD，线性时间；Mamba-2 的 core layer 比 Mamba 快 2–8×。
- **证据**：Mamba-2 125M–1.3B，300B tokens；130M 与 Mamba 打平（42.6 vs 42.4），370M 49.0 vs 48.7，1.3B 56.4 vs GPT-Neo 49.7；混合 H3 没有显著超过 Transformer。也就是说，SSM 在亚十亿是“可比”，不是“更好”，且需要 300B tokens 才看得出来。
- **我们的可借鉴性**：低；8B tokens 下数据效率不占优，且没有 production kernel。
- **成本**：3–6 周。
- **预期**：0.000–0.005 bpb。
- **结论**：不做。

#### 3.3.3 Hymba（[2411.13676](https://arxiv.org/abs/2411.13676)，2024，ICLR 2025，126 引）与混合层消融（[2603.22473](https://arxiv.org/abs/2603.22473)，2026-03）

- **Hymba 做法**：attention 头 + SSM 头并行，meta token，跨层 KV 共享，partial sliding window。
- **证据**：125M/350M/1.5B；1.5B 比 Llama-3.2-3B 高 1.32%，cache −11.67×，吞吐 3.49×。主要是推理/长上下文指标。
- **混合层消融**：Qwen3.5-0.8B、Falcon-H1-0.5B；删 attention 或线性/SSM 任一路都显著掉点；似然对线性路更敏感；重要性随深度变化。说明“全换线性”有风险，保留 attention 是对的。
- **我们的可借鉴性**：低；训练 bpb 收益不明确，且换注意力要重写 kernel。
- **结论**：不做；把“保留 attention”作为已验证的设计选择。

### 3.4 稀疏/记忆/参数效率：MoE、Engram、embedding/weight sharing

#### 3.4.1 MoE 小模型（OLMoE [2409.02060](https://arxiv.org/abs/2409.02060)，2024；MobileMoE [2605.27358](https://arxiv.org/abs/2605.27358)，2026-05；MiniCPM-MoE 产品）

- **做法**：细粒度专家 + 共享专家 + top-k 路由；总参数远大于 active。
- **证据**：
  - OLMoE-1B-7B：1.3B active/7B total，5T tokens；
  - MobileMoE：0.3/0.5/0.9B active，1.3/2.8/5.3B total，~6T tokens；约 2× token 效率；MobileMoE-S（272M active）平均 46.5；
  - MiniCPM-MoE 是 2024 产品发布，缺少同等级技术报告（**未证实**）；MiniCPM 论文主体是 2.4B dense + 数据/训练策略，MiniCPM4 是 0.5B/8B、8T tokens + InfLLM v2 稀疏注意力。
- **我们的可借鉴性**：低。单卡 5090 没有 expert parallelism/all-to-all；8B tokens 远不够专家收敛；我们的 Engram 已经提供“稀疏容量”且不增加 per-token FLOPs。MoE 还会把 optimizer/memory 推高，32GB 不够。
- **成本**：2–4 周；不建议。
- **预期**：8B tokens 下大概率负；能力收益需要 100B+ tokens 才显现。
- **结论**：不做。

#### 3.4.2 Engram（[2601.07372](https://arxiv.org/abs/2601.07372)，2026-01，104 引，ACL 2026）

- **做法**：把 N-gram 哈希查表做成 O(1) 的静态记忆；与 MoE 组成“条件计算 + 条件记忆”两个稀疏轴；用 U 形 allocation law 分配稀疏预算。
- **证据**：27B/3.8B active/262B tokens，iso-param/iso-FLOPs 胜 MoE-27B：MMLU +3.4、CMMLU +4.0、BBH +5.0、ARC-C +3.7、HumanEval +3.0、MATH +2.4；NIAH 84.2→97.0。小规模控制：12L/3B MoE（0.56B active）+1.6B Engram/100B tokens，val loss 1.808→1.768（−0.04）；第 2 层注入最好；把同样 1.6B 拆成第 2、6 层两个小模块更好；稀疏预算的 20–25% 给记忆最优。
- **我们的可借鉴性**：**已有 220M 表**，占 active 的 76%，比论文 0.56B+1.6B 的“2.86× active”保守；所以“220M 是否过大”不能直接套 allocation law。可做的是**表大小消融**：64M vs 220M（vs 512M 如果显存允许），看边际收益。
- **成本**：1 人日（config/表 resize/SparseRowAdam）+ 2 臂×0.6B≈5.4 GPU·h≈¥16。
- **预期**：−0.003–+0.005 bpb；能力 +0.2–0.5 分。
- **结论**：可选 E4；不是 Top-3，但便宜、正交。

#### 3.4.3 Embedding sharing / weight sharing（MobileLLM-LS 2024；onóna [2611.10556](https://arxiv.org/abs/2611.10556)，2026-11；Share Your Attention [2508.04581](https://arxiv.org/abs/2508.04581)，2025-08）

- **做法**：输入/输出 embedding 共享；block-wise 层共享；字典学习共享 attention 权重。
- **证据**：MobileLLM：embedding sharing 省 ~10% 参数、−0.2/−0.6 分；LS 在 1T tokens 上再 +0.7/+0.8。onóna：twin embeddings + tied unembeddings，142M/330M/1B 降 val loss（2026-11，**极新/未证实**）。Share Your Attention：减 66.7% attention 参数、性能持平（2025-08，**新**）。
- **我们的可借鉴性**：低。我们 V=41,280、d=1024，embedding+head 约 84M（~29% 的 288M）；共享能省 42M，但 288M/8B 下容量更稀缺，MobileLLM 的收益是在 1T tokens 上、且用省下的参数加层。我们加层会撞显存/深度上限。
- **成本**：2–5 人日 + 2 臂×1B≈¥27。
- **预期**：−0.005–+0.002 bpb；大概率掉点。
- **结论**：不做。

### 3.5 Tokenizer/词表大小（[2407.13623](https://arxiv.org/abs/2407.13623)，2024，NeurIPS，144 引；[2605.01188](https://arxiv.org/abs/2605.01188)，2026-05，4 引）

- **2407.13623 做法/证据**：33M–3B、500B chars；词表参数 N_v∝C^0.42、非词表参数 N_nv∝C^0.50。302M 模型、43B chars、数据匹配时最优 V≈16K；数据瓶颈时降到 10K；数据过多（overtraining）时升到 24K。3B/1.3e21 最优 37–43K；Llama2-70B 应 216K。
- **2605.01188 做法/证据**：988 个 latent + 320 个 subword 模型，50M–6.7B；压缩率 T 进入 scaling law；subword 最优 T 接近 BPE 4.57，并随 compute 缓降；低算力下最优 T 可高于 BPE。
- **对我们的含义**：我们 V=41,280、4.4426 B/token，比 302M/43B chars 的 16–24K 高 1.7–2.6×；但 4.44 与 BPE 4.57 接近，2026 论文认为这是合理区间。bpb 是 bytes 归一化，换词表不会自动改善 bpb；重分词 8.5B tokens 要 2–3 人日 + CPU 成本。**不优先**。
- **成本**：2–3 人日 + 2 臂×0.6B≈¥25。
- **预期**：−0.005–+0.005 bpb；能力 ±0.3 分。
- **结论**：不做；如果要做，只做 24K vs 41K 的 0.6B 小锚点，且先确认 token_bytes 面板对齐。

### 3.6 其他 2026 热门改动（manifold-constrained、深度可塑、layer-wise）

| 工作 | 做法/证据 | 我们的判断 |
|---|---|---|
| **TEMPER**（[2608.07851](https://arxiv.org/abs/2608.07851)，2026-08，新/未证实） | 把 mHC 的 pre/res/post 生成器张量化，减少 stream 数增长带来的参数；号称保持双随机约束 | 与我们的 Sinkhorn 路径同源；但无大规模证据/无 kernel；**watch**，不实测 |
| **Delta Attention Residuals**（[2605.18855](https://arxiv.org/abs/2605.18855)，2026-05，新/未证实） | AttnRes 对累积 hidden state 做 attention 会路由坍缩；改为对“每层增量 delta”做 attention | 若 AttnRes 在 E2 成功，这是下一步；当前不直接上 |
| **Post-LayerNorm Is Back**（[2601.19895](https://arxiv.org/abs/2601.19895)，2026-01，新/未证实） | 重新审视 Post-LN；认为 Pre-LN 的残差路径导致深层梯度消失，Post-LN 可支持更深 | 与 mHC 的 identity mapping 目标重叠；换归一化会动 production kernel；**watch**，不在 32L 实验里混入 |
| **onóna**（[2611.10556](https://arxiv.org/abs/2611.10556)，2026-11，极新/未证实） | 两个 embedding 表取平均做输入，输出头混合两个 softmax；保持 tied unembedding | 与 embedding sharing 相关；只在 142M/330M/1B 有初步结果；**watch** |
| **Relaxed Recursive Transformers**（[2410.20672](https://arxiv.org/abs/2410.20672)，2024-10） | 层间参数共享 + 每层 LoRA，把已有 LLM 递归化 | 降参思路；我们 288M 容量更紧，不适用 |
| **Share Your Attention**（[2508.04581](https://arxiv.org/abs/2508.04581)，2025-08） | attention Q/K/V/O 用共享字典原子分解，减 66.7% 参数、性能持平 | 压缩/部署向；不解决 bpb；不做 |

---

## 4. 特别回答

### 4.1 (a) 在我们 288M、8B tokens 下，哪 2–3 个架构改动最值得实测？

**排序（按“证据贴合度 × 成本 × 上限”）：**

| 排名 | 实验 | 为什么是它 | 成本 | 预期 | 风险 |
|---|---|---|---|---|---|
| **1** | **E1：shape×mHC 2×2**（24×1024 vs 32×896，mHC on/off，2B 锚点） | 唯一能同时回答“24L 是否已过深”和“mHC 在 depth 24 值多少”的实验；配置级改动，1–2 人日；MobileLLM 与 Depth Delusion 冲突，必须用自己的数据裁决 | 4 臂×2B≈35.5 GPU·h≈**¥110** + eval ¥4 | 形状主效应 ±0.010；mHC 主效应 0.004–0.008；交互 ±0.005 | 32L 可能 OOM；mHC off 可能掉出快路径；LR 需匹配 |
| **2** | **E2：Block AttnRes 3 臂**（mHC / AttnRes / mHC+AttnRes，1B） | 296M/62B 上 AttnRes 1.804 vs mHC 1.807，436M 上 1.737 vs 1.747；I/O 5.5d vs 34d，理论上可替代 mHC 并省内存；2026 高关注（88 引） | 3 臂×1B≈13.5 GPU·h≈**¥40** + 3–5 人日 + 调试 ¥30–60 | 0.005–0.012 bpb；能力 +0.5–1.0 | kernel 工作量大；与 mHC 叠加未验证；可能只打平 |
| **3** | **E3：MUDDFormer-lite 2 臂**（baseline vs 块级 dense connections，1B） | 405M/24L/d1024/7B 就是我们的形状，公开 1.8× compute；如果迁移，是上限最高的改动 | 2 臂×1B≈11 GPU·h≈**¥35** + 5–8 人日 + 调试 ¥60–90 | full 0.008–0.015；lite 0.004–0.010 | 内存 +29% 会顶爆 32GB；full 版不可行；lite 可能只拿到一半收益 |

**如果只能做一个**：做 E1。它最便宜、最可解释，而且无论结果如何都能校准后续（如果 32L 不行，E2/E3 的深度相关改动都要降级）。
**如果只能押一个“高上限”**：E3 MUDDFormer-lite；但先做 12L/0.2B 的内存/速度原型，确认不 OOM 再投入。
**如果只能押一个“低风险”**：E2 Block AttnRes；它的 296M 锚点最接近我们，且 I/O 比 mHC 低。

### 4.2 (b) 我们的 mHC/HC 在 depth 24 的收益是否可能被高估？

**结论：相对增益没有被高估，绝对 bpb 被高估了；如果期待 >0.01 bpb，就是高估。**

把公开证据统一换算（1 nat ≈ 0.00325 bpb）：

| 设置 | baseline | +HC/mHC | Δ nats | 相对 | 换算 bpb | 来源 |
|---|---:|---:|---:|---:|---:|---|
| OLMo-1B / 500B | 2.811 | 2.781 | −0.030 | 1.07% | 0.0097 | HC |
| OLMo-7B / 500B | 2.581 | 2.559 | −0.022 | 0.85% | 0.0071 | HC |
| 27B / 262B | — | — | −0.021 | ~0.8–1% | ~0.0068 | mHC |
| 296M / 62.1B | 1.829 | 1.807（mHC-lite） | −0.022 | 1.20% | 0.0071 | AttnRes 表 2 |
| 0.5B / 10B | 2.796 | 2.764（mHC-4） | −0.032 | 1.14% | 0.0104 | DAR 表 3 |

几个判断：

1. **相对增益大致 scale-invariant**：从 0.3B 到 27B 都在 0.8–1.3%。所以“27B/262B 的证据”不意味着 288M 上绝对收益更大；它只说明相对比例可能保持。
2. **绝对收益随 bpb 下降而下降**：bpb 1.066 上 1% ≈ 0.0107；但 8B 只有 62B 的 1/8，AttnRes 论文说 gap 在 decay 阶段才拉大，8B 时大概率只拿到 50–70% 的相对增益 → **0.004–0.008 bpb**。
3. **mHC 的非 bpb 价值**：mHC 论文在 27B 上 HC 有 loss spike，mHC 没有；如果关掉 mHC 后 24L 在 8e-4 下不稳定，那 mHC 的价值是“允许更高 LR/更稳”，不是“同 LR 下更低 bpb”。这一点 2×2 的固定 LR 设计测不出来，需要加一个 LR 探针（见 E1b）。
4. **不要加 stream**：HC 论文 n=8 相对 n=4 几乎无增益（V2 2.778 vs 2.781），但 I/O 线性涨；我们的 32GB 余量 <4GiB，hc_mult=6/8 风险高、收益低。

**建议**：保留 mHC；在 E1 里用“mHC off”作为对照，若 24L 上 off−on <0.004 bpb，则承认“mHC 对 bpb 中性，价值在稳定性/已有 kernel”，不再为它做新工程。若 off−on ≥0.006，说明它仍有实际收益，保留并考虑在 32L 上继续用。

### 4.3 (c) deep-thin（32×768）× mHC 的交互是否值得测？怎么设计 2×2 锚点实验？

**是否值得：值得，但要先纠正一个设计错误——32×768 不是 iso-param。**

按我们的 SwiGLU 形状粗算（忽略 MLA 低秩细节）：

| 配置 | 每层参数（4d²+3dh） | ×层数 | 相对 24×1024 | L×d（激活代理） |
|---|---:|---:|---:|---:|
| 24L×1024，h=2304 | 11.27M | 270.5M | 1.00 | 24,576 |
| **32L×896，h=2016** | 8.63M | **276.2M** | **1.02** | 28,672（+16.7%） |
| 32L×768，h=1728 | 6.34M | 202.9M | 0.75 | 24,576（持平） |

所以：
- **2×2 主实验用 32×896**，才是 iso-param；32×768 是“少 25% 参数/FLOPs”的另一个点。
- 32×896 的 `L×d` 高 16.7%，按 27.78 GiB 峰值推算会顶到 ~30 GiB，需实测；若 OOM 需要 activation checkpointing（慢 20–30%）或先降到 seq 4096 做原型（但最终对比必须 seq 8192）。
- 32×768 的 `L×d` 与 24×1024 相同，显存友好，但参数少 25%；如果一定要测这个点，应做 **iso-FLOPs 扩展**：给 32×768 10.7B tokens（8B/0.75），成本约 ¥142，与 24×1024 @8B（¥106）比。

**2×2 设计（E1）**：

| 臂 | 深度/宽度 | h | mHC | tokens | 目的 |
|---|---|---|---:|---:|---|
| A | 24L×1024 | 2304 | on | 2B | 基线（P1 形状） |
| B | 24L×1024 | 2304 | off | 2B | mHC 主效应 |
| C | 32L×896 | 2016 | on | 2B | deep-thin 主效应 |
| D | 32L×896 | 2016 | off | 2B | 交互 |

- **数据/seed**：同一数据顺序、seed 1234、同 LR 8e-4（先不做 per-shape 调 LR）；eval 在 0.5/1/1.5/2B 四个点，3M panel，raw+EMA。
- **成本**：4×2B=8B tokens 总量；按 62.5k tok/s 是 35.5 GPU·h≈¥106；32L 慢 15–20%，约 ¥120；eval ¥4；合计 **¥110–125**。1–2 人日（配置、mHC off 路径、吞吐/显存 benchmark）。
- **判据**（2B 时噪声 ~0.004）：
  - 深度主效应 `ΔD = bpb(32,mHC) − bpb(24,mHC)`：≤−0.008 → deep-thin 胜；≥+0.008 → deep-thin 败；中间 → 不结论，优先 24L。
  - mHC 主效应 `ΔM = bpb(24,off) − bpb(24,on)`：≥+0.006 → mHC 有用；|ΔM|≤0.004 → 中性。
  - 交互 `ΔΔ = [bpb(32,off)−bpb(32,on)] − [bpb(24,off)−bpb(24,on)]`：≥+0.005 → mHC 在 32L 更有价值（深窄需要 mHC）；≤−0.005 → mHC 在 32L 反而有害。
  - 同时记录 tokens/s、峰值显存、bpb/¥；如果 32L 的 bpb/token 赢但 bpb/¥ 输，按“同预算效率”判负。
- **LR 混淆**：8e-4 是为 24L 调的；32L 可能需要更低 LR，mHC off 可能需要更低 LR。加 **E1b：0.2B LR 探针**（32L 在 0.6/0.8/1.0×，3 臂×0.2B≈0.6B tokens≈¥8），只用于选 LR，不进入主比较。若 32L 在 8e-4 发散，先降 LR 再重跑，不要直接判死。
- **8B 确认**：如果 32L+mHC 在 2B 胜 ≥0.008，跑 2 臂 8B（24+mHC vs 32+mHC）≈71 GPU·h≈**¥212**；总 E1+确认 ≈¥322，略超 ¥300，可把 screen 改成 1.5B（4×1.5B≈¥80）使总预算 ≈¥292。
- **32×768 扩展（E1-ext）**：如果团队坚持 MobileLLM 的 32×768 点，做 2 臂 iso-FLOPs：24×1024 @8B（¥106）vs 32×768 @10.7B（¥142），总 ¥248；判据 ≥0.008 bpb。不要把它塞进 2×2。

### 4.4 (d) 哪些架构改动在单卡 5090、我们的 kernel 支持（dim 1024 生产快路径）下风险过高？

| 改动 | 为什么风险高 | 证据/约束 | 建议 |
|---|---|---|---|
| **MoE（OLMoE/MobileMoE/MiniCPM-MoE）** | 单卡无 EP/all-to-all；专家并行/负载均衡/分组 GEMM 都要新 kernel；总参数和 optimizer 把 32GB 顶爆；8B tokens 远不够专家收敛 | OLMoE 5T、MobileMoE 6T tokens；我们 8B | 不做；Engram 已是更便宜的稀疏容量 |
| **KDA/Mamba/混合线性注意力** | 需要 chunkwise KDA、SSM scan、与 MLA 混合的层间调度；3–6 周；长上下文收益为主，训练 bpb 证据在 1.4T tokens/3B active | Kimi Linear 48B/3B/1.4T；Mamba-2 300B tokens | 不做；保留现有 attention |
| **full DAR** | 两条流 + 跨流深度 attention + 门控写回；O(L) 状态；与 mHC Sinkhorn 两套逻辑；只有 1 篇/1 引 | 0.5B/10B 的 5% 需要独立复现 | 不做；等复现或先做 12L 原型 |
| **full MUDDFormer** | O(L) 深度状态；论文 +29% 激活内存、训练速度降到 81%；我们余量 <4GiB | 405M/7B 的 1.8× 很诱人，但 full 版 memory 不可行 | 只做块级/截断 lite；先 12L/0.2B 原型 |
| **hc_mult>4** | I/O 随 stream 线性涨；HC n=8 相对 n=4 几乎无增益；kernel 快路径按 4 调 | HC 表 1：n=4 2.781 vs n=8 2.777（V2） | 不做 |
| **vocab 64K+** | head FLOPs +50%+、embedding/optimizer 内存涨；重分词 8.5B tokens；bpb 不自动改善 | 2407.13623 的 302M 最优点在 16–24K；2026 论文说 4.44 已接近 BPE | 不做；最多 24K vs 41K 小锚点 |
| **weight sharing / embedding tying** | 省参数但 288M/8B 容量更紧；MobileLLM 的收益在 1T tokens 且用省下的参数加层 | MobileLLM-LS +0.7/0.8（1T tokens） | 不做 |
| **TEMPER / Delta AttnRes / Post-LN / onóna** | 2026 新、无 production kernel、无大规模独立复现；会动 mHC/归一化路径 | 多为 0–1 引/单篇 | watch；不在主实验里混入 |
| **Engram 表 >512M** | SparseRowAdam 状态、表内存、host offload 不在训练路径；可能 OOM | 27B 论文的 1.6B 表是在 MoE 稀疏预算内 | 最多做到 512M，且先 0.6B 锚点 |
| **改归一化/残差缩放（Post-LN、scaled residual）** | 与 mHC Sinkhorn 数值敏感路径交互；production kernel 可能失效 | 项目已记录 mHC 数值敏感 | 不在 32L 实验里同时改 |

---

## 5. 候选实验（2×2 / 3 臂，含预算与判据）

### 5.1 实验总表

| ID | 问题 | 臂 | tokens/臂 | GPU·h | ¥ | 工程 | 成功判据 | 决策 |
|---|---|---|---:|---:|---:|---|---|---|
| **E1** | depth 24 vs 32、mHC 值多少、两者交互 | 24×1024±mHC；32×896±mHC（4 臂） | 2B | 35.5–40 | **110–125** | 1–2 人日 | 2B 上 ΔD≤−0.008 / ΔM≥+0.006 / ΔΔ≥+0.005（分别判） | 赢的臂上 8B 确认 |
| **E1b** | 32L/mHC-off 的 LR 是否被 8e-4 坑了 | 32L × {0.6,0.8,1.0}×LR（3 臂） | 0.2B | 2.5 | **8–10** | 0.5 人日 | 最优点比 8e-4 好 ≥0.006 | 只用于选 LR |
| **E1-ext** | 32×768 在 iso-FLOPs 下是否值得 | 24×1024@8B vs 32×768@10.7B | 8/10.7B | 83 | **248** | 1 人日 | 32×768 胜 ≥0.008 | 仅当坚持 32×768 |
| **E2** | AttnRes 能否替代/叠加 mHC | mHC；Block AttnRes；mHC+Block AttnRes（3 臂） | 1B | 13.5 | **40** + 调试 30–60 | 3–5 人日 | AttnRes−mHC≤−0.006 且吞吐≥95%；或叠加−max≤−0.004 | 赢的臂上 2B 确认（¥54） |
| **E3** | 块级 dense connections 能否拿到 1.8× compute 的一部分 | baseline；MUDD-lite（2 臂） | 1B | 11 | **35** + 调试 60–90 | 5–8 人日 | ≥0.010 bpb 且吞吐损失≤10%、内存<30GiB | 过则 2B；否则停 |
| **E4** | Engram 220M 是否过大 | 64M vs 220M（2 臂） | 0.6B | 5.4 | **16** | 1 人日 | 64M 在 0.003 内 → 缩；220M 胜 ≥0.005 → 留 | 可选 |
| **E5** | V=41,280 是否偏大 | 24K vs 41K（2 臂） | 0.6B | 5.4 | **25**（含重分词） | 2–3 人日 | ≥0.005 bpb | 可选，不优先 |

> GPU·h 按 62.5k tok/s、¥2.98/h；32L/896 和 MUDD-lite 按慢 15–20% 估。3M eval 约 ¥1/次，已含在 ¥ 栏的粗估里。

### 5.2 推荐执行顺序（总预算控制在 ¥300 内）

**Stage 1（¥160–175，1–2 周）**
1. E1 2×2 @2B：¥110–125；同时跑 E1b LR 探针 ¥8–10。
2. E2 三臂 @1B：¥40；先做 12L/0.2B 原型（¥5）验证 Block AttnRes kernel。
3. 读结果：如果 E1 的 32L+mHC 胜 ≥0.008，进入 Stage 2a；如果 E2 的 AttnRes 胜 ≥0.006，进入 Stage 2b；两者都赢则优先 E1（形状收益更便宜）。

**Stage 2（二选一，¥212 或 ¥54）**
- **2a（E1 确认）**：24×1024+mHC vs 32×896+mHC @8B，2 臂≈¥212。这是唯一能在真实 8B 端点回答“deep-thin 是否值得”的实验。
- **2b（E2 确认）**：mHC vs AttnRes @2B，2 臂≈¥54；若赢，再考虑 8B（¥212）。

**Stage 3（可选，¥35–60）**
- E3 MUDDFormer-lite @1B；只在 Stage 1 证明“残差改动有 ≥0.006 bpb 空间”且工程有余力时做。
- E4 Engram 表大小 @0.6B（¥16）；可与 E1 并行，几乎不占工程。

**不建议在本轮做**：E5 vocab（收益不确定、重分词贵）、E1-ext 32×768（除非团队坚持）、任何 MoE/KDA/DAR full。

### 5.3 判据的统一口径

- 主指标：**held-out bpb（3M panel，raw）**；EMA 作为辅助（P1 实测 EMA −0.0042，稳定）。
- 次指标：**bpb/¥**（同预算效率）、tokens/s、峰值显存。
- 统计：同一数据顺序、seed 1234；接近的臂（|Δ|<0.006）跑第二 seed 或延到 2B。
- 能力：如果便宜，加 CORE/ARC-easy 小评测；但 288M/8B 的下游噪声大，**不要用单点下游分数否决 bpb 结论**。
- 早停：2B 时 Δ 超过 ±0.015 或显存/吞吐不可接受，立即停。

---

## 6. 风险清单

| 风险 | 表现 | 缓解 |
|---|---|---|
| **短跑排名反转** | P1 的 β 从 0.6B 的 0.154 降到 8B 的 0.06–0.08；1B 赢的臂 8B 可能输 | 2B screen；只有 ≥0.008 才上 8B；报告 0.5/1/1.5/2B 多点曲线 |
| **kernel/吞吐混淆** | mHC off 掉出 fused 快路径；896/768 宽度不是 dim1024 tile；32L 启动开销大 | 每臂测 tokens/s、峰值显存、kernel 时间；同时报 bpb/token 和 bpb/¥；不要用“同 tokens”掩盖吞吐差 |
| **32L OOM** | 27.78 GiB 峰值；32×896 激活 +16.7%；b2 已 OOM | b1、activation checkpointing、先 12L/seq4096 原型；32×768 fallback |
| **LR 不匹配** | 8e-4 为 24L 调；32L/mHC-off 可能需要更低；mHC 可能允许更高 | E1b 0.2B LR 探针；主实验固定 LR，但记录发散/尖峰 |
| **统计噪声** | 3M panel 0.003–0.005；单 seed | 配对数据顺序；接近的臂 2 seeds；最终 5M panel |
| **新论文不可复现** | DAR 1 引、Depth Delusion 1 引、TEMPER/onóna 0–1 引；代码未集成 | 先 12L/0.2B 原型；要求 ≥2 个独立信号；不直接上 8B |
| **形状混淆** | 32×768 少 25% 参数/FLOPs；iso-param 与 iso-FLOPs 不同 | 2×2 用 32×896；32×768 只做 iso-FLOPs 扩展；明确口径 |
| **Engram/词表改动副作用** | 表 resize/SparseRowAdam 状态、重分词、token_bytes 面板对齐 | 0.6B 小锚点；先验证 eval harness；不改生产 tokenizer |
| **工程机会成本** | AttnRes/MUDD 3–8 人日；数据侧可能更划算 | 架构工程上限 2 周；E1 先做；E2/E3 二选一 |
| **把 bpb 当能力** | bpb 对 web val 敏感，下游能力可能不同 | 加小 CORE/ARC；结论只写“bpb + 初步能力”，不夸大 |

---

## 7. 参考文献

> 引用数为 Semantic Scholar 2026-11-18 快照；“经典/高引用”与“2026 新/未证实”为定性标签。

**残差/深度**

1. Hyper-Connections. *arXiv 2024*；ICLR 2025；113 引；经典. <https://arxiv.org/abs/2409.19606>
2. mHC: Manifold-Constrained Hyper-Connections. *arXiv 2025-12*；113 引；2025 末新/高关注. <https://arxiv.org/abs/2512.24880>
3. Attention Residuals. *arXiv 2026-03*；88 引；2026 新/高关注. <https://arxiv.org/abs/2603.15031>
4. Dual Attention Residuals. *arXiv 2026-07*；1 引；2026 新/未证实. <https://arxiv.org/abs/2607.18730>
5. DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging. *arXiv 2024*；NeurIPS 2024；72 引；经典. <https://arxiv.org/abs/2402.02622>
6. MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections. *arXiv 2025*；ICML 2025；43 引；高引用. <https://arxiv.org/abs/2502.12170>
7. Frac-Connections: Fractional Extension of Hyper-Connections. *arXiv 2025-03*；9 引；新. <https://arxiv.org/abs/2503.14125>
8. Delta Attention Residuals. *arXiv 2026-05*；新/未证实. <https://arxiv.org/abs/2605.18855>
9. TEMPER: Tensorized Efficient Manifold-constrained Parameterization for Expressive Residual Routing. *arXiv 2026-08*；新/未证实. <https://arxiv.org/abs/2608.07851>
10. Post-LayerNorm Is Back: Stable, ExpressivE, and Deep. *arXiv 2026-01*；新/未证实. <https://arxiv.org/abs/2601.19895>

**形状/深度 scaling**

11. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. *arXiv 2024*；ICML 2024；376 引；经典. <https://arxiv.org/abs/2402.14905>
12. The Depth Delusion: Why Transformers Should Be Wider, Not Deeper. *arXiv 2026-01*；1 引；2026 新/未证实. <https://arxiv.org/abs/2601.20994>

**注意力/混合架构**

13. Kimi Linear: An Expressive, Efficient Attention Architecture. *arXiv 2025-10*；245 引；高引用. <https://arxiv.org/abs/2510.26692>
14. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2). *arXiv 2024*；经典. <https://arxiv.org/abs/2405.21060>
15. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv 2023*；经典. <https://arxiv.org/abs/2312.00752>
16. Hymba: A Hybrid-head Architecture for Small Language Models. *arXiv 2024*；ICLR 2025；126 引；经典. <https://arxiv.org/abs/2411.13676>
17. Component Ablation for Efficient Hybrid Language Model Architectures: Performance, Resilience, and Compression Implications. *arXiv 2026-03*；新. <https://arxiv.org/abs/2603.22473>

**稀疏/记忆/参数效率**

18. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram). *arXiv 2026-01*；ACL 2026；104 引；2026 新/高关注. <https://arxiv.org/abs/2601.07372>
19. OLMoE: Open Mixture-of-Experts Language Models. *arXiv 2024*；经典. <https://arxiv.org/abs/2409.02060>
20. MobileMoE: Scaling On-Device Mixture of Experts. *arXiv 2026-05*；新. <https://arxiv.org/abs/2605.27358>
21. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. *arXiv 2024*；经典. <https://arxiv.org/abs/2401.06066>
22. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. *arXiv 2024-10*；新. <https://arxiv.org/abs/2410.20672>
23. Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning. *arXiv 2025-08*；新. <https://arxiv.org/abs/2508.04581>
24. onóna: Twin Embeddings with Tied Unembeddings. *arXiv 2026-11*；极新/未证实. <https://arxiv.org/abs/2611.10556>
25. MiniCPM4: Ultra-Efficient LLMs on End Devices. *arXiv 2025-06*；高关注. <https://arxiv.org/abs/2506.07900>
26. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. *arXiv 2024*；高引用. <https://arxiv.org/abs/2404.06395>

**Tokenizer/词表**

27. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies. *arXiv 2024*；NeurIPS 2024；144 引；经典. <https://arxiv.org/abs/2407.13623>
28. Compute Optimal Tokenization. *arXiv 2026-05*；4 引；2026 新. <https://arxiv.org/abs/2605.01188>

**内部依据**

- [r93-p1-final-report.md](../r93-p1-final-report.md)：P1 8B 终值、β、成本、显存、EMA。
- [r93-p0a-p0b-results.md](../r93-p0a-p0b-results.md)：d24_c1 配置、0.6B 校准、27.78 GiB 峰值、b2 OOM。
- [r94-small-model-methods-survey.md](../r94-small-model-methods-survey.md)：数据/KD/超参侧调研（与本篇互补）。
- [architecture.md](../architecture.md)：mHC/Engram/MLA 的实现与 kernel 约束。

---

### 附：一页“做/不做”清单

**做**
- E1：24×1024 vs 32×896 × mHC on/off，2B，¥110–125。
- E2：mHC / Block AttnRes / 叠加，1B，¥40 + 3–5 人日。
- E3（stretch）：MUDDFormer-lite，1B，¥35 + 5–8 人日。
- E4（可选）：Engram 64M vs 220M，0.6B，¥16。

**不做**
- MoE、KDA/Mamba、full DAR、full MUDDFormer、hc_mult>4、vocab 64K+、weight sharing、TEMPER/Post-LN 混入主实验。

**一句话**
> 在 288M/8B 上，架构是“0.005–0.015 bpb 的精细活”；先用 ¥110 的 2×2 把“深度/mHC”这两个最大不确定性钉死，再用 ¥40 的 AttnRes 三臂决定是否换掉 mHC。其余大改动的收益/成本比都不支持在单卡 5090 上做。
