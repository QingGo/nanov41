# P1 8B 完整测评报告（2026-11-18）

> 覆盖：训练曲线 mark、全量验证集 bpb/loss、EMA 增益、评测抽样误差、能力基准（MMLU/HellaSwag/GSM8K/ARC-easy）、
> 多域 bpb（web/code/math）、生成样例，以及与 nanochat 评测口径的对齐说明。所有数字来自本地归档报告与实例日志。

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| **全量 val raw bpb** | **1.06235**（CI95 [1.06028, 1.06407]，56.75M tokens，nats/token 3.2991） |
| **全量 val EMA(0.995) bpb** | **1.05808**（CI95 [1.05602, 1.05980]，nats/token 3.2858） |
| EMA 增益 | **−0.00427 bpb**（与 3M 面板的 −0.0042 一致） |
| 3M 面板 vs 全量 | 3M 子集估 1.06624（+0.0039，在噪声内）；8 个随机 3M 子集 std **0.00452**，偏差范围 [−0.0056, +0.0099] |
| 域内异质性 | 10 个连续 ~5.67M 段：bpb 1.0480–1.0714，std 0.00736 |
| 能力（修复版快照） | MMLU **23.3%**（5-shot，n=150，≈随机 25%）；HellaSwag **34.0%**（10-shot，n=150）；GSM8K **5.6%**（8-shot，n=18）；ARC-easy **29.3%**（25-shot，n=150） |
| 与 nanochat 同数据对比 | nanochat d24（~4e19 FLOPs）val_bpb **0.71800**、CORE 0.2626；我们（1.38e19 FLOPs）val_bpb **1.06235** |

## 2. 模型与训练（P1）

d24_c1 dense（288M active；dim 1024/24 层/FFN 2304；V=41,280；seq 8192；mHC hc_mult=4；Engram 220M；
CSA2 稀疏注意力 window 128 + topk 64 + 压缩 KV）；8.0e9 tokens；LR 8e-4 WSD(30/0.9/0.1)；z-loss 1e-4；
EMA 0.995；bf16 + fp8 head；62,569 tok/s，MFU 51.60%，训练 35.51h，实例成本 ≈¥128。

## 3. 验证数据与协议

- **数据**：`karpathy/climbmix-400b-shuffle` 是 nanochat 对 **`nvidia/Nemotron-ClimbMix`** 的重新打包：
  源数据是 GPT-2 tokenized 的 `tokens` 列（NVIDIA 用 **CLIMB 框架**选出的 400B-token 数据混合，来自
  ClimbLab 的 20 个语义簇——含教育/百科/数学逻辑/科技新闻等**网页衍生簇**，按嵌入+K-means 聚类、
  fasttext 质量过滤后按比例混合，**降重后以单一 token 流发布，没有分域标签**）；repackage 时 decode 回 text、
  shuffle(seed=42)、写成 6542 个 `text` 单列 parquet。**最后一个 shard `shard_06542.parquet` 作 val（56,778,205 tokens）**。
- **我们的语料** `p2_climbmix_40k` 用同一份 6542 shard 重 tokenize 到 K=40k，训练时显式排除 06542，因此
  我们的 val 与 nanochat 的 val 是**同一段文本**（只是 tokenizer 不同）。
- **评测面板**：`p0_eval_40k`，域 `climbmix_val`，window 2048，非重叠窗口、首个 token 不计分；
  bpb 按 `token_bytes` 逐 token 字节加权（sum nats / (ln2·sum bytes)），与 nanochat `loss_eval.evaluate_bpb` 同一定义。
- **nanochat 协议差异**（对比时需注意）：nanochat 的 val_bpb 默认 `--split-tokens 40×524288 ≈ 20.97M tokens`
  子采样 + BOS 前缀 best-fit 打包；我们本轮用**全量 56.75M**、非重叠窗口。两者同 shard、同 bpb 定义，
  仅子采样与窗口化细节不同（噪声 ±0.005 量级）。
- **能力评测协议**：few-shot 上下文打分（MMLU 5-shot 选字母、HellaSwag 10-shot 选结尾、GSM8K 8-shot 生成抽答案），
  n 限制与说明见 §5；与 lm-eval-harness 的数字不完全等价（后者是标准实现，我们的是自研轻量版）。

## 4. bpb / loss 轨迹

| tokens | val bpb（3M 面板） | nats/token |
|---|---:|---:|
| 1B | 1.2771 | 3.9462 |
| 2B | 1.1897 | 3.6761 |
| 4B | 1.1176 | 3.4531 |
| 6B | 1.0905 | 3.3695 |
| **8B raw（全量）** | **1.06235** | **3.2991** |
| **8B EMA（全量）** | **1.05808** | **3.2858** |

- 局部 β：1→2B 0.102、2→4B 0.090、4→6B 0.060、6→8B 0.078；带地板项拟合 `L=E+A·D^-β`
  给 E≈0.887、β≈0.372（残差 ≤0.003 bpb），说明**短跑曲线外推出的 0.89 目标本来就不可达**（见调研报告）。
- 训练末步 train loss 3.2923（与 val 3.2991 基本一致，单 epoch 0.93 无过拟合）。

## 5. 能力基准（P1 final raw ckpt，3M/面板无污染）

| 基准 | 我们 | 说明 |
|---|---:|---|
| MMLU（5-shot，n=150） | **23.33%** | 字母 logprob 打分；随机基线 25% → **低于随机（噪声内）** |
| HellaSwag（10-shot，n=150） | **34.0%** | 4 选 1 结尾平均 logprob；随机 25% |
| GSM8K（8-shot，n=18） | **5.56%** | greedy 生成+抽答案，时间预算截断到 18 题；随机≈0 |
| ARC-easy（25-shot，n=150） | **29.33%** | 修复版重跑（首跑因长上下文截断 bug 报错）；随机 25% |
| 生成样例 | 见 §6 | 贪心循环重复；采样偶尔合理 |

**预期对照**（调研报告 kd-capability 的锚点估计：288M/8B 应为 MMLU 25–27、ARC-E 45–52、HellaSwag 30–35、GSM8K 0–1%）：
MMLU 低于预期（23.3 vs 25–27），HellaSwag 在带内，GSM8K 5.6% 高于预期（小样本 n=18，置信区间宽）。

## 6. 生成样例（P1 final，greedy / sample 各一）

| Prompt | 输出节选 |
|---|---|
| "Once upon a time... fisherman who" | greedy: "loved to fish and had a special bond with their loved ones..." / sample: "loved exploring the bustling city life in the nearby town of Lake District..." |
| 冻水 | greedy: "The water in the glass will evaporate, and the water will evaporate. …"（循环） |
| 17+26 | greedy: "2.5 m long and 1.5 m long. 2. 3. 4. 5. …"（无算术） |
| 法国首都 | greedy: "The capital of France is the capital of France. …"；sample 模式答对 "Paris" |
| `def fibonacci(n)` | greedy: "return Fibonacci Fibonacci Fibonacci…" |

## 7. 与 nanochat 的对齐与差距（评测口径已按源码核对）

**nanochat 的评测（`scripts/base_eval.py`）**：
- **val_bpb**：`loss_eval.evaluate_bpb` = sum(nats)/(ln2·sum bytes)，按 `token_bytes` 加权、排除 special token；
  默认 `--split-tokens 40×524288 ≈ 20.97M tokens`（子采样，不是全 shard），loader 为 BOS 前缀 best-fit、seq=2048；
- **CORE**（主指标/排行榜指标）：下载 `eval_bundle.zip`（karpathy S3）→ `core.yaml` 的 **22 个 ICL 任务**
  （DCLM CORE），逐任务 few-shot 准确率 → centered=(acc−0.01·random)/(1−0.01·random) → **CORE=22 任务均值**；
  GPT-2 CORE=0.256525，排行榜主指标是"time-to-GPT-2 CORE"；
- **samples**：7 固定 prompt + 8 无条件采样；chat 阶段另有 MMLU/GSM8K/ARC/HumanEval 任务评测（`tasks/`）。
- 排行榜（2026-03）：d24 ~4e19 FLOPs，**val_bpb 0.71800**、**CORE 0.2626**（Jan-2026 旧数据 0.74833 → 换 ClimbMix 后 0.71854 → autoresearch 0.71800）。

**对比**：
- 同数据同 val shard：我们 **1.06235** vs nanochat d24 **0.71800**。差距主因是**算力/规模**
  （我们 1.38e19 FLOPs / 288M active vs nanochat ~4e19 FLOPs / ~730M 宽）；按仓库内 nanochat 插值线
  （iso-FLOPs，C=1.38e19 → bpb≈0.94）仍差 +0.12；
- 口径差异：nanochat val_bpb 是 21M 子采样（±0.005 噪声）、BOS best-fit；我们全量 56.75M、非重叠窗口；
  bpb 已按 byte 归一化，tokenizer 不同不影响可比性（但窗口/子采样细节有 ~0.005 量级差异）；
- **能力**：nanochat d24 CORE 0.2626（GPT-2 0.2565）；我们尚未实现 CORE，MMLU/HellaSwag 处于
  "随机到略高于随机"的 base 小模型水平——**下一步建议实现 CORE 评测**（公开 eval_bundle，¥1–2/轮）。
- **nanochat 也没有"分域 bpb"**：ClimbMix 是单 token 流、无域标签；分域信息只体现在 CORE 的任务和 chat eval 上。


### 7b. CORE 的 22 个任务是什么，代表性如何

CORE = DCLM（DataComp-LM）的 22 任务"低方差"子集（`eval/additional_aggregation.json::low_variance_datasets`），
nanochat 的 eval_bundle `core.yaml` 用同一套协议（few-shot ICL、centered 平均，随机基线=0、GPT-2=0.2565）：

| 组别 | 任务（shots） |
|---|---|
| 常识/语言理解（9） | HellaSwag 0-shot、HellaSwag 10-shot、ARC-Easy 10、ARC-Challenge 10、COPA 0、CommonSenseQA 10、PIQA 10、OpenBookQA 0、LAMBADA-OpenAI 0 |
| Winograd 类（2） | Winograd 0、WinoGrande 0 |
| 阅读理解（3） | SQuAD 10、CoQA 0、BoolQ 10 |
| 世界知识/语言建模（3） | Jeopardy 10、BIG-Bench QA-WikiData 10、BIG-Bench Language Identification 10 |
| 符号/推理（5） | BIG-Bench Dyck Languages 10、AGI-Eval LSAT-AR 3、BIG-Bench CS Algorithms 10、BIG-Bench Operators 10、BIG-Bench Repeat Copy Logic 10 |


**centered 公式的推导**：random 基线以百分数存储（如 4 选 1 = 25），
`centered = (acc − 0.01·random)/(1 − 0.01·random)` 等价于 `(acc% − random%)/(100% − random%)`——
即**把"随机水平"映射到 0、"满分"映射到 1 的 min-max 归一化**，衡量"模型吃掉了多少超过随机的可提升空间"。
目的：① 不同任务 chance level 不同（2 选 1 与 4 选 1 的 raw acc 不可比），归一化后 22 任务等权平均才有意义；
② 防止高 chance 任务"白捡分"主导平均；③ 线性变换、有界（低于随机时为负），计算简单且不需要 logit。
局限：低于随机会得负分；等权平均让低 chance 任务（headroom 大）对均值影响更大；线性尺度对接近饱和的任务不敏感。

**每个任务有多个题目**：是的，每个任务是一个 jsonl 数据集（ARC-Easy ~2,376 题、HellaSwag val ~10,042、
PIQA val ~1,838、WinoGrande val ~1,267、BoolQ val ~3,270、SQuAD ~10,570、LAMBADA ~5,153、COPA ~500…），
最终 accuracy = 该任务全部（或 `--max-per-task` 子采样）题目的平均；few-shot 的示例是"每道题前面拼 N 个示例"，
不是从题库里扣题。注意题量差异极大（BIG-Bench Repeat Copy Logic 仅 ~32 题、Operators ~210 题），
这也是 CORE 用 centered/等权平均时小任务噪声大、DCLM 需要"低方差任务筛选"的原因；我们单卡评测建议
`--max-per-task 150–300` 子采样（~3,300–6,600 题，15–25min，¥1–2），并报告子采样噪声。

合计 22。每个任务：few-shot 上下文 + 准确率 → centered = (acc − 0.01·random)/(1 − 0.01·random) → CORE = 平均。
（DCLM 2025-09 修过 baseline 值 bug：v1/v2 的 CORE 数值不完全可比；nanochat 的 0.2626 按其 bundle 计算。）

**代表性评估**：
- ✅ 作为"数据集质量/模型能力"的低方差代理，它在 400M–7B、英语 base 模型上被 DCLM/nanochat/OLMo/Marin 广泛使用；
  任务覆盖常识、阅读理解、世界知识、符号推理，速度快、方差小（这正是 DCLM 选 22 个任务的原因：95% CI 筛选）；
- ⚠️ 局限：① 纯英语、base 模型 ICL，无代码生成/数学推理（GSM8K/MMLU 不在 CORE 内，DCLM 单独报 MMLU）；
  ② 无指令跟随/多轮/长上下文/多语言；③ 0-shot/few-shot 对数概率打分，对 tokenizer 和提示格式敏感；
  ④ 22 任务等权平均，小样本任务噪声大；⑤ 对 288M/8B 这种小模型会接近随机（我们的预期 CORE 0.02–0.05），
  区分度低——**小模型诊断应同时看分域 bpb（code/math/web）、MMLU/HellaSwag 与生成质量**。
- 结论：CORE 用于与 nanochat 对齐必须实现；但我们自己的 P2 决策应以"分域 bpb + CORE + 少量任务评测"三线并进，
  不能只看 CORE 一个数。

## 8. 多域 bpb（诊断用，非 nanochat 口径）

每域 3M tokens、同一 panel/window/byte 归一化（web 列给出全量口径另注）：

| 域 | raw bpb（3M） | EMA bpb（3M） | 备注 |
|---|---:|---:|---|
| web（climbmix_val） | 1.0662（全量 **1.06235**） | 1.0620（全量 **1.05808**） | 训练分布内 |
| code_py（本地 Python 源码） | **1.8005** | 1.7928 | **分布外：比 web 高 0.73 bpb**（训练语料无代码域） |
| math_web（open-web-math） | **1.5714** | 1.5638 | 分布外：比 web 高 0.51 bpb |
| 宏观平均（3 域等权） | 1.4794 | 1.4729 | — |
| 字节加权 bpb | ~1.44 | 1.4370 | 以各域字节数加权 |

解读：模型在 web 域 1.06，但代码 1.80 / 数学 1.57——**训练数据是单一 ClimbMix 混合流，没有代码/数学域数据，
OOD 损失显著更高**；这也解释了能力评测里 MMLU/GSM8K 接近随机。若 P2 要提升代码/数学能力，需要显式混入对应数据
（但这会占用 web 域的 token 预算）。
（注：nanochat 本身**不分域评测**（ClimbMix 是单 token 流，无域标签）；分域 bpb 是 DCLM/FineWeb 式实践，
我们用来看训练分布外的损失。）

## 9. 成本

| 项目 | 成本 |
|---|---|
| 全量 val 单 ckpt | ~32min ≈ ¥1.6 |
| 3M 面板单 ckpt | ~2min ≈ ¥0.1 |
| 能力快照（4 bench 子集） | ~15–20min ≈ ¥1 |
| 多域 eval（3 域 × 3M × 2 ckpt） | ~12min ≈ ¥0.6 |
| 本轮全部评测 | ≈¥4（5090 ¥3.08/h） |

## 10. 文件与命令

- 全量 val 结果：`runs/r93/p1/eval_{final,ema0p995}.json`（3M）→ 全量：`runs/r93/cap/fullval_{raw,ema}.json`
- 能力快照：`runs/r93/cap/snapshot2.json`（+ `snapshot_arc.json`）
- 多域：`runs/r93/cap/multidomain_{raw,ema}.json`
- 脚本：`tools/r93_fullval_eval.py`、`tools/r93_capability_snapshot.py`、`tools/r95_build_domain_eval.py`
