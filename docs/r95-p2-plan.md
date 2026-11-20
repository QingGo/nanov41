# P2 未来计划（2026-11-18 制定）

## 0. 目标与约束

- **目标**：从零训练（不能续 P1 ckpt）达到 **bpb<1.0（loss<3.09）**，越低越好；次级目标：能力可测。
- **约束**：单张 RTX 5090（¥2.78–3.14/h，62.5k tok/s，bf16 现状）；单轮预算 ¥100–300；
  **尽量保持 DeepSeek-V4.1-Flash 官方架构**（CSA2/CED/mHC/indexer/Engram；dim 1024 快路径）；
  语料目前 8.57B tokens 单 epoch（可重复至 4 epoch）。
- **baseline**：P1 = 8B tokens，raw bpb **1.06235** / EMA **1.05808**（全量 val），MFU 51.6%。

## 1. 三档预算与预期

| 档 | 内容 | 成本 | 预期 bpb（loss） | bpb<1 概率 |
|---|---|---|---:|---:|
| **A. ¥100** | 7.3B tokens 从零 + 全部零成本/系统杠杆（FP8/TWEO、MuonH、decay/min-lr、curriculum、EMA） | ~¥100–115 | **1.00–1.03**（3.09–3.18） | ~40–55% |
| **B. ¥150** | 11B tokens（1.3 epoch）+ 同杠杆 | ~¥150 | **0.99–1.01**（3.06–3.12） | ~60–70% |
| **C. ¥250** | 18B tokens（2.1 epoch）+ 同杠杆 + 可选 AttnRes/indexer 改进 | ~¥250 | **0.97–1.00**（3.00–3.09） | ~80% |
| D. ¥350–600 | 600–730M active × 10–20B tokens 或 KD+放大（冲 0.9 以下） | ¥350–600 | 0.90–0.97 | 需先锚点校准 |

## 2. 阶段与实验（按顺序）

### Phase 0（¥0，1–2 天）：代码/配置项（无 GPU 花费）
1. **FP8 body + TWEO 补丁**（ZGCM 路线）：body GEMM 走 fp8（delayed scaling），TWEO 式 outlier 正则
   （抑制激活外点，保 fp8 稳定）。我们已有 fp8 head/linear-fp8 内核基础，估算 3–10 人日；
   目标吞吐 **+30–60%**；
2. **MuonH（Hyperball 范数约束）**：`W ← R·normalize(W − ηR·normalize(u))`，3 行，1 人日；
3. **调度配置**：线性 decay、decay ratio 0.5、min-lr 绝对值 1e-5、warmup 1%（现 WSD 10% decay 保留为对照）；
4. **indexer aux loss**（官方缺口）：DSA 风格让 indexer 预测主 attention 的 top-k 分布，2–4 人日；
5. **评测管线**：全量 val（32min/ckpt，¥1.6）+ lm-eval-harness（¥1–2/轮，17min）固化为一键脚本。

### Phase 1（¥35–60，1 天）：0.5B 锚点 A/B（判据 Δbpb≥0.008）
| 实验 | 臂 | 成本 |
|---|---|---:|
| E1 FP8+TWEO | bf16 对照 vs fp8+TWEO | ¥12 |
| E2 MuonH | Muon 对照 vs MuonH | ¥12 |
| E3 调度 | WSD-10% vs 线性-decay-50%/min-lr 1e-5 | ¥12 |
| E4 数据 | 纯 ClimbMix vs 末段15%高质量（FineMath/Stack-Edu/Cosmopedia）| ¥12 |

### Phase 2（¥100–130，2–3 天）：从零 8B 全叠加（档 A）
启动命令基于 `tools/r93_p1_train.sh` 修改：FP8+TWEO、MuonH、decay-0.5、min-lr 1e-5、
末段 curriculum、EMA 0.995；`--init-from` 无（从零）；保存 2B/4B/6B/8B mark + EMA；
评测：4 个 mark + 终评 raw/EMA 全量 val + lm-eval 子集。
**Gate**：若 8B 终评 bpb ≤1.01 → 追加 2.5B tokens（+¥35，档 B）稳定跨过 1.0；若 ≤1.03 且锚点显示
FP8 未达标 → 补做 kernel 优化再跑。

### Phase 3（¥230–260，2 天）：2 epoch 17–19B（档 C，bpb<1 把握大）
数据 2 epoch（≤4 safe），同全栈；终评全量 val + lm-eval。

### Phase 4（可选，¥350–600）：模型放大/KD 路线
- 600–730M active × 10–20B tokens（先按 `docs/r94-shape-scaling.md` 形状律：0.6B→25L×1480、1B→29L×1820）；
- 或 288M student + 离线数据精炼（MiniPLM-lite，teacher 1.7B，¥40–60 打分）+ 更长训练；
- 必须先用 0.5B/1B 锚点校准，否则不投。

## 3. 预算汇总（不含 P1 已花）

| Phase | 内容 | 成本 |
|---|---|---:|
| 0 | 代码/配置/评测 | ¥0 |
| 1 | 锚点 A/B ×4 | ¥48 |
| 2 | 8B 全叠加（档 A） | ¥105–115 |
| 2' | 追加 2.5B（条件） | ¥35 |
| 3 | 18B 2-epoch（档 C） | ¥235–255 |
| 4 | 放大/KD（可选） | ¥350–600 |
| **最小达标路径（0→2→2'）** | | **≈¥190–200** |

## 4. 不做清单（来自各调研报告的风险项）

LR re-warmup、fp8 直接上 8B 不先锚点、logit soft-capping、换 SOAP/Lion、切 muP、
2^22 tokens/step（已实测差）、QAT 进预训练、单改数据顺序、>4 epoch 重复、
在线 KD（等算力不赚）、EDI/LIMO 式少量 SFT（Phi-4-Mini 反证）、
不必要地替换 CSA2→SWA（8K 不省 FLOPs，且破坏官方对齐）、CED/Reindex 模式（8K 实测 +7% wall）。

## 5. 时间表（相对）

| 天 | 事项 |
|---|---|
| D0 | Phase 0 代码（FP8/TWEO、MuonH、调度、indexer aux loss、评测固化） |
| D1 | Phase 1 锚点 4×0.5B + 判据评估 |
| D2–D4 | Phase 2 8B 全叠加（35.5h GPU + 评测） |
| D4 | 看终评；决定是否追加 2.5B / 是否进 Phase 3 |
| D5–D7 | Phase 3（若启动，2 epoch 76h GPU） |

## 6. 交付物与复用资产

- 代码：`tools/r95_*`（FP8/TWEO 补丁、MuonH、调度、indexer aux loss、评测一键脚本）
- 文档：`docs/r95-p2-plan.md`（本文件）、`r95-small-model-research-synthesis.md`、`r95-p1-eval-report.md`
- 数据：8.57B 语料（在实例数据盘）+ 公开合成数据（FineMath/Stack-Edu/Cosmopedia）
- 评测：全量 val 评测脚本 `tools/r93_fullval_eval.py`、能力评测 `tools/r93_capability_snapshot.py`
