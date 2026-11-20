# R93 P0a/P0b 计划（2026-11-15）：K=40k 词表/数据/eval 迁移 + KD 数据准备 + dense 校准 run

> 用户决定（2026-11-15）：启动 P0a（词表/数据/eval 迁移 + KD 数据准备，纯 CPU/无卡，¥2–5），
> 数据管线跑的同时安排 P0b 校准 run；**后续主跑倾向 dense（非 MoE）方案**。
> 本文件是执行契约：工作流、验收数字、预算与止损。上游依据：`docs/scaling-learnings.md`
> §8.1/8.1b/8.1c/8.1f/8.1g、`docs/dense-vs-moe-design.md`、`docs/r90-final-report.md`、`docs/r91-search-protocol.md`。

## 1. 决策摘要

| 项 | 决定 | 依据 |
|---|---|---|
| 词表 | **官方 BPE 前缀截断 K=40,000 + 1,280 reserved specials = V=41,280** | K 权衡表 text/¥ 在 32k–48k 平坦；head 效率模型最优 48k–56k，40k 是低风险点；E3 已验证 32k 版本整套栈可用（125k→32k 吞吐 1.35× bf16 / ~1.19× fp8） |
| 训练 tokenizer | 直接按 K=40k tokenizer 编码原始文本（等价于官方 129k 编码后按 merge-tree 展开，property 已在 E3 验证） | 避免 129k 中间流；bytes/token 期望 4.2706（官方 4.6036 的 1.078 展开） |
| 主跑架构 | **dense FFN twin（aligned 家族，保留 attention/mHC/indexer/Engram）**，`ffn_impl="dense"`、`n_routed_experts=1` | 用户倾向 dense；R91 A-screen：等 FLOPs 下 dense 优于 MoE 且吞吐更高（MFU ~43% vs 38%）；R91 D1/D2 缺，但 dense 是大方向 |
| P1 目标 | C ≈ 1.0e19 FLOPs，D ≈ 5.8B tokens，N_active ≈ 285–295M（仓库约定 C = 6·N_active·D） | §8.1b 行：274M active × 5.5B ≈ 9.0e18 → bpb<1（32h ≈ ¥90 级） |
| KD | Go（中等规模）：从 Arcee 公开 logits 取 0.35–0.5B tokens 投影到 K=40k，w≈0.5、rest-bucket，**盘上 ≤22–24GB** | E3：129k+KD 等 bytes −5.2%；24% 覆盖 ≈1.11×；实际盘限制下 6–9% 覆盖 ≈1.04–1.08× |
| P0b | 用 P1 候选 dense config 跑 0.6B tokens 校准（gate 见 §4） | 验证 K=40k 数据/eval/吞吐/配方健康，再决定 P1 |

## 2. 目标数学（仓库约定，已按 recon 核对）

- `fwd_flops_per_token = 2·N_active`；dense：`N_active = backbone 非嵌入参数 + head(V·dim)`，Engram 表不计入 N_active。
- P1：`C = 3·fwd_flops·D = 6·N_active·D`。取 `N_active = 287M`、`D = 5.8B` ⇒ **C = 9.99e18**。
- bpb 插值线（两个锚点，±2× 不确定）：`bpb ≈ 0.748·(C/4.33e19)^−0.20`。

| D (tokens) | C | 预测 bpb |
|---|---:|---:|
| 0.4B | 6.89e17 | **1.71** |
| 0.6B | 1.03e18 | **1.58** |
| 0.8B | 1.38e18 | **1.49** |
| 2B | 3.44e18 | 1.32 |
| 4B | 6.89e18 | 1.09 |
| 5.8B | 9.99e18 | **1.00** |
| 6.2B | 1.07e19 | 0.99 |

- 生产对照锚：d12 MoE 91.5M active 在 5090 上 145,174 tok/s / MFU 38.04%（b2×s8192×accum128）；
  dense 287M active 期望 tok/s ≈ 40–48k（同 MFU 下 ∝ 1/3），MFU 目标 ≥33–35%。

## 3. 工作流（并行）

| # | owner | 主机 | 交付 | 验收 |
|---|---|---|---|---|
| P0a-data | data-ops | 无卡 5090 `<instance-id>`（¥0.1/h） | `p2_climbmix_40k`（5.8B pieces ≈23.2GB u32）+ `p0_eval_40k`（shard_06542，token_bytes 41,280） | manifest tokens ≥5.8e9；bytes/token 4.2706±0.5%；ids<41280；EOS=1；decode 抽查无损 |
| P0a-kd | kd-ops | 无卡 3090 `<instance-id>`（¥0.1/h） | K=40k 投影 KD 数据集（≤24GB，0.35–0.5B tokens，top-k + rest） | dtype/shape/mask 校验；teacher 覆盖率表；可 mmap |
| P0b-calib | calib | 先 Mac（离线准备），后 GPU 实例（Lead 分配） | dense 候选 config + runner + 0.6B-token 校准 run + 分析 | gate 见 §4；GPU 花费 ≤≈¥20 |
| 统筹 | Lead | — | 实例开关机/预算/文档/最终汇报 | 余额记账；完成后关闭自建实例 |

数据/KD 两台无卡实例并行；P0b 在 GPU 容量出现后立即开始（5090 优先，否则 4090 bf16 降级）。

## 4. P0b gate（0.6B tokens，dense 候选配置）

- **通过**：0.4B/0.6B 点 bpb ≤ 预测+0.05（1.71/1.58）且曲线斜率正常；无 NaN/发散；5090 上 tok/s ≥30k（4090 ≥10k）。
- **需修配方**：bpb 偏差 >+0.05 或 loss 异常 → 查 tokenizer/数据/LR/head 配置，P1 前必须修复。
- **中止**：NaN 反复、吞吐 <10k（5090）且无廉价解释。
- 产出：`runs/r93/p0b/report.{md,json}` + 曲线；同时记录实际 N_active、fwd_flops、MFU。

## 5. 存储/盘位预算（50GB 数据盘）

| P1 实例（<instance-id>） | GB |
|---|---:|
| 训练语料 `p2_climbmix_40k`（u32） | 23.2 |
| eval 面板 `p0_eval_40k` | ~0.25 |
| KD 数据集（待 kd-ops 定稿） | ≤22 |
| core_bundle/杂项/日志 | ~1 |
| **合计** | **≤46.5**（留 ≥3.5GB） |

旧的 `p2_*_32k`、`p5_data`、models/merged/adapters、calib_* 等在本轮清理；`p2_climbmix_129k`
（2.9GB）在 KD 定稿前保留，必要时删除。P0b 子语料与 eval 面板建在 GPU 实例上，不占此盘。

## 6. 预算与实例

| 项 | 预算 |
|---|---|
| 无卡实例（2 台 × ≤8h × ¥0.1/h） | ~¥1.6 |
| P0b GPU（5090 优先：~5–6h 含数据准备；或 4090 降级 ~7–8h） | ~¥13–20 |
| 合计 | **~¥15–22** |
| P1（后续，需充值） | 5.8B tokens @ ¥<redacted>（5090 35h 级） |

当前余额 **¥<redacted>**。安全规则：不动用户实例 `<instance-id>`（运行中）、`<instance-id>`、`<instance-id>`；
自己启动的实例用完关机/释放并记账；GPU 实例一律设自动关机 TTL。

## 7. 风险与对策

- **5090 无容量**：0 台可租（21 台自有 5090 全部关机且机器满卡）→ 后台每分钟轮询，60min 内出现即占用；
  超时降级 4090（CUDA 13.0 宿主机，bf16，NO_FP8=1），P0b token 预算降至 ~0.4B。
- **P0b 偏差**：bpb 线来自生产 MoE 两个锚点（±2×）；gate 用 +0.05 绝对容差，偏差先归因（LR/head/数据）再考虑改配方。
- **KD 盘位**：语料 23.2GB 后只剩 ~25GB；KD 超 22GB 则减覆盖或减 token；KD 非 P1 阻塞项（可先 hard-CE 启动）。
- **词表迁移 bug**：eval harness 要求 token_bytes 长度 == 模型 vocab（41,280）；P0b 前必须跑通 eval_t0.py。
- **单卡长跑可靠性**：P1 需 200–500M tokens 级 checkpoint/resume；P0b 阶段先验证 save/resume 链路。

## 8. 里程碑

1. 无卡实例清理完成 + 下载速率确认（T+1h）
2. p0_eval_40k 建好 + P0b 数据/配置就绪（T+2h）
3. `p2_climbmix_40k` 完成（T+4–6h）
4. KD 规范批准 → 数据集完成（T+6–8h）
5. GPU 出现 → P0b smoke（20M）→ 0.6B run → 分析（T+6–10h）
6. Lead 汇报 + P1 go/no-go（当日）
