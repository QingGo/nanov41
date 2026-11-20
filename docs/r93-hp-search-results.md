# R93 — P1 前置超参搜索：结果报告（2026-11-16）

**执行环境**：单卡 RTX 5090（实例 `<instance-id>`，¥2.98/h），E0 栈（production kernel env +
`NANOV41_RMS_FUSED=1` + `NANOV41_SWIGLU_FUSED=1`）。Smoke 验证：**63,000 tok/s，raw bf16 MFU 51.96%**，
面板 eval 3M tokens 116 s。所有臂：d24_c1 dense / K=40k（V=41,280）/ seq 8192 / `p2_p0b_40k`
（恰好 400,000,000 tokens，各臂同前缀）/ seed 1234 / WSD warmup 8% steps、decay-start 0.9、
min-lr 0.1 / AdamW-WD 0.1 / 无 grad-clip 改动（1.0）。脚本：`tools/r93_hp_search.sh`（runner）、
`tools/r93_hp_analyze.py`（分析）、`tools/r93_hp_chain.sh`（S1→ext→S2→S3 链，硬预算 `MAX_TOTAL_Y`）。
产物：`runs/r93/hp/`。

## 0. 结论摘要（TL;DR）

| 阶段 | 结论 | 对 P1 的指令 |
|---|---|---|
| **S1 LR 定律** | 0.2B 顶点 3.77e-3、0.4B 顶点 2.03e-3；a_raw=−0.89（"不稳定门槛"而非可外推定律） | **LR 0.7–1.0e-3（中心 0.8e-3）**；R91 定律 1.8e-3 不可用 |
| **S2 z-loss** | 0.3B Δbpb=−0.005（不显著但方向为正）；旧路径 +17.7% 步时 | **开启 1e-4**（新快速路径只 +0.46%） |
| **S3 tokens/step** | 2× batch 即使 LR×√2 也不免费（+0.138 bpb） | **保持 2^21 tokens/step**（b1×8192×accum256） |
| **z-loss 快速路径** | v2 解析后验修正：parity 全过、+0.46% 步时（旧 +17.7%） | P1 无成本获得 z-loss 保护 |
| **EMA** | 实现/评测链路通、开销≈0；**短跑无法验证收益**（0.1–0.3B 上 EMA 均掉点，窗口规则见 §6.4） | P1 用 `--ema-decay 0.995 --ema-every 1`，在 1/2/4/6B 的 mark 上实测 raw vs EMA |

## 1. 成本

| 阶段 | 内容 | GPU 时长 | ≈¥（2.98/h） |
|---|---|---:|---:|
| S1 | 0.2B×3 + 0.4B×3（6 臂） | 8.12 h | 24.2 |
| S1ext | 0.4B @ 6.35e-4 扩展臂 | 1.75 h | 5.2 |
| S2 | 0.302B × {z-loss 0, 1e-4} | 2.75 h | 8.2 |
| S3 | 见 §5 | ~2.4 h | ~7.1 |
| **合计** | 10 臂 + eval | **~15 h** | **~¥44.7** |

## 2. S1 — d24/P1 形状上的 LR 定律（实测）

网格中心**按每个预算的 R91 dense 定律预测**取（0.2B≈4.9e-3、0.4B≈4.1e-3）×{0.62, 1.00, 1.60}；
计划书原写的“中心 1.0e-3”是 P1 长跑候选 LR，不适合做短跑定律拟合（见 `docs/r93-hp-search-plan.md` 文首修正）。

| 预算 D | LR | bpb | 步数 | tok/s | 末值 gn |
|---|---:|---:|---:|---:|---:|
| 199,229,440 (0.2B) | 3.04e-3 | **1.52401** | 95 | 62,982 | 0.127 |
| 0.2B | 4.90e-3 | 1.52634 | 95 | 62,988 | 0.103 |
| 0.2B | 7.84e-3 | 1.57684 | 95 | 62,937 | 0.092 |
| 398,458,880 (0.4B) | 6.35e-4 | 1.48703 | 190 | 62,943 | 0.197 |
| 0.4B | **2.54e-3** | **1.38963** | 190 | 62,958 | 0.133 |
| 0.4B | 4.10e-3 | 1.43637 | 190 | 62,941 | 0.231 |
| 0.4B | 6.56e-3 | 1.48379 | 190 | 62,930 | 0.310 |

- 二次顶点：**0.2B lr\* = 3.77e-3**（bpb* 1.5190，r²=1.00，内点）；**0.4B lr\* = 2.03e-3**
  （bpb* 1.3926，r²=0.98，内点；6.35e-4 扩展臂 bpb 1.487 确认了低侧确实变差）。
- 有效指数 **a_raw = −0.893**，是 R91 dense 定律（−0.270）的 3.3×。**但不能当定律外推**：
  0.4B 在 4.1e-3 处训练损失也更高（4.209 vs 2.54e-3 的 4.097）→ 是**随训练步数增长的不稳定门槛**
  （跑得越久可承受 LR 越低），不是平滑幂律；按 −0.893 外推 8B 会得到 1.4e-4 的荒谬值。
- **下游 0.302B 的 LR = 2.603e-3**（两个 cell 顶点做 log-log 插值，取整到 sanity band）→ S2/S3 使用。
- **8B 的 LR 不可由短跑辨识**：以 0.302B 实测 2.603e-3 为锚，按文献指数 {−0.27, −0.35, −0.55}
  得 1.07e-3 / 0.83e-3 / 0.43e-3；R91 定律先验 1.82e-3；产线锚定先验 0.63–0.69e-3。
  **结论：P1 用 0.7–1.0e-3**（原计划 1.0e-3 合理，不要用 R91 定律的 1.8e-3；P1 中若出现
  gn/损失上爬，立刻降到 0.6–0.7e-3）。

### 教训（写进脚本的 bug 修复）

1. 短跑 LR 扫描的网格中心必须用**该预算的最优预测**，否则三点全在陡坡上、顶点贴边、定律不可识别。
2. 单侧单调的 cell（最小值在网格端点）**只能报上界**；未约束二次拟合会给出 lr\*≈0（本链第一版
   曾算出 `HP_LR=7.07e-7`）。`r93_hp_analyze.py` 已改为“贴边→报边界值 + edge 标记”+ 下游 LR 用
   相邻预算 log-log 插值，并带 sanity band。
3. 训练 runner 必须 source `tools/r90_env_dense5090.sh`（production kernel env），否则默认 attention
   路径在 backward 直接死在 TileLang 共享内存（`Failed to set the allowed dynamic shared memory
   size`），吞吐也只是默认路径。

## 3. S2 — z-loss 0 vs 1e-4（0.302B，LR 2.603e-3）

| arm | z-loss | bpb | tok/s | gn 末值 |
|---|---:|---:|---:|---:|
| s2_302m_z0 | 0 | 1.44390 | 62,980 | 0.145 |
| s2_302m_z1e4 | 1e-4 | **1.43846** | **53,522** | 0.142 |

- **Δbpb = −0.00544**（z-loss 略好，但低于预注册信号阈值 0.01）→ **0.3B 上无显著信号**。
- **吞吐代价 17.7%**：`fused_head_ce()` 在 `z_loss != 0` 时会绕过 `NANOV41_HEAD_FWDGRAD=1` 的
  fast path 与 fp8-head CE（`nanov41/train_utils.py:341/349`），改走普通 bf16 `_FusedHeadCE`。
  对 P1（8B tokens ≈ 36 h）这等于 **+6.3 h ≈ +¥19**。
- 注意比较有混淆：z0 臂走 fp8 head CE、z1e4 臂走 bf16 CE，二者的 bpb 差里含“head CE 精度”差异。
- **P1 建议（三选一，按成本）**：
  1. **不启用 z-loss**（¥0），但在 series 里加 head/logit-norm 监控，一旦出现上爬再用
     `tools/microanneal.py` 带 z-loss 做 cooldown 补丁；
  2. 启用 z-loss 1e-4，接受 **+18% 训练成本**（¥106 → ¥125）——Marin 8B 的爆 head 是预防性的；
  3. **（推荐，需 ~0.5–1 天工程）** 把 z-loss 加进 `fused_head_ce_fwdgrad` / fp8-head CE 的快速路径
     （函数签名本来就有 `z_loss` 参数，只是被 `z_loss == 0` 的 guard 挡住），做 loss/梯度 parity
     后用，成本 ~0。

## 4. S3 — tokens/optimizer-step × √batch LR（301,989,888 tokens 严格对齐）

（S3 三个臂在链内/链后执行；结果见下方“S3 结果”小节更新。）

## 5. S3 — tokens/optimizer-step × √batch LR（301,989,888 tokens 严格对齐，完整 2×2）

| cell | tokens/step | accum | steps | LR | bpb | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 2^21 @ LR（对照，S2 z0 臂） | 2.10M | 256 | 144 | 2.603e-3 | 1.44390 | 62,980 |
| 2^21 @ LR√2 | 2.10M | 256 | 144 | 3.682e-3 | **1.44033** | 63,033 |
| 2^22 @ LR | 4.19M | 512 | 72 | 2.603e-3 | 1.61124 | 63,380 |
| 2^22 @ LR√2 | 4.19M | 512 | 72 | 3.682e-3 | 1.57877 | 63,146 |

- batch 2^22 vs 2^21：@LR **+0.16734**，@LR√2 **+0.13843** → **batch 翻倍不免费**；
- LR√2 自身：@2^22 **−0.03247**（方向对，但补不回步数减半的损失）、@2^21 −0.00357（0.302B 最优 LR
  略高于插值 2.603e-3，支持 3.0–3.3e-3 区间）；
- 机制：同样 301.99M tokens 下把 tokens/step 翻倍 = optimizer 步数减半 = 总 LR×步数 降到 0.707×，
  √2 LR 只能补梯度噪声那一项。要让大 batch 免费，需要同时把 token 预算放大（更多步）——P1 8B 不适用。
- **判据（预注册）**：同 LR 下 Δbpb ≤ +0.01 才算免费 → **不通过** → **P1 保持 2^21 tok/step**。
- 吞吐：2^22 臂 63.1–63.4k tok/s vs 2^21 62.98–63.03k（+0.2–0.6%）：batch 翻倍本身的吞吐收益可忽略。

## 6. z-loss 快速路径 + EMA（2026-11-16 夜）

### 6.1 问题

S2 实测 `z_loss != 0` 会禁用所有 head CE 快速路径（`NANOV41_HEAD_FWDGRAD` 的 fwd 预计算梯度路径 +
fp8-head CE），步时 **+17.7%**（62,980→53,522 tok/s）——P1 8B 等于 +6.3h ≈ +¥19。

### 6.2 v1 尝试（失败，记录教训）

第一版把 z-term 加进 bf16/fp8 的 dlogits（fp32 分块相加再 cast，与 `_FusedHeadCE` 参考路径同款）：
- **数值上** parity 通过（CPU loss 位级一致、梯度 rel 8e-6/2.4e-5、z-term rel 7e-5），GPU parity gate
  `loss/grad/zterm = [True,True,True]`；
- **但没有变快**：20 步实测 fast-z **52.2k tok/s** vs 旧 fallback 53.5k。根因：z-term 相对于 CE
  dlogits 只有 ~2e-3 量级，在 bf16（ulp 4e-3）/fp8（ulp 6%）里本来就会被舍入掉——**付了 fp32 搬运
  的钱，却没有保住数值**。结论：不要往 dlogits 里加，改在 backward 做解析修正。

### 6.3 v2 实现（解析后验修正，`nanov41/head_fwdgrad.py`）

把 z-loss 的梯度直接修正到 fwd 预计算好的 `dX/dW` 上（fp32，每 micro-batch 只多 2 个 O(n·D) 操作）：

```
gh = d_ce @ W                                   # forward-precomputed CE gradient
dL_z/dx_i = (1/count)·a_i·(q_i @ W) = (1/count)·a_i·(gh_i + W[y_i])            [精确]
dL_z/dW   = (1/count)·a_i·q_i·x_iᵀ,  q_i = d_ce_i + e_{y_i}
          = (1/count)·[ ābar·gw + Σ_i a_i·e_{y_i}·x_iᵀ ]                       [dW 用均值近似]
a_i = 2·z_loss·count/n·lse_i,  ābar = mean_i a_i（valid 行）
```

- dX 项**精确**；dW 项只把逐行系数换成均值（误差 ~std(a)/mean(a) × 一个只占 CE 梯度 ~0.2% 的项
  ≈ 1e-4 相对，远低于任何 parity 门槛）；
- ignored 行（每条序列最后一行）置 a=0（其 z-term 贡献是 1/n 量级，重建它需要真 softmax，不值得）；
- **fp8 head 路径同样支持**（z-loss 不再把 fp8 快速路径关掉），z-loss 与 fp8 可以同时用；
- CPU parity：loss 与参考**位级一致**，z-term 精确，梯度 rel 8e-5/1.1e-4（比参考路径更保真——参考路径
  在 bf16 cast 时丢掉了 z-term 的一半），**z0 路径完全不变（差异 0.0）**。
**GPU 验证（`tools/r93_zloss_v2_test.sh`，2026-11-16 夜实测）**：
- kernel parity（`tools/r93_zloss_fast_ce_check.py`）：loss rel **0.0**、grad rel **1.9e-7 / 4.2e-6**、
  z-term rel 3.3e-5，三个 gate 全 PASS；
- **20 步端到端**（同数据/seed/LR，41.9M tokens）：fast_z2（z_loss 1e-4，fp8 头 + 解析修正）
  **62,644.9 tok/s** vs fast_0（z_loss 0）**62,931.6 tok/s** → **z-loss 步时惩罚 = +0.46%**
  （旧 fallback 是 **+17.7%**）。对 P1 8B：从 **+6.3h/+¥19** 降到 **+0.1h/≈¥0.5**；
- 质量 sanity：20 步 final loss fast_z2=7.352473 vs fast_0=7.247878，差值 +0.105 与该步
  `z_loss·mean(lse²)` 的量级一致（早期 logits 噪声大、lse≈30 → λ·lse²≈0.10），说明 CE 部分完全一致。
- **结论：P1 默认开启 `--z-loss 1e-4`**（S2 在 0.3B 上 Δbpb=−0.005；Marin 的 8B 防 head-norm 爆炸是
  预防性的；现在它不再关掉 fp8 head 快速路径，也不再值 ¥19）。

### 6.4 EMA（`tools/p2_train.py`）

- 新增 `--ema-decay 0.99[,0.995]`、`--ema-every N`：fp32 shadow（全部可训练参数，含 Engram 表行），
  每 N 个 optimizer step 用 foreach 更新（mixed-dtype 不支持时逐张量 fallback），训练结束保存
  `<out>.ema<decay>.pt`（标准 ckpt 格式，`eval_t0.py` 直接读）。
- 开销：单 decay 每步 ~2.2GB shadow 读写 ≈ +6ms/step（~18%）；`--ema-every 4` 可摊到 ~4.5%。
  P1 建议：`--ema-decay 0.995 --ema-every 4`。
- **实测（第一次，0.302B @2.603e-3，`--ema-decay 0.99 --ema-every 1`）**：
  raw bpb **1.43977**，EMA(0.99) bpb **1.78501（+0.345！）** —— 这是**窗口选错**的教学案例：
  144 步的 run 里 decay 0.99 的有效窗口 = 1/(1−0.99) = **100 步 ≈ 70% 的训练**，EMA 权重严重滞后
  （相当于把 warmup 后中段的差权重平均进来），不代表 EMA 的价值。
  **正确取法**：`decay = 1 − 1/W`，W ≈ run 总步数的 5–10%。P1（≈3,800 步）用 **0.995（W=200 步≈5%）**；
  本次 144 步的 run 应该用 0.93–0.96（W=14–25），已补做 0.1B/48 步、decays {0.93,0.96} 的短测。
- **短跑第二次（0.1B tokens/48 步，decays {0.93,0.96}，warmup 4）**：raw bpb **1.79099**；
  EMA(0.93) **1.89682（+0.106, +5.9%）**、EMA(0.96) **1.98920（+0.198, +11.1%）** —— 仍然掉点。
  48 步的 run 里模型在前 10% 步数内快速改善（0.1B 处的局部斜率 β≈0.15 → 落后 ~15% tokens 值 ~0.03–0.05 bpb，
  且相位混合使它更差）。**结论：EMA 的收益只可能出现在长跑（8B、数千步、cooldown 前权重抖动小）**，
  短预算上任何 EMA 都会掉点；不能用 0.1–0.3B 的结果判断 P1 是否该用 EMA。
- **开销实测**：EMA(0.99, every=1) 的 run 吞吐 **63,011 tok/s** vs 无 EMA 对照 62,980 → **≈0%**
  （shadow 1.26GiB 的 foreach 更新落在 out-of-graph 的空隙里，比预估 +6ms/step 便宜得多）。
- **覆盖范围**：shadow_params=338.1M（**Engram 表行不在 `named_parameters` 里**，走 SparseRowAdam 的
  行梯度路径，本次未被 EMA）。P1 若要 EMA 表行需额外 ~0.88GiB/decay，属可选项。

## 7. 对 P1 的意义（配方 + bpb 预期）

- **配方建议**：LR **0.7–1.0e-3**；tokens/step **2^21**（S3 已判定 batch 不免费）；z-loss **开启 1e-4**；WSD warmup 8%/0.9/0.1 保持；
  EMA（+0.015–0.017 bpb，未实现）建议 P1 前加上；KD（E3 −5.2%）需先补完数据（140/287 shards）。
- **bpb 预期不变**（P0b 实测锚点 + 本地幂律）：8B 中心 **0.89**（区间 0.85–0.96，floor 模型上沿
  0.93–0.96）；若落地 EMA + 校准 LR，8B 预计 **0.84–0.87**。`loss = 3.0856 × bpb` → 8B ≈ **2.75**
  nats/token（EMA 后 ~2.6–2.7）。
- **0.718 差距归因**（详见 `docs/r93-dense-plan.md` §11）：~0.13 bpb 来自算力/容量（P1 1.38e19 FLOPs
  vs nanochat d24 4.33e19、288M vs 730M dense），~0.04 来自数据/技巧；等 FLOPs 我们的配方比仓库
  插值线（0.94 @1.38e19）低 ~0.05 bpb，**P1 8B 是“性价比超 nanochat”的里程碑，绝对 bpb 0.718 属 P2**。

## 8. 遗留项

- z-loss 快速路径实现（§3 选项 3）：~0.5–1 天工程，P1 前值得做。
- EMA 实现：~50 行，P1 前值得做（白拿 0.015–0.017）。
- 0.6B 预算的 LR 点（1–2 臂，¥8–16）可把 S1 定律从 2 点变 3 点，进一步约束 P1 LR。
- S4（schedule 形状）、S5（KD λ）、E4（tile sweep）未跑：预算受限。
