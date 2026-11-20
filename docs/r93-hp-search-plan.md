# R93 — P1 超参搜索现状与计划（2026-11-16）

> **2026-11-16 UPDATE（S1 网格修正 + 实测成本模型 + 已开跑）**
> 1. **计划里的 S1 网格是错的，已修正**：原文写 "LR 中心 1.0e-3 × {0.6,1.0,1.6}"，但 1.0e-3 是
>    **P1 长跑（8B）的候选 LR**，不是短预算最优。R91 dense 定律 `5.927e-3 @100.7M`、
>    `4.405e-3 @302M`，按 μP 宽度 `√(512/1024)` 与 `√batch` 换算到 d24/2^21 tok/step 后
>    不变（两个因子正好抵消）→ **0.2B 预算最优 ≈4.9e-3、0.4B ≈4.1e-3**。若用 0.6–1.6e-3 网格，
>    三点全落在陡峭左坡上（R91 在 100.7M 处 1.55e-3 vs 5.9e-3 的 bpb 差 +0.17），顶点必然贴边、
>    定律不可识别，白烧 ¥12。**正确做法：每个预算的网格中心 = R91 定律在该预算的预测值**，
>    ×{0.62, 1.00, 1.60}：0.2B → {3.04, 4.90, 7.84}e-3；0.4B → {2.54, 4.10, 6.56}e-3。
> 2. **成本用实测重算**（E0 栈 b1×8192×accum256 = **62,848 tok/s**，面板 eval 3M tokens =
>    **116 s**，均来自 `runs/r93/mfu45/final_{a,b}.*`）：0.2B 臂 ≈55 min≈¥2.75、
>    0.4B 臂 ≈108 min≈¥5.4、0.302B 臂 ≈82.5 min≈¥4.1。**S1 6 臂≈8.2h≈¥24.4**（原估 ¥18 偏低）、
>    **S2 2 臂≈2.75h≈¥8.2**、**S3 3 臂≈4.1h≈¥12.3**（S3 用 S2 的 z0 臂当 2^21@LR 对照，省 1 臂）。
>    合计 **≈15.1h≈¥45**；余额 ¥<redacted> → 缓冲仅 ~¥<redacted>，**只够 1 个扩展臂（0.2B）**；若 0.4B
>    cell 贴边需要扩展，就先砍 S3 的第 3 臂（¥4.1）再补。
> 3. **执行**：`tools/r93_hp_search.sh`（STAGE=s1/s1ext/s2/s3，resume-safe、带 wall 预算拦截）+
>    `tools/r93_hp_analyze.py`（复用 R91 的二次顶点法 + 边缘检测 + 配对 Δbpb 判据）。
>    所有臂同数据前缀（`p2_p0b_40k` 恰好 400,000,000 tokens）、同 seed 1234、WSD
>    warmup=8% steps/decay-start 0.9/min-lr 0.1、面板 eval 3M tokens 同口径；数据覆盖
>    0.4B 臂（398.5M<400M）。
> 4. **结果已出**：见 `docs/r93-hp-search-results.md`（S1 定律、S2 z-loss、S3 batch 的实测表与 P1 配方结论）。
> 5. **S2/S3 的新设计**：S2 的两臂在 0.302B（=144 步×2^21）跑 z-loss 0 vs 1e-4，**z0 臂同时是
>    S3 的 2^21@LR 对照**；S3 只需 3 个新臂（2^22@LR、2^22@LR√2、2^21@LR√2），token 数严格相等
>    （301,989,888），Δbpb 判据 ≤+0.01 = batch 免费。


问题：LR、batch size 等超参我们到底搜过哪些？对照 Marin（`docs/marin-data-lessons.md`）还缺哪些
重要超参？P1（8B tokens，d24_c1，K=40k）开跑前应该怎么补搜？

## 1. 已搜 / 未搜清单（真实现状）

| 超参 | 当前值（P1 计划） | 怎么定的 | 是否搜过 |
|---|---|---|---|
| **峰值 LR** | **1.0e-3**（区间 0.6–2.0e-3） | R91 C 层实测的 **dense LR 定律**（d12-aligned，dim512，1.05M tok/step，24 arms，¥29.3）：`lr* = 10^-2.733·D^-0.270`（dense；moe −0.79；mean −0.53），再乘 μP 宽度 `√(512/dim)` 与 `√(batch/1.05M)` | **间接搜过**：定律在 d12 形状实测；**d24/P1 形状从未直接扫**（2.8× 不确定度） |
| tokens/step（等效 batch） | **2^21 = 2.10M**（b1×8192×accum256；b2 时 b2×8192×accum128） | P0b 沿用；与 R91 搜索批 1.05M 不同 | **只搜过吞吐**（b1/b2/b3/accum64 等工程实验），**没有质量维度的 batch 搜索** |
| micro-batch / seq | b1、seq 8192 | 显存 + 与 P0b 可比 | 未搜（b2 昨天有质量弱证据 +2.9% 吞吐） |
| LR schedule | WSD，warmup 30 步（≈21% of P0b run，P1 相对更短）、decay-start 0.9、min-lr 0.1 | R91 协议 + P0b | 未在 P1 搜（decay 形状/min-LR 未扫） |
| weight decay | 0.1（Muon + AdamW，decoupled） | 官方 V4.1 报告 | 未搜 |
| β1 / β2 / eps | 0.9 / 0.95 / **1e-20** | 官方 | 未搜（eps 1e-20 是官方值，bf16/fp8 下是否稳健未验） |
| grad clip | 1.0 | 本仓库默认 | 未搜；Marin 是**完全关掉 clip**（靠 z-loss+LR） |
| **z-loss** | **0（关闭）** | 沿用 runner 默认 | **未搜，且 Marin 明确说它是刚需 1e-4**（见 §2） |
| Muon 细节 | update-RMS 0.18、Nesterov 0.95、NS 5 iters、head-wise Q | 官方 | 只搜过**速度**（R78/R79），质量未搜 |
| Sinkhorn（Engram） | γ 0.18、K=11、τ 1e-3、ε 1e-20、Engram LR ×5 | 官方 | 未搜 |
| Engram 表大小/层位 | 220M、layers [1,4] | R91 B 层搜过 table 大小（R91: dense+eng815 1.3028 / eng440 1.3044 / eng220 1.3110，单调饱和） | 分配搜过；P1 固定 220M 可复议 |
| 架构 | d24_c1 dense | R91 A 层（8 arms）+ R93 P0b | 已搜 |
| 层模式（Full/Reuse/Reindex） | 1 group per half（官方式变体被 A/B 拒绝） | R93 A/B（2026-11-16） | 已搜 |
| 词表/分词器 | K=40k（V=41,280） | R90/R92 多轮 | 已搜 |
| 数据配方 | ClimbMix 纯 web | R90/R92 数据阶段（DCLM/FineWeb/mix） | 阶段搜过；P1 固定 |
| KD 权重 λ（若 P1 用 KD） | **未定** | E3 pilot（−5.2% bpb） | **未定/未搜**，P1 若开 KD 必须先定 |
| EMA 评测 | 未实现 | —— | 未做；Marin WSD+EMA 白拿 ~0.015–0.017 BPB |

结论：**只有 LR 有"定律级"的搜索（还是在 d12 代理上）；batch 只有工程实验；z-loss 甚至没开。**
P1 的 8B 训练预期花费 ≈¥106（E0 栈 62.8k tok/s），一次训练 Δbpb 0.03 ≈ 值 ¥30–60，
所以花 ¥25–35 在 P1 前补搜 LR/batch/z-loss 是明显正期望。

## 2. Marin 给的重点（`docs/marin-data-lessons.md`，全部有出处）

1. **z-loss 1e-4 是刚需**（Marin 8B Retro §2.1）：lm_head 范数爆炸 → loss 缓慢上爬，换优化器/wd
   都无解，logit z-loss 修好；"从零训练也该用"。我们的 E5B 在 LR 1.5e-3 发散（gn 1→8、loss 上爬）
   正是同款症状。**我们的 fused CE 已实现 z_loss 参数（`nanov41/train_utils.py`），但 P0b/P1 runner
   都传 `--z-loss 0`** → 改成 1e-4 并与 LR 一起重扫（Marin 因此敢关 grad clip）。
2. **大 batch 是被验证的道路，我们 batch 偏小**（§2.7）：Marin 8B 从 4M → 16M tokens/step（按 √batch
   提 LR），535B 46M；我们 2.1M tokens/step → 比 Marin 8B 小 2–8×。梯度累积把 tokens/step 提到
   4–8M 不增加 FLOPs（只改 accum），需要按 √batch 重扫 LR。
3. **LR 随 token 预算下降**：Marin 实测 `lr* ∝ tokens^-0.346`（R91 已引用）；我们的 dense 拟合
   a=−0.270（2 点/FFN，低置信）。含义：**短跑最优 LR 明显偏高，P1 长跑 LR 必须按定律外推**，
   不能直接用 0.3B 扫描的赢家。
4. **WSD + EMA**（§2.6）：Marin 用 EMA(β=0.995) 评测，稳定 ~0.015–0.017 BPB 收益；我们 WSD 已有，
   EMA 未实现（评估时用 EMA 权重即可，成本是显存里多一份权重）。
5. **cooldown/microannealing、mix 切换要平滑过渡 + LR rewarm**（§2.2–2.4）：P1 是单阶段 ClimbMix，
   暂不涉及；若 P1 加 KD cooldown 阶段则要设计（20% tokens 的 cooldown + LR rewarm）。
6. **linear decay 到 0.05、无 grad clip**（Marin 535B 配方，§1 表）：与我们 WSD min-lr 0.1 + clip 1.0
   不同，可作为一个 schedule 臂。

## 3. P1 前建议的补搜计划（按性价比；成本按 E0 栈 63k tok/s、¥3.0/h）

单臂成本（**实测口径 2026-11-16**：E0 栈 62,848 tok/s、面板 eval 3M tokens 116 s、¥2.98/h）：
0.2B≈55 min≈**¥2.75**；0.302B≈82.5 min≈**¥4.1**；0.4B≈108 min≈**¥5.4**；0.6B≈2.7h≈**¥8.2**。

### S1（必做，最高优先级）— 在 d24/P1 形状直接测 LR 定律：2 预算 × 3 LR = 6 arms
- 预算 {0.2B, 0.4B}（95 / 190 步 × 2^21 tok/step），每预算 3 LR、**网格中心 = R91 dense 定律
  在该预算的预测**（不是 P1 的 1.0e-3；见文首 UPDATE）：0.2B → {3.04, 4.90, 7.84}e-3，
  0.4B → {2.54, 4.10, 6.56}e-3（= 中心 ×{0.62, 1.00, 1.60}）；WSD warmup=8% steps/0.9/0.1，
  seed 1234，数据 `p2_p0b_40k` 前缀（各臂可比）。
- 拟合 `log10 lr* = c + a·log10(D/D_ref)`（a 与 R91 dense −0.270 对比），外推 0.302B 与 8B →
  P1 LR 及其置信区间；同时给出下一阶段需要的 `HP_LR`。
- 成本 **≈8.2h≈¥24.4**；判据：两个 cell 的顶点都在网格内部（`edge=none`）；贴边则用
  `STAGE=s1ext S1X_ARMS="<budget>:<lr>..."` 补 1 臂（0.2B ¥2.75 / 0.4B ¥5.4），扩展后重算。
- `tools/r93_hp_analyze.py` 复用 R91 的二次顶点法（`r91_analyze_c.py` 同款统计），只是换 d24 结果目录。

### S2（必做）— z-loss：2 arms（0.302B，z0 臂同时是 S3 的 2^21@LR 对照）
- 在 S1 定律给出的 0.302B LR（`HP_LR`）上跑 `--z-loss 0` vs `1e-4`，各 301,989,888 tokens
  （144 步 × 2^21），≈82.5 min/臂 ≈ **¥8.2**；顺带看 gn 曲线；若 z-loss 允许更高 LR，用
  `S2_BOOST=1` 加 1 臂（HP_LR×1.6，¥4.1）。判据 |Δbpb|≥0.01 才算信号（bpb 更小 = 更好）。

### S3（高价值）— tokens/step × √batch LR：3 个新臂（第 4 格用 S2 的 z0 对照）
- 固定 301,989,888 tokens：{2^22 @ HP_LR（72 步×accum512）、2^22 @ HP_LR·√2、
  2^21 @ HP_LR·√2（144 步）} + S2 的 {2^21 @ HP_LR} 对照 = 完整 2×2；≈4.1h ≈ **¥12.3**。
- 判据（预先写死）：同 LR 下 `bpb(2^22) − bpb(2^21) ≤ +0.01` 且 √2 LR 不劣 → batch 可免费翻倍；
  若正向，P1 用 2^22 tok/step（b1×8192×accum512）并把 LR 相应 ×√2。
- 若 S1 需要扩展臂吃掉预算，先砍 `2^21 @ HP_LR·√2`（¥4.1），保留 2 个 2^22 臂 + z0 对照。

### S4（可延后/可选）— schedule 形状：3 arms
- decay-start {0.8, 0.9}、min-lr {0.05, 0.1}、可选 Marin 式 linear→0.05，0.3B/arm ≈ **¥12**；
  预期收益 0.005–0.02 bpb。

### S5（若 P1 用 KD）— KD λ：2–3 arms（0.2B）
- λ {0.3, 0.5, 0.8} + pure-CE 对照，≈¥8–12；与 bpb 一起看（KD 目标是降 bpb/等质量省 token）。

**总预算（2026-11-16 实测口径）**：S1 **¥24.4** + S2 **¥8.2** + S3(3 臂) **¥12.3** ≈ **¥45**
（另加开机/sync/smoke ~¥0.8）；余额 **¥<redacted>** → 缓冲 ~¥<redacted>，只够 1 个 0.2B 扩展臂。
P1（8B ≈¥106）仍需另充值 ~¥<redacted>。

## 4. 执行注意（R91 的教训）

- **短跑 LR 不能直接用于长跑**：R91 实测短跑最优比长跑高 4–8×；必须按 a≈0.27–0.35 外推到 8B，
  再用 2–3 个点验证外推（比如在 0.6B 上验证预测 LR，1 臂 ¥8）。
- **同签名比较**：每臂必须同 data/seed/signature，否则结果作废（R91 signature 机制）。
- **搜索必须用 E0 栈**（63k tok/s）否则成本 ×1.4；并固定 micro 形状（b1 或 b2）使臂间可比。
- **判据要预先写死**：bpb 差值 ≥0.01 才算真信号（我们 0.1B/50 步的 A/B 噪声就在 ±0.01 量级），
  所以 LR 扫描用 ≥0.2B 臂、每个 cell ≥3 点取顶点。
- **z-loss 可能与 LR 耦合**：Marin 是先加 z-loss 再重扫 LR；我们两个都还没做，顺序上先 S1 再 S2，
  若 S2 显示可以更高 LR，则回到 S1 格点补两臂。

## 5. 其他"值得但不在本轮"的项

- **EMA 评测**（Marin 白拿 0.015–0.017 BPB）：实现 ~50 行 + 评估同口径，建议 P1 直接带上（不占搜索预算）。
- **Muon update-RMS / NS 步数 / head-wise**：只影响优化器效率/稳定性，8B 规模下收益预计 <0.01 bpb；
  NS 5→3 可省 optimizer 时间（~1–2% 墙钟），质量需单独 A/B。
- **init（hc_scale/base、embed/head std）**、**β2/eps**、**attention softmax scale**：低优先级，
  官方值在中等规模一般稳健；只有在 S1/S2 显示异常（如发散、gn 异常）时才动。
- **seq length**：8192 → 16K/32K 是能力/长上下文问题，不是 P1 cost-loss 主指标，暂不搜。
