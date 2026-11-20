# R93 — MFU 45% 的口径、FP8 对齐与达标路线（2026-11-16）

> **UPDATE 2026-11-16 — sprint executed; 45% target exceeded.** E0 (the two existing fused kernels
> missing from the production env) alone gives **raw MFU 36.2% → 51.8%** (+43% tok/s), b2 adds
> →53.4%, and fp8 FFN is speed-validated (−11.4 ms/replay) but quality-unverified and OOMs together
> with b2. See **`docs/r93-mfu45-sprint.md`** (final results) and `runs/r93/mfu45/` for the data.
> P1 stack should be E0 (+b2 guarded, bpb-monitored); body fp8 stays OFF until its own 0.3B bpb A/B.

**问题**：把单卡 5090 的 MFU 优化到 ≥45% —— 这个 45% 是 bf16 口径，还是 precision-discount 口径？
用 FP8 对齐官方（DeepSeek-V4.1-Flash 预训练）是否能帮忙？
**结论先说**：
1. 此前所有数字（P0b 36.56% / 复现 37.17%）都是 **bf16 口径（raw）**：`F_model / (t × 209.5 TF)`；
   **同一份实测在官方 precision-discount 口径下是 33.88% / 34.44%**（只因为 head 是 fp8，被折 0.5）。
2. **45% 只能按 bf16 口径（raw）声明**。precision-discount 口径下，FP8 主体必降分：
   α(fp8)=0.5 与 5090 的 2× 峰值比一致，只有当 fp8 GEMM 达到 bf16 实测吞吐的 **>2×（≥364 TF/s = fp8 峰值的 87%）**
   时才不降分；我们实测端到端只有 1.26–1.40×（230–290 TF/s）→ **开 body fp8 后官方口径从 34.4% 掉到 ~25–28%**，
   而 raw 从 37.2% 升到 ~39–40%。这是记账规则决定的，不是 kernel 不够快。
3. **raw ≥45% 可达但需要三件套**（t ≤ 150.1 ms/replay = 54,580 tok/s，+21%）：fp8 FFN（−8~−14 ms）
   + 已存在的 RMS/SwiGLU 融合（−9~−17 ms）+ b2 解锁（+3~5 pp）。前两件中央合计只到 **43.2%**（乐观 44.7%），
   **单靠 fp8+融合过不了 45%**；叠 b2/其余 EW 融合后中央 **45–47%**、上端 ~51%，把握 ~60–65%。
   fp8 本身贡献 +2~3.2 pp，主杠杆是非 GEMM（113 ms）与 b2。
4. 若一定要官方口径 ≥45%：**不能开 body fp8**，只能走 bf16 主体 + 全量融合（那时 raw≈disc，上限 ~47%），
   与"对齐官方 dtype"目标冲突 → 需要你拍板 KPI（§6）。
5. 可复现：`tools/r93_mfu45_scenarios.py` + `runs/r93/mfu45_scenarios.json`（本文件所有数字由脚本生成）；
   fp8 细节见 `docs/r93-fp8-d24-memo.md`，官方架构对照见 `docs/v41-flash-arch-diff.md`。

---

## 1. 三种口径的定义与今日实测值

`F_model = 3 × 575,953,504 × 8192 = 14.1546 TFLOP/replay`；实测 replay **181.76 ms**（45,070 tok/s），
train loop **182.4 ms**（44,334 tok/s）；峰值：bf16 209.5 / fp8 419 / fp4 838 TF（repo 标准，`tools/bench_graph.py`）。

| 口径 | 公式 | 当前值（replay / train） | 说明 |
|---|---|---:|---|
| **raw（bf16 口径）** | `F_model / (t·209.5TF)` | **37.17% / 36.56%** | 我们历来报的数；与 tok/s 严格成正比；**精度混合不同则 raw 上限不同**（纯 bf16=100%，纯 fp8=200%） |
| **disc（官方口径）** | `Σα_p F_p / (t·209.5TF)`，α = bf16 1 / fp8 0.5 / fp4 0.25 | **34.44% / 33.88%** | V4.1-Flash 报告 Fig.2 caption 的折算（`tools/mfu_accounting.py`）；等价于"该混合精度的理想时间 / 实际时间"，上限恒 100% |
| **fp8 峰值口径** | `F_model / (t·419TF)` | **18.6% / 18.3%** | 若把分母换成 fp8 峰值；不建议用（跨精度不可比） |

当前 precision mix：head fp8（14.7% FLOPs）+ body bf16（85.3%）→ disc = raw × **0.927**。
`docs/m6-mfu-roadmap.md` §0 已记录 2026-11-10 的决定：**tok/s 北极星、raw 作跨配置口径、disc 降为辅助**——
本文件把这条决定落到 R93 的 45% 目标上。

## 2. "45%" 在三种口径下各需要多少时间（决定性的算术）

| 声明口径 | 45% ⇔ replay 时间 | 相对现在要省 | 可达性 |
|---|---:|---:|---|
| raw（bf16 口径） | **≤150.1 ms**（54,580 tok/s） | −31.7 ms | **可达**（§4） |
| disc（官方口径，保持当前 head-fp8 mix） | ≤139.1 ms | −42.7 ms | 难（≈把非 GEMM 砍 41%） |
| disc（官方口径，全 bf16 含 head） | ≤150.1 ms | −31.7 ms + head bf16 的 +8.6 ms | 可达但必须**放弃 body fp8** |
| disc（官方口径，body 全 fp8） | ≤**75.1 ms** | −106.6 ms | **物理不可达**（非 GEMM 现在就有 113 ms，极限融合后仍 ≥60 ms） |
| fp8 峰值口径 | ≤75.1 ms | −106.6 ms | 不可达 |

> 参照：42% raw ⇔ 160.9 ms（省 21 ms）；50% raw ⇔ 135.1 ms（省 46.7 ms）。

## 3. FP8 的账：墙钟赚，官方口径亏

### 3.1 为什么（闭式条件）
把 FLOPs 占比 q 的部分从 bf16 换成 fp8：`disc_new/disc_old = (1 − 0.5q)·(t_old/t_new)`。
- 要 disc 不降，必须 `t_new/t_old ≤ 1 − 0.5q`；即 **fp8 那部分的加速倍数必须 ≥2×**（与峰值比 2× 完全对应）。
- 我们实测（R82/R83，graph-timed，含量化开销）：wo_b 1.26×、engram_wkv 1.28×、dense-FFN twin 1.08×、
  **full-V4.1 形状 1.40–1.41×**；fwd 单项 1.69–1.84×，dgrad 1.73–1.84×，**wgrad 仍 bf16**（fp8 wgrad R84/R85 gate 拒绝）。
  → 端到端达不到 2×，**只要 q 大，disc 必降**。
- 结论：**在 5090 上 FP8 是 tok/s、¥/token、dtype 对齐的杠杆，不是 MFU 口径的杠杆。**

### 3.2 数字（脚本输出）
| 场景 | replay ms | tok/s | raw | disc | fp8 FLOPs 占比 |
|---|---:|---:|---:|---:|---:|
| S0 基线（P0b 实测） | 181.8 | 45,070 | **37.17%** | **34.44%** | 15% |
| S2 fp8 dense-FFN only（端到端 1.35×，中央） | 169.4 | 48,365 | 39.89% | **25.20%** | 74% |
| S2hi fp8 FFN（1.40×，乐观） | 168.2 | 48,713 | 40.18% | 25.38% | 74% |
| S3 fp8 FFN + RMS/SwiGLU 融合（中央） | 156.4 | 52,385 | 43.20% | 27.29% | 74% |
| S3hi fp8 FFN + 融合（乐观） | 151.2 | 54,191 | **44.69%** | 28.23% | 74% |
| S4 S3hi + 其余 EW/mHC 融合 + tile sweep（全乐观） | 133.2 | 61,516 | **50.74%** | 32.05% | 74% |

fp8 在 FFN 上的单笔时间收益（GEMM 桶 68.7 ms 中 FFN 占 66.8% = 45.9 ms）：
**1.20× → −7.6 ms；1.35× → −12.4 ms；1.40× → −13.6 ms**（含 engram.wkv；wo_b 已在产线 scope 内，不重复计），
按 raw 口径折 **+1.9~3.2 pp**（与 `docs/r93-fp8-d24-memo.md` §5 的 −12.4/−15.8/−18.2 ms 一致，
后者含已生效的 wo_b 切片）。
注意 d24 的 K=1024 比 full-V4.1 的 K=5120 短，fp8 收益会比 1.40× 低——中央取 1.26× 是保守的。

## 4. 达标路线（推荐顺序：先融合、后 fp8、再 b2）

**主要杠杆（按确定性/金额）**

| # | 杠杆 | 依据 | Δms | Δraw | 工期/GPU | 数值/风险 |
|--:|---|---|---:|---:|---|---|
| 1 | **RMS_FUSED + SWIGLU_FUSED**（代码已存在，产线 env 忘开） | R45 d12 已逐位验证；d24 未测 | −9~−17 | +2.3~4.3 pp | 0.5 h GPU | fp32 内部，语义不变；R83 的 +0.66% 慢是 fp8-producer 变体，非本项 |
| 2 | **fp8 dense FFN**（`LINEAR_FP8_V2_SCOPE` 扩到 FFN w1/w3/w2） | R82 实测 1.26–1.41×；d24 形状待测 | −7.1~−12.2 | +1.5~3.3 pp | 1 h micro + 2–4 h 集成 | wgrad 留 bf16；loss rel 目标 <1e-5；官方口径 −5~8 pp（预期内） |
| 3 | **b2×8192 解锁**（FFN 激活重算 或 bf16 中间量） | MoE b1→b2 +5.8 pp；dense 估计 +3~5 pp | −10~−17（等效） | +3~5 pp | 1–2 天 + 3–4 h GPU | 差 ~8–10 GiB；CE_CHUNK 无效；难点是 capture+fp8 wgrad 兼容 |
| 4 | 其余 EW/mHC 融合（sinkhorn wrapper muls、compare/clamp、copy、mHC v5） | 静态归因（`runs/r93/p0b/mfu_profile.md` §2b） | −6~−14 | +1.5~3.5 pp | 1–2 天 + 2 h | 收益分散；需 `--profile-eager 3` 先定位 476 muls/414 copies |
| 5 | d24 GEMM tile 重调（旧 tile 是给 d12 dim512 的） | `docs/r93-mfu-decomposition.md` §3B | −0~−4 | 0~1 pp | 1 h | 逐位等价 |

**组合（脚本场景）**
- 路线 A（fp8 对齐）：1+2 中央 → **43.2% raw / disc 27.3%**；1+2 乐观 → 44.7% / 28.2%；
  **单靠 1+2 到不了 45%**；再叠 3（b2）或 4 → **46~51% raw / disc 28~32%**。
- 路线 B（bf16 纯度，不开 body fp8）：1+4（上界）→ **46.0% raw = disc 42.7%**；再加 b2 → 50.9% raw / disc 47.2%。
- 结论：**两条路线都能让 raw ≥45%；差别只在官方口径（A ≈29–31%，B ≈43–47%）与 tok/s（A 更快 3–8%）**。

**成本**：P1 8B tokens @45% raw（t=150.1 ms → 54,580 tok/s）≈ 40.7 h ≈ **¥122**（¥3/h）；
@42.2%（S3）≈ 43.4 h ≈ ¥130；@36.6% 基线 ≈ 50.1 h ≈ ¥150；10B 全部 ×1.25。即 **45% 让 8B 的 P1 省 ~¥28**。

## 5. 与官方 dtype 对齐：真相表（不要照抄"官方 MoE 是 fp8"）

依据：`docs/v41-report-precision-facts.md`（V4.1 报告 arXiv:2609.19969 全文核对 + V4 报告 arXiv:2606.19348 §5.2.1）。

| 位置 | 官方（报告原文/阶段） | 我们现在 | 对齐动作 |
|---|---|---|---|
| **预训练 linear / GEMM 计算** | **FP8 训练框架**（V4 §5.2.1：FP32 master → FP4 量化 → **反量化回 FP8 计算**，反向 STE；"fully reuse the existing FP8 training framework"）；"linear 层 FP8" 本身是据发布 `dtype:fp8` + DeepGEMM 的**推断**，报告未逐字说 | body bf16 + head fp8 | **O1/v2 body fp8（本次路线核心）** |
| **routed experts 权重** | **FP4 仅作后训练 QAT 的存储/模拟格式**；FP4×FP8 峰值 = FP8×FP8（当前硬件），"FP4 reduces storage rather than accelerating matmul" | dense FFN（无专家） | 不需要 FP4 计算；FP4 是 post-training 项 |
| **Engram 表 + KV 投影** | 报告**确认 FP8**（§2.4.2、§3.1.3：表 FP8 存储/读取、scales 直接进 GEMM） | bf16 表（220M） | O4：表 fp8 存储 + 反量化 gather（+0.3~0.8% tok/s，省 ~220 MB） |
| **main KV / indexer QK** | **FP4，post-training QAT**（V4.1 §2.4.4，推理/部署；训练用 bf16/FP8 KV） | 训练 bf16 KV（无持久 cache） | post-training 阶段再说 |
| **SWA KV** | FP8（推理） | — | 部署项 |
| **head / lm_head** | 发布推理 **不量化**（fp32/bf16） | **我们 fp8**（主动偏离，50 步 loss rel ≤3.6e-5） | 若追求"严格对齐"，head 回 bf16：**+8.6 ms、disc +1 pp**——不建议为对齐牺牲 8.6 ms |
| 推理发布权重 | `dtype: fp8`, `expert_dtype: fp4`（config.json） | — | 导出时再量化 |

结论：**对齐的正确目标是"FP8 计算 + Engram FP8"，不是 FP4 训练**；我们的 head fp8 是唯一"反向偏离"（官方 head 不量化）。

## 6. 建议的 KPI 声明（需要拍板）

**推荐：以 raw（bf16 口径）报 45%，并把官方口径作为强制附带披露**（每份报告一行）：

```
d24_c1 b1x8192 replay: 150.1 ms | 54,580 tok/s | raw MFU 45.0% (bf16 calibre, 209.5 TF)
  | disc MFU 28.4% (V4.1-report calibre, fp8=0.5) | fp8 FLOPs 74% | t_min/t 28.4% | fp8 FFN 1.26x
```

理由：
- raw 是工业界通常说的 MFU，也是我们历来所有数字的口径（与 nanochat/Marin/M6 可比）；
- 官方口径下 45% 与"body fp8 对齐"物理上互斥（§2），二者只能选一个作为 45% 的定义；
- 项目已有 2026-11-10 决定（tok/s 北极星、raw 跨配置、disc 辅助），本方案只是把它显式化。

**若你更看重"官方口径 45%"**：走路线 B（body 保持 bf16、head 可留 fp8），目标 raw 45~46% / disc 42~43%，
代价是放弃 body fp8 对齐、墙钟慢 3–8%，且 b2+EW 融合必须都成立（否则只能到 ~43%）。

## 7. 执行计划（P1 之前，GPU 约 8–10 h ≈ ¥24–30）

| 步 | 内容 | GPU | 验收门 |
|---|---|---|---|
| E0 | `NANOV41_RMS_FUSED=1` / `NANOV41_SWIGLU_FUSED=1` A/B（b1 graph-only，A B B A 交错 + `--profile 3` 对桶） | 0.5 h | replay −9~−17 ms 且逐位/数值 parity；否则查 R83 式反效果 |
| E1 | d24 形状 fp8 microbench：FFN w1/w3/w2 [8192,1024]×[2304,1024] / [8192,2304]×[1024,2304]，fwd/dgrad/wgrad 分项 | 1 h | 端到端 ≥1.25× 且 quant ≤15%；否则不集成 |
| E2 | `LINEAR_FP8_V2_SCOPE` 扩到 FFN → G2 in-graph profile + 20 步 loss | 2–4 h | replay ≥−7 ms；loss rel <1e-5；可 capture |
| E3 | b2 显存工程（FFN 激活重算 / bf16 中间量）+ b1→b2 replay | 3–4 h | b2 capture 通过、2^21 tok/step 3 步 loss 正常、replay MFU ≥前置值 |
| E4 | d24 tile sweep（FFN/attn v3/mHC） | 1 h | 逐位等价、≥1% |
| E5 | 50 步 token-matched A/B（总账）+ `--profile-loop` 复核 | 2 h | raw ≥45% 才算达标；同时归档 disc/fp8 占比 |

**预算**：E0+E1+E2+E4+E5 ≈4–6 GPU·h；E3（b2 显存工程）再加 2–3 h ⇒ 合计 **7–9 GPU·h ≈ ¥21–27**
（fp8 单项验证按 `docs/r93-fp8-d24-memo.md` §7 只需 ~2 h ≈ ¥6）。当前余额 ¥<redacted> → 建议充值 ¥<redacted>
（留出失败重试 + P1 启动），或先只做 E0（0.5 h，¥1.5）验证最便宜杠杆再决定。

## 8. 风险与已知负结果（不要重复踩）

- **fp8 不是 2×**：d24 的 K=1024 偏短，端到端预计 1.2–1.4×；wgrad 保持 bf16（fp8 wgrad 无 split-K 时慢 5–7×，
  R84/R85 gate 拒绝）。若实际 <1.2×：只开 head+a-wo 的小 scope，别强推。
- **producer-fused cast 曾 +0.66% 变慢**（R83）：d12 上只有 1/3 的 cast 有可融合 producer；d24 张量翻倍可能改善，
  但必须实测，不要假设。
- **R83 的 d12 全 body fp8 是 NO-GO**（step −6.5~−26%）；d24 形状不同（N/K=1024/2304 而非 64–512），
  需要 E1 先验证，不许直接翻默认。
- **b2 的坑**：OOM 在 FFN fwd，缺 ~8–10 GiB，CE_CHUNK 无效；FFN 激活重算必须与 capture、fp8 wgrad 兼容
  （否则 E3 变成 2–3 天）。
- **时钟漂移**：容器内无法 `nvidia-smi -lgc`，所有 <1% 的结论必须 A B B A 交错（R83 §7.3）。
- **45% 依赖 b2/融合**：若 b2 失败且融合只取到中央值，raw 落在 **42–44%**；此时应如实报 42–44%，
  不把 fp8 的 raw 抬升说成"效率提升"（官方口径会同时下降）。
