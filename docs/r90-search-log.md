# R90 训练搜索 sprint log（2026-11-12，对标 nanochat）

> 预算 ¥70.02（批准的计划见 `docs/search-plan-2026-11-12.md`；实际执行按结果动态重排 stage 顺序，
> 预算上限不变）。目标：同 ¥ 下 CORE/val_bpb 超过 nanochat（GPT-2 CORE 0.256525、val_bpb 0.71800）。
> GPU：全部 5090 retained 实例所在机器 **0/8 空闲卡**（全市场 0 台可租）→ 改用自有 retained
> **4090 <instance-id>**（¥2.18/h，16 vCPU，hf-mirror ~3.7 MB/s）。所有 GPU job 走
> `flock /root/gpu.lock`；实例 TTL 已延到 11:02。

## Stage 0 — 资产 + 首个评测数据点

| 项 | 结果 |
|---|---|
| ClimbMix val `shard_06542` | 91.70 MB parquet → 84,992 docs / **52,429,588 tokens**（129k tokenizer，4.808 B/tok）；P0 面板 `$ROOT/p0_eval`（nanov41-eval-data/1，精确 byte 表，EOS=IGNORE，window 2048） |
| ClimbMix 训练 shards | 00000–00025 + val（27 文件，2.5 GB，~1.5B tokens @129k） |
| CORE eval_bundle | 官方 26,060,410 B bundle 已取回并解开（86 entries，22 tasks，`core.yaml`+`eval_data`+`eval_metadata.csv`） |
| E5B ckpt | `E-default.step152590.pt`（2.5B tokens，d12+Engram 193.9M+128.0M）上传到 4090，首评用 |
| CORE-lite 首测（E5B） | arc_easy 0.3480 / arc_challenge 0.2300 / piqa 0.5820 / hellaswag 0.2560 / winogrande 0.4920 / boolq (跑完补) → **CORE-lite ≈ 0.04–0.06**，离 GPT-2 0.2565 很远（预期：193M 模型只训 2.5B tokens） |
| 评测速度 | CORE-lite 6 题×1000 例 ≈ 4 min；完整 22 题（77k 例）≈ 25–45 min（Engram eval cache 限制 rows/hot）→ Stage1/2/3 用 **CORE-lite**，Stage4 最终 ckpt 用完整面板 |
| 已知坑 | eval_t0 `--window` 必须 ≤ 模型 `max_seq_len`（E5B=1024，Panel 建 2048 会在 eval 时炸） |

## Stage 2（提前做，最大杠杆）— 32k vocab/head 消融

动机：aligned d12 @129k 的 head 占 fwd FLOPs 72.3%（183.0M/token）；32k vocab 后 **84.2M/token
（0.460×）**。配合 tokenizer 压缩 4.6131 vs 4.8387 B/tok（1.049×）= **训练 FLOPs/byte 0.482×（2.07× 便宜）**。

CPU 侧（head-cache 完成，全部在 <instance-id>）
- 32k BPE tokenizer（2.0 GB ClimbMix，105 s）→ `$ROOT/tok32k/tokenizer.json`
- 129k 控制语料 `p2_climbmix_129k` **300M tokens**（4.8077 B/tok）、32k 语料 `p2_climbmix_32k` **480M tokens**（4.5888 B/tok）
- 32k val 面板 `p0_eval_32k` **55,086,667 tokens**
- 配置：`docs/aligned/d12_aligned_v32768.json`（84.16M fwd FLOPs/token）、`d16_aligned_v32768.json`（dim 640, 16L, 127.51M）、`d20_aligned_v32768.json`（dim 640, 20L, 148.82M）
- Engram 不能开：32k tokenizer 的 compressed-vocab=23,225 ≠ 配置硬编码 99,092 → 两臂均 `--engram none`

GPU 臂（4090 实测：aligned d12 @129k **62.6k tok/s**，@32k 预计 ~137k）
- A0 = 129k 控制臂，250M tokens，adamw 2.6e-3，b16×s1024
- A1 = 32k 臂，250M tokens（iso-token）；A2 = iso-cost 32k（等 GPU 秒，~480M tokens）待 A0/A1 实测后决定是否跑
- 评测：P0 val_bpb（各自 tokenizer 的面板，20M 子集）+ CORE-lite

## 关键决策记录
1. **4090 替代 5090**：5090 全部无空闲卡；4090 的 ¥/token 与 5090 接近（32k vocab 下 215M vs 269M tok/¥），
   但 129k 大模型臂慢 ~1.7×。搜索阶段可接受，最终大 run 若 5090 有空位再切。
2. **Stage2 提前**：vocab 杠杆 2.07× >> 架构（~5%），先做。
3. **Stage1 架构臂改产线家族**：旧 CONFIGS 家族（193M/129k）与最终产线（aligned d12/d16/d20 @32k）不一致，
   改为在产线家族上做 **D16/D20 深度探针**（equal-FLOPs tokens），预算从 ¥12 缩到 ~¥2.2。
4. Engram 消融放弃（32k 下不可用；129k 下已有 P2 结论）。
5. CORE-lite 子集 = arc_easy, arc_challenge, hellaswag, winogrande, boolq, piqa（每臂 1000–2000 题）。

## 进行中/下一步
- [进行中] vocab A0（250M @129k，~75 min）→ A1（250M @32k，~32 min）→ evals
- [并行 CPU] DCLM（1.4 GB）+ FineWeb-Edu（2.15 GB）下载 → `tools/r90_stage3_prep.sh` 转 32k 语料 + 13-gram 去污染
- 之后：D16/D20 深度探针 → Stage3 数据臂（5–6 arms × 0.25B）→ LR/HP 短筛 → Stage4 最终长跑 + 完整 CORE
- 记账：4090 00:37 起租 ¥2.18/h（截至 01:40 ≈ ¥2.3）；AutoDL 余额 ¥<redacted> 起

---

## 03:55 状态快照（Stage-4 最终 run 已启动）

### 硬件迁移与事故
- 02:01 起拿到 retained 5090 `<instance-id>`（¥2.78/h）→ 放弃 4090（4090 02:15 关机）。
- **03:14 实例 host-level crash（shutdown_by_running_error）**：数据盘全部完好（head-cache 立即上报并复核）；
  03:23 重新开机成功，`p2_final_129k`（4.03B tokens）与 Stage-3 语料 `--check PASS` 与 crash 前一致。
  损失：A1-32k 训练中断（141/190 步）、深度探针中断。余额从 ¥<redacted> 降到 ¥<redacted>（含 4090 时代开销）。

### 5090 实测吞吐（本机，生产栈）
| 配置 | tok/s | MFU | 备注 |
|---|---:|---:|---|
| aligned d12 @129k + Engram, v41, b2×s8192 accum64 | **148.6k** | 38.9% | 生产配方基线 |
| aligned d12 @32k（无 Engram）, v41, accum64 | **181.0k** | 21.8% | vocab 杠杆 ×1.22 |
| aligned d12 @129k（无 Engram）, adamw, accum64 | 153.2k | 40.1% | A0 控制臂 |
| aligned d12 @32k（无 Engram）, adamw, accum64 | 176.3k | 21.2% | A1（中断前）|
| aligned d12 + Engram, v41, b2×s8192 accum128 | **146.7k** | 38.5% | 最终 run 实测 |

**关键结论（修正 FLOPs 估算）**：32k vocab 的理论 FLOPs/byte 收益 2.07× 在真实吞吐上只兑现 **~1.16× text/¥**
（head FLOPs 减少无法转化为等比例 tok/s，因为非 head 开销/固定开销占比大）。因此最终 run 选择
**129k + Engram**（Engram 的 P2 配对增益 −1.9~4% BPB 大于 vocab 的 ~1-1.5% 等效增益，且生产配方已验证）。

### 已完成实验
| 实验 | 设置 | 结果 |
|---|---|---|
| A0-129k vocab 控制臂 | aligned d12@129k no-Engram, adamw 2.6e-3, 200M tok, accum64 | wall 1316s, **153.2k tok/s**, ¥1.02, **P0 val_bpb 1.6625** |
| A1-32k | aligned d12@32k, 200M tok | 训练至 141/190 步被 crash 打断（176.3k tok/s）；postrun 重跑 |
| 深度探针 d12-ref | 129k+Engram, v41 μP 8.22e-4, 140M tok, accum128（仅 66 步！） | **设计缺陷**：步数太少（warmup 100 > 66 步），BPB 3.25 无意义；放弃该探针，最终 run 选 d12（生产已验配方） |
| CORE-lite E5B（193.9M, 2.5B tok, 4090） | 6 题×1000 | arc_easy .348 / arc_challenge .230 / piqa .582 / hellaswag .256 / winogrande .492（≈随机~低分；离 GPT-2 CORE 0.2565 很远，符合小模型预期） |
| ClimbMix val 面板 | 129k | 52,429,588 tok / 84,992 docs / 4.808 B/tok（P0 基准，nanochat 锚点 0.71800/0.74833）|
| 32k 面板 | 32k tokenizer | 55,086,667 tok / 5.6724→实测 4.6131 B/tok |

### Stage-3 数据语料（head-cache 完成，32k、13-gram 去污染 filter）
| 语料 | tokens | 去污染命中率 |
|---|---:|---:|
| p2_dclm_dc_32k | 220,638,481 | 0.309% |
| p2_fwe_dc_32k | 250,000,000 | 2.082% |
| p2_fwehq_dc_32k (score≥3.5) | 250,000,000 | 2.831% |
| p2_mix_dc_32k | 855,378,884 | — |
（数据臂未跑：时间让给最终 run；postrun 里跑 3 个 100M-token 臂）

### Stage-4 最终 run（in progress）
- `docs/aligned/d12_aligned_final.json`（= aligned d12 + Engram + 129k；仅 max_batch_size 8→32 以加速 eval）
- data `p2_final_129k`（**4,033,742,930 tokens** ClimbMix，77 shards，val 排除，epoch≈1.86）
- b2×s8192×accum128 = 2.097M tok/step，3576 steps，LR 8.22e-4 cosine→0.1，warmup 100，save-every 1000
- 实测 146.7k tok/s / MFU 38.5% → **ETA ~14.2h（约 18:10 CST）**，预计成本 ¥39.5
- checkpoints: step1000/2000/3000 + final → 用于 CORE-vs-¥ 前沿
- postrun（自动）：全 P0 + 全 CORE(22 题) + CORE-lite×4 ckpt、A1-32k 重跑、3 个数据臂

### 需要报告的对比口径
- 成本：¥2.78/GPU-h × wall（+ eval 成本）；nanochat 锚点 ¥285 list / ¥95 spot（1.65h 8×H100 含 ~4B tokens 训练）
- 指标：CORE（tokenizer 无关，GPT-2 0.256525 / nanochat best 0.2626–0.2690）、P0 val_bpb（tokenizer 无关，nanochat 0.71800/0.74833）

### 07:50 磁盘事故预防（已处理）
- step1000 ckpt（2.09GB）写入 `/root/nanov41/runs`（**系统盘** 30GB）后系统盘仅剩 1.6GB → 后续 3 个 ckpt 会写失败。
- 处理：删掉不再需要的 probe 臂 ckpt（`runs/r90/probe5090/*.pt` 5 个共 ~7.6GB）+ 废弃 depth d12-ref ckpt（2.1GB）→ 系统盘 15GB free；
  剩余 3 个 ckpt（step2000/3000/final，各 2.09GB）空间足够。
- 注：head-cache 的 watchdog 只监控 `/root/autodl-tmp`（数据盘），系统盘需人工关注；已在本轮处理。

### 14:00 进度检查
- final run step **2531/3576（70.8%）**，loss 3.965（step2000=4.029 → 2511=3.965），tok/s 145.6k，MFU 38.2%，无 NaN；
  step2000 ckpt 11:50 已存（2.09GB），step3000 预计 ~15:50，final ~18:15–18:20。
- 磁盘：数据盘 25GB free、系统盘 13GB free（再写 2 个 ckpt 约 4.2GB，够）。
- ⚠️ 发现 **4090 <instance-id> 于 12:29 被开机**，13:57 起在跑（用户另一个项目的推理任务）→ 判定为用户自己的任务，**不动**。它与 5090 共享 AutoDL 余额（¥<redacted>/h）。
- 余额 14:00 **¥<redacted>**（10:00 读数 ¥<redacted>，账单有滞后）；仅 5090 时 4h 应扣 ~¥<redacted>，4090 12:29 起另有 ~¥<redacted> → 18:15 预计余额 ¥<redacted>（4090 若已关）/ ¥<redacted>（4090 一直开着）。
- **postrun 预算计划（18:40 决策）**：若 4090 仍在跑或余额 <¥<redacted> → 只跑必要项（4 ckpt 的 P0 full + CORE-lite@500 + final 全量 22 题 CORE，约 1.7h ≈ ¥4.7）；否则跑全量（含 A1-32k 重跑、3 条数据臂，约 4h ≈ ¥11）。
- 实例 TTL 已从 18:22 延到 **22:00**，避免 final 刚结束就被关机打断 postrun。

### 18:12 final run 完成 + postrun 评测结果
- final DONE 18:12:59 CST：3,576/3,576 步，wall 51,658s=**14.35h**，**145,174 tok/s**，MFU **38.04%**，
  终态 loss **3.93788**；7,499,415,552 tokens（4.03B unique ClimbMix × 1.86 epoch）。
- ckpts：step1000（07:49）、step2000（11:50）、step3000（15:53）、final（18:12），各 2.09GB。
- **评测结果**（P0 = ClimbMix val 全量 52.4M tokens；CORE-lite = 6 题×500；CORE = 官方 22 题全量）：
  | ckpt | tokens | bpb | CORE-lite | CORE(22) |
  |---|---:|---:|---:|---:|
  | step1000 | 2.10B | 1.2719 | −0.0690 | — |
  | step2000 | 4.19B | 1.2180 | −0.0448 | — |
  | step3000 | 6.29B | 1.2041 | −0.0213 | — |
  | final | 7.50B | **1.2006** | **−0.0088** | **0.0179** |
- full CORE 明细（21:22 完成，4151s）：正贡献 piqa +0.213、bigbench_language_identification +0.178、
  arc_easy +0.167、winograd +0.099；知识类 ≈0（squad 0.0020、jeopardy 0、coqa 0.0322、lambada 0.0085）；
  boolq −0.516（恒预测 False）。对照 GPT-2 0.256525 / nanochat d24 0.2585 / best 0.2626。
- postrun 操作：21:12 杀掉 postrun 父进程以免自动进入数据臂（full CORE 子进程继续跑完）；21:25 单独起 A1-32k 臂。
- 评测成本：P0×4=5096s + CORE-lite×4=760s + full CORE=4151s ≈ 2.78h ≈ ¥7.7。
- TTL 23:40；余额约 ¥<redacted>（21:17 读数）。

### 21:25–21:52 A1-32k 消融 + 收尾（sprint 结束）
- A1-32k（`docs/aligned/d12_aligned_v32768.json`, engram none, adamw 2.6e-3, 200M tokens, ClimbMix-32k）：
  wall 1109s，**182.5k tok/s（MFU 22.0%）**，¥0.86；P0 bpb **1.6770**（5M 子样本）。
  对照 A0-129k（no Engram，153.2k tok/s，20M 子样本）bpb **1.6625** → 32k 等 token bpb 差 **0.87%**，
  吞吐只快 1.19× → **vocab 杠杆的 text/¥ 几乎打平**（远小于 2.07× 理论值），129k+Engram 是正确选择。
- A1 的 CORE-lite 无法评（eval_bundle 是 129k 词表，32k 模型 index 越界 → CUDA device-side assert）；
  需 32k 版 eval bundle，留下一轮。
- 21:52 **关机 <instance-id>**（按要求完成即关机；实例保留、数据盘与 4 个 ckpt 均在）。
- 余额 ¥<redacted>（起始值 <redacted> → 共扣 ¥<redacted>；含 sprint 自用 ~¥<redacted>、期间用户 another-project 在 <instance-id> 的 4090 用量、
  以及 若干保留实例的数据盘费；账单可能仍有 1h 滞后）。
- 交付物：`docs/r90-final-report.md`、`runs/r90/frontier.md/.json`、`runs/r90/frontier.png`、
  `runs/r90/eval/`（9 个评测 JSON）、`runs/r90/final_d12.{train,series}.json`、本 log。

### 2026-11-13 dense vs MoE + 粒度扫描（租用 5090 <instance-id> @¥2.98/h）
- 背景：<instance-id> 所在机器整夜无空闲卡；改用市场租 5090（¥2.98/h），从 ydz 无卡模式同区内网 rsync 迁移
  环境（/root/t141 3.85GB，42s ≈ 92MB/s）+ 数据（p2_climbmix_129k 2.94GB + p0_eval 0.25GB）+ final ckpt（2.1GB，25s）。
- 补全探针（final ckpt）：greedy 循环（France→"largest city in the world" 重复）、事实错误（Eiffel→London）、
  fibonacci→"return False"；结果 `runs/r90/continuation_final.json`。确认 base 模型水平与 CORE 0.0179 一致。
- 200M-token 屏幕结果（P0 5M 子样本，adamw 2.6e-3，warmup 15，warmup=15，b2×8192×accum64）：
  | arm | 结构 | bpb | tok/s | MFU | wall |
  |---|---|---:|---:|---:|---:|
  | M0-moe | MoE G=4（aligned） | 2.0024 | 148.7k | 39.0% | 22.3min |
  | D1-eqflops | dense mr384（等 FLOPs） | **1.9658** | 159.1k | 41.7% | 20.9min |
  | D2-mr1536 | dense 1.46× | 1.9647 | 99.8k | 38.3% | 33.3min |
  | D3-mr6144 | dense 3.32×（100M tok） | 2.2291 | 40.3k | 35.1% | 41.2min |
  | G1-g2 | MoE G=2 | 2.0539 | 143.7k | 39.6% | 23.1min |
  | G2-g8 | MoE G=8 | **1.9871** | 145.2k | 37.1% | 22.9min |
  | G3-g16 | MoE G=16 | 2.0064 | 135.0k | 34.0% | 24.6min |
  | D4-capacity | dense 7.5×/752M 全激活 | OOM×2 | — | — | — |
- **结论**：① 等 FLOPs 下 dense 赢 MoE（−1.83% bpb、+7% 吞吐）→ 本尺度 EL≈0.73–1.00；
  ② G=8 比 aligned G=4 好 −0.77%，G=16 回退 → 论文 G*≈8–12 复现，下一轮可把产线 G 改成 8；
  ③ 200M-token 屏幕对更大的 dense 是 data-limited，不能据此判断 dense 扩展性。
- 失败/修复记录：dense twin（1 专家）被官方 TileLang topk-gate 的 `num_routed_experts%4==0` 断言挡住 →
  `NANOV41_TL_GATE=0`；G16 (inter=64) 要 `NANOV41_MOE_DW2_TILE=64,128,32,4,3`（不是 MOE_DW_TILE）；
  D3/D4 大 dense OOM → D3 用 b1×8192×accum128 跑通，D4 即使 b1×s4096+CE_CHUNK=1024 仍 OOM（graph pool 24.8GB）→ 放弃。
- 成本：有效 run 2.05h ≈ ¥6.1；冒烟/重试 ≈ 0.6h ≈ ¥1.8；**实例 00:18–12:00 共 ¥35 左右，其中 ~6.4h 空转
  （01:51–08:15）≈ ¥19 是本轮监控失误造成的浪费**（教训：排队任务结束就必须设下一次 wake-up）。
- 交付：`docs/dense-vs-moe-design.md`（新增 §7 结果）、`runs/r90/dense_moe/`（p0/train JSON + queue 日志）、
  `runs/r90/continuation_final.json`、11 个 twin 配置、arms/queue/fallback env 脚本。

### 2026-11-13 21:35 少数据搜参：R91 四层级联协议（只写脚本，未实跑）
- 背景：用户问"我们可以用更少的数据量搜参吗"。结论：分四层，结构类可压到 100.7M tokens，
  Engram 分配必须 ≥0.5B，LR/参数分配的最优值必须用两档预算拟合或长跑确认。
- 交付（脚本已通过离线 fixture 测试，未上 GPU）：
  - `tools/r91_common.py`（常量/阶段定义/plan 读写）、`tools/r91_plan.py`（生成 A/B/C arms+env）、
    `tools/r91_promote.py`（A/B→D1→D2 晋级）、`tools/r91_analyze_c.py`（LR 定律 + C2 越界扩展）、
    `tools/r91_rank_transfer.py`（前缀 vs 全预算 Spearman）、`tools/r91_queue.sh`（队列：依赖顺序/
    预算护栏/preflight/断点续跑/QUEUE_DONE）。
  - 通用臂 runner `tools/r90_arms5090.sh` 扩展：WSD 参数（`LR_SCHEDULE/DECAY_START/MIN_LR_RATIO`）、
    `WARMUP_FRAC`、per-arm 覆盖列（9–12）、`SAVE_AT_TOKENS` 前缀 ckpt + 前缀 P0 评测（marks.json）、
    OOM 重试、以 `p0.json` 为完成标记的可重入跳过。
  - `tools/p2_train.py` 新增 `--save-at-tokens`（token 预算前缀保存 + `[save-mark]` 机器可读行）。
  - 协议文档 `docs/r91-search-protocol.md`（含 WSD 参数、每层细节、命令、成本、风险、fixture 覆盖）。
- 设计要点：C 先跑（2 FFN × 2 预算 × 3 LR = 12 臂）→ 测出 `lr*(tokens)` 后再跑 A（8 结构臂 @100.7M）
  与 B（5 Engram 臂 @503.3M）→ D1（A 前二 + B 第一 + 产线对照 @1.007B）→ D2（前二 @2.013B）。
  A/B 各在 50% 预算处存前缀 ckpt，`rank_transfer` 直接量化"100M 排名能否迁移到 1B"。
  放弃原"3 宽度"设计：A/B/D 候选全是 dim=512，不需要宽度外推；改为 FFN-conditioned 定律。
- 预算：全量投影 24–31h ≈ ¥71–92（脚本按 0.36 MFU 保守估）；lean（C+A+B，B=480 步）≈11–14h ≈ ¥34–41（B 砍到 288 步 ≈10–12h ≈ ¥30–36）。
  `MAX_COST_Y` 护栏会逐层拦截；当前余额不足以跑全量，等充值/决策后再起 GPU（完成后按规则关机）。
- **2026-11-13 21:35–21:55 独立 review（read-only subagent）+ 修复**：审出 3 个 blocker / 12 个 major，
  全部修复并加测试：
  - B1 新实例上 C 层训练后**从不分析**（分析被自己的输出文件门住）→ 改为以 C 结果数为门无条件分析；
    C 不可用时默认跳过 A/B/D（`ALLOW_PRIOR_LAW=1` 才用先验）；
  - B2 重启后 C 网格被自己拟合的定律移动、旧 bpb 配新 LR → C 网格固定用先验、分析读
    `<tag>.train.json` 的**真实 peak_lr**、每臂 `schedule.txt` 写 provenance signature，签名变化
    自动把旧目录挪成 `<arm>.stale.<HHMMSS>` 重训（M4）；
  - B3 `QUEUE_DONE` 无条件写（dry-run/失败也写）→ 仅当"每个请求层所有臂都有有效 p0.json"才写
    `QUEUE_DONE`，否则写 `QUEUE_STOPPED`（含原因、exit 1）；dry-run 不写任何 sentinel；run_stage
    失败计 FAIL（M5 也修：plan 失败不再执行旧 arms/env）；
  - M1 预算护栏空转（D1/D2 在 promote 前检查=0、spent_y 每次重置、依赖被拦仍继续）→ 先 promote
    再按投影检查、`cost.log` 续算累计、依赖失败跳过下游；非数值投影直接报错；
  - M2 跨层去重失效（A5≡B0 同一模型）→ 语义指纹（engine_config+engram 哈希）去重，保留高预算证据；
  - M3 位置式 D tag 复用旧结果 → tag 改为 `D<k>-<短名>-lr<LR>`（LR 变了落新目录）；
  - M6 mark 评测失败永不重试 → 缺 mark 的臂在下次运行时重试（ckpt 评测失败即保留，p0_t JSON 严格校验）；
  - M7 无完整性门 → law 带 `quality/warnings`，promote 结果不足 top-k 记 `partial`；
  - M9 `--out-dir` 不生效、M10 `LR_LAW` 无效、M11 非原子 ckpt（改 `.tmp`+`os.replace`；P0 失败把 `.pt`
    挪 `.pt.bad` 下次重训）、M12 `--save-at-tokens`+resume 丢点（改为显式警告）。
  - 其它：rank_transfer 每层各写一份/有限值过滤/winner-aware verdict 并加"前缀无 cooldown"口径说明；
    D 层不再存前缀点；P0 mark 面板统一 5M；`R91_B_STEPS` 支持 lean；MAX_COST_Y 默认 95；
    protocol doc 全面更新（§0 成本、§4 命令、§5 口径、§6 保底、§10 修复记录）。
  - 测试：`/tmp/r91_fixture_test_v3.py`（定律真实 LR/越界扩展/指纹去重/tag/rank/dry-run 无 sentinel/
    QUEUE_STOPPED→QUEUE_DONE 门控）全过；假解释器端到端（前缀评测/重试/签名隔离/坏 ckpt 隔离/
    DEL_CKPT/重入跳过）全过。仍未上 GPU。

### 2026-11-14 R91 lean 搜索实跑（C→A→B，用户借用的 5090 <instance-id> @¥2.98/h）

**准备**：该实例是干净环境；从本机同步了含 review 修复的 `tools/`（含 R91 全部脚本），用密钥 rsync 从保留实例
`<instance-id>` 拉了 `/root/t141` venv + `/root/dsv41` tokenizer + `p2_final_129k`(16G)/`p2_climbmix_129k`(2.8G)/`p0_eval`，
跨区约 12 分钟（~32MB/s），拉完关机 ydz；10 步 MoE/dense 冒烟各通过（143.8k/161.7k tok/s），修掉 runner 里
多 mark 解析的 `local` bug 后重测通过。

**队列**：`STAGES=C,A,B R91_B_STEPS=288 MAX_COST_Y=70 PRICE_Y_H=2.98`。运行 15.1h、38 臂，**总花费 ¥44.8**
（cost.log 累计 ¥44.04 到 B 结束 + A3 补跑 ~¥0.8）。实际 sentinel 是 `QUEUE_STOPPED`（completion gate 时
A 层 7/8，因 A3 G16 首跑 tile 失败；A3 已在队列后用 tile env 补跑成功）。

**事故与修复**：C2 规模（12 臂）会把原 MAX_COST_Y=45 的 A/B 挤掉，我中途 kill 队列想改上限 → 误杀正在跑的
C 阶段（rc=143，只完成 9/12 臂）；随后用 cost.log 续算、`MAX_COST_Y=70` 干净重启，C 剩余 3 个 dense 长臂补跑完，
总浪费约 20 分钟/¥0.6。**教训：不要 kill 正在跑的队列改参数——应在启动前按 C2 最坏情况预留预算。**

**C/C2（LR 定律，24 臂，¥29.3）**
- 4 个 cell 全部 high-edge（先验网格太低）→ 自动 C2 扩展 12 臂；
- 最优 LR：moe 100.7M **7.36e-3**、moe 302M **4.47e-3**、dense 100.7M **6.50e-3**、dense 302M **3.78e-3**；
- 拟合 token 指数：dense **a=−0.270**、moe **a=−0.790**、mean **a=−0.530**（每 FFN 仅 2 格，低置信）；
- **产线 LR 反向验证**：mean 定律外推 7.5B（搜参 batch 6.15e-4）乘 √2（产线 batch）得 **8.70e-4**，
  产线实际 8.22e-4 → **ratio 1.058**。即：产线长跑 LR 基本最优，错的是短跑直接抄长跑 LR（低 ~8×）。

**A（结构，8 臂 @100.7M，¥6.2）**：A7 dense h3072 **1.5196** < A6 h1536 1.5211 < A5 h768 **1.5225**（等 FLOPs）
< A4 h384 1.5339 < A2 MoE G2 1.5410 < A0 MoE G4 1.5460 < A1 G8 1.5486 < A3 G16 1.5658。
→ 等 FLOPs dense 赢 MoE **−0.0235 bpb**（复现 R90）；dense 加宽收益递减；**G 排序被 LR 标定翻转**（G2≈G4>G8>G16，
R90 的"上 G8"结论不成立）。rank-transfer：**ρ=0.714**，winner 在 50%→100% 从 h1536 换到 h3072。

**B（Engram 分配，5 臂 @302M，¥8.9）**：B2 dense+eng815 **1.3028** < B1 eng440 1.3044 < B0 eng220 1.3110
< B4 MoE E16+eng748 1.3230 < B3 MoE E128+eng220 1.3446。
→ 表大小收益单调但饱和（220→440 −0.0066，440→815 仅 −0.0016）；dense 相对 MoE 的优势随 tokens 增大
（同表 220M：200M 时 −0.0015 → 302M 时 **−0.0336**）；MoE 往 memory 挪仍有效（B3→B4 −0.0216）但追不上 dense。
rank-transfer：**ρ=1.0（winner 不变）** → 分配类屏幕可再减半。B3/B4 的 LR 用 C2 实测顶点 3.5e-3（不是拟合 2.59e-3）。

**结论**：产线候选从"MoE+eng220"换成 **dense h768 + Engram 440–815M**（iso-param 600–976M、iso-FLOPs 183M，
200M/302M 两个预算都第一，且快 ~10%）；是否改 7.5B 产线等 D1（1.007B tokens）确认。
**交付**：`runs/r91/`（38 臂 p0/train/marks/rank_transfer、lr_law.json、plan.json、cost.log、search_summary.md）、
`docs/scaling-learnings.md` 全面更新（L15/L16、§5/§6.2/§7/§9/§12）、`docs/r91-search-protocol.md` §11 实跑结果。

> **收尾（2026-11-14 13:50）**：结果已 rsync 回 `runs/r91/`（38 臂、lr_law、rank_transfer、cost.log、queue.console）；
> A3 已补跑（bpb **1.5658**，G16 最差，符合预期）；`docs/scaling-learnings.md` 已更新为实测值
> （新增 L15/L16，重写 §5/§6.2/§7/§9/§12），并附 9 张图（`runs/r91/figs/`，脚本 `tools/r91_make_figures.py`）
> 与自包含 HTML 报告 `runs/r91/scaling-learnings.html`（脚本 `tools/r91_make_report_html.py`）。**已关机 <instance-id> 与 <instance-id>**（用户借用的 5090 用完即关）。
> 本轮搜索记账：**15.1h GPU ≈ ¥44.8**（cost.log 累计 ¥44.04 + A3 补跑 ~¥0.8）；AutoDL 余额 ¥<redacted>
> （期间用户其他实例也在计费，余额降幅不能全算到本轮）。
