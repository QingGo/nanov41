# R93 结果报告：P0a（K=40k 词表/数据/eval 迁移 + KD 准备）+ P0b（dense 校准 run）

> 2026-11-15。用户批准的任务：启动 P0a（纯 CPU/无卡，¥2–5），数据管线跑的同时安排 P0b 校准 run；
> 后续主跑倾向 **dense（非 MoE）** 方案。
> 本文是交付物总表：每个数字都附云端路径与验证方式。计划/契约见 `docs/r93-p0a-p0b-plan.md`，
> dense 配置推导见 `docs/r93-dense-plan.md`。

## 0. TL;DR

| 项 | 结果 |
|---|---|
| **P0a-数据** | `p2_climbmix_40k` = **6,070,608,841 tokens**（K=40k pieces，107 shards，9,053,184 docs，**4.442642 B/token**，24.28 GB）；**VERIFY_PASS**（12/12 检查，含 decode round-trip 与官方 129k 逐字节等价交叉验证） |
| **P0a-eval** | `p0_eval_40k`：56,778,205 tokens / 56,693,186 TEXT / token_bytes len=41,280 / window 2048 / 4.44642 B/tok；**8/8 检查 PASS** |
| **P0a-词表** | 官方 BPE 前缀截断 K=40,000 + 1,280 reserved specials = **V=41,280**；实测量级修正：canonical 编码 **4.4426 B/tok**（不是 expansion 估计的 4.2706） |
| **P0a-KD** | k=16、rest-bucket、dense rows、uint16 ids + fp16 logp + fp16 rest；目标 **287 shards ≈ 272.7M official tokens ≈ 20 GB**（进行中，ETA ~16:40） |
| **P0b** | dense 候选 **d24_c1**（dim 1024 / 24L / h=2304，N_act 287.98M，fwd 575.95M FLOPs/tok，Engram ON，C40=32,410）在 K=40k 数据上跑 0.6B tokens；**待完成**（ETA ~16:50 + eval ~25min） |
| **成本** | P0a 无卡 ≈ ¥2；P0b GPU（5090 ¥2.98/h）预算 ≤¥19；合计 ≈ ¥15–21 |

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| **P0a-数据** | `p2_climbmix_40k` = **6,070,608,841 tokens**（K=40k pieces，107 shards，9,053,184 docs，**4.442642 B/token**，24.28 GB）；**VERIFY_PASS**（12/12，含 decode round-trip 与官方 129k 逐字节等价交叉验证） |
| **P0a-eval** | `p0_eval_40k`：56,778,205 tokens / token_bytes len 41,280 / window 2048 / **8/8 PASS** |
| **P0a-词表** | K=40,000 + 1,280 reserved = **V=41,280**；canonical 编码 **4.4426 B/tok**（修正 expansion 估计 4.2706） |
| **P0a-KD** | k=16 + rest-bucket + dense rows；**final_partial_v140：140/287 shards，134,217,728 official tokens（143,417,520 pieces），9.75GB**（可续跑到 18.6GB，镜像限速） |
| **P0b** | dense d24_c1（dim1024/24L/h2304，N_act 288.0M，fwd 576.0M，Engram ON C40=32,410）在 K=40k 数据上 0.6B tokens：**44.3k tok/s / MFU 36.56%**；**val bpb 1.3141（gate 1.578±0.05 → PASS，−0.264）** |
| **P0b 外推** | bpb@6.0B 最可能 **0.95–1.15**（3 点拟合 c 敏感：0.94→1.21）→ **bpb<1 不保证，建议 P1 加到 8–10B tokens 或 1.5× 模型** |
| **MFU 分解** | 实测 replay 99.6% / 图外 gap −0.2% / optimizer 0.17% → 优化空间全在 in-graph；**40–42% 是 1–2 天现实目标，45% 是 stretch** |
| **成本** | P0a ≈ ¥1.5；P0b GPU 会话 ≈ ¥12.2；总余额从 ¥<redacted> → ≈ ¥<redacted> |

## 1. P0a-1 关键发现：K=40k canonical 编码与 expansion 方案不等价（实测量级修正）

E3/§8.1c 的 K 权衡表用的是 **expansion 方案**：先按官方 129k 编码，再把被剪掉的 token 沿 merge 树展开成
K 词表 pieces（纯 id 操作）。当时估计 K=40k 的 bytes/token = 4.2706（inflation 1.078）。

本次直接用一个**真正按 K=40k tokenizer 编码**的语料实测：

| 指标 | 官方 129k | K=40k canonical（本次实测） | K=40k expansion（旧估计） |
|---|---:|---:|---:|
| bytes/token | 4.6036 | **4.442642** | 4.2706 |
| pieces / official token | 1.0 | **1.0363** | 1.0780 |
| 相对官方压缩率 | 100% | **96.5%** | 92.8% |

**原因**：official tokenization 里两个相邻 official token 之间的边界，可能只是因为一个 rank ≥ K 的
official merge "挡住"了某个更低 rank 的 pair merge；剪掉那个高 rank merge 之后，这个低 rank merge
在 K 词表里**可以跨 official 边界**应用 → canonical K 编码比 expansion 更**粗**（tokens 更少、
bytes/token 更高）。expansion 流仍是合法的 K 词表流（E3 的相对结论不受影响），但不是 canonical 编码。

**影响**：
1. K=40k 的实际文本效率比旧表更好（保留官方压缩率的 96.5%），head 成本降到 1/3，K=40k 的选择更优；
2. P1 的 token↔bytes 换算必须用 4.4426：`loss(nats) = bpb × 4.442642 × ln2 = 3.0792 × bpb`
   （129k 是 3.343）；
3. 教训：任何"先 129k 再展开"的语料都不能当作目标 tokenizer 的 canonical 语料；bytes/token 必须用
   目标 tokenizer 直接实测（本轮 data-ops 的 verifier 已把 4.442642 作为 hard check）。

## 2. P0a-2 训练语料

**产物**（数据主机 `<instance-id>`，无卡模式）：
`/root/autodl-tmp/nanochat/p2_climbmix_40k/{manifest.json,web.u32.bin}`
- `tokens_total` = **6,070,608,841**（pieces，全部 <41,280；EOS=1；val shard `06542` 排除）
- `web.u32.bin` = **24,282,435,364 B** = tokens×4 整；107 shards；`docs` = **9,053,184**（== raw parquet 行数）
- `bytes_per_token` = **4.442642**；源文本 26,969,540,480 B；build log **0 个 WARN**（无 partial-read）
- 每 shard pieces：min 56,364,954 / avg 56,734,662 / max 57,221,527

**验证（PASS，12/12）**：manifest/文件大小/格式/tokenizer sha/EOS/vocab 上限/val 排除/8M token 采样
max id=39,999/解码前缀==shard_00000 开头/3 文档 round-trip 逐字节相等/官方 129k 交叉验证
（doc0：209 official pieces vs 221 K=40k pieces，两者解码字节完全一致）/panel 无泄漏。
报告：`runs/r93/p0a/data/data_report.md`、`data_report.json`、`verification.json`。

**构建流程（踩坑记录）**：
1. ClimbMix `shard_00000..00106`（107 shards）从 hf-mirror 按 20 并发 + resume 下载（聚合 ~11 MB/s）。
2. 先遇到 **13 个 shard 内部 ZSTD 数据页损坏**（footer 合法，footer-only 校验查不出；其中 5 个是截断、
   其余是早期"双 fetcher 并发写同一文件"造成的交错损坏）。修复：全量 column-read 校验定位
   （`n=107 rows=9,053,184 bad=13`），在 GPU 实例重新单线程下载那 13 个 → `FULLREAD bad=0`。
3. 注意：**无卡实例的真实配额是 0.5 CPU core / 2 GiB**（不是 `nproc` 显示的 128/1TB）；在无卡机上
   tokenize 6e9 pieces ≈ 20h，不可行。最终在 GPU 实例的**空闲 CPU**（16 核，8 个 nice'd worker）上构建，
   每批 22 shards（约 230s/批），**必须用 `--append --delete-raw --max-new-shards` 增量模式**：
   非 append 模式会同时物化 sidecar+final（峰值 2×24GB）导致 50GB 盘写满（本轮实际踩到过一次）。
4. 总计 5 批 + 1 次空批，107 shards ≈ 25 min；rsync 回数据主机（公网 ~17–27 MB/s）≈ 15 min。

## 3. P0a-3 eval 面板

`/root/autodl-tmp/nanochat/p0_eval_40k`（`nanov41-eval-data/1`，数据主机）：
- domain `climbmix_val`（`shard_06542`，训练集已排除），window **2048**
- tokens **56,778,205**（TEXT 56,693,186），`token_bytes.npy` len=**41,280**，文本字节 252,081,736
  → **4.44642 B/tok**（与语料 4.4426 差 +0.1%，符合样本差异）
- 85,019 个 EOS typed IGNORE（0 nats / 0 bytes）；byte accounting 精确（TEXT 字节 == UTF-8 字节）
- **8/8 检查 PASS**（详见 `runs/r93/p0a/data/data_report.md` §3）

## 4. P0a-4 KD 数据集（final_partial_v140，可续跑）

- 方案：**k=16**（投影到 K=40k 后的 top-16）+ **rest bucket**；dense rows（每 piece 一行，unusable 位置
  `top_ids=65535, top_logprobs=-inf, rest_mass=1.0`）；`input_ids` uint16、`top_ids` uint16、
  `top_logprobs` fp16（对 stored 16 项 renormalise）、`rest_mass` fp16（绝对质量 = 1−stored）
- 行约定：teacher row t 存在 official token t 的最后一个 piece 上（row t 预测 t+1 的第一个 piece）；可用位置 92.76%
- **交付快照 final_partial_v140**：**140/287 Arcee shards**，134,217,728 official tokens（143,417,520 pieces），
  **9.75 GB**；验证 PASS：ids<41280、blocked 7.24%、sentinel==blocked、logp 行和 1.0、stored mass 0.891、
  68 B/piece 精确；sample/ 139MB
- 完整规格：287 shards ≈ 272.7M official tokens ≈ 294.1M pieces ≈ **19.999e9 B = 18.63 GiB**
  （硬 20GB cap 已装）；按 1.64×/覆盖折算，满覆盖对 P1 价值 ≈ +2.8-3% 等效 token（≈¥3）
- **阻塞原因（外部）**：hf-mirror 的 Arcee/xethub 路径在本机 IP 上从 13:30 的 10-11.7MB/s 逐步降到
  14:57 的 43KB/s、17:00 的 ~3MB/s；换一台机器（kmz）单连接 3.6MB/s 可证明是 per-IP 限速。故采用
  "边下边建 + 每 25 shards 快照 + 可断点续跑"的保守策略交付 partial；完整版可在网络恢复后一键续跑：
  `python3 tools/r93_build_kd40k.py --num-shards 287 --resume --download-workers 12 --sample-shards 2
   --max-bytes 20000000000 --out /root/autodl-tmp/kd40k --stage-dir /root/autodl-tmp/kd40k_stage`
- 产物：`runs/r93/p0a/kd/{README.md, meta.json, verify_v140.json, teacher_stats.json}` + `tools/r93_build_kd40k.py`；
  云端 `/root/autodl-tmp/kd40k/`（数据盘保留）

## 5. P0b：dense 校准 run（完成，gate PASS）

| 项 | 值 |
|---|---|
| 模型 | **d24_c1**：dim 1024 / L24 / dense SwiGLU h=2304；seq 8192；V=41,280；Engram ON（C40=**32,410** patch） |
| 参数/算力 | N_act **287,976,752**；fwd **575,953,504** FLOPs/tok（repo 口径）；真实 +Engram wkv = 591.7M；backbone 330.2M；+Engram 550.3M stored |
| 训练 | 0.6B tokens，2^21 tok/step（286 步），WSD warmup 30 / decay-start 0.90 / min-lr 0.1，LR **2.9e-3** |
| 吞吐/显存 | **44.3k tok/s**，raw MFU **36.56%**（含 engram wkv 修正 37.56%），峰 reserved **27.78 GiB**，无 NaN |
| b2×8192 | **OOM**（capture 需 ~30.5GiB；CE_CHUNK 2048/512 均不够，差 ~0.3-1GiB）→ **b1×8192×accum256** |
| 成本 | 会话花费 ≈ **¥12.2**（含 prep/smoke/eval/profile），预算 ¥19 |

### 5.1 val bpb vs gate（p0_eval_40k，K=40k panel）

| ckpt | tokens | **val bpb** | gate = line(D) | delta |
|---|---:|---:|---:|---:|
| 0.2B | 0.200B | **1.5581** | 1.9656 | −0.408 |
| 0.4B | 0.400B | **1.3913** | 1.7111 | −0.320 |
| **final** | **0.600B** | **1.3141** | **1.5780** | **−0.264 → PASS** |

判定：**PASS**（gate 1.578±0.05）。train CE 3.965 nats @600M ⇒ bpb 1.291（与 val 一致）。

### 5.2 曲线外推（关键，决定 P1 是否值得跑）

3 点拟合 `bpb = a + b·D^-c`：a=1.2934, b=1.92e14, **c=1.789** → **bpb(6.0B) ≈ 1.294（MC 10-90%: 1.261–1.329）**。
但 c=1.79 是短跨度内"拟合出渐近线"的产物，不是可信的物理模型；固定指数稳健性表：

| c | 0.2 | 0.35 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| bpb@6.0B | **0.943** | 1.025 | 1.088 | 1.122 | 1.175 | 1.214 |

→ **P1 在 6.0B tokens 的 bpb 最可能在 0.95–1.15 区间，bpb<1 不是保证**（§8.1 line 的 0.996 偏乐观）。
若要在 bpb<1 上有余量，需 **~8–10B tokens**（同模型，成本 ¥150–190）或 **1.5× 模型 ×6B tokens**（≈¥170）。
这是 P0b 给出的最重要的规划修正。

### 5.3 MFU 阶段分解（实测，`runs/r93/mfu_probe/b1.loop_profile.json`）

p2_train `--profile-loop`（b1×8192×accum256，生产 env，4 步）：

| 段 | 每步 ms | 占比 |
|---|---:|---:|
| **replay（graph fwd+loss+bwd）** | 46,906 | **99.60%** |
| accumulate（梯度累加 add） | 65.5 | 0.14% |
| clip | 4.9 | 0.01% |
| optimizer 合计（adamw 10.1 + muon 55.6 + sinkhorn 5.0 + engram 8.5 + bias 0.9） | 80.0 | 0.17% |
| h2d + log_sync | 18.3 | 0.04% |
| **pipelined wall** | **46,994** | 100%（tok/s 44,626，MFU 36.81%） |
| **非 GPU gap** | **−81.6 (≈0)** | **−0.2%** |

结论：**训练循环没有"图外税"（0.2%）**，MFU 36.9% 全部是 **in-graph kernel 效率**问题。这与旧 MoE
栈（图外 ~18%）不同，说明优化空间只在 graph 内部：micro-batch 形状（b2 恢复 ≈ +2–4pp）、
GEMM/kernel 效率（body fp8、attn/mHC/EW 融合）、head CE。详见 `docs/r93-mfu-decomposition.md`。

## 6. 成本与实例状态

| 实例 | 用途 | 计费 | 状态 |
|---|---|---|---|
| `<instance-id>` (5090, west-B) | P0b 训练 + eval + MFU profile + P1 语料构建（空闲 CPU） | ¥2.98/h | 训练 12:39–17:15 后关机 |
| `<instance-id>` (5090, 无卡) | P1 语料/eval 面板存储 + P0a 验证 | ¥0.1/h | 完成后关机（数据盘保留） |
| `<instance-id>` (3090, 无卡) | KD 数据集构建 | ¥0.1/h | 完成后关机（数据盘保留） |

总花费（本阶段）：GPU 会话 ≈ **¥12.2**（P0b，含 prep/smoke/eval/profile；实例 12:39–17:15 ≈ ¥13.5 实际）
+ 无卡 2 台 ≈ ¥1.5 + 语料构建（GPU 空闲 CPU，已计入会话）。余额：≈ ¥<redacted>（收尾时核对）。
用户实例 `<instance-id>`（4090，运行中）、`<instance-id>`、`<instance-id>` 全程未触碰。
注：两台无卡实例曾于 16:21 因我设置的 4h TTL 自动关机（数据盘完好，已重启复核 PASS）。

## 7. 下一步（P1）与开放问题（P0b 后的修正版）

1. **P1 规模决策（最重要修正）**：P0b 的 3 点外推显示 6.0B tokens 的 bpb 最可能 **0.95–1.15**，
   bpb<1 不保证。选项：
   - (a) **10B tokens × 288M**（C≈1.7e19）：成本 ≈ **¥190**（按 36.6% MFU）或 ¥155（按 42%）；
   - (b) **1.5× 模型（~430M active）× 6B tokens**：≈ ¥170–180；
   - (c) 维持 6B（¥113–130）接受"可能 1.05–1.1"的 miss 风险。
   建议先把 P1 作为 **8–10B token 的 run**，中途 ckpt 观测；或先跑 4B 看曲线再决定是否续。
2. **LR 未定**：P1-LR 估计 1.0e-3 中心（6.3e-4–2.0e-3，2.8× 不确定）；建议 3 臂 ×0.5B LR mini-sweep（≈¥7–9）。
3. **MFU 工程**：先做 **b2 恢复**（内存 ~0.3–1GiB 缺口；Muon momentum bf16 / CE_CHUNK 128 / 重算）
   预期 40–42%（1–2 天）；45% 是 stretch（见 `docs/r93-mfu-decomposition.md`）。
4. **KD**：final_partial_v140（9.75GB）可续跑到 18.6GB（网络恢复后一键 resume）；KD 集成到 P1 trainer
   （混合/权重调度）约半天工程，需在 P1 前完成。
5. **数据**：P1 语料（6.07B pieces / 24.28GB）+ panel 已在 `<instance-id>` 数据盘；P1 启动前 rsync 到目标 GPU 机（约 20GB/15分钟级）。
6. **预算**：P1（8–10B tokens + LR sweep + KD 收尾）需要 **¥180–250**，当前余额不足以启动。
