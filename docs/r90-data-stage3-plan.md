# R90 Stage 3 数据配方计划（2026-11-12）：ClimbMix vs DCLM-Baseline vs FineWeb-Edu

> 目标：在 **¥10 / ~3.5 GPU-h** 内回答 4 个数据问题，为 Stage-4 预算匹配验证（8–10B tokens）
> 选出数据配方，并让 P0 `val_bpb`（与 nanochat 直接可比，tokenizer 无关）成为可信指标：
>
> 1. **同源对比**：nanochat 使用的 ClimbMix 在我们的模型上是否优于 DCLM-Baseline / FineWeb-Edu？
> 2. **质量清洗**：FineWeb-Edu 的 edu-score 过滤（`score ≥ 3.5`）是否值得（data-strategy-v2 §4 的
>    "质量分层"最简单的可执行版本）？
> 3. **配比**：web 域内 ClimbMix / DCLM / FineWeb-Edu 混合是否优于单一源（多样性 vs 质量）？
> 4. **去污染**：DCLM/FineWeb-Edu 与 ClimbMix val（`shard_06542`）的重叠对 `val_bpb` 有多大虚高？
>    所有 arm 的 P0 指标是否可信？
>
> 复用 `docs/search-plan-2026-11-12.md` §3 Stage3 的预算（¥10）与协议；数据准备全部在 **CPU
> 无卡实例（¥0.1/h）** 完成，不占 GPU 预算。脚本：`tools/r90_data_fetch_{dclm,fineweb}.sh`、
> `tools/r90_data_prepare.{sh,py}`。

---

## 0. 实验协议（固定，与 Stage-1 一致）

| 项 | 值 |
|---|---|
| 模型 | `d12` + Engram `d12`（P2 容量 gate 的胜者：E-default），AdamW peak 3e-3 |
| 数据量 | **0.30B tokens / arm**（= 18,311 steps @ b16×s1024；若需补 seed 复测，用 0.25B，见 §6） |
| batch/seq | `batch 16 × seq 1024` = 16,384 tokens/step，warmup 300，grad-clip 1.0，z-loss 0 |
| 吞吐假设 | **~204k tok/s**（R90 Stage-1 实测 s1024；Stage-3 数据不改栈） |
| 主指标 | **P0 = ClimbMix val_bpb**（`shard_06542` → `p0_eval`，bits/byte，越低越好；nanochat 1.65h=0.71800 / d24=0.74833 锚点） |
| 副指标 | **CORE-lite**（`arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa` × 2000 题） |
| 判定 | 固定 token 数 → **质量比较不需要 ABBA**（ABBA 只用于等 wall 的比较）；Δ`val_bpb` ≥ 0.01 视为信号（P2 A/B 报告的可分辨量级），否则判平 |
| 顺序 | 单 seed；arm 顺序不影响指标（固定 tokens），用 D0 控制臂锚定数据质量尺度 |

训练/评测命令见 §7（`tools/r90_stage1_env.sh` 的栈 + `p2_train.py --tokens 300000000`）。

---

## 1. 数据源（2026-11-12 用 `curl -sI` 对 hf-mirror.com 实测）

| 源 | hf-mirror URL（`https://hf-mirror.com/datasets/...`） | 布局 / 已核实大小 | token 估算 |
|---|---|---|---|
| **ClimbMix-400B**（nanochat 同源） | `karpathy/climbmix-400b-shuffle/resolve/main/shard_XXXXX.parquet` | `shard_00000`=**92.29MB**，val `shard_06542`=**91.70MB**；训练 shards `00000..06541`。已下载 00000–00009 + 06542（1.29GB 中已有 0.90GB，其余在飞行中） | ~**61M tok/shard**（`r90_build_climbmix_val.py` 实测），14 shards ≈ **854M** tokens |
| **DCLM-Baseline 1.0** | `mlfoundations/dclm-baseline-1.0/resolve/main/global-shard_%02d_of_10/local-shard_0_of_10/shard_%08d_processed.jsonl.zst` | `global-shard_01..10_of_10/local-shard_0..9_of_10` × **279 文件**（共 27,900 个 jsonl.zst）。每个 global shard 首个文件实测：141.3 / 133.2 / 130.1 / 149.8 / 89.9 / 129.4 / 134.3 / 197.5 / 141.5 / 215.3 MB → **10 文件 = 1.40GB** | ~0.65–0.75 tok/byte（4T tokens/≈5.5–6TB）→ 10 文件 ≈ **0.9–1.4B** tokens（`r90_data_prepare.py` 给出精确值） |
| **FineWeb-Edu**（10BT 采样） | `HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet` | **14 文件 × 2.152GB** = 28.52GB；列 `text,id,dump,url,file_path,language,language_score,token_count,score` | 10B tokens → **0.71B tok/文件**；1 文件够 2 个 0.3B arm 复用 |
| FineWeb-Edu（100BT 采样） | `.../sample/100BT/%03d_%05d.parquet` | **140 文件 × 2.153GB** = 286GB | 100B tokens，约 0.71B tok/文件（需要更多数据时才用） |

要点（已在实例上复核）：hf-mirror 的 `resolve` 先 **302 到 xethub CAS**，`curl -L` 必须带；
HEAD 的 `x-linked-size` 就是真实文件大小（下载脚本用它做**尺寸校验/断点续传**）。实例到
hf-mirror 可达（DCLM/FineWeb/ClimbMix 三种 HEAD 均 302+x-linked-size 正常）。
**数据量足够**：D0/D1/D2/D5/D6 复用同一个 ClimbMix/DCLM 语料；D3/D4 复用同一个 FineWeb 文件。
因此全 Stage-3 只需 **DCLM 1.4GB + FineWeb 2.15GB**（ClimbMix 已在下载），磁盘峰值 < 20GB（49GB 盘）。

---

## 2. 新增工具（本 Stage 的交付物）

| 文件 | 作用 |
|---|---|
| `tools/r90_data_fetch_dclm.sh` | DCLM web shards → `/root/autodl-tmp/nanochat/dclm/`：resumable（`curl -C -`）、并行（`xargs -P`）、HEAD 尺寸校验、断点续传、磁盘/失败保护、`files.list`+`sizes.txt`+`fetch.log`。默认 10 个 global shard 的首文件（1.40GB）。`DRY_RUN=1` 只列计划+尺寸。 |
| `tools/r90_data_fetch_fineweb.sh` | FineWeb-Edu → `.../fineweb_edu/sample/<SAMPLE>/`：同机制；`SAMPLE=10BT|100BT`、`N_FILES` 控制量（默认 1 文件 = 2.15GB ≈ 0.71B tokens）。 |
| `tools/r90_data_prepare.py` | 原始源 → P2 语料（`nanov41-p2-corpus/1`）：`--kind parquet-text\|dclm-jsonlzst`、多进程 tokenize、`--tokens` 截断、`EOS` 逐文档、**自动排除 ClimbMix val shard**、`--score-col score --score-min 3.5` 质量过滤、`--decontam-val` 13-gram 去污染（`sample` 估计 / `filter` 过滤）、`--combine` 生成多域目录（供 `--mix` 训练）、`--check` 校验、`--dry-run`。 |
| `tools/r90_data_prepare.sh` | 上面的薄包装（`bash tools/r90_data_prepare.sh ...`，默认 `/root/t141/bin/python`）。 |

**2026-11-12 实例冒烟（无 GPU、只读/临时文件）**：`r90_data_fetch_dclm.sh DRY_RUN=1` 正常解析
HEAD 尺寸（g01=141.3MB, g02=133.2MB）；`r90_data_prepare.py` 的三条路径（ClimbMix parquet、
DCLM jsonl.zst、FineWeb-Edu `score` 过滤）+ `--check` 全部通过，ClimbMix 实测 **84.5k tok/s
（单 worker、0.5 核）**；decontam sample 模式在 20 篇文档上正常输出 hit_rate（0/20）。

语料输出（与 `p2_train.py:TokenMixLoader` 完全兼容）：

```
<out>/manifest.json   {"format": "nanov41-p2-corpus/1", "domains": {"web": {"tokens": N}}, ...}
<out>/web.u32.bin     little-endian uint32，逐文档 EOS（4 B/token）
<out>/build_report.json / decontam.json
```

混合 arm 的目录用 `--combine` 生成（多 domain + manifest），训练时
`--mix climbmix=0.50,dclm=0.25,fineweb=0.25`（`TokenMixLoader` 原生支持多 domain 加权采样）。

---

## 3. 一次性数据准备（GPU 之前完成，¥≈0.1–0.5）

```bash
cd /root/nanov41
ROOT=/root/autodl-tmp/nanochat; PY=/root/t141/bin/python

# --- 1) 拉取原始数据（CPU 实例；可反复重跑，断点续传）
bash tools/r90_data_fetch_dclm.sh                         # 10 文件 1.40GB，~10–45 min @0.5–3MB/s
N_FILES=1 bash tools/r90_data_fetch_fineweb.sh            # 1 文件 2.15GB，~15–70 min
# （ClimbMix 训练 shards 00000–00009 已由 r90_fetch_assets.sh 在飞行中）

# --- 2) 建 P2 语料（CPU；--workers 在 0.5 核实例上只能 1，在 GPU 机 CPU 上可 8–12）
$PY tools/r90_data_prepare.py --kind parquet-text \
    --src $ROOT/climbmix --out $ROOT/p2_climbmix_s3 --tokens 600000000 --workers 1   # val 自动排除
$PY tools/r90_data_prepare.py --kind dclm-jsonlzst \
    --src $ROOT/dclm --out $ROOT/p2_dclm --tokens 500000000 --workers 1 \
    --decontam-val $ROOT/climbmix/shard_06542.parquet            # 默认 sample 模式（估计泄漏率）
$PY tools/r90_data_prepare.py --kind parquet-text \
    --src $ROOT/fineweb_edu/sample/10BT --out $ROOT/p2_fineweb --tokens 500000000 --workers 1 \
    --decontam-val $ROOT/climbmix/shard_06542.parquet
$PY tools/r90_data_prepare.py --kind parquet-text \
    --src $ROOT/fineweb_edu/sample/10BT --out $ROOT/p2_fineweb_hq --tokens 500000000 \
    --score-col score --score-min 3.5 --workers 1
# --- 2b) 若 sample 泄漏率 > 0.1%：重建"去污染版" DCLM/FineWeb 语料（filter 模式，CPU 时间 ~2–4×）
$PY tools/r90_data_prepare.py --kind dclm-jsonlzst --src $ROOT/dclm \
    --out $ROOT/p2_dclm_dc --tokens 400000000 --decontam-mode filter --workers 1 \
    --decontam-val $ROOT/climbmix/shard_06542.parquet

# --- 3) 混合目录
$PY tools/r90_data_prepare.py --combine \
    climbmix=$ROOT/p2_climbmix_s3,dclm=$ROOT/p2_dclm,fineweb=$ROOT/p2_fineweb_hq --out $ROOT/p2_mix_web
$PY tools/r90_data_prepare.py --check --out $ROOT/p2_climbmix_s3   # 每个语料建完都可 --check
```

**吞吐/耗时预期**：
- 下载：hf-mirror 单连接实测 **0.13–1.34 MB/s**（正在跑的 ClimbMix 日志），`PARALLEL=4` 聚合
  约 0.5–3 MB/s → DCLM 1.40GB **~10–45min**，FineWeb 2.15GB **~15–70min**；CPU 实例 ¥0.1/h，
  总成本 **< ¥0.3**（可与 ClimbMix 下载并行）。
- tokenize：**0.5 核实例实测 84.5k tok/s**（`--workers 1`，ClimbMix parquet，含进程启动；
  合成冒烟测试中 DCLM jsonl.zst 与 FineWeb-Edu `score` 过滤两条路径也通过）→ 2.1B tokens
  ≈ **7–9h（¥0.7–0.9）**，墙钟较长但成本可忽略；**推荐把 tokenize 放到 GPU 机的 CPU**
  （`--workers 8–12`，25 核级 ≈2.5–3M tok/s）→ **10–15min**，安排在 Stage-1/2 的 GPU 训练时段
  同步跑（CPU 空闲，不占 GPU、不增加实例时间）。
- 磁盘：原始 3.6GB + 语料 ~2.4+2+2+2GB ≈ 最多 12GB，49GB 盘充足。

---

## 4. 去污染（针对 ClimbMix val `shard_06542`）

**为什么必须做**：Stage-3 的主指标是 ClimbMix val `val_bpb`。ClimbMix 训练/val 是**按 shard
切分**的（val=06542），所以 **D0 控制臂天然无泄漏**；但 DCLM/FineWeb-Edu 是从 Common Crawl 独立
构建的公开集，**其中的网页可能也出现在 ClimbMix val 里**，这会让 D1/D3 的 `val_bpb` 虚低
（虚高竞争力），也可能污染"清洗/配比"结论。Marin/Jellyfish 与 `docs/data-strategy-v2.md` §4
的标准做法就是 **word-level 13-gram 命中即删/即报**。

**实现**（`tools/r90_data_prepare.py`）：
- 归一化：小写 + 去 HTML 标签 + 空白折叠（与 data-strategy-v2 的预处理一致性要求一致）；
- 13-gram 的 word 级哈希 → int64 多项式 rolling hash（**向量化**，不依赖 Python 循环），
  val 索引默认上限 20M grams（160MB，2GB 容器可跑）；fork worker 共享（COW）；
- `--decontam-mode sample`（默认）：每个源读 3000 篇文档估算 **hit_rate**（秒级–分钟级，
  不改变语料）；`filter`：tokenize 时逐文档检查，命中即跳过（CPU +2–4×），报告 filtered 文档数；
- **自动排除** `shard_06542.parquet`（`--allow-val` 仅调试用，会打印警告）。

**执行与判定**：
1. 先跑 §3 里带 `--decontam-val` 的 sample 模式（DCLM、FineWeb，各 ~1–2min）；
2. 判定：hit_rate **> 0.1%** → 追加 `filter` 模式重建 DCLM（且 FineWeb 也做），用去污染语料
   训 D2，并在结果表并列 D1（raw）/D2（filtered）量化泄漏大小；hit_rate ≤0.1% → D1 raw 即可，
   D2 可省下 1 个 arm 的预算；
3. 额外诊断（不花 GPU）：用同一 13-gram 采样检查 **ClimbMix train ↔ DCLM** 交叉重叠率。
   若 >30%，说明 D1 与 D0 高度同源，D0-vs-D1 的差异小、应把结论重心放到 FineWeb/mix 上。

---

## 5. 8 条 arm 计划（6–8 条，含 mix/clean 变体）

固定：`d12 + engram d12`、b16×s1024、AdamW 3e-3、warmup 300、0.30B tokens、seed 0。
成本口径：训练 0.30B @204k tok/s = **24.5min**；init/build ~1min；P0 val_bpb ~1min；
CORE-lite 6×2000 ~3min → **~29.5min/arm ≈ ¥1.37**（¥2.78/GPU-h）。

| arm | 数据 | 语料目录 / `--mix` | 目的 | train wall | 成本 |
|---|---|---|---|---:|---:|
| **D0** control | ClimbMix（nanochat 同源） | `p2_climbmix_s3` `--mix web=1.0` | 基准（也校准我们 vs nanochat 同数据） | 24.5min | ¥1.37 |
| **D1** dclm-raw | DCLM-Baseline | `p2_dclm` `--mix web=1.0` | DCLM 纯源质量 | 24.5min | ¥1.37 |
| **D2** dclm-decontam | DCLM（13-gram 过滤） | `p2_dclm_dc` `--mix web=1.0` | 量化 val 泄漏对 `val_bpb` 的虚高；最干净的 DCLM 数据点 | 24.5min | ¥1.37 |
| **D3** fineweb-raw | FineWeb-Edu 10BT | `p2_fineweb` `--mix web=1.0` | edu-filtered 源 vs DCLM/ClimbMix | 24.5min | ¥1.37 |
| **D4** fineweb-hq | FineWeb-Edu `score≥3.5` | `p2_fineweb_hq` `--mix web=1.0` | 质量分层是否值得（raw vs HQ 同源对照） | 24.5min | ¥1.37 |
| **D5** mix-A | 50% ClimbMix + 25% DCLM + 25% FineWeb-HQ | `p2_mix_web` `--mix climbmix=0.50,dclm=0.25,fineweb=0.25` | 多样性 mix | 24.5min | ¥1.37 |
| **D6** mix-B | 40% ClimbMix + 40% DCLM + 20% FineWeb-HQ | 同上 `--mix climbmix=0.40,dclm=0.40,fineweb=0.20` | mix 比例消融（DCLM 权重加倍） | 24.5min | ¥1.37 |
| D7（可选） | 最优臂重复 seed | 同上 | 2σ 复核（预算允许时） | 24.5min | ¥1.37 |

**推荐排序（¥10 内的最优信息量）**：
- **首选 7 臂：D0–D6 = ¥9.59**（3.5 GPU-h），留 ¥0.4 余量给 eval 波动。
  7 臂覆盖"纯源对比 + 清洗 + mix"三类问题；D2 是否跑取决于 §4 的 sample 泄漏率
  （hit_rate ≤0.1% 则跳过 D2，改跑 D7 = 最优臂复测，总价仍是 7 臂）。
- 若只想花 **¥5**：D0 + D3 + D4 + D5 = ¥5.48（问题 1/2/3 各取一个最省答案）。
- **省预算变体**：0.15B tokens/arm（train 12.3min，arm ≈ 17min ≈ ¥0.79），可做 ~12 臂；
  代价是噪声更大 → 只建议用于 **mix 比例粗筛**（D0/D5/D6 + 一个 ratio 点），胜者再用 0.3B 复核。
  正式交付建议仍用 0.3B（与 Stage-1 同口径，便于横向对比）。
- 不建议：8×0.30B（=¥11.0，超 ¥10）；把 DCLM 拉到 1B tokens 单臂（数据源只值 0.3B 的结论）。

---

## 6. 决策规则（跑完之后写进 R90 Stage-3 结果）

1. **去污染优先**：先看 `decontam.json` 的 hit_rate。D1/D3/D4 的 `val_bpb` 若被泄漏显著抬高
   （D1 vs D2 Δ>0.01），**结论一律用 D2（filtered）代替 D1**，并在表里同时报告 raw 值以便解释
   nanochat 口径（nanochat 自己也可能有轻微泄漏，同口径比较时注明）。
2. **纯源**：`val_bpb` 最低的源为 Stage-4 web 底座候选；ClimbMix（D0）vs DCLM（D1/D2）差值
   ≥0.01 BPB 才算显著。
3. **清洗**：D4 vs D3 的 Δ 就是 score≥3.5 过滤的收益；同时看 token 生成率
   （HQ 语料每 GB 的 tokens 更少、但训练 token 数固定 → 只比质量）。
4. **mix**：D5/D6 vs 最好的纯源 ≥0.01 → 采用 mix，并把比例作为 Stage-4 默认；
   D5 vs D6 不显著 → 采用更简单的 50/25/25。
5. **CORE-lite 只做副指标**（6 题 ×2000 噪声 ~±1–2%），仅用于排除"BPB 好但任务崩"的情况。
6. 结果落盘：`runs/r90/stage3/<arm>/<arm>.{pt,p0.json,core_lite.json,run.json}` +
   `runs/r90/stage3/summary.json`（含每臂 tokens/tok-s/wall/¥/decontam hit_rate）。

---

## 7. 训练与评测命令（固定协议）

```bash
cd /root/nanov41
. tools/r90_stage1_env.sh                      # 同一 R69/R77 加速栈 + tokenizer/TOK 路径
ROOT=/root/autodl-tmp/nanochat; OUT=runs/r90/stage3; TOK=300000000
PY=/root/t141/bin/python

# 纯源 arm（D0/D1/D2/D3/D4）: --mix web=1.0
$PY tools/p2_train.py --arm D0-climbmix --config d12 --engram d12 \
    --tokens $TOK --data-dir $ROOT/p2_climbmix_s3 --mix web=1.0 \
    --batch 16 --seq 1024 --optimizer adamw --peak-lr 3.0e-3 --warmup 300 \
    --grad-clip 1.0 --log-every 500 --nan-check-every 8 --out $OUT/D0-climbmix

# mix arm（D5/D6）: 多域加权
$PY tools/p2_train.py --arm D5-mixA --config d12 --engram d12 \
    --tokens $TOK --data-dir $ROOT/p2_mix_web --mix climbmix=0.50,dclm=0.25,fineweb=0.25 \
    --batch 16 --seq 1024 --optimizer adamw --peak-lr 3.0e-3 --warmup 300 \
    --grad-clip 1.0 --log-every 500 --nan-check-every 8 --out $OUT/D5-mixA

# P0 val_bpb（ClimbMix val，与 nanochat 同口径；20M-token 子集够筛，最终模型再全量）
$PY tools/eval_t0.py --dataset $ROOT/p0_eval --domains climbmix_val \
    --ckpt $OUT/D0-climbmix/D0-climbmix.pt --device cuda --dtype bf16 --window 2048 \
    --batch-size 8 --max-tokens-per-domain 20000000 --out $OUT/D0-climbmix/p0.json

# CORE-lite (6 tasks x 2000)
$PY tools/eval_core.py --ckpt $OUT/D0-climbmix/D0-climbmix.pt --bundle $ROOT/core_bundle/eval_bundle \
    --max-per-task 2000 --task arc_easy --task arc_challenge --task hellaswag \
    --task winogrande --task boolq --task piqa --out $OUT/D0-climbmix/core_lite.json
```

（也可是 `bash tools/r90_stage1_arms.sh` 的同一模板改数据目录/mix；CORE bundle 与 `p0_eval`
由 Stage-0 产出：`tools/r90_build_climbmix_val.py` 建 P0、`eval_bundle` 由 `r90_fetch_assets.sh` 拉。）

---

## 8. 成本汇总（¥2.78/GPU-h、204k tok/s）

| 项 | 数量 | wall | 成本 |
|---|---:|---:|---:|
| 数据下载（CPU 实例，hf-mirror） | 3.55GB | 25–115min | ¥0.05–0.2 |
| tokenize + 去污染（CPU 实例 0.5 核，实测 84.5k tok/s） | ~2.1B tokens | 7–9h | ¥0.7–0.9 |
| （推荐）tokenize 在 GPU 机 CPU 上 `--workers 8–12` | ~2.1B tokens | 10–15min | GPU 训练时段零额外成本 |
| **7 臂训练+评测（D0–D6）** | 7×0.30B | 3.5 GPU-h | **¥9.59** |
| Stage-3 合计（推荐方案） | | | **≈¥9.8–10.2** |
| 若 6 臂（省 D6 或 D2） | 6×0.30B | 3.0 GPU-h | ¥8.22 |

预算护栏：Stage-3 GPU 硬上限 ¥10；数据准备放 CPU 实例不计入 GPU 预算（¥0.1/h 级）。
每臂跑完立刻 eval + 落 JSON；7 臂按顺序跑完即关机，**不补跑**（不足信息量时再申请预算）。

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| hf-mirror 慢/断流 | 脚本已尺寸校验+断点续传+重试；必要时 DCLM 减到 6 文件（0.84GB）、FineWeb 只 1 文件（2.15GB）；两者都拿不到时退化到"ClimbMix + 现有 P5 语料"对照，并注明 |
| DCLM 0.3B tokens 相当于 2–3 个 ~140MB 文件流式处理 | 默认 10 文件覆盖全部 global shard，多样性足够；`--tokens 0.5B` 的语料每个 arm 只用 0.3B |
| 去污染 filter 模式 CPU 太慢 | 默认 sample 估计；只有 hit_rate>0.1% 才 filter，且 filter 只对 DCLM/FineWeb 各做一次（0.4-0.5B tokens，CPU 实例 1–2h，¥0.1/h） |
| 单 seed 噪声 | 固定 tokens 比较，BPB 的 run-to-run 噪声远小于 0.01 BPB（P2 A/B 报告：0.01–0.03 BPB 可分辨）；若最优与 D0 差距 <0.01 → 用 D7/0.25B 复测（§5 变体） |
| DCLM/FineWeb 与 ClimbMix 高度同源 → 实验信息量低 | §4 步骤 3 的交叉重叠率采样，若 >30% 则把结论重心放到 FineWeb-HQ 与 mix，并在报告中量化同源度 |
| CORE-lite 噪声 / bundle 未就绪 | CORE-lite 只做副指标；bundle 缺失时只报 P0（Stage-0 已确保 22 题 bundle 拉到） |
| 预处理差异（Marin §2.5 尾随空格/格式） | 所有源只做 tokenizer 直通、不做额外 text 变换（去污染归一化不影响语料）；报告各源 tokens/byte（fertility）作为 sanity：若某源偏离 >15% 提示格式差异 |
| 磁盘/实例 TTL | 原始+语料峰值 <20GB/49GB；数据准备与 Stage-0 共用 CPU 实例，不新开实例 |

---

## 10. 立即执行清单（给 Lead）

1. **现在（CPU 实例，零 GPU 成本）**：`bash tools/r90_data_fetch_dclm.sh` + `N_FILES=1 bash tools/r90_data_fetch_fineweb.sh`（可与 ClimbMix 下载并行；~25–115min）。拉完各自日志里有 `summary: ok=.. bad=0`。
2. 用 `--decontam-val .../climbmix/shard_06542.parquet` 的 **sample 模式**建 4 个语料（§3）；
   看 `decontam.json` 的 hit_rate 决定是否追加 filtered DCLM/FineWeb（§4 判定）。
3. `--combine` 建 `p2_mix_web`；`--check` 全部通过。
4. GPU 上线后按 §7 跑 D0–D6（7×0.30B，3.5h，¥9.6），每臂立刻 P0 + CORE-lite，结果落
   `runs/r90/stage3/summary.json`；按 §6 判定规则给出 Stage-4 数据配方建议。

---

## 11. 执行结果（2026-11-12 03:1x，task-15 收尾；5090 `<instance-id>`）

> 说明：本节是实际执行记录（与 §1–§10 的计划对应）。所有产物在 `/root/autodl-tmp/nanochat/`（50GB 数据盘）。

### 11.1 下载

| 源 | 结果 |
|---|---|
| DCLM-Baseline | **20/20 文件, 2.9 GB**（`global-shard_00..10` 各 2 文件）→ `dclm/`；fetch 脚本 rc=0，尺寸校验通过 |
| FineWeb-Edu 10BT sample | **3/3 文件, 6.457 GB**（2,152.8/2,152.2/2,151.8 MB）→ `fineweb_edu/sample/10BT/`；fetch 脚本 rc=0 |

### 11.2 32k 去污染语料（`tools/r90_stage3_prep2.sh`，13-gram filter vs `shard_06542`，`--workers 8`）

| 语料 | tokens | 源文件 | decontam filtered docs | build 时间 | `--check` |
|---|---:|---:|---:|---:|---|
| `p2_dclm_dc_32k` | **220,638,481** | 20 | 542 | 73.4 s | **OK** (0.82 GiB) |
| `p2_fwe_dc_32k` | **250,000,000**（cap） | 3 | 5,139 | 141.5 s | **OK** (0.93 GiB) |
| `p2_fwehq_dc_32k`（score≥3.5） | **250,000,000**（cap） | 3 | 6,828 | 173.4 s | **OK** (0.93 GiB) |
| `p2_mix_dc_32k`（combine） | **855,378,884** | symlink | — | <1 s | **OK** |

- `p2_mix_dc_32k` = `climbmix=384.7M + dclm=220.6M + fineweb=250.0M`（symlink 到源域 `web.u32.bin`），
  builder 给出的默认配比 `--mix climbmix=0.4498,dclm=0.2579,fineweb=0.2923`。
- **D1 臂 token 预算按实际值 220.6M**（DCLM 原始 2.9GB jsonl.zst 只够 ~220.6M tokens，未到 250M cap）；
  D2/D3 = 250M；mix 可用 3 域任意比例（总 855.4M）。
- 去污染：13-gram（val n-gram index cap 20M grams，mode=filter）——**实测 hit rate**：
  DCLM **0.309%**（542/175,201 docs）、FineWeb **2.082%**（5,139/246,784）、FineWeb-HQ **2.831%**
  （6,828/241,152）→ DCLM 与 ClimbMix val 重叠最低，mix 配方宜偏 DCLM。
- **bytes/token 未落盘**：builder 未持久化源文本字节数（原始数据建完即删）；以 32k tokenizer 在
  ClimbMix 上实测的 4.5888 B/tok 作参考，英文 web 源同量级（4.5–4.8）。
- 建完即删原始源（DCLM 2.9GB + FineWeb 6.46GB）→ 数据盘占用回到 ~36/50 GB（15GB free）。

### 11.3 同时进行的 Stage-4 语料（非 task-15）

`p2_final_129k` 已建到 **4,033,742,930 tokens / 77 shards / 15.03 GiB / 4.8105 B/tok**，`--check PASS`
（追加式增量构建 + 原始 shard 用后即删；详见 `tools/r90_final129k_loop.sh`）。

### 11.4 事件

2026-11-12 03:14 CST，5090 `<instance-id>` 被 AutoDL 以 `shutdown_by_running_error` 关机（系统/宿主层，
非本任务写入所致；所有语料 manifest 原子写入且 `--check` 已通过）。**03:23 重启后复核：全部产物完好**：
`p2_final_129k` 仍为 4,033,742,930 tokens / 77 shards / `--check PASS`；4 个 Stage-3 语料
（220,638,481 / 250,000,000 / 250,000,000 / 855,378,884 tokens）`--check OK`；数据盘 25 GB free。
未 tokenize 的 ClimbMix 原始 shard 在关机中丢失（语料已封顶，无影响）。
