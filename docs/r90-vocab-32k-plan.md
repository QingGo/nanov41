# R90 Stage-2 — 32k vocab / head 消融实施方案（vocab ablation plan）

> 状态：2026-11-12 起草（head-cache）。目标：执行 `docs/search-plan-2026-11-12.md` §5 的
> **vocab/head 消融**（129,280 vs 32,768），定量「head 成本」对 val_bpb / CORE / ¥ 的影响。
> 已按 Lead 决策收敛：**只用 aligned 家族 + `--config-json`（不需要改 p2_train.py）**、
> **两臂都 `--engram none`**、语料用本方案的 retokenizer（129k 控制臂也重切，保证 doc/EOS 策略一致）。
> 工具：`tools/r90_vocab_{train_tok,retokenize,configs,eval_panel}.py`；一键 runner：
> `tools/r90_vocab_stage2.sh`（`STAGE=prep|train|eval|all`、`SUBSET="A0 A1 A2"`、`ORDER=ABBA`、
> `DRY=1` 只打印命令；不启动/关机实例、不抢 GPU 锁，由 Lead 放进自己的 arm runner 执行）。

## 0. TL;DR

- aligned d12（dim 512，vocab 129,280）的 head/embed = **132.4 MFLOPs/token**，占 **fwd 183.0
  MFLOPs/token 的 72.3%**；vocab=32,768 后 head = 33.55M，**fwd FLOPs/token 84.16M = 0.460×**，
  head 占比降到 **39.9%**（config 只改 `vocab_size`，body 完全不动）。
- **已在实例上训出最终 32k tokenizer**（byte-level BPE，`tokenizers`，gpt2 ByteLevel 切分，
  2.0 GB ClimbMix / 671k docs / 105 s @8 threads）。ClimbMix val shard holdout 上 **4.6131
  bytes/token**；pinned DeepSeek 129,280 tokenizer 在**同一文本**上 **4.8387** → 32k 需要
  **4.9% 更多 token**（同字节文本）。⇒ 每字节训练成本比 ≈ `0.460 × 1.049 = 0.482`，即 **2.07×
  更便宜/字节**（FLOPs 口径）；按吞吐 1.6× 保守折算，同 ¥ 下 32k 臂可多看 **~1.5× 原始文本**。
  （注：早期 30MB 预训练时测得 0.845× 是 `Split` 无 pretokenise 的 bug，已修正并对齐，见 §2。）
- 成本（5090 ¥2.78/h，0.3B tokens/臂 @ 150k tok/s）：129k 臂 ≈ **¥2.0**（train+eval）、
  32k iso-token 臂 ≈ **¥1.4**、32k iso-cost 臂（0.48B tokens）≈ **¥2.0**；
  **A0+A1+A2 三臂 ≈ ¥5.5**（+65k 臂 +¥1.7，可选）。
- 数据：一条 shard 解压后 ≈ **254 MB 文本**（129k ≈ 52.5M tokens, 32k ≈ 55.1M tokens）；三臂
  全部（含 iso-cost 2.21 GB）只需 **~10 shards**，均已就绪（25 train shards + val shard 在箱上）。
- **Engram 不能与 32k tokenizer 同时开**（硬 assert，见 §6）→ 两臂 Engram off，消融只测
  embed+head 尺寸 + tokenizer 两个变量（Lead 已拍板）。

## 1. 机制：head 是每 token 成本的主导项

`docs/aligned/d12_aligned.json` 的 accounting（已由 `tools/r90_vocab_configs.py` 精确重算）：

| vocab | embed+head 参数 | active 参数 | fwd FLOPs/token | head 占比 | 相对 129k 的 fwd FLOPs |
|---:|---:|---:|---:|---:|---:|
| 129,280（基线） | 132,382,720 | 91,495,560 | 182,991,120 | 72.3% | 1.000× |
| **32,768** | **33,554,432** | **42,081,416** | **84,162,832** | **39.9%** | **0.460×** |
| 65,536（可选中点） | 67,108,864 | 58,860,832 | 117,717,264 | 57.0% | 0.643× |

- 只改 `engine_config.vocab_size`；`embed+head = 2·V·dim`、`active += (V−V0)·dim`、
  `fwd_flops = 2·active` 是完全解析的，因此 accounting 重算是精确的（tool 里有恒等式断言）。
- 参数/显存侧：embed+head 少 98.8M 参数（bf16 权重 ≈ −198 MB，Sinkhorn/Adam 状态同比），
  backbone 752.2M → 653.4M；**没有 Engram 时**总参数 653.4M。
- 训练 FLOPs/token ≈ 3×fwd（+attention），所以 **iso-token 下训练成本比也是 0.46×**（attention
  会略微抬高 32k 臂相对成本，量级 ~2-5%）。

## 2. Tokenizer：32k byte-level BPE（nanochat 口径）

`tools/r90_vocab_train_tok.py`：byte-level BPE、无 normalizer（lossless，任意 UTF-8 可逆）、
`initial_alphabet`=256 字节符号、默认 GPT-4 风格 split pattern（nanochat rustbpe 的
`SPLIT_PATTERN`；`--split-pattern gpt2` 可切换做 A/B）、special = `<|pad|>=0,<|bos|>=1,<|eos|>=2`
（含在 32,768 内）。BPE 训练对固定语料顺序确定性；同语料同参数逐位可复现。

**已测（实例 <instance-id>，最终产物 `/root/autodl-tmp/nanochat/tok32k/tokenizer.json`）**：
训练 2.0 GB ClimbMix 文本 / 671,006 docs，**104.7 s**（8 rayon threads）；
holdout = val shard `shard_06542` 的前 4 MB（1,338 docs，与训练文本不同 shard）：

| tokenizer | vocab | holdout bytes/token | holdout tokens（同 5.93MB 文本） | 相对 token 数 |
|---|---:|---:|---:|---:|
| **本方案 32k BPE（final）** | 32,768 | **4.6131** | 867,207（4MB 计） | **1.049×** |
| pinned DeepSeek 129,280 | 129,280 | 4.8387 | 同文本 1.049× 更少 | 1.000× |
| 32k BPE，60 MB 预训练（本机交叉验证） | 32,768 | 4.557 | — | 1.052× |

- 交叉验证：把 60MB 训练的 gpt2-pattern tokenizer 拿到本机、在**同一** holdout 上测 → 4.557
  vs DS 4.794（+0.5% 期望一致）⇒ 数字稳定。round-trip **精确**（2,000/2,000 docs + 12 unicode
  边界用例），byte-table 严格校验 500 docs 全过（`sum(token_bytes[ids]) == len(utf8)`）。
- **结论修正**：32k tokenizer 在此语料上压缩率略**差于** 129k DS tokenizer（同文本需 +4.9% token），
  但 head FLOPs 0.46× 远大于该劣势 ⇒ **净 2.07× 更便宜/字节**。这仍是最大的成本杠杆。
- **实现 caveat（重要，已修正）**：`tokenizers` 的 `pre_tokenizers.Split(pattern=<str>)` **静默不切分**
  （整篇文档成为一个 pre-token → BPE 学到跨词长 token，得到虚假的 5.67 B/tok）；必须传
  `tokenizers.Regex(...)`（已修）。修正后 gpt4 风格 pattern 可正确切分，但比内置 ByteLevel GPT-2
  regex 慢 **~15-30×**（60MB: >90s vs 6s）→ 工具默认 `--split-pattern gpt2`（标准 GPT-2 切分，
  数字/空白按 GPT-2 规则）。要与 nanochat 的 rustbpe 完全同 pattern 需换 rustbpe（未安装），本消融
  不必要：BPB/CORE 都 tokenizer 无关，两臂又是同一 tokenizer 对比 vocab。
- 复现（本机 / 实例）：见 §4.1、§7。

## 3. 语料：两臂独立重切（同文本、同 doc/EOS 策略）

`tools/r90_vocab_retokenize.py`：parquet `text` → P2 uint32（`manifest.json` format
`nanov41-p2-corpus/1` + `web.u32.bin`），**排除 val shard `shard_06542.parquet`**，每 doc 后加 EOS
（`--doc-sep eos` 默认；129k 用 `<｜end▁of▁sentence｜>` id 1，32k 用 `<|eos|>` id 2），
多进程 per-shard 并行、可断点续跑、按 shard 边界取整（`--truncate` 可精确截断），manifest 记录
tokenizer sha/vocab/eos、每 shard doc/token/源字节数、整体 **bytes/token**。

为什么 129k 控制臂也要重切：Stage-1 的 `tools/make_climbmix_p2.py` **不写 doc 分隔符**，而本消融
要求两臂 doc/EOS 策略完全一致；重切同一条 129k 语料是唯一干净做法（CPU 成本可忽略）。

## 4. 精确命令（在 GPU-mode 实例的 CPU 侧跑；不占 GPU）

变量：`ROOT=/root/autodl-tmp/nanochat`、`PY=/root/miniconda3/bin/python`（4090）或
`/root/t141/bin/python`（5090，两者都带 tokenizers/pyarrow/torch）；
`RAW=$ROOT/climbmix`、`TOK32=$ROOT/tok32k/tokenizer.json`。

### 4.1 训练 32k tokenizer（~1-2GB 文本，16+ 核，预计 5-20 min）

```bash
# 注意：--train-parquet-dir 会把目录下所有 *.parquet 都算进去；若仍有 shard 在下（partial）会告警
# 跳过（工具已容错），但最好显式传完整 shard 列表：
$PY tools/r90_vocab_train_tok.py \
  --train-parquet $(ls $RAW/shard_0*.parquet | grep -v 06542) \
  --sample-bytes 2000000000 --vocab-size 32768 --split-pattern gpt2 \
  --holdout-parquet $RAW/shard_06542.parquet --holdout-bytes 4000000 --holdout-docs 2000 \
  --compare-tokenizer /root/dsv41/tokenizer.json --engram-check --engram-ref-check \
  --out $ROOT/tok32k/tokenizer.json
# -> tokenizer.json + tokenizer.stats.json（bytes/token 双 tokenizer、roundtrip、engram 风险数）
# 实测（16 核箱、8 threads）：2.0GB 文本 105s；tokenizer.json 2.3MB；holdout 4.6131 vs 4.8387 B/tok
```

### 4.2 重切两条语料（CPU；16 workers；预计各 3-10 min）

```bash
# 两边都截到**正好 0.30B tokens**（iso-token 对比；--truncate 允许 shard 中间截断）。
# 一条 shard ≈ 254MB 解压文本 → 129k 52.5M tok / 32k 55.1M tok；6-7 shards 足够。
# 实测：6-7 个 worker 各 ~135s/7 shards（129k 2.3M tok/s total）。
$PY tools/r90_vocab_retokenize.py --shards-dir $RAW --exclude-shard shard_06542.parquet \
  --min-shard-bytes 85000000 --limit-shards 7 --tokenizer /root/dsv41/tokenizer.json \
  --out $ROOT/p2_climbmix_129k --tokens 300000000 --truncate --workers 7
$PY tools/r90_vocab_retokenize.py --shards-dir $RAW --exclude-shard shard_06542.parquet \
  --min-shard-bytes 85000000 --limit-shards 8 --tokenizer $TOK32 \
  --out $ROOT/p2_climbmix_32k --tokens 300000000 --truncate --workers 8
# iso-cost A2 若要单独更大 32k 语料：--tokens 480000000 --limit-shards 9（同一条 corpus 也可
# 用 --tokens 480M 训练时截断；p2_train --tokens 控制训练步数，语料多切一些更灵活）
```

### 4.3 生成 aligned 配置变体（CPU，秒级；已生成本地副本）

```bash
$PY tools/r90_vocab_configs.py --vocab-sizes 32768,65536 \
  --base-aligned docs/aligned/d12_aligned.json --out-dir runs/r90/configs --mode aligned
# 产物：runs/r90/configs/aligned/d12_aligned_v32768.json、…_v65536.json、index.json
# （repo 内也提交了 docs/aligned/d12_aligned_v32768.json 供 --config-json 直接引用）
```
`--config-json` 路径**无需改 p2_train.py**：`tools/aligned_build.install_aligned()` 会注册
`CONFIGS["d12_aligned"]` 并以 `--engram none` 关掉 Engram；JSON 里只改 `engine_config.vocab_size`
与派生 accounting，其余（engram 字段、aligned_spec、optimizer…）逐字节不变（tool 会校验）。

### 4.4 32k val BPB 面板（CPU；整 val shard 预计 3-10 min）

```bash
$PY tools/r90_vocab_eval_panel.py --shard $RAW/shard_06542.parquet \
  --tokenizer $TOK32 --out $ROOT/p0_eval_32k --window 2048 --batch-docs 512
# nanov41-eval-data/1：exact byte table（32,768）、EOS typed IGNORE、含 nanochat anchors
# 129k 面板沿用 Lead 已在建的 $ROOT/p0_eval（同 shard、同 conventions）
```

### 4.5 训练 + 评测三臂（或 `STAGE=train|eval bash tools/r90_vocab_stage2.sh`）

**优化器/LR 建议（arm 脚本已内置）**：与 Stage-1 同协议 `--optimizer adamw --batch 16 --seq 1024
--warmup 300 --lr-schedule cosine`，`--peak-lr 2.6e-3`（= Stage-1 P2-d12 anchor 3e-3 @dim384 按
width 缩放到 aligned d12 dim 512：3e-3·√(384/512)）；两臂完全相同，消融是 within-pair 比较。
（若要与 Stage-4 的生产 v41 配方对齐，可改 `--optimizer v41 --peak-lr 8.22e-4`；不要两臂混用。）

公共 env（生产栈，两臂完全一致；见 `tools/r90_stage1_env.sh` + R79/R82/R83/R85/R88）：

```bash
export NANOV41_HEAD_FP8=v2 NANOV41_HEAD_FWDGRAD=1 NANOV41_HEAD_WCACHE=1 \
       NANOV41_CE_CHUNK=8192 NANOV41_MHC_V4=1 NANOV41_V4_MODE=split \
       NANOV41_MOE_LAYOUT=fast NANOV41_ATTN_DKV_BF16=1 NANOV41_MOE_DWS_INLINE=1 \
       NANOV41_MOE_STACKED=1 NANOV41_MOE_TILE=64,64,64,4,3 NANOV41_MOE_DW_TILE=128,64,64,4,3 \
       NANOV41_ATTN_V2=0 FAST_ATTN_TL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```bash
# A0 129k iso-token（控制臂）
$PY tools/p2_train.py --arm A0-129k-isotok --optimizer v41 \
  --config-json docs/aligned/d12_aligned.json --engram none \
  --data-dir $ROOT/p2_climbmix_129k --mix web=1.0 \
  --batch 16 --seq 1024 --peak-lr 8.22e-4 --lr-schedule constant --warmup 300 \
  --balance-loss 1e-4 --bias-speed 1e-3 --z-loss 0 --grad-clip 1.0 \
  --tokens 300000000 --log-every 500 --nan-check-every 8 \
  --out runs/r90/stage2/A0-129k-isotok

# A1 32k iso-token
$PY tools/p2_train.py --arm A1-32k-isotok --optimizer v41 \
  --config-json runs/r90/configs/aligned/d12_aligned_v32768.json --engram none \
  --data-dir $ROOT/p2_climbmix_32k --mix web=1.0 \
  --batch 16 --seq 1024 --peak-lr 8.22e-4 --lr-schedule constant --warmup 300 \
  --balance-loss 1e-4 --bias-speed 1e-3 --z-loss 0 --grad-clip 1.0 \
  --tokens 300000000 --log-every 500 --nan-check-every 8 \
  --out runs/r90/stage2/A1-32k-isotok

# A2 32k iso-cost（同 A0 的 wall/¥；0.48B tokens；语料用重建后的 480M 版）
$PY tools/p2_train.py --arm A2-32k-isocost ...同 A1... --tokens 480000000 \
  --out runs/r90/stage2/A2-32k-isocost
```

评测（P0 BPB 用**各自 tokenizer 的面板**；BPB tokenizer 无关，可直接比）：

```bash
# 129k 臂
$PY tools/eval_t0.py --dataset $ROOT/p0_eval --domains climbmix_val \
  --ckpt runs/r90/stage2/A0-129k-isotok.pt --device cuda --dtype bf16 \
  --window 2048 --batch-size 8 --max-tokens-per-domain 20000000 \
  --out runs/r90/stage2/A0.p0.json
# 32k 臂（panel + ckpt vocab 都是 32,768；不需要 --allow-vocab-mismatch）
$PY tools/eval_t0.py --dataset $ROOT/p0_eval_32k --domains climbmix_val \
  --ckpt runs/r90/stage2/A1-32k-isotok.pt --device cuda --dtype bf16 \
  --window 2048 --batch-size 8 --max-tokens-per-domain 20000000 \
  --out runs/r90/stage2/A1.p0.json
# CORE-lite（tokenizer 无关的 accuracy；每臂必须用**自己的** tokenizer 编码 prompt）
$PY tools/eval_core.py --ckpt runs/r90/stage2/A1-32k-isotok.pt \
  --bundle $ROOT/core_bundle/eval_bundle --max-per-task 2000 \
  --task arc_easy --task arc_challenge --task hellaswag --task winogrande \
  --task boolq --task piqa --tokenizer $TOK32 \
  --out runs/r90/stage2/A1.core_lite.json
# A0 用 --tokenizer /root/dsv41/tokenizer.json
```

## 5. 成本、数据与判定

吞吐假设（**待实测校正**；依据：生产 aligned-d12 全栈 160k tok/s @b2/s8192，搜索臂 b16/s1024
小 batch 更慢，且本消融无 Engram）：129k 臂 150k tok/s、32k 臂 240k tok/s（≈1.6×，FLOPs 上限
2.17× 的保守折算）、65k 臂 190k tok/s。GPU ¥2.78/h（5090；4090 ¥1.98/h 更便宜）。

| arm | vocab | tokens | 相对成本口径 | 训练 GPU-h | 训练 ¥ | 语料文本量 | eval ¥ | **单臂合计 ¥** |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| A0 控制 | 129k | 0.300B | iso-token（基线） | 0.556 | 1.54 | 1.45 GB | ~0.50 | **2.04** |
| A1 32k | 32k | 0.300B | iso-token（更便宜） | 0.347 | 0.97 | 1.38 GB | ~0.45 | **1.42** |
| A2 32k | 32k | 0.480B | **iso-cost**（≈A0 ¥） | 0.556 | 1.54 | 2.21 GB | ~0.45 | **1.99** |
| A3 65k（可选） | 65k | 0.300B | iso-token | 0.439 | 1.22 | 1.38 GB | ~0.45 | **1.67** |
| A4 32k（可选） | 32k | 0.315B | iso-data（同文本量） | 0.293 | 0.82 | 1.45 GB | ~0.45 | **1.27** |

- **A0+A1+A2 ≈ ¥5.5**（推荐主实验）；A0+A1 对 ≈ ¥3.5（先筛“32k 是否掉点”）；+A3 ≈ +¥1.7；
  2 个 seed/ABBA 复核胜者约 +¥2-3。tokenizer 训练 + 重切 + 面板 = CPU（no-GPU 模式 ¥0.1/h 或与 GPU 重叠）≈ 0 额外 GPU 成本。
- 数据需求：**共享原始文本池 = max(A2) ≈ 2.21GB 解压文本 ≈ 9 shards**（一条 shard ≈ 254MB 文本）。
  只跑 A0+A1 需 1.45GB ≈ 6 shards。当前箱上 25 train shards + val shard 已全部就绪。
- 判定规则（建议）：
  1. **A1 vs A0（iso-token）**：BPB/CORE 差距在 Stage-0 实测噪声内（BPB ≤ ~0.005、CORE-lite ≤2σ）
     且 A1 吞吐实测 ≥1.4× → 32k 是明确 ¥/质量赢家；
  2. **A2 vs A0（iso-cost）**：A2 显著更好（BPB 更低 ≥ 噪声 2×）→ “同 ¥ 换更多数据”路线成立，
     可直接进 Stage-4 预算匹配验证；
  3. 若 32k 反而更差 → 记录“129k head 成本被 tokenizer 压缩率部分抵消”的负结果，Stage-3/4 不再
     优先 vocab，转 HP/数据。
- **iso-data（A4）是最干净的因果臂**（同文本、只变 tokenizer+head），预算允许时建议跑。

## 6. Engram 风险分析（为什么必须 `--engram none`）

- `ref/engram.py:146`：`build_compressed_token_map(tokenizer)` 的返回值必须 **等于**
  `engine_config.engram_compressed_vocab_size`（129k tokenizer = **99,092**），否则 **hard assert**。
  该 compressed 空间由 tokenizer 词表按归一化折叠得到，**32k 词表最多 ~32k 个 key** → 必不相等。
  **已实测**（`r90_vocab_train_tok.py --engram-check`，纯 `tokenizers` 实现、无需 torch；在 129k 上
  精确复现参考实现的 99,092 作为交叉验证）：
  - pinned 129,280 → **99,092**（`matches_129k_config: true`）
  - 本方案 32k BPE（30MB 预训练版）→ 24,942；**最终 32k tokenizer（2GB 预训练）→ 23,225**
    （`pad_id=0`）≠ 99,092 → assert 必失败。
- `verify`：pure 实现在 129k tokenizer 上得到 99,092 == 参考值，故 32k 的 24,942 可信（实例上可用
  `NANOV41_R90_ALLOW_TORCH=1 --engram-ref-check` 直接对拍 `ref.engram`）。
- 且 `nanov41/nano_engram.py:107 apply_preset()` 把 `engram_compressed_vocab_size` 硬编码回
  99,092，`tools/aligned_build.install_aligned()` 也不透传该字段 ⇒ **当前代码路径无法给 32k
  tokenizer 开 Engram**。
- 若未来必须保留 Engram（Stage-4 生产候选），最小补丁（Lead 决定是否做，不在本文件改）：
  1. `nano_engram.ENGRAM_PRESETS[name]` 允许携带 `engram_compressed_vocab_size` / `engram_pad_id`，
     `apply_preset()` 用 preset 值（fallback 99,092）代替硬编码；
  2. `aligned_build.install_aligned()` 把 JSON `engram` 段的这两个字段透传进 preset；
  3. variant JSON 的 `engine_config.engram_compressed_vocab_size` 填实测 compressed size。
  重置 hash multipliers 后表从头训练即可（新 tokenizer 的哈希空间自洽）。
- 本消融两臂均无 Engram → 绝对 BPB/CORE 与生产（972M 参数含 220M Engram）不可直接比，**只在
  两臂之间比较**；报告里注明。

## 7. 复现 / 本机验证（已做）

```bash
# 本机（Mac，tokenizers-only；训练 32k/30MB 约 4 min）
python3 tools/r90_vocab_train_tok.py --train-jsonl /tmp/r90_vocab/train_sample.jsonl \
  --sample-bytes 30000000 --vocab-size 32768 --split-pattern gpt4 \
  --holdout-jsonl /tmp/r90_vocab/holdout_sample.jsonl --holdout-bytes 6000000 \
  --holdout-docs 2000 --compare-tokenizer /tmp/r90_vocab/ds_tokenizer.json \
  --out /tmp/r90_vocab/tok32k_local.json
# 端到端验证（本地 tiny parquet，两个 tokenizer 都跑过）
python3 tools/r90_vocab_retokenize.py --shards-dir /tmp/r90_vocab/tiny_shards \
  --tokenizer /tmp/r90_vocab/tok32k_local.json --out /tmp/r90_vocab/tiny_p2_32k \
  --tokens 30000 --truncate --workers 2
python3 tools/r90_vocab_eval_panel.py --shard /tmp/r90_vocab/tiny_shards/shard_06542.parquet \
  --tokenizer /tmp/r90_vocab/tok32k_local.json --out /tmp/r90_vocab/tiny_p0_32k --window 256
python3 tools/r90_vocab_configs.py --vocab-sizes 32768,65536 --out-dir /tmp/r90_configs --mode both
```
验证结果：byte-level round-trip 精确（2,000 docs + 12 unicode edge cases）、byte-table 严格校验
500 docs 全过、corpus EOS 数 == docs、面板 `EvalDataset.verify()` 通过、config 变体仅 vocab+accounting
改变（tool 断言）。`r90_vocab_configs.py` 的 p2_train-path 安装回放需要 torch，只在实例上跑
（`NANOV41_R90_ALLOW_TORCH=1`）。**实例实测（<instance-id>，16 核）**：
tokenizer 2.0GB/105s（8 threads）；retok 129k 7 shards × 52.5M tok ≈ 135s/7 workers；
32k panel（val shard，~55M tokens）≈ 60-90s（6 rayon threads）；产出语料与面板见 §8 交付物。

## 8. 风险 / 注意

1. **tokenizer 训练样本量**：30MB 只够验证；正式模型用 ~1-2GB（nanochat 用 ~2B chars）。压缩率
   对样本量敏感，正式 `stats.json` 的数字才写进最终报告。
2. **域内 tokenizer 优势**：32k 在 ClimbMix 上训练，DS tokenizer 是通用词表 → 15.5% token 节省
   含“域内”成分；这正是 nanochat 式端到端管线的真实收益，但结论需标注。
3. **seq=1024 搜索臂 ≠ 生产 8192**：vocab 若进 Stage-4，必须在生产 seq/优化器下复验（attention
   成本占比变化会改变 32k 的相对收益）。
4. **fp8 head 与 V=32768**：需在 A1 首步 sanity（`_scaled_mm` N=32768、triton CE kernels 任意 V
   均支持；冒烟 20 步 + nan-check）。engram=None 时 `NANOV41_DS_TOKENIZER_JSON` 不需要设置。
5. **`--config-json` 的注册名固定为 `d12_aligned`**（`install_aligned` 硬编码 name）：日志里
   `[capture] cfg=d12_aligned` 不代表 129k；用 `--arm` 名字区分（A0/A1/A2），报告 JSON 的
   `config_name` 会重复，不要用它区分臂。
6. **数据下载依赖**：iso-cost 臂要 2.72GB 原始文本（~40 shards）；当前 11。若下载受限，先跑
   A0+A1（1.7GB）并降级 A2 为 A4（iso-data，1.44GB）。
7. 每臂 0.3B tokens 的训练在 b16/s1024 下约 20-35 min；三个臂 + eval 合计约 1.5-2 GPU-h，建议
   与 tokenizer/重切 CPU 工作交错，避免 CPU 成为 GPU 空转等待。
8. 消融只测 vocab 一个变量：两臂除 tokenizer/语料/vocab 外**所有超参、seed、数据顺序、env、batch/
   seq 必须一致**；数据顺序按 shard 名排序 + 固定 seed（与 Stage-1 协议一致）。

## 9. 需要 Lead 确认/执行

1. 下载至少 **25 shards**（A0+A1）或 **40 shards**（含 A2 iso-cost）到 `$ROOT/climbmix`；
   `shard_06542`（val）必须下载用于 32k 面板。
2. 给一个 **CPU 窗口**（GPU-mode 实例，16-32 核）：跑 §4.1-4.4（tokenizer→两条语料→32k 面板），
   预计总 wall 20-40 min（可与 GPU 训练交错）。
3. 确认 §5 的三臂预算（≈¥5.5）与判定规则；确认后我改由你在 arm runner 中执行（本方案不直接跑 GPU）。
4. p2_train.py **无需改动**（aligned 路径已够）；`tools/r90_vocab_configs.py --mode plain` 只是为
   configs.py 家族预留的 spec（若未来要在 d8/d16… 上做 vocab 臂，需要 ~6 行 `--config-variant`
   loader：`CONFIGS[spec["name"]] = dict(CONFIGS[spec["base_config"]], **spec["overrides"])`；
   Lead 已决定本轮不需要）。

## 10. 交付物 / 运行产物（实例 <instance-id>）

| 产物 | 路径（实例） | 备注 |
|---|---|---|
| 32k tokenizer + stats | `/root/autodl-tmp/nanochat/tok32k/tokenizer.json` / `.stats.json` | vocab 32,768；holdout 4.6131 B/tok；engram compressed 23,225 |
| 129k 控制语料 | `/root/autodl-tmp/nanochat/p2_climbmix_129k/` | **300,000,000 tokens**, 483,888 docs, 1.442GB 文本, **4.8077 B/tok**, eos=1，val shard 排除 |
| 32k 语料 | `/root/autodl-tmp/nanochat/p2_climbmix_32k/` | 先 300,000,000 tokens (461,828 docs, 1.376GB, **4.5882 B/tok**, eos=2)，随后重建为 **480M** tokens 供 A2/A4（前 300M 与 A1 完全一致） |
| 32k val BPB panel | `/root/autodl-tmp/nanochat/p0_eval_32k/` | **55,086,667 tokens** / 84,992 docs / 85,022 IGNORE；token_bytes=32,768，window 2048，verify ✓ |
| 129k val BPB panel | `/root/autodl-tmp/nanochat/p0_eval/` | Lead 构建：52,429,588 tokens, token_bytes=129,280, window 2048 ✓ |
| 配置变体 | `runs/r90/configs/aligned/d12_aligned_v{32768,65536}.json`（+`docs/aligned/d12_aligned_v32768.json`） | `--config-json` 直接用，无需改 p2_train |
| arm 脚本 | `tools/r90_vocab_arms.sh` | A0/A1/A2/A4 + P0/CORE-lite；adamw, b16×1024, lr 2.6e-3, warmup 300 |
| runner | `tools/r90_vocab_stage2.sh` | prep/train/eval/all 全流程 |

## 11. 生产家族 depth probe：d12 vs d16 vs d20 @ 32k vocab（Lead 追加，可选）

目标：在**同一生产家族/aligned 规则**下做小规模等 wall（一阶等 FLOPs）深度 probe —— 全部
`vocab=32768` + `--engram none`，**无需改 p2_train**（`--config-json` + 现有路径）。

生成方式（实例上）：先用 `tools/make_aligned_config.py` 生成 base（accounting 需要 torch，故在
GPU 机上跑，~20 s/个），再用 `tools/r90_vocab_configs.py` 只替换 `engine_config.vocab_size=32768`
并精确重算派生 accounting（engram/spec/optimizer 字段逐字节不变，tool 断言）：

```bash
# on the instance (CPU, torch available)
$PY tools/make_aligned_config.py --layers 16 --dim 640 --seq 8192 --out /tmp/aligned/d16_aligned.json
$PY tools/make_aligned_config.py --layers 20 --dim 640 --seq 8192 --out /tmp/aligned/d20_aligned.json
$PY tools/make_aligned_config.py --layers 20 --dim 768 --seq 8192 --out /tmp/aligned/d20d768_aligned.json
python3 tools/r90_vocab_configs.py --base-aligned /tmp/aligned/d16_aligned.json \
  --vocab-sizes 32768 --out-dir /tmp/r90cfg --mode aligned     # 其余同理
```
产物（repo 已提交 `--config-json` 可直接用）：`docs/aligned/d16_aligned_v32768.json`、
`docs/aligned/d20_aligned_v32768.json`（dim 640）；可选 `runs/r90/configs/aligned/d20d768_aligned_v32768.json`。

| model @ vocab 32768 | dim | layers | active 参数 | **fwd FLOPs/token** | head FLOPs/token | head 占比 | 等 FLOPs tokens（A1=300M 基准） |
|---|---:|---:|---:|---:|---:|---:|---:|
| d12_aligned | 512 | 12 | 42,081,416 | **84,162,832** | 33,554,432 | 39.9% | 300.0M |
| d16_aligned | 640 | 16 | 63,754,784 | **127,509,568** | 41,943,040 | 32.9% | 198.0M |
| d20_aligned | 640 | 20 | 74,411,168 | **148,822,336** | 41,943,040 | 28.2% | 169.7M |
| d20d768_aligned（可选） | 768 | 20 | 107,977,416 | **215,954,832** | 50,331,648 | 23.3% | 116.9M |

（= `2 × active_params`，与 aligned accounting 完全一致；训练 FLOPs ≈ 3× + attention。）

**AdamW peak-LR 建议**（全 AdamW、与 Stage-1 同协议；两/多臂必须一致）：

| model | dim | Stage-1 约定 `3e-3·√(384/dim)`（**推荐**） | aligned μP `2.6e-4·√(5120/dim)`（v41 套件用） |
|---|---:|---:|---:|
| d12_aligned | 512 | **2.598e-3 → 2.6e-3** | 8.22e-4 |
| d16_aligned | 640 | **2.324e-3 → 2.32e-3** | 7.354e-4 |
| d20_aligned | 640 | **2.324e-3 → 2.32e-3** | 7.354e-4 |
| d20d768_aligned | 768 | **2.121e-3 → 2.12e-3** | 6.71e-4 |

推荐理由：这些臂是 **AdamW-only** 的等 wall 架构 probe，与 Stage-1 的 AdamW 臂口径一致；
aligned μP 值是为 **v41 Muon+Sinkhorn** 套件（update-RMS 0.18 / Sinkhorn γ=0.18）标定的，AdamW 直接
用 8.22e-4 会低 ~3.2×、欠训练；probe 本身是 within-pair 比较，绝对值只需两臂一致且合理。

运行：`SUBSET="D16 D20" bash tools/r90_vocab_arms.sh`（脚本默认 tokens 已按上表等 FLOPs 折算；
`D16_TOKENS`/`D20_TOKENS` 可覆盖；数据/面板/tokenizer 完全复用 A1）。

## 12. 增量式 129k 语料（Stage-4 用）：append 模式 + 驱动脚本

`tools/r90_vocab_retokenize.py` 新增磁盘友好的**增量 append 模式**（用于把 ClimbMix 边下载边
tokenize，语料一路长大；每批只处理"还没进 manifest"的 shard，写完 manifest 后删 sidecar/原始
parquet，崩溃可自愈）：

```bash
# 一次 append（幂等；无新 shard 时是廉价 no-op）
/root/t141/bin/python tools/r90_vocab_retokenize.py \
  --shards-dir /root/autodl-tmp/nanochat/climbmix --exclude-shard shard_06542.parquet \
  --min-shard-bytes 90000000 --tokenizer /root/dsv41/tokenizer.json \
  --out /root/autodl-tmp/nanochat/p2_final_129k \
  --append --delete-raw --workers 12 [--max-new-shards K]

# 驱动脚本（默认每 300 s 一轮；STOP_TOKENS 到顶后 idle；磁盘 < MIN_FREE_GB 时 idle）
#   STOP_TOKENS=4000000000 MIN_FREE_GB=12 INTERVAL=300 setsid nohup \
#     bash tools/r90_final129k_loop.sh >> logs/final129k_loop.log 2>&1 &
```

要点 / 安全性：
- 磁盘：每 shard 净增 ≈ +118 MB（corpus +210 MB − raw 92 MB），sidecar 在 append 后即删；
  1B tokens ≈ +2.2 GB 净占用（4.8 B/tok × 4 B/token）。
- `--check`（新）：校验 manifest/文件长度=4×tokens、各 shard 求和不矛盾、抽样 token id 上界、
  抽样 EOS 计数与 docs 量级一致；任何异常 rc≠0。
- 崩溃自愈：manifest 是唯一事实来源；若 append 后 manifest 未落盘，下次运行会把语料截断回
  `tokens×4` 字节再继续。
- `--min-shard-bytes 90000000` 跳过仍在下载的 partial shard；val shard 永远排除且不删除；
  只删除本轮成功 append 的 shard 的原始 parquet。
- 2026-11-12 实测：12 workers 下 tokenize-active **≈5.3M tok/s**（52.4M tok/shard，~90-100 s/worker），
  4.03B tokens 语料端到端 ~29 min（含等待下载的空闲）。
