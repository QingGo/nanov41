# R91 — 用更少数据搜参：四层级联搜索协议（脚本化 2026-11-13，review 修复版）

> 问题：我们能否用**远少于目标 run 的 token** 搜超参/结构/容量分配，并让结论可迁移到
> 1–7.5B token 的目标 run？答案：分四层（C/A/B/D，D 拆成 D1/D2 两档），结构类可压到
> 100.7M tokens，Engram 分配必须 ≥0.5B，LR/参数分配的最优值必须用两档预算拟合或长跑确认。
>
> 状态：脚本 + 队列已写完，并通过离线 fixture / 假解释器端到端测试；**尚未在 GPU 上实跑**
> （全量约 ¥75–92，当前余额不足，等充值/决策）。

## 0. TL;DR

| 层 | 目的 | 臂数 | 每臂 token | 单层 wall（估） | 单层 ¥（估） |
|---|---|---:|---:|---:|---:|
| **C** | LR 定律（2 FFN × 2 预算 × 3 LR） | 12 | 100.7M / 302.0M | 4.5–5.8h | 13–17 |
| **C2** | C 的最优 LR 越出网格时自动扩展（3 点/格） | 0–12 | 同 C | 0–5.8h | 0–17 |
| **A** | 结构粗筛（MoE 粒度 / 真 dense FFN 宽度） | 8 | 100.7M | 1.8–2.4h | 5–7 |
| **B** | Engram 分配/表大小（在领先 FFN 上） | 5 | 503.3M | 4.6–5.5h | 14–16 |
| **D1** | 级联确认（A/B 晋级 + 产线对照） | 3–4 | 1.007B | 5.7–8.6h | 17–26 |
| **D2** | 最终确认（D1 前二） | 2 | 2.013B | 7.3–8.6h | 22–26 |
| 合计 | | 30–31 | | **24–31h** | **¥71–92** |

- **lean**（只 C→A→B，B=480 步）≈ **11–14h ≈ ¥34–41**；`R91_B_STEPS=288` 再砍到 ≈ **10–12h ≈ ¥30–36**。
  D1/D2 只对赢家付大钱；`MAX_COST_Y` 逐层拦截（默认 95）。
- 完成/停止语义：`runs/r91/QUEUE_DONE` = **所有请求层全部臂都有有效结果**（才可关机）；
  否则写 `QUEUE_STOPPED`（含原因、exit 1）。**dry-run 不写任何 sentinel。**

## 1. 为什么"少数据"能搜、以及哪里不能搜

| 机制 | 现象 | 对策（已写进脚本/协议） |
|---|---|---|
| **最优 LR 随 token 预算漂移** | Marin 实测 `lr* ∝ tokens^-0.346`（issue #7856/#8003）；7.5B 的 LR 直接用到 100M 会低 ~4.5× | **先跑 C 层**测 `lr*(tokens)`；A/B/D 全用定律 LR。C 网格**固定用先验定律**，不会因重启而移动 |
| **短预算下大模型欠训 → iso-FLOP 最优点左移** | 100M tokens 下 h3072 输给 h768 不代表 1B 仍输 | A 只在同预算排名；D1 在 1.007B 复核前二；A 的半预算前缀点量化 rank 迁移 |
| **Engram/稀疏容量需要行覆盖** | R90 P2：256M 表在 0.6B tokens 下回退 | Engram 分配（B）强制 ≥0.5B tokens；A 不搜表大小 |

两个省钱技巧：

1. **WSD 稳定段 + 前缀 checkpoint**：A/B 每臂在 50% 预算存一次 ckpt（`--save-at-tokens`），
   训练后评测再删 → 一次 run 得到"稳定段前缀 vs 最终"两个点；`r91_rank_transfer.py` 给 Spearman ρ
   与 winner-aware verdict。**注意**：前缀点没有 cooldown，是保守代理（见 §5）。
2. **级联只给赢家加钱**：D1 只跑 A 前二 + B 第一 + 产线对照；D2 只跑 D1 前二；D 层不存前缀点。

## 2. WSD schedule 参数（唯一真源：`r91_common.wsd_schedule()` + `stage_plan().warmup_frac`）

所有臂统一 WSD，不用产线 cosine：

```
--lr-schedule wsd --decay-start 0.9 --min-lr-ratio 0.1   # 前 90% 峰值稳定，最后 10% 余弦退火到 0.1×
warmup = round(WARMUP_FRAC * steps)                       # A/C 0.08、B 0.06、D1/D2 0.05
tokens/step = 2×8192×accum64 = 2^20                       # 产线是 2^21；减半让 100.7M 仍有 96 步
```

- A/B 在 50% 预算存前缀 ckpt 并评测（同一 P0 5M 面板；`KEEP_MARK_CKPT=0` 评测后删除）；
- **LR 从运行记录读**：`p2_train` 写 `<tag>.train.json`（含 `peak_lr`/`tokens`），`r91_analyze_c.py`
  以它为准（plan 只当请求，训练报告才是记录）；
- 每臂 `schedule.txt` 带 `signature=`（LR/tokens/schedule/warmup/cfg/data）。**签名变化 → 旧结果目录
  被挪到 `<arm>.stale.<HHMMSS>` 并重训**，绝不静默复用（review M4/B2）；
- C 层网格固定用先验：`lr*(tokens,dim=512) = 8.2219e-4 × 0.707 × (7.4994e9/tokens)^0.346`
  （WSD、搜索 batch；产线 cosine 只作 sanity check）。

## 3. 各层细节

### C — LR 定律（12 arms，分析必然执行）

| 维度 | 取值 |
|---|---|
| FFN 家族 | `moe`（`d12_aligned.json`，E128 top2）与 `dense`（`d12_dense_true_h768.json`，FLOPs 等价） |
| token 预算 | 96 步 = 100,663,296 / 288 步 = 301,989,888 |
| LR 网格 | 先验中心 ×{0.60, 1.00, 1.60} |
| 数据 | `p2_final_129k` |
| 分析 | `r91_analyze_c.py`：每 (FFN,budget) cell 对 `bpb = f(log10 lr)` 做最小二乘二次拟合取顶点；拟合 `log10 lr* = c + a·log10(tokens/7.5B)`；越界（low/high/nonconvex）自动生成 **C2**（围绕预测顶点的 0.45/1.0/2.2 三点） |

为什么"2 FFN × 2 预算"而不是原"3 宽度"：A/B/D 候选全是 dim=512，不需要宽度外推；
A/B 同时含 MoE 与 dense，**FFN-conditioned 定律**更值钱（宽度指数仅记录先验）。
分析总是从 C 结果重算；`lr_law.json` 带 `quality`（ok/partial）与 `warnings`（cell≥3 点等完整性检查，review M7）。

### A — 结构粗筛（8 arms，100.7M tokens）

| tag | cfg | 结构 |
|---|---|---|
| A0-moe128 | `d12_aligned.json` | 产线 MoE E128 top2 + Engram 220M（对照） |
| A1-moe-g8 | `d12_g8.json` | E256 top4（G=8） |
| A2-moe-g2 | `d12_g2.json` | E64 top1（G=2） |
| A3-moe-g16 | `d12_g16.json` | E512 top8（G=16） |
| A4-dense-h384 | `d12_dense_true_h384.json` | 真 dense h=384 |
| A5-dense-h768 | `d12_dense_true_h768.json` | 真 dense h=768（FLOPs 等价 183.0M） |
| A6-dense-h1536 | `d12_dense_true_h1536.json` | 真 dense h=1536（1.16× FLOPs） |
| A7-dense-h3072 | `d12_dense_true_h3072.json` | 真 dense h=3072（1.46× FLOPs） |

LR 用 C 定律在该 FFN 家族、100.7M 预算下的值；50% 处存前缀点；数据 `p2_climbmix_129k`。

### B — Engram 分配（5 arms，503.3M tokens）

| tag | cfg | 含义 |
|---|---|---|
| B0-dense-h768-eng220 | `..._eng220.json` | dense h768 + 表 220.0M（与 A5 **同一模型**；promote 用语义指纹去重） |
| B1-dense-h768-eng440 | `..._eng440.json` | 表 439.7M（2×） |
| B2-dense-h768-eng815 | `..._eng815.json` | 表 814.6M（与产线 iso-param） |
| B3-moe128-eng220 | `d12_aligned.json` | 产线 MoE 对照 |
| B4-moe-e16-eng748 | `d12_alloc_e16.json` | 全内存 MoE split |

lean 模式：`R91_B_STEPS=288`（302.0M tokens）。

### D1 — 级联确认（3–4 arms，1.007B tokens）

`r91_promote.py --stage D1`：

```
top-2 stage A + top-1 stage B + 产线对照 A0-moe128（按语义指纹去重；不在则补）
```

- 按 **模型语义指纹**（engine_config+engram 的哈希）跨层去重：A5 与 B0 是同一模型只留一个；
- tag = `D1-<短名>-lr<LR>`（含 LR，重训/LR 变化会自动落到新目录，review M3）；
- D 层不存前缀点（无消费者，省 2GB/臂 + 评测时间）；`--include-ref ''` 可 3 臂；
- 结果不足 top-k 时 promote 仍继续但记 `partial=true` 并告警。

### D2 — 最终确认（2 arms，2.013B tokens）

D1 前二，LR = 定律在 2.013B 的预测值。

## 4. 命令

数据需就位（保留实例 `<instance-id>`）：`/root/autodl-tmp/nanochat/{p2_climbmix_129k, p2_final_129k, p0_eval}`
+ `/root/dsv41/tokenizer.json` + `/root/t141` venv（新租机器先同区内网 rsync；`p2_final_129k` ≈15GiB）。
queue 启动 preflight 检查这四项 + 磁盘 ≥12GiB + nvidia-smi（测试可用 `R91_SKIP_PREFLIGHT=1` 跳过）。

```bash
cd /root/nanov41
mkdir -p /root/autodl-tmp/nanochat/runs/r91

# 0) 干跑：只生成 plan/arms/env 并打印，不碰 GPU，不写任何完成标记
DRY_RUN=1 STAGES=C,A,B,D1 MAX_COST_Y=40 bash tools/r91_queue.sh

# 1) 全量（约 24–31h；看到 QUEUE_DONE 才关机）
setsid nohup bash tools/r91_queue.sh \
  > /root/autodl-tmp/nanochat/runs/r91/queue.console 2>&1 < /dev/null &

# 2) lean：只跑三层（B=480 步 ≈11–14h；R91_B_STEPS=288 再省 2–3h）
STAGES=C,A,B R91_B_STEPS=288 MAX_COST_Y=50 bash tools/r91_queue.sh
```

常用旋钮：

| 变量 | 默认 | 作用 |
|---|---|---|
| `STAGES` | `C,A,B,D1,D2` | 跑哪些层（顺序强制：C→A/B→D1→D2；依赖不满足会跳过并写进 QUEUE_STOPPED 原因） |
| `MAX_COST_Y` | `95` | 预算护栏：每层（含 D1/D2）在真正开跑前用 plan 投影检查；超限则跳过该层并记 `SKIPPED` |
| `LR_LAW` | `auto` | `auto`=用 `<base>/c/lr_law.json`；给路径=强制使用（不存在直接报错退出） |
| `ALLOW_PRIOR_LAW` | `0` | C 失败时是否允许 A/B 用先验定律继续（默认不允许，避免在错误 LR 上烧钱） |
| `C_EXTEND` | `1` | C 最优越界时自动跑 C2 |
| `R91_B_STEPS` | `480` | B 层预算步数（lean 用 288） |
| `R91_BASE` / `R91_PLAN` | `.../runs/r91` | 结果目录 / plan 路径 |
| `PRICE_Y_H` | `2.98` | 成本记账单价（写入 summary，不再被 env 覆盖） |
| `DRY_RUN` | `0` | 只生成 + 打印 |

单层手动跑（调试）：`r91_plan.py --stages C` → `source tools/r91_env_C.sh` →
`bash tools/r90_run_gpu.sh tools/r90_arms5090.sh` → `r91_analyze_c.py`。
分析已拉回本地的结果：`r91_analyze_c.py --plan <local plan.json> --out-dir <本地 C 结果目录>`。

## 5. 读结果与"多少数据才够"

- `runs/r91/plan.json`：各层 arm 定义、LR 定律、晋级来源、（可能的）`partial` 标记。
- `runs/r91/c/lr_law.json` + `c_summary.md`：`quality`/`warnings`、每 cell `edge`、
  `fits.{moe,dense,mean}`（`a` 越负 = 短跑需要越高 LR）、`predict`（96/288/480/960/1920 步预算）、
  `production_check.ratio`（>1 表示产线 LR 可能偏保守）。
- `runs/r91/{a,b}/rank_transfer.{json,md}`（每层各自一份）：
  - 比较的是 **WSD 稳定段 50% 前缀（无 cooldown）** vs **完全退火的最终模型**，是保守代理；
  - verdict 同时要求 ρ≥0.9 **且 winner 不变**才建议"屏幕减半"；`UNSTABLE` 可能由缺失 cooldown 造成，
    只能说"不能据此减半"；
  - 前缀与最终都用 5M token P0 面板（同分辨率、配对抽样）。
- `runs/r91/search_summary.md`：每层排名表 + winner + 成本。
- `runs/r91/QUEUE_DONE`（完成，可关机）**或** `runs/r91/QUEUE_STOPPED`（含原因，exit 1，不可当完成）：
  `QUEUE_DONE` 仅在"每个请求层都存在且其所有臂都有有效 p0.json"时写出；dry-run 不写任何 sentinel。

## 6. 风险与保底（含 2026-11-13 独立 review 修复）

| 风险 | 保底 |
|---|---|
| C 层没分析 → 定律永远测不到 | 分析在 C 训练后**无条件执行**（以 C 结果数为门），失败即 FAIL；C 不可用时 A/B/D 默认直接跳过（`ALLOW_PRIOR_LAW=1` 才用先验） |
| 重启后 C 网格被"自己拟合的定律"移动、旧 bpb 配新 LR | C 网格**固定用先验**；分析读 `<tag>.train.json` 的真实 LR；每臂 signature 变化即隔离旧目录重训 |
| 前缀 ckpt 评测算失败后永远不补 | 评测失败保留 ckpt；下次运行对缺 mark 的臂**重试**；p0_t JSON 用"非空+bpb+完整括号"校验 |
| 评测崩溃留下坏 ckpt / 写一半的 ckpt | `p2_train` 原子写（`.tmp`+`os.replace`）；P0 失败则把 `.pt` 挪成 `.pt.bad` 并在下次重训 |
| 短跑 LR 先验偏（C2 仍越界） | C2 自动扩展；二次仍越界则只用 in-grid cell 拟合，law 标 `quality=partial` + warnings |
| OOM（d12+dense+Engram 长跑） | runner `RETRY_OOM=1`：同 batch/schedule 下用 `CE_CHUNK=1024 + expandable_segments` 重试一次 |
| 队列中断/实例被抢占 | 臂级可重入：p0.json + 全部 mark 有效才算 done，否则重训/补评测；`cost.log` 续算累计花费 |
| 单臂失败/预算不够 | 该臂无 bpb；promote 记 `partial=true`；`QUEUE_STOPPED` 写明哪层缺多少臂 |
| 磁盘（前缀 ckpt ~2GB/个） | 前缀点评测后删除；preflight 要求 ≥12GiB |
| C 层测的是 WSD 下的 LR，产线是 cosine | 全链路自洽（A/B/D 也用 WSD）；产线 LR 只作 sanity check 记录 |
| 100.7M 屏幕只有 ~1.1× active（不含 head ~4×），远低于 Marin 最小 30× active | 前缀点 + D1 复核量化迁移性；ρ/verdict 直接回答"100M 排名能否迁移 1B" |
| 产线对照占一个 D1 臂（+¥5.6） | `--include-ref ''` 可省 |

## 7. 成本

投影公式：`Σ tokens/(0.36·209.5T/3·fwd) + 180s/臂 + 评测`（0.36 MFU 保守；实测 d12 MoE 145k tok/s、
dense h768 163k tok/s 时更便宜）：

- 全量投影 **24–31h ≈ ¥71–92**（含 C2 时 +0–17）；
- lean（C+A+B，B=480 步，无 C2）≈ **11–14h ≈ ¥34–41**；`R91_B_STEPS=288` ≈ **10–12h ≈ ¥30–36**；
- 每层投影会打印，`MAX_COST_Y` 在**开跑前**（含 D1/D2，promote 之后）逐层拦截。

> 当前余额不足以跑全量（见 `docs/r90-search-log.md` 末尾），需先确认余额/充值或先跑 lean。

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `tools/r91_common.py` | 常量、阶段定义、plan/arms/env 生成、`config_fingerprint()`、LR 先验、结果读取 |
| `tools/r91_plan.py` | 生成 A/B/C 的 arms/env/plan（`--lr-law` 只影响 A/B） |
| `tools/r91_promote.py` | A/B→D1→D2：指纹去重、LR-aware tag、partial 标记 |
| `tools/r91_analyze_c.py` | LR 定律拟合（真实 LR）、C2 越界扩展、`quality`/`warnings` |
| `tools/r91_rank_transfer.py` | 稳定段前缀 vs 最终：Spearman/top-k/winner-aware verdict（每层各写一份） |
| `tools/r91_queue.sh` | 队列：依赖顺序、预算护栏、preflight、sentinel 语义、断点续跑、成本续算 |
| `tools/r91_make_figures.py` | 生成 `runs/r91/figs/*.png`（loss/bpb 趋势、LR sweep/law、结构/分配、rank-transfer、成本） |
| `tools/r91_make_report_html.py` | 把 `docs/scaling-learnings.md` 渲染成自包含 HTML（图表内嵌）：`runs/r91/scaling-learnings.html` |
| `tools/r90_arms5090.sh` | 臂 runner：WSD 参数/per-arm 覆盖/前缀 ckpt 评测/provenance signature/OOM 重试 |
| `tools/p2_train.py` | 训练入口（`--save-at-tokens`、原子 ckpt、mark resume 警告） |
| `tools/r91_arms_*.txt`、`tools/r91_env_*.sh`、`tools/r91_plan.json` | 生成物 |

实例产物：`runs/r91/{a,b,c,d1,d2}/<arm>/`（`train.log`、`schedule.txt`（含 signature）、
`<tag>.train.json`、`p0.json`、`p0_t<mark>.json`、`marks.json`）、`plan.json`、`cost.log`、
`search_summary.md`、`QUEUE_DONE` xor `QUEUE_STOPPED`。

## 9. 离线验证

`/tmp/r91_fixture_test_v3.py`（本地，无 torch/GPU）：

1. **定律恢复 + 真实 LR**：人为把 plan 的 C LR 全乘 1.7（模拟重启后网格移动），分析器按
   `<tag>.train.json` 记录值恢复出 `a=-0.300`，1B 外推误差 <2%；
2. **越界扩展**：dense 定律整体平移 4× → 自动 C2（6 arms）→ 重拟合恢复 `a=-0.300`、`extend=False`；
3. **晋级去重**：A5 与 B0（同一模型）同时胜出 → D1 只保留一个；tag 含 LR、D 层无前缀 mark；
4. **rank 迁移**：单调构造成对 → ρ=1.0 且 winner-aware verdict 为 RANK-STABLE；
5. **队列 dry-run**：只生成 plan/arms，**不写** QUEUE_DONE/QUEUE_STOPPED；
6. **完成门控**：无结果 → `QUEUE_STOPPED`（exit 1）；补齐全部 A 臂结果 → `QUEUE_DONE`（exit 0）。

另用假解释器端到端跑过 `r90_arms5090.sh`：per-arm 覆盖、warmup 比例、前缀 ckpt 保存→评测→
`marks.json`→删除、`DEL_CKPT`、坏结果隔离（`.pt.bad`）、"已 done 即跳过"、签名变化隔离重训。

## 10. 参考与修复记录

> 本轮 scaling 探索的**跨项目 learning 汇总**见 `docs/scaling-learnings.md`（含 Chinchilla 适用性、LR×预算定律、
> dense/MoE/Engram 结论、成本口径与下一步）。

- Marin 535B hero run 与 HP 定律：`marin-community/marin` #8435 / #7856 / #8003（`lr ∝ tokens^-0.346`）；
  WSD-S 用于 scaling ladder（同仓 #654）。
- 2026-11-13 独立 review（read-only）发现并已修复：C 从不分析（B1）、重启后 C 网格漂移污染定律
  （B2）、QUEUE_DONE 无条件写（B3）、D1/D2 预算护栏失效（M1）、跨层去重失效（M2）、位置式 tag
  复用旧结果（M3）、p0.json 跳过不看 provenance（M4）、plan 失败仍跑旧 arms（M5）、mark 评测不重试
  （M6）、无完整性门（M7）、rank 对比口径（M8）、`--out-dir` 不生效（M9）、`LR_LAW` 无效（M10）、
  非原子 ckpt（M11）、`--save-at-tokens`+resume 静默丢点（M12）。

---

## 11. 实跑结果（2026-11-14，lean：C→A→B，38 臂，15.1h，¥44.8）

实例：用户借用的 5090 `<instance-id>`（¥2.98/h）；预算 `MAX_COST_Y=70`（因 C2 扩展把上限从 45 调高）。
sentinel 为 `QUEUE_STOPPED`：completion gate 时 A 层 7/8（A3 G16 tile 断言失败，队列后补跑成功），B 层 5/5 完整。

**C + C2（24 臂，¥29.3）**
- 4/4 cell 全部 high-edge → 自动 C2 12 臂；最优 LR：moe 100.7M **7.36e-3**、moe 302M **4.47e-3**、
  dense 100.7M **6.50e-3**、dense 302M **3.78e-3**；
- token 指数：dense a=−0.270、moe a=−0.790、mean a=−0.530（2 格/FFN，低置信）；
- 产线 LR 校验：mean 定律外推 7.5B（×√2 batch 修正）= **8.70e-4** vs 产线 8.22e-4 → ratio 1.058。

**A（8 臂 @100.7M，¥6.2）**
- dense：h3072 1.5196 / h1536 1.5211 / **h768 1.5225（等 FLOPs）** / h384 1.5339；
- MoE：G2 1.5410 / G4 1.5460 / G8 1.5486 / G16 1.5658；
- **等 FLOPs dense 赢 MoE −0.0235 bpb；G8 没有优势（与 R90 低 LR 屏幕相反）**；
- rank-transfer：ρ=0.714（winner 会换）→ 结构屏幕需 D1 复核。

**B（5 臂 @302M，¥8.9）**
- dense+eng815 **1.3028** / eng440 1.3044 / eng220 1.3110 / MoE E16+eng748 1.3230 / 产线 MoE+eng220 1.3446；
- 表大小收益单调但饱和（220→440 −0.0066；440→815 −0.0016）；dense 优势随 tokens 增大；
- rank-transfer：**ρ=1.0** → 分配类屏幕可用半预算。

**协议本身的修正**
1. **C2 是常态不是例外**：先验 LR 与实测差 ~8× 时，4/4 cell 越界；预算要按"C 的两倍"预留（本次 C2 ≈ ¥14.4）。
2. A3/G16 这类非标准 expert inter 需要 kernel tile env（`NANOV41_MOE_DW2_TILE=64,128,32,4,3`），
   否则整臂在 28s 内失败；v2 协议应让 runner 按 config 自动注入 env。
3. `QUEUE_STOPPED` 的语义符合设计（缺臂即不写 DONE）；补跑缺失臂后需人工重算排名（或 v2 协议加自动补跑）。
4. 结构类屏幕用全预算（100.7M）比半预算可靠；分配类可用半预算（ρ=1.0）。
5. 完整 conclusion 与 learning 见 `docs/scaling-learnings.md` §5/§6.2/§7/§9。
