# R93 — P1 8B 启动配方（2026-11-17，HP 搜索后）
> **2026-11-18 结果修正**：P1 8B 实际终值 raw bpb **1.0662**、EMA **1.0620**（预期 0.89 未达到；短预算曲线外推过长跑被证明过于乐观）。详见 `docs/r93-p1-final-report.md`。


**状态**：前置超参搜索完成（S1 LR 定律 / S2 z-loss / S3 tokens-per-step，见
`docs/r93-hp-search-results.md`）；z-loss 快速路径（+0.46% 步时）与 EMA 实现已就绪。
P1 本身尚未启动，**需要充值 + 补语料**（见 §4/§7）。

## 1. 模型与计算

| 项 | 值 |
|---|---|
| 模型 | `docs/r93/configs/d24_c1_dense40k.json`（d24 dense，288M active，Engram 220M，K=40k / V=41,280） |
| 序列 / micro-batch | seq 8192 / b1 |
| tokens/optimizer-step | **2^21 = 2,097,152**（b1×8192×accum256；S3 已判定 2^22 即使 LR×√2 也 +0.138 bpb → **保持 2^21**） |
| P1 8B 步数 | 8e9 / 2.097e6 = **3,815 步** |
| FLOPs/token | 1.728 GFLOP（2×N_active×3）→ 8B 总 C≈**1.38e19** |
| 吞吐（E0 栈） | 62.8k tok/s → 8B ≈ **35.4 GPU·h ≈ ¥106**（¥2.98/h） |

## 2. 超参（本轮搜索的结论）

- **峰值 LR：0.7–1.0e-3，中心 0.8e-3**。
  依据：S1 实测 d24 上 0.2B 顶点 3.77e-3、0.4B 顶点 2.03e-3（a_raw=−0.89，是"不稳定门槛"不是
  可外推定律，按它外推 8B 会得到 1.4e-4）；R91 定律的 1.82e-3 **不可用**；产线锚定先验 0.63–0.69e-3
  作为下沿。**P1 中若 gn/损失上爬，立刻降到 0.6–0.7e-3**。
- **WSD**：`--lr-schedule wsd --warmup 30 --decay-start 0.9 --min-lr-ratio 0.1`
  （warmup 30 步 ≈ P1 的 0.8%；cooldown 0.1 = 381 步 ≈ 0.8B tokens）。
- **z-loss 1e-4：默认开启**。S2 实测 0.3B Δbpb=−0.005（差异不显著但方向为正）；新快速路径只
  +0.46% 步时（旧路径 +17.7%），且保留 fp8 head 快速路径（见 `r93-hp-search-results.md` §6.3）。
- **EMA：`--ema-decay 0.995 --ema-every 1`**（窗口 200 步 ≈ 5% 训练量）。
  评测 raw 与 EMA 两个权重；在 1/2/4/6/8B 的 mark checkpoint 上都跑。
  注意：当前 EMA 只覆盖 338.1M 主体参数（Engram 表行走 SparseRowAdam、不在 named_parameters）。
- **其他不变**：`--grad-clip 1.0 --seed 1234 --optimizer v41`、Muon wd 0.1、
  AdamW eps 1e-20、Sinkhorn γ0.18/K=11/τ1e-3、Engram LR ×5。
- **环境**：production kernel env（`tools/r90_env_dense5090.sh`，含 `NANOV41_HEAD_FWDGRAD=1`、
  `NANOV41_HEAD_FP8=v2`、`NANOV41_HEAD_WCACHE=1`、linear-fp8 wo_b）+ E0 融合核
  `NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1` + `CE_CHUNK=8192`。
  （不 source 生产 env 会在 backward 死在 TileLang 共享内存，见 `r93-hp-search-results.md` §2 教训。）

## 3. 评测计划

- 面板：`p0_eval_40k`，`--domains climbmix_val --window 2048 --batch-size 8
  --max-tokens-per-domain 3000000`（~2 min/ckpt）。
- `--save-at-tokens 1b,2b,4b,6b`（每个 mark 存 ckpt，eval 后删 ckpt 保留 json）。
- **中途 gate**（省钱的早停）：2B 时 bpb 若 > 0.99 或 4B 时 > 0.93（相对 §6 预期偏差 >0.04），
  停跑并先查 LR/数据；6B 时 bpb 若 > 0.90 重新评估最终 8B 是否值得。
- 最终 eval（3M tokens）**按 5M/3M 两种口径各出一份**以便与 P0b 的 5M 曲线直接对比。

## 4. 语料（**必须先解决**）

P1 8B 需要 **≥8B tokens 的单 epoch 语料**。磁盘现状：
`<instance-id>` 上的 `p2_climbmix_40k` ≈ 24.28 GB u32 = **6.07B tokens**（4.4414 B/token）→
直接跑 8B 会是 1.32 epoch，按 §10 估计 bpb 约 +0.01–0.02。两个选项：

1. **补 fetch 到 ≥8.5B tokens**（推荐）：用 `tools/r93_fetch40k*.sh` / `tools/r93_lead_p1corpus*.sh`
   再拉 ~2.5–3B tokens（~12–15 GB 下载；实例网络 + 磁盘成本 ~¥1）；单 epoch 干净。
2. 接受 1.32 epoch，或缩到 **6B tokens（¥80，bpb 预期 0.92–0.94）**。

## 5. 启动命令（实例上，repo root）

```bash
# 0) 数据就位（p2_climbmix_40k ≥ 8.5B tokens）+ tokenizer + 面板
# 1) 冒烟（5 步，确认 63k tok/s、z-loss/EMA 都正常、显存峰值 < 30 GiB）
env OPT=v41 NANOV41_DS_TOKENIZER_JSON=$TOK CE_CHUNK=8192 \
    NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1 \
    python tools/r93_p0b_train.py --tokenizer $TOK -- \
    --arm p1_smoke --config-json docs/r93/configs/d24_c1_dense40k.json \
    --data-dir /root/autodl-tmp/nanochat/p2_climbmix_40k --mix web=1.0 \
    --batch 1 --seq 8192 --grad-accum 256 --optimizer v41 \
    --peak-lr 8e-4 --lr-schedule wsd --warmup 2 --decay-start 0.9 --min-lr-ratio 0.1 \
    --z-loss 1e-4 --ema-decay 0.995 --ema-every 1 --grad-clip 1.0 \
    --tokens 10485760 --seed 1234 --out runs/r93/p1/p1_smoke

# 2) P1 主跑（~36–38 h，后台，带 TTL 与预算守护）
( source tools/r90_env_dense5090.sh
  export NANOV41_DS_TOKENIZER_JSON=$TOK
  nohup setsid env CE_CHUNK=8192 NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1 \
    python tools/r93_p0b_train.py --tokenizer $TOK -- \
    --arm p1_8b --config-json docs/r93/configs/d24_c1_dense40k.json \
    --data-dir /root/autodl-tmp/nanochat/p2_climbmix_40k --mix web=1.0 \
    --batch 1 --seq 8192 --grad-accum 256 --optimizer v41 \
    --peak-lr 8e-4 --lr-schedule wsd --warmup 30 --decay-start 0.9 --min-lr-ratio 0.1 \
    --z-loss 1e-4 --ema-decay 0.995 --ema-every 1 --grad-clip 1.0 \
    --tokens 8000000000 --save-at-tokens 1b,2b,4b,6b \
    --log-every 10 --nan-check-every 1 --seed 1234 \
    --out runs/r93/p1/p1_8b > /root/p1_8b.log 2>&1 < /dev/null &
)
```

## 6. 预期（`bpb ≈ 1.21·(D/1B)^-0.154`，P0b 实测锚点外推）

| D | bpb 中心 | 区间 | loss (nats/tok) | ≈费用 |
|---:|---:|---|---:|---:|
| 6B | 0.93 | 0.87–1.00 | 2.87 | ¥80 |
| **8B** | **0.89** | **0.85–0.96** | **2.75** | **¥106** |
| 10B | 0.86 | 0.79–0.93 | 2.65 | ¥132 |

- 加 EMA/z-loss/LR 校准后 8B 预计 **0.87–0.90**；若 KD（−5.2%，数据需补完 140/287 shards）→ ~0.84。
- **性价比对照**：同 FLOPs（C=1.38e19）nanochat 插值线 bpb=0.94 → **P1 8B 预计低 ~0.05**，
  即"同算力性价比超 nanochat"的里程碑；绝对 bpb 追 0.718 需 ~30B tokens（¥400）或 P2 放大模型
  （见 `docs/r93-dense-plan.md` §11）。

## 7. 预算与注意事项

- **余额核对**：P1 8B ≈ ¥<redacted> + 评测/sync ≈ ¥<redacted> → **¥<redacted>**；当前余额需 ≥¥<redacted>，另留缓冲。
  注意**同账户其他实例也在计费**（每台 4090 ≈ ¥<redacted>/h）：P1 跑 36h 期间若 2 台 4090 也开着，会额外
  消耗 ~¥150。启动 P1 前请确认这两台是否必要。
- 实例：5090 ¥2.98/h；开机要等空闲卡（本账号 18 台 5090 常满），预留 0–2h 找卡时间。
- TTL 设到预计结束 +1h；每 2B token eval 后把 `runs/r93/p1/*.json` 同步回本地（防实例丢失）。
- 失败保护：`--abort-on-nan` 建议加；`--save-at-tokens` 的 mark ckpt 是断点续跑的落点
  （`--resume <ckpt>` 可续，但 optimizer state 默认不存，续跑要 `--save-optimizer`，文件会 +4–5 GB）。
