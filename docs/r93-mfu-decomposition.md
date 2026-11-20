# R93 dense d24_c1 — MFU 阶段分解与优化空间（实测 + 判断）

> 2026-11-15，单 5090 32GB，生产 kernel 栈（TL attn v3 / mHC v4 / fused_ew / linear-fp8 v2 /
> head-fp8 v2 / CUDA graph / v41），模型 d24_c1（dim 1024 / 24L / dense SwiGLU h=2304，Engram 220M，
> V=41,280），seq 8192，P0b 训练形状 b1×8192×accum256（2^21 tok/step）。
> 数据：`runs/r93/mfu_probe/b1.loop_profile.json`（p2_train `--profile-loop 4`）。

## 1. 实测：step 时间去哪了

| 段 | 每步 ms | 占 wall |
|---|---:|---:|
| **replay（graph: fwd + loss + bwd）** | 46,906 | **99.60%** |
| accumulate（grad-accum add） | 65.5 | 0.14% |
| clip | 4.9 | 0.01% |
| optimizer（adamw 10.1 + muon 55.6 + sinkhorn 5.0 + engram 8.5 + bias 0.9） | 80.0 | 0.17% |
| h2d + log_sync | 18.3 | 0.04% |
| pipelined wall | 46,994 | 100% |
| **非 GPU gap（loop/data/launch 税）** | **−81.6** | **−0.2%** |

- pipelined: **44,626 tok/s → raw MFU 36.81%**（与 P0b 全程 44.3k / 36.56% 一致）。
- **结论：没有"图外税"**。optimizer 只占 0.17%（accum=256 摊薄 + 全部与 GPU 重叠）；
  dataloader/launch 也无空泡。36.9% 是纯粹的 **in-graph kernel 效率**。
- 对比旧 MoE d12 栈（b2×8192×accum128）：graph replay raw 46.57% vs 训练 38.04% → 图外 ~18%。
  dense 这套没有这个问题（更大模型+更大 accum 让 CPU 侧完全隐藏）。

## 2. 为什么 in-graph 只有 36.9%（而 MoE d12 replay 是 46.6%）

口径：fwd 只算 `2·N_active`（575.95M FLOPs/token；Engram wkv 与 attention 未计 → 真实效率略高）。
36.9% ⇒ 每 micro（8192 tok）replay 183.2ms；train FLOPs = 3×575.95M×8192 = 14.16 TFLOP。

结构性差异（相对 MoE d12 replay）：
1. **micro-batch 形状**：我们是 b1×8192 = 8k token/micro；MoE replay 是 b2×8192 = 16k token/micro。
   生产实测 b1 vs b2 训练 MFU = 31.0% vs 36.8%（+5.8pp）→ **b2 恢复是最确定的单项杠杆**。
2. **非 GEMM 段占比更高**：dense FFN 的 24 层 mHC（每层 sinkhorn 20 iter）、attention v3、
   indexer、norms/EW、Engram gate；MoE 栈有大块 expert GEMM 拉高平均利用率。
3. head CE：head = 41280×1024 → fwd 84.5M（15%），fp8-v2 已启用。
4. dense 层数 24、dim 1024 → 每层 GEMM（M=8192, K=1024, N=2304×3）形状本身不算差，
   但 tile 配置是给 d12（dim 512）调的，**未对 dim1024 重新 sweep**。

## 3. 优化空间（按确定性排序）

| 杠杆 | 需要做什么 | 预期 | 把握 |
|---|---|---|---|
| **A. 恢复 b2×8192（16k micro）** | b2 在 capture 处 OOM：PyTorch 分配 ~30.5GiB（容量 31.36GiB），差 ~0.3–1GiB。CE_CHUNK 2048/512 不够。可试：CE_CHUNK 128、**Muon momentum bf16（−~0.5GB）**、FFN/mHC 激活重算、graph pool 裁剪、减少 engram 优化器缓冲 | **+2–4pp**（生产 b1→b2 = +5.8pp，dense 折半） | **高**（0.5–1 天） |
| **B. GEMM/kernel 形状工程** | dim 1024/h 2304 的 tile/mapping 重调（旧 tile 是 d12 调的）；body linear fp8（d12 上 NO-GO 因 N=64-512 太小；d24 是 1024/2304，需 1–2h microbench） | +2–5pp | 中 |
| **C. 残余 EW/attn 融合** | norms/residual/mHC 的 add+copy、rope、indexer 融合（R79 清单在新形状下未重测） | +1–2pp | 中 |
| **D. head CE** | fp8-v2 已开；chunk 与 softmax 融合余量小 | ≤1pp | 低 |
| **E. 更大 batch 语义** | accum 已 256；提高 micro 才是 A；seq 8192 不可降（分布一致性） | — | — |

**结论**：
- **40–42% 是 1–2 天工程的现实目标**（A + 部分 B/C），把握高；对应 P1 成本从 ¥113 降到 ¥99–104。
- **45% 是 stretch**：需要 A（b2）+ B（fp8/tile 正收益）+ C 全部落在上界，且 b2 的显存工程成功；
  把握 ~40–50%，需 3–5 天。**不应写进 P1 预算**（保守按 36.6% 记 ¥113，优化成功是 bonus）。
- 反过来看：我们已把"图外 18%"这一项彻底消掉（dense 配置的意外收获），剩下的都是 kernel 级工作。


## 5. 最终版：kernel 分桶实测（详见 `runs/r93/p0b/mfu_profile.md`）

| 桶 | ms/replay | 占比 | 关键事实 |
|---|---:|---:|---|
| bf16 GEMMs | 68.7 | 37.8% | 19 种 cutlass/nvjet kernel；**归因 ~12.5 TFLOP 后 ≈182 TF ≈ 87% peak → GEMM 不是问题** |
| **elementwise/copy/norm** | **~68** | **37.5%** | **fp32 `MulFunctor` 17.40ms ×476 calls**（≈ RMSNorm 未融合，fwd 3 muls + bwd muls，~96-121 calls/step × 3-5）、copy 5.66×414、bf16-copy 4.49×481、clamp 4.40×100、compare 4.22×72、silu/silu_bwd 5.1 |
| mHC kernels | 21.5 | 11.8% | _post_bwd_dual 8.56 / _mixes_pre_bwd 5.51 / _post_fwd 4.11 / _mixes_pre_fwd 2.41 / split_sinkhorn_bwd 0.93 |
| attention v3 | 9.8 | 5.4% | TL kernel 7.84×48 + _dkv_win_tri 1.94 |
| head/CE fp8 | 2.5 | 1.4% | V=41k + fp8 head 几乎免费 |
| 长尾 | 1.25 | 0.7% | |

replay = 181.76 ms/micro（45,070 tok/s，raw MFU **37.17%**），6,439 kernels/replay；图外 gap **0.3%**。
**结构性结论：GEMM 已经很满（~87%），~50% 的时间在非 GEMM（EW/copy/norm+mHC）→ 所有剩余 MFU 都在这里。**
这解释了为什么 dense 36.9% < 旧 MoE 46.6%（MoE grouped-GEMM 主导、EW 只占 ~10%）。

### 5.1 最大单项 & 最便宜的杠杆（重要发现）

1. **`NANOV41_RMS_FUSED` 在产线 env 里没开**：`nanov41/fused_rmsnorm.py`（R45 已验证：fp32 内部、
   replay 逐位 parity、6-7 个 aten kernel → 1 fwd + 2 bwd）存在，但 `tools/r90_env5090.sh` 未导出该变量
   → 当前 476 个 fp32 mul 主要来自未融合的 RMSNorm。**只需开 env + 10 min A/B**，预期 −6-10ms。
2. **`NANOV41_SWIGLU_FUSED` 同样没开**：`nanov41/fused_swiglu.py` 融合 clamp+silu+mul+cast，且 patch
   的正是 DenseFFN 使用的 `weights is None` 路径；预期 −3-7ms。
3. 两者合并预期 **−9-17 ms ≈ +2.5-4 pp**（需在 d24 形状 A/B 验证，R83 提示 producer-fusion 偶有反效果）。
4. **b2×8192 不是 CE_CHUNK 问题**：capture 阶段 OOM 在 FFN fwd（`[16384,2304]` fp32 buffer，缺 ~8-10 GiB，
   不是 ~1GiB）→ 必须做 **FFN 激活重算**（1-2 天）或 bf16 中间量，预期 +3-5pp。
5. 其余：`--profile-eager 3` 精确定位 476 muls/414 copies（10 min）→ sinkhorn-wrapper/compare 融合（≤1ms）→
   body fp8 linear 扩到 FFN GEMM（2-5 天，+2.5-4pp）。

### 5.2 45% 是否现实（最终判断）

- 42% ⇔ 需省 ~21ms 非 GEMM；45% ⇔ 需省 ~32ms（非 GEMM 的 ~31%）。
- **42%：1–2 天可达成（把握中高）**：开两个已存在的融合 kernel（0.5h GPU）+ b2 激活重算（1-2 天）。
- **45%：3–5 天 kernel campaign（把握 ~50-60%）**，需要 b2 + EW/copy 腰斩（RMS/SwiGLU + 其余融合）+
  body fp8 扩到 FFN 三者都成立；若 fp32 norm/mHC 语义是不可动的（bitwise parity 约束），上界 ~42-44%。
- 结论：**不把 45% 写进 P1 预算**；P1 前先花 0.5-1 天做 #1+#2+#4（最便宜），拿到 40-42% 再启动。

## 4. 待办（P1 前，按顺序）

1. b2 显存工程：`NANOV41_MUON_MOMENTUM_BF16=1` / CE_CHUNK 128 / 激活重算 → 目标 b2×8192×accum128
   跑通 3 步，测 replay MFU；若 ≥40% 直接采用。
2. 用 torch.profiler 或 R79 工具对 d24 replay 做 kernel 分桶（当前只有段级，没有 kernel 级）。
3. tile sweep（FFN GEMM 1024×2304、attn v3、mHC）各 1 组；body fp8 microbench。
4. 每次改动都跑 3 步 loss parity + 用 `--profile-loop` 复核真实 MFU。
