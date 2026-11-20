# R93 — d24_c1 推理速度估算（单流 / 批量），与实测锚点（2026-11-16）

问题：当前模型（R93 d24_c1 dense，288M active，V=41,280，seq 8192，Engram 220M，CSA2(win128/topk64)，
mHC=4）在单张 5090 上的**单流 decode 速度**和**批量吞吐**能到多少？

**先说结论**：我们**还没有 d24_c1 的 decode 实测**——仓库里现有解码栈（M1/M2/M2c/M2d/M3）是为旧的
d12 模型（193.9M 总 / 19.8M 非嵌入 active / 129k vocab / 384 dim / ctx 1024）在 **4090** 上做的。
下面是按实测锚点 + 带宽/层数放缩的估算（±30–50%），最后给了把现有栈接到 d24_c1 上实测的最短路径（~0.5 GPU·h，¥2–3）。

## 1. 估算结果（bf16 权重、单张 5090、32GB）

| 场景 | 估算 | 约束来源 |
|---|---:|---|
| **单流 decode（现有 d12 风格解码栈移植+调优）** | **~300–500 tok/s**（2.0–3.3 ms/token） | kernel 链延迟主导（24 层 × ~100 µs/层） |
| 单流 decode（batch-1 专项 kernel campaign 后） | ~600–1,000 tok/s | 合并/常驻 kernel；d12 的 M2→M2d 用同类手段拿到 3.1→1.23 ms |
| 单流带宽下限（bf16 权重） | **~2,600–2,800 tok/s**（~0.36 ms/token） | 每 token 流式读 576 MB（245.7M×2B + head 84.5MB）/ 1.6–1.79 TB/s |
| 单流带宽下限（body+head fp8，官方推理 dtype） | ~5,000–5,500 tok/s | 同上但权重减半（~288 MB） |
| **prefill（8k prompt）** | 8,192 tok：~50–60 ms（≈140–160k tok/s）；未优化路径 100–200 ms | 4.72 TFLOP / (84–105 TF @40–50% MFU) |
| **批量 decode，B=16** | ~5–7k tok/s | 步骤时间 ≈ 2.5 ms 固定链 + 16×0.006 ms 计算 |
| B=64 | ~18–25k tok/s | 同上，计算项线性增 |
| B=128 | ~32–40k tok/s | |
| B=256 | ~50–70k tok/s | |
| B=512 | ~80–100k tok/s | 计算项开始主导；KV ~2 MB/序列 → 1 GB |
| 理论计算上限 | ~360k tok/s（0.576 GFLOP/token @209.5 TF 100%） | 实际有效 50% ⇒ ~180k |

显存不是问题：权重 bf16 ≈ 1.1 GB（backbone 491 MB + head 84.5 MB + embed 84.5 MB）+ Engram 表
440 MB（bf16；fp8 220 MB）+ KV ≈ 2 MB/序列（win128 + 两个压缩组 + index K，8192 ctx）→ B=256–512 可以放下。

## 2. 依据：旧模型实测锚点（4090，d12，19.8M active，129k vocab）

| 里程碑 | 单流 decode | 每步耗时 | 说明 |
|---|---:|---:|---|
| eager | 23.3 tok/s | 42.9 ms | 94% 是 CPU launch 开销（1019 kernel/step） |
| M1 graph-safe | 306–314 | ~3.2 ms | 图捕获消掉 dispatch，ceil 324–331 |
| M2 各 kernel 提速 | 396–397 | ~2.5 ms | ceil 433 |
| M2c | 487.5 | ~2.05 ms | ceil 537.6 |
| **M2d（fused MoE+RMSNorm+head）** | **790** | **1.231 ms** | ceil 812.4；token 与 reference 完全一致 |
| M3 批量 | — | **1.882 ms/step @B=16 → 8,776 tok/s** | bit-identical per sequence |

换算到我们的 d24_c1：层数 ×2（12→24）、非嵌入权重 ×6.2（39.4M→245.7M bf16 流式读）、head 更便宜
（129k×384 fp32=198.5MB → 41k×1024 bf16=84.5MB）。单流仍以 kernel 链延迟为主（d12 M2d 1.23ms 里
GEMM 带宽占比很小），所以 **~2× 层数 ≈ 2.2–2.6 ms/step ⇒ 380–450 tok/s** 是合理中位；
批量侧 d12 M3 B=16 的 1.882 ms 里固定链约 1.8 ms，换到 d24 约 2.5 ms，计算项 16 tok 只加 ~0.1 ms。

## 3. 为什么单流离带宽下限差 6–8×（可优化空间）

- 单流是**逐层串行的小 kernel 链**：24 层 × ~15–20 个 kernel（q_a/q_b/kv/wo_a/wo_b/w1/w3/w2/attn/
  indexer/mHC/norm…）≈ 400–500 个 kernel，即使全进 CUDA graph，每个 ~1–3 µs → 0.5–1.5 ms 下限；
- 权重流式读 576 MB 只要 0.36 ms，所以 **延迟而不是带宽**是瓶颈 → batch-1 的 kernel 融合
  （persistent/fused mHC+RMSNorm+GEMM epilogue、把 q_a→q_norm→q_b 融合、engram gather 融合）
  是最直接的提速方向；
- 官方推理 dtype 是 **fp8 linear + fp4 expert**：我们若把 body 权重也走 fp8（repo 已有 fp8 linear
  与 fp8 head kernel），流式读减半、GEMM 更快，单流预计 +20–40%，批量侧更明显；
- **DSpark 投机解码**（官方 post-train 组件，我们未实现）可再给单流 2–2.5×；
- **CED prefill 捷径**（decoder global KV 由 encoder 末层投影，未实现）让 prefill 计算≈减半。

## 4. 把估算变成实测的最短路径（建议下一步，~0.5–1 GPU·h ≈ ¥2–3）

1. 用 **P0b 的 d24_c1 最终 ckpt**（在 `<instance-id>` 数据盘）加载，先跑 eager decode 正确性
   （4–5 个 prompt × 16 token，与训练 checkpoint 的 CE/eval 一致性 sanity）；
2. 把 `nanov41/decode_graph.py` / `batch_decode.py` / `attn_decode.py` / `engram_decode.py` 的
   config 适配 d24_c1：dense FFN decode 路径（无 MoE gate）、head V=41,280、ctx 8192 KV（现栈按
   ctx 1024 设计，需要 paged/长 ctx 版本）、mHC/Engram decode kernel 的 shape 校验；
3. benchmark 矩阵：batch {1, 2, 8, 16, 32, 64}, ctx {512, 2048, 8192}，报 **单流 tok/s、每步 ms、
   TTFT（8k prompt prefill + 1 step）、峰值显存**；A B B A 交错消时钟漂移；
4. 可选：fp8 权重/head 开关、batch-1 fused kernel 实验，验证 §3 的 +20–40%。

预计实测值应与 §1 的"现有栈移植"档一致（300–500 tok/s 单流、B=16 ~5–7k、B=64 ~20k）；
若显著低于该区间，说明是移植/长 ctx KV 的实现问题而不是物理极限。

## 5. 和产品/竞品口径的关系

- 单流 300–500 tok/s（bf16、无投机）对 288M-active dense 是"中等"水平（decode 受 launch 延迟限制）；
  优化后 600–1,000 属于同类模型上游；带宽下限 2,600+/5,000+（fp8）是最终天花板。
- 批量侧 8k ctx 下单卡 20–70k tok/s，是"单卡可服务中小并发"的量级；对照 nanochat 的 8×H100
  部署口径时，要按每 ¥ 吞吐比，而不是单卡绝对数。
