#!/usr/bin/env bash
# R93 — P1 8B host runner (recipe: docs/r93-p1-launch-recipe.md).
#   d24_c1 dense / K=40k / seq 8192 / 2^21 tokens-per-step (b1 x accum256, S3 verdict)
#   LR 8e-4, WSD warmup 30 / decay-start 0.9 / min-lr 0.1
#   z-loss 1e-4 (fast path, +0.46%), EMA 0.995 every step (overhead ~0)
#   saves checkpoints at 1B/2B/4B/6B tokens (eval loop consumes+deletes them) and the final
#   raw + EMA weights.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
PY="${PY:-/root/t141/bin/python}"
TOK="${TOK:-$REPO/runs/r92/vocab/K40000_s1280/tokenizer.json}"
CFG="${CFG:-$REPO/docs/r93/configs/d24_c1_dense40k.json}"
DATA="${DATA:-/root/autodl-tmp/nanochat/p2_climbmix_40k}"
OUT="${OUT:-/root/autodl-tmp/nanochat/runs/p1}"
TOKENS="${TOKENS:-8000000000}"
LR="${LR:-8e-4}"
WARMUP="${WARMUP:-30}"
# shellcheck disable=SC1091
source tools/r90_env_dense5090.sh
export NANOV41_DS_TOKENIZER_JSON="$TOK"
mkdir -p "$OUT"
echo "=== R93 P1 start $(date -u +%FT%TZ) tokens=$TOKENS lr=$LR warmup=$WARMUP" | tee -a "$OUT/p1_launch.log"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$OUT/p1_launch.log"
exec env CE_CHUNK=8192 NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1 \
  "$PY" tools/r93_p0b_train.py --tokenizer "$TOK" -- \
  --arm p1_8b --config-json "$CFG" --data-dir "$DATA" --mix web=1.0 \
  --batch 1 --seq 8192 --grad-accum 256 --optimizer v41 \
  --peak-lr "$LR" --lr-schedule wsd --warmup "$WARMUP" --decay-start 0.9 --min-lr-ratio 0.1 \
  --balance-loss 0 --bias-speed 0 --z-loss 1e-4 --grad-clip 1.0 \
  --ema-decay 0.995 --ema-every 1 \
  --tokens "$TOKENS" --save-at-tokens 1b,2b,4b,6b \
  --log-every 10 --nan-check-every 1 --seed 1234 --abort-on-nan \
  --out "$OUT/p1_8b"
