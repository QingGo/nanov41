#!/bin/bash
# P2b: seed re-validation of E-none vs E-default at 0.6B tokens (paired by seed).
set -u
cd /root/nanov41 || exit 1
source tools/final_env.sh
export NANOV41_MHC_V4=1 NANOV41_V4_MODE=split NANOV41_MOE_LAYOUT=fast
export NANOV41_ATTN_DKV_BF16=1 NANOV41_MOE_DWS_INLINE=1 NANOV41_MOE_TILE=64,64,64,4,3
export NANOV41_DS_TOKENIZER_JSON=/root/dsv41/tokenizer.json
PY=${PY:-/root/miniconda3/bin/python}
DATA=/root/autodl-tmp/p2_data
TOKENS=600000000
mkdir -p runs/p2b
run_arm() {
  local arm="$1" cfg="$2" engram="$3" seed="$4"
  local name="${arm}_s${seed}"
  echo "===== P2b $name cfg=$cfg engram=$engram seed=$seed $(date -u +%H:%M:%S) =====" | tee -a runs/p2b/arms.log
  $PY tools/p2_train.py --arm "$name" --config "$cfg" --engram "$engram" \
      --tokens $TOKENS --data-dir "$DATA" --batch 16 --seq 1024 --seed "$seed" \
      --peak-lr 3e-3 --engram-lr 1e-2 --log-every 1000 --save-every 10000 \
      --out "runs/p2b/$name" >> "runs/p2b/$name.train.log" 2>&1
  local rc=$?
  echo "  train rc=$rc" | tee -a runs/p2b/arms.log
  [ $rc -ne 0 ] && { echo "FAILED $name" | tee -a runs/p2b/arms.log; return $rc; }
  $PY tools/eval_t0.py --dataset /root/nanov41/runs/eval/data_pilot \
      --ckpt "runs/p2b/$name.pt" --device cuda --dtype bf16 --window 1024 --batch-size 16 \
      --out "runs/p2b/$name.t0.json" >> "runs/p2b/$name.eval.log" 2>&1
  echo "  eval rc=$? $(date -u +%H:%M:%S)" | tee -a runs/p2b/arms.log
}
# paired order: same seed, both arms back-to-back
run_arm E-none    d12 none 1235
run_arm E-default d12 d12  1235
run_arm E-none    d12 none 1236
run_arm E-default d12 d12  1236
echo "ALL DONE $(date -u +%H:%M:%S)" | tee -a runs/p2b/arms.log
