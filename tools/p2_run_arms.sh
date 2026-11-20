#!/bin/bash
# P2 A/B: five 0.6B-token arms on the R69-tuned graph+Engram stack, then T0 multi-domain BPB.
set -u
cd /root/nanov41 || exit 1
source tools/final_env.sh
export NANOV41_MHC_V4=1 NANOV41_V4_MODE=split NANOV41_MOE_LAYOUT=fast
export NANOV41_ATTN_DKV_BF16=1 NANOV41_MOE_DWS_INLINE=1 NANOV41_MOE_TILE=64,64,64,4,3
export NANOV41_DS_TOKENIZER_JSON=/root/dsv41/tokenizer.json
export TILE_KERNELS_PATH=/root/TileKernels PYTHONPATH=/root/TileKernels
PY=/root/t141/bin/python
DATA=/root/autodl-tmp/p2_data
TOKENS=600000000
mkdir -p runs/p2
run_arm() {
  local name="$1" cfg="$2" engram="$3"
  echo "===== ARM $name cfg=$cfg engram=$engram $(date -u +%H:%M:%S) =====" | tee -a runs/p2/arms.log
  $PY tools/p2_train.py --arm "$name" --config "$cfg" --engram "$engram" \
      --tokens $TOKENS --data-dir "$DATA" --batch 16 --seq 1024 \
      --peak-lr 3e-3 --engram-lr 1e-2 --log-every 1000 --save-every 10000 \
      --out "runs/p2/$name" >> "runs/p2/$name.train.log" 2>&1
  local rc=$?
  echo "  train rc=$rc" | tee -a runs/p2/arms.log
  if [ $rc -ne 0 ]; then echo "FAILED $name" | tee -a runs/p2/arms.log; return $rc; fi
  $PY tools/eval_t0.py --dataset /root/nanov41/runs/eval/data_pilot \
      --ckpt "runs/p2/$name.pt" --device cuda --dtype bf16 --window 1024 --batch-size 16 \
      --out "runs/p2/$name.t0.json" >> "runs/p2/$name.eval.log" 2>&1
  echo "  eval rc=$? $(date -u +%H:%M:%S)" | tee -a runs/p2/arms.log
}
run_arm E-none    d12    none
run_arm E-thin    d12    d12_thin
run_arm E-default d12    d12
run_arm E-thick   d12    d12_thick
run_arm d12e32    d12e32 none
echo "ALL DONE $(date -u +%H:%M:%S)" | tee -a runs/p2/arms.log
