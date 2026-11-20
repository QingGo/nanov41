#!/usr/bin/env bash
# R93 — P1 evaluation loop: as soon as a `--save-at-tokens` mark checkpoint (or the final raw/EMA
# checkpoint) appears, score it on the K=40k ClimbMix-val panel (3M tokens), then delete the mark
# checkpoint to keep the data disk clean.  Runs alongside training (eval model ~2 GiB on top of the
# ~20 GiB training footprint).
# Output: $OUT/eval_<tag>.json ; per-ckpt log $OUT/eval_<tag>.log ; loop log $OUT/p1_eval.log
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
PY="${PY:-/root/t141/bin/python}"
TOK="${TOK:-$REPO/runs/r92/vocab/K40000_s1280/tokenizer.json}"
PANEL="${PANEL:-/root/autodl-tmp/nanochat/p0_eval_40k}"
OUT="${OUT:-/root/autodl-tmp/nanochat/runs/p1}"
MAXTOK="${MAXTOK:-3000000}"
INTERVAL="${INTERVAL:-300}"
L="$OUT/p1_eval.log"
mkdir -p "$OUT"
echo "=== R93 P1 eval loop $(date -u +%FT%TZ) panel=$PANEL max_tokens=$MAXTOK" >> "$L"

eval_one() {  # eval_one <ckpt> <tag> [delete-after]
  local ck="$1" tag="$2" del="${3:-0}"
  local ev="$OUT/eval_${tag}.json"
  [ -f "$ev" ] && { echo "[p1eval] have $ev" >> "$L"; return 0; }
  echo "[p1eval] === $tag $(date -u +%FT%TZ)" >> "$L"
  NANOV41_DS_TOKENIZER_JSON="$TOK" "$PY" tools/eval_t0.py --dataset "$PANEL" \
    --domains climbmix_val --ckpt "$ck" --provider model --device cuda --dtype bf16 \
    --window 2048 --batch-size 8 --max-tokens-per-domain "$MAXTOK" \
    --out "$ev" >> "$OUT/eval_${tag}.log" 2>&1
  local rc=$?
  if [ "$rc" = "0" ]; then
    "$PY" - "$ev" >> "$L" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
m = ((d.get("dataset", {}).get("domains") or {}).get("climbmix_val") or {}).get("metrics", {}).get("all", {})
print(f"[p1eval] {sys.argv[1].split('/')[-1]} bpb={m.get('bpb')} bits/token={m.get('bits_per_token')}")
PYEOF
    [ "$del" = "1" ] && rm -f "$ck"
  else
    echo "[p1eval] FAILED $tag rc=$rc (see $OUT/eval_${tag}.log)" >> "$L"
  fi
  return 0
}

trap 'echo "[p1eval] exit $(date -u +%FT%TZ)" >> "$L"; exit 0' TERM INT
while true; do
  for ck in "$OUT"/p1_8b.t*.pt; do
    [ -f "$ck" ] || continue
    base="$(basename "$ck" .pt)"                     # p1_8b.t1000000000
    tag="$(echo "$base" | sed -E 's/^p1_8b\.t([0-9]+)$/t\1/')"
    eval_one "$ck" "$tag" 0   # keep mark ckpts: insurance against budget/instance interruption
  done
  if [ -f "$OUT/p1_8b.pt" ]; then
    eval_one "$OUT/p1_8b.pt" final 0
    [ -f "$OUT/p1_8b.ema0p995.pt" ] && eval_one "$OUT/p1_8b.ema0p995.pt" ema0p995 0
    break
  fi
  if ! pgrep -f "arm p1_8b" >/dev/null 2>&1; then
    echo "[p1eval] training process gone but no final ckpt -> exit" >> "$L"; break
  fi
  sleep "$INTERVAL"
done
echo "R93_P1_EVAL_DONE $(date -u +%FT%TZ)" >> "$L"
