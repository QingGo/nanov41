#!/usr/bin/env bash
# R93 — mid-run LR probe. From the 4B-token checkpoint of the P1 run (trained at LR 8e-4),
# resume weights+data-position (fresh optimizer) and train a short 0.2B-token segment at a
# constant LR, then evaluate the visible panel bpb. This measures whether 8e-4 is below the
# optimum for this long run, without touching the main run.
#   arms: 8e-4 (control) 1.2e-3 1.8e-3 2.6e-3   (env ARMS to override)
#   each arm: --tokens 4.2e9 from step 1907 -> 96 steps x 2^21 tokens = 0.2B tokens
# Output: /root/autodl-tmp/nanochat/runs/probe/arm_<tag>.{pt,train.json,series.json,eval.json}
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
PY="${PY:-/root/t141/bin/python}"
TOK="${TOK:-/root/autodl-tmp/nanochat/tok40k/tokenizer.json}"
CFG="${CFG:-$REPO/docs/r93/configs/d24_c1_dense40k.json}"
DATA="${DATA:-/root/autodl-tmp/nanochat/p2_climbmix_40k}"
PANEL="${PANEL:-/root/autodl-tmp/nanochat/p0_eval_40k}"
CKPT="${CKPT:-/root/autodl-tmp/nanochat/runs/probe/p1_8b.t4000000000.pt}"
OUT="${OUT:-/root/autodl-tmp/nanochat/runs/probe}"
ARMS="${ARMS:-8e-4 1.2e-3 1.8e-3 2.6e-3}"
TOKENS_TOTAL="${TOKENS_TOTAL:-4200000000}"
L="$OUT/probe.log"
mkdir -p "$OUT"
# shellcheck disable=SC1091
source tools/r90_env_dense5090.sh
export NANOV41_DS_TOKENIZER_JSON="$TOK"
echo "=== R93 probe start $(date -u +%FT%TZ) arms=[$ARMS] ckpt=$CKPT" | tee -a "$L"
for LR in $ARMS; do
  tag="$(echo "$LR" | tr '.' 'p')"
  O="$OUT/arm_$tag"
  if [ -f "$O.eval.json" ]; then echo "[probe] skip lr=$LR (already has eval)" | tee -a "$L"; continue; fi
  echo "[probe] === arm lr=$LR $(date -u +%FT%TZ)" | tee -a "$L"
  rm -f "$O.train.json" "$O.pt"
  env CE_CHUNK=8192 NANOV41_RMS_FUSED=1 NANOV41_SWIGLU_FUSED=1 \
    "$PY" tools/r93_p0b_train.py --tokenizer "$TOK" -- \
    --arm "probe_$tag" --config-json "$CFG" --data-dir "$DATA" --mix web=1.0 \
    --batch 1 --seq 8192 --grad-accum 256 --optimizer v41 \
    --peak-lr "$LR" --lr-schedule wsd --warmup 0 --decay-start 1.0 --min-lr-ratio 1.0 \
    --balance-loss 0 --bias-speed 0 --z-loss 1e-4 --grad-clip 1.0 --ema-decay 0 \
    --resume "$CKPT" --tokens "$TOKENS_TOTAL" \
    --log-every 5 --nan-check-every 1 --seed 1234 --abort-on-nan \
    --out "$O" >> "$L" 2>&1
  rc=$?
  echo "[probe] arm lr=$LR train rc=$rc $(date -u +%FT%TZ)" | tee -a "$L"
  if [ -f "$O.pt" ]; then
    NANOV41_DS_TOKENIZER_JSON="$TOK" "$PY" tools/eval_t0.py --dataset "$PANEL" \
      --domains climbmix_val --ckpt "$O.pt" --provider model --device cuda --dtype bf16 \
      --window 2048 --batch-size 8 --max-tokens-per-domain 3000000 \
      --out "$O.eval.json" >> "$L" 2>&1
    echo "[probe] arm lr=$LR eval rc=$? $(date -u +%FT%TZ)" | tee -a "$L"
  fi
done
# summary
"$PY" - "$OUT" <<'PYEOF' | tee -a "$L"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = []
for ev in sorted(out.glob("arm_*.eval.json")):
    try:
        d = json.loads(ev.read_text())
        m = ((d.get("dataset", {}).get("domains") or {}).get("climbmix_val") or {}).get("metrics", {}).get("all", {})
        rows.append({"arm": ev.name.replace("arm_", "").replace(".eval.json", ""), "bpb": m.get("bpb")})
    except Exception as exc:  # noqa: BLE001
        print("[probe] eval parse failed", ev, exc)
rows.sort(key=lambda r: r["bpb"] or 9e9)
print("[probe] SUMMARY", json.dumps(rows))
if rows:
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
PYEOF
echo "R93_PROBE_DONE $(date -u +%FT%TZ)" | tee -a "$L"
