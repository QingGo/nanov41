# P0a KD dataset (task-17): Arcee teacher logits -> K=40k projected KD data for P1

**Status: FINAL FROZEN SNAPSHOT = `final_partial_v140`** (2026-11-15 17:20 CST) — a valid,
verified, loadable, **resumable** dataset: **140 of 287 shards, 134,217,728 official tokens,
143,417,520 piece rows, 9,752,391,872 B = 9.75 GB (9.08 GiB)**, built by `tools/r93_build_kd40k.py`
on AutoDL host `<instance-id>` (RTX 3090, **no-card mode**: 0.5 CPU core, 2 GiB RAM, 45 GB
data disk; no GPU used). Frozen per Lead's call; the data disk persists across shutdown and the build
can be continued later with the exact command in the "FINAL FROZEN SNAPSHOT" section below.

## ⚠️ Build interruption: host shutdown (2026-11-15 ~16:15 CST)
The host `<instance-id>` went down at ~16:21 via its **4 h AutoDL TTL** (Lead's scheduling
miss, not a crash; auto-shutdown), was **booted again in no-card mode at ~16:38 with TTL 21:38**, and
the build resumed from the durable snapshot (`partial_v100` → `partial_v117`). The data disk survived
intact and the wrapper was relaunched. The builder now trims the resume point to the on-disk snapshot
length (meta can be ahead of the header between milestones), so interleaved snapshots are always
loadable and resuming never restarts from zero. UUID note: ours is `<instance-id>`; the user's
3090 `<instance-id>` is never touched.

## FINAL FROZEN SNAPSHOT (`final_partial_v140`, 2026-11-15 17:20 CST)
Build was frozen per Lead's call (host TTL 21:38; **data disk persists**, artifact can be continued
later with the same command). Verified PASS (`verify_v140.json`).

| item | value |
|---|---|
| path | `/root/autodl-tmp/kd40k/` on host `<instance-id>` (+ `sample/`) — dataset persists on the data disk across shutdown |
| status | `final_partial_v140`, **resumable** |
| shards | **140 of 287** (48.8% of the full target) |
| official tokens | **134,217,728** (= 140 x 950,272); usable teacher rows ~134,217,588; blocked/unusable = 7.25% |
| piece rows | **143,417,520** (dense) |
| **bytes** | **9,752,391,872 B = 9.75 GB (9.08 GiB)** = exactly 68 B/piece + 128 B headers |
| files | `input_ids.npy` u16[P] 286,835,168 B; `top_ids.npy` u16[P,16] 4,589,360,768 B; `top_logprobs.npy` f16[P,16] 4,589,360,768 B; `rest_mass.npy` f16[P] 286,835,168 B; `meta.json`; `sample/` (2,090,600 pieces, 139 MB) |
| missing vs full target | remaining 147 shards -> 287 shards = 294.0M pieces = **19.99 GB (18.62 GiB)**; the 20 GB hard cap + `--max-bytes` trim is armed |
| P1 context | P1 sizing moved to **8-10B tokens** after P0b, so full KD (~4.4% coverage at 5.75B, ~2.7-3.4% at 9-10B) is useful but not blocking. This 134.2M-token snapshot = **1.34-1.49% of P1 tokens** => ~+0.86-0.95% effective tokens at the measured 1.64x KD data efficiency |

**Verification PASS** (`--verify` on the final frozen state; `verify_v140.json`): `input_ids` max
**39997 < 41280**; dtypes u16/u16/f16/f16; shapes [P]/[P,16]/[P,16]/[P]; blocked fraction **0.0725**
(expected 0.0724); sentinel rows == blocked rows exactly; blocked `rest_mass` = 1.0; renormalised
log-prob rows sum **1.0000**; **stored absolute teacher mass 0.89146** (k=16 expected 0.89151);
byte total exactly 68 B/piece.

### Exact resume command (after the host is booted again; no data loss)
```bash
# 1) one build step (resumes from meta; ~25 min; --resume trims to the on-disk snapshot now)
cd /root/nanov41 && /root/miniconda3/bin/python -u tools/r93_build_kd40k.py --num-shards 287 --resume \
    --download-workers 8 --sample-shards 2 --snapshot-every 25 --max-bytes 20000000000 \
    --out /root/autodl-tmp/kd40k --stage-dir /root/autodl-tmp/kd40k_stage \
    >> /root/autodl-tmp/kd40k_build.log 2>&1
# 2) or the unattended wrapper (already on the data disk; runs 25-min iterations + auto-snapshots)
cd /root/nanov41 && setsid nohup /root/autodl-tmp/kd40k_resume.sh > /root/autodl-tmp/kd40k_resume.out 2>&1 &
# verify / force a snapshot
/root/miniconda3/bin/python tools/r93_build_kd40k.py --verify --out /root/autodl-tmp/kd40k
/root/miniconda3/bin/python tools/r93_build_kd40k.py --finalize-partial --status-tag partial_vN --shards-done N --out /root/autodl-tmp/kd40k
```
Note: the builder needs the repo + K40k tokenizer files on the host (`/root/nanov41` system disk,
`runs/p5_local/tokenizer.json`, `runs/r92/vocab/K40000_s1280/tokenizer.json` + `special_map.json` --
all already there and small; re-sync from this Mac repo if the system disk is ever reset).

### History of snapshots (all loadable, same format)
v75 = 4.865 GiB (`meta_v75.json`/`verify_v75.json`), v100, v117 = 7.59 GiB, **v140 = 9.08 GiB
(final)**. Milestone snapshots overwrite the same directory and remain resumable.

## Spec / conventions (unchanged from the approved design)
- teacher: `arcee-ai/DeepSeek-DCLM-Logits-Packed-8192` (legacy top-128 compression, top-32 exact);
  950,272 tokens/shard (116 x 8192).
- student vocab **K=40k scheme A**: keep `id < 40,000`; base vocab is 128,000 ids + 1,280 added
  specials remapped to **40,000..41,279** ⇒ **V=41,280**; pruned ids expand via the BPE merge tree.
- **dense rows over piece positions** (1.0785 pieces/official token measured); teacher row `t`
  stored at the **last piece of official token t** (student predicts the first piece of t+1);
  projection: id<40k stays, 40k..127,999 → first BPE piece, added specials 128,000..129,279 →
  40,000+rank, ≥129,280 unrepresentable (mass ~0); **collisions summed**; row renormalised over the
  k=16 stored entries (rank order) with per-row `rest_mass = 1 - absolute teacher mass captured`.
- KD target = `(1 - rest_mass) * softmax(top_logprobs)`; loss
  `-Σ p_abs·log q - rest_mass·log(1-Σ q)` (TAD-style rest bucket).
- unusable positions (~7.25%): `top_ids=65535`, `top_logprobs=-inf`, `rest_mass=1.0`.

## Measured teacher mass coverage (5 decoded shards = 4.75M tokens; `teacher_stats.json`)
| k | absolute teacher top-k mass | rest | projected retained | collision entries | top-1 agree |
|---|---|---|---|---|---|
| 8 | 0.8280 | 0.1720 | 0.8280 | 1.52% | 0.5072 |
| **16** | **0.8915** | **0.1085** | 0.8915 | 2.05% | 0.5072 |
| 24 | 0.9211 | 0.0789 | 0.9211 | 2.43% | 0.5072 |
| 32 | 0.9392 | 0.0608 | 0.9392 | 2.74% | 0.5072 |
Projection loses **zero** teacher mass (all top-32 ids representable in V=41,280); `m32_abs=0.93921`.
Scheme-A expansion: **inflation 1.078523**, usable frac **0.927194**.

## ⚠️ Incident: Arcee/xethub mirror throttling (not a code issue)
Timeline of the 287-shard build on `<instance-id>` (per-IP/host throttle, Lead measured a *fresh* host
`<instance-id>` getting 3.6 MB/s on the same URL at the same time):
- 12:45–13:40 CST: **10–11.7 MB/s aggregate** (20 parallel conns) → first ~75 shards built;
- 14:00–14:30: fell to 2–4 MB/s; stale sockets hung 15–22 min (curl had no stall guard) → fixed with
  `--speed-limit 8KB/s --speed-time 20s`, `--max-time 110s` (forces fresh connection on resume)
  and fewer workers; transient recovery, then:
- 14:45–15:05: **collapse to 43 KB/s single-conn**, parallel requests hang (7–12 B/s),
  direct xethub CAS 0.6–0.75 MB/s, **huggingface.co unreachable**; ~45–50 GB had been downloaded
  by then. 15:05: partial recovery to ~1 MB/s. Not mirror-global: per-IP/host.
- Also found (and fixed) that some stage files from the killed runs had right size but **corrupt
  snappy pages** (duplicated resume); the builder now validates the first row group after download,
  rolls back `.npy` cursors and re-downloads on any read error (retry 3x), so corrupt stage files
  cannot enter the dataset.

## How to continue (on the host; everything is resumable, nothing deleted)
```bash
cd /root/nanov41
# one build step (resumes from meta.json; snapshots at every 25 shards; 20 GB hard cap)
/root/miniconda3/bin/python -u tools/r93_build_kd40k.py --num-shards 287 --resume \
    --download-workers 8 --sample-shards 2 --snapshot-every 25 --max-bytes 20000000000 \
    --out /root/autodl-tmp/kd40k --stage-dir /root/autodl-tmp/kd40k_stage \
    >> /root/autodl-tmp/kd40k_build.log 2>&1
# auto-resume wrapper (already running as /root/autodl-tmp/kd40k_resume.sh; log kd40k_retry.log)
# force a loadable snapshot at the current state:
/root/miniconda3/bin/python tools/r93_build_kd40k.py --finalize-partial --status-tag partial_vN \
    --shards-done N --out /root/autodl-tmp/kd40k
# verify:
/root/miniconda3/bin/python tools/r93_build_kd40k.py --verify --out /root/autodl-tmp/kd40k
```
If a feeder host (`<instance-id>`) is used: pre-populate `--stage-dir` with
`train-XXXXX-of-01377.parquet` files (they are validated by size+schema before use); the dataset must
still be appended on `<instance-id>` (or rebuilt elsewhere with the same builder and the sample synced).

## P1 integration notes (r91_kd_train-style KD loss)
```python
import numpy as np
d = "/root/autodl-tmp/kd40k"
ids  = np.load(f"{d}/input_ids.npy", mmap_mode="r")      # u16 [P] piece stream (CE input)
tids = np.load(f"{d}/top_ids.npy", mmap_mode="r")        # u16 [P,16], 65535 = sentinel
lps  = np.load(f"{d}/top_logprobs.npy", mmap_mode="r")   # f16 [P,16] log-space, renorm over 16
rest = np.load(f"{d}/rest_mass.npy", mmap_mode="r")      # f16 [P], 1 - captured absolute mass
# usable mask: np.isfinite(lps[:,0]) (or tids[:,0] != 65535)
# absolute target: p_abs = (1 - rest) * softmax(lps)  (rank-ordered slots)
# rest bucket:     L = -(p_abs * log q).sum(-1) - rest * log(1 - q.sum(-1))
# gather head rows for tids per usable position; clamp sentinel ids (65535) to 0 before gather.
```
`samples/` (2.09M pieces) has the identical format for fast integration tests; `sample/sample_meta.json`
notes the convention. `runs/r93/p0a/kd/teacher_stats.json` = coverage evidence;
`runs/r93/p0a/kd/{measure_teacher_stats.py,probe_one_shard.py}` = evidence scripts;
`tools/r93_build_kd40k.py` = builder/verify/finalize (canonical deliverable).
