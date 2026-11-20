#!/usr/bin/env python3
"""R93 — capability snapshot for a nanov41 checkpoint: text generation samples + a few
few-shot multiple-choice benchmarks (MMLU / ARC-easy / HellaSwag) and GSM8K (generation).

The point is orientation, not a leaderboard-grade eval: formats follow lm-eval conventions
loosely (few-shot, mean logprob over choice tokens for MC), so numbers are indicative.

Usage:
  HF_ENDPOINT=https://hf-mirror.com NANOV41_DS_TOKENIZER_JSON=<tok.json> \
  python3 tools/r93_capability_snapshot.py --ckpt /path/p1_8b.pt \
      --out runs/r93/cap/snapshot_raw.json [--ema-ckpt /path/p1_8b.ema0p995.pt] [--gen-only]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanov41.eval.provider import ModelProvider  # noqa: E402

PROMPTS = [
    ("story", "Once upon a time, in a small village by the sea, there lived a fisherman who"),
    ("story2", "The old library was silent except for the sound of rain. She opened the book and"),
    ("common_sense", "Q: If you put a glass of water in the freezer, what happens to the water?\nA:"),
    ("common_sense2", "Q: Why do leaves fall from trees in autumn?\nA:"),
    ("math", "Q: What is 17 + 26?\nA: The answer is"),
    ("math2", "Q: A train travels 60 miles per hour for 3 hours. How far does it travel?\nA: The train travels"),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("code2", "def is_palindrome(s):\n    \"\"\"Return True if s reads the same backwards.\"\"\"\n"),
    ("knowledge", "Q: What is the capital of France?\nA:"),
    ("knowledge2", "Q: Which planet is known as the Red Planet?\nA:"),
]


class EvalWrapper:
    def __init__(self, ckpt: str, dtype: str = "bf16", device: str = "cuda", max_len: int = 1024):
        self.provider = ModelProvider(checkpoint=ckpt, dtype=dtype, device=device)
        from tokenizers import Tokenizer

        tok_path = os.environ["NANOV41_DS_TOKENIZER_JSON"]
        self.tok = Tokenizer.from_file(tok_path)
        self.max_len = max_len
        self.eos = 1

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @torch.no_grad()
    def next_logits(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-self.max_len:]], dtype=torch.long)
        logits = self.provider.logits(x)
        return logits[0, -1].float()

    @torch.no_grad()
    def seq_logprobs(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-self.max_len:]], dtype=torch.long)
        logits = self.provider.logits(x)[0].float()
        return torch.log_softmax(logits, dim=-1)

    def generate(self, prompt: str, max_new: int = 96, temperature: float = 0.0,
                 top_k: int | None = 40) -> str:
        ids = self.encode(prompt)
        out = []
        for _ in range(max_new):
            logits = self.next_logits(ids + out)[: self.provider.vocab_size]
            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                    logits[logits < kth] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == self.eos:
                break
            out.append(nxt)
        return self.decode(out)

    def choice_logprob(self, context: str, choice: str) -> tuple[float, int]:
        """mean logprob of `choice` tokens given `context` (single forward, prefix assumption)."""
        ctx_ids = self.encode(context)
        full = self.encode(context + choice)
        # find the longest prefix of full equal to ctx_ids (tokenizer merges at the boundary are rare)
        start = len(ctx_ids) if full[: len(ctx_ids)] == ctx_ids else None
        if start is None or start >= len(full):
            start = max(len(ctx_ids) - 1, 0)
        ids = full
        if len(ids) > self.max_len:  # keep the tail that actually fits; shift the scoring offset
            drop = len(ids) - self.max_len
            ids = ids[drop:]
            start = max(1, start - drop)
        if start >= len(ids):
            return float("nan"), 0
        lp = self.seq_logprobs(ids)
        total, n = 0.0, 0
        for pos in range(start, len(ids)):
            total += float(lp[pos - 1, ids[pos]])
            n += 1
        return total / max(n, 1), n


def first_line(s: str, n: int = 120) -> str:
    s = " ".join(s.split())
    return s[:n]


def load_bench(name: str):
    import datasets

    ds = {
        "mmlu": lambda: datasets.load_dataset("cais/mmlu", "all", split="test"),
        "arc_easy": lambda: datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"),
        "hellaswag": lambda: datasets.load_dataset("Rowan/hellaswag", split="validation"),
        "gsm8k": lambda: datasets.load_dataset("openai/gsm8k", "main", split="test"),
    }[name]()
    # NOTE: slicing a Dataset yields a dict-of-lists; materialise rows as plain dicts so that
    # few-shot loops can slice/index them naturally.
    return [dict(r) for r in ds]


def run_mc(ev: EvalWrapper, rows, n_shot: int, n_eval: int, kind: str, budget_s: float = 400.0):
    """kind: 'mmlu' (score A-D letters), 'arc' (score option texts), 'hella' (score 4 endings)."""
    t0 = time.time()
    correct = 0
    total = 0
    for i in range(n_shot, min(n_shot + n_eval, len(rows))):
        r = rows[i]
        shots = rows[i - n_shot:i]
        if kind == "mmlu":
            head = "".join(
                f"{s['question']}\n" + "".join(f"{j+1}. {c}\n" for j, c in enumerate(s["choices"])) +
                f"Answer: {'ABCD'[int(s['answer'])]}\n\n" for s in shots)
            head += (f"{r['question']}\n" + "".join(f"{j+1}. {c}\n" for j, c in enumerate(r["choices"])) +
                     "Answer:")
            scores = [ev.choice_logprob(head, " " + L)[0] for L in "ABCD"]
            pred = int(torch.tensor(scores).argmax())
            gold = int(r["answer"])
            total += 1
            correct += int(pred == gold)
        elif kind == "arc":
            head = ""
            for s in shots:
                labels = [f"{L}. {t}" for L, t in zip(s["choices"]["label"], s["choices"]["text"])]
                head += f"Question: {s['question']}\n" + "\n".join(labels) + f"\nAnswer: {s['answerKey']}\n\n"
            opts = r["choices"]
            labels = [f"{L}. {t}" for L, t in zip(opts["label"], opts["text"])]
            head += f"Question: {r['question']}\n" + "\n".join(labels) + "\nAnswer:"
            scores = [ev.choice_logprob(head, " " + t)[0] for t in opts["text"]]
            pred = opts["label"][int(torch.tensor(scores).argmax())]
            total += 1
            correct += int(pred == r["answerKey"])
        else:  # hella
            head = "".join(f"{s['ctx']} {s['endings'][int(s['label'])]}\n" for s in shots)
            ctx = r["ctx"]
            scores = [ev.choice_logprob(ctx, " " + c)[0] for c in r["endings"]]
            pred = int(torch.tensor(scores).argmax())
            total += 1
            correct += int(pred == int(r["label"]))
        if time.time() - t0 > budget_s:
            break
    return (correct / total if total else float("nan")), total


def run_gsm8k(ev: EvalWrapper, rows, n_shot: int, n_eval: int, budget_s: float = 420.0):
    t0 = time.time()
    correct = 0
    total = 0
    for i in range(n_shot, min(n_shot + n_eval, len(rows))):
        q = rows[i]["question"]
        shots = "".join(f"Q: {s['question']}\nA: {s['answer']}\n\n" for s in rows[i - n_shot:i])
        prompt = shots + f"Q: {q}\nA:"
        gen = ev.generate(prompt, max_new=200, temperature=0.0)
        # gold = number after #### in the reference answer
        ref = rows[i]["answer"].split("####")[-1].strip()
        pred = ""
        for tok in gen.replace(",", " ").split():
            if any(ch.isdigit() for ch in tok):
                pred = tok.strip(".")
        total += 1
        correct += int(pred == ref)
        if time.time() - t0 > budget_s:
            break
    return (correct / total if total else float("nan")), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ema-ckpt", default="")
    ap.add_argument("--out", default="runs/r93/cap/snapshot.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--n-per-bench", type=int, default=150)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--benches", default="mmlu,arc_easy,hellaswag,gsm8k")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"ckpt": args.ckpt, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ev = EvalWrapper(args.ckpt, dtype=args.dtype, device=args.device)
    report["model"] = {"params": ev.provider.n_params, "flops_per_token": ev.provider.flops_per_token,
                       "config": ev.provider.config_name}
    samples = []
    for tag, prompt in ([] if args.bench_only else PROMPTS):
        for temp, label in ((0.0, "greedy"), (0.8, "sample")):
            txt = ev.generate(prompt, max_new=args.max_new, temperature=temp)
            samples.append({"tag": tag, "mode": label, "prompt": prompt, "completion": txt})
            print(f"[gen:{label}] {tag}: {first_line(txt)}", flush=True)
    report["generation"] = samples

    if not args.gen_only:
        benches = {}
        want = [b.strip() for b in args.benches.split(",") if b.strip()]
        try:
            if "mmlu" not in want: raise RuntimeError("skip")
            mmlu = load_bench("mmlu")
            acc, n = run_mc(ev, mmlu, 5, args.n_per_bench, "mmlu")
            benches["mmlu"] = {"acc": acc, "n": n, "shots": 5, "note": "letter logprob, n<=150"}
        except Exception as exc:  # noqa: BLE001
            benches["mmlu"] = {"error": repr(exc)[:200]}
        print("[bench] mmlu", benches.get("mmlu"), flush=True)
        try:
            if "arc_easy" not in want: raise RuntimeError("skip")
            arc = load_bench("arc_easy")
            acc, n = run_mc(ev, arc, 25, args.n_per_bench, "arc")
            benches["arc_easy"] = {"acc": acc, "n": n, "shots": 25}
        except Exception as exc:  # noqa: BLE001
            benches["arc_easy"] = {"error": repr(exc)[:200]}
        print("[bench] arc_easy", benches.get("arc_easy"), flush=True)
        try:
            if "hellaswag" not in want: raise RuntimeError("skip")
            hella = load_bench("hellaswag")
            acc, n = run_mc(ev, hella, 10, args.n_per_bench, "hella")
            benches["hellaswag"] = {"acc": acc, "n": n, "shots": 10}
        except Exception as exc:  # noqa: BLE001
            benches["hellaswag"] = {"error": repr(exc)[:200]}
        print("[bench] hellaswag", benches.get("hellaswag"), flush=True)
        try:
            if "gsm8k" not in want: raise RuntimeError("skip")
            gsm = load_bench("gsm8k")
            acc, n = run_gsm8k(ev, gsm, 8, 25)
            benches["gsm8k"] = {"acc": acc, "n": n, "shots": 8, "note": "greedy, exact match, n<=25"}
        except Exception as exc:  # noqa: BLE001
            benches["gsm8k"] = {"error": repr(exc)[:200]}
        print("[bench] gsm8k", benches.get("gsm8k"), flush=True)
        report["benchmarks"] = benches
    del ev

    if args.ema_ckpt:
        ev2 = EvalWrapper(args.ema_ckpt, dtype=args.dtype, device=args.device)
        ema_samples = []
        for tag, prompt in PROMPTS[:6]:
            txt = ev2.generate(prompt, max_new=args.max_new, temperature=0.0)
            ema_samples.append({"tag": tag, "mode": "greedy", "prompt": prompt, "completion": txt})
        report["generation_ema"] = ema_samples
        del ev2

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(report, indent=1))
    print(f"[cap] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
