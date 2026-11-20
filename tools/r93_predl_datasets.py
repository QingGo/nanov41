import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import datasets
print("datasets", datasets.__version__, flush=True)
jobs = [("cais/mmlu", "all", "test"), ("Rowan/hellaswag", None, "validation"),
        ("openai/gsm8k", "main", "test"), ("allenai/ai2_arc", "ARC-Challenge", "test")]
for name, cfg, split in jobs:
    t0 = time.time()
    try:
        d = datasets.load_dataset(name, cfg, split=split) if cfg else datasets.load_dataset(name, split=split)
        print(f"{name} {cfg} {split}: {len(d)} rows in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"{name} FAILED: {repr(e)[:200]}", flush=True)
print("PREDL_DONE", flush=True)
