"""Efficiency of every method x keep-ratio WITHOUT training (inference only):
GMACs, reducer params, throughput (img/s), peak VRAM.
  python scripts/bench.py --ckpt vim_t.pth --out RESULTS_DIR
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kanvim import metrics  # noqa: E402
from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.reducers import ALL_REDUCERS, build_reducer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="tiny")
ap.add_argument("--ckpt", default="vim_t.pth")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5, 0.25])
ap.add_argument("--reducers", nargs="+", default=[r for r in ALL_REDUCERS if r != "random_drop"])
ap.add_argument("--out", default="results")
a = ap.parse_args()

dim, depth = {"tiny": (192, 24), "small": (384, 24)}[a.backbone]
dev = "cuda" if torch.cuda.is_available() else "cpu"
backbone = load_vim(a.backbone, a.ckpt, 100)
rows = []
for name, ratio in [("identity", 1.0)] + [(r, q) for q in a.ratios for r in a.reducers]:
    model = ReducedVim(backbone, build_reducer(name, dim, (14, 14), ratio))
    row = {"method": name, "keep": ratio, "tokens": model.reducer.m,
           "GMACs": round(metrics.total_macs(model.cpu(), dim, depth, num_classes=100) / 1e9, 3),
           "reducer_params": sum(p.numel() for p in model.reducer.parameters())}
    if dev == "cuda":
        row["img_per_s"] = round(metrics.throughput(model, batch=a.batch, iters=a.iters, warmup=2), 1)
        row["vram_MB"] = round(metrics.peak_vram_mb(model, batch=a.batch), 0)
        model.cpu(); torch.cuda.empty_cache()
    rows.append(row)
    print(row, flush=True)

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"bench_{a.backbone}.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
    w.writeheader(); w.writerows(rows)
print("saved", path)
