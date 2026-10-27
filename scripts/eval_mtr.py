"""MTR (training-free) on the fine-tuned baseline: val + test top-1, GMACs, throughput, VRAM.
  python scripts/eval_mtr.py --dataset cifar100 --teacher_ckpt results_plan/teacher_identity.pt \
      --val_size 5000 --ratios 0.75 0.5 0.25 --out results_plan/mtr
Writes one json per ratio in the same format as train.py (reducer = 'mtr')."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kanvim import metrics  # noqa: E402
from kanvim.data import build_loaders_val  # noqa: E402
from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.mtr import MTRVim  # noqa: E402
from kanvim.reducers import build_reducer  # noqa: E402
from kanvim.vim_pure import VimPure  # noqa: E402

DIMS = {"tiny": (192, 24), "small": (384, 24), "stub": (64, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--backbone", default="tiny", choices=list(DIMS))
    ap.add_argument("--teacher_ckpt", required=True, help="fine-tuned identity baseline (train.py --save_ckpt)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5, 0.25])
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--test_subset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_bench", action="store_true")
    ap.add_argument("--out", default="results/mtr")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    dim, depth = DIMS[a.backbone]

    _, va, te, nc = build_loaders_val(a.dataset, a.data_root, a.batch, a.workers, a.val_size, 0, a.test_subset)
    os.makedirs(a.out, exist_ok=True)
    for ratio in a.ratios:
        name = f"{a.dataset}_{a.backbone}_mtr_r{ratio}_s{a.seed}.json"
        if os.path.exists(os.path.join(a.out, name)):
            print("exists", name); continue
        bb = VimPure(dim, depth, nc) if a.backbone == "stub" else load_vim(a.backbone, None, nc, "pure")
        rv = ReducedVim(bb, build_reducer("identity", dim, (14, 14), 1.0))
        rv.load_state_dict(torch.load(a.teacher_ckpt, map_location="cpu"))
        model = MTRVim(rv, ratio).to(dev).eval()

        def score(loader):
            hit = n = 0
            loss = 0.0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                        out = model(x)
                    loss += torch.nn.functional.cross_entropy(out.float(), y, reduction="sum").item()
                    hit += (out.argmax(1) == y).sum().item(); n += y.numel()
            return 100.0 * hit / n, loss / n

        (vacc, vloss), (tacc, tloss) = score(va), score(te)
        res = {"args": dict(vars(a), reducer="mtr", ratio=ratio, epochs=0, lr=None, impl="pure"), "log": [],
               "stopped": "training-free", "val_top1": vacc, "val_loss": vloss, "best_epoch": None,
               "top1": tacc, "test_loss": tloss, "reducer_params": 0,
               "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
               "mean_tokens": sum(model.layer_tokens()) / depth - 1,
               "GMACs": (metrics.vim_macs_layers(model.layer_tokens(), dim, num_classes=nc) + model.extra_macs(dim)) / 1e9,
               "scan_impl": "pure", "tokens_per_block": model.tokens}
        if dev == "cuda" and not a.no_bench:
            res["throughput"] = metrics.throughput(model, batch=64)
            res["vram_MB"] = metrics.peak_vram_mb(model, batch=64)
        json.dump(res, open(os.path.join(a.out, name), "w"), indent=1)
        print(f"MTR keep {ratio}: val {vacc:.2f} test {tacc:.2f} GMACs {res['GMACs']:.3f} "
              f"(tokens {model.tokens[0]}->{model.tokens[-1]})", flush=True)


if __name__ == "__main__":
    main()
