"""Print (or run) the experiment grid.
  python scripts/sweep.py --dataset cifar100 --backbone tiny --ckpt vim_t.pth > cifar.sh
  python scripts/sweep.py ... --run
Ratios 1.0 = Vim baseline (reducer 'identity'), run once per seed."""
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar100")
ap.add_argument("--backbone", default="tiny")
ap.add_argument("--ckpt", default="vim_t.pth")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5, 0.25])
ap.add_argument("--reducers", nargs="+", default=["avg_pool", "max_pool", "mlp_pool", "kan_pool", "mlp_prune", "kan_prune"])
ap.add_argument("--extra", default="", help="extra args appended to every command, e.g. \"--grad_ckpt --batch 32 --out DIR\"")
ap.add_argument("--no_baseline", action="store_true")
ap.add_argument("--run", action="store_true")
a = ap.parse_args()

base = f"python train.py --dataset {a.dataset} --backbone {a.backbone} --ckpt {a.ckpt} --epochs {a.epochs} {a.extra}".strip()
cmds = [] if a.no_baseline else [f"{base} --reducer identity --ratio 1.0 --seed {s}" for s in a.seeds]
cmds += [f"{base} --reducer {r} --ratio {q} --seed {s}" for s in a.seeds for q in a.ratios for r in a.reducers]
for c in cmds:
    print(c)
    if a.run:
        subprocess.run(c, shell=True, check=True)
