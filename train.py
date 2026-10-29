"""Fine-tune a pretrained Vim with one token reducer, then evaluate accuracy + efficiency.

  python train.py --dataset cifar100 --reducer kan_prune --ratio 0.5 --seed 0 --backbone tiny --ckpt vim_t.pth
  python train.py --backbone stub --epochs 1 --smoke        # CPU smoke test, no Vim needed

Standard protocol (used by scripts/run_plan.py): --val_size 5000 holds out part of the train set;
every epoch is scored on val, the best-val model is kept (early stopping with --patience) and only
that model is evaluated on the test set, once. Without --val_size the old behaviour is kept
(test set scored every epoch, last epoch reported).
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from kanvim import metrics
from kanvim.kan_layers import count_params
from kanvim.model import ReducedVim, load_vim
from kanvim.reducers import build_multistage, build_reducer
from kanvim.vim_pure import VimPure

DIMS = {"tiny": (192, 24), "small": (384, 24), "stub": (64, 2)}
PAUSED = 3  # exit code: stopped before --deadline, resume by running the same command again
AMP_DTYPE = torch.float16  # --bf16 switches autocast to torch.bfloat16


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    return evaluate_full(model, loader, dev, amp)[0]


@torch.no_grad()
def evaluate_full(model, loader, dev, amp):
    """-> (top-1 %, mean cross-entropy)"""
    model.eval()
    hit = n = 0
    loss = 0.0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev, dtype=AMP_DTYPE, enabled=amp and dev == "cuda"):
            out = model(x)
        loss += F.cross_entropy(out.float(), y, reduction="sum").item()
        hit += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return 100.0 * hit / n, loss / n


def mix_batch(x, y, a):
    """Mixup / CutMix (one of them per batch). -> x, (y_a, y_b, lam) or None"""
    if a.mixup <= 0 and a.cutmix <= 0:
        return x, None
    cut = a.cutmix > 0 and (a.mixup <= 0 or random.random() < 0.5)
    lam = float(np.random.beta(a.cutmix, a.cutmix) if cut else np.random.beta(a.mixup, a.mixup))
    perm = torch.randperm(x.size(0), device=x.device)
    if cut:
        H, W = x.shape[2:]
        r = math.sqrt(1 - lam)
        h, w = int(H * r), int(W * r)
        cy, cx = random.randrange(H), random.randrange(W)
        y1, y2, x1, x2 = max(cy - h // 2, 0), min(cy + h // 2, H), max(cx - w // 2, 0), min(cx + w // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, (y, y[perm], lam)


def train_step(model, teacher, x, y, opt, scaler, a, dev, amp, mix=None):
    with torch.autocast(dev, dtype=AMP_DTYPE, enabled=amp):
        out = model(x)
        if mix is None:
            ce = F.cross_entropy(out, y, label_smoothing=0.1)
        else:
            ya, yb, lam = mix
            ce = lam * F.cross_entropy(out, ya, label_smoothing=0.1) + (1 - lam) * F.cross_entropy(out, yb, label_smoothing=0.1)
        loss = ce + a.l1 * model.reg_loss()
        if teacher is not None:
            with torch.no_grad():
                t = teacher(x)
            loss = loss + a.distill * F.kl_div(F.log_softmax(out.float(), 1), F.softmax(t.float(), 1),
                                               reduction="batchmean")
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if a.clip_grad > 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
    scaler.step(opt)
    scaler.update()
    return loss, out


@torch.no_grad()
def probe_stats(model, pruner, x, prev, dev, amp):
    """Stability of the learned selection on a fixed batch (eval mode):
    raw_std = std of the scores before the z-score, scorer_norm = L2 norm of the scorer weights,
    flip = fraction of the kept tokens that were not kept at the previous epoch (None at the first one)."""
    model.eval()
    with torch.autocast(dev, dtype=AMP_DTYPE, enabled=amp and dev == "cuda"):
        model(x)
    idx = pruner.aux["idx"]
    kept = torch.zeros(idx.shape[0], pruner.n, device=idx.device).scatter_(1, idx, 1.0)
    flip = None if prev is None else ((kept - prev).clamp(min=0).sum() / kept.sum()).item()
    norm = sum(p.detach().float().pow(2).sum() for p in pruner.parameters()).sqrt().item()
    return {"raw_std": float(pruner.aux["raw_std"]), "scorer_norm": norm, "flip": flip}, kept


def _cycle(loader):
    while True:
        yield from loader


def timing(model, teacher, tr, te, opt, scaler, a, dev, amp, warm=5, steps=20, eval_batches=10):
    """Speed only. Separates data-loading wait from GPU compute; extrapolates to a full epoch."""
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    model.train()
    it_tr = _cycle(tr)
    t_data = t_comp = 0.0
    for i in range(warm + steps):
        t = time.time()
        x, y = next(it_tr)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        sync(); t1 = time.time()
        train_step(model, teacher, x, y, opt, scaler, a, dev, amp)
        sync(); t2 = time.time()
        if i >= warm:
            t_data += t1 - t; t_comp += t2 - t1
    step = (t_data + t_comp) / steps
    model.eval()
    it_te = _cycle(te)
    t = time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=AMP_DTYPE, enabled=amp):
        for _ in range(eval_batches):
            model(next(it_te)[0].to(dev))
    sync()
    ev = (time.time() - t) / eval_batches
    ep_min = (step * len(tr) + ev * len(te)) / 60
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else float("nan")
    print(f"TIMING reducer={a.reducer} keep={a.ratio} batch={a.batch}: "
          f"{step:.2f}s/step (data {t_data / steps:.2f}s + gpu {t_comp / steps:.2f}s) = {a.batch / step:.1f} img/s | "
          f"eval {ev:.2f}s/batch | 1 epoch ~= {ep_min:.0f} min ({len(tr)} steps + {len(te)} eval batches) | "
          f"peak VRAM {mem:.1f} GB", flush=True)
    if a.timing_out:
        json.dump({"train_img_s": a.batch / step, "eval_img_s": 2 * a.batch / ev, "data_frac": t_data / steps / step,
                   "peak_GB": mem, "batch": a.batch}, open(a.timing_out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--reducer", default="identity")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--reduce_at", type=int, default=0,
                    help="block before which the single reducer runs (0 = right after patch embedding). L>0: blocks "
                         "0..L-1 see all 196 tokens, so the scorer sees contextualised tokens (DynamicViT / DyVM)")
    ap.add_argument("--backbone", default="tiny", choices=list(DIMS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone lr (DyVM: 5e-5)")
    ap.add_argument("--lr_reducer", type=float, default=1e-3)
    ap.add_argument("--lr_head", type=float, default=1e-3, help="new classifier head starts random -> needs a larger lr")
    ap.add_argument("--train_subset", type=int, default=0, help=">0: use N training images (pilot runs)")
    ap.add_argument("--test_subset", type=int, default=0, help=">0: use N test images (pilot runs)")
    ap.add_argument("--val_size", type=int, default=0,
                    help=">0: hold out N train images as validation (model selection / early stopping)")
    ap.add_argument("--patience", type=int, default=0, help=">0: early stop after N epochs without val improvement")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 = off)")
    ap.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha (0 = off)")
    ap.add_argument("--clip_grad", type=float, default=0.0, help=">0: clip gradient norm")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--wd_exclude", action="store_true",
                    help="backbone: no weight decay on 1-d params (norms, biases, D), pos_embed, cls_token, A_log "
                         "(timm / Mamba convention)")
    ap.add_argument("--bf16", action="store_true", help="autocast in bfloat16 instead of float16 (no loss scaling)")
    ap.add_argument("--kan_hidden", type=int, default=32)
    ap.add_argument("--grid_size", type=int, default=5)
    ap.add_argument("--l1", type=float, default=1e-4, help="KAN L1 regularisation")
    ap.add_argument("--prune_mode", default="st", choices=["st", "legacy"],
                    help="prune reducers: st = straight-through mask over all tokens + fused token; legacy = first version")
    ap.add_argument("--prune_temp", type=float, default=1.0, help="st: temperature of the sigmoid surrogate mask")
    ap.add_argument("--prune_norm", action="store_true",
                    help="prune: z-score the scores per image before noise / top-M / sigmoid (scale-free selection)")
    ap.add_argument("--prune_cover", type=float, default=0.0,
                    help="prune st: fraction of the kept slots given to evenly spaced tokens (Static Drop positions), "
                         "the scorer picks the rest")
    ap.add_argument("--gumbel", type=float, default=1.0,
                    help="prune: Gumbel noise scale at the start of training, annealed linearly to 0 (legacy: fixed 1)")
    ap.add_argument("--wd_reducer", type=float, default=None,
                    help="weight decay of the reducer (default: --wd). With --prune_norm the selection is invariant to "
                         "the scorer's scale, so weight decay only shrinks it and raises the effective lr: use 0")
    ap.add_argument("--freeze_reducer", type=float, default=0.0,
                    help="prune st: fraction of the last epochs with the scorer frozen (fixed selection, no noise); "
                         "the Gumbel noise then reaches 0 when the freeze starts")
    ap.add_argument("--distill", type=float, default=0.5, help="KL weight to the un-reduced teacher")
    ap.add_argument("--teacher_ckpt", default=None,
                    help="fine-tuned identity baseline on the SAME dataset (saved with --save_ckpt); no distillation if absent")
    ap.add_argument("--save_ckpt", default=None, help="save the fine-tuned (best-val if --val_size) state_dict here")
    ap.add_argument("--impl", default="pure", choices=["pure", "official"])
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (less VRAM, slower)")
    ap.add_argument("--max_steps", type=int, default=0, help=">0: stop each epoch after N steps")
    ap.add_argument("--timing", action="store_true",
                    help="only measure speed: ~25 train steps + 10 eval batches, print estimates, exit")
    ap.add_argument("--timing_out", default=None, help="with --timing: also write the measured speeds to this json")
    ap.add_argument("--no_bench", action="store_true", help="skip throughput/VRAM measurement after training")
    ap.add_argument("--no_test", action="store_true", help="with --val_size: skip the final test evaluation (LR search)")
    ap.add_argument("--resume_dir", default=None, help="save last.pt here every epoch and resume from it if present")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: pause (exit 3) before an epoch that would not finish")
    ap.add_argument("--epoch_est", type=float, default=0, help="seconds per epoch guess, used by --deadline before epoch 1")
    ap.add_argument("--tag", default="", help="suffix of the result file name")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (Colab has 2 CPUs)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if a.bf16 else torch.float16

    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    dim, depth = DIMS[a.backbone]

    va = None
    if a.smoke:
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.randn(16, 3, 224, 224), torch.randint(0, 10, (16,)))
        tr = te = DataLoader(ds, 8)
        nc = 10
    elif a.val_size > 0:
        from kanvim.data import build_loaders_val
        tr, va, te, nc = build_loaders_val(a.dataset, a.data_root, a.batch, a.workers, a.val_size,
                                           a.train_subset, a.test_subset)
        print(f"data: train {len(tr.dataset)} | val {len(va.dataset)} | test {len(te.dataset)}", flush=True)
    else:
        from kanvim.data import build_loaders
        tr, te, nc = build_loaders(a.dataset, a.data_root, a.batch, a.workers, a.train_subset, a.test_subset)

    backbone = VimPure(dim, depth, nc) if a.backbone == "stub" else load_vim(a.backbone, a.ckpt, nc, a.impl)
    if a.reducer.endswith("_ms"):  # multi-stage pruning (DyVM-style), same mean tokens as single stage
        model = ReducedVim(backbone, None, grad_ckpt=a.grad_ckpt,
                           stages=build_multistage(a.reducer, dim, a.ratio, depth, a.kan_hidden, a.grid_size,
                                                   prune_mode=a.prune_mode, prune_temp=a.prune_temp,
                                                   prune_norm=a.prune_norm)).to(dev)
    else:
        red = build_reducer(a.reducer, dim, (14, 14), a.ratio, a.kan_hidden, a.grid_size,
                            a.prune_mode, a.prune_temp, a.prune_norm, a.prune_cover)
        if a.reduce_at > 0:  # identity at the input, the reducer on the residual stream before block reduce_at
            assert a.reduce_at < depth, (a.reduce_at, depth)
            model = ReducedVim(backbone, build_reducer("identity", dim, (14, 14), 1.0), grad_ckpt=a.grad_ckpt,
                               stages=[(a.reduce_at, red)]).to(dev)
        else:
            model = ReducedVim(backbone, red, grad_ckpt=a.grad_ckpt).to(dev)
    pruners = [r for r in [model.reducer, *model.stages] if hasattr(r, "noise")]
    reducer = model.reducer
    red_params = lambda: sum(p.numel() for p in model.reducer.parameters()) + sum(p.numel() for p in model.stages.parameters())
    teacher = None
    if a.distill > 0 and a.reducer != "identity" and a.teacher_ckpt:
        tb = copy.deepcopy(backbone)
        teacher = ReducedVim(tb, build_reducer("identity", dim, (14, 14), 1.0))
        teacher.load_state_dict(torch.load(a.teacher_ckpt, map_location="cpu"))
        teacher = teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    print(f"reducer={a.reducer} tokens per block={sorted(set(model.layer_tokens()), reverse=True)}, reducer params={red_params()}, "
          f"distill={'on' if teacher else 'off'}", flush=True)

    head_ids = {id(p) for p in model.backbone.head.parameters()}
    body = [(n, p) for n, p in model.backbone.named_parameters() if id(p) not in head_ids]
    no_wd = lambda n, p: a.wd_exclude and (p.ndim <= 1 or n in ("pos_embed", "cls_token")
                                           or getattr(p, "_no_weight_decay", False))  # Mamba marks A_log, D
    groups = [{"params": [p for n, p in body if not no_wd(n, p)], "lr": a.lr}]
    if a.wd_exclude:  # default: same 3 param groups as before
        groups.append({"params": [p for n, p in body if no_wd(n, p)], "lr": a.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups + [
        {"params": list(model.backbone.head.parameters()), "lr": a.lr_head},
        {"params": list(model.reducer.parameters()) + list(model.stages.parameters()), "lr": a.lr_reducer,
         "weight_decay": a.wd if a.wd_reducer is None else a.wd_reducer}],
        weight_decay=a.wd)
    per_epoch = min(len(tr), a.max_steps) if a.max_steps else len(tr)
    steps = a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=steps,
                                                pct_start=min(0.15, 5 / max(a.epochs, 1)), anneal_strategy="cos")
    scaler = (torch.amp.GradScaler(enabled=amp and not a.bf16) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp and not a.bf16))  # torch < 2.3 (official Vim kernels need torch 2.1)

    if a.timing:
        return timing(model, teacher, tr, va if va is not None else te, opt, scaler, a, dev, amp)

    # ---- resume -------------------------------------------------------------------------------
    log, start_ep, bad = [], 0, 0
    best = {"val_top1": -1.0, "epoch": -1}
    best_state = None
    last_pt = best_pt = None
    if a.resume_dir:
        os.makedirs(a.resume_dir, exist_ok=True)
        last_pt, best_pt = os.path.join(a.resume_dir, "last.pt"), os.path.join(a.resume_dir, "best.pt")
        if os.path.exists(last_pt):
            s = torch.load(last_pt, map_location="cpu", weights_only=False)
            model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
            sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
            log, start_ep, bad, best = s["log"], s["epoch"] + 1, s["bad"], s["best"]
            random.setstate(s["rng"][0]); np.random.set_state(s["rng"][1]); torch.set_rng_state(s["rng"][2])
            print(f"resumed from epoch {start_ep} (best val {best['val_top1']:.2f} @ {best['epoch']})", flush=True)

    # ---- train --------------------------------------------------------------------------------
    freeze_ep = a.epochs
    if pruners and a.freeze_reducer > 0:
        freeze_ep = min(a.epochs - 1, max(1, round(a.epochs * (1 - a.freeze_reducer))))
    probe_x = next(iter(va if va is not None else te))[0][:64].to(dev) if pruners else None
    probe_prev = None
    stopped = "completed"
    for ep in range(start_ep, a.epochs):
        if a.patience and bad >= a.patience:
            stopped = f"early stop @ epoch {ep - 1}"
            break
        if a.deadline:
            est = log[-1]["total_min"] * 60 if log else a.epoch_est
            if time.time() + 1.1 * est > a.deadline:
                print(f"PAUSED before epoch {ep}: not enough time left. Run again to resume.", flush=True)
                sys.exit(PAUSED)
        model.train()
        if ep >= freeze_ep:  # fixed selection: only the backbone adapts to it
            for r in pruners:
                r.frozen = True
                r.requires_grad_(False)
        t0 = time.time()
        loss_sum = hit = n = 0
        for it, (x, y) in enumerate(tr):
            if it >= per_epoch:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            x, mix = mix_batch(x, y, a)
            for r in pruners:  # st: Gumbel noise a.gumbel -> 0 at freeze_ep (default: end); legacy: fixed (first version)
                r.noise = a.gumbel * max(0.0, 1 - (ep * per_epoch + it) / (freeze_ep * per_epoch)) \
                    if a.prune_mode == "st" else 1.0
            loss, out = train_step(model, teacher, x, y, opt, scaler, a, dev, amp, mix)
            sched.step()
            loss_sum += loss.item() * y.numel()
            hit += (out.argmax(1) == y).sum().item()
            n += y.numel()
        t_train = time.time() - t0
        acc, eval_loss = evaluate_full(model, va if va is not None else te, dev, amp)
        rec = {"epoch": ep, "loss": loss_sum / n, "train_loss": loss_sum / n, "train_top1": 100.0 * hit / n,
               "top1": acc, "val_loss": eval_loss, "lr": sched.get_last_lr()[0],
               "train_min": t_train / 60, "total_min": (time.time() - t0) / 60}
        if pruners:
            stats, probe_prev = probe_stats(model, pruners[0], probe_x, probe_prev, dev, amp)
            rec.update(stats)
        log.append(rec)
        if acc > best["val_top1"]:
            best, bad = {"val_top1": acc, "epoch": ep}, 0
            if va is not None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if best_pt:
                    torch.save(best_state, best_pt)
        else:
            bad += 1
        print(f"epoch {ep} train_loss {rec['train_loss']:.3f} train_top1 {rec['train_top1']:.2f} | "
              f"{'val' if va is not None else 'test'}_loss {eval_loss:.3f} top1 {acc:.2f} "
              f"(best {best['val_top1']:.2f} @ {best['epoch']}) | train {t_train / 60:.1f} min, "
              f"train+eval {rec['total_min']:.1f} min"
              + (f" | raw_std {rec['raw_std']:.3g} scorer_norm {rec['scorer_norm']:.3g} flip "
                 + ("-" if rec["flip"] is None else f"{rec['flip']:.3f}") + (" (frozen)" if ep >= freeze_ep else "")
                 if pruners else ""), flush=True)
        if last_pt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "log": log, "bad": bad, "best": best,
                        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state())}, last_pt)

    # ---- final evaluation ---------------------------------------------------------------------
    res = {"args": vars(a), "log": log, "stopped": stopped}
    if va is not None:
        if best_state is None and best_pt and os.path.exists(best_pt):
            best_state = torch.load(best_pt, map_location="cpu")
        model.load_state_dict(best_state)
        res.update(val_top1=best["val_top1"], best_epoch=best["epoch"])
        if a.no_test:
            res["top1"] = None
        else:
            res["top1"], res["test_loss"] = evaluate_full(model, te, dev, amp)
            print(f"TEST top1 {res['top1']:.2f} (model of epoch {best['epoch']}, val {best['val_top1']:.2f})", flush=True)
    else:
        res["top1"] = log[-1]["top1"]
    if a.save_ckpt:
        torch.save(model.state_dict(), a.save_ckpt)

    res.update(params_M=count_params(model) / 1e6,
               reducer_params=red_params(), mean_tokens=sum(model.layer_tokens()) / depth - 1,
               GMACs=metrics.total_macs(model.cpu(), dim, depth, num_classes=nc) / 1e9)
    model.to(dev)
    res["scan_impl"] = a.impl if a.backbone != "stub" else "stub"
    if dev == "cuda" and not a.no_bench:
        res["throughput"] = metrics.throughput(model, batch=64)
        res["vram_MB"] = metrics.peak_vram_mb(model, batch=64)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.dataset}_{a.backbone}_{a.reducer}_r{a.ratio}_s{a.seed}{'_' + a.tag if a.tag else ''}.json"
    json.dump(res, open(os.path.join(a.out, name), "w"), indent=1)
    print("saved", name, {k: v for k, v in res.items() if k not in ("log", "args")})
    for p in (last_pt, best_pt):  # finished: the json is the result, drop the resume state
        if p and os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
