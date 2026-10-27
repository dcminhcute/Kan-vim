"""TOÀN BỘ thí nghiệm của đồ án. Chạy lại lệnh này bao nhiêu lần cũng được: phần nào xong thì bỏ qua,
lần train đang dở thì chạy tiếp từ epoch cuối đã lưu.

  python scripts/run_plan.py --profile gpu --budget_hours 30          # máy thuê
  python scripts/run_plan.py --profile kaggle --stop_after_hours 11   # Kaggle (phiên tối đa 12 giờ)
  python scripts/run_plan.py --profile cpu-test                       # kiểm tra cả quy trình trên CPU, dữ liệu giả
  python scripts/run_plan.py --profile gpu --phases 1-4               # chỉ chạy một số bước

Các bước (theo thứ tự ưu tiên; hết giờ thì các bước sau chưa chạy, lần sau chạy tiếp):
  0. Đo tốc độ -> tự chọn số ảnh train / số epoch để TOÀN BỘ kế hoạch vừa --budget_hours -> plan.json
  1. Dò learning rate cho baseline (chọn theo validation)                          [CIFAR-100]
  2. Baseline Vim-Ti (100% token), lưu làm teacher                                   [CIFAR-100]
  3. 6 phương pháp ở mức giữ 50%, distillation từ baseline                            [CIFAR-100]
  4. MTR (training-free, trên baseline đã fine-tune) ở mức giữ 75/50/25%              [CIFAR-100]
  5. 6 phương pháp ở mức giữ 75% và 25%                                               [CIFAR-100]
  6. Lặp lại baseline + 6 phương pháp (50%) với seed 1 và 2 -> trung bình ± độ lệch   [CIFAR-100]
  7. Đối chứng static drop / random drop + giảm token nhiều tầng (KAN/MLP Prune multi-stage) [CIFAR-100]
  8. Dataset thứ hai: baseline + 6 phương pháp (50%) + MTR (50%)                     [Tiny-ImageNet]
  9. Ablation KAN: grid size 3/8, hidden 16/64, bỏ L1, bỏ distillation                [CIFAR-100]
 10. Báo cáo: scripts/report_plan.py -> BAO_CAO.md, ket_qua.csv, hình
Mỗi lần train: train / val (tách cố định từ tập train) / test; chọn model theo val, early stopping, test 1 lần.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PAUSED = 3

PROFILES = {
    "gpu": dict(batch=64, workers=min(16, os.cpu_count() or 4), budget_hours=30, dataset="cifar100", dataset2="tiny-imagenet", backbone="tiny"),
    "kaggle": dict(batch=32, workers=4, budget_hours=40, dataset="cifar100", dataset2="tiny-imagenet", backbone="tiny"),
    "colab": dict(batch=64, workers=min(8, os.cpu_count() or 2), budget_hours=30, dataset="cifar100",
                  dataset2="tiny-imagenet", backbone="tiny"),  # Colab Pro: A100 / L4; batch is halved if out of memory
    "cpu-test": dict(batch=16, workers=0, budget_hours=1, dataset="fake", dataset2="fake", backbone="stub"),
}
# (train pool before the val split, test size)
SIZES = {"cifar100": (50000, 10000), "cifar10": (50000, 10000), "tiny-imagenet": (100000, 10000), "food101": (75750, 25250), "fake": (1200, 200)}
METHODS = ["kan_pool", "mlp_pool", "avg_pool", "max_pool", "kan_prune", "mlp_prune"]
CONTROLS = ["static_drop", "random_drop", "kan_prune_ms", "mlp_prune_ms"]
ABLATIONS = [("g3", ["--grid_size", "3"]), ("g8", ["--grid_size", "8"]), ("h16", ["--kan_hidden", "16"]),
             ("h64", ["--kan_hidden", "64"]), ("noL1", ["--l1", "0"]), ("nodistill", ["--distill", "0"])]
ABL_METHODS = ["kan_pool", "kan_prune"]
PHASE_NAMES = {1: "dò learning rate", 2: "baseline", 3: "6 phương pháp @ mức giữ chính", 4: "MTR (training-free)",
               5: "các mức giữ khác", 6: "seed 1, 2", 7: "đối chứng + multi-stage", 8: "dataset thứ hai",
               9: "ablation KAN"}


def log(*s):
    print(time.strftime("[%H:%M:%S]"), *s, flush=True)


def result_name(dataset, backbone, reducer, ratio, seed, tag=""):
    return f"{dataset}_{backbone}_{reducer}_r{ratio}_s{seed}{'_' + tag if tag else ''}.json"


# ------------------------------------------------------------------------------------------------
# running jobs, one per GPU in parallel
# ------------------------------------------------------------------------------------------------
class Job:
    def __init__(self, name, args, out_json, script="train.py"):
        self.name, self.args, self.out_json, self.script = name, args, out_json, script


def _pump(proc, prefix, logf):
    with open(logf, "a", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line); f.flush()
            line = line.rstrip()
            if any(k in line for k in ("epoch", "TEST", "saved", "PAUSED", "Error", "error", "resumed", "data:",
                                       "TIMING", "Traceback", "MTR")):
                print(f"   [{prefix}] {line}", flush=True)


def run_jobs(jobs, gpus, logdir, stop_flag):
    """-> dict name -> 'done' | 'paused' | 'failed' | 'not started'"""
    os.makedirs(logdir, exist_ok=True)
    status = {j.name: "done" for j in jobs if os.path.exists(j.out_json)}
    pending, running = [j for j in jobs if j.name not in status], {}
    while pending or running:
        for g in gpus:
            if g in running or not pending or stop_flag["stop"]:
                continue
            j = pending.pop(0)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            log(f"BẮT ĐẦU {j.name}" + (f" (GPU {g})" if g is not None else ""))
            p = subprocess.Popen([PY, os.path.join(ROOT, *j.script.split("/"))] + j.args, cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace")
            t = threading.Thread(target=_pump, args=(p, j.name, os.path.join(logdir, j.name + ".log")), daemon=True)
            t.start()
            running[g] = (j, p, t, time.time())
        time.sleep(2)
        for g in list(running):
            j, p, t, t0 = running[g]
            if p.poll() is None:
                continue
            t.join()
            del running[g]
            mins = (time.time() - t0) / 60
            if p.returncode == 0 and os.path.exists(j.out_json):
                status[j.name] = "done"; log(f"XONG {j.name} ({mins:.0f} phút)")
            elif p.returncode == PAUSED:
                status[j.name] = "paused"; stop_flag["stop"] = True
                log(f"TẠM DỪNG {j.name} (sắp hết giờ) - chạy lại lệnh để tiếp tục")
            else:
                status[j.name] = "failed"
                log(f"LỖI {j.name} (exit {p.returncode}), xem {os.path.join(logdir, j.name + '.log')}")
    for j in pending:
        status.setdefault(j.name, "not started")
    return status


# ------------------------------------------------------------------------------------------------
# step 0: speed measurement and sizing
# ------------------------------------------------------------------------------------------------
def pick_impl(a):
    if a.impl != "auto":
        return a.impl
    if a.backbone == "stub":
        return "pure"
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "check_impl.py"), a.ckpt], cwd=ROOT,
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return "official" if r.returncode == 0 and "USE_OFFICIAL" in r.stdout else "pure"


def common(a, dataset=None):
    c = ["--dataset", dataset or a.dataset, "--data_root", a.data_root, "--backbone", a.backbone,
         "--workers", str(a.workers)]
    if a.backbone != "stub":
        c += ["--ckpt", a.ckpt]
    return c + a.extra.split()


def measure(a, impl, reducer, ratio, batch, grad_ckpt, gpu):
    out = os.path.join(a.out, "timing", f"{reducer}_{batch}{'_gc' if grad_ckpt else ''}.json")
    if os.path.exists(out):
        return json.load(open(out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [PY, os.path.join(ROOT, "train.py"), "--timing", "--timing_out", out, "--reducer", reducer,
           "--ratio", str(ratio), "--batch", str(batch), "--impl", impl,
           "--val_size", "200" if a.dataset == "fake" else "2000"] + common(a)
    if grad_ckpt:
        cmd.append("--grad_ckpt")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n".join("   " + l for l in r.stdout.splitlines() if "TIMING" in l or "data:" in l), flush=True)
    if r.returncode or not os.path.exists(out):
        print("   (lỗi khi đo)", r.stderr[-1200:], flush=True)
        return None
    return json.load(open(out))


def sizes_for(a, n, E):
    """all sizes derived from (n train images on dataset 1, E epochs)"""
    pool, n_test = SIZES[a.dataset]
    pool2, n_test2 = SIZES[a.dataset2]
    if a.dataset == "fake":
        n_val, n_val2 = 200, 200
    else:
        n_val = 5000 if n >= 20000 else 2000
        n_val2 = min(2 * n_val, 10000)
    n = min(n, pool - n_val)
    n2 = min(n * pool2 // pool, pool2 - n_val2)  # same images per class on dataset 2
    return dict(n_train=n, epochs=E, val_size=n_val, n_test=n_test, n_train2=n2, val_size2=n_val2, n_test2=n_test2,
                search_n=min(n, 10000), search_epochs=max(2 if a.dataset == "fake" else 4, E // 5))


def phase_costs(a, sp, z, G):
    """hours per phase (wall clock with G GPUs in parallel)"""
    vi, ei, vr, er = sp["identity"]["train_img_s"], sp["identity"]["eval_img_s"], sp["red"]["train_img_s"], sp["red"]["eval_img_s"]
    bench = 120 if sp.get("gpu") else 0
    E = z["epochs"]
    ident = lambda n, nv, nt, ep=E: ep * (n / vi + nv / ei) + nt / ei + bench
    meth = lambda n, nv, nt: E * (n * (1 / vr + 1 / ei) + nv / er) + nt / er + bench  # + teacher forward
    waves = lambda k: math.ceil(k / G)
    n, nv, nt = z["n_train"], z["val_size"], z["n_test"]
    n2, nv2, nt2 = z["n_train2"], z["val_size2"], z["n_test2"]
    nm, nr = len(a.methods), len(a.other_ratios)
    mtr = lambda k, nv_, nt_: k * 1.5 * (nv_ + nt_) / ei + k * bench
    c = {1: waves(len(a.lrs)) * ident(z["search_n"], nv, 0, z["search_epochs"]),
         2: ident(n, nv, nt),
         3: waves(nm) * meth(n, nv, nt),
         4: mtr(1 + nr, nv, nt),
         5: waves(nm * nr) * meth(n, nv, nt),
         6: waves(len(a.extra_seeds) * (nm + 1)) * meth(n, nv, nt),
         7: waves(len(CONTROLS)) * meth(n, nv, nt),
         8: ident(n2, nv2, nt2) + waves(nm) * meth(n2, nv2, nt2) + mtr(1, nv2, nt2),
         9: waves(len(ABLATIONS) * len(ABL_METHODS)) * meth(n, nv, nt)}
    return {k: 1.1 * v / 3600 for k, v in c.items() if k in a.phase_set}


def make_plan(a, gpus):
    log("BƯỚC 0: chọn cài đặt, đo tốc độ")
    impl = pick_impl(a)
    log(f"backbone implementation: {impl}")
    g0 = gpus[0]
    batch, gc, sp = a.batch, False, {}
    for b, g in [(a.batch, False), (a.batch // 2, False), (a.batch // 2, True), (a.batch // 4, True)]:
        if b < 4:
            continue
        s1 = measure(a, impl, "identity", 1.0, b, g, g0)
        s2 = s1 and measure(a, impl, "kan_prune", a.ratio, b, g, g0)
        if s1 and s2:
            batch, gc, sp = b, g, {"identity": s1, "red": s2}
            break
        log(f"batch {b}{' + grad_ckpt' if g else ''} lỗi (có thể thiếu VRAM), thử nhỏ hơn")
    if not sp:
        sys.exit("Không đo được tốc độ: xem lỗi ở trên.")
    sp["gpu"] = g0 is not None

    if a.train_subset or a.epochs:
        cands = [(a.train_subset or 45000, a.epochs or 30)]
    elif a.dataset == "fake":
        cands = [(800, 3)]
    else:
        cands = [(45000, 30), (30000, 30), (20000, 30), (20000, 20), (10000, 20), (5000, 20), (5000, 15), (2500, 15)]
    for n, E in cands:
        z = sizes_for(a, n, E)
        costs = phase_costs(a, sp, z, len(gpus))
        h = sum(costs.values())
        log(f"  {z['n_train']} ảnh train, {E} epoch, val {z['val_size']}: toàn bộ kế hoạch ~{h:.1f} giờ")
        if h <= a.budget_hours:
            break
    else:
        log(f"CẢNH BÁO: cấu hình nhỏ nhất vẫn cần ~{h:.0f} giờ > ngân sách {a.budget_hours} giờ. Vẫn dùng cấu hình "
            f"nhỏ nhất; các bước chạy theo thứ tự ưu tiên, bước nào chưa xong thì chạy lại lệnh để tiếp tục.")
    pool = SIZES[a.dataset][0]
    plan = dict(impl=impl, batch=batch, grad_ckpt=gc, speeds=sp, lrs=a.lrs, gpus=len(gpus), created=time.ctime(),
                phase_hours=costs, est_hours=h, **z)
    plan["train_subset"] = 0 if z["n_train"] >= pool - z["val_size"] else z["n_train"]
    plan["train_subset2"] = 0 if z["n_train2"] >= SIZES[a.dataset2][0] - z["val_size2"] else z["n_train2"]
    vi, ei, vr, er = sp["identity"]["train_img_s"], sp["identity"]["eval_img_s"], sp["red"]["train_img_s"], sp["red"]["eval_img_s"]
    plan["epoch_s_identity"] = z["n_train"] / vi + z["val_size"] / ei
    plan["epoch_s_method"] = z["n_train"] * (1 / vr + 1 / ei) + z["val_size"] / er
    log(f"KẾ HOẠCH: {z['n_train']} ảnh train ({a.dataset}), {z['n_train2']} ảnh ({a.dataset2}), {E} epoch, "
        f"batch {batch}{' + grad_ckpt' if gc else ''}, ước tính tối đa {h:.1f} giờ (early stopping thường ngắn hơn)")
    for k, v in costs.items():
        log(f"   bước {k} ({PHASE_NAMES[k]}): ~{v:.1f} giờ")
    return plan


# ------------------------------------------------------------------------------------------------
def parse_phases(s):
    out = set()
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default="gpu")
    ap.add_argument("--dataset", default=None, help="dataset chính (mặc định cifar100)")
    ap.add_argument("--dataset2", default=None, help="dataset thứ hai, bước 8 (mặc định tiny-imagenet)")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "vim_t.pth"))
    ap.add_argument("--impl", default="auto", choices=["auto", "pure", "official"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results_plan"))
    ap.add_argument("--gpus", default=None, help="vd '0,1'. Mặc định: tất cả GPU thấy được")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget_hours", type=float, default=None, help="tổng thời gian muốn bỏ ra cho TOÀN BỘ kế hoạch")
    ap.add_argument("--stop_after_hours", type=float, default=0, help=">0: tự tạm dừng an toàn sau N giờ (Kaggle: 11)")
    ap.add_argument("--phases", default="1-9", help="vd '1-4' hoặc '1-3,8'")
    ap.add_argument("--train_subset", type=int, default=0, help="ép số ảnh train (bỏ qua tự chọn)")
    ap.add_argument("--epochs", type=int, default=0, help="ép số epoch (bỏ qua tự chọn)")
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-5, 5e-5, 1e-4])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--ratio", type=float, default=0.5, help="mức giữ chính")
    ap.add_argument("--other_ratios", type=float, nargs="+", default=[0.75, 0.25])
    ap.add_argument("--extra_seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr2", type=float, default=0, help=">0: lr backbone riêng cho dataset thứ hai (bước 8)")
    ap.add_argument("--epochs2", type=int, default=0, help=">0: số epoch riêng cho dataset thứ hai (bước 8)")
    ap.add_argument("--extra", default="", help="tham số thêm cho train.py")
    ap.add_argument("--replan", action="store_true", help="đo và chọn lại cấu hình (xoá plan.json)")
    a = ap.parse_args()
    for k, v in PROFILES[a.profile].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.phase_set = parse_phases(a.phases)
    if a.profile == "cpu-test":
        a.extra = (a.extra + " --max_steps 3").strip()
        a.lrs, a.patience = a.lrs[:2], 2
    t_start = time.time()
    deadline = t_start + a.stop_after_hours * 3600 if a.stop_after_hours else 0
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, ROOT)

    import torch
    if a.gpus:
        gpus = [int(x) for x in a.gpus.split(",")]
    elif torch.cuda.is_available():
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [None]
    log(f"profile={a.profile} dataset={a.dataset} + {a.dataset2} GPU={gpus if gpus != [None] else 'không có (CPU)'} "
        f"bước={sorted(a.phase_set)} out={a.out}")
    if gpus == [None] and a.profile != "cpu-test":
        sys.exit("Không thấy GPU. Bật GPU (Kaggle: Settings -> Accelerator -> GPU T4 x2) rồi chạy lại.")
    if a.backbone != "stub" and not os.path.exists(a.ckpt):
        log("tải checkpoint Vim-Ti từ Hugging Face")
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download("hustvl/Vim-tiny-midclstok", "vim_t_midclstok_76p1acc.pth"), a.ckpt)
    from kanvim.data import build_datasets

    def download(ds):  # once here, never in parallel jobs
        if ds != "fake":
            log("kiểm tra / tải dữ liệu " + ds)
            build_datasets(ds, a.data_root)
    download(a.dataset)

    plan_f = os.path.join(a.out, "plan.json")
    if a.replan and os.path.exists(plan_f):
        os.remove(plan_f)
    if os.path.exists(plan_f):
        plan = json.load(open(plan_f))
        log(f"dùng lại kế hoạch đã có: {plan['n_train']} ảnh train, {plan['epochs']} epoch, batch {plan['batch']}, "
            f"impl {plan['impl']}")
    else:
        plan = make_plan(a, gpus)
        json.dump(plan, open(plan_f, "w"), indent=1)

    logs, state = os.path.join(a.out, "logs"), os.path.join(a.out, "state")
    stop = {"stop": False}
    summary = {}

    def targs(dataset, val_size, epoch_est):
        c = common(a, dataset) + ["--impl", plan["impl"], "--batch", str(plan["batch"]), "--val_size", str(val_size),
                                  "--mixup", str(a.mixup), "--cutmix", str(a.cutmix), "--clip_grad", "1.0"]
        if plan["grad_ckpt"]:
            c.append("--grad_ckpt")
        if deadline:
            c += ["--deadline", str(deadline), "--epoch_est", str(int(epoch_est))]
        return c

    def job(folder, reducer, ratio, seed, lr, teacher=None, tag="", extra=(), second=False, save=None):
        ds = a.dataset2 if second else a.dataset
        if second and a.lr2:
            lr = a.lr2
        epochs = a.epochs2 if second and a.epochs2 else plan["epochs"]
        scale = (plan["n_train2"] / plan["n_train"]) if second else 1.0
        est = (plan["epoch_s_identity"] if reducer == "identity" else plan["epoch_s_method"]) * scale
        sub = plan["train_subset2"] if second else plan["train_subset"]
        name = f"{'d2_' if second else ''}{reducer}_r{ratio}_s{seed}{'_' + tag if tag else ''}"
        args = targs(ds, plan["val_size2"] if second else plan["val_size"], est) + [
            "--reducer", reducer, "--ratio", str(ratio), "--seed", str(seed), "--lr", str(lr),
            "--epochs", str(epochs), "--patience", str(a.patience), "--out", os.path.join(a.out, folder),
            "--resume_dir", os.path.join(state, name)] + (["--train_subset", str(sub)] if sub else []) + list(extra)
        if tag:
            args += ["--tag", tag]
        if teacher:
            args += ["--teacher_ckpt", teacher]
        if save:
            args += ["--save_ckpt", save]
        return Job(name, args, os.path.join(a.out, folder, result_name(ds, a.backbone, reducer, ratio, seed, tag)))

    def mtr_job(ratios, teacher, second=False):
        ds = a.dataset2 if second else a.dataset
        args = ["--dataset", ds, "--data_root", a.data_root, "--backbone", a.backbone, "--teacher_ckpt", teacher,
                "--ratios", *map(str, ratios), "--val_size", str(plan["val_size2"] if second else plan["val_size"]),
                "--batch", str(plan["batch"]), "--workers", str(a.workers), "--out", os.path.join(a.out, "mtr")]
        last = os.path.join(a.out, "mtr", result_name(ds, a.backbone, "mtr", ratios[-1], 0))
        return Job(f"{'d2_' if second else ''}mtr", args, last, script="scripts/eval_mtr.py")

    def phase(k, jobs):
        if k not in a.phase_set:
            return True
        if stop["stop"]:
            summary[k] = "chưa chạy (đã tạm dừng)"
            return False
        todo = [j for j in jobs if not os.path.exists(j.out_json)]
        log(f"BƯỚC {k}: {PHASE_NAMES[k]} - {len(jobs) - len(todo)}/{len(jobs)} đã xong")
        st = run_jobs(jobs, gpus, logs, stop)
        bad = {n: s for n, s in st.items() if s != "done"}
        summary[k] = "xong" if not bad else ", ".join(f"{n}: {s}" for n, s in bad.items())
        return not bad

    # ---- 1. LR search ------------------------------------------------------------------------
    best_f = os.path.join(a.out, "best_lr.json")
    if os.path.exists(best_f):
        best_lr = json.load(open(best_f))["lr"]
    elif len(plan["lrs"]) == 1:
        best_lr = plan["lrs"][0]
    else:
        sj = []
        for lr in plan["lrs"]:
            tag = f"lr{lr:g}"
            est = plan["epoch_s_identity"] * plan["search_n"] / plan["n_train"]
            sj.append(Job(f"search_{tag}", targs(a.dataset, plan["val_size"], est) + [
                "--reducer", "identity", "--ratio", "1.0", "--seed", "0", "--lr", str(lr),
                "--epochs", str(plan["search_epochs"]), "--train_subset", str(plan["search_n"]), "--no_test",
                "--no_bench", "--tag", tag, "--out", os.path.join(a.out, "search"),
                "--resume_dir", os.path.join(state, f"search_{tag}")],
                os.path.join(a.out, "search", result_name(a.dataset, a.backbone, "identity", 1.0, 0, tag))))
        a.phase_set.add(1)
        if not phase(1, sj):
            return finish(a, summary, t_start)
        scores = {lr: json.load(open(j.out_json))["val_top1"] for lr, j in zip(plan["lrs"], sj)}
        best_lr = max(scores, key=scores.get)
        json.dump({"lr": best_lr, "val_top1": scores}, open(best_f, "w"), indent=1)
        log("kết quả dò LR (val top-1): " + ", ".join(f"{k:g}: {v:.2f}" for k, v in scores.items()) + f" -> chọn {best_lr:g}")

    # ---- 2. baseline (teacher) ----------------------------------------------------------------
    teacher = os.path.join(a.out, "teacher_identity.pt")
    base = job("main", "identity", 1.0, 0, best_lr, save=teacher)
    if not os.path.exists(teacher) and os.path.exists(base.out_json):
        os.remove(base.out_json)  # json without the teacher weights: redo
    need_teacher = a.phase_set & {2, 3, 4, 5, 6, 7, 9}
    if need_teacher:
        a.phase_set.add(2)
        if not phase(2, [base]):
            return finish(a, summary, t_start)

    # ---- 3..7, 9 on dataset 1; 8 on dataset 2 -------------------------------------------------
    R, S = a.ratio, a.extra_seeds
    phase(3, [job("main", m, R, 0, best_lr, teacher) for m in a.methods])
    phase(4, [mtr_job(sorted({R, *a.other_ratios}, reverse=True), teacher)])
    phase(5, [job("main", m, q, 0, best_lr, teacher) for q in a.other_ratios for m in a.methods])
    phase(6, [job("main", "identity", 1.0, s, best_lr) for s in S]
          + [job("main", m, R, s, best_lr, teacher) for s in S for m in a.methods])
    phase(7, [job("main", m, R, 0, best_lr, teacher) for m in CONTROLS])
    if 8 in a.phase_set and not stop["stop"]:
        download(a.dataset2)
        teacher2 = os.path.join(a.out, "teacher_identity_d2.pt")
        b2 = job("second", "identity", 1.0, 0, best_lr, second=True, save=teacher2)
        if not os.path.exists(teacher2) and os.path.exists(b2.out_json):
            os.remove(b2.out_json)
        if phase(8, [b2]):
            phase(8, [job("second", m, R, 0, best_lr, teacher2, second=True) for m in a.methods]
                  + [mtr_job([R], teacher2, second=True)])
    phase(9, [job("ablation", m, R, 0, best_lr, teacher, tag=t, extra=x) for t, x in ABLATIONS for m in ABL_METHODS])
    finish(a, summary, t_start)


def finish(a, summary, t_start):
    report(a)
    log("TÓM TẮT:")
    for k in sorted(summary):
        log(f"   bước {k} ({PHASE_NAMES[k]}): {summary[k]}")
    if all(v == "xong" for v in summary.values()):
        log(f"XONG TOÀN BỘ. Thời gian lần chạy này: {(time.time() - t_start) / 3600:.1f} giờ")
    elif any("paused" in v or "chưa chạy" in v or "not started" in v for v in summary.values()):
        log("ĐÃ TẠM DỪNG. Chạy lại đúng lệnh này (Kaggle: xem HUONG_DAN_CHAY.md) để chạy tiếp.")
    else:
        log("Có lần chạy lỗi (xem TÓM TẮT và logs/). Sửa lỗi rồi chạy lại lệnh: các phần đã xong được giữ.")
        sys.exit(1)


def report(a):
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "report_plan.py"), a.out], cwd=ROOT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode:
        return
    import zipfile
    z = a.out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:  # small: no weights / resume states
        for d, _, files in os.walk(a.out):
            for f in files:
                if not f.endswith(".pt"):
                    p = os.path.join(d, f)
                    zf.write(p, os.path.relpath(p, os.path.dirname(a.out)))
    log(f"báo cáo: {os.path.join(a.out, 'BAO_CAO.md')} | file nén để tải về: {z}")


if __name__ == "__main__":
    main()
