"""Chạy lại nhánh Prune đã sửa (--prune_mode st) trên CIFAR-100, cùng cấu hình với results_plan
(45000 ảnh train, val 5000, 30 epoch, patience 6, lr 1e-4, mixup 0.2, clip 1.0, teacher = baseline seed 0).

  bash scripts/run_fix.sh          # chạy nền, log: run_fix.log
  python scripts/run_fix.py        # chạy trực tiếp

Thứ tự ưu tiên (chạy song song 1 job / GPU, chạy lại lệnh thì bỏ qua job đã xong, job dở chạy tiếp):
  A. KAN Prune, MLP Prune, giữ 50%, seed 0/1/2                      6 job
  B. kiểm tra learning rate (seed 0, giữ 50%): lr bộ chấm điểm 1e-4, lr backbone 5e-5   4 job
  C. KAN Prune, MLP Prune, giữ 75% và 25%, seed 0                     4 job
Kết quả: results_fix/*.json, bảng so sánh results_fix/KET_QUA_FIX.md, log từng job: results_fix/logs/.

Đợt so sánh lr (--lr_round, dùng thư mục --out riêng): lr backbone 5e-5, giữ 50%
  KANPool, Static Drop, MLP Pool, Random Drop seed 0 (chạy trước), rồi KAN/MLP Prune seed 0/1/2   10 job
  thêm seed 1/2 cho KANPool, MLP Pool, Static Drop (để mọi dòng của bảng đều 3 seed)                6 job
  Bảng: <out>/KET_QUA_LR.md

Đợt sửa độ ổn định của Prune (--stab_round, tự bật --prune_norm, lr backbone 5e-5, giữ 50%):
  KAN/MLP Prune x {wd0: bộ chấm điểm không weight decay, g03: nhiễu Gumbel 0.3,
                   frz: đóng băng bộ chấm điểm 1/3 số epoch cuối}
  seed 1 trước (KAN seed 1 sụt nặng nhất: val 82,92 -> 74,06), rồi seed 0                        12 job
  Bảng: <out>/KET_QUA_STAB.md (mốc so sánh: cùng seed, lr 5e-5 của results_lr)

Đợt giảm token sau khối L (--ctx_round, mặc định L = --reduce_at 6, tự bật --prune_norm, lr backbone 5e-5, giữ 50%):
  bộ chấm điểm thấy token đã qua L khối Mamba (có ngữ cảnh) thay vì patch embedding thô.
  KAN/MLP Prune, KANPool/MLP Pool, Static Drop x seed 0/1/2 (seed 0 của cả 5 phương pháp chạy trước)   15 job
  Checkpoint seed 0 của 4 phương pháp có học: <out>_ckpt/ (để vẽ hình diễn giải)
  Bảng: <out>/KET_QUA_CTX.md (mốc so sánh: giảm token ở đầu vào, lr 5e-5 của results_lr)

Đợt giữ 25% sau khối L (--r25_round, cùng cấu hình --ctx_round, chỉ đổi tỉ lệ giữ 0.5 -> 0.25):
  có học có thắng chọn đều khi bỏ 3/4 số token không. KANPool, MLP Pool, Static Drop, Avg Pool x seed 0/1/2   12 job
  rồi KAN/MLP Prune + phủ đều (--prune_cover 0.5: nửa số slot ở vị trí Static Drop, nửa kia bộ chấm điểm chọn)  6 job
  Bảng: <out>/KET_QUA_R25.md (KPI đặt trước: có học hơn Static Drop ở cả 3 seed và trung bình > 2 x độ lệch)

Đợt 6 (--dot6): đủ ma trận đề cương. Giảm token ở đầu vào, lr backbone 5e-5, --prune_norm, patience 0 (tự bật).
  Pha 1. CIFAR-100, 30 epoch: baseline 100% token x 3 seed, rồi KANPool, MLP Pool, Avg Pool, Max Pool, KAN Prune,
         MLP Prune x giữ 75 / 50 / 25% x 3 seed (seed 0 của mọi mức chạy trước); Static Drop 75% / 25% chạy cuối.
         16 job giữ 50% đã có (results_lr, cùng cấu hình) nằm ở prev_results/lr5e-5/, được chép vào --out: không chạy lại.
         Cùng pha: baseline Tiny-ImageNet và ImageNet-100 (seed 0, --ext_epochs epoch), lưu làm teacher.
  Pha 2. Tiny-ImageNet, ImageNet-100 rút gọn: 6 phương pháp, giữ 50%, seed 0, chưng cất từ teacher cùng dataset.
         Thiếu data/imagenet-100 (scripts/prep_imagenet100.py) thì bỏ qua phần ImageNet-100.
  Bảng: <out>/KET_QUA_DOT6.md, hình: <out>/fig_dot6_cifar100.png
"""
import argparse
import glob
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable

# kết quả bản cũ (prune_mode legacy, results_plan) để so sánh
OLD = {("kan_prune", 0.5): [73.88, 75.71, 77.00], ("mlp_prune", 0.5): [83.08, 74.40, 76.95],
       ("kan_prune", 0.75): [84.00], ("mlp_prune", 0.75): [85.05],
       ("kan_prune", 0.25): [61.85], ("mlp_prune", 0.25): [66.51]}
REF = [("Vim-Ti (100% token)", "87,85 ± 0,46 (3 seed)"), ("KANPool", "84,27 ± 0,16 (3 seed)"),
       ("MLP Pool", "84,24 ± 0,14 (3 seed)"), ("Static Drop", "84,48 (seed 0)"), ("Random Drop", "81,74 (seed 0)")]
NAMES = {"kan_prune": "KAN Prune", "mlp_prune": "MLP Prune"}
# --lr_round: kết quả lr backbone 1e-4 (seed 0, 1, 2) để so sánh. Pool/Drop: results_plan; Prune: bản sửa (results_fix_norm)
LR1E4 = {"kan_prune": [78.78, 79.55, 78.28], "mlp_prune": [79.31, 79.41, 78.61], "kan_pool": [84.43, 84.11, 84.27],
         "mlp_pool": [84.14, 84.19, 84.40], "static_drop": [84.48], "random_drop": [81.74]}
LR_NAMES = {**NAMES, "kan_pool": "KANPool", "mlp_pool": "MLP Pool", "static_drop": "Static Drop",
            "random_drop": "Random Drop"}


def jobs():
    out = []
    for s in (0, 1, 2):
        for r in ("kan_prune", "mlp_prune"):
            out.append((r, 0.5, s, "", []))
    for r in ("kan_prune", "mlp_prune"):
        out.append((r, 0.5, 0, "lrred1e-4", ["--lr_reducer", "1e-4"]))
        out.append((r, 0.5, 0, "lr5e-5", ["--lr", "5e-5"]))
    for ratio in (0.75, 0.25):
        for r in ("kan_prune", "mlp_prune"):
            out.append((r, ratio, 0, "", []))
    return out


def jobs_lr():
    # Pool/Drop trước: sớm biết lr thấp có giúp mọi phương pháp không. Prune seed 0 đã có ở results_fix_norm
    out = [(r, 0.5, 0, "lr5e-5", ["--lr", "5e-5"]) for r in ("kan_pool", "static_drop", "mlp_pool", "random_drop")]
    for s in (0, 1, 2):
        for r in NAMES:
            out.append((r, 0.5, s, "lr5e-5", ["--lr", "5e-5"]))
    for s in (1, 2):
        for r in ("kan_pool", "mlp_pool", "static_drop"):
            out.append((r, 0.5, s, "lr5e-5", ["--lr", "5e-5"]))
    return out


# --stab_round: mốc lr 5e-5 chưa sửa (results_lr): (test, val tốt nhất, epoch tốt nhất, val epoch cuối)
STAB_BASE = {("kan_prune", 1): (82.37, 82.92, 17, 74.06), ("mlp_prune", 1): (79.10, 78.90, 19, 78.14),
             ("kan_prune", 0): (83.81, 84.46, 26, 84.40), ("mlp_prune", 0): (81.65, 81.66, 18, 78.80)}
STAB_SEEDS = (1, 0)
STAB = [("wd0", "không weight decay cho bộ chấm điểm", ["--wd_reducer", "0"]),
        ("g03", "nhiễu Gumbel 0.3", ["--gumbel", "0.3"]),
        ("frz", "đóng băng bộ chấm điểm 1/3 cuối", ["--freeze_reducer", "0.33"])]


def jobs_stab():
    return [(r, 0.5, s, f"lr5e-5_{t}", ["--lr", "5e-5"] + extra)
            for s in STAB_SEEDS for t, _, extra in STAB for r in NAMES]


# --ctx_round: mốc = giảm token ngay sau patch embedding, lr 5e-5, giữ 50% (results_lr), seed 0/1/2
CTX_BASE = {"kan_prune": [83.81, 82.37, 82.72], "mlp_prune": [81.65, 79.10, 81.13], "kan_pool": [84.74, 84.93, 84.98],
            "mlp_pool": [84.82, 85.60, 85.29], "static_drop": [85.68, 85.46, 85.77]}
CTX_PAIRS = [("kan_prune", "mlp_prune"), ("kan_pool", "mlp_pool")]


def ctx_tag(a):
    return f"L{a.reduce_at}_lr5e-5"


def jobs_ctx(a):
    os.makedirs(a.out + "_ckpt", exist_ok=True)
    out = []
    for s in (0, 1, 2):
        for r in CTX_BASE:
            extra = ["--lr", "5e-5", "--reduce_at", str(a.reduce_at)]
            if s == 0 and r != "static_drop":
                name = os.path.splitext(result_name(r, 0.5, s, ctx_tag(a)))[0]
                extra += ["--save_ckpt", os.path.join(a.out + "_ckpt", name + ".pt")]
            out.append((r, 0.5, s, ctx_tag(a), extra))
    return out


# --r25_round: giữ 25% sau khối L. Mốc giữ 50% = results_ctx (cùng vị trí, cùng cấu hình), seed 0/1/2
# phương án: (reducer, đuôi tag, cờ thêm). "cov": một nửa số slot là vị trí Static Drop, nửa kia bộ chấm điểm chọn
R25_VARIANTS = {"kan_pool": ("kan_pool", "", []), "mlp_pool": ("mlp_pool", "", []),
                "static_drop": ("static_drop", "", []), "avg_pool": ("avg_pool", "", []),
                "kan_prune_cov": ("kan_prune", "_cov05", ["--prune_cover", "0.5"]),
                "mlp_prune_cov": ("mlp_prune", "_cov05", ["--prune_cover", "0.5"])}
R25_BASE = {"kan_pool": [87.37, 87.46, 87.30], "mlp_pool": [87.26, 87.35, 87.67],
            "static_drop": [87.41, 86.66, 87.27], "avg_pool": [], "kan_prune_cov": [], "mlp_prune_cov": []}
R25_NAMES = {**LR_NAMES, "avg_pool": "Avg Pool", "kan_prune_cov": "KAN Prune + phủ đều",
             "mlp_prune_cov": "MLP Prune + phủ đều"}
R25_PAIRS = [("kan_pool", "static_drop"), ("mlp_pool", "static_drop"), ("kan_prune_cov", "static_drop"),
             ("mlp_prune_cov", "static_drop"), ("kan_pool", "avg_pool"), ("kan_pool", "mlp_pool"),
             ("kan_prune_cov", "mlp_prune_cov")]


def jobs_r25(a):
    extra = ["--lr", "5e-5", "--reduce_at", str(a.reduce_at)]
    # 4 phương pháp gốc trước (đã giao cho Minh), 2 phương án phủ đều thêm vào cuối
    order = list(R25_VARIANTS)[:4], list(R25_VARIANTS)[4:]
    return [(R25_VARIANTS[v][0], 0.25, s, ctx_tag(a) + R25_VARIANTS[v][1], extra + R25_VARIANTS[v][2])
            for group in order for s in (0, 1, 2) for v in group]


# --dot6: đủ ma trận đề cương (giảm ở đầu vào, lr 5e-5). Job = (reducer, ratio, seed, tag, cờ thêm[, dataset])
DOT6_TAG = "lr5e-5"
DOT6_METHODS = ["kan_pool", "mlp_pool", "avg_pool", "max_pool", "kan_prune", "mlp_prune"]
DOT6_RATIOS = (0.75, 0.5, 0.25)
DOT6_NAMES = {**LR_NAMES, "identity": "Vim-Ti (100% token)", "avg_pool": "Avg Pool", "max_pool": "Max Pool"}
DOT6_PREV = os.path.join(ROOT, "prev_results", "lr5e-5")  # 16 job giữ 50% của results_lr, cùng cấu hình
DOT6_EXT = {"tiny-imagenet": "Tiny-ImageNet", "imagenet-100": "ImageNet-100"}
# (a, b, tên): hiệu số a − b ghép cặp theo seed ở từng mức giữ
DOT6_PAIRS = [("kan_pool", "kan_prune", "Gộp − loại bỏ (KAN)"), ("mlp_pool", "mlp_prune", "Gộp − loại bỏ (MLP)"),
              ("kan_pool", "mlp_pool", "KAN − MLP (gộp)"), ("kan_prune", "mlp_prune", "KAN − MLP (loại bỏ)"),
              ("kan_pool", "avg_pool", "KANPool − Avg Pool"), ("kan_pool", "max_pool", "KANPool − Max Pool"),
              ("kan_pool", "static_drop", "KANPool − Static Drop")]


def ext_teacher(a, ds):
    return os.path.join(a.out, f"teacher_{ds}.pt")


def jobs_dot6(a):
    """-> [pha 1, pha 2]. Pha 2 cần teacher do pha 1 lưu, nên chỉ bắt đầu khi pha 1 xong."""
    lr = ["--lr", "5e-5"]
    ext = lr + ["--epochs", str(a.ext_epochs)]
    cifar = [("identity", 1.0, s, DOT6_TAG, lr) for s in (0, 1, 2)]
    cifar += [(r, ratio, s, DOT6_TAG, lr) for s in (0, 1, 2) for ratio in DOT6_RATIOS for r in DOT6_METHODS]
    teachers = [("identity", 1.0, 0, DOT6_TAG, ext + ["--save_ckpt", ext_teacher(a, ds)], ds) for ds in a.ext_ds]
    static = [("static_drop", ratio, s, DOT6_TAG, lr) for s in (0, 1, 2) for ratio in (0.75, 0.25)]
    rest = [(r, 0.5, 0, DOT6_TAG, ext + ["--teacher_ckpt", ext_teacher(a, ds)], ds)
            for ds in a.ext_ds for r in DOT6_METHODS]
    if a.quick:  # chạy thử: đủ mọi loại job (baseline, gộp, loại bỏ, teacher, mỗi dataset) nhưng ít job
        keep = {("identity", 1.0, 0), ("kan_pool", 0.25, 0), ("max_pool", 0.75, 0), ("kan_prune", 0.25, 0)}
        return [[j for j in cifar if j[:3] in keep] + teachers, [j for j in rest if j[0] in ("kan_pool", "kan_prune")]]
    # thứ tự pha 1: baseline + seed 0 mọi mức (đủ một đường cong sớm), teacher, seed 1/2, Static Drop
    return [cifar[:21] + teachers + cifar[21:] + static, rest]


def result_name(r, ratio, seed, tag, dataset="cifar100"):
    return f"{dataset}_tiny_{r}_r{ratio}_s{seed}{'_' + tag if tag else ''}.json"


def done(a, j):
    if not os.path.exists(os.path.join(a.out, result_name(*j[:4], *j[5:]))):
        return False
    extra = j[4]  # teacher của --dot6: phải có cả file checkpoint
    return not (a.dot6 and "--save_ckpt" in extra and not os.path.exists(extra[extra.index("--save_ckpt") + 1]))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(a):
    if not os.path.exists(a.ckpt):
        log("tải checkpoint Vim-Ti pretrain ImageNet")
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download("hustvl/Vim-tiny-midclstok", "vim_t_midclstok_76p1acc.pth"), a.ckpt)
    log("tải CIFAR-100 (một lần)")
    from kanvim.data import build_datasets
    build_datasets("cifar100", a.data_root)
    if a.dot6:
        log("tải Tiny-ImageNet (một lần, ~250 MB)")
        build_datasets("tiny-imagenet", a.data_root)
        a.ext_ds = ["tiny-imagenet"]
        if os.path.isdir(os.path.join(a.data_root, "imagenet-100", "train")):
            a.ext_ds.append("imagenet-100")
        else:
            log(f"CẢNH BÁO: chưa có {a.data_root}/imagenet-100 (xem scripts/prep_imagenet100.py) -> bỏ qua ImageNet-100")
        os.makedirs(a.out, exist_ok=True)
        for f in glob.glob(os.path.join(DOT6_PREV, "*.json")):
            if not os.path.exists(os.path.join(a.out, os.path.basename(f))):
                shutil.copy(f, a.out)
    if a.impl == "auto":
        r = subprocess.run([PY, os.path.join(ROOT, "scripts", "check_impl.py"), a.ckpt], cwd=ROOT,
                           capture_output=True, text=True)
        a.impl = "official" if r.returncode == 0 and "USE_OFFICIAL" in r.stdout else "pure"
        if a.impl == "pure":
            log("CẢNH BÁO: không dùng được kernel Vim chính thức -> bản pure (chậm hơn ~25%, kết quả như nhau)")
            log(r.stdout[-800:] + r.stderr[-800:])
    log(f"backbone implementation: {a.impl}")


def command(a, r, ratio, seed, tag, extra, dataset="cifar100"):
    name = os.path.splitext(result_name(r, ratio, seed, tag, dataset))[0]
    cmd = [PY, os.path.join(ROOT, "train.py"), "--dataset", dataset, "--data_root", a.data_root,
           "--backbone", "tiny", "--ckpt", a.ckpt, "--impl", a.impl, "--batch", str(a.batch), "--workers", str(a.workers),
           "--val_size", "5000", "--mixup", "0.2", "--cutmix", "0.0", "--clip_grad", "1.0",
           "--epochs", "30", "--patience", str(a.patience), "--lr", "1e-4", "--teacher_ckpt", a.teacher,
           "--reducer", r, "--ratio", str(ratio), "--seed", str(seed), "--prune_mode", "st",
           "--out", a.out, "--resume_dir", os.path.join(a.out, "state", name)] + (["--prune_norm"] if a.prune_norm else []) + extra
    if a.quick:  # vài bước mỗi epoch: chỉ để bắt lỗi trước khi chạy thật
        cmd += ["--epochs", "2", "--max_steps", "20", "--no_bench"]
    if tag:
        cmd += ["--tag", tag]
    return name, cmd


def job_phases(a):
    """Các pha chạy lần lượt; trong một pha, mỗi GPU lấy job tiếp theo trong hàng đợi."""
    if a.dot6:
        return jobs_dot6(a)
    return [jobs_r25(a) if a.r25_round else jobs_ctx(a) if a.ctx_round else jobs_stab() if a.stab_round
            else jobs_lr() if a.lr_round else jobs()]


def run_all(a, gpus):
    phases = job_phases(a)
    n_all = sum(len(p) for p in phases)
    n_todo = sum(not done(a, j) for p in phases for j in p)
    log(f"{n_all} job, đã xong {n_all - n_todo}, còn {n_todo}, GPU {gpus}")
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    failed = []
    left = a.limit
    for phase in phases:
        todo = [j for j in phase if not done(a, j)]
        if a.limit:
            todo, left = todo[:left], left - len(todo[:left])
            log(f"chạy thử: chỉ {len(todo)} job đầu của pha này")
        run_phase(a, gpus, todo, failed)
    return failed


def run_phase(a, gpus, todo, failed):
    q = queue.Queue()
    for j in todo:
        if "--teacher_ckpt" in j[4] and not os.path.exists(j[4][j[4].index("--teacher_ckpt") + 1]):
            name = os.path.splitext(result_name(*j[:4], *j[5:]))[0]
            log(f"bỏ qua {name}: chưa có teacher (job baseline cùng dataset lỗi?)")
            failed.append(name)
            continue
        q.put(j)

    def worker(g):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            name, cmd = command(a, *j)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log(f"GPU {g} bắt đầu {name}")
            t = time.time()
            with open(os.path.join(a.out, "logs", name + ".log"), "a") as f:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            log(f"GPU {g} xong {name}: {'OK' if rc == 0 else f'LỖI (exit {rc}), xem logs/{name}.log'} "
                f"({(time.time() - t) / 60:.0f} phút)")
            if rc:
                failed.append(name)
            report(a)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def fmt(xs):
    if not xs:
        return "-"
    s = f"{statistics.mean(xs):.2f}".replace(".", ",")
    return s + (f" ± {statistics.stdev(xs):.2f}".replace(".", ",") if len(xs) > 1 else "")


def fmt_d(d):
    """Hiệu số có dấu: '+1,23 ± 0,20'."""
    if not d:
        return "-"
    s = f"{statistics.mean(d):+.2f}".replace(".", ",")
    return s + (f" ± {statistics.stdev(d):.2f}".replace(".", ",") if len(d) > 1 else "")


def kpi(d):
    """KPI đặt trước: dương ở cả 3 seed và trung bình > 2 x độ lệch chuẩn."""
    if len(d) < 3:
        return f"chờ ({len(d)}/3 seed)"
    return "**đạt**" if all(x > 0 for x in d) and statistics.mean(d) > 2 * statistics.stdev(d) else "không đạt"


def load_results(out, with_ds=False):
    """with_ds: khoá có thêm dataset ở đầu (--dot6 có nhiều dataset trong cùng thư mục)."""
    res = {}
    for f in glob.glob(os.path.join(out, "*.json")):
        try:
            d = json.load(open(f))
        except ValueError:  # job ở GPU kia đang ghi dở file này, lần cập nhật sau sẽ đọc
            continue
        k = d["args"]
        key = (k["reducer"], k["ratio"], k["seed"], k.get("tag", ""))
        res[(k["dataset"],) + key if with_ds else key] = d
    return res


def report(a):
    if a.dot6:
        return report_dot6(a)
    res = load_results(a.out)
    if a.r25_round:
        return report_r25(a, res)
    if a.ctx_round:
        return report_ctx(a, res)
    if a.stab_round:
        return report_stab(a, res)
    if a.lr_round:
        return report_lr(a, res)
    L = ["# Kết quả chạy lại nhánh Prune đã sửa (CIFAR-100)", "",
         "Bản cũ (legacy): token bị loại không có gradient, nhiễu Gumbel cố định. Bản mới (st): mask straight-through "
         "trên toàn bộ token + token gộp các token bị loại (EViT), nhiễu Gumbel giảm dần về 0. Cùng cấu hình huấn luyện"
         + (f", trừ dừng sớm: patience {a.patience} (bản cũ: 6; 0 = chạy đủ 30 epoch)." if a.patience != 6 else ".")
         + (" Điểm của scorer được chuẩn hoá theo từng ảnh (--prune_norm)." if a.prune_norm else ""), "",
         "## Giữ 50%, 3 seed: test top-1 (%)", "",
         "| Phương pháp | seed 0 | seed 1 | seed 2 | **Bản mới: trung bình ± độ lệch** | Bản cũ | Epoch tốt nhất (bản mới) |",
         "|---|---|---|---|---|---|---|"]
    for r in NAMES:
        runs = [res.get((r, 0.5, s, "")) for s in (0, 1, 2)]
        xs = [d["top1"] for d in runs if d]
        cells = [f"{d['top1']:.2f}".replace(".", ",") if d else "-" for d in runs]
        ep = ", ".join(f"{d['best_epoch']}/{len(d['log'])}" for d in runs if d) or "-"
        L.append(f"| {NAMES[r]} | {' | '.join(cells)} | **{fmt(xs)}** | {fmt(OLD[(r, 0.5)])} | {ep} |")
    L += ["", "Tham chiếu (results_plan): " + "; ".join(f"{n} {v}" for n, v in REF) + ".", "",
          "## Các mức giữ (seed 0): test top-1 (%)", "", "| Phương pháp | giữ 75% (mới / cũ) | giữ 50% (mới / cũ) | giữ 25% (mới / cũ) |",
          "|---|---|---|---|"]
    for r in NAMES:
        cells = []
        for ratio in (0.75, 0.5, 0.25):
            d = res.get((r, ratio, 0, ""))
            cells.append(f"{fmt([d['top1']]) if d else '-'} / {fmt(OLD[(r, ratio)][:1])}")
        L.append(f"| {NAMES[r]} | {' | '.join(cells)} |")
    L += ["", "## Kiểm tra learning rate (giữ 50%, seed 0): test top-1 (%)", "",
          "| Phương pháp | mặc định (lr backbone 1e-4, lr bộ chấm điểm 1e-3) | lr bộ chấm điểm 1e-4 | lr backbone 5e-5 |",
          "|---|---|---|---|"]
    for r in NAMES:
        cells = [res.get((r, 0.5, 0, t)) for t in ("", "lrred1e-4", "lr5e-5")]
        L.append(f"| {NAMES[r]} | " + " | ".join(fmt([d["top1"]]) if d else "-" for d in cells) + " |")
    open(os.path.join(a.out, "KET_QUA_FIX.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def report_lr(a, res):
    L = ["# So sánh lr backbone 5e-5 với 1e-4 (CIFAR-100, giữ 50%)", "",
         "Cùng cấu hình results_plan, chỉ đổi lr backbone 1e-4 -> 5e-5 (lr head và lr bộ chấm điểm vẫn 1e-3); "
         f"patience {a.patience} (0 = chạy đủ 30 epoch). Prune là bản sửa (st"
         + (" + --prune_norm" if a.prune_norm else "") + "). Cột lr 1e-4: Pool/Drop lấy từ results_plan, Prune lấy "
         "từ results_fix_norm.", "",
         "## Test top-1 (%)", "",
         "| Phương pháp | lr 1e-4 (seed 0) | **lr 5e-5 (seed 0)** | Chênh | lr 1e-4 (các seed) | lr 5e-5 (các seed) "
         "| Val tốt nhất, lr 5e-5 (seed 0) |",
         "|---|---|---|---|---|---|---|"]
    for r, name in LR_NAMES.items():
        new = [d["top1"] for d in (res.get((r, 0.5, s, "lr5e-5")) for s in (0, 1, 2)) if d]
        d0 = res.get((r, 0.5, 0, "lr5e-5"))
        diff = f"{d0['top1'] - LR1E4[r][0]:+.2f}".replace(".", ",") if d0 else "-"
        L.append(f"| {name} | {fmt(LR1E4[r][:1])} | **{fmt([d0['top1']]) if d0 else '-'}** | {diff} "
                 f"| {fmt(LR1E4[r])} (n={len(LR1E4[r])}) | {fmt(new)}{f' (n={len(new)})' if new else ''} "
                 f"| {fmt([d0['val_top1']]) if d0 else '-'} |")
    open(os.path.join(a.out, "KET_QUA_LR.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def report_stab(a, res):
    f2 = lambda x: f"{x:.2f}".replace(".", ",")
    L = ["# Sửa độ ổn định của nhánh Prune (CIFAR-100, giữ 50%)", "",
         "Cấu hình: lr backbone 5e-5, st + --prune_norm, chạy đủ 30 epoch. Mốc = cùng cấu hình, chưa sửa (results_lr). "
         "\"Sụt cuối\" = val tốt nhất - val epoch cuối: càng gần 0 càng ổn định. \"Đổi token\" = tỉ lệ token được giữ "
         "khác epoch trước trên một batch val cố định (trung bình 10 epoch cuối).", "",
         "| Phương pháp | Seed | Cách sửa | Test top-1 | Val tốt nhất | Epoch tốt nhất | Val epoch cuối | Sụt cuối "
         "| Đổi token |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in NAMES:
        for s in STAB_SEEDS:
            t, v, e, last = STAB_BASE[(r, s)]
            L.append(f"| {NAMES[r]} | {s} | mốc (chưa sửa) | {f2(t)} | {f2(v)} | {e} | {f2(last)} | {f2(v - last)} | - |")
            for tag, desc, _ in STAB:
                d = res.get((r, 0.5, s, f"lr5e-5_{tag}"))
                if not d:
                    L.append(f"| {NAMES[r]} | {s} | {desc} | - | - | - | - | - | - |")
                    continue
                last = d["log"][-1]["top1"]
                flips = [x["flip"] for x in d["log"][-10:] if x.get("flip") is not None]
                flip = f2(100 * statistics.mean(flips)) + "%" if flips else "-"
                L.append(f"| {NAMES[r]} | {s} | {desc} | **{f2(d['top1'])}** | {f2(d['val_top1'])} | {d['best_epoch']} "
                         f"| {f2(last)} | {f2(d['val_top1'] - last)} | {flip} |")
    L += ["", "Diễn biến từng epoch (raw_std, scorer_norm, flip) nằm trong `log` của từng file json và cuối mỗi dòng "
          "của logs/*.log."]
    open(os.path.join(a.out, "KET_QUA_STAB.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def report_ctx(a, res):
    f2 = lambda x: f"{x:.2f}".replace(".", ",")
    tag = ctx_tag(a)
    runs = {r: [res.get((r, 0.5, s, tag)) for s in (0, 1, 2)] for r in CTX_BASE}
    gmacs = next((d["GMACs"] for ds in runs.values() for d in ds if d), None)
    L = [f"# Giảm token sau khối {a.reduce_at} (CIFAR-100, giữ 50%)", "",
         f"Một bộ giảm token đặt trước khối {a.reduce_at} của Vim-Ti: {a.reduce_at} khối đầu chạy đủ 196 token, nên bộ "
         "chấm điểm thấy token đã có ngữ cảnh. Mốc = giảm token ngay sau patch embedding (results_lr). Cùng cấu hình: "
         f"lr backbone 5e-5, st + --prune_norm, patience {a.patience}, chọn mô hình theo val.", "",
         f"GMACs: sau khối {a.reduce_at} = {f2(gmacs) if gmacs else '-'}; ở đầu vào = 0,76; baseline 100% token = 1,47.", "",
         "## Test top-1 (%)", "",
         f"| Phương pháp | seed 0 | seed 1 | seed 2 | **Sau khối {a.reduce_at}** | Ở đầu vào | Chênh |",
         "|---|---|---|---|---|---|---|"]
    for r, base in CTX_BASE.items():
        xs = [d["top1"] for d in runs[r] if d]
        cells = [f2(d["top1"]) if d else "-" for d in runs[r]]
        diff = f"{statistics.mean(xs) - statistics.mean(base):+.2f}".replace(".", ",") if xs else "-"
        L.append(f"| {LR_NAMES[r]} | {' | '.join(cells)} | **{fmt(xs)}** | {fmt(base)} | {diff} |")
    L += ["", "## KAN trừ MLP, ghép cặp theo seed (điểm %)", "",
          f"| Cặp | Sau khối {a.reduce_at}: từng seed | Trung bình | Ở đầu vào: từng seed | Trung bình |",
          "|---|---|---|---|---|"]
    for k, m in CTX_PAIRS:
        d = [x["top1"] - y["top1"] for x, y in zip(runs[k], runs[m]) if x and y]
        d0 = [x - y for x, y in zip(CTX_BASE[k], CTX_BASE[m])]
        sd = lambda v: " / ".join(f"{x:+.2f}".replace(".", ",") for x in v) or "-"
        L.append(f"| {LR_NAMES[k]} − {LR_NAMES[m]} | {sd(d)} | {fmt(d)} | {sd(d0)} | {fmt(d0)} |")
    L += ["", f"Checkpoint seed 0 (KAN/MLP Prune, KANPool/MLP Pool): {os.path.basename(a.out)}_ckpt/."]
    open(os.path.join(a.out, "KET_QUA_CTX.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def report_r25(a, res):
    f2 = lambda x: f"{x:.2f}".replace(".", ",")
    tag = ctx_tag(a)
    runs = {v: [res.get((r, 0.25, s, tag + t)) for s in (0, 1, 2)] for v, (r, t, _) in R25_VARIANTS.items()}
    gmacs = next((d["GMACs"] for ds in runs.values() for d in ds if d), None)
    L = [f"# Giữ 25% token sau khối {a.reduce_at} (CIFAR-100)", "",
         f"Cùng cấu hình results_ctx (lr backbone 5e-5, patience {a.patience}, chọn mô hình theo val), chỉ đổi tỉ lệ giữ "
         f"50% -> 25%. GMACs: {f2(gmacs) if gmacs else '-'} (giữ 50%: 0,94; baseline: 1,47).", "",
         "## Test top-1 (%)", "",
         "| Phương pháp | seed 0 | seed 1 | seed 2 | **Giữ 25%** | Giữ 50% (results_ctx) |",
         "|---|---|---|---|---|---|"]
    for r, base in R25_BASE.items():
        xs = [d["top1"] for d in runs[r] if d]
        cells = [f2(d["top1"]) if d else "-" for d in runs[r]]
        L.append(f"| {R25_NAMES[r]} | {' | '.join(cells)} | **{fmt(xs)}** | {fmt(base)} |")
    L += ["", "## Hiệu số ghép cặp theo seed (điểm %)", "",
          "KPI đặt trước khi chạy: **đạt** nếu hiệu số dương ở cả 3 seed và trung bình > 2 × độ lệch chuẩn. "
          "Có 4 phương pháp có học được so với Static Drop: báo cáo cả 4 dòng, không chỉ dòng tốt nhất.", "",
          "| Cặp | Từng seed | Trung bình ± độ lệch | KPI |", "|---|---|---|---|"]
    for k, m in R25_PAIRS:
        d = [x["top1"] - y["top1"] for x, y in zip(runs[k], runs[m]) if x and y]
        L.append(f"| {R25_NAMES[k]} − {R25_NAMES[m]} | {' / '.join(f'{x:+.2f}'.replace('.', ',') for x in d) or '-'} "
                 f"| {fmt(d)} | {kpi(d)} |")
    open(os.path.join(a.out, "KET_QUA_R25.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def report_dot6(a):
    f2 = lambda x: f"{x:.2f}".replace(".", ",")
    res = load_results(a.out, with_ds=True)
    runs = lambda ds, r, ratio, seeds=(0, 1, 2): [res.get((ds, r, ratio, s, DOT6_TAG)) for s in seeds]
    top = lambda ds, r, ratio: [d["top1"] for d in runs(ds, r, ratio) if d]
    base = top("cifar100", "identity", 1.0)
    rows = DOT6_METHODS + ["static_drop"]
    gm = {ratio: next((d["GMACs"] for d in runs("cifar100", "kan_pool", ratio) if d), None) for ratio in DOT6_RATIOS}
    L = ["# Đợt 6: đủ ma trận đề cương (giảm token ở đầu vào)", "",
         "Cấu hình chung: Vim-Ti pretrain ImageNet-1K, lr backbone 5e-5, lr head và bộ chấm điểm 1e-3, Prune st + "
         "--prune_norm, mixup 0,2, chưng cất từ baseline cùng dataset, chọn mô hình theo val, test dùng 1 lần. "
         "Baseline không chưng cất.", "",
         f"## CIFAR-100 (30 epoch, 3 seed)", "",
         f"Baseline Vim-Ti (100% token, 1,47 GMACs): **{fmt(base)}** (n={len(base)}).", "",
         "Test top-1 (%), trung bình ± độ lệch; trong ngoặc: chênh so với trung bình baseline. "
         "n < 3 = chưa đủ seed. Static Drop giữ 50% lấy từ results_lr.", "",
         "| Phương pháp | " + " | ".join(f"Giữ {round(r * 100)}% ({f2(gm[r]) if gm[r] else '-'} GMACs)"
                                      for r in DOT6_RATIOS) + " |",
         "|---|---|---|---|"]
    for r in rows:
        cells = []
        for ratio in DOT6_RATIOS:
            xs = top("cifar100", r, ratio)
            c = fmt(xs) + (f" (n={len(xs)})" if 0 < len(xs) < 3 else "")
            if xs and base:
                c += f" ({statistics.mean(xs) - statistics.mean(base):+.2f})".replace(".", ",")
            cells.append(c)
        L.append(f"| {DOT6_NAMES[r]} | {' | '.join(cells)} |")
    L += ["", "### Hiệu số ghép cặp theo seed (điểm %)", "",
          "KPI đặt trước: **đạt** nếu hiệu số dương ở cả 3 seed và trung bình > 2 × độ lệch chuẩn. "
          "Hai dòng đầu trả lời \"khi nào gộp, khi nào loại bỏ\" ở từng mức giữ.", "",
          "| Cặp | Giữ 75% | Giữ 50% | Giữ 25% |", "|---|---|---|---|"]
    for k, m, name in DOT6_PAIRS:
        cells = []
        for ratio in DOT6_RATIOS:
            d = [x["top1"] - y["top1"] for x, y in zip(runs("cifar100", k, ratio), runs("cifar100", m, ratio)) if x and y]
            cells.append(f"{fmt_d(d)}, {kpi(d)}" if d else "-")
        L.append(f"| {name} | {' | '.join(cells)} |")
    b0 = res.get(("cifar100", "identity", 1.0, 0, DOT6_TAG))
    if b0 and b0.get("throughput"):
        L += ["", "### Tốc độ và bộ nhớ khi suy luận (seed 0, lô 64, đo ngay sau huấn luyện)", "",
              f"Baseline: {b0['throughput']:.0f} ảnh/s, {b0['vram_MB']:.0f} MB. Ô: tăng tốc so với baseline / VRAM (MB). "
              "GPU còn lại đang chạy job khác nên số có nhiễu; số để báo cáo nên đo lại bằng scripts/bench.py.", "",
              "| Phương pháp | Giữ 75% | Giữ 50% | Giữ 25% |", "|---|---|---|---|"]
        for r in rows:
            cells = []
            for ratio in DOT6_RATIOS:
                d = res.get(("cifar100", r, ratio, 0, DOT6_TAG))
                cells.append(f"{d['throughput'] / b0['throughput']:.2f}× / {d['vram_MB']:.0f}".replace(".", ",")
                             if d and d.get("throughput") else "-")
            L.append(f"| {DOT6_NAMES[r]} | {' | '.join(cells)} |")
    for ds, title in DOT6_EXT.items():
        b = res.get((ds, "identity", 1.0, 0, DOT6_TAG))
        L += ["", f"## {title} (rút gọn: giữ 50%, seed 0, {a.ext_epochs} epoch)", "",
              "| Phương pháp | Test top-1 (%) | Chênh so với baseline | Epoch tốt nhất |", "|---|---|---|---|",
              f"| {DOT6_NAMES['identity']} | {f2(b['top1']) if b else '-'} | — | "
              f"{b['best_epoch'] if b else '-'}/{len(b['log']) if b else '-'} |"]
        for r in DOT6_METHODS:
            d = res.get((ds, r, 0.5, 0, DOT6_TAG))
            diff = f"{d['top1'] - b['top1']:+.2f}".replace(".", ",") if d and b else "-"
            L.append(f"| {DOT6_NAMES[r]} | {f2(d['top1']) if d else '-'} | {diff} | "
                     f"{d['best_epoch'] if d else '-'}/{len(d['log']) if d else '-'} |")
    open(os.path.join(a.out, "KET_QUA_DOT6.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    plot_dot6(a, res, base)


def plot_dot6(a, res, base):
    """Đường cong đánh đổi Top-1 theo GMACs trên CIFAR-100 (trung bình ± độ lệch qua các seed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in DOT6_METHODS + ["static_drop"]:
        pts = []
        for ratio in DOT6_RATIOS:
            ds = [res.get(("cifar100", r, ratio, s, DOT6_TAG)) for s in (0, 1, 2)]
            xs = [d["top1"] for d in ds if d]
            if xs:
                pts.append((next(d["GMACs"] for d in ds if d), statistics.mean(xs),
                            statistics.stdev(xs) if len(xs) > 1 else 0.0))
        if pts:
            ax.errorbar(*zip(*[(g, m) for g, m, _ in pts]), yerr=[s for *_, s in pts], marker="o", capsize=3,
                        linestyle="--" if "prune" in r or r == "static_drop" else "-", label=DOT6_NAMES[r])
    if base:
        ax.errorbar([1.467], [statistics.mean(base)], yerr=[statistics.stdev(base) if len(base) > 1 else 0.0],
                    marker="*", markersize=12, color="black", capsize=3, label=DOT6_NAMES["identity"])
    ax.set_xlabel("GMACs / ảnh")
    ax.set_ylabel("Top-1 trên tập kiểm tra (%)")
    ax.set_title("CIFAR-100: độ chính xác theo chi phí (nét liền: gộp, nét đứt: loại bỏ)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_dot6_cifar100.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "vim_t.pth"))
    ap.add_argument("--teacher", default=os.path.join(ROOT, "teacher", "teacher_cifar100.pt"))
    ap.add_argument("--impl", default="auto", choices=["auto", "pure", "official"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results_fix"))
    ap.add_argument("--patience", type=int, default=6, help="0 = chạy đủ 30 epoch (vẫn chọn model theo val tốt nhất)")
    ap.add_argument("--prune_norm", action="store_true", help="chuẩn hoá điểm theo từng ảnh (sửa lỗi điểm KAN sụp / MLP phình)")
    ap.add_argument("--limit", type=int, default=0, help=">0: chỉ chạy N job đầu còn thiếu (chạy thử)")
    ap.add_argument("--lr_round", action="store_true",
                    help="đợt so sánh lr backbone 5e-5 (Pool/Drop + Prune, giữ 50%%), bảng KET_QUA_LR.md")
    ap.add_argument("--stab_round", action="store_true",
                    help="đợt sửa độ ổn định của Prune (wd0 / g03 / frz, lr 5e-5, seed 1 rồi 0), bảng KET_QUA_STAB.md")
    ap.add_argument("--ctx_round", action="store_true",
                    help="đợt giảm token sau khối --reduce_at (KAN/MLP Prune, KAN/MLP Pool, Static Drop, 3 seed), "
                         "bảng KET_QUA_CTX.md")
    ap.add_argument("--r25_round", action="store_true",
                    help="đợt giữ 25%% sau khối --reduce_at (KAN/MLP Pool, Static Drop, Avg Pool, KAN/MLP Prune + phủ "
                         "đều, 3 seed), bảng KET_QUA_R25.md")
    ap.add_argument("--dot6", action="store_true",
                    help="đợt 6: CIFAR-100 đủ 7 phương pháp x giữ 75/50/25%% x 3 seed + Tiny-ImageNet / ImageNet-100 "
                         "rút gọn, giảm ở đầu vào, lr 5e-5; bảng KET_QUA_DOT6.md")
    ap.add_argument("--ext_epochs", type=int, default=10, help="--dot6: số epoch cho Tiny-ImageNet và ImageNet-100")
    ap.add_argument("--reduce_at", type=int, default=6, help="--ctx_round / --r25_round: giảm token trước khối này")
    ap.add_argument("--quick", action="store_true",
                    help="chạy thử: mỗi job 2 epoch x 20 bước, không đo tốc độ (dùng --out riêng, vd results_quick)")
    ap.add_argument("--batch", type=int, default=64, help="chỉ đổi khi chạy thử trên GPU ít VRAM (kết quả thật: 64)")
    ap.add_argument("--gpus", default=None, help="vd '0,1'. Mặc định: tất cả GPU")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    a = ap.parse_args()
    a.ext_ds = []  # --dot6: prepare() điền các dataset phụ có sẵn
    if a.stab_round or a.ctx_round or a.r25_round or a.dot6:
        a.prune_norm = True
    if a.dot6:
        a.patience = 0  # như results_lr: 16 job giữ 50% dùng lại phải cùng cấu hình
    import torch
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else list(range(torch.cuda.device_count()))
    if not gpus:
        sys.exit("KHÔNG THẤY GPU")
    if not os.path.exists(a.teacher):
        sys.exit(f"thiếu teacher {a.teacher}")
    prepare(a)
    failed = run_all(a, gpus)
    report(a)
    shutil.make_archive(a.out, "zip", os.path.dirname(a.out), os.path.basename(a.out))
    done = "XONG CHẠY THỬ" if a.limit else "XONG TOÀN BỘ"
    log(f"{done if not failed else 'XONG, có job lỗi: ' + ', '.join(failed)}. "
        f"Bảng: {a.out}/{'KET_QUA_DOT6.md' if a.dot6 else 'KET_QUA_R25.md' if a.r25_round else 'KET_QUA_CTX.md' if a.ctx_round else 'KET_QUA_STAB.md' if a.stab_round else 'KET_QUA_LR.md' if a.lr_round else 'KET_QUA_FIX.md'}"
        f" | tải về: {a.out}.zip")


if __name__ == "__main__":
    main()
