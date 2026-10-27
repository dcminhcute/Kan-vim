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
  Bảng: <out>/KET_QUA_LR.md
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
    return out


def result_name(r, ratio, seed, tag):
    return f"cifar100_tiny_{r}_r{ratio}_s{seed}{'_' + tag if tag else ''}.json"


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
    if a.impl == "auto":
        r = subprocess.run([PY, os.path.join(ROOT, "scripts", "check_impl.py"), a.ckpt], cwd=ROOT,
                           capture_output=True, text=True)
        a.impl = "official" if r.returncode == 0 and "USE_OFFICIAL" in r.stdout else "pure"
        if a.impl == "pure":
            log("CẢNH BÁO: không dùng được kernel Vim chính thức -> bản pure (chậm hơn ~25%, kết quả như nhau)")
            log(r.stdout[-800:] + r.stderr[-800:])
    log(f"backbone implementation: {a.impl}")


def command(a, r, ratio, seed, tag, extra):
    name = os.path.splitext(result_name(r, ratio, seed, tag))[0]
    cmd = [PY, os.path.join(ROOT, "train.py"), "--dataset", "cifar100", "--data_root", a.data_root,
           "--backbone", "tiny", "--ckpt", a.ckpt, "--impl", a.impl, "--batch", "64", "--workers", str(a.workers),
           "--val_size", "5000", "--mixup", "0.2", "--cutmix", "0.0", "--clip_grad", "1.0",
           "--epochs", "30", "--patience", str(a.patience), "--lr", "1e-4", "--teacher_ckpt", a.teacher,
           "--reducer", r, "--ratio", str(ratio), "--seed", str(seed), "--prune_mode", "st",
           "--out", a.out, "--resume_dir", os.path.join(a.out, "state", name)] + (["--prune_norm"] if a.prune_norm else []) + extra
    if tag:
        cmd += ["--tag", tag]
    return name, cmd


def run_all(a, gpus):
    all_jobs = jobs_lr() if a.lr_round else jobs()
    todo = [j for j in all_jobs if not os.path.exists(os.path.join(a.out, result_name(*j[:4])))]
    log(f"{len(all_jobs)} job, đã xong {len(all_jobs) - len(todo)}, còn {len(todo)}, GPU {gpus}")
    if a.limit:
        todo = todo[:a.limit]
        log(f"chạy thử: chỉ {len(todo)} job đầu")
    q = queue.Queue()
    for j in todo:
        q.put(j)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    failed = []

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
    return failed


def fmt(xs):
    if not xs:
        return "-"
    s = f"{statistics.mean(xs):.2f}".replace(".", ",")
    return s + (f" ± {statistics.stdev(xs):.2f}".replace(".", ",") if len(xs) > 1 else "")


def load_results(out):
    res = {}
    for f in glob.glob(os.path.join(out, "*.json")):
        try:
            d = json.load(open(f))
        except ValueError:  # job ở GPU kia đang ghi dở file này, lần cập nhật sau sẽ đọc
            continue
        k = d["args"]
        res[(k["reducer"], k["ratio"], k["seed"], k.get("tag", ""))] = d
    return res


def report(a):
    res = load_results(a.out)
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
    ap.add_argument("--gpus", default=None, help="vd '0,1'. Mặc định: tất cả GPU")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    a = ap.parse_args()
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
        f"Bảng: {a.out}/{'KET_QUA_LR.md' if a.lr_round else 'KET_QUA_FIX.md'} | tải về: {a.out}.zip")


if __name__ == "__main__":
    main()
