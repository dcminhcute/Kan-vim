"""Report of a run_plan.py output folder:  python scripts/report_plan.py results_plan
Writes BAO_CAO.md, ket_qua.csv, fig_*.png into that folder. Works on partial results too."""
import csv
import glob
import json
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "results_plan"
METHODS = ["kan_pool", "mlp_pool", "avg_pool", "max_pool", "kan_prune", "mlp_prune"]
CONTROLS = ["static_drop", "random_drop", "kan_prune_ms", "mlp_prune_ms"]
ORDER = ["identity"] + METHODS + CONTROLS + ["mtr"]
NAMES = {"identity": "Vim-Ti (100% token)", "kan_pool": "KANPool", "mlp_pool": "MLP Pool", "avg_pool": "Avg Pool",
         "max_pool": "Max Pool", "kan_prune": "KAN Prune", "mlp_prune": "MLP Prune", "static_drop": "Static drop",
         "random_drop": "Random drop", "kan_prune_ms": "KAN Prune (3 tầng)", "mlp_prune_ms": "MLP Prune (3 tầng)",
         "mtr": "MTR (không train)"}
GROUP = {"identity": "baseline", "kan_pool": "gộp, học (KAN)", "mlp_pool": "gộp, học (MLP)", "avg_pool": "gộp, cố định",
         "max_pool": "gộp, cố định", "kan_prune": "loại bỏ, học (KAN)", "mlp_prune": "loại bỏ, học (MLP)",
         "static_drop": "loại bỏ, cố định", "random_drop": "loại bỏ, ngẫu nhiên", "kan_prune_ms": "loại bỏ nhiều tầng (KAN)",
         "mlp_prune_ms": "loại bỏ nhiều tầng (MLP)", "mtr": "gộp nhiều tầng, không train"}
COLORS = {"identity": "#222222", "kan_pool": "#c8641a", "mlp_pool": "#2f74c0", "avg_pool": "#8c8c8c",
          "max_pool": "#bdbdbd", "kan_prune": "#e8a33d", "mlp_prune": "#72b7b2", "static_drop": "#9c755f",
          "random_drop": "#bab0ac", "kan_prune_ms": "#b07aa1", "mlp_prune_ms": "#59a14f", "mtr": "#e15759"}
TEST_N = {"cifar10": 10000, "cifar100": 10000, "tiny-imagenet": 10000, "food101": 25250}


def load(sub):
    out = []
    for f in sorted(glob.glob(os.path.join(R, sub, "*.json"))):
        try:
            r = json.load(open(f, encoding="utf-8"))
            r["_folder"] = sub
            out.append(r)
        except Exception:  # noqa: BLE001
            pass
    return out


def key(r):
    a = r["args"]
    return a["reducer"], float(a["ratio"]), int(a.get("seed", 0)), a.get("tag", "")


def f2(v, d=2):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def ci95(r):
    n = TEST_N.get(r["args"]["dataset"])
    p = r.get("top1")
    return None if n is None or p is None else 1.96 * math.sqrt(p * (100 - p) / n)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    return f"{st.mean(v):.2f}" + (f" ± {st.stdev(v):.2f}" if len(v) > 1 else "")


plan = json.load(open(os.path.join(R, "plan.json"))) if os.path.exists(os.path.join(R, "plan.json")) else {}
search, main, second, abl = load("search"), load("main"), load("second"), load("ablation")
mtr_all = load("mtr")
ds1 = (main or search or [{"args": {"dataset": "?"}}])[0]["args"]["dataset"]
mtr1 = [r for r in mtr_all if r["args"]["dataset"] == ds1]
mtr2 = [r for r in mtr_all if r["args"]["dataset"] != ds1]
M = {key(r): r for r in main + mtr1}
S2 = {key(r): r for r in second + mtr2}
A = {key(r): r for r in abl}
base = M.get(("identity", 1.0, 0, ""))
ratio_main = 0.5
ratios = sorted({k[1] for k in M if k[1] < 1.0}, reverse=True)
md = ["# Kết quả thí nghiệm", "", "_Tự sinh bởi `scripts/report_plan.py`._", ""]
csv_rows = []


def row_of(r, b, section):
    k = r["args"]["reducer"]
    x = {"section": section, "dataset": r["args"]["dataset"], "method": NAMES.get(k, k), "reducer": k,
         "keep": r["args"]["ratio"], "seed": r["args"].get("seed", 0), "tag": r["args"].get("tag", ""),
         "val_top1": r.get("val_top1"), "best_epoch": r.get("best_epoch"), "epochs_run": len(r["log"]),
         "stopped": r.get("stopped"), "test_top1": r.get("top1"), "ci95": ci95(r),
         "delta_vs_baseline": (r["top1"] - b["top1"]) if b and r.get("top1") is not None else None,
         "GMACs": r["GMACs"], "GMACs_cut_pct": (1 - r["GMACs"] / b["GMACs"]) * 100 if b else None,
         "mean_tokens": r.get("mean_tokens"), "reducer_params": r["reducer_params"], "img_s": r.get("throughput"),
         "speedup": (r["throughput"] / b["throughput"]) if b and r.get("throughput") and b.get("throughput") else None,
         "vram_MB": r.get("vram_MB"), "train_h": sum(l["total_min"] for l in r["log"]) / 60}
    csv_rows.append(x)
    return x


def table(rs, b, section):
    out = ["| Phương pháp | Nhóm | Giữ | Val top-1 | Epoch tốt nhất / đã chạy | **Test top-1** | Δ baseline | GMACs (giảm) | Tham số module | ảnh/s (tăng tốc) | Giờ train |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        x = row_of(r, b, section)
        ep = "-" if x["best_epoch"] is None else f"{x['best_epoch']} / {x['epochs_run']}"
        out.append(f"| {x['method']} | {GROUP.get(x['reducer'], '')} | {x['keep']:.0%} | {f2(x['val_top1'])} | {ep} | "
                   f"**{f2(x['test_top1'])}** ± {f2(x['ci95'], 1)} | "
                   f"{('%+.2f' % x['delta_vs_baseline']) if x['delta_vs_baseline'] is not None else '-'} | "
                   f"{x['GMACs']:.3f} ({f2(x['GMACs_cut_pct'], 0)}%) | {x['reducer_params']:,} | {f2(x['img_s'], 1)} "
                   f"({f2(x['speedup'])}×) | {x['train_h']:.1f} |")
    return out + [""]


def sort_methods(rs):
    return sorted(rs, key=lambda r: (ORDER.index(r["args"]["reducer"]) if r["args"]["reducer"] in ORDER else 99,
                                     -float(r["args"]["ratio"])))


# ---- 1. setup ------------------------------------------------------------------------------------
any_run = next((r for r in main + search if r["log"]), None)
if any_run:
    a = any_run["args"]
    md += ["## 1. Thiết lập", "",
           f"- Dataset chính **{ds1}**: train {plan.get('n_train', '?')} ảnh, validation {plan.get('val_size', a.get('val_size'))} ảnh "
           f"(tách cố định, cân bằng theo lớp, từ tập train gốc), test = tập test chính thức ({TEST_N.get(ds1, '?')} ảnh), "
           f"**test chỉ dùng 1 lần** trên model có val tốt nhất. Ảnh resize 224×224."]
    if second:
        a2 = second[0]["args"]
        test2 = "tập val chính thức" if a2["dataset"] == "tiny-imagenet" else "tập test chính thức"
        md.append(f"- Dataset thứ hai **{a2['dataset']}**: train {plan.get('n_train2', '?')} ảnh, "
                  f"validation {plan.get('val_size2', '?')} ảnh, test = {test2} ({TEST_N.get(a2['dataset'], '?')} ảnh); "
                  f"tối đa {a2['epochs']} epoch, lr backbone {a2['lr']:g}.")
    md += [f"- Backbone Vim-{a['backbone']} pretrain ImageNet-1K, cài đặt scan `{plan.get('impl', a['impl'])}`.",
           f"- Tối đa {a['epochs']} epoch, batch {a['batch']}, AdamW (weight decay {a['wd']}), OneCycle cosine, fp16; "
           f"lr backbone chọn ở bước dò LR, lr head {a['lr_head']}, lr module {a['lr_reducer']}.",
           f"- Chống overfitting: random crop + lật ngang, label smoothing 0.1, weight decay {a['wd']}, mixup α={a['mixup']}"
           + (f", cutmix α={a['cutmix']}" if a.get("cutmix") else "")
           + f", gradient clipping {a['clip_grad']}, early stopping theo val top-1, chọn model theo val. KAN: L1 = {a['l1']}, "
           f"grid {a['grid_size']}, hidden {a['kan_hidden']}; MLP cùng số tham số với KAN.",
           f"- Phương pháp giảm token: distillation (KL, trọng số {a['distill']}) từ baseline seed 0 đã fine-tune.",
           "- Giảm token nhiều tầng: loại bỏ trước block 0, 8, 16, mỗi tầng giữ cùng một tỉ lệ, chọn sao cho số token "
           "trung bình qua 24 block bằng bản 1 tầng (cùng GMACs).",
           "- MTR (không train): sau mỗi block, chấm điểm token bằng Δ, gộp nhóm 'source' vào 'target' giống nhất; "
           "tỉ lệ chọn để GMACs bằng các phương pháp 1 tầng cùng mức giữ; chạy trên baseline đã fine-tune.",
           "- ±: khoảng tin cậy 95% do kích thước tập test; 'a ± b' ở bảng nhiều seed là trung bình ± độ lệch chuẩn.", ""]

# ---- 2. LR search --------------------------------------------------------------------------------
if search:
    search.sort(key=lambda r: r["args"]["lr"])
    best = max(search, key=lambda r: r["val_top1"])
    md += [f"## 2. Dò learning rate (baseline, {search[0]['args']['epochs']} epoch, {search[0]['args']['train_subset']} ảnh)", "",
           "| lr backbone | val top-1 tốt nhất | epoch tốt nhất | train loss cuối | val loss cuối |", "|---|---|---|---|---|"]
    for r in search:
        l = r["log"][-1]
        md.append(f"| {r['args']['lr']:g}{' **(chọn)**' if r is best else ''} | {r['val_top1']:.2f} | {r['best_epoch']} | "
                  f"{l['train_loss']:.3f} | {l['val_loss']:.3f} |")
    md.append("")
    plt.figure(figsize=(5.5, 3.6))
    for r in search:
        plt.plot([l["epoch"] + 1 for l in r["log"]], [l["top1"] for l in r["log"]], "o-", label=f"lr {r['args']['lr']:g}")
    plt.xlabel("epoch"); plt.ylabel("val top-1 (%)"); plt.title("Dò learning rate"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_lr_search.png"), dpi=150); plt.close()
    md += ["![](fig_lr_search.png)", ""]

# ---- 3. main -------------------------------------------------------------------------------------
main_rs = sort_methods([r for k, r in M.items() if k[2] == 0 and not k[3] and (k[1] == ratio_main or k[0] == "identity")
                        and k[0] in ["identity"] + METHODS])
if main_rs:
    md += [f"## 3. Kết quả chính ({ds1}, giữ {ratio_main:.0%} token, seed 0)", ""] + table(main_rs, base, "main")

# ---- 4. seeds ------------------------------------------------------------------------------------
seeds = sorted({k[2] for k in M})
if len(seeds) > 1:
    md += [f"## 4. Độ ổn định qua seed ({ds1}, giữ {ratio_main:.0%})", "",
           "| Phương pháp | " + " | ".join(f"seed {s}" for s in seeds) + " | **Trung bình ± độ lệch** | Δ baseline (trung bình) |",
           "|---|" + "---|" * (len(seeds) + 2)]
    bmean = [M[("identity", 1.0, s, "")]["top1"] for s in seeds if ("identity", 1.0, s, "") in M]
    for m in ["identity"] + METHODS:
        q = 1.0 if m == "identity" else ratio_main
        v = [M[(m, q, s, "")]["top1"] if (m, q, s, "") in M else None for s in seeds]
        got = [x for x in v if x is not None]
        d = f"{st.mean(got) - st.mean(bmean):+.2f}" if got and bmean and m != "identity" else "-"
        md.append(f"| {NAMES[m]} | " + " | ".join(f2(x) for x in v) + f" | **{ms(v)}** | {d} |")
        for s in seeds:
            if s and (m, q, s, "") in M:
                row_of(M[(m, q, s, "")], base, "seeds")
    md.append("")

# ---- 5. ratios -----------------------------------------------------------------------------------
if len(ratios) > 1:
    md += [f"## 5. Các mức giữ token ({ds1}, seed 0): test top-1 (GMACs)", "",
           "| Phương pháp | " + " | ".join(f"giữ {q:.0%}" for q in ratios) + " |", "|---|" + "---|" * len(ratios)]
    if base:
        md.append(f"| {NAMES['identity']} (100%) | " + " | ".join(f"{base['top1']:.2f} ({base['GMACs']:.2f})" for _ in ratios) + " |")
    for m in METHODS + CONTROLS + ["mtr"]:
        cells = []
        for q in ratios:
            r = M.get((m, q, 0, ""))
            cells.append("-" if r is None else f"{f2(r['top1'])} ({r['GMACs']:.2f})")
        if any(c != "-" for c in cells):
            md.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    md.append("")
    plt.figure(figsize=(7, 4.6))
    for m in METHODS + ["mtr", "kan_prune_ms", "mlp_prune_ms"]:
        pts = sorted([(M[(m, q, 0, "")]["GMACs"], M[(m, q, 0, "")]["top1"]) for q in ratios if (m, q, 0, "") in M])
        if pts:
            style = "--" if m in ("avg_pool", "max_pool", "mtr") else "-"
            plt.plot(*zip(*pts), style, marker="o", color=COLORS[m], label=NAMES[m])
    if base:
        plt.plot(base["GMACs"], base["top1"], "k*", ms=12, label=NAMES["identity"])
    plt.xlabel("GMACs / ảnh"); plt.ylabel("Test top-1 (%)"); plt.grid(alpha=.3); plt.legend(fontsize=7)
    plt.title(f"Đánh đổi độ chính xác - chi phí ({ds1})")
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig_acc_vs_gmacs.png"), dpi=150); plt.close()
    md += ["![](fig_acc_vs_gmacs.png)", ""]
    for q in ratios:
        for m in METHODS + CONTROLS + ["mtr"]:
            if (m, q, 0, "") in M and q != ratio_main:
                row_of(M[(m, q, 0, "")], base, "ratios")

# ---- 6. controls / multi-stage / MTR -------------------------------------------------------------
ctl = sort_methods([M[k] for k in M if k[0] in CONTROLS + ["mtr", "kan_prune", "mlp_prune"] and k[1] == ratio_main
                    and k[2] == 0 and not k[3]])
if any(r["args"]["reducer"] in CONTROLS + ["mtr"] for r in ctl):
    md += [f"## 6. Đối chứng, giảm token nhiều tầng và MTR ({ds1}, giữ {ratio_main:.0%}, seed 0)", ""] + table(ctl, base, "controls")

# ---- 7. dataset 2 --------------------------------------------------------------------------------
if second:
    b2 = S2.get(("identity", 1.0, 0, ""))
    ds2 = second[0]["args"]["dataset"]
    md += [f"## 7. Dataset thứ hai: {ds2} (giữ {ratio_main:.0%}, seed 0)", ""] + table(sort_methods(list(S2.values())), b2, "dataset2")

# ---- 8. ablation ---------------------------------------------------------------------------------
if abl:
    tags = {"": "mặc định (grid 5, hidden 32, L1, distill)", "g3": "grid size 3", "g8": "grid size 8",
            "h16": "hidden 16", "h64": "hidden 64", "noL1": "không L1", "nodistill": "không distillation"}
    md += [f"## 8. Ablation KAN ({ds1}, giữ {ratio_main:.0%}, seed 0): test top-1 (tham số module)", "",
           "| Biến thể | KANPool | KAN Prune |", "|---|---|---|"]
    for t, label in tags.items():
        cells = []
        for m in ["kan_pool", "kan_prune"]:
            r = M.get((m, ratio_main, 0, "")) if t == "" else A.get((m, ratio_main, 0, t))
            cells.append("-" if r is None else f"{f2(r['top1'])} ({r['reducer_params']:,})")
            if r is not None and t:
                row_of(r, base, "ablation")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

# ---- 9. overfitting ------------------------------------------------------------------------------
curve_rs = [r for r in main_rs + (sort_methods(second) if second else []) if r["log"]]
if curve_rs:
    md += ["## 9. Kiểm tra overfitting", "",
           "Train loss/accuracy đo trên ảnh đã augmentation + mixup nên không phản ánh đúng độ khớp trên ảnh sạch; dấu hiệu "
           "overfitting là val loss tăng lại trong khi train loss vẫn giảm (xem hình).", "",
           "| Dataset | Phương pháp | train loss (tốt nhất → cuối) | val loss (tốt nhất → cuối) | val loss thấp nhất @ epoch | val top-1 (tốt nhất → cuối) | Nhận xét |",
           "|---|---|---|---|---|---|---|"]
    for r in curve_rs:
        L, b = r["log"], r.get("best_epoch", len(r["log"]) - 1)
        lb = next(l for l in L if l["epoch"] == b)
        le = L[-1]
        vmin = min(L, key=lambda l: l["val_loss"])
        rising = le["val_loss"] > vmin["val_loss"] * 1.05 and le["train_loss"] < lb["train_loss"]
        note = ("val loss tăng lại trong khi train loss giảm: có dấu hiệu overfitting, đã xử lý bằng chọn model theo val"
                if rising else "chưa thấy overfitting rõ")
        if str(r.get("stopped", "")).startswith("early"):
            note += "; dừng sớm"
        md.append(f"| {r['args']['dataset']} | {NAMES.get(r['args']['reducer'])} | {lb['train_loss']:.3f} → {le['train_loss']:.3f} | "
                  f"{lb['val_loss']:.3f} → {le['val_loss']:.3f} | {vmin['val_loss']:.3f} @ {vmin['epoch']} | "
                  f"{lb['top1']:.2f} → {le['top1']:.2f} | {note} |")
    md.append("")
    for tag, rs in [("", main_rs), ("_d2", sort_methods(second) if second else [])]:
        rs = [r for r in rs if r["log"]]
        if not rs:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for r in rs:
            k, L = r["args"]["reducer"], r["log"]
            e = [l["epoch"] + 1 for l in L]
            for i, f in enumerate(["train_loss", "val_loss", "top1"]):
                ax[i].plot(e, [l[f] for l in L], "-", color=COLORS.get(k), label=NAMES.get(k, k))
            if r.get("best_epoch") is not None:
                ax[2].plot(r["best_epoch"] + 1, r["val_top1"], "*", color=COLORS.get(k), ms=10)
        for x, t in zip(ax, ["train loss", "val loss", "val top-1 (%)  (* = model được chọn)"]):
            x.set_xlabel("epoch"); x.set_title(f"{rs[0]['args']['dataset']}: {t}"); x.grid(alpha=.3)
        ax[2].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(R, f"fig_curves{tag}.png"), dpi=150); plt.close()
        md += [f"![](fig_curves{tag}.png)", ""]

# ---- 10. efficiency ------------------------------------------------------------------------------
eff = sort_methods([r for k, r in M.items() if k[2] == 0 and not k[3]])
if eff:
    md += [f"## 10. Chi phí tính toán ({ds1}, đo khi suy luận, batch 64, fp16)", "",
           "| Phương pháp | Giữ | Token trung bình / block | GMACs | Tham số module | ảnh/s | Tăng tốc | VRAM MB |", "|---|---|---|---|---|---|---|---|"]
    for r in eff:
        sp = (r["throughput"] / base["throughput"]) if base and r.get("throughput") and base.get("throughput") else None
        md.append(f"| {NAMES.get(r['args']['reducer'])} | {float(r['args']['ratio']):.0%} | {f2(r.get('mean_tokens'), 1)} | "
                  f"{r['GMACs']:.3f} | {r['reducer_params']:,} | {f2(r.get('throughput'), 1)} | {f2(sp)}× | {f2(r.get('vram_MB'), 0)} |")
    md += ["", "_Tốc độ đo với cài đặt scan ghi ở mục 1; so sánh tương đối giữa các phương pháp, không so với số công bố của Vim._", ""]
    xs = [r for r in main_rs if r.get("top1") is not None]
    if xs:
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(xs)), [r["top1"] for r in xs], yerr=[ci95(r) or 0 for r in xs], capsize=3,
                color=[COLORS.get(r["args"]["reducer"]) for r in xs])
        for i, r in enumerate(xs):
            plt.text(i, r["top1"] + 1, f"{r['top1']:.1f}", ha="center", fontsize=8)
        plt.xticks(range(len(xs)), [NAMES[r["args"]["reducer"]] for r in xs], rotation=25, ha="right")
        plt.ylabel("Test top-1 (%)"); plt.title(f"{ds1}, giữ {ratio_main:.0%}"); plt.grid(axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(R, "fig_test_top1.png"), dpi=150); plt.close()
        md += ["![](fig_test_top1.png)", ""]


# ---- 11. key comparisons -------------------------------------------------------------------------
def acc(m, q, tbl):
    v = [r["top1"] for k, r in tbl.items() if k[0] == m and k[1] == q and not k[3] and r.get("top1") is not None]
    return (st.mean(v), len(v)) if v else (None, 0)


pairs = [("KAN vs MLP, gộp (cùng tham số)", "kan_pool", "mlp_pool"), ("KAN vs MLP, loại bỏ (cùng tham số)", "kan_prune", "mlp_prune"),
         ("KANPool vs Avg Pool (học vs cố định)", "kan_pool", "avg_pool"), ("KANPool vs KAN Prune (gộp vs loại bỏ)", "kan_pool", "kan_prune"),
         ("KAN Prune vs Random drop", "kan_prune", "random_drop"), ("KAN Prune 3 tầng vs 1 tầng", "kan_prune_ms", "kan_prune"),
         ("KANPool vs MTR (có train vs không train)", "kan_pool", "mtr"), ("KAN Prune vs MTR", "kan_prune", "mtr")]
lines = []
for tbl, dsn in [(M, ds1)] + ([(S2, second[0]["args"]["dataset"] + " (dataset 2)")] if second else []):
    for q in sorted({k[1] for k in tbl if k[1] < 1.0}, reverse=True):
        for label, x, y in pairs:
            (ax_, nx), (ay, ny) = acc(x, q, tbl), acc(y, q, tbl)
            if ax_ is not None and ay is not None:
                lines.append(f"| {dsn} | {q:.0%} | {label} | {ax_:.2f} | {ay:.2f} | **{ax_ - ay:+.2f}** | {min(nx, ny)} |")
if lines:
    md += ["## 11. So sánh rút gọn (dùng cho phần thảo luận)", "",
           "Hiệu = A − B (điểm %, trung bình các seed có sẵn). Nếu |hiệu| nhỏ hơn khoảng tin cậy (~±1 điểm với 10.000 ảnh "
           "test) hoặc nhỏ hơn độ lệch giữa các seed thì chưa đủ để kết luận khác biệt.", "",
           "| Dataset | Giữ | So sánh (A vs B) | A | B | Hiệu | số seed |", "|---|---|---|---|---|---|---|"] + lines + [""]

if csv_rows:
    with open(os.path.join(R, "ket_qua.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader(); w.writerows(csv_rows)
open(os.path.join(R, "BAO_CAO.md"), "w", encoding="utf-8").write("\n".join(md))
print(f"wrote {os.path.join(R, 'BAO_CAO.md')}: {len(main)} chính, {len(second)} dataset 2, {len(abl)} ablation, "
      f"{len(mtr_all)} MTR, {len(search)} dò LR")
