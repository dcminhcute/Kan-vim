"""Tables (markdown + CSV) and figures from bench_*.csv and run *.json.
  python scripts/report.py RESULTS_DIR
Outputs in RESULTS_DIR: report.md, results_summary.csv, fig_efficiency.png, fig_acc_<dataset>.png
"""
import csv
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "results"
ORDER = ["identity", "kan_pool", "mlp_pool", "avg_pool", "max_pool", "kan_prune", "mlp_prune", "static_drop", "random_drop"]
NAMES = {"identity": "Vim-Ti (100%)", "avg_pool": "Avg Pool", "max_pool": "Max Pool", "static_drop": "Static drop",
         "random_drop": "Random drop", "mlp_pool": "MLP Pool", "kan_pool": "KANPool",
         "mlp_prune": "MLP Prune", "kan_prune": "KAN Prune"}
COLORS = {"identity": "#000000", "avg_pool": "#8c8c8c", "max_pool": "#bdbdbd", "static_drop": "#d9a066",
          "mlp_pool": "#2f74c0", "kan_pool": "#c8641a", "mlp_prune": "#72b7b2", "kan_prune": "#e8a33d"}
STYLE = {"kan_pool": "o-", "mlp_pool": "o-", "avg_pool": "s--", "max_pool": "s--", "kan_prune": "^-", "mlp_prune": "^-"}
md = []


def mstd(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


# ---------------- efficiency (no training)
bench = []
for f in glob.glob(os.path.join(R, "bench_*.csv")):
    bench += list(csv.DictReader(open(f)))
if bench:
    base = next(r for r in bench if r["method"] == "identity")
    md += ["## Chi phí tính toán (không cần train)", "",
           "| Phương pháp | Giữ | #token | GMACs | Giảm GMACs | Tham số module | img/s | Tăng tốc | VRAM (MB) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in bench:
        g, g0 = float(r["GMACs"]), float(base["GMACs"])
        sp = f"{float(r['img_per_s']) / float(base['img_per_s']):.2f}x" if r.get("img_per_s") else "-"
        md.append(f"| {NAMES.get(r['method'], r['method'])} | {float(r['keep']):.0%} | {r['tokens']} | {g:.3f} | "
                  f"{(1 - g / g0):.1%} | {int(r['reducer_params']):,} | {r.get('img_per_s', '-')} | {sp} | {r.get('vram_MB', '-')} |")
    md.append("")
    if base.get("img_per_s"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for m in sorted({r["method"] for r in bench} - {"identity"}):
            rs = sorted([r for r in bench if r["method"] == m], key=lambda r: float(r["keep"]))
            ks = [float(r["keep"]) for r in rs] + [1.0]
            ax[0].plot(ks, [float(r["GMACs"]) for r in rs] + [float(base["GMACs"])], "o-", label=NAMES[m], color=COLORS.get(m))
            ax[1].plot(ks, [float(r["img_per_s"]) for r in rs] + [float(base["img_per_s"])], "o-", label=NAMES[m], color=COLORS.get(m))
        ax[0].set(xlabel="Tỉ lệ token giữ lại", ylabel="GMACs", title="Chi phí tính toán")
        ax[1].set(xlabel="Tỉ lệ token giữ lại", ylabel="ảnh/giây", title="Throughput (T4)")
        ax[1].legend(fontsize=7)
        for x in ax:
            x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(R, "fig_efficiency.png"), dpi=150); plt.close(fig)

# ---------------- accuracy runs, grouped by dataset
runs = {}
for f in glob.glob(os.path.join(R, "*.json")):
    r = json.load(open(f)); a = r["args"]
    runs.setdefault(a["dataset"], {}).setdefault((a["reducer"], float(a["ratio"])), []).append(r)

summary = []
for ds in sorted(runs):
    groups = runs[ds]
    a0 = next(iter(groups.values()))[0]["args"]
    md += [f"## Độ chính xác: {ds} (train {a0.get('train_subset') or 'full'} ảnh, test {a0.get('test_subset') or 'full'} ảnh, "
           f"{a0['epochs']} epoch)", "",
           "| Phương pháp | Giữ | Top-1 (%) mean ± std | n seed | GMACs | Tham số module | Train (phút) |",
           "|---|---|---|---|---|---|---|"]
    keys = sorted(groups, key=lambda k: (-k[1], ORDER.index(k[0]) if k[0] in ORDER else 99))
    for m, q in keys:
        rs = groups[(m, q)]
        mu, sd = mstd([r["top1"] for r in rs])
        tmin = st.mean(sum(e.get("train_min", 0) for e in r["log"]) for r in rs)
        seeds = sorted(r["args"]["seed"] for r in rs)
        md.append(f"| {NAMES.get(m, m)} | {q:.0%} | {mu:.2f} ± {sd:.2f} | {len(rs)} | {rs[0]['GMACs']:.3f} | "
                  f"{rs[0]['reducer_params']:,} | {tmin:.1f} |")
        summary.append({"dataset": ds, "method": m, "keep": q, "top1_mean": round(mu, 2), "top1_std": round(sd, 2),
                        "n_seeds": len(rs), "seeds": " ".join(map(str, seeds)), "GMACs": round(rs[0]["GMACs"], 3),
                        "reducer_params": rs[0]["reducer_params"], "train_min": round(tmin, 1)})
    md.append("")

    # accuracy vs keep ratio, one line per method, baseline at keep=1.0
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if ("identity", 1.0) in groups:
        b_mu, b_sd = mstd([r["top1"] for r in groups[("identity", 1.0)]])
        ax.axhline(b_mu, color="k", lw=1, ls=":", label=f"Vim-Ti 100% ({b_mu:.1f}%)")
    for m in [x for x in ORDER if x != "identity"]:
        pts = sorted((q, mstd([r["top1"] for r in groups[(mm, q)]])) for (mm, q) in groups if mm == m)
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts],
                    fmt=STYLE.get(m, "o-"), color=COLORS.get(m), label=NAMES[m], capsize=3, ms=5)
    ax.set(xlabel="Tỉ lệ token giữ lại", ylabel="Top-1 (%)", title=f"Độ chính xác theo mức giữ token: {ds}")
    ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(R, f"fig_acc_{ds}.png"), dpi=150); plt.close(fig)
    md += [f"![acc](fig_acc_{ds}.png)", ""]

if summary:
    with open(os.path.join(R, "results_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

open(os.path.join(R, "report.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
