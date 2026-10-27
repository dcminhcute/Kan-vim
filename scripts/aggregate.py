"""results/*.json -> markdown table of mean +- std over seeds.  python scripts/aggregate.py results"""
import collections
import glob
import json
import statistics as st
import sys

rows = collections.defaultdict(list)
for f in glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'results'}/*.json"):
    r = json.load(open(f)); a = r["args"]
    rows[(a["dataset"], a["backbone"], a["reducer"], a["ratio"])].append(r)

def ms(v):
    v = [x for x in v if x is not None]
    return "-" if not v else f"{st.mean(v):.2f} ± {st.pstdev(v) if len(v) > 1 else 0:.2f}"

print("| dataset | backbone | method | keep | n | top1 | GMACs | red.params | img/s | VRAM MB |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(rows):
    rs = rows[k]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(rs)} | {ms([r['top1'] for r in rs])} | "
          f"{ms([r['GMACs'] for r in rs])} | {rs[0]['reducer_params']} | "
          f"{ms([r.get('throughput') for r in rs])} | {ms([r.get('vram_MB') for r in rs])} |")
