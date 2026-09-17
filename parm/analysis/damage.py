"""Cross-dataset summary: is the IMM Delta real shift, or encoder damage?"""
import json
from glob import glob
from pathlib import Path
import numpy as np

REG = ["U-lo", "U-hi", "T-frag", "NV-frag"]

def load(root, ds, proto):
    cells, comp = {}, {}
    for f in glob(str(Path(root) / ds / proto / "*.json")):
        r = json.load(open(f))
        tr = r["config"]["train_regime"]
        for te, c in r["cells"].items():
            (comp if te == "NONE" else cells).setdefault(
                tr if te == "NONE" else (tr, te), []).append(c["acc2"])
    return cells, comp

def delta(cells, exclude=()):
    rows = [r for r in REG if r not in exclude]
    per = {}
    for te in rows:
        if (te, te) not in cells:
            continue
        others = [np.mean(cells[(tr, te)]) for tr in rows
                  if tr != te and (tr, te) in cells]
        if others:
            per[te] = float(np.mean(cells[(te, te)]) - np.mean(others))
    return float(np.mean(list(per.values()))) if per else float("nan")

print(f"{'dataset':8s} {'proto':5s} {'D(all)':>8s} {'D(-Tfrag)':>10s} "
      f"{'complete acc2 by train regime':>44s}")
print("-" * 82)
for ds in ["MOSI", "SIMS"]:
    for proto in ["IMM", "FMM"]:
        cells, comp = load("results_bert", ds, proto)
        if not cells:
            continue
        best = max(np.mean(v) for v in comp.values()) if comp else float("nan")
        cstr = "  ".join(
            f"{r.split('-')[0][:4]}:{np.mean(comp[r]):.3f}" for r in REG if r in comp)
        drop = (best - np.mean(comp["T-frag"])) if "T-frag" in comp else float("nan")
        print(f"{ds:8s} {proto:5s} {delta(cells):+8.4f} {delta(cells,('T-frag',)):+10.4f} "
              f"   {cstr}")
        print(f"{'':14s} -> T-frag encoder damage vs best regime: {drop*100:+.1f} pts")
