"""Separate 'pattern shift' from 'this model is simply damaged'.

The IMM matrix shows Delta = +.034 (GREEN), but the T-frag TRAINING row is bad
in every column, not just off-diagonal. If fine-tuning on mostly-erased text
damages the encoder outright, that row drags down the off-diagonal mean of
every other column and inflates Delta mechanically. Test: recompute Delta with
that row excluded, and compare against FMM (where no damage appears) and
against the frozen-feature baseline.
"""
import json, sys
from glob import glob
from pathlib import Path
import numpy as np

REG = ["U-lo", "U-hi", "T-frag", "NV-frag"]

def load(root, proto):
    cells = {}          # (train, test) -> list of acc2
    complete = {}       # train -> list of unmasked acc2
    for f in glob(str(Path(root) / "MOSI" / proto / "*.json")):
        r = json.load(open(f))
        tr = r["config"]["train_regime"]
        for te, c in r["cells"].items():
            if te == "NONE":
                complete.setdefault(tr, []).append(c["acc2"])
            else:
                cells.setdefault((tr, te), []).append(c["acc2"])
    return cells, complete

def delta(cells, exclude_rows=()):
    rows = [r for r in REG if r not in exclude_rows]
    per = {}
    for te in REG:
        if te not in rows:
            continue                      # no matched cell available
        m = cells.get((te, te))
        if not m:
            continue
        others = [np.mean(cells[(tr, te)]) for tr in rows
                  if tr != te and (tr, te) in cells]
        if not others:
            continue
        per[te] = float(np.mean(m) - np.mean(others))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per

for label, root in [("frozen", "results"), ("fine-tuned BERT", "results_bert")]:
    for proto in ["IMM", "FMM"]:
        cells, comp = load(root, proto)
        if not cells:
            continue
        d_all, per_all = delta(cells)
        d_ex, per_ex = delta(cells, exclude_rows=("T-frag",))
        print(f"\n===== {label} / {proto} =====")
        print(f"  Delta (all rows)          = {d_all:+.4f}")
        print(f"  Delta (excl. T-frag row)  = {d_ex:+.4f}")
        if comp:
            print("  complete-input acc2 by TRAIN regime (encoder health):")
            for tr in REG:
                if tr in comp:
                    v = comp[tr]
                    print(f"     {tr:8s} {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")
