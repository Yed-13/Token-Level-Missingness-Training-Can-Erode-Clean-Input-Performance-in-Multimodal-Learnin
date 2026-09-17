"""Dose-response: complete-input accuracy vs text-erasure rate during training.

Holds the corpus fixed, so it isolates erasure rate from the dataset-size
confound that complicates the cross-dataset alpha comparison.
"""
import json, re
from glob import glob
from pathlib import Path
import numpy as np

for ds in ["MOSI", "SIMS", "MOSEI"]:
    rows = {}
    for f in glob(f"results_sweep/{ds}/*/recon__train-T@*__seed*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        m = re.search(r"train-T@([0-9.]+)__seed", Path(f).name)
        if not m or "NONE" not in r.get("cells", {}):
            continue
        rate, proto = float(m.group(1)), r["config"]["protocol"]
        rows.setdefault(proto, {}).setdefault(rate, []).append(
            r["cells"]["NONE"]["acc2"])
    if not rows:
        continue
    print(f"\n=== {ds}: complete-input acc2 by training text-erasure rate ===")
    rates = sorted({x for p in rows.values() for x in p})
    hdr = "  rate  " + "".join(f"{p:>16s}" for p in sorted(rows))
    print(hdr)
    for rate in rates:
        line = f"  {rate:<6.1f}"
        for p in sorted(rows):
            v = rows[p].get(rate)
            line += f"{np.mean(v):>10.4f}±{np.std(v):.3f}" if v else f"{'--':>16s}"
        print(line)
    for p in sorted(rows):
        vals = [(r, np.mean(v)) for r, v in sorted(rows[p].items())]
        if len(vals) > 1:
            base = vals[0][1]
            worst = min(vals, key=lambda x: x[1])
            print(f"  {p}: baseline(rate 0)={base:.4f}  worst={worst[1]:.4f} "
                  f"at rate {worst[0]}  drop={base-worst[1]:+.4f}")
