"""Do any of the three candidate mitigations prevent the encoder damage?

Baseline is the plain IMM run. A mitigation works if complete-input accuracy
under the text-scarce (T-frag) training regime recovers toward the level of a
benign regime such as U-lo.
"""
import json
from glob import glob
import numpy as np

SETUPS = [
    ("baseline IMM",        "results_bert",              "IMM"),
    ("M1 train-FMM/test-IMM", "results_mitig_crossproto", "FMM"),
    ("M2 freeze 6 layers",  "results_mitig_freeze",      "IMM"),
    ("M3 bert-lr 5e-6",     "results_mitig_lowlr",       "IMM"),
]

for ds in ["MOSI", "SIMS"]:
    print(f"\n=== {ds}: complete-input acc2 by training regime ===")
    print(f"  {'setup':24s} {'U-lo':>14s} {'T-frag':>14s} {'damage':>10s}")
    for name, root, proto in SETUPS:
        got = {}
        for f in glob(f"{root}/{ds}/{proto}/recon__train-*__seed*.json"):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "NONE" not in r.get("cells", {}):
                continue
            got.setdefault(r["config"]["train_regime"], []).append(
                r["cells"]["NONE"]["acc2"])
        if "U-lo" not in got or "T-frag" not in got:
            continue
        u, t = np.mean(got["U-lo"]), np.mean(got["T-frag"])
        print(f"  {name:24s} {u:>8.4f}±{np.std(got['U-lo']):.3f} "
              f"{t:>8.4f}±{np.std(got['T-frag']):.3f} {t-u:>+10.4f}")
    print("  (damage = T-frag minus U-lo; closer to 0 is better)")
