"""Representation-collapse diagnostics on trained models.

Tests the mechanism directly instead of inferring it from accuracy. A damaged
encoder should show high pairwise cosine similarity and low uncentred
effective rank.
"""
import json
from glob import glob
import numpy as np

for ds in ["MOSI", "SIMS", "MOSEI"]:
    for proto in ["IMM", "FMM"]:
        got = {}
        for f in glob(f"results_mech/{ds}/{proto}/*.json"):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            d = r.get("diagnostics", {}).get("text_repr")
            if not d:
                continue
            got.setdefault(r["config"]["train_regime"], []).append(
                (d.get("cos_offdiag"), d.get("eff_rank_raw"),
                 d.get("eff_rank_centred"),
                 r["cells"].get("NONE", {}).get("acc2", float("nan"))))
        if not got:
            continue
        print(f"\n=== {ds} / {proto}: text-encoder representation health ===")
        print(f"  {'train regime':12s} {'cos_offdiag':>12s} {'eff_rank_raw':>13s}"
              f" {'eff_rank_cen':>13s} {'complete acc2':>14s}")
        for reg in ["U-lo", "U-hi", "T-frag", "NV-frag"]:
            if reg not in got:
                continue
            a = np.array(got[reg], dtype=float)
            print(f"  {reg:12s} {np.nanmean(a[:,0]):>12.4f} {np.nanmean(a[:,1]):>13.2f}"
                  f" {np.nanmean(a[:,2]):>13.2f} {np.nanmean(a[:,3]):>14.4f}")
        print("  (collapse => cos_offdiag toward 1.0 and eff_rank_raw toward 1)")
