"""Unpaired sensitivity analysis of archived runs, not a stable-seed rerun.

Intervals assume independent groups as a working model. Shared initialization
seeds can induce dependence; this calculation does not repair old selection.
"""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ("analysis" if (ROOT / "analysis/studies.py").exists() else "paper")
sys.path.insert(0, str(ANALYSIS))
from studies import load, select, clean_acc
from summarize_review import describe


def unpaired_change(benign, scarce):
    a, b = np.asarray(benign, float), np.asarray(scarce, float)
    if min(len(a), len(b)) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Two finite runs per group required")
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = float(np.sqrt(va+vb))
    delta = float(b.mean()-a.mean())
    df = float((va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))) if se else None
    margin = float(t.ppf(.975, df)*se) if se else 0.0
    return dict(benign=describe(a), text_scarce=describe(b), mean=delta,
                standard_error=se, degrees_of_freedom=df,
                ci95=[delta-margin, delta+margin])


def summarize():
    specs = [("main","BERT",ds,p,"full") for ds in ("MOSI","MOSEI","SIMS")
             for p in ("IMM","FMM")]
    specs += [("roberta","RoBERTa",ds,p,n) for ds,n in (("MOSI","full"),("MOSEI",1284))
              for p in ("IMM","FMM")]
    specs += [("unk","BERT/UNK","MOSI","IMM","full")]
    records = []
    for study,encoder,dataset,proto,n in specs:
        data = select(load(study),dataset=dataset,protocol=proto,n_train=n)
        groups = {regime:sorted([r for r in data if r["config"]["train_regime"]==regime],
                               key=lambda r:r["config"]["seed"]) for regime in ("U-lo","T-frag")}
        for group in groups.values():
            seeds = [r["config"]["seed"] for r in group]
            expected = set(range(5 if study=="main" else 3))
            if len(seeds) != len(set(seeds)) or set(seeds) != expected:
                raise ValueError("Duplicate or incomplete archived condition")
        result = unpaired_change(*[[100*clean_acc(r) for r in groups[g]] for g in ("U-lo","T-frag")])
        records.append(dict(study=study,encoder=encoder,dataset=dataset,protocol=proto,
                            training_size=n, seeds={g:[r["config"]["seed"] for r in rs]
                                                   for g,rs in groups.items()}, change=result,
                            provenance=[dict(path=str(Path(r["_file"]).relative_to(ROOT)),
                                             sha256=hashlib.sha256(Path(r["_file"]).read_bytes()).hexdigest())
                                        for rs in groups.values() for r in rs]))
    return dict(analysis="Welch unpaired sensitivity; scarce minus benign, percentage points",
                assumptions="Independent groups as a working model; approximate normality; fixed dataset; pointwise intervals, no multiplicity adjustment or significance claims",
                caveat="Historical validation masks were not matched. Shared seed labels may induce covariance. These are exploratory comparisons, not stable-seed replications.",
                records=records)


if __name__ == "__main__":
    path = ANALYSIS / "data/archived_uncertainty.json"
    path.write_text(json.dumps(summarize(),indent=2,allow_nan=False)+"\n")
    print(path)
