"""Strict, seed-paired analysis of the prespecified review controls.

No incomplete matrix is summarized. Raw records and archived SD fields remain
unchanged; this revision reports sample SD and pointwise paired t intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t, shapiro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from review_common import SUITES


def describe(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite seed-level observations")
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    margin = float(t.ppf(.975, len(a)-1) * sd / np.sqrt(len(a)))
    normality = None
    if len(a) >= 3 and np.ptp(a) > 0:
        test = shapiro(a)
        normality = dict(W=float(test.statistic), p=float(test.pvalue))
    return dict(n=len(a), values=a.tolist(), mean=mean, sd=sd,
                median=float(np.median(a)), minimum=float(a.min()), maximum=float(a.max()),
                ci95=[mean-margin, mean+margin], shapiro_diagnostic=normality)


def condition_values(records, metric):
    cells = [r["cells"] for r in records]
    out = {"NONE": [r["NONE"][metric] for r in cells]}
    for protocol in ("IMM", "FMM"):
        for regime in SUITES:
            out[f"{protocol}/{regime}"] = [r[protocol][regime][metric] for r in cells]
    return out


def summarize_group(records):
    output = {m: {k: describe(v) for k, v in condition_values(records, m).items()}
              for m in ("acc2", "mae")}
    # Sensitivity analysis, not an estimate of deployment prevalence.
    output["mixture"] = {}
    for metric in ("acc2", "mae"):
        values = condition_values(records, metric)
        clean = np.asarray(values["NONE"])
        missing = (np.asarray(values["IMM/T-frag"]) + np.asarray(values["FMM/T-frag"])) / 2
        output["mixture"][metric] = {f"{p:.1f}": describe(p*clean + (1-p)*missing)
                                       for p in np.linspace(0, 1, 11)}
    return output


def paired_difference(left, right, metric="acc2", condition="NONE"):
    scale = 100 if metric == "acc2" else 1
    a = np.asarray(left[metric][condition]["values"])
    b = np.asarray(right[metric][condition]["values"])
    return describe(scale*(a-b))


def collect(paths):
    records, provenance, suite = [], [], None
    for path in paths:
        content = path.read_bytes()
        record = json.loads(content)
        if record.get("suite") is not None:
            if suite is not None and suite != record["suite"]:
                raise ValueError("Unmatched deterministic evaluation suites")
            suite = record["suite"]
        records.append(record)
        provenance.append(dict(path=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(content).hexdigest()))
    return records, provenance


def common_summary(paths):
    records, provenance = collect(paths)
    expected = {(w,p,r,s) for w in (0.0,0.5) for p in ("IMM","FMM")
                for r in ("U-lo","T-frag") for s in range(5)}
    indexed = {}
    for r in records:
        c = r["config"]
        key = (c["lambda_recon"],c["protocol"],c["train_regime"],c["seed"])
        if key in indexed:
            raise ValueError("Duplicate common-evaluation condition")
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError("Incomplete/unexpected common evaluation matrix")
    groups = {}
    for w in (0.0,0.5):
        for p in ("IMM","FMM"):
            for regime in ("U-lo","T-frag"):
                groups[f"{w}/{p}/{regime}"] = summarize_group([indexed[w,p,regime,s] for s in range(5)])
    return dict(n_runs=40, groups=groups, suite=records[0]["suite"], provenance=provenance)


def selection_summary(paths):
    records, provenance = collect(paths)
    expected = {(p,r,s) for p,r in (("IMM","U-lo"),("IMM","T-frag"),
                                   ("FMM","U-lo"),("FMM","T-frag"),("IMM","NONE"))
                for s in range(5)}
    indexed, suite, source, shared_config, runtime = {}, None, None, None, None
    for r in records:
        c = r["config"]
        fixed = {k:v for k,v in c.items() if k not in ("protocol","train_regime","seed")}
        if shared_config is not None and shared_config != fixed:
            raise ValueError("Unmatched training configurations")
        shared_config = fixed
        if (c["lambda_recon"], c["validation_regime"], c["amp"], c["max_epochs"]) != (0.5,"NONE","bf16",40):
            raise ValueError("Unexpected selection configuration")
        key = (c["protocol"],c["train_regime"],c["seed"])
        if key in indexed:
            raise ValueError("Duplicate selection condition")
        if r["epochs_run"] != 40 or r["early_stopping"] or r["n_train"] != 1284:
            raise ValueError("Unmatched training budget")
        if source is not None and source != r["source_sha256"]:
            raise ValueError("Unmatched review training source")
        source = r["source_sha256"]
        current_runtime = (r["torch"], r["attention"])
        if runtime is not None and runtime != current_runtime:
            raise ValueError("Unmatched training runtimes")
        runtime = current_runtime
        best_value, best_epoch = float("inf"), None
        if [h["epoch"] for h in r["history"]] != list(range(1,41)):
            raise ValueError("Incomplete epoch history")
        for h in r["history"]:
            value = h["clean_validation_mae"]
            if not np.isfinite(value):
                raise ValueError("Non-finite validation history")
            if value < best_value - 1e-5:
                best_value, best_epoch = value, h["epoch"]
        if r["selections"]["clean_best"]["epoch"] != best_epoch:
            raise ValueError("Checkpoint selection disagrees with validation history")
        if set(r["selections"]) != {"clean_best","fixed40"}:
            raise ValueError("Incomplete checkpoint selections")
        for mode, sel in r["selections"].items():
            if not 1 <= sel["epoch"] <= 40 or (mode == "fixed40" and sel["epoch"] != 40):
                raise ValueError("Invalid checkpoint epoch")
            current = sel["evaluation"]["suite"]
            if suite is not None and suite != current:
                raise ValueError("Unmatched selection test suite")
            suite = current
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError("Incomplete/unexpected selection matrix")
    groups, changes, anchors, contrasts = {}, {}, {}, {}
    for mode in ("clean_best","fixed40"):
        for p, regime in sorted({(p,r) for p,r,s in expected}):
            rows = [indexed[p,regime,s]["selections"][mode]["evaluation"] for s in range(5)]
            group = summarize_group(rows)
            group["selected_epochs"] = [indexed[p,regime,s]["selections"][mode]["epoch"] for s in range(5)]
            groups[f"{mode}/{p}/{regime}"] = group
        for metric in ("acc2","mae"):
            for p in ("IMM","FMM"):
                changes[f"{mode}/{p}/{metric}"] = paired_difference(
                    groups[f"{mode}/{p}/T-frag"], groups[f"{mode}/{p}/U-lo"], metric)
                for regime in ("U-lo","T-frag"):
                    anchors[f"{mode}/{p}/{regime}/{metric}"] = paired_difference(
                        groups[f"{mode}/{p}/{regime}"], groups[f"{mode}/IMM/NONE"], metric)
            contrasts[f"{mode}/{metric}"] = describe(
                np.asarray(changes[f"{mode}/IMM/{metric}"]["values"]) -
                np.asarray(changes[f"{mode}/FMM/{metric}"]["values"]))
    return dict(n_trajectories=25, n_selections=50, groups=groups, changes=changes,
                clean_anchor_changes=anchors, protocol_contrasts=contrasts,
                suite=suite, source_sha256=source, provenance=provenance,
                environment=dict(torch=runtime[0],attention=runtime[1],device="cuda"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("common","selection"))
    args = parser.parse_args()
    pattern = "common/*.json" if args.phase == "common" else "selection/*/result.json"
    paths = sorted((ROOT / "results_review_cuda").glob(pattern))
    summary = common_summary(paths) if args.phase == "common" else selection_summary(paths)
    summary["statistics"] = {"sd": "sample, ddof=1", "ci": "pointwise paired t, 95%",
        "replication_unit": "training seed", "multiplicity_adjustment": "none; no significance claims",
        "scope": "conditional on fixed dataset; normality and seed independence assumed",
        "normality_note": "Shapiro values are diagnostics; five seeds cannot establish normality"}
    dest = ROOT / f"analysis/data/review_{args.phase}.json"
    dest.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
