"""Build cross-pattern transfer matrices and compute the robustness gap.

THE DEFINITION THAT MATTERS
---------------------------
Let M[i][j] be the test metric for a model TRAINED under regime i and
EVALUATED under regime j.

A naive "gap" compares cells across a ROW (one model, different test regimes).
That is wrong: it measures how intrinsically hard each regime is, not how much
train/test mismatch costs. T-frag is harder than U-lo for every model, because
text is usually gone.

The difficulty-controlled gap is COLUMN-wise -- fix the test regime, vary what
the model was trained on:

    Delta_j       = M[j][j] - mean_{i != j} M[i][j]      (per test regime)
    Delta         = mean_j Delta_j                        (headline number)
    WorstDrop_j   = M[j][j] - min_i M[i][j]

Delta > 0 means training on the target regime genuinely helps, i.e. pattern
shift costs real performance and the premise of this project holds.
Delta ~ 0 means the training regime barely matters and the premise fails.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

from ..data.masking import REGIME_NAMES


def load(results_dir: str, dataset: str, protocol: str = "IMM"):
    """-> {method: {train_regime: {test_regime: [values per seed]}}}"""
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta = defaultdict(set)
    for f in sorted(glob(str(Path(results_dir) / dataset / protocol / "*.json"))):
        r = json.load(open(f))
        m, tr, seed = r["config"]["method"], r["config"]["train_regime"], r["config"]["seed"]
        meta[m].add(seed)
        for te, cell in r["cells"].items():
            store[m][tr][te].append((seed, cell))
    return store, {k: sorted(v) for k, v in meta.items()}


def matrix(store, method: str, metric: str = "acc2"):
    """Returns mean matrix, std matrix, and per-seed count, over REGIME_NAMES."""
    n = len(REGIME_NAMES)
    mean = np.full((n, n), np.nan)
    std = np.full((n, n), np.nan)
    cnt = np.zeros((n, n), dtype=int)
    for i, tr in enumerate(REGIME_NAMES):
        for j, te in enumerate(REGIME_NAMES):
            vals = [c[metric] for _, c in store[method].get(tr, {}).get(te, [])]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                mean[i, j], std[i, j], cnt[i, j] = np.mean(vals), np.std(vals), len(vals)
    return mean, std, cnt


def gaps(mean: np.ndarray, higher_better: bool = True):
    n = mean.shape[0]
    per_regime, worst = {}, {}
    for j, te in enumerate(REGIME_NAMES):
        matched = mean[j, j]
        others = [mean[i, j] for i in range(n) if i != j and not np.isnan(mean[i, j])]
        if np.isnan(matched) or not others:
            continue
        sign = 1.0 if higher_better else -1.0
        per_regime[te] = sign * (matched - float(np.mean(others)))
        worst[te] = sign * (matched - (min(others) if higher_better else max(others)))
    delta = float(np.mean(list(per_regime.values()))) if per_regime else float("nan")
    return delta, per_regime, worst


def render(mean, std, cnt, method, metric):
    lines = [f"\n=== {method}  [{metric}]  rows = TRAIN regime, cols = TEST regime ==="]
    lines.append("            " + "".join(f"{r:>16s}" for r in REGIME_NAMES))
    for i, tr in enumerate(REGIME_NAMES):
        row = f"  {tr:<10s}"
        for j in range(len(REGIME_NAMES)):
            if np.isnan(mean[i, j]):
                row += f"{'--':>16s}"
            else:
                tag = "*" if i == j else " "
                row += f"{mean[i,j]:>10.4f}±{std[i,j]:.3f}{tag}"
        lines.append(row)
    lines.append(f"  (* = matched cell; n seeds per cell: "
                 f"{cnt.min()}-{cnt.max()})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="MOSI")
    ap.add_argument("--protocol", default="IMM")
    ap.add_argument("--metric", default="acc2")
    a = ap.parse_args()

    store, meta = load(a.results, a.dataset, a.protocol)
    if not store:
        print(f"no results under {a.results}/{a.dataset}/{a.protocol}")
        return

    print(f"dataset={a.dataset} protocol={a.protocol} metric={a.metric}")
    summary = {}
    for method in sorted(store):
        mean, std, cnt = matrix(store, method, a.metric)
        if np.isnan(np.diag(mean)).all():
            continue
        print(render(mean, std, cnt, method, a.metric))
        delta, per_regime, worst = gaps(mean, higher_better=(a.metric != "mae"))
        summary[method] = {"delta": delta, "per_regime": per_regime, "worst": worst,
                           "seeds": meta[method]}
        print(f"  Delta (difficulty-controlled, column-wise) = {delta:+.4f}")
        print("    per test regime: " +
              "  ".join(f"{k}={v:+.4f}" for k, v in per_regime.items()))
        print("    worst drop:      " +
              "  ".join(f"{k}={v:+.4f}" for k, v in worst.items()))

    print("\n=== VERDICT ===")
    for m, s in summary.items():
        d = s["delta"]
        verdict = ("GREEN: pattern shift costs real performance" if d >= 0.03 else
                   "AMBER: small but present" if d >= 0.015 else
                   "RED: training regime barely matters -- premise fails")
        print(f"  {m:16s} Delta={d:+.4f}  {verdict}")
    Path(a.results, f"summary_{a.dataset}_{a.protocol}_{a.metric}.json").write_text(
        json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
