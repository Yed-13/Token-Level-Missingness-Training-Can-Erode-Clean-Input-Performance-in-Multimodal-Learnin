"""Evaluation metrics, following the conventions used by MMSA and the
missing-modality MSA literature so our matched-protocol diagonal is
directly comparable to published numbers."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score


def _acc_bins(pred, truth, edges):
    pb = np.digitize(pred, edges)
    tb = np.digitize(truth, edges)
    return float((pb == tb).mean())


def regression_metrics(pred: np.ndarray, truth: np.ndarray,
                       label_range: float = 3.0) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    R = float(label_range)
    p = np.clip(pred, -R, R)
    t = np.clip(truth, -R, R)

    out = {
        "mae": float(np.abs(p - t).mean()),
        "corr": float(pearsonr(p, t)[0]) if p.std() > 1e-8 else 0.0,
    }

    # Acc-2 / F1: standard convention excludes truly-neutral (==0) samples.
    nz = t != 0
    if nz.sum() > 0:
        pb, tb = p[nz] > 0, t[nz] > 0
        out["acc2"] = float((pb == tb).mean())
        out["f1"] = float(f1_score(tb, pb, average="weighted", zero_division=0))
    else:
        out["acc2"] = out["f1"] = float("nan")

    # Acc-2 including neutrals as non-negative (the "has0" variant).
    out["acc2_has0"] = float(((p >= 0) == (t >= 0)).mean())

    if R == 3.0:
        out["acc7"] = float((np.round(p) == np.round(t)).mean())
        out["acc5"] = _acc_bins(p, t, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    else:  # SIMS, labels in [-1, 1]
        out["acc5"] = _acc_bins(p, t, np.array([-0.6, -0.2, 0.2, 0.6]))
        out["acc3"] = _acc_bins(p, t, np.array([-0.2, 0.2]))
    return out


PRIMARY = "acc2"
HIGHER_BETTER = {"acc2": True, "f1": True, "acc7": True, "acc5": True,
                 "acc3": True, "acc2_has0": True, "corr": True, "mae": False}
