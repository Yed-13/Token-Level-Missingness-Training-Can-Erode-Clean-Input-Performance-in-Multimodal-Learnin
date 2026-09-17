"""Estimate per-modality informativeness alpha_k.

alpha_k is the validation degradation caused by removing modality k, measured
on a probe trained on COMPLETE data. Computed once per dataset, cached to
datasets/alpha_<DS>.json, and reported as a table in the paper.

Uses the train/valid splits only -- never test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.datasets import collate, load_dataset
from .data.masking import MODALITIES
from .metrics import regression_metrics
from .models.backbones.fusion import MultimodalFusion
from .train import pick_device, to_device


def probe_eval(model, loader, device, drop=None, label_range=3.0):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for feats, valid, y, _, _b in loader:
            feats, valid = to_device(feats, valid, device)
            if drop is not None:
                feats = dict(feats); valid = dict(valid)
                feats[drop] = torch.zeros_like(feats[drop])
                valid[drop] = torch.zeros_like(valid[drop])
            P.append(model(feats, valid).float().cpu().numpy())
            T.append(y.numpy())
    return regression_metrics(np.concatenate(P), np.concatenate(T), label_range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MOSI")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--d", type=int, default=64)
    a = ap.parse_args()

    device = pick_device()
    splits, dims, seqs, label_range = load_dataset(a.dataset, a.root)
    tr = DataLoader(splits["train"], batch_size=32, shuffle=True, collate_fn=collate)
    va = DataLoader(splits["valid"], batch_size=64, shuffle=False, collate_fn=collate)

    per_seed = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        model = MultimodalFusion(dims, seqs, d=a.d, use_recon=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(a.epochs):
            model.train()
            for feats, valid, y, _, _b in tr:
                feats, valid = to_device(feats, valid, device)
                loss = F.l1_loss(model(feats, valid), y.to(device))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            m = probe_eval(model, va, device, None, label_range)["mae"]
            if m < best - 1e-5:
                best, bad = m, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 6: break
        model.load_state_dict(best_state)

        full = probe_eval(model, va, device, None, label_range)
        row = {"seed": seed, "full_mae": full["mae"], "full_acc2": full["acc2"]}
        for k in MODALITIES:
            d = probe_eval(model, va, device, k, label_range)
            row[f"drop_{k}_mae"] = d["mae"]
            row[f"deg_{k}"] = d["mae"] - full["mae"]
        per_seed.append(row)
        print(f"  seed {seed}: full MAE {full['mae']:.4f} acc2 {full['acc2']:.4f} | "
              + " ".join(f"-{k}:{row[f'deg_{k}']:+.4f}" for k in MODALITIES))

    raw = {k: float(np.mean([r[f"deg_{k}"] for r in per_seed])) for k in MODALITIES}
    s = sum(max(v, 0.0) for v in raw.values()) or 1.0
    alpha = {k: max(raw[k], 0.0) / s for k in MODALITIES}

    out = {"dataset": a.dataset, "alpha": alpha, "raw_degradation": raw,
           "per_seed": per_seed, "n_seeds": a.seeds}
    Path(a.root, f"alpha_{a.dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{a.dataset} alpha: " + "  ".join(f"{k}={alpha[k]:.3f}" for k in MODALITIES))
    print(f"raw MAE degradation: " + "  ".join(f"{k}={raw[k]:+.4f}" for k in MODALITIES))


if __name__ == "__main__":
    main()
