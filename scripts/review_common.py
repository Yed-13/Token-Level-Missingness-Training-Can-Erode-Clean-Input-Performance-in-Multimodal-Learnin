"""Shared deterministic review evaluation; uses the archived masking implementation."""
import hashlib
import json
from pathlib import Path

SUITES = ("U-lo", "T-frag", "T@0.0", "T@0.2", "T@0.4", "T@0.6", "T@0.8")
REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def suite_spec():
    return {"dataset": "MOSI", "partition": "test", "batch_size": 32,
            "shuffle": False, "protocols": ["IMM", "FMM"],
            "regimes": list(SUITES), "repeats": REPEATS,
            "clean": "NONE", "seed_base": 20260915,
            "source_hashes": {p: sha256(ROOT / p) for p in
                              ("parm/train.py", "parm/seeding.py", "parm/data/masking.py",
                               "parm/data/datasets.py", "scripts/review_common.py")}}


def common_evaluate(model, loader, cfg, label_range, device):
    from parm.train import evaluate
    if cfg.dataset != "MOSI" or cfg.batch_size != 32 or device != "cuda":
        raise ValueError("Review evaluation requires MOSI, batch 32 and CUDA")
    cells = {"NONE": evaluate(model, loader, cfg, "NONE", label_range, device,
                              repeats=1, protocol="IMM")}
    for protocol in ("IMM", "FMM"):
        cells[protocol] = {regime: evaluate(model, loader, cfg, regime, label_range,
                                          device, repeats=REPEATS, protocol=protocol)
                           for regime in SUITES}
    return {"suite": suite_spec(), "cells": cells}


def write_new_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(encoded)
