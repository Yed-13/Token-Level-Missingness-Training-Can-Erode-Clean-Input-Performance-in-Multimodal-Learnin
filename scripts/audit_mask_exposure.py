"""Measure masking exposure on MOSI training lengths using the training operator."""
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parm.data.datasets import _valid_masks
from parm.data.masking import MODALITIES, apply_mask, sample_rates
from parm.data.download import SPECS, sha256_of


def run():
    source = ROOT / "datasets/MOSI/unaligned_50.pkl"
    digest = sha256_of(source)
    assert digest == SPECS["MOSI"]["sha256"], "Unexpected MOSI feature file"
    with source.open("rb") as handle:
        train = pickle.load(handle)["train"]
    n = len(train["text_bert"])
    masks = _valid_masks(train, n)
    del train
    valid = {k: torch.from_numpy(masks[k]) for k in MODALITIES}
    feats = {k: torch.ones(n, valid[k].shape[1], 1) for k in MODALITIES}
    rows = []
    for regime in ("U-lo", "T-frag"):
        for protocol in ("IMM", "FMM"):
            values = []
            for repeat in range(10):
                rng = np.random.default_rng(31000 + repeat)
                rates = torch.tensor(np.array([sample_rates(regime, rng) for _ in range(n)]),
                                     dtype=torch.float32)
                generator = torch.Generator().manual_seed(41000 + repeat)
                _, masked = apply_mask(feats, valid, rates, protocol, generator)
                text_absent = ~masked["t"].any(dim=1)
                text_intact = (masked["t"] == valid["t"]).all(dim=1)
                missing = torch.stack([~masked[k].any(dim=1) for k in MODALITIES], dim=1)
                assert (~missing.all(dim=1)).all()
                values.append([float(text_intact.float().mean()),
                               float(text_absent.float().mean()),
                               float(missing.sum(dim=1).float().mean())])
            mean = np.mean(values, axis=0)
            rows.append(dict(regime=regime, protocol=protocol,
                             text_intact=float(mean[0]), text_absent=float(mean[1]),
                             reconstruction_pairs_per_example=float(mean[2]),
                             repeat_values=values))
    result = dict(dataset="MOSI", partition="train", n=n, repeats=10,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  source_sha256=digest, rate_seed_base=31000, mask_seed_base=41000,
                  operator_sha256=hashlib.sha256((ROOT / "parm/data/masking.py").read_bytes()).hexdigest(),
                  rows=rows)
    destination = ROOT / "analysis/data/masking_exposure.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
