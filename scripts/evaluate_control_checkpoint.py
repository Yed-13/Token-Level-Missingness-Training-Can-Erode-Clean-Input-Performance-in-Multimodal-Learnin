"""Evaluate one selected checkpoint on both fixed missingness protocols."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parm.data.datasets import collate, load_dataset
from parm.data.masking import REGIME_NAMES
from parm.train import BERT_FOR, Cfg, METHODS, build_model, evaluate, limit_threads, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    output = args.checkpoint.with_suffix(".common_eval.json")
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    allowed = {field.name for field in fields(Cfg)}
    cfg = Cfg(**{k: v for k, v in state["config"].items() if k in allowed})
    if cfg.method != "recon":
        raise ValueError("This evaluator currently supports the reconstruction control only")
    limit_threads(4)
    device = pick_device()
    name = BERT_FOR[cfg.dataset] if cfg.bert_name == "auto" else cfg.bert_name
    tokenizer = name if not name.startswith("bert-base") else ""
    splits, dims, seqs, label_range = load_dataset(cfg.dataset, cfg.root, tokenizer=tokenizer)
    loader = DataLoader(splits["test"], batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    model, _ = build_model(cfg, METHODS[cfg.method], dims, seqs, None, device)
    model.load_state_dict(state["state_dict"])
    cells = {}
    for protocol in ("IMM", "FMM"):
        cells[protocol] = {}
        for regime in (*REGIME_NAMES, "NONE"):
            cells[protocol][regime] = evaluate(
                model, loader, cfg, regime, label_range, device,
                repeats=1 if regime == "NONE" else 5, protocol=protocol)
            print(protocol, regime, cells[protocol][regime], flush=True)
    result = dict(config=state["config"], best_epoch=state["best_epoch"],
                  evaluation_device=device, evaluation_torch=torch.__version__,
                  cells_by_protocol=cells,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
