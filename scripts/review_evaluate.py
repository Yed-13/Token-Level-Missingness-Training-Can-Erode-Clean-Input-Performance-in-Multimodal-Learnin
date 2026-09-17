"""Evaluate an existing selected CUDA checkpoint; never retrains a model."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from parm.data.datasets import collate, load_dataset
from parm.train import Cfg, METHODS, build_model, limit_threads
from review_common import common_evaluate, sha256, write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; local evaluation is disabled")
    limit_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Cfg(**{k: v for k, v in state["config"].items()
                 if k in {f.name for f in fields(Cfg)}})
    splits, dims, seqs, label_range = load_dataset(cfg.dataset, cfg.root)
    loader = DataLoader(splits["test"], batch_size=32, shuffle=False,
                        collate_fn=collate, num_workers=0)
    model, _ = build_model(cfg, METHODS[cfg.method], dims, seqs, None, "cuda")
    model.load_state_dict(state["state_dict"])
    result = common_evaluate(model, loader, cfg, label_range, "cuda")
    original = json.loads(args.checkpoint.with_suffix(".json").read_text())
    for metric in ("acc2", "mae"):
        if abs(result["cells"]["NONE"][metric] - original["cells"]["NONE"][metric]) > 1e-5:
            raise ValueError(f"Clean checkpoint reproduction failed: {metric}")
    result.update(config=state["config"], best_epoch=state["best_epoch"],
                  checkpoint=str(args.checkpoint.relative_to(ROOT)),
                  checkpoint_sha256=sha256(args.checkpoint),
                  evaluator_sha256=sha256(__file__), device="cuda", torch=torch.__version__)
    write_new_json(args.output, result)
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
