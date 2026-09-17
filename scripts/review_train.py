"""Fixed-budget CUDA reconstruction study with paired checkpoint-selection rules.

Reuses model, corruption, objective and evaluation primitives without modifying
the archived implementation. Only clean validation selects the best checkpoint;
the second selection is the prespecified last epoch. No test-time selection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from parm.data.datasets import collate, load_dataset
from parm.data.masking import apply_mask
from parm.train import (Cfg, METHODS, autocast_ctx, batch_patterns, build_model,
                        evaluate, limit_threads, to_device)
from review_common import common_evaluate, sha256, write_new_json


def train(protocol, regime, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no local training")
    cfg = Cfg(dataset="MOSI", method="recon", protocol=protocol,
              train_regime=regime, seed=seed, text_encoder="bert",
              batch_size=32, max_epochs=40, lambda_recon=0.5, amp="bf16",
              num_workers=0, torch_threads=4, validation_regime="NONE",
              save_checkpoint=1, out="results_review_cuda/selection")
    output = ROOT / cfg.out / f"{protocol}-{regime}-seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    limit_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits, dims, seqs, label_range = load_dataset(cfg.dataset, cfg.root)
    order_generator = torch.Generator().manual_seed(1000 + seed)
    loaders = {name: DataLoader(partition, batch_size=32, shuffle=name == "train",
                               generator=order_generator if name == "train" else None,
                               collate_fn=collate, num_workers=0)
               for name, partition in splits.items()}
    model, _ = build_model(cfg, METHODS["recon"], dims, seqs, None, "cuda")
    optimizer = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")],
          "lr": cfg.lr},
         {"params": [p for n, p in model.named_parameters() if n.startswith("bert.")],
          "lr": cfg.bert_lr}], weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator().manual_seed(2000 + seed)
    best_mae, best_epoch, best_state = float("inf"), -1, None
    history = []
    started = time.time()
    for epoch in range(40):
        model.train()
        loss_sum = 0.0
        for feats, valid, y, _, bert in loaders["train"]:
            feats, valid = to_device(feats, valid, "cuda")
            y, bert = y.to("cuda"), bert.to("cuda")
            rates, _ = batch_patterns(regime, len(y), rng, cfg.granularity)
            fm, vm = apply_mask(feats, valid, rates.to("cuda"), protocol, mask_generator)
            with autocast_ctx(cfg, "cuda"):
                pred, parts = model(fm, vm, return_parts=True, bert=bert,
                                    orig_text_mask=valid["t"])
            parts = {k: ({kk: vv.float() for kk, vv in v.items()}
                         if isinstance(v, dict) else v.float()) for k, v in parts.items()}
            task = F.l1_loss(pred.float(), y)
            with torch.no_grad(), autocast_ctx(cfg, "cuda"):
                target = model.encode(feats, valid, bert, orig_text_mask=valid["t"])
            recon = model.recon_loss(parts, {k: v.float() for k, v in target.items()}, None)
            loss = task + cfg.lambda_recon * recon
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(task.detach()) * len(y)
        # Validation must not advance training/dropout/DataLoader RNG streams.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            metrics = evaluate(model, loaders["valid"], cfg, "NONE", label_range,
                               "cuda", repeats=1, protocol="IMM")
        value = metrics["mae"]
        if not np.isfinite(value):
            raise ValueError("Non-finite validation MAE")
        history.append(dict(epoch=epoch + 1, train_task=loss_sum / len(splits["train"]),
                            clean_validation_mae=value))
        if value < best_mae - 1e-5:
            best_mae, best_epoch = value, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch+1}/40 validation_mae={value:.6f}", flush=True)
    states = {"fixed40": (40, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}),
              "clean_best": (best_epoch, best_state)}
    selections = {}
    for selection, (epoch, state) in states.items():
        checkpoint = output / f"{selection}.pt"
        torch.save({"state_dict": state, "config": asdict(cfg), "best_epoch": epoch - 1},
                   checkpoint)
        model.load_state_dict(state)
        evaluation = common_evaluate(model, loaders["test"], cfg, label_range, "cuda")
        selections[selection] = dict(epoch=epoch, checkpoint=checkpoint.name,
                                     checkpoint_sha256=sha256(checkpoint), evaluation=evaluation)
        print("EVALUATED", selection, flush=True)
    record = dict(config=asdict(cfg), n_train=len(splits["train"]), epochs_run=40,
                  early_stopping=False, clean_best_mae=best_mae, history=history,
                  selections=selections, wall_seconds=time.time()-started,
                  source_sha256=sha256(__file__), torch=torch.__version__,
                  train_order_seed=1000 + seed, train_mask_seed=2000 + seed,
                  attention=model.bert.bert.config._attn_implementation)
    write_new_json(output / "result.json", record)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("IMM", "FMM"), required=True)
    parser.add_argument("--regime", choices=("U-lo", "T-frag", "NONE"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if args.regime == "NONE" and args.protocol != "IMM":
        raise ValueError("Use one shared clean anchor per seed")
    train(args.protocol, args.regime, args.seed)
