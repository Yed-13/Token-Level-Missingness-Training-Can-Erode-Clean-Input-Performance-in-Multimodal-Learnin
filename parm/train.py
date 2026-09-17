"""Train one configuration and evaluate it under EVERY test regime.

One training run yields a full row of the cross-pattern transfer matrix, which
is what makes the 4x4 design affordable: 4 training runs per (method, dataset,
seed), not 16.

Discipline enforced here:
  * Validation/early stopping defaults to the training regime; an explicit
    validation-regime override supports sensitivity runs. Test scores are
    excluded from checkpoint selection.
  * Test masks are drawn from a generator seeded by (dataset, regime, repeat)
    and are therefore IDENTICAL across methods and training seeds.
  * `recon @ train_regime=ALL` is the pattern-distribution-training baseline
    (M3S-style). PARM at train_regime=ALL differs from it only by DRO,
    invariance and capacity allocation -- that is the isolated contribution.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.datasets import collate, load_dataset
from .data.masking import (MODALITIES, REGIME_NAMES, apply_mask, n_groups,
                           pattern_group, sample_rates)
from .metrics import regression_metrics
from .seeding import stable_eval_seed
from .models.backbones.fusion import (MultimodalFusion, param_count,
                                      representation_diagnostics)
from .models.parm.capacity import allocate_hidden, normalise_alpha, uniform_hidden
from .models.parm.dro import GroupDRO
from .models.parm.invariance import PatternDiscriminator, conditional_mmd

METHODS: Dict[str, dict] = {
    "complete":     dict(mask_train=False, recon=False, dro=False, inv=0.0, cap="uniform"),
    "moddrop":      dict(mask_train=True,  recon=False, dro=False, inv=0.0, cap="uniform"),
    "recon":        dict(mask_train=True,  recon=True,  dro=False, inv=0.0, cap="uniform"),
    "parm":         dict(mask_train=True,  recon=True,  dro=True,  inv=0.1, cap="alpha"),
    "parm-nodro":   dict(mask_train=True,  recon=True,  dro=False, inv=0.1, cap="alpha"),
    "parm-noinv":   dict(mask_train=True,  recon=True,  dro=True,  inv=0.0, cap="alpha"),
    "parm-nocap":   dict(mask_train=True,  recon=True,  dro=True,  inv=0.1, cap="uniform"),
    "parm-uncond":  dict(mask_train=True,  recon=True,  dro=True,  inv=0.1, cap="alpha", inv_mode="uncond"),
    "parm-disc":    dict(mask_train=True,  recon=True,  dro=True,  inv=0.1, cap="alpha", inv_mode="disc"),
    # --- v2: excess-risk DRO + tempered capacity (see dro.py rationale) ---
    "parm2":        dict(mask_train=True, recon=True, dro=True, inv=0.05, cap="alpha",
                         dro_mode="excess", alpha_temp=3.0, q_max_ratio=4.0),
    "parm2-nodro":  dict(mask_train=True, recon=True, dro=False, inv=0.05, cap="alpha",
                         alpha_temp=3.0),
    "parm2-noinv":  dict(mask_train=True, recon=True, dro=True, inv=0.0, cap="alpha",
                         dro_mode="excess", alpha_temp=3.0, q_max_ratio=4.0),
    "parm2-nocap":  dict(mask_train=True, recon=True, dro=True, inv=0.05, cap="uniform",
                         dro_mode="excess", q_max_ratio=4.0),
    "parm2-rawdro": dict(mask_train=True, recon=True, dro=True, inv=0.05, cap="alpha",
                         dro_mode="raw", alpha_temp=3.0, q_max_ratio=4.0),
    "parm2-caponly":dict(mask_train=True, recon=True, dro=False, inv=0.0, cap="alpha",
                         alpha_temp=3.0),
}

BERT_FOR = {"MOSI": "bert-base-uncased", "MOSEI": "bert-base-uncased",
            "SIMS": "bert-base-chinese"}

EVAL_REPEATS = 5          # independent mask draws per TEST regime
VAL_REPEATS = 2           # cheaper draws for early stopping only
EVAL_SEED_BASE = 20260915


def limit_threads(per_job: int = 0) -> int:
    """Cap torch's intra-op threads to the container's real CPU budget.

    torch sizes its thread pool from nproc, which inside a container reports
    the HOST's core count (128 here) rather than the cgroup quota (32). Four
    concurrent jobs then spawn ~512 threads for 32 usable CPUs; the resulting
    oversubscription starves DataLoader IPC and the main process blocks in
    select/poll indefinitely. Observed: 4 jobs stuck at 11+ minutes, 2.7% CPU,
    zero GPU utilisation.

    Reads cpu.max (cgroup v2) for the true quota and divides it across the
    concurrent jobs we expect to run.
    """
    quota = None
    try:
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw[0] != "max":
            quota = max(1, int(int(raw[0]) / int(raw[1])))
    except Exception:
        pass
    if quota is None:
        quota = os.cpu_count() or 8
    n = per_job if per_job > 0 else max(2, quota // 8)
    n = max(1, min(n, quota))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(min(2, n))
    except RuntimeError:
        pass  # already initialised
    return n


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_generator(dataset: str, regime: str, repeat: int) -> torch.Generator:
    """Deterministic, method-independent test masks."""
    return torch.Generator().manual_seed(
        stable_eval_seed(dataset, regime, repeat, "mask", EVAL_SEED_BASE))


def batch_patterns(regime: str, bsz: int, rng: np.random.Generator,
                   granularity: int):
    rates = np.stack([sample_rates(regime, rng) for _ in range(bsz)])
    groups = np.array([pattern_group(tuple(r), granularity) for r in rates])
    return (torch.from_numpy(rates).float(),
            torch.from_numpy(groups).long())


def autocast_ctx(cfg, device):
    import contextlib
    if cfg.amp == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(feats, valid, device):
    return ({k: feats[k].to(device) for k in MODALITIES},
            {k: valid[k].to(device) for k in MODALITIES})


@dataclass
class Cfg:
    dataset: str = "MOSI"
    method: str = "parm"
    train_regime: str = "ALL"
    protocol: str = "IMM"          # training protocol
    test_protocol: str = ""        # evaluation protocol; "" = same as training
    validation_regime: str = ""   # empty uses the original selection rule
    save_checkpoint: int = 0       # persist selected weights for re-evaluation
    seed: int = 0
    d: int = 64
    layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 8
    lambda_recon: float = 0.5
    lambda_inv: float = 0.1
    dro_eta: float = 0.01
    dro_ema: float = 0.1
    granularity: int = 8
    label_bins: int = 5
    inv_mode: str = "cmmd"
    alpha_temp: float = 1.0
    dro_mode: str = "raw"
    dro_warmup_steps: int = 100
    q_max_ratio: float = 0.0
    text_encoder: str = "frozen"      # "frozen" | "bert"
    bert_name: str = "auto"           # "auto" -> chinese for SIMS, uncased otherwise
    bert_lr: float = 2e-5
    freeze_bert_layers: int = 0    # freeze embeddings + first N encoder layers
    amp: str = "bf16"                 # "bf16" | "off"
    num_workers: int = 2       # 32-CPU cgroup quota shared across jobs
    torch_threads: int = 0     # 0 = derive from the cgroup CPU quota
    text_mask_mode: str = "remove"   # "remove" (ours) | "unk" (literature)
    subsample_train: int = 0   # cap training-set size (0 = use all)
    verbose: int = 0
    root: str = "datasets"
    out: str = "results"


def build_model(cfg: Cfg, spec: dict, dims, seqs, alpha, device):
    total_hidden = 6 * cfg.d          # identical budget for both allocations
    if spec["cap"] == "alpha" and alpha is not None:
        hidden = allocate_hidden(
            normalise_alpha(alpha, temperature=spec.get("alpha_temp", 1.0)),
            total_hidden)
    else:
        hidden = uniform_hidden(total_hidden)
    bert_name = (BERT_FOR[cfg.dataset] if cfg.bert_name == "auto"
                 else cfg.bert_name)
    unk_id = -1
    if cfg.text_encoder == "bert" and cfg.text_mask_mode == "unk":
        # The tokenizer may not be cached: for BERT we consume the token ids
        # stored in the benchmark files and never instantiate a tokenizer, so
        # under HF_HUB_OFFLINE the lookup fails. Fall back to the documented
        # vocabulary index rather than aborting the run.
        FALLBACK_UNK = {"bert-base-uncased": 100, "bert-base-chinese": 100,
                        "roberta-base": 3}
        try:
            from transformers import AutoTokenizer
            unk_id = int(AutoTokenizer.from_pretrained(bert_name).unk_token_id)
        except Exception:
            if bert_name not in FALLBACK_UNK:
                raise RuntimeError(
                    f"cannot resolve [UNK] id for {bert_name}: tokenizer "
                    f"unavailable offline and no documented fallback")
            unk_id = FALLBACK_UNK[bert_name]
            print(f"    [warn] tokenizer unavailable; using documented "
                  f"unk_token_id={unk_id} for {bert_name}", flush=True)
    m = MultimodalFusion(dims, seqs, d=cfg.d, n_layers=cfg.layers,
                         nhead=cfg.nhead, dropout=cfg.dropout,
                         recon_hidden=hidden, use_recon=spec["recon"],
                         text_encoder=cfg.text_encoder,
                         bert_name=bert_name, unk_id=unk_id).to(device)
    return m, hidden


def evaluate(model, loader, cfg: Cfg, regime: str, label_range: float,
             device: str, repeats: int = EVAL_REPEATS,
             protocol: str | None = None) -> dict:
    """Mean metrics over `repeats` fixed mask draws (identical across methods)."""
    model.eval()
    per_repeat = []
    for rep in range(repeats):
        gen = eval_generator(cfg.dataset, regime, rep)
        rng = np.random.default_rng(
            stable_eval_seed(cfg.dataset, regime, rep, "rates", EVAL_SEED_BASE))
        preds, trues = [], []
        with torch.no_grad():
            for feats, valid, y, _, bert in loader:
                feats, valid = to_device(feats, valid, device)
                bert = bert.to(device)
                rates, _ = batch_patterns(regime, y.shape[0], rng, cfg.granularity)
                fm, vm = apply_mask(feats, valid, rates.to(device),
                                    protocol or cfg.protocol, gen)
                with autocast_ctx(cfg, device):
                    out = model(fm, vm, bert=bert, orig_text_mask=valid["t"])
                preds.append(out.float().cpu().numpy())
                trues.append(y.numpy())
        per_repeat.append(regression_metrics(np.concatenate(preds),
                                             np.concatenate(trues), label_range))
    keys = per_repeat[0].keys()
    return {k: float(np.nanmean([r[k] for r in per_repeat])) for k in keys}


def run(cfg: Cfg) -> dict:
    spec = dict(METHODS[cfg.method])
    spec.setdefault("inv_mode", cfg.inv_mode)
    spec.setdefault("alpha_temp", cfg.alpha_temp)
    spec.setdefault("dro_mode", cfg.dro_mode)
    spec.setdefault("q_max_ratio", cfg.q_max_ratio)
    n_threads = limit_threads(cfg.torch_threads)
    device = pick_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    bert_name = (BERT_FOR[cfg.dataset] if cfg.bert_name == "auto" else cfg.bert_name)
    # Only BERT can reuse the stored WordPiece ids; anything else needs a cache.
    needs_retok = (cfg.text_encoder == "bert"
                   and not bert_name.startswith("bert-base"))
    splits, dims, seqs, label_range = load_dataset(
        cfg.dataset, cfg.root, tokenizer=(bert_name if needs_retok else ""))

    # Break the alpha/size confound. Across our three corpora, text dominance
    # and training-set size are entangled: MOSI is small with alpha_text=.950
    # and shows 20.8 pts of damage; MOSEI has alpha_text=.948 but 12.7x the
    # data and shows 3.7. Subsampling MOSEI to MOSI's size holds alpha fixed
    # and varies only n, which isolates which factor drives the damage.
    if cfg.subsample_train > 0:
        from torch.utils.data import Subset
        tr = splits["train"]
        n = min(cfg.subsample_train, len(tr))
        # Seeded independently of cfg.seed so every seed sees the SAME subset;
        # we are varying training-set size, not resampling noise.
        idx = np.random.default_rng(12345).permutation(len(tr))[:n].tolist()
        splits["train"] = Subset(tr, idx)
    loaders = {
        sp: DataLoader(splits[sp], batch_size=cfg.batch_size,
                       shuffle=(sp == "train"), collate_fn=collate,
                       drop_last=False, num_workers=cfg.num_workers,
                       pin_memory=(device == "cuda"),
                       persistent_workers=(cfg.num_workers > 0))
        for sp in splits
    }

    alpha = None
    if spec["cap"] == "alpha":
        ap = Path(cfg.root) / f"alpha_{cfg.dataset}.json"
        if ap.exists():
            alpha = json.loads(ap.read_text())["alpha"]

    model, hidden = build_model(cfg, spec, dims, seqs, alpha, device)
    # BERT needs a much smaller LR than the randomly-initialised fusion stack;
    # using 1e-3 on pretrained weights destroys them in the first few steps.
    if cfg.freeze_bert_layers > 0 and getattr(model, "bert", None) is not None:
        enc = model.bert.bert
        for prm in enc.embeddings.parameters():
            prm.requires_grad_(False)
        for lyr in enc.encoder.layer[: cfg.freeze_bert_layers]:
            for prm in lyr.parameters():
                prm.requires_grad_(False)

    bert_params = [p for n, p in model.named_parameters()
                   if n.startswith("bert.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith("bert.") and p.requires_grad]
    groups = [{"params": rest, "lr": cfg.lr}]
    if bert_params:
        groups.append({"params": bert_params, "lr": cfg.bert_lr})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    G = n_groups(cfg.granularity)
    dro = (GroupDRO(G, cfg.dro_eta, cfg.dro_ema, device,
                    mode=spec.get("dro_mode", "raw"),
                    warmup_steps=(cfg.dro_warmup_steps
                                  if spec.get("dro_mode") == "excess" else 0),
                    q_max_ratio=spec.get("q_max_ratio", 0.0))
           if spec["dro"] else None)
    disc = (PatternDiscriminator(cfg.d, G).to(device)
            if spec.get("inv_mode") == "disc" and spec["inv"] > 0 else None)
    if disc is not None:
        opt.add_param_group({"params": disc.parameters()})

    rng = np.random.default_rng(cfg.seed)
    recon_w = (normalise_alpha(alpha, temperature=spec.get("alpha_temp", 1.0))
               if (spec["cap"] == "alpha" and alpha) else None)

    best = {"val": float("inf"), "state": None, "epoch": -1}
    history = []
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        agg = {"task": 0.0, "recon": 0.0, "inv": 0.0, "n": 0}
        for feats, valid, y, _, bert in loaders["train"]:
            feats, valid = to_device(feats, valid, device)
            bert = bert.to(device)
            y = y.to(device)
            B = y.shape[0]

            if spec["mask_train"]:
                rates, groups = batch_patterns(cfg.train_regime, B, rng,
                                               cfg.granularity)
                groups = groups.to(device)
                fm, vm = apply_mask(feats, valid, rates.to(device), cfg.protocol)
            else:
                fm, vm = feats, valid
                groups = torch.zeros(B, dtype=torch.long, device=device)

            with autocast_ctx(cfg, device):
                pred, parts = model(fm, vm, return_parts=True, bert=bert,
                                    orig_text_mask=valid["t"])
            pred = pred.float()
            # Cast the returned representations to fp32 once, here: under bf16
            # autocast the MMD kernel and the reconstruction MSE are both
            # numerically fragile in half precision.
            parts = {k: ({kk: vv.float() for kk, vv in v.items()}
                         if isinstance(v, dict) else v.float())
                     for k, v in parts.items()}
            per_ex = F.l1_loss(pred, y, reduction="none")
            task = dro(per_ex, groups) if dro is not None else per_ex.mean()

            loss = task
            rl = torch.zeros((), device=device)
            if spec["recon"]:
                with torch.no_grad(), autocast_ctx(cfg, device):
                    z_full = model.encode(feats, valid, bert,
                                          orig_text_mask=valid["t"])
                z_full = {k: v.float() for k, v in z_full.items()}
                rl = model.recon_loss(parts, z_full, recon_w)
                loss = loss + cfg.lambda_recon * rl

            il = torch.zeros((), device=device)
            if spec["inv"] > 0 and spec["mask_train"]:
                if spec.get("inv_mode") == "disc":
                    il = disc(parts["z"], y, groups)
                elif spec.get("inv_mode") == "uncond":
                    il = conditional_mmd(parts["z"], groups, torch.zeros_like(y),
                                         1, label_range)
                else:
                    il = conditional_mmd(parts["z"], groups, y,
                                         cfg.label_bins, label_range)
                loss = loss + cfg.lambda_inv * il

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["task"] += float(task.detach()) * B
            agg["recon"] += float(rl.detach()) * B
            agg["inv"] += float(il.detach()) * B
            agg["n"] += B

        # Use the predefined validation condition; test scores never select epochs.
        val_regime = cfg.validation_regime or (cfg.train_regime if spec["mask_train"] else "U-lo")
        vm_metrics = evaluate(model, loaders["valid"], cfg, val_regime,
                              label_range, device, repeats=VAL_REPEATS)
        if cfg.verbose:
            print(f"    ep {epoch:3d}  task {agg['task']/max(agg['n'],1):.4f}"
                  f"  recon {agg['recon']/max(agg['n'],1):.4f}"
                  f"  inv {agg['inv']/max(agg['n'],1):.4f}"
                  f"  val_mae {vm_metrics['mae']:.4f}"
                  f"  val_acc2 {vm_metrics['acc2']:.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        history.append({"epoch": epoch,
                        "train_task": agg["task"] / max(agg["n"], 1),
                        "train_recon": agg["recon"] / max(agg["n"], 1),
                        "train_inv": agg["inv"] / max(agg["n"], 1),
                        "val_mae": vm_metrics["mae"],
                        "val_acc2": vm_metrics["acc2"]})
        if vm_metrics["mae"] < best["val"] - 1e-5:
            best = {"val": vm_metrics["mae"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg.patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # Mechanism probe: is the encoder itself damaged, or only the decision
    # layer mismatched? Measured on COMPLETE (unmasked) test input, so it
    # reflects encoder health rather than robustness to the test regime.
    model.eval()
    Zt, Zf = [], []
    with torch.no_grad():
        for feats, valid, y, _, bert in loaders["test"]:
            feats, valid = to_device(feats, valid, device)
            bert = bert.to(device)
            with autocast_ctx(cfg, device):
                z_obs = model.encode(feats, valid, bert,
                                     orig_text_mask=valid["t"])
                _, parts = model(feats, valid, return_parts=True, bert=bert,
                                 orig_text_mask=valid["t"])
            Zt.append(z_obs["t"].float().cpu())
            Zf.append(parts["z"].float().cpu())
    diagnostics = {
        "text_repr": representation_diagnostics(torch.cat(Zt)),
        "fused_repr": representation_diagnostics(torch.cat(Zf)),
    }

    test_proto = cfg.test_protocol or cfg.protocol
    cells = {r: evaluate(model, loaders["test"], cfg, r, label_range, device,
                         protocol=test_proto)
             for r in REGIME_NAMES}
    # Complete-input performance, for comparison against published numbers.
    # Excluded from REGIME_NAMES so it never pollutes the transfer matrix.
    cells["NONE"] = evaluate(model, loaders["test"], cfg, "NONE",
                             label_range, device, repeats=1,
                             protocol=test_proto)

    rec = {
        "config": asdict(cfg),
        "method_spec": spec,
        "cells": cells,
        "alpha": alpha,
        "recon_hidden": hidden,
        "params": param_count(model),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "val_best_mae": best["val"],
        "history": history,
        "dro_q": (dro.q.cpu().tolist() if dro is not None else None),
        "dro_reference": (dro.reference.cpu().tolist() if dro is not None else None),
        "dro_running": (dro.running.cpu().tolist() if dro is not None else None),
        "wall_seconds": round(time.time() - t0, 1),
        "diagnostics": diagnostics,
        "test_protocol": test_proto,
        "subsample_train": cfg.subsample_train,
        "text_mask_mode": cfg.text_mask_mode,
        "n_train": len(splits["train"]),
        "freeze_bert_layers": cfg.freeze_bert_layers,
        "text_encoder": cfg.text_encoder,
        "bert_name": (BERT_FOR[cfg.dataset] if cfg.bert_name == "auto" else cfg.bert_name),
        "env": {"device": device, "torch": torch.__version__,
                "bert_attention": (getattr(model.bert.bert.config, "_attn_implementation", None)
                                   if model.bert is not None else None),
                "torch_threads": n_threads,
                "platform": platform.platform(), "python": platform.python_version()},
    }

    out = Path(cfg.out) / cfg.dataset / cfg.protocol
    out.mkdir(parents=True, exist_ok=True)
    tag = f"__test-{cfg.test_protocol}" if cfg.test_protocol else ""
    if cfg.subsample_train > 0:
        tag += f"__n{cfg.subsample_train}"
    if cfg.text_mask_mode != "remove":
        tag += f"__{cfg.text_mask_mode}"
    if cfg.validation_regime:
        tag += f"__val-{cfg.validation_regime}"
    enc = (BERT_FOR[cfg.dataset] if cfg.bert_name == "auto" else cfg.bert_name)
    if not enc.startswith("bert-base"):
        tag += f"__{enc.replace('/', '-')}"
    fn = f"{cfg.method}__train-{cfg.train_regime}{tag}__seed{cfg.seed}.json"
    if cfg.save_checkpoint:
        checkpoint_path = out / fn.replace(".json", ".pt")
        torch.save({"state_dict": best["state"], "config": asdict(cfg),
                    "best_epoch": best["epoch"]}, checkpoint_path)
        rec["checkpoint"] = checkpoint_path.name
    (out / fn).write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    for f, v in asdict(Cfg()).items():
        ap.add_argument(f"--{f.replace('_','-')}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = Cfg(**{k.replace("-", "_"): v for k, v in vars(a).items()})
    rec = run(cfg)
    complete = rec["cells"].get("NONE", {})
    acc = {r: v["acc2"] for r, v in rec["cells"].items() if r != "NONE"}
    mean_all, worst = float(np.mean(list(acc.values()))), min(acc.values())
    if cfg.train_regime in acc:
        diag = acc[cfg.train_regime]
        off = float(np.mean([v for r, v in acc.items() if r != cfg.train_regime]))
        summary = f"matched={diag:.4f} mismatch={off:.4f} gap={diag-off:+.4f}"
    else:
        summary = f"mean={mean_all:.4f} (trained on union support; no matched cell)"
    comp = (f"complete: acc2={complete.get('acc2', float('nan')):.4f} "
            f"mae={complete.get('mae', float('nan')):.4f} | " if complete else "")
    print(f"{cfg.method:14s} {cfg.dataset} [{cfg.text_encoder}] train={cfg.train_regime:8s} "
          f"seed={cfg.seed} | {comp}{summary} worst={worst:.4f} "
          f"| {rec['epochs_run']}ep {rec['wall_seconds']}s {rec['params']/1e6:.2f}M")


if __name__ == "__main__":
    main()
