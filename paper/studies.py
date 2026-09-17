"""Named run sets ("studies") and the single loader used by every analysis.

WHY THIS EXISTS
---------------
Effect sizes were pool-dependent. Pooling `results_bert` with `results_mech`
and `results_sweep` gave MOSI token-level degradation of -21.0 points, while
`results_bert` alone gave -20.8; CH-SIMS moved -5.4 vs -4.6, and the shift gap
+.0094 vs +.0134. Neither figure is wrong -- they answer different questions,
because the directories differ in seed count and purpose. But a manuscript
cannot quote a number whose value depends on an unstated pooling decision.

Every claim must therefore name exactly one study. Figures and prose both go
through this module, so they cannot diverge.

The studies are NOT interchangeable:
  main    5 seeds, full training sets, both protocols, 4x4 grid
  size    MOSEI subsampled, 3 seeds, alpha_text held fixed
  sweep   parameterised text-erasure rate T@r, 3 seeds
  mech    3 seeds, run after representation diagnostics were added
  roberta RoBERTa encoder, 3 seeds, re-tokenised inputs
  mitig   three candidate interventions, 3 seeds
  unk     text masked by [UNK] substitution (the literature's operation)
  frozen  the earlier frozen-feature study; a DIFFERENT model, not a seed
          variant of the others
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

STUDIES: dict[str, dict] = {
    "main": dict(
        dirs=["results_bert"],
        desc="4x4 transfer matrices, fine-tuned BERT, full training sets",
        defines=["shift gap", "headline degradation", "backbone competitiveness"],
    ),
    "size": dict(
        dirs=["results_size"],
        desc="CMU-MOSEI subsampled, alpha_text fixed at .948",
        defines=["size-response"],
    ),
    "sweep": dict(
        dirs=["results_sweep"],
        desc="dose-response over training text-erasure rate",
        defines=["dose-response", "onset threshold"],
    ),
    "mech": dict(
        dirs=["results_mech"],
        desc="runs carrying representation diagnostics",
        defines=["collapse diagnostics"],
    ),
    "roberta": dict(
        dirs=["results_roberta"],
        desc="RoBERTa text encoder, tokenisation matched by lowercasing",
        defines=["encoder generality"],
    ),
    "mitig": dict(
        dirs=["results_mitig_crossproto", "results_mitig_freeze",
              "results_mitig_lowlr"],
        desc="three candidate mitigations",
        defines=["mitigation comparison"],
    ),
    "unk": dict(
        dirs=["results_unk"],
        desc="text masked by [UNK] substitution, the operation used in the "
             "literature, rather than removal from the attention mask",
        defines=["operator-robustness check"],
    ),
    "frozen": dict(
        dirs=["results"],
        desc="earlier study with frozen pre-extracted text features",
        defines=["shift gap under a frozen encoder"],
    ),
}


def load(study: str) -> list[dict]:
    """All result records for one named study. Never pools across studies."""
    if study not in STUDIES:
        raise KeyError(f"unknown study {study!r}; known: {sorted(STUDIES)}")
    out = []
    for d in STUDIES[study]["dirs"]:
        for f in glob(str(ROOT / d / "*" / "*" / "*.json")):
            try:
                with open(f) as handle:
                    r = json.load(handle)
            except Exception:
                continue
            r["_study"] = study
            r["_file"] = f
            out.append(r)
    return out


def counts() -> dict[str, int]:
    return {s: len(load(s)) for s in STUDIES}


def encoder_of(rec: dict) -> str:
    name = rec["config"].get("bert_name", "auto")
    if name == "auto" or name.startswith("bert-base"):
        return "bert-base"
    return name


def clean_acc(rec: dict):
    """End-to-end accuracy on unmasked input, not an encoder-only measurement."""
    return rec.get("cells", {}).get("NONE", {}).get("acc2")


def clean_mae(rec: dict):
    return rec.get("cells", {}).get("NONE", {}).get("mae")


def select(recs: Iterable[dict], **eq):
    """Filter records on config fields; `n_train` and `encoder` are derived."""
    out = []
    for r in recs:
        c = r["config"]
        ok = True
        for k, v in eq.items():
            if k == "encoder":
                got = encoder_of(r)
            elif k == "n_train":
                got = r.get("n_train") if c.get("subsample_train") else "full"
            else:
                got = c.get(k)
            if got != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out
