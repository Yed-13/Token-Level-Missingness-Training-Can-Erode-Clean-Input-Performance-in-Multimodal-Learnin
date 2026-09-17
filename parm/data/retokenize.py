"""Re-tokenise raw_text for a non-BERT encoder.

The MMSA pickles store text as BERT WordPiece ids in `text_bert`, so a
different encoder cannot reuse them -- feeding RoBERTa BERT's ids produces
garbage. This regenerates ids/attention-mask from the `raw_text` field that
the pickles also carry, caching one .npz per (dataset, model).

Sequence length is kept identical to the original (50 for MOSI/MOSEI, 39 for
CH-SIMS) so the encoder comparison is not confounded by context length.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .datasets import DATASET_FILES

SEQ_LEN = {"MOSI": 50, "MOSEI": 50, "SIMS": 39}


def cache_path(root: str, dataset: str, model: str,
               lowercase: bool = True) -> Path:
    safe = model.replace("/", "__")
    suffix = "" if lowercase else "__cased"
    return Path(root) / dataset / f"tokens_{safe}{suffix}.npz"


def build(dataset: str, model: str, root: str = "datasets",
          overwrite: bool = False, lowercase: bool = True) -> Path:
    """Build a token cache.

    `lowercase` defaults True because 100% of MOSI transcripts are ALL CAPS.
    bert-base-uncased lowercases internally, but RoBERTa's BPE is
    case-sensitive, so raw capitals fragment into subwords. At the 50-token
    limit on MOSI training data, mean retained lengths are 14.78 for BERT,
    20.49 for original-case RoBERTa and 14.81 for lowercased RoBERTa.
    True truncation rates (untruncated length > 50) are 0.55%, 3.58% and
    0.62%, respectively. Lowercasing brings length exposure closer; the
    encoders still use distinct tokenizers. See scripts/audit_tokenization.py.
    """
    out = cache_path(root, dataset, model, lowercase)
    if out.exists() and not overwrite:
        print(f"[{dataset}/{model}] cache exists: {out}")
        return out

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    max_len = SEQ_LEN[dataset]

    with open(Path(root) / DATASET_FILES[dataset], "rb") as f:
        raw = pickle.load(f)

    arrays = {}
    for sp in ("train", "valid", "test"):
        texts = [str(t) for t in raw[sp]["raw_text"]]
        if lowercase:
            texts = [t.lower() for t in texts]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=max_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in enc:
            ttype = enc["token_type_ids"].astype(np.int64)
        else:
            ttype = np.zeros_like(ids)
        # Same (N, 3, T) layout as the original text_bert field.
        arrays[sp] = np.stack([ids, mask, ttype], axis=1)
        kept = mask.sum(axis=1)
        print(f"  {sp}: n={len(texts)} shape={arrays[sp].shape} "
              f"tokens/example mean={kept.mean():.1f} max={kept.max()} "
              f"at_length_limit={(kept == max_len).sum()}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{sp: arrays[sp] for sp in arrays})
    print(f"[{dataset}/{model}] wrote {out}")
    return out


def load_cache(root: str, dataset: str, model: str, lowercase: bool = True):
    p = cache_path(root, dataset, model, lowercase)
    if not p.exists():
        return None
    z = np.load(p)
    return {sp: z[sp] for sp in ("train", "valid", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MOSI"])
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cased", action="store_true",
                    help="keep original casing (default lowercases; see build())")
    a = ap.parse_args()
    for ds in a.datasets:
        print(f"=== {ds} / {a.model} "
              f"({'cased' if a.cased else 'lowercased'}) ===")
        build(ds, a.model, a.root, a.overwrite, lowercase=not a.cased)


if __name__ == "__main__":
    main()
