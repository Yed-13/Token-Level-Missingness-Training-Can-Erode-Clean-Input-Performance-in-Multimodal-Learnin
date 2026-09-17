"""Audit MOSI token lengths and true truncation using cached tokenizers.

This is a CPU-only data audit. It never rewrites the training token cache.
Sequences exactly at the length limit are distinguished from truncated ones.
"""
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parm.data.download import SPECS, sha256_of


def main():
    source = ROOT / "datasets/MOSI/unaligned_50.pkl"
    digest = sha256_of(source)
    assert digest == SPECS["MOSI"]["sha256"]
    with source.open("rb") as handle:
        raw = pickle.load(handle)
    cache_path = ROOT / "datasets/MOSI/tokens_roberta-base.npz"
    cache = np.load(cache_path)
    tokenizers = {name: AutoTokenizer.from_pretrained(name, local_files_only=True)
                  for name in ("bert-base-uncased", "roberta-base")}
    rows = []
    for split in ("train", "valid", "test"):
        texts = [str(x) for x in raw[split]["raw_text"]]
        stored_lengths = np.asarray(raw[split]["text_bert"])[:, 1, :].sum(1)
        for model, lowercase in (("bert-base-uncased", False),
                                 ("roberta-base", False), ("roberta-base", True)):
            inputs = [x.lower() for x in texts] if lowercase else texts
            tok = tokenizers[model]
            tokens = tok(inputs, truncation=False, padding=False)["input_ids"]
            lengths = np.array([len(x) for x in tokens])
            row = dict(partition=split, model=model, lowercase=lowercase,
                       n=len(texts), all_caps_fraction=float(np.mean([x == x.upper() for x in texts])),
                       mean_untruncated_length=float(lengths.mean()),
                       mean_capped_length=float(np.minimum(lengths, 50).mean()),
                       count_truncated=int((lengths > 50).sum()),
                       count_exactly_at_limit=int((lengths == 50).sum()),
                       truncation_fraction=float((lengths > 50).mean()),
                       stored_bert_mean_length=float(stored_lengths.mean()))
            if model == "roberta-base" and lowercase:
                padded = tok(inputs, truncation=True, padding="max_length",
                             max_length=50, return_tensors="np")
                row["matches_archived_ids"] = bool(np.array_equal(
                    padded["input_ids"], cache[split][:, 0, :]))
                row["matches_archived_mask"] = bool(np.array_equal(
                    padded["attention_mask"], cache[split][:, 1, :]))
            rows.append(row)
    result = dict(dataset="MOSI", max_length=50, lengths_include_special_tokens=True,
                  source_sha256=digest, token_cache_sha256=sha256_of(cache_path), rows=rows)
    destination = ROOT / "analysis/data/tokenization_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
