"""MMSA feature-pickle loading.

Schema (verified against the downloaded files):
  MOSI  text (N,50,768)  audio (N,375,5)   vision (N,500,20)   labels [-3,3]
  SIMS  text (N,39,768)  audio (N,400,33)  vision (N,55,709)   labels [-1,1]
  MOSEI text (N,50,768)  audio (N,500,74)  vision (N,500,35)   labels [-3,3]

Validity masks: text from text_bert[:,1,:] (the BERT attention mask);
audio/vision from the *_lengths arrays. Features are z-scored with training-set
statistics computed over valid frames only, and NaN/Inf are scrubbed (MOSEI
acoustic features are known to contain them).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .masking import MODALITIES

DATASET_FILES = {
    "MOSI": "MOSI/unaligned_50.pkl",
    "MOSEI": "MOSEI/unaligned_50.pkl",
    "SIMS": "SIMS/unaligned_39.pkl",
}
LABEL_RANGE = {"MOSI": 3.0, "MOSEI": 3.0, "SIMS": 1.0}


@dataclass
class Norm:
    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]


def _valid_masks(split: dict, n: int) -> Dict[str, np.ndarray]:
    out = {}
    tb = np.asarray(split["text_bert"])          # (N, 3, T) -> row 1 is attn mask
    out["t"] = tb[:, 1, :].astype(bool)
    for k, key, feat in (("a", "audio_lengths", "audio"),
                         ("v", "vision_lengths", "vision")):
        T = np.asarray(split[feat]).shape[1]
        lengths = np.asarray(split[key]).astype(int).clip(0, T)
        m = np.arange(T)[None, :] < lengths[:, None]
        # A zero length would erase the example entirely; keep one frame.
        m[lengths <= 0, 0] = True
        out[k] = m
    return out


def _raw(split: dict) -> Dict[str, np.ndarray]:
    return {
        "t": np.asarray(split["text"], dtype=np.float32),
        "a": np.nan_to_num(np.asarray(split["audio"], dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0),
        "v": np.nan_to_num(np.asarray(split["vision"], dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0),
    }


def fit_norm(feats: Dict[str, np.ndarray], valid: Dict[str, np.ndarray]) -> Norm:
    mean, std = {}, {}
    for k in MODALITIES:
        x, m = feats[k], valid[k]
        flat = x[m]                                      # (n_valid_frames, D)
        mean[k] = flat.mean(axis=0)
        s = flat.std(axis=0)
        std[k] = np.where(s < 1e-6, 1.0, s)              # guard constant dims
    return Norm(mean=mean, std=std)


def apply_norm(feats, valid, norm: Norm):
    out = {}
    for k in MODALITIES:
        z = (feats[k] - norm.mean[k]) / norm.std[k]
        out[k] = (z * valid[k][..., None]).astype(np.float32)   # keep pads at 0
    return out


class MMSASplit(Dataset):
    def __init__(self, feats, valid, labels, ids, bert=None):
        self.feats, self.valid = feats, valid
        self.labels, self.ids = labels, ids
        # bert: (N, 3, T) = input_ids / attention_mask / token_type_ids.
        # Row 1 is the same array we use as valid["t"], so masking the text
        # validity mask IS masking BERT's attention mask -- token-level
        # erasure, identical semantics to frame erasure on the other modalities.
        self.bert = bert

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return (
            {k: torch.from_numpy(self.feats[k][i]) for k in MODALITIES},
            {k: torch.from_numpy(self.valid[k][i]) for k in MODALITIES},
            torch.tensor(self.labels[i], dtype=torch.float32),
            i,
            (torch.from_numpy(self.bert[i]) if self.bert is not None
             else torch.zeros(1, dtype=torch.long)),
        )


def collate(batch):
    feats = {k: torch.stack([b[0][k] for b in batch]) for k in MODALITIES}
    valid = {k: torch.stack([b[1][k] for b in batch]) for k in MODALITIES}
    y = torch.stack([b[2] for b in batch])
    idx = torch.tensor([b[3] for b in batch], dtype=torch.long)
    bert = torch.stack([b[4] for b in batch])
    return feats, valid, y, idx, bert


def load_dataset(name: str, root: str = "datasets", tokenizer: str = ""):
    """Returns (splits dict of MMSASplit, dims, seqs, label_range).

    `tokenizer` selects a re-tokenised cache built by parm/data/retokenize.py.
    Needed for any encoder other than BERT: the stored `text_bert` ids are
    WordPiece, so RoBERTa must be given ids from its own tokeniser.
    """
    path = Path(root) / DATASET_FILES[name]
    with open(path, "rb") as f:
        raw = pickle.load(f)

    retok = None
    if tokenizer:
        from .retokenize import load_cache
        retok = load_cache(root, name, tokenizer)
        if retok is None:
            raise FileNotFoundError(
                f"no token cache for {name}/{tokenizer}; run "
                f"python -m parm.data.retokenize --datasets {name} "
                f"--model {tokenizer}")

    prepared, norm = {}, None
    for sp in ("train", "valid", "test"):
        s = raw[sp]
        if retok is not None:
            # Swap in the new ids/mask. The text FEATURE array is unused when a
            # transformer encoder is active (it overwrites feats["t"]), but its
            # sequence length must match so positional embeddings line up.
            s = dict(s)
            s["text_bert"] = retok[sp]
            T = retok[sp].shape[-1]
            s["text"] = np.zeros((len(s["id"]), T, 768), dtype=np.float32)
        n = len(s["id"])
        feats, valid = _raw(s), _valid_masks(s, n)
        if sp == "train":
            norm = fit_norm(feats, valid)
        feats = apply_norm(feats, valid, norm)
        prepared[sp] = MMSASplit(
            feats, valid,
            np.asarray(s["regression_labels"], dtype=np.float32),
            np.asarray(s["id"]),
            bert=np.asarray(s["text_bert"], dtype=np.int64),
        )

    dims = {k: prepared["train"].feats[k].shape[-1] for k in MODALITIES}
    seq = {k: prepared["train"].feats[k].shape[1] for k in MODALITIES}
    return prepared, dims, seq, LABEL_RANGE[name]
