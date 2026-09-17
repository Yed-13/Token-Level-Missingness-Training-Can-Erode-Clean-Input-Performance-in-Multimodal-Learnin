"""Missingness-pattern sampling — the SINGLE source of truth for masking.

Every method (ours and all baselines) MUST import from this module. Subtly
different masking between methods is the classic way a missing-modality
comparison becomes meaningless.

Two protocols from the literature:
  IMM (intra-modal missingness): a fraction r_k of temporal positions of
      modality k is erased.
  FMM (fixed-modality missingness): modality k is entirely absent.

A *regime* is a distribution over per-modality missing rates. Existing work
trains and tests under the same regime; this project evaluates mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

MODALITIES: Tuple[str, str, str] = ("t", "a", "v")

# Per-modality missing-rate ranges (lo, hi), sampled uniformly per example.
REGIMES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "U-lo":    {"t": (0.1, 0.3), "a": (0.1, 0.3), "v": (0.1, 0.3)},
    "U-hi":    {"t": (0.5, 0.7), "a": (0.5, 0.7), "v": (0.5, 0.7)},
    "T-frag":  {"t": (0.6, 0.8), "a": (0.1, 0.3), "v": (0.1, 0.3)},
    "NV-frag": {"t": (0.0, 0.2), "a": (0.6, 0.8), "v": (0.6, 0.8)},
}
REGIME_NAMES = tuple(REGIMES.keys())

# Union of all regime supports: used for PARM's DRO group coverage so the
# worst-case objective is taken over the full pattern space, not one regime.
REGIME_ALL = {k: (0.0, 0.8) for k in MODALITIES}

DEGRADED_THRESHOLD = 0.5  # r_k above this counts as "heavily degraded"


@dataclass(frozen=True)
class MaskSpec:
    """Per-example missingness: a rate per modality, plus derived group id."""
    rates: Tuple[float, float, float]      # (r_t, r_a, r_v)
    group: int                             # DRO group id
    signature: Tuple[int, int, int]        # binary degradation signature


def pattern_group(rates, granularity: int = 8) -> int:
    """Map per-modality missing rates to a DRO group id.

    granularity=8 : binary degradation signature over (t, a, v)   -> 8 groups
    granularity=16: signature x coarse global-rate bucket          -> 16 groups
    granularity=4 : (text degraded?) x (any nonverbal degraded?)   -> 4 groups
    granularity=2 : any modality degraded or not                   -> 2 groups
    """
    sig = tuple(int(r > DEGRADED_THRESHOLD) for r in rates)
    if granularity == 8:
        return sig[0] * 4 + sig[1] * 2 + sig[2]
    if granularity == 16:
        base = sig[0] * 4 + sig[1] * 2 + sig[2]
        return base * 2 + int(float(np.mean(rates)) > 0.4)
    if granularity == 4:
        return sig[0] * 2 + int(sig[1] or sig[2])
    if granularity == 2:
        return int(any(sig))
    raise ValueError(f"unsupported granularity: {granularity}")


def n_groups(granularity: int = 8) -> int:
    return {2: 2, 4: 4, 8: 8, 16: 16}[granularity]


def parse_regime(regime: str):
    """Resolve a regime name to per-modality (lo, hi) missing-rate ranges.

    Accepts the four named regimes, "ALL" (union support), "NONE", and
    "T@<rate>" -- text missing at a fixed rate with nonverbal held at 0.1.
    The T@ form gives a dose-response curve over text scarcity, which is what
    locates the threshold where encoder damage begins.
    """
    if regime.startswith("T@"):
        r = float(regime[2:])
        return {"t": (r, r), "a": (0.1, 0.1), "v": (0.1, 0.1)}
    if regime == "ALL":
        return REGIME_ALL
    return REGIMES[regime]


def sample_rates(regime: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Draw per-modality missing rates for one example from a regime.

    "NONE" means no missingness at all -- used to measure complete-input
    performance so we can check the backbone against published numbers. It is
    deliberately NOT in REGIME_NAMES, so it never enters the transfer matrix.
    """
    if regime == "NONE":
        return (0.0, 0.0, 0.0)
    spec = parse_regime(regime)
    return tuple(float(rng.uniform(*spec[k])) for k in MODALITIES)


def sample_mask_spec(regime: str, rng: np.random.Generator,
                     granularity: int = 8) -> MaskSpec:
    rates = sample_rates(regime, rng)
    return MaskSpec(rates=rates,
                    group=pattern_group(rates, granularity),
                    signature=tuple(int(r > DEGRADED_THRESHOLD) for r in rates))


def apply_mask(
    feats: Dict[str, torch.Tensor],
    valid: Dict[str, torch.Tensor],
    rates: torch.Tensor,
    protocol: str = "IMM",
    gen: torch.Generator | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply missingness to a batch.

    feats: modality -> (B, T_k, D_k) float
    valid: modality -> (B, T_k) bool, True where a real (non-padding) frame sits
    rates: (B, 3) float, per-example per-modality missing rate
    protocol: "IMM" (erase a fraction of frames) or "FMM" (erase whole modality)
    gen:   a CPU torch.Generator. ALL randomness is drawn on CPU and then moved
           to the feature device, so masks are bit-identical on MPS and CUDA.
           This matters: development happens on MPS, reported numbers come from
           CUDA, and the two must agree exactly.

    Guarantees at least one modality survives per example.

    Returns masked features (erased positions zeroed) and updated validity
    masks. Erased content is genuinely inaccessible downstream -- no leakage.
    """
    if protocol not in ("IMM", "FMM"):
        raise ValueError(f"unknown protocol: {protocol}")

    device = feats[MODALITIES[0]].device
    r_cpu = rates.detach().to("cpu")
    B = r_cpu.shape[0]
    out_f, out_v = {}, {}

    if protocol == "FMM":
        u = torch.rand(B, 3, generator=gen)
        drop = u < r_cpu
        all_gone = drop.all(dim=1)
        if all_gone.any():
            idx = torch.nonzero(all_gone, as_tuple=True)[0]
            keep = r_cpu[idx].argmin(dim=1)
            drop[idx, keep] = False
        for j, k in enumerate(MODALITIES):
            f, v = feats[k], valid[k]
            km = (~drop[:, j]).to(device)
            out_f[k] = f * km.view(B, *([1] * (f.dim() - 1))).to(f.dtype)
            out_v[k] = v & km.view(B, 1)
        return out_f, out_v

    # IMM: erase each real frame independently with probability r_k.
    v_cpu = {k: valid[k].detach().to("cpu") for k in MODALITIES}
    keep_cpu = {}
    for j, k in enumerate(MODALITIES):
        u = torch.rand(v_cpu[k].shape, generator=gen)
        erase = (u < r_cpu[:, j].view(B, 1)) & v_cpu[k]
        keep_cpu[k] = v_cpu[k] & ~erase

    # Restore the least-degraded modality for examples left fully empty.
    nonempty = torch.stack([keep_cpu[k].any(dim=1) for k in MODALITIES], dim=1)
    all_gone = ~nonempty.any(dim=1)
    if all_gone.any():
        idx = torch.nonzero(all_gone, as_tuple=True)[0]
        keep = r_cpu[idx].argmin(dim=1)
        for j, k in enumerate(MODALITIES):
            sel = idx[keep == j]
            if sel.numel():
                keep_cpu[k][sel] = v_cpu[k][sel]

    for k in MODALITIES:
        kv = keep_cpu[k].to(device)
        out_v[k] = kv
        out_f[k] = feats[k] * kv.unsqueeze(-1).to(feats[k].dtype)
    return out_f, out_v


def observed_rate(valid_masked, valid_orig) -> torch.Tensor:
    """Fraction of originally-real frames that survived, per modality. Diagnostic."""
    out = []
    for k in MODALITIES:
        denom = valid_orig[k].sum(dim=1).clamp(min=1)
        out.append(valid_masked[k].sum(dim=1) / denom)
    return torch.stack(out, dim=1)
