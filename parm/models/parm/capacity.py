"""Informativeness-weighted capacity allocation.

alpha_k measures how much the model degrades when modality k is removed,
estimated once on complete training data with a short probe run. Reconstruction
capacity (decoder width) and reconstruction loss weight are then allocated
proportional to alpha_k -- capacity goes where absence hurts most.

Total parameter count is held fixed against the uniform-allocation ablation, so
any gain is attributable to allocation rather than to size.
"""
from __future__ import annotations

from typing import Dict

from ...data.masking import MODALITIES


def normalise_alpha(raw: Dict[str, float], floor: float = 0.05,
                    temperature: float = 1.0) -> Dict[str, float]:
    """Normalise informativeness to a simplex.

    `temperature` > 1 flattens the distribution (alpha ** (1/T)). This matters:
    on MOSI the raw split is t=.95 a=.04 v=.01, and allocating decoder width
    proportionally starves the audio/vision heads to the floor. Those are the
    heads doing the *easy and useful* job -- predicting nonverbal streams from
    text -- whereas the text head chases a target that audio+vision barely
    determine. T=3 keeps the ordering while keeping every head viable.
    """
    vals = {k: max(float(raw.get(k, 0.0)), 0.0) for k in MODALITIES}
    s = sum(vals.values())
    if s <= 0:
        return {k: 1.0 / len(MODALITIES) for k in MODALITIES}
    a = {k: v / s for k, v in vals.items()}
    if temperature != 1.0:
        a = {k: v ** (1.0 / temperature) for k, v in a.items()}
        t = sum(a.values()) or 1.0
        a = {k: v / t for k, v in a.items()}
    # Floor then renormalise: a modality with alpha 0 still needs some decoder.
    a = {k: max(v, floor) for k, v in a.items()}
    s = sum(a.values())
    return {k: v / s for k, v in a.items()}


def allocate_hidden(alpha: Dict[str, float], total_hidden: int,
                    min_hidden: int = 16, multiple_of: int = 8) -> Dict[str, int]:
    """Split a fixed hidden-unit budget across modalities in proportion to alpha.

    total_hidden is the budget the uniform baseline also gets (3 * 2d by
    default), so parameter counts stay comparable.
    """
    raw = {k: alpha[k] * total_hidden for k in MODALITIES}
    out = {k: max(min_hidden, int(round(v / multiple_of)) * multiple_of)
           for k, v in raw.items()}
    # Correct rounding drift so the budget is respected.
    drift = sum(out.values()) - total_hidden
    order = sorted(MODALITIES, key=lambda k: -out[k])
    i = 0
    while drift != 0 and i < 100:
        k = order[i % len(order)]
        step = -multiple_of if drift > 0 else multiple_of
        if out[k] + step >= min_hidden:
            out[k] += step
            drift += step
        i += 1
    return out


def uniform_hidden(total_hidden: int, multiple_of: int = 8) -> Dict[str, int]:
    per = max(multiple_of, int(round(total_hidden / 3 / multiple_of)) * multiple_of)
    return {k: per for k in MODALITIES}
