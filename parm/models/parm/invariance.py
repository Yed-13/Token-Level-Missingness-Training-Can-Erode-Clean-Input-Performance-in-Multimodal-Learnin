"""Conditional pattern-invariance.

We want the fused representation to carry no information about *which*
missingness pattern produced it, given the label: z indep m | y.
Unconditional invariance would be wrong -- pattern and label are not
independent in effect, so forcing marginal invariance destroys label signal.

Two interchangeable estimators (chosen empirically, the loser goes in the
ablation): label-binned MMD, and a pattern discriminator with gradient reversal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased-ish MMD^2 with a median-heuristic Gaussian kernel."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device)
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z, p=2).pow(2)
    with torch.no_grad():
        med = d2[d2 > 0].median() if (d2 > 0).any() else torch.tensor(1.0, device=x.device)
        sigma2 = (med / 2).clamp(min=1e-6)
    k = torch.exp(-d2 / (2 * sigma2))
    n, m = x.shape[0], y.shape[0]
    kxx = (k[:n, :n].sum() - k[:n, :n].diag().sum()) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - k[n:, n:].diag().sum()) / (m * (m - 1))
    kxy = k[:n, n:].mean()
    return (kxx + kyy - 2 * kxy).clamp(min=0.0)


def conditional_mmd(z: torch.Tensor, groups: torch.Tensor, y: torch.Tensor,
                    n_bins: int = 5, label_range: float = 3.0,
                    min_per_cell: int = 2) -> torch.Tensor:
    """Mean pairwise MMD^2 between pattern groups, within label bins."""
    edges = torch.linspace(-label_range, label_range, n_bins + 1, device=y.device)[1:-1]
    bins = torch.bucketize(y, edges)
    total, count = torch.zeros((), device=z.device), 0
    for b in bins.unique():
        sel = bins == b
        gz, gg = z[sel], groups[sel]
        uniq = gg.unique()
        if uniq.numel() < 2:
            continue
        for i in range(uniq.numel()):
            for j in range(i + 1, uniq.numel()):
                a, c = gz[gg == uniq[i]], gz[gg == uniq[j]]
                if a.shape[0] < min_per_cell or c.shape[0] < min_per_cell:
                    continue
                total = total + _gaussian_mmd2(a, c)
                count += 1
    return total / count if count else torch.zeros((), device=z.device)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class PatternDiscriminator(nn.Module):
    """Adversarial alternative: predict the pattern group from (z, y) through a
    gradient-reversal layer, so the encoder is pushed to remove pattern info."""

    def __init__(self, d: int, n_groups: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.GELU(),
            nn.Linear(hidden, n_groups),
        )

    def forward(self, z, y, groups, lambd: float = 1.0):
        zr = _GradReverse.apply(z, lambd)
        logits = self.net(torch.cat([zr, y.unsqueeze(-1)], dim=-1))
        return F.cross_entropy(logits, groups)
