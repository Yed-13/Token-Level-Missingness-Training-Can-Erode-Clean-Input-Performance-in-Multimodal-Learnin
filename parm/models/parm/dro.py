"""Group DRO over missingness-pattern groups.

Instead of average-case risk under one regime, minimise worst-case risk across
pattern groups. Weights follow exponentiated gradient ascent on running-average
group losses (raw per-batch group losses are far too noisy at these batch
sizes, where a group may contribute only a handful of examples).
"""
from __future__ import annotations

import torch


class GroupDRO:
    def __init__(self, n_groups: int, eta: float = 0.01, ema: float = 0.1,
                 device: str = "cpu", mode: str = "raw",
                 warmup_steps: int = 0, q_max_ratio: float = 0.0):
        """mode="raw":    classic group DRO on absolute group risk.
        mode="excess": DRO on group risk MINUS a per-group reference level
            estimated during warmup under uniform weighting.

            Absolute worst-case risk is the wrong target when groups differ in
            *irreducible* difficulty. A text-absent group is hard because the
            information is gone, not because it is under-fit; up-weighting it
            sacrifices learnable groups for nothing. Excess risk asks instead
            which groups are furthest from their own achievable level.

        q_max_ratio > 0 clamps max(q)/min(q) to bound how aggressive the
        reweighting can get (a CVaR-like trust region).
        """
        self.n = n_groups
        self.eta = eta
        self.ema = ema
        self.mode = mode
        self.warmup_steps = warmup_steps
        self.q_max_ratio = q_max_ratio
        self.steps = 0
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.running = torch.zeros(n_groups, device=device)
        self.reference = torch.zeros(n_groups, device=device)
        self.seen = torch.zeros(n_groups, device=device)

    def state_dict(self):
        return {"q": self.q.cpu(), "running": self.running.cpu(),
                "reference": self.reference.cpu(), "seen": self.seen.cpu(),
                "mode": self.mode, "steps": self.steps}

    def __call__(self, per_example_loss: torch.Tensor,
                 groups: torch.Tensor) -> torch.Tensor:
        """Weighted loss. per_example_loss (B,), groups (B,) int64."""
        dev = per_example_loss.device
        if self.q.device != dev:
            self.q, self.running, self.seen = (self.q.to(dev),
                                               self.running.to(dev),
                                               self.seen.to(dev))
        present = torch.bincount(groups, minlength=self.n).float()
        sums = torch.zeros(self.n, device=dev).index_add_(
            0, groups, per_example_loss.detach())
        means = sums / present.clamp(min=1)

        # Update running estimates only for groups present in this batch.
        mask = present > 0
        self.running = torch.where(
            mask, (1 - self.ema) * self.running + self.ema * means, self.running)
        self.seen = self.seen + present

        self.steps += 1

        # Warmup: uniform weighting while we learn each group's achievable level.
        if self.steps <= self.warmup_steps:
            self.reference = self.running.clone()
            return per_example_loss.mean()

        signal = (self.running - self.reference).clamp(min=0.0) \
            if self.mode == "excess" else self.running

        # Exponentiated-gradient ascent on q, then renormalise.
        with torch.no_grad():
            q = self.q * torch.exp(self.eta * signal)
            # Never let an unseen group hoard weight.
            q = torch.where(self.seen > 0, q, torch.zeros_like(q))
            q = q / q.sum().clamp(min=1e-12)
            if self.q_max_ratio > 0:
                live = q[self.seen > 0]
                if live.numel() > 1:
                    lo = live.max() / self.q_max_ratio
                    q = torch.where(self.seen > 0, q.clamp(min=float(lo)), q)
                    q = q / q.sum().clamp(min=1e-12)
            self.q = q

        # Reweight: each example scaled by q_g / (empirical share of g).
        share = (present / present.sum().clamp(min=1)).clamp(min=1e-8)
        scale = (self.q / share)[groups]
        return (per_example_loss * scale).mean()
