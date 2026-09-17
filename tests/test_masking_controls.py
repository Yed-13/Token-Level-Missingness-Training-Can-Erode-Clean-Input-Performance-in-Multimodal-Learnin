import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "training dependencies required")
class MaskingControlTests(unittest.TestCase):
    def test_reconstruction_loss_uses_absent_pairs(self):
        from parm.models.backbones.fusion import MultimodalFusion
        keys = ("t", "a", "v")
        model = MultimodalFusion({k: 2 for k in keys}, {k: 3 for k in keys},
                                 d=4, n_layers=1, nhead=2)
        predictions = {k: torch.ones(2, 4, requires_grad=True) for k in keys}
        targets = {k: torch.zeros(2, 4, requires_grad=True) for k in keys}
        availability = {k: torch.ones(2) for k in keys}
        availability["t"][0] = 0
        parts = {"z": torch.zeros(2, 4), "z_hat": predictions, "avail": availability}
        loss = model.recon_loss(parts, targets)
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        self.assertTrue((predictions["t"].grad[0] != 0).all())
        self.assertTrue((predictions["t"].grad[1] == 0).all())
        self.assertIsNone(predictions["a"].grad)
        self.assertTrue(all(target.grad is None for target in targets.values()))
        availability["t"][0] = 1
        self.assertEqual(model.recon_loss(parts, targets).item(), 0.0)

    def test_restoration_and_granularity(self):
        from parm.data.masking import MODALITIES, apply_mask
        feats = {k: torch.ones(64, 10, 1) for k in MODALITIES}
        valid = {k: torch.ones(64, 10, dtype=torch.bool) for k in MODALITIES}
        for protocol in ("IMM", "FMM"):
            for rate in (0.0, 0.7, 1.0):
                rates = torch.full((64, 3), rate)
                _, observed = apply_mask(feats, valid, rates, protocol,
                                         torch.Generator().manual_seed(20))
                counts = torch.stack([observed[k].sum(1) for k in MODALITIES], 1)
                self.assertTrue((counts.sum(1) > 0).all())
                if protocol == "FMM":
                    self.assertTrue(((counts == 0) | (counts == 10)).all())
                if rate == 0:
                    self.assertTrue((counts == 10).all())
                if rate == 1:
                    self.assertTrue((counts[:, 0] == 10).all())
                    self.assertTrue((counts[:, 1:] == 0).all())

    def test_seeded_masks_are_identical_across_calls(self):
        from parm.data.masking import MODALITIES, apply_mask
        feats = {k: torch.ones(12, 8, 1) for k in MODALITIES}
        valid = {k: torch.ones(12, 8, dtype=torch.bool) for k in MODALITIES}
        rates = torch.full((12, 3), 0.7)
        for protocol in ("IMM", "FMM"):
            _, first = apply_mask(feats, valid, rates, protocol,
                                  torch.Generator().manual_seed(99))
            _, second = apply_mask(feats, valid, rates, protocol,
                                   torch.Generator().manual_seed(99))
            for k in MODALITIES:
                self.assertTrue(torch.equal(first[k], second[k]))
