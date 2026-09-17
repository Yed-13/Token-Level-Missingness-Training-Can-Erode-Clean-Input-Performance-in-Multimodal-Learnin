"""Review-control analysis tests; synthetic fixtures are not experimental evidence."""
from pathlib import Path
import sys
import unittest
import json
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_review import describe, summarize_group, paired_difference, common_summary, selection_summary
from review_common import SUITES


class ReviewRevisionTests(unittest.TestCase):
    def test_sample_sd_and_interval(self):
        s = describe([1,2,3,4,5])
        self.assertAlmostEqual(s["sd"], np.sqrt(2.5))
        self.assertAlmostEqual(s["mean"], 3)
        self.assertAlmostEqual(s["ci95"][1]-3, 2.7764451051977987/np.sqrt(2), places=6)
        with self.assertRaises(ValueError):
            describe([1,float("nan")])

    def test_common_suite_and_mixture_are_seed_paired(self):
        records = []
        for seed in range(5):
            clean = .8 + seed*.01
            missing = .6 + seed*.01
            cells = {"NONE": {"acc2":clean, "mae":1-clean}}
            for proto in ("IMM","FMM"):
                cells[proto] = {r:{"acc2":missing,"mae":1-missing} for r in SUITES}
            records.append({"cells":cells})
        group = summarize_group(records)
        self.assertAlmostEqual(group["mixture"]["acc2"]["0.0"]["mean"], .62)
        self.assertAlmostEqual(group["mixture"]["acc2"]["1.0"]["mean"], .82)
        self.assertAlmostEqual(group["mixture"]["acc2"]["0.5"]["mean"], .72)
        same = paired_difference(group,group)
        self.assertEqual(same["values"], [0]*5)

    def test_incomplete_matrices_rejected(self):
        for summarize in (common_summary,selection_summary):
            with self.assertRaises(ValueError):
                summarize([])

    def test_selection_pairing_and_validation_rejection(self):
        # Fixture files live under the project only to exercise provenance paths;
        # TemporaryDirectory removes them after this test, never treating them as runs.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            paths = []
            for seed in range(5):
                for proto,regime in (("IMM","U-lo"),("IMM","T-frag"),
                                    ("FMM","U-lo"),("FMM","T-frag"),("IMM","NONE")):
                    score = .8 + seed*.01
                    if regime == "T-frag":
                        score -= .2 if proto=="IMM" else .02
                    cells = {"NONE":{"acc2":score,"mae":1-score}}
                    for test in ("IMM","FMM"):
                        cells[test] = {r:{"acc2":score,"mae":1-score} for r in SUITES}
                    record = dict(config=dict(protocol=proto,train_regime=regime,seed=seed,
                                              lambda_recon=.5,validation_regime="NONE",amp="bf16",max_epochs=40),
                                  epochs_run=40,early_stopping=False,n_train=1284,source_sha256="synthetic",
                                  torch="synthetic",attention="synthetic",
                                  history=[dict(epoch=e,clean_validation_mae=1/e) for e in range(1,41)],
                                  selections={m:dict(epoch=40,evaluation=dict(suite={"synthetic":True},cells=cells))
                                              for m in ("clean_best","fixed40")})
                    path = Path(directory)/f"{proto}-{regime}-{seed}.json"
                    path.write_text(json.dumps(record))
                    paths.append(path)
            result = selection_summary(paths)
            self.assertEqual(result["n_trajectories"],25)
            self.assertAlmostEqual(result["protocol_contrasts"]["clean_best/acc2"]["mean"],-18)
            self.assertAlmostEqual(result["clean_anchor_changes"]["fixed40/IMM/T-frag/acc2"]["mean"],-20)
            record = json.loads(paths[0].read_text())
            record["selections"]["clean_best"]["epoch"] = 2
            paths[0].write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"selection disagrees"):
                selection_summary(paths)

    def test_completed_common_records_match_original_checkpoints(self):
        summary = json.loads((ROOT/"analysis/data/review_common.json").read_text())
        paths = [ROOT/p["path"] for p in summary["provenance"]]
        self.assertEqual(common_summary(paths)["groups"],summary["groups"])
        previous = json.loads((ROOT/"analysis/data/control_matrix_cuda.json").read_text())
        self.assertEqual({p["checkpoint_sha256"] for p in previous["provenance"]},
                         {json.loads(p.read_text())["checkpoint_sha256"] for p in paths})
    def test_completed_selection_records_reproduce_summary(self):
        summary = json.loads((ROOT/"analysis/data/review_selection.json").read_text())
        paths = [ROOT/p["path"] for p in summary["provenance"]]
        recomputed = selection_summary(paths)
        self.assertEqual(len(paths),25)
        for key,value in recomputed.items():
            self.assertEqual(value,summary[key])
        for mode in ("clean_best","fixed40"):
            # Every mixture is affine: endpoint ordering establishes the full range.
            for weight in ("0.0","1.0"):
                mod = summary["groups"][f"{mode}/FMM/T-frag"]["mixture"]["acc2"][weight]["mean"]
                for proto,regime in (("IMM","T-frag"),("IMM","NONE")):
                    other = summary["groups"][f"{mode}/{proto}/{regime}"]["mixture"]["acc2"][weight]["mean"]
                    self.assertGreater(mod,other)


if __name__ == "__main__":
    unittest.main()
