"""Analysis tests use synthetic fixtures; they are not experimental results."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_reconstruction_controls import configurations, result_path
from summarize_reconstruction_controls import summarize
from run_cuda_controls import cuda_configurations, verify_sources


class ControlMatrixTests(unittest.TestCase):
    def records(self):
        records = []
        for cfg in configurations():
            # Seed-dependent offsets must cancel inside each paired difference.
            acc = 0.8 + 0.01 * cfg["seed"]
            if cfg["train_regime"] == "T-frag":
                acc -= 0.2 if cfg["protocol"] == "IMM" else 0.02
                acc += 0.02 * cfg["lambda_recon"]
            records.append(dict(config=cfg, env={"device": "synthetic-fixture"},
                                cells={"NONE": {"acc2": acc, "mae": 2 - acc}}))
        return records

    def test_exact_matrix_and_unique_paths(self):
        configs = list(configurations())
        self.assertEqual(len(configs), 40)
        self.assertEqual(len({result_path(c) for c in configs}), 40)
        self.assertEqual({c["seed"] for c in configs[:8]}, {0})

    def test_cuda_matrix_changes_only_output_namespace(self):
        local = list(configurations())
        remote = list(cuda_configurations())
        self.assertEqual(len(remote), 40)
        for before, after in zip(local, remote):
            self.assertEqual(after["out"], before["out"].replace("mps", "cuda"))
            self.assertEqual({k: v for k, v in before.items() if k != "out"},
                             {k: v for k, v in after.items() if k != "out"})
        verify_sources()

    def test_pairing_and_interaction(self):
        result = summarize(self.records())
        self.assertEqual(result["n_runs"], 40)
        rows = {(r["protocol"], r["coefficient"]): r for r in result["rows"]}
        self.assertAlmostEqual(rows[("IMM", 0.0)]["acc2"]["change"]["mean"], -20)
        self.assertAlmostEqual(rows[("FMM", 0.5)]["acc2"]["change"]["mean"], -1)
        self.assertAlmostEqual(rows[("IMM", 0.0)]["acc2"]["change"]["sd"], 0)
        self.assertAlmostEqual(result["protocol_change_contrasts"]["acc2"]["0.0"]["mean"], -18)
        self.assertAlmostEqual(result["protocol_change_contrasts"]["acc2"]["interaction"]["mean"], 0)

    def test_incomplete_duplicate_or_mismatched_records_rejected(self):
        original = self.records()
        for records in (original[:-1], original + [original[0]]):
            with self.assertRaises(ValueError):
                summarize(records)
        records = copy.deepcopy(original)
        records[0]["config"]["text_encoder"] = "different"
        with self.assertRaisesRegex(ValueError, "Unmatched configurations"):
            summarize(records)
        records = copy.deepcopy(original)
        records[0]["env"]["device"] = "different"
        with self.assertRaisesRegex(ValueError, "Unmatched runtime"):
            summarize(records)

    def test_nonfinite_results_rejected(self):
        records = self.records()
        records[0]["cells"]["NONE"]["mae"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize(records)

    def test_completed_cuda_study_matches_hashed_records(self):
        report = json.loads((ROOT / "paper/data/control_matrix_cuda.json").read_text())
        self.assertEqual(len(report["provenance"]), 40)
        records = []
        for item in report["provenance"]:
            content = (ROOT / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
            self.assertEqual(len(item["checkpoint_sha256"]), 64)
            records.append(json.loads(content))
        recalculated = summarize(records)
        for key in ("rows", "protocol_change_contrasts", "env", "config"):
            self.assertEqual(report[key], recalculated[key])
        self.assertEqual(report["env"]["device"], "cuda")
        expected = {("IMM", 0.0): -22.96, ("IMM", 0.5): -16.52,
                    ("FMM", 0.0): -1.10, ("FMM", 0.5): -0.61}
        table = (ROOT / "paper/tables/control_matrix_cuda.tex").read_text()
        for row in report["rows"]:
            change = row["acc2"]["change"]
            self.assertEqual(round(change["mean"], 2), expected[row["protocol"], row["coefficient"]])
            self.assertIn(f"{change['mean']:+.2f} $\\pm$ {change['sd']:.2f}", table)
        contrasts = report["protocol_change_contrasts"]["acc2"]
        self.assertTrue(all(x < 0 for w in ("0.0", "0.5") for x in contrasts[w]["values"]))
        self.assertEqual(round(contrasts["interaction"]["mean"], 2), 5.95)
        self.assertEqual(round(contrasts["interaction"]["sd"], 2), 14.65)


if __name__ == "__main__":
    unittest.main()
