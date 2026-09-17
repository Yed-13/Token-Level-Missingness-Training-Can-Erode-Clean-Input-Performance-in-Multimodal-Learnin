"""Check the manuscript's selected numerical claims against archived records."""
from pathlib import Path
import re
import json
import hashlib
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from studies import clean_acc, clean_mae, load, select


class ResultEvidenceTests(unittest.TestCase):
    def paired(self, study, metric, **filters):
        records = load(study)
        groups = []
        for regime in ("U-lo", "T-frag"):
            rows = select(records, train_regime=regime, **filters)
            pairs = {r["config"]["seed"]: metric(r) for r in rows}
            self.assertEqual(len(pairs), len(rows), "duplicate seed")
            self.assertTrue(pairs)
            self.assertNotIn(None, pairs.values())
            groups.append(pairs)
        u, t = groups
        self.assertEqual(set(u), set(t))
        return np.array([t[s] - u[s] for s in sorted(u)])

    def test_headline_accuracy_changes(self):
        for study, dataset, protocol, n_train, expected in [
            ("main", "MOSI", "IMM", "full", -20.8),
            ("roberta", "MOSI", "IMM", "full", -27.7),
            ("roberta", "MOSEI", "FMM", 1284, -2.2),
            ("unk", "MOSI", "IMM", "full", -29.6),
        ]:
            with self.subTest(study=study, protocol=protocol):
                changes = self.paired(study, clean_acc, dataset=dataset,
                                      protocol=protocol, n_train=n_train)
                self.assertEqual(round(100 * changes.mean(), 1), expected)

    def test_regression_changes(self):
        expected = {("MOSI", "IMM"): .510, ("MOSI", "FMM"): .063,
                    ("MOSEI", "IMM"): .078, ("MOSEI", "FMM"): .016,
                    ("SIMS", "IMM"): .039, ("SIMS", "FMM"): -.021}
        for (dataset, protocol), value in expected.items():
            changes = self.paired("main", clean_mae, dataset=dataset,
                                  protocol=protocol, n_train="full")
            self.assertEqual(len(changes), 5)
            self.assertEqual(round(changes.mean(), 3), value)

    def test_within_protocol_onset(self):
        for dataset, expected in [("MOSI", .5), ("SIMS", .6)]:
            rows = select(load("sweep"), dataset=dataset, protocol="IMM")
            by_rate = {}
            for r in rows:
                rate = float(r["config"]["train_regime"].split("@")[1])
                by_rate.setdefault(rate, []).append(clean_acc(r))
            reference = np.mean(by_rate[0.0])
            onset = min(rate for rate, values in by_rate.items()
                        if 100 * (reference - np.mean(values)) > 2)
            self.assertEqual(onset, expected)

    def test_mask_exposure_table_matches_measurements(self):
        record = json.loads((ROOT / "analysis/data/masking_exposure.json").read_text())
        self.assertEqual(record["n"], 1284)
        self.assertEqual(record["repeats"], 10)
        for row in record["rows"]:
            means = np.mean(row["repeat_values"], axis=0)
            np.testing.assert_allclose(means, [row["text_intact"], row["text_absent"],
                                             row["reconstruction_pairs_per_example"]])

    def test_pilot_summary_matches_completed_runs(self):
        result = json.loads((ROOT / "analysis/data/control_pilot.json").read_text())
        self.assertEqual([row["coefficient"] for row in result["rows"]], [0.0, 0.5])
        for row in result["rows"]:
            path = ROOT / row["result_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["result_sha256"])
            run = json.loads(path.read_text())
            self.assertEqual(row["acc2"], run["cells"]["NONE"]["acc2"])
            self.assertEqual(row["mae"], run["cells"]["NONE"]["mae"])
            self.assertEqual(run["epochs_run"], 12)
            self.assertEqual(run["best_epoch"], 3)

    def test_common_suites_reproduce_clean_scores(self):
        pilot = json.loads((ROOT / "analysis/data/control_pilot.json").read_text())
        common = json.loads((ROOT / "analysis/data/control_common_eval.json").read_text())
        self.assertEqual(len(common["evaluations"]), 2)
        self.assertEqual(len(common["rows"]), 4)
        for record, reference in zip(common["evaluations"], pilot["rows"]):
            for protocol in ("IMM", "FMM"):
                cells = record["cells_by_protocol"][protocol]
                self.assertEqual(cells["NONE"]["acc2"], reference["acc2"])
                self.assertAlmostEqual(cells["NONE"]["mae"], reference["mae"], places=6)

    def test_tokenization_audit_defines_true_truncation(self):
        audit = json.loads((ROOT / "analysis/data/tokenization_audit.json").read_text())
        self.assertEqual(audit["max_length"], 50)
        rows = audit["rows"]
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertAlmostEqual(row["truncation_fraction"], row["count_truncated"] / row["n"])
            self.assertGreaterEqual(row["mean_untruncated_length"], row["mean_capped_length"])
            self.assertEqual(row["all_caps_fraction"], 1)
            if row["model"] == "roberta-base" and row["lowercase"]:
                self.assertTrue(row["matches_archived_ids"])
                self.assertTrue(row["matches_archived_mask"])
        training = [r for r in rows if r["partition"] == "train"]
        self.assertEqual([r["count_truncated"] for r in training], [7, 46, 8])
        self.assertEqual([r["count_exactly_at_limit"] for r in training], [2, 3, 2])


if __name__ == "__main__":
    unittest.main()
