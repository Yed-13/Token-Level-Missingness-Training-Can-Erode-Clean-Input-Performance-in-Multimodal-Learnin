"""Unpaired sensitivity calculation and complete archived-record coverage."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from summarize_archived_uncertainty import unpaired_change, summarize, ANALYSIS


class ArchivedUncertaintyTests(unittest.TestCase):
    def test_welch_matches_scipy_and_is_order_invariant(self):
        a,b = [71,82,79,80,81],[60,72,59]
        result = unpaired_change(a,b)
        reference = ttest_ind(b,a,equal_var=False).confidence_interval()
        np.testing.assert_allclose(result["ci95"],[reference.low,reference.high])
        self.assertEqual(result["ci95"],unpaired_change(a,list(reversed(b)))["ci95"])
        with self.assertRaises(ValueError):
            unpaired_change([1],[2,3])
        with self.assertRaises(ValueError):
            unpaired_change([1,2],[2,float("nan")])

    def test_archived_summary_reproduces_all_records(self):
        summary = json.loads((ANALYSIS/"data/archived_uncertainty.json").read_text())
        self.assertEqual(summarize(),summary)
        self.assertEqual(len(summary["records"]),11)
        for r in summary["records"]:
            d = r["change"]
            self.assertAlmostEqual(d["mean"],d["text_scarce"]["mean"]-d["benign"]["mean"])
            self.assertEqual(len(r["provenance"]),d["benign"]["n"]+d["text_scarce"]["n"])


if __name__ == "__main__":
    unittest.main()
