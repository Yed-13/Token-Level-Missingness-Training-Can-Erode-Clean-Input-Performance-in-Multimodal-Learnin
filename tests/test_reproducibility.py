import subprocess
import sys
import unittest

import numpy as np

from parm.seeding import stable_eval_seed


class EvaluationSeedTests(unittest.TestCase):
    def test_seed_is_stable_across_processes(self):
        code = (
            "from parm.seeding import stable_eval_seed; "
            "print(stable_eval_seed('MOSI', 'U-lo', 0, 'mask'))"
        )
        values = [
            subprocess.check_output([sys.executable, "-c", code], text=True).strip()
            for _ in range(2)
        ]
        self.assertEqual(values[0], values[1])

    def test_streams_and_repeats_are_distinct(self):
        seeds = {
            stable_eval_seed("MOSI", "U-lo", repeat, stream)
            for repeat in range(5)
            for stream in ("mask", "rates")
        }
        self.assertEqual(len(seeds), 10)

    def test_seed_reconstructs_identical_rate_stream(self):
        seed = stable_eval_seed("SIMS", "T-frag", 3, "rates")
        left = np.random.default_rng(seed).uniform(size=16)
        right = np.random.default_rng(seed).uniform(size=16)
        np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
