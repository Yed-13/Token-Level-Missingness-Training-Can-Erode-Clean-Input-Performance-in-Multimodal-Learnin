import importlib.util
from pathlib import Path
import unittest


class ControlPlanTests(unittest.TestCase):
    def test_matched_conditions_and_unique_outputs(self):
        path = Path(__file__).resolve().parents[1] / "scripts/plan_reconstruction_control.py"
        spec = importlib.util.spec_from_file_location("control_plan", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        commands = list(module.commands())
        self.assertEqual(len(commands), 40)
        outputs = set()
        groups = {}
        for cmd in commands:
            options = dict(zip(cmd[3::2], cmd[4::2]))
            self.assertEqual(options["--method"], "recon")
            self.assertEqual(options["--text-encoder"], "bert")
            output = tuple(options[k] for k in
                           ("--out", "--protocol", "--train-regime", "--seed"))
            self.assertNotIn(output, outputs)
            outputs.add(output)
            key = tuple(options[k] for k in ("--protocol", "--train-regime", "--seed"))
            groups.setdefault(key, set()).add(options["--lambda-recon"])
        self.assertEqual(len(groups), 20)
        self.assertTrue(all(values == {"0.0", "0.5"} for values in groups.values()))


if __name__ == "__main__":
    unittest.main()
