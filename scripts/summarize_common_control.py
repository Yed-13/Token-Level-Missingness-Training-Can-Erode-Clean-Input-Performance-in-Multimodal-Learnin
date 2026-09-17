"""Validate and archive the two pilot checkpoints' common-suite evaluations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    pilot = json.loads((ROOT / "analysis/data/control_pilot.json").read_text())
    evaluations, rows = [], []
    for condition, reference in zip(("loss0", "loss05"), pilot["rows"]):
        path = ROOT / f"results_controls_mps/{condition}/MOSI/IMM/recon__train-T-frag__seed0.common_eval.json"
        record = json.loads(path.read_text())
        config = {k: v for k, v in record["config"].items() if k not in ("lambda_recon", "out")}
        assert config == pilot["config"]
        assert record["config"]["lambda_recon"] == reference["coefficient"]
        for protocol in ("IMM", "FMM"):
            cells = record["cells_by_protocol"][protocol]
            assert abs(cells["NONE"]["acc2"] - reference["acc2"]) < 1e-10
            assert abs(cells["NONE"]["mae"] - reference["mae"]) < 1e-6
            rows.append(dict(coefficient=reference["coefficient"], protocol=protocol,
                             acc2=cells["T-frag"]["acc2"], mae=cells["T-frag"]["mae"]))
        evaluations.append(record)
    result = dict(rows=rows, evaluations=evaluations)
    (ROOT / "analysis/data/control_common_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
