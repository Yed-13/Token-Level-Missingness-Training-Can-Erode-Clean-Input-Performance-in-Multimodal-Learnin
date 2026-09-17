"""Summarize the matched seed-zero auxiliary-loss pilot after both runs finish."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / f"results_controls_mps/{condition}/MOSI/IMM/recon__train-T-frag__seed0.json"
             for condition in ("loss0", "loss05")]
    records = [json.loads(path.read_text()) for path in paths]
    configs = [{k: v for k, v in rec["config"].items() if k not in ("lambda_recon", "out")}
               for rec in records]
    assert configs[0] == configs[1], "Pilot configurations are not matched"
    assert records[0]["env"] == records[1]["env"], "Pilot environments differ"
    assert [r["config"]["lambda_recon"] for r in records] == [0.0, 0.5]
    rows = []
    for path, rec in zip(paths, records):
        assert rec["n_train"] == 1284
        assert rec["config"]["seed"] == 0
        archived_path = ROOT / "analysis/data/control_runs" / (path.parents[2].name + ".json")
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        archived_path.write_bytes(path.read_bytes())
        rows.append(dict(coefficient=rec["config"]["lambda_recon"],
                         acc2=rec["cells"]["NONE"]["acc2"],
                         mae=rec["cells"]["NONE"]["mae"],
                         best_epoch=rec["best_epoch"], epochs_run=rec["epochs_run"],
                         result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         result_path=str(archived_path.relative_to(ROOT)),
                         source_result_path=str(path.relative_to(ROOT))))
    result = dict(config=configs[0], env=records[0]["env"], rows=rows)
    (ROOT / "analysis/data/control_pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
