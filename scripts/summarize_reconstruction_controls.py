"""Validate and summarize the complete five-seed factorial control study.

No incomplete matrix is accepted. Statistics are descriptive population SDs;
the units of replication are training seeds, not test mask repetitions.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_reconstruction_controls import configurations, result_path, validate
from record_control_provenance import main as verify_implementation


def moments(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values)), values=values)


def summarize(records):
    indexed = {}
    common = None
    environment = None
    for rec in records:
        cfg = rec["config"]
        for metric in ("acc2", "mae"):
            value = rec["cells"]["NONE"][metric]
            if not np.isfinite(value):
                raise ValueError(f"Non-finite clean metric: {metric}")
        if not 0 <= rec["cells"]["NONE"]["acc2"] <= 1:
            raise ValueError("Accuracy is outside [0, 1]")
        key = (cfg["protocol"], cfg["lambda_recon"], cfg["train_regime"], cfg["seed"])
        if key in indexed:
            raise ValueError(f"Duplicate condition: {key}")
        indexed[key] = rec
        fixed = {k: v for k, v in cfg.items() if k not in
                 ("protocol", "lambda_recon", "train_regime", "seed", "out")}
        if common is not None and common != fixed:
            raise ValueError("Unmatched configurations")
        if environment is not None and environment != rec["env"]:
            raise ValueError("Unmatched runtime environments")
        common, environment = fixed, rec["env"]
    expected = {(p, w, r, s) for p in ("IMM", "FMM") for w in (0.0, 0.5)
                for r in ("U-lo", "T-frag") for s in range(5)}
    if set(indexed) != expected:
        raise ValueError(f"Incomplete/unexpected matrix: missing={expected-set(indexed)}, "
                         f"extra={set(indexed)-expected}")
    rows = []
    for protocol in ("IMM", "FMM"):
        for weight in (0.0, 0.5):
            row = dict(protocol=protocol, coefficient=weight, n_seeds=5)
            for metric in ("acc2", "mae"):
                by_regime = {r: [indexed[(protocol, weight, r, s)]["cells"]["NONE"][metric]
                                 for s in range(5)] for r in ("U-lo", "T-frag")}
                scale = 100 if metric == "acc2" else 1
                row[metric] = dict(benign=moments(by_regime["U-lo"]),
                                   text_scarce=moments(by_regime["T-frag"]),
                                   change=moments([scale * (t - u) for t, u in
                                                   zip(by_regime["T-frag"], by_regime["U-lo"])]))
            rows.append(row)
    by_condition = {(r["protocol"], r["coefficient"]): r for r in rows}
    contrasts = {}
    for metric in ("acc2", "mae"):
        contrast = {}
        for weight in (0.0, 0.5):
            tok = by_condition[("IMM", weight)][metric]["change"]["values"]
            mod = by_condition[("FMM", weight)][metric]["change"]["values"]
            contrast[str(weight)] = moments([t-m for t, m in zip(tok, mod)])
        contrast["interaction"] = moments([a-b for a, b in zip(
            contrast["0.5"]["values"], contrast["0.0"]["values"])])
        contrasts[metric] = contrast
    return dict(n_runs=40, seeds=list(range(5)), config=common, env=environment,
                rows=rows, protocol_change_contrasts=contrasts)


def main():
    verify_implementation()
    paths = [result_path(cfg) for cfg in configurations()]
    records = [validate(path, cfg) for path, cfg in zip(paths, configurations())]
    result = summarize(records)
    archive = ROOT / "paper/data/control_matrix_runs"
    archive.mkdir(exist_ok=True)
    provenance = []
    for path, rec in zip(paths, records):
        cfg = rec["config"]
        destination = archive / (f"{cfg['protocol']}-{cfg['train_regime']}-"
                                 f"loss{cfg['lambda_recon']}-seed{cfg['seed']}.json")
        content = path.read_bytes()
        destination.write_bytes(content)
        provenance.append(dict(path=str(destination.relative_to(ROOT)),
                               sha256=hashlib.sha256(content).hexdigest()))
    result["provenance"] = provenance
    (ROOT / "paper/data/control_matrix.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.path.insert(0, str(ROOT))
    from paper.make_tables import t10
    t10()
    print(json.dumps({k: result[k] for k in ("rows", "protocol_change_contrasts")}, indent=2))


if __name__ == "__main__":
    main()
