"""Validate all 40 CUDA records/checkpoints and publish separate summaries."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cuda_controls import cuda_configurations, result_path, validate, verify_sources
from summarize_reconstruction_controls import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tables", action="store_true",
                        help="Accepted for compatibility; this release writes JSON summaries only")
    args = parser.parse_args()
    verify_sources()
    configs = list(cuda_configurations())
    paths = [result_path(cfg) for cfg in configs]
    records = [validate(path, cfg) for path, cfg in zip(paths, configs)]
    result = summarize(records)
    result["study"] = "cuda_factorial_20260917"
    archive = ROOT / "analysis/data/control_matrix_cuda_runs"
    archive.mkdir(exist_ok=True)
    provenance = []
    for path, cfg in zip(paths, configs):
        dest = archive / (f"{cfg['protocol']}-{cfg['train_regime']}-"
                          f"loss{cfg['lambda_recon']}-seed{cfg['seed']}.json")
        content = path.read_bytes()
        dest.write_bytes(content)
        checkpoint = path.with_suffix(".pt")
        with checkpoint.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        provenance.append(dict(path=str(dest.relative_to(ROOT)),
                               sha256=hashlib.sha256(content).hexdigest(),
                               checkpoint=str(checkpoint.relative_to(ROOT)),
                               checkpoint_sha256=digest))
    result["provenance"] = provenance
    (ROOT / "analysis/data/control_matrix_cuda.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("rows", "protocol_change_contrasts")}, indent=2))


if __name__ == "__main__":
    main()
