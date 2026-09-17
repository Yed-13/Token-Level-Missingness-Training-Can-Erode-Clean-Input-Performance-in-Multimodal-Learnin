"""Resume the predefined 40-run local control matrix, one training job at a time.

Completed records are checked rather than overwritten. Every run has a log and
selected checkpoint. A directory lock prevents simultaneous matrix runners.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_controls_mps"


def configurations():
    # Finish complete factorial blocks before advancing the training seed.
    for seed in range(5):
        for protocol in ("IMM", "FMM"):
            for regime in ("U-lo", "T-frag"):
                for coefficient, condition in ((0.0, "loss0"), (0.5, "loss05")):
                    yield dict(dataset="MOSI", method="recon", text_encoder="bert",
                               protocol=protocol, train_regime=regime, seed=seed,
                               lambda_recon=coefficient,
                               out=f"results_controls_mps/{condition}",
                               save_checkpoint=1, num_workers=0, torch_threads=4,
                               verbose=1)


def result_path(cfg):
    return ROOT / cfg["out"] / "MOSI" / cfg["protocol"] / (
        f"recon__train-{cfg['train_regime']}__seed{cfg['seed']}.json")


def validate(path, cfg):
    record = json.loads(path.read_text())
    for key, value in cfg.items():
        if record["config"][key] != value:
            raise ValueError(f"Configuration mismatch at {path}: {key}")
    assert record["n_train"] == 1284
    assert record["env"]["device"] == "mps"
    assert record["config"]["max_epochs"] == 40
    assert record["config"]["patience"] == 8
    assert record["config"]["validation_regime"] == ""
    assert record["history"] and record["cells"]["NONE"]
    assert path.with_suffix(".pt").is_file()
    return record


def main():
    OUTPUT.mkdir(exist_ok=True)
    lock = OUTPUT / "matrix.lock"
    lock.mkdir()  # Fail closed if another runner is active.
    try:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        logs = OUTPUT / "logs"
        logs.mkdir(exist_ok=True)
        for index, cfg in enumerate(configurations(), 1):
            path = result_path(cfg)
            if path.exists():
                validate(path, cfg)
                print(f"VERIFIED {index}/40 {path.relative_to(ROOT)}", flush=True)
                continue
            if path.with_suffix(".pt").exists():
                raise FileExistsError(f"Checkpoint without result: {path}")
            if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB free; stopping before the next run")
            name = f"{cfg['protocol']}-{cfg['train_regime']}-{cfg['lambda_recon']}-seed{cfg['seed']}"
            command = [sys.executable, "-u", "-m", "parm.train"]
            for key, value in cfg.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            print(f"START {index}/40 {name}", flush=True)
            with (logs / f"{name}.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            rec = validate(path, cfg)
            clean = rec["cells"]["NONE"]
            print(f"DONE {index}/40 {name}: Acc2={clean['acc2']:.6f} "
                  f"MAE={clean['mae']:.6f} epochs={rec['epochs_run']}", flush=True)
        print("COMPLETE: all 40 training records verified", flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    main()
