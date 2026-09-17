"""Run the complete, separate CUDA factorial study with bounded concurrency.

Uses the unchanged training implementation and original CUDA BF16 default.
No local Metal result is used to fill a CUDA condition.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_reconstruction_controls import configurations

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_controls_cuda"


def cuda_configurations():
    for cfg in configurations():
        cfg["out"] = cfg["out"].replace("results_controls_mps", "results_controls_cuda")
        yield cfg


def result_path(cfg):
    return ROOT / cfg["out"] / "MOSI" / cfg["protocol"] / (
        f"recon__train-{cfg['train_regime']}__seed{cfg['seed']}.json")


def validate(path, cfg):
    record = json.loads(path.read_text())
    for key, value in cfg.items():
        if record["config"][key] != value:
            raise ValueError(f"Configuration mismatch: {path}: {key}")
    assert record["env"]["device"] == "cuda"
    assert record["config"]["amp"] == "bf16"
    assert record["config"]["max_epochs"] == 40
    assert record["config"]["patience"] == 8
    assert record["config"]["validation_regime"] == ""
    assert record["n_train"] == 1284
    assert record["history"] and record["cells"]["NONE"]
    assert path.with_suffix(".pt").is_file()
    return record


def verify_sources():
    snapshot = json.loads((ROOT / "paper/data/control_implementation.json").read_text())
    for name, digest in snapshot["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Training source changed: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    verify_sources()
    OUTPUT.mkdir(exist_ok=True)
    lock = OUTPUT / "matrix.lock"
    lock.mkdir()
    running = []
    try:
        logs = OUTPUT / "logs"
        logs.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        pending = []
        for index, cfg in enumerate(cuda_configurations(), 1):
            path = result_path(cfg)
            if path.exists():
                validate(path, cfg)
                print(f"VERIFIED {index}/40 {path.relative_to(ROOT)}", flush=True)
            elif path.with_suffix(".pt").exists():
                raise FileExistsError(f"Checkpoint without result: {path}")
            else:
                pending.append((index, cfg, path))
        while pending or running:
            while pending and len(running) < args.jobs:
                if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                    raise RuntimeError("Less than 5 GiB free")
                index, cfg, path = pending.pop(0)
                name = f"{cfg['protocol']}-{cfg['train_regime']}-{cfg['lambda_recon']}-seed{cfg['seed']}"
                command = [sys.executable, "-u", "-m", "parm.train"]
                for key, value in cfg.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                log = (logs / f"{name}.log").open("a")
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                running.append((proc, log, index, cfg, path))
                print(f"START {index}/40 {name}", flush=True)
            for item in list(running):
                proc, log, index, cfg, path = item
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                running.remove(item)
                if code:
                    raise RuntimeError(f"Run {index}/40 failed with exit code {code}")
                rec = validate(path, cfg)
                print(f"DONE {index}/40 Acc2={rec['cells']['NONE']['acc2']:.6f} "
                      f"MAE={rec['cells']['NONE']['mae']:.6f} "
                      f"epochs={rec['epochs_run']}", flush=True)
            if running:
                time.sleep(5)
        print("COMPLETE: all 40 CUDA records verified", flush=True)
    finally:
        for proc, log, *_ in running:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            log.close()
        lock.rmdir()


if __name__ == "__main__":
    main()
