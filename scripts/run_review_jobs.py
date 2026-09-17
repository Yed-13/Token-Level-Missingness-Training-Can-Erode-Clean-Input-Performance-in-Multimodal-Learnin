"""Bounded, resumable CUDA-only review jobs; no automatic experiment expansion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cuda_controls import verify_sources
from review_common import SUITES, suite_spec


def jobs(phase):
    if phase == "evaluate":
        for weight in ("loss0", "loss05"):
            for protocol in ("IMM", "FMM"):
                for regime in ("U-lo", "T-frag"):
                    for seed in range(5):
                        name = f"{weight}-{protocol}-{regime}-seed{seed}"
                        checkpoint = ROOT / f"results_controls_cuda/{weight}/MOSI/{protocol}/recon__train-{regime}__seed{seed}.pt"
                        output = ROOT / f"results_review_cuda/common/{name}.json"
                        yield name, output, ["scripts/review_evaluate.py", str(checkpoint), str(output)]
    else:
        for seed in range(5):
            for protocol, regime in (("IMM", "U-lo"), ("IMM", "T-frag"),
                                     ("FMM", "U-lo"), ("FMM", "T-frag"), ("IMM", "NONE")):
                name = f"{protocol}-{regime}-seed{seed}"
                output = ROOT / f"results_review_cuda/selection/{name}/result.json"
                yield name, output, ["scripts/review_train.py", "--protocol", protocol,
                                     "--regime", regime, "--seed", str(seed)]


def validate(path, phase):
    record = json.loads(path.read_text())
    if phase == "evaluate":
        if record["suite"] != suite_spec():
            raise ValueError(f"Suite mismatch: {path}")
        for proto in ("IMM", "FMM"):
            if set(record["cells"][proto]) != set(SUITES):
                raise ValueError("Incomplete test suite")
    else:
        if record["epochs_run"] != 40 or len(record["selections"]) != 2:
            raise ValueError("Incomplete selection study")
        for selection in record["selections"].values():
            if selection["evaluation"]["suite"] != suite_spec():
                raise ValueError("Selection evaluation suite mismatch")
            if not (path.parent / selection["checkpoint"]).is_file():
                raise ValueError("Missing checkpoint")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("evaluate", "train"))
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no local training")
    verify_sources()
    output = ROOT / "results_review_cuda"
    output.mkdir(exist_ok=True)
    lock = output / (args.phase + ".lock")
    lock.mkdir()
    running = []
    try:
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        pending = []
        for name, path, command in jobs(args.phase):
            if path.exists():
                validate(path, args.phase)
                print("VERIFIED", name, flush=True)
            else:
                pending.append((name, path, command))
        while pending or running:
            while pending and len(running) < args.jobs:
                if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                    raise RuntimeError("Less than 5 GiB free")
                name, path, command = pending.pop(0)
                log = (logs / f"{args.phase}-{name}.log").open("a")
                proc = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT,
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append((proc, log, name, path))
                print("START", name, flush=True)
            for item in list(running):
                proc, log, name, path = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                log.close()
                if proc.returncode:
                    raise RuntimeError(f"Job failed: {name} exit={proc.returncode}")
                validate(path, args.phase)
                print("DONE", name, flush=True)
            if running:
                time.sleep(3)
        print("COMPLETE", args.phase, flush=True)
    finally:
        for proc, log, *_ in running:
            proc.terminate()
            proc.wait()
            log.close()
        lock.rmdir()


if __name__ == "__main__":
    main()
