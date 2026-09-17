#!/usr/bin/env bash
# Gate 2: train the strong reconstruction baseline under EACH regime and
# evaluate under all four -> the 4x4 transfer matrix. Its column-wise Delta
# decides whether this project has a premise.
set -u
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for tr in U-lo U-hi T-frag NV-frag; do
    ./.venv/bin/python -m parm.train --dataset MOSI --method recon \
      --train-regime "$tr" --seed "$seed" --out results 2>/dev/null | tail -1
  done
done
