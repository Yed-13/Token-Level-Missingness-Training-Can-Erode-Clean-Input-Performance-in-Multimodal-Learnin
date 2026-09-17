#!/usr/bin/env bash
# Gate 2, FMM protocol. IMM (temporal frame erasure) may be too MILD to
# distinguish regimes: at 70% erasure, 30% of frames survive and text is still
# partially readable. Under FMM a modality is entirely absent, which is where
# T-frag and NV-frag genuinely differ. If pattern shift costs anything, it
# should show up here or nowhere.
set -u
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for tr in U-lo U-hi T-frag NV-frag; do
    ./.venv/bin/python -m parm.train --dataset MOSI --method recon --protocol FMM \
      --train-regime "$tr" --seed "$seed" --out results 2>/dev/null | tail -1
  done
done
