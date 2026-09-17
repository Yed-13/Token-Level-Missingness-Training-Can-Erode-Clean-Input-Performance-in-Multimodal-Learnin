# Clean-input performance under missingness training

Experiment code and results for *Token-Level Missingness Training Can Erode
Clean-Input Performance in Multimodal Sentiment Learning*.

## Reproduce numerical results

Python 3.11 is recommended. No GPU or dataset download is needed for this check.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python analysis/summarize_results.py
python -m unittest discover -s tests -q
```

The analysis keeps the archived studies separate: 388 fine-tuned-text-pathway
runs, 24 frozen-feature runs, and the independent 40-run CUDA control matrix.
Archived summaries retain their original population-standard-deviation fields.
The review-control summaries use sample standard deviations and pointwise paired
t intervals across training seeds. Tests recompute summaries and verify
raw-record hashes. Intervals condition on fixed data splits and approximately
normal seed differences; they do not measure subset-selection uncertainty.

## Re-run experiments

```bash
pip install -r requirements-train.txt
python parm/data/download.py --root datasets
python -m parm.train --dataset MOSI --method recon --text-encoder bert \
  --protocol IMM --train-regime T-frag --seed 0 --out results_x
```

`IMM` applies temporal position-level erasure; `FMM` applies whole-modality
absence. `--text-mask-mode remove|unk` chooses attention-mask removal or `[UNK]`
substitution. Datasets are obtained from the upstream distributor and checked
against recorded SHA-256 digests; dataset binaries are not bundled.

Current evaluation masks use stable digest-based seeds. Archived studies predate
this correction and did not guarantee identical validation masks across Python
processes. Complete-input evaluation applies no corruption.

## Independent CUDA controls

The fixed matrix has five seeds (0–4), two protocols (IMM/FMM), two training
regimes (U-lo/T-frag), and two reconstruction-loss weights (0/0.5): 40 runs.
It uses full MOSI training data, BERT, batch size 32, at most 40 epochs, and
patience 8. Checkpoints are selected by validation MAE under the training
regime with two fixed mask repetitions. See `requirements-controls-cuda.txt`
for the recorded core environment.

On a provisioned CUDA host with inputs available:

```bash
python scripts/run_cuda_controls.py --jobs 4
python scripts/summarize_cuda_controls.py
```

The runner verifies training-source hashes and completed configurations before
skipping runs. The summarizer requires all 40 records and their checkpoints.
Released checkpoint hashes are in `analysis/data/control_matrix_cuda.json`;
the large checkpoint binaries are not included. Use the analysis command above
to inspect bundled results without checkpoints or training.

The prespecified contrasts are text-scarce minus benign within each protocol
and weight, the token-level minus modality-level contrast, and its change
between weights. The clean accuracy changes are -22.96 and -16.52 percentage
points for token-level training at weights 0 and 0.5, versus -1.10 and -0.61
for modality-level training. This is an end-to-end system comparison; it does
not isolate an encoder gradient path. Separate partial Metal-runtime results
must not be pooled with this matrix.

## Shared selection, clean anchor, and common test suites

The review study adds 25 full-MOSI trajectories: five seeds for each of
IMM/U-lo, IMM/T-frag, FMM/U-lo, FMM/T-frag, and a shared clean-training anchor
with all missing rates zero. Each trajectory runs exactly 40 epochs and yields
two checkpoints: minimum clean-validation MAE and fixed epoch 40. Both use the
same trajectory, not independent repetitions. Test scores never select epochs.

All 40 original CUDA checkpoints and these 50 selected checkpoints are evaluated
on common deterministic clean, token-erasure, and whole-modality suites. Each
corrupted suite averages five mask draws. Mixture scores vary clean-input weight
from 0 to 1, splitting the remainder equally between the two text-scarce suites;
these weights are sensitivity scenarios, not measured deployment frequencies.

Recompute the bundled results without training or checkpoints:

```bash
python scripts/summarize_review.py common
python scripts/summarize_review.py selection
python -m unittest discover -s tests -q
```

On a provisioned CUDA host, with the recorded inputs and original control
checkpoints available, run `python scripts/run_review_jobs.py evaluate --jobs 4`
and `python scripts/run_review_jobs.py train --jobs 4`. The runners refuse local
CPU/MPS execution. Released result directories already exist: use a separate
checkout without those output directories for a fresh replication. Checkpoint
binaries are not bundled. Training uses the pinned CUDA control environment.

Relative to benign training, token-level clean-accuracy changes are -8.96 points
under clean-validation selection and -16.28 points at epoch 40; modality-level
changes are +0.49 and -0.12 points. Relative to clean training, the text-scarce
token-level changes are -9.63 and -17.62 points, versus +0.15 and +0.24 points
for modality-level training. The result records include all seeds, histories,
selected epochs, common test scores, and checkpoint/source hashes.

## Layout

- `parm/`: data loading, masking, models, training, and evaluation.
- `scripts/`: experiment runners, audits, and summary generation.
- `analysis/`: study registry, numerical summaries, and provenance records.
- `results*/`: per-run configurations and measurements.
- `tests/`: reproducibility, evidence, and control-design checks.
- `datasets/alpha_*.json`: calibration metadata.
