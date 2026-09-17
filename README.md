# Clean-input performance under missingness training

Experiment code and results for *Token-Level Missingness Training Can Erode
Clean-Input Performance in Multimodal Learning*.

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
Statistics are descriptive, with seed-paired differences and population standard
deviations. Tests recompute the control summary and verify raw-record hashes.

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

## Layout

- `parm/`: data loading, masking, models, training, and evaluation.
- `scripts/`: experiment runners, audits, and summary generation.
- `analysis/`: study registry, numerical summaries, and provenance records.
- `results*/`: per-run configurations and measurements.
- `tests/`: reproducibility, evidence, and control-design checks.
- `datasets/alpha_*.json`: calibration metadata.

This code-and-results release excludes manuscript PDFs, LaTeX sources,
publication figures, review notes, dataset binaries, and model weights.
