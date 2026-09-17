# Clean-input performance under missingness training

This repository contains the code, per-run results, and manuscript sources for
*Token-Level Missingness Training Can Erode Clean-Input Performance in
Multimodal Learning*.

Repository: https://github.com/Yed-13/Token-Level-Missingness-Training-Can-Erode-Clean-Input-Performance-in-Multimodal-Learnin

The manuscript PDF and editable LaTeX source are in `paper/`; the complete
40-run CUDA control results are in `paper/data/control_matrix_cuda.json`.

Incomplete multimodal systems are commonly trained with either partial
temporal erasure or whole-modality absence. At matched nominal rate schedules,
these choices have different effects when the trained system is evaluated on
complete input. Across CMU-MOSI, CMU-MOSEI, and CH-SIMS, high-rate token-level
erasure changes clean-input binary accuracy by as much as -27.7 points, while
the largest decrease with whole-modality absence is 2.2 points in the corresponding
comparisons. These are end-to-end system measurements; they do not isolate a
single encoder layer or gradient path.

## Reproduce the paper artifacts

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
./scripts/reproduce_paper.sh
```

The script recomputes the study summaries, regenerates tables and
figures from the bundled JSON results, tests selected numerical claims and
deterministic evaluation seeds,
and builds `paper/main.pdf` when `pdflatex` and `bibtex` are installed.

## Re-run experiments

```bash
pip install -r requirements-train.txt
python parm/data/download.py --root datasets
python -m parm.train --dataset MOSI --method recon --text-encoder bert \
  --protocol IMM --train-regime T-frag --seed 0 --out results_x
```

`--protocol IMM` applies position-level erasure; `--protocol FMM` applies
whole-modality absence. `--text-mask-mode remove|unk` selects attention-mask
removal or `[UNK]` substitution. Evaluation masks use a stable digest of the
dataset, regime, repeat, and stream names, so separate Python processes recover
the same evaluation samples for the same configuration. Archived runs predate
this seed correction; their masks were not guaranteed identical across
processes. Complete-input evaluations apply no corruption, while checkpoint
selection used sampled validation masks.

## Protocol controls

The measured masking-exposure study and local reconstruction-loss controls are
documented in `paper/CONTROL_STUDY.md`. The exposure measurements and source
checksums are stored in `paper/data/masking_exposure.json` and regenerate the
corresponding manuscript table without requiring training dependencies.
Re-measuring exposure requires the dataset and PyTorch:

```bash
python scripts/audit_mask_exposure.py
```

Training supports `--save-checkpoint 1` and `--validation-regime NONE` for a
predefined complete-input validation sensitivity run. The latter changes the
validation condition, not the training corruption. To evaluate a selected
reconstruction-model checkpoint under both test corruption protocols:

```bash
python scripts/evaluate_control_checkpoint.py PATH_TO_CHECKPOINT
```

Local Metal controls use eager BERT attention to preserve attention dropout.
Their results are kept separate from archived CUDA studies.

The completed auxiliary-loss control matrix uses five seeds, two training
operators, two regimes and two loss coefficients (40 fresh CUDA runs):

```bash
python scripts/run_cuda_controls.py --jobs 4
python scripts/summarize_cuda_controls.py
```

The runner executes up to four jobs at a time, validates completed records before
skipping them, saves selected checkpoints, and refuses conflicting outputs.
Its training logs are in `results_controls_cuda/logs/`. The summary requires all
40 matched records; it does not report a partial matrix as a complete study.
See `paper/CONTROL_ANALYSIS_PLAN.md` for the fixed contrasts and interpretation.
The complete summary and hashed raw records are bundled under
`paper/data/control_matrix_cuda*`; selected checkpoint hashes are included.
Token-level clean accuracy changes by -22.96 points at reconstruction weight
zero and -16.52 at weight 0.5, versus -1.10 and -0.61 for modality-level
training. These are seed-paired means, with variation reported in the paper.
The partial Metal study is retained separately and was stopped at the user's
request; do not combine its cells with the CUDA matrix.

## Repository layout

- `parm/`: data loading, masking, models, training, and analysis.
- `paper/`: manuscript, claim verifier, generated tables, and figures.
- `results_*/`: one JSON record per run, including configuration and metrics.
- `tests/`: reproducibility checks.
- `scripts/reproduce_paper.sh`: artifact-regeneration entry point.

The archived study comprises 388 fine-tuned-text-pathway runs and 24 frozen-feature
runs. Reported comparisons use seed-paired means and standard deviations where
paired runs are available. The analysis is descriptive and reports no
null-hypothesis tests. The primary reference is benign missingness training.
