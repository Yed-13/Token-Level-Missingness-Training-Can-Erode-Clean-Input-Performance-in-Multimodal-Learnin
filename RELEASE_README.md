# Manuscript and experiment release

## Contents

- `paper/main.pdf`: final 24-page manuscript, without a highlights page.
- `paper/main.tex`, `refs.bib`, `figs/`, `tables/`: editable LaTeX manuscript.
- `paper/Highlights.txt`: original highlights retained separately as text.
- `parm/`: training, masking, data preparation and analysis implementation.
- `scripts/`: reproducibility, auditing, control training and summary scripts.
- `tests/`: numerical, provenance, masking and implementation checks.
- `results*/*.json` (nested by dataset/protocol): raw experiment records.
- `paper/data/control_matrix_cuda.json`: complete 40-run CUDA summary,
  paired seed changes, interactions, record hashes and checkpoint hashes.
- `paper/data/control_matrix_cuda_runs/`: all 40 archived CUDA raw records.
- `paper/data/control_runs/`: supplementary two-run Metal pilot.
- `results_controls_mps/`: partial Metal study, stopped before completion.
- `MANIFEST.sha256`: SHA-256 digest for every packaged file except itself.

The CUDA matrix contains five seeds for every combination of training protocol,
training regime and reconstruction-loss coefficient. Do not combine its records
with the partial Metal matrix or historical CUDA studies. See
`paper/CONTROL_ANALYSIS_PLAN.md` and `paper/CONTROL_STUDY.md` for interpretation.

## Reproduce manuscript artifacts

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
bash scripts/reproduce_paper.sh
```

The full PDF build requires LaTeX with `elsarticle`, TikZ and BibTeX.
Numerical tests and figure/table generation use the bundled result records;
they do not need model checkpoints or datasets. The release was checked with
22 passing tests and a warning-free manuscript build.

For the LaTeX-only ZIP, compile in its root:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Training environment

The completed control study used Python 3.12.3, PyTorch 2.8.0+cu128,
Transformers 4.57.6, BF16 and SDPA on an RTX PRO 6000 Blackwell GPU.
`requirements-controls-cuda.txt` records the core package pins and PyTorch
installation command. `requirements-train.txt` is the broader environment
specification, not an exact lockfile for the completed study.

Training entry point: `python scripts/run_cuda_controls.py --jobs 4`.
This requires the verified MOSI dataset and cached BERT model. The input
preparation script verifies their hashes. After training,
`python scripts/summarize_cuda_controls.py` additionally requires the selected
checkpoints. The packaged summary and raw JSON records already suffice for
manuscript regeneration and tests.

## Intentionally excluded

Dataset pickle/NPZ files, pretrained-model caches, selected model checkpoints,
virtual environments, Git history, sample papers, server connection details,
transfer scripts and temporary LaTeX files are excluded. Checkpoint hashes are
included, but the approximately 17-GiB binary checkpoint collection is a
separate backup and is not part of these ZIP files.

## GitHub upload

Extract the GitHub ZIP and review its changes against the existing repository
before committing. Its `.gitignore` keeps checkpoints, datasets and build
temporary files out of Git while retaining experiment JSON records. Creating
this package does not commit or push anything to GitHub.
