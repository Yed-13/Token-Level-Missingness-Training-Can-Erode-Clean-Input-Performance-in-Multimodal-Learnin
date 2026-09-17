# Protocol control study

## Completed five-seed CUDA control (2026-09-17)

All 40 predefined conditions completed on the RTX PRO 6000 Blackwell server.
The CUDA study uses BF16/SDPA, PyTorch 2.8.0+cu128 and Transformers 4.57.6.
Source hashes and input hashes were verified before training. The partial
MPS study was stopped at the user's request and remains separate.

| Protocol | Reconstruction weight | Paired clean Acc-2 change (pp) | Paired MAE change |
|---|---:|---:|---:|
| Token-level | 0 | -22.96 +/- 8.56 | +0.553 +/- 0.117 |
| Token-level | 0.5 | -16.52 +/- 5.09 | +0.458 +/- 0.068 |
| Modality-level | 0 | -1.10 +/- 2.61 | +0.087 +/- 0.053 |
| Modality-level | 0.5 | -0.61 +/- 3.10 | +0.130 +/- 0.094 |

Values are paired means +/- population SDs over five training seeds. The
token-minus-modality contrast is negative in every seed at both weights.
The coefficient interaction is +5.95 +/- 14.65 pp, spanning both signs across
seeds. Persistence of the contrast at zero loss weight shows that auxiliary
reconstruction supervision is not necessary in this configuration. It does
not remove imputation or isolate an encoder-specific mechanism.

`paper/data/control_matrix_cuda.json` includes all seed values, exact
configuration/environment, raw-record hashes and selected-checkpoint hashes.
`scripts/summarize_cuda_controls.py` verifies the complete matrix; the
`--skip-tables` option runs on the training server without LaTeX dependencies.
Checkpoint retrieval is ongoing; the originals remain on the server.
The following sections retain the earlier measurement/pilot history.

## Completed observation-exposure measurement

The manuscript's observation-exposure table is generated from
`paper/data/masking_exposure.json`. It uses all 1,284 MOSI training validity masks,
ten repetitions per regime/protocol, shared rate draws across protocols, and the
production masking implementation. The input SHA-256 is checked before loading.

Reproduce with the training environment:

```bash
python scripts/audit_mask_exposure.py
python paper/make_tables.py
```

This measurement quantifies intact text, absent text, and reconstruction-eligible
pairs after restoration. It does not measure task performance or assign causal
contributions to individual training components.

## Reconstruction-loss control

The local control uses the full MOSI split, BERT, the reconstruction architecture,
batch size 32, 40-epoch maximum, eight-epoch patience, and deterministic validation
masks. Auxiliary reconstruction coefficients are 0 and 0.5. The reconstruction
heads remain in the prediction path in both conditions. The complete-input target
forward pass is retained at coefficient zero, keeping its dropout random-number
consumption in the training procedure.

The local runtime is PyTorch 2.14.0 / Transformers 4.57.6 on Metal, using full
precision and eager BERT attention to support attention dropout. Its environment
differs from the archived CUDA studies; results belong to a separate study.
Key package versions are in `requirements-controls-local.txt`.

Selected checkpoints are saved with `--save-checkpoint 1`. Re-evaluate one on
both missingness protocols using:

```bash
python scripts/evaluate_control_checkpoint.py PATH_TO_CHECKPOINT
```

The evaluator uses the same test ordering, batch size and deterministic mask
streams for compared checkpoints. It writes a separate `common_eval.json` file
and refuses to overwrite an existing evaluation. It must be run from the project
root when the saved dataset path is relative.

## Interpretation and completion criteria

The planned 40-run matrix is now being executed with
`scripts/run_reconstruction_controls.py`. It resumes compatible completed
records and runs the remaining conditions sequentially. The analysis plan is
in `CONTROL_ANALYSIS_PLAN.md`; the complete-matrix summarizer requires all five
seeds for every condition. The completed pilot below remains the available
evidence until the matrix has finished and been validated.

Progress on 2026-09-17: the two seed-zero token-level benign runs have also
completed, bringing the completed count to four. Their clean Acc-2 / MAE are
0.8369 / 0.8016 at coefficient zero and 0.8186 / 0.7958 at coefficient 0.5.
The runner has advanced to seed-zero modality-level benign training. These
partial results have not been promoted to five-seed manuscript claims. There
is no scheduled continuation enabled; creating one requires user approval.

Two full training runs have completed for token-level text-scarce MOSI, seed
zero. Both used the prescribed early-stopping rule, ran 12 epochs and selected
epoch four. Clean Acc-2 / MAE are 0.5442 / 1.4150 at coefficient zero and
0.5732 / 1.3813 at coefficient 0.5. The comparison does not show a clean-input
benefit from removing the auxiliary loss in this pilot. Run records are copied
into `paper/data/control_runs/` with checksums in `paper/data/control_pilot.json`.
The checkpoint files remain in `results_controls_mps/`.

Common-suite evaluation has also completed for both checkpoints. All four
missingness regimes were tested under both operators with five deterministic
mask repetitions, plus clean evaluation. Re-evaluation reproduced both original
clean scores. `paper/data/control_common_eval.json` contains the full results;
the appendix table reports the text-scarce test condition. Its token-erasure
Acc-2 values are 0.5521 and 0.5826, and whole-modality-dropout Acc-2 is 0.5390
for both coefficients. These are comparisons between two token-erasure-trained
models, not between token- and modality-dropout-trained models.


- A coefficient comparison within one regime is an auxiliary-loss sensitivity
  check. It is not the benign-to-text-scarce clean-input change.
- Estimating that change requires both training regimes for every compared
  coefficient, protocol and seed.
- Seed-level pilot outcomes remain descriptive. The planned five-seed matrix
  supports reporting paired-seed means and variation.
- This control retains one fusion architecture. Replication on a published
  architecture is a separate experiment.
- `--validation-regime` enables a specified validation regime for sensitivity
  runs; test scores remain excluded from checkpoint selection.
