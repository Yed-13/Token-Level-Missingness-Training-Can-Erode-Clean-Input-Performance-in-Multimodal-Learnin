# Five-seed reconstruction-loss control: analysis plan

Specified on 2026-09-17 while the first additional benign-regime run was
training. The two seed-zero, token-level text-scarce pilot outcomes were already
known. This is a documented analysis plan, not an external preregistration.

## Question and fixed design

Does the contrast in benign-to-text-scarce clean-input change between token-
and modality-level training persist when auxiliary reconstruction supervision
is removed?

- Full CMU-MOSI training partition; existing validation and test partitions.
- BERT reconstruction-based fusion architecture, unchanged across conditions.
- Factors: training operator (IMM/FMM), regime (U-lo/T-frag), auxiliary loss
  coefficient (0/0.5); training seeds 0–4 for every cell: 40 runs in total.
- Original 40-epoch maximum, eight-epoch patience, batch size 32 and learning
  rates. Checkpoint selection uses training-regime validation MAE, two fixed
  corruption draws. Test data do not select checkpoints or hyperparameters.
- Local Metal FP32/eager-attention runtime, reported separately from archived
  CUDA results. The two compatible pilot runs are retained, not rerun until
  a preferred result is obtained.
- Reconstruction heads and the detached complete-input target forward pass
  remain active at both loss weights. The intervention removes the auxiliary
  objective, not the prediction pathway for fully absent streams.

## Primary summaries

For each operator and coefficient, report complete-input Acc-2 and MAE under
both training regimes. Compute the text-scarce-minus-benign change within each
seed before reporting its mean and population standard deviation.

Then compute the token-minus-modality contrast between these paired changes,
separately at each coefficient. The coefficient interaction is the contrast
at 0.5 minus that at zero, also computed within seed. Acc-2 changes use percentage
points; MAE changes use the original sentiment scale.

All five seeds enter every summary. No seed exclusion, alternative checkpoint
selection, tuned stopping rule or best-seed presentation is planned. Missing or
failed runs block the complete-matrix summary; they are not silently dropped.
The estimates are descriptive, without converting five seeds into a population
confidence claim.

## Interpretation boundaries

Persistence at coefficient zero would show that the auxiliary reconstruction
objective is not necessary for the protocol contrast in this configuration.
Attenuation or reversal would restrict that interpretation. Either outcome must
be reported. The interaction does not identify a unique causal mechanism:
observation distributions, availability flags and use of reconstructed features
still differ between training operators.

Stable-seed reruns also provide an independent local check of the archived
pattern. Because runtime and execution seeds differ, they do not isolate the
effect of replacing historical process-dependent validation hashes.

This study supplies no independent fusion architecture or additional dataset.
The paper's architectural scope remains explicit.

## Compute migration amendment (2026-09-17, before CUDA outcomes)

The user supplied a dedicated RTX PRO 6000 Blackwell server to accelerate
completion and requested that local training stop. Completed MPS records are
preserved as a partial supplementary study. The full 40-condition factorial
design will be run afresh on CUDA; no MPS record fills a CUDA cell. The
predefined paired contrasts and all five seeds are unchanged.

CUDA runs use the unchanged production training source, its default BF16
autocast, and CUDA-supported attention implementation. This differs from the
MPS execution's effective FP32/eager attention and is recorded in provenance.
All CUDA conditions share one environment. Four independent processes use
four CPU threads each on the allocated 22-core, 110-GB host with one 96-GB
GPU. Data and BERT weight hashes must match the local inputs. No test outcome
is used to select the hardware, precision, hyperparameters, or seeds.
