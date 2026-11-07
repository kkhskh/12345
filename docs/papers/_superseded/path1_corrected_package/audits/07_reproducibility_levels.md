# Reproducibility levels

Three levels are distinguished. Passing a higher-level engineering test is not claimed.

## Level 1 — Numerical verification from saved outputs

Recompute softmax, JS, \(Y=G-Q\), pairwise contrasts, coverage, and clustered bootstrap from archived logits, legal masks, and pair manifests.

| Experiment | Level 1 in this workspace | Evidence |
|---|---|---|
| 08R2 validation | NOT RUN | Arrays absent |
| FIBER geometry | NOT RUN | Contrast matrix / SVD report absent; rank 124/388 is a recorded protocol result |
| SPEC | NOT RUN | `fiber_spec_full_logits.npz` absent |
| Original COST | NOT RUN | Arrays absent |
| Fixed-JS | NOT RUN | Arrays absent |
| NST | NOT RUN | Arrays absent; pooled stdout from a prior Colab paste is archived in `reported_numbers.json` |
| SEM | Arithmetic identity only | Git-tracked mean-level correction |
| FRESH | NOT RUN from NPZ | Stage 5/6 numbers from operator-pasted stdout; Stage 6 reported 768/768 match **on Colab** |
| VNF | NOT RUN from NPZ | Stage 5/6 numbers from operator-pasted stdout; Stage 6 reported 128/128 **on Colab** |

This package’s `reproduce_tables.py` regenerates manuscript tables from `reported_numbers.json` with provenance tags. That is **not** Level 1.

## Level 2 — Intervention replay

Load the frozen checkpoint, reconstruct directions, and reproduce model logits.

Status: **NOT PERFORMED HERE**. Checkpoint file absent. Hardware used for the original Colab runs is **NOT AVAILABLE FROM ARCHIVED ARTIFACTS** (no recorded GPU model or wall-clock in this package).

A replay command is documented in `REPRODUCIBILITY.md`. It must be executed against the Drive tree. This package does not pretend it was executed.

## Level 3 — Training replay

Refit the 08c probe and the 64-dimensional PCA/ridge maps from original training data and frozen configs.

Status: **NOT PERFORMED**. Doing so would be a new scientific training run, which this assignment forbids. The frozen coefficients live on Drive (`null_state_pca_ridge.npz`).

## What an independent lab can do without Drive

- Read protocols and source.
- Run unit tests that check \(Y=G-Q\), JS symmetry, probe-null projection identities, and witness-hash constants.
- See that official letters and recorded means are internally consistent with the preregistered gates.

They cannot currently recompute the headline 48-pair or 32-triplet tables from raw logits in this workspace.
