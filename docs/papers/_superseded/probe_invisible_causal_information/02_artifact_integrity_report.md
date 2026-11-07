# Artifact integrity report

This package was assembled from the git-tracked protocols, source, and unit tests in the local repository, together with operator-pasted Colab stdout from the completed real-model stages. It was **not** assembled from a local copy of the Drive result directories.

## What is present locally

- Source and runners under `src/` and `run_08r2_*.py`.
- Frozen preregistrations under `docs/experiments/`.
- Official letter constants in `src/othello_r2.py`: `OFFICIAL_0A_LETTER = "C / PARTIAL"`, `SECTION_53_STATUS = "PARTIAL"`, `OBSTRUCTION_READY = "NO"`.
- Fresh triplet hash constant `3ae44e1eabf2684cb3e0b5025ccb9b76cabad6a1a6176454999ebc4b5ce70972` in `src/othello_r2_anchor_interaction.py`.
- Local `artifacts/toy_othello/data/games.json` only. This is not the evaluation archive.

## What is missing locally

The Colab/Drive tree

`/content/drive/MyDrive/ml_research_artifacts/toy_othello/`

is not mounted in this workspace. The following result directories are therefore **NOT AVAILABLE FROM ARCHIVED ARTIFACTS** here:

- `results/08r2/` validation
- `results/08r2/fiber/`
- `results/08r2/fiber_spec/` including `fiber_spec_full_logits.npz`
- `results/08r2/fiber_cost/` and `fiber_cost_forensic/`
- `results/08r2/fiber_cost_fixed_js/`
- `results/08r2/null_state_transfer/`
- `results/08r2/state_semantics/`
- `results/08r2/symmetric_target/` leftover
- `results/08r2/symmetric_target_census_audit/`
- `results/08r2/symmetric_target_fresh/` including `fresh_full_logits.npz`
- `results/08r2/visible_null_factorial/`

Checkpoint `synthetic_model.pth` and the 08c probe pickle are also absent locally. Their SHA-256 values below were recorded in Colab dependency prints, not hashed in this workspace:

- checkpoint `aef97afb1e7ec9dbef1d9007174458830a5c428d62f1e2338128879b179bd5c4`
- probe `d587d61d7cba059ecc16059679120c96c7b660ad9fdf00aa18bab1806f3a2a31`

## Hash checks performed in this package

| Asset | Expected SHA-256 | Local verification |
|---|---|---|
| SPEC pair manifest | `2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3` | Protocol text only; JSON not present |
| NST pair manifest | `39abdc79090211ab4a69e3e5f79f14319231705640f231dcc146233a9f5092e6` | Protocol text only |
| FRESH triplet | `3ae44e1eabf2684cb3e0b5025ccb9b76cabad6a1a6176454999ebc4b5ce70972` | Matches git-tracked constant; Drive JSON absent |
| FRESH protocol | `663335988190e1fd0fe710643db3c55c6bcadd7311c84aeb55ba5cce4cd87231` | Colab stdout only |
| VNF triplet | `d7ceb056d16b5c089c8ea491ad162bc9ce72c6c2a502e2ceb5d8b9b53ac81d25` | Colab stdout only |
| Checkpoint | `aef97afb…` | NOT VERIFIED locally |
| Probe | `d587d61d…` | NOT VERIFIED locally |

## Consequence

Level 1 numerical verification (softmax, JS, Y, bootstrap from saved logits) **cannot be executed in this workspace**. The manuscript reports official letters and recorded means with explicit provenance tags. It does not claim that this package recomputed SPEC, fixed-JS, or NST from NPZ files.

Historical experiment directories on Drive must not be modified. This package writes only under `docs/papers/probe_invisible_causal_information/`.
