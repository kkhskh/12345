# Reproducibility

This package is private. Do not upload artifacts, push a public branch, or submit the paper.

## Commands in this workspace (no GPU)

From `docs/papers/probe_invisible_causal_information/`:

```bash
python verified_data/verify_fixed_js.py
python reproduce_tables.py
python reproduce_figures.py
python verify_paper_claims.py
python -m pytest tests/test_paper_reproducibility.py -q
```

`verified_data/fixed_js_verified_data.csv` is the 336-row saved-output table used for Figure 2 and the matched-JS means. It is not a checkpoint replay.

These regenerate tables/figures from `reported_numbers.json` and check manuscript integrity. They do **not** replay Othello-GPT.

## Level 1 (requires Drive)

Mount the archive used on Colab:

`/content/drive/MyDrive/ml_research_artifacts/toy_othello/`

Then, without changing files, recompute \(Y\) from:

- `results/08r2/fiber_spec/fiber_spec_full_logits.npz`
- `results/08r2/fiber_cost_fixed_js/` (exact filename on Drive)
- `results/08r2/null_state_transfer/` logits
- `results/08r2/symmetric_target_fresh/fresh_full_logits.npz`
- `results/08r2/visible_null_factorial/` logits

using `src/othello_r2_fiber_spec.y_components` and the archived legal-set JSON. Bootstrap seeds: 82 (48-pair), 9482 (FRESH), 10482 (VNF).

This Level 1 run was **not** executed in the Path 1 workspace because the arrays are absent.

## Level 2 replay (requires checkpoint; not run here)

```bash
# Colab, after mounting Drive and cloning the repo at the frozen commit
python run_08r2_fiber_spec.py --help
python run_08r2_null_state_transfer.py --help
python run_08r2_symmetric_target_fresh.py --stage 6 --results-dir ...
python run_08r2_visible_null_factorial.py --stage 6 --results-dir ...
```

Stage 6 of FRESH and VNF is a read-only recomputation of saved logits. It is the intended verification, not a new experiment. Do not open confirmation. Do not retrain.

## Level 3

Not performed. Do not refit the 08c probe or the 64-D map for this package.

## Missing resources

Checkpoint and probe files, GPU identity, wall-clock, and Drive NPZ hashes verified from local bytes: **NOT AVAILABLE FROM ARCHIVED ARTIFACTS** here. Recorded SHA-256 strings are in `reported_numbers.json` from Colab prints.

## Redistribution

Third-party Othello-GPT weights follow Li et al. / Nanda redistribution notes. Legal permission to publish those files with a paper is **NOT VERIFIED**. Do not release them from this package.
