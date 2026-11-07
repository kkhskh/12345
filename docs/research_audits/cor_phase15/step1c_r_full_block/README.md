# Step 1C-R — one pre-norm block

Phase 1.5, Step 1. Not Step 2. Not Phase 2.

**Verdict:** `STEP1C_R_ONLY_GENERIC_COMPUTATIONAL_RESULTS_SURVIVE`

The corrected LayerNorm rank, the deletion argument, and the differentiated block are `02` through `09`. The register is `16`. The verdict and the thirty questions are `23_final_step1c_r_verdict.md`.

`PHASE_2 = NO GO`. `OBSTRUCTION_READY = NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` is unchanged.

No Transformer was run. The checks are finite maps in this directory. A passing check is arithmetic on those maps.

Run the commands below from this directory, the package root. They do not use a repository path and they do not need a `PYTHONPATH`.

## 1. Dependencies

```text
python -m pip install "numpy==1.26.4" "sympy==1.14.0" "pytest==8.4.2"
```

The hardcoded floats were produced by `numpy.random.default_rng` on NumPy 1.26.4. Another NumPy build can change that stream.

## 2. LayerNorm

```text
python -m pytest tests/test_block.py -q -k "layer_norm or positive_epsilon or singular_value or zero_gamma"
```

## 3. Two-path necessity

```text
python -m pytest tests/test_block.py -q -k "zero_bias or biased_addition"
```

## 4. Block derivative

```text
python -m pytest tests/test_block.py -q -k "block_directional"
```

## 5. Attention derivative against finite differences

```text
python -m pytest tests/test_block.py -q -k "attention_directional"
```

## 6. Causal interventions

```text
python -m pytest tests/test_block.py -q -k "shift_is_exact or value_only or zero_first_order or same_projection or head_derivatives"
```

## 7. Counterexamples

```text
python -m pytest tests/test_block.py -q -k "rank_one or exact_equations or value_only or same_projection or zero_first_order"
```

## 8. Full suite

```text
python -m pytest tests -q
```

## 9. Manifest

```text
python scripts/verify_manifest.py
```

The manifest lists only files in the archive. It does not list `__pycache__` or `.pyc` files. A cache created later by `pytest` is ignored by the checker and is not an archived asset.

## 10. Recreate the archive

From the directory that contains `cor_phase15/`:

```text
python cor_phase15/step1c_r_full_block/scripts/write_manifest.py
zip -r -X cor_phase15_step1c_r_full_block.zip cor_phase15/step1c_r_full_block -x '*__pycache__*' '*.pyc'
```

`write_manifest.py` hashes the files present at that moment. Run it before the zip, after the files are final.
