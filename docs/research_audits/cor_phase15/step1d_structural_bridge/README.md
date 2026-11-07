# Step 1D — structural certificates and the relational question

Correction of the overnight reconstruction's reading of Theorems U and L, on the same pre-norm block, plus a direct attempt to find a joint-path finite-response bound that beats separate-path bounds under the same computational budget.

The overnight directory is unchanged. Phase 2 stays closed.

**Verdict:** `STEP1D_STRUCTURAL_CERTIFICATES_ONLY`

## Run

From this directory, with no `PYTHONPATH` set:

```text
python3 -m pytest -q --tb=line
python3 scripts/verify_manifest.py
```

Dependencies are `numpy==1.26.4`, `pytest==9.2.0`, and `sympy==1.14.0`. PyTorch is not installed in this environment. The second route is a loop-based NumPy forward with central differences, plus SymPy identities that do not call the block.

## What was proved

For the raw read-token intervention, with `ε>0`,

```text
|R| ≤ |1_{t=r} w·δ| + ||w|| (L_A L_1 + L_M L_2 (1 + L_A L_1)) ||δ||.
```

`L_A`, `L_M`, `L_1`, and `L_2` are global Lipschitz constants derived in `06_direction_specific_finite_bounds.md`. When `w·δ = 0` this is the audit formula. It is a separate-path certificate. It is not a joint-path mechanism.

On the seed-0 output-scale ray, the certificate is strictly smaller than `||∇m|| ||δ||` for every readout-orthogonal `δ` precisely when the branch rate is below `1/2`, i.e. for scales `α < 1.0387929173812708×10^{-4}`. At scale `1` the same certificate is about `1.63×10^6` and does not undercut the Cauchy number. Both facts are tests.

## What was not proved

No relation between the attention path and the MLP path, measurable under the same operations, produced a finite-response bound strictly tighter than the separate aligned triangle bound. The path-derivative sum is the branch part of `∇m`. Pointwise cancellation of those derivatives does not control the finite response: at step `2` the recomputed response is `-0.16901883579206622` while the linear term is numerically zero.

## Status line

`COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`. `OBSTRUCTION_READY = NO`. `PHASE_2 = NO GO`.
