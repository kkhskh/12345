# COR Phase 1.5, Step 1E — jointly reachable pathway increments

This package studies the set of attention and MLP increments produced by one intervention in one frozen pre-norm block. It does not replace the overnight reconstruction or Step 1D, and it does not open Phase 1.5 Step 2 or Phase 2.

**Verdict:** `STEP1E_VERIFIED_RELATIONAL_CERTIFICATE_KNOWN_THEORY`

On one invariant line of a specified two-head block, the jointly reachable branch readouts are strictly anti-monotone. Their support maximum is strictly smaller than the sum of the marginal support maxima. For one gain the joint positive branch maximum is exactly 0, while the attention marginal is `0.7070997102248`. The comparison is the product of exact marginals, not a norm-ball relaxation. The argument is the support function of a monotone graph, together with the chain rule. It is not a new decision theory, and it does not revive an obstruction score.

Preserved status:

- `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- `PHASE_1.5_STEP_1 = ACTIVE`

## How to run

From this directory, with no `PYTHONPATH` set:

```text
python3 -m pytest
python3 scripts/verify_manifest.py
```

Dependencies, as installed for this run: `numpy==1.26.4`, `pytest==9.2.0`, `sympy==1.14.0`. PyTorch is not installed. The second route is the loop implementation in `proofs/reference_block.py` and the SymPy identities in `proofs/sympy_identities.py`.

The notes `00` through `23` are the scientific record. `proofs/` is the mathematics that those notes cite. Tests check the implementations; they are not the proofs.
