# 0. Source register

## Present and used

The branch already contained, and this step did not edit:

- `docs/research_audits/cor_phase15/overnight_reconstruction/`, commit `2655a7b`
- `docs/research_audits/cor_phase15/step1d_structural_bridge/`, commit `90342e4`
- the historical Phase 1 and earlier Phase 1.5 audit directories
- the Direction 1 paper

Step 1D's theorem and counterexample notes were read for scope, not rewritten. The results used as premises are Theorem U, Theorem L, Theorem S, Theorem Align, Theorem Kernel-out, the product-ball ceiling, and the statement that a base-point derivative cancellation does not control a finite response. Their numerical witnesses were not retuned.

## Absent

These files are not in the repository. They were not read.

- `INDEPENDENT_STEP1D_SCOPE_AUDIT.md`
- `independent_stress.py`

The prompt's summary of that audit is the starting point: Step 1D's R2 result is an independent-radius product-ball statement, not a description of the jointly reachable set.

Local untracked archives on another machine, including `cor_phase15_overnight_reconstruction.zip` and `cor_phase15_step1d_structural_bridge.zip`, are not part of this record and were not used.

## Runtime

`import torch` raises `ModuleNotFoundError`. No timings were invented. The numerical routes are NumPy `1.26.4` and a separate loop implementation, plus SymPy `1.14.0`. Pytest is `9.2.0`.

Proofs are exact statements about real functions. The tests are IEEE-754 float64 checks of the implementations, with the tolerances written in `tests/test_step1e.py`. They are not a rounding-error enclosure of the float program.
