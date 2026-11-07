# COR Phase 1.5, Step 1 — causal bridge

**Verdict:** `PHASE15_STEP1_ONLY_GENERIC_BOUNDS_SURVIVE`

Phase 2 remains **NO GO**. `OBSTRUCTION_READY` remains **NO**. No Transformer was run. The synthetic checks are finite matrices.

Start with `18_final_causal_bridge_verdict.md`. It answers the thirty questions. The other files are the proofs, the counterexamples, the five reviews, and the gate matrix.

## What was proved

A consistency score built from projections, transport maps, and metrics does not determine a decision. If a transport map exactly matches a fixed computation, the residual on a recomputed trace is zero. If it does not, the residual is approximation error. A squared distance has no sign. The sign of a repair step is an inner product with the output gradient. The printed stability proposition does not give obstruction a role: a pointwise Jacobian does not bound a finite step, and a real increment bound does not need the obstruction term.

What remains is a Lipschitz bound, a zero-residual identity for maps that actually intertwine, and ordinary gradient arithmetic. None of these is a distinctive COR prediction. `16_surviving_prediction_candidates.md` names no candidate.

## How to read the folder

| File | Contents |
|---|---|
| `01_authoritative_source_register.md` | Original definitions and conflicts |
| `02`–`04` | Impossibility, exact computation, intervention semantics |
| `05`–`06` | Families A–H and which of them are the same object |
| `07`–`09` | Identifiability, coordinates, sign |
| `10_counterexample_catalog.md` | CE-01 through CE-20 |
| `11`–`12` | Literature and attribution baselines |
| `13_theorem_and_proof_obligations.csv` | Claim register |
| `14`–`15` | Review resolution and where a cause would have to enter |
| `16`–`18` | No surviving candidate, gates, verdict |
| `reviews/` | Five separate review passes |

The checks live at `tests/test_causal_bridge_math.py` in the repository and are included in the zip. From the repository root:

```text
python -m pytest tests/test_causal_bridge_math.py -q
```

That command does not load a model. Do not run a Transformer, fit an obstruction score, or open the confirmation split on the strength of this package.

Official letters are unchanged, including `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`.
