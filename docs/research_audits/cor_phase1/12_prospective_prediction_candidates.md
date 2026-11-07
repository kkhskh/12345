# 12 — Prospective prediction candidates

After Parts 1–11: **none of the original COR causal claims is ready to freeze as a distinctive obstruction-specific test.**

Below are at most three *conditional* candidates, ranked by mathematical defensibility. None is recommended for Phase 2 execution. `OBSTRUCTION_READY` remains `NO`.

## Candidate 1 (most defensible, not distinctive)

**Statement.** For a fully frozen `(K,P,ρ,M,W)` fit on training-clean data only, `O(x)` can be computed before any evaluation intervention.

**Why it is not a COR prediction.** It is the definition of a residual. 2B already instantiated it. Distinctiveness vs activation geometry / PCA reconstruction error is not established.

Falsification: none as a causal claim. As a monitor, 2A already failed; 2B passed for one PCA residual on Pythia-1B factual conflict.

**Verdict:** not obstruction-specific.

## Candidate 2 (would be distinctive if an extra assumption were true)

**Statement.** If (B1) `L_c` factors through `ker d` for frozen maps, then interventions with `d(s+δ)=0` cannot change `L_c`, and interventions in `ker(∇L_c)` that change `O` cannot exist.

**Required extra assumption:** B1, not in the original theorems, not supported by Othello factorial or SPEC (probe-invisible effects live in `ker W_probe`, which is a different kernel).

**Simple counterexample outside B1:** Attack A.

**Falsifier:** any frozen-map intervention in `ker d` that changes `L_c`, or a change in `O` with `ΔL=0` while claiming B1.

**Leakage test:** maps must be frozen without `ΔL`. If they are, this is interchangeable with causal-abstraction interchange, not a new obstruction score.

**Verdict:** CONDITIONAL on B1. Running it would test interchange, not `O` vs `D`. Do not relabel interchange as obstruction.

## Candidate 3 (the original closed loop)

**Statement.** Rank sites by `S=-ΔO+βΔm-γCost` and beat attribution at repair.

**Already tested.** IOI negative; 2D repair-AUC lift entirely negative. Including `βΔm` fails the leakage test unless `β=0`, in which case 2D obstruction-only is weak for `Δm`.

**Escape clauses to forbid:** wrong layer/basis/scale/β. Those were available and were used historically.

**Verdict:** empirically failed in the form that was actually computed. Not a new prospective test.

## Strong-baseline comparison

Under current assumptions, COR’s operational score is a graph residual plus optional attribution. There is no remaining prediction that ordinary attribution, `∇m`, activation distance, or the residual itself would not already make.

## Conclusion

**Distinctive prospective obstruction-specific prediction: NO.**

Do not design Phase 2 to search for a positive `O`–repair correlation by changing rank, `β`, or metric.
