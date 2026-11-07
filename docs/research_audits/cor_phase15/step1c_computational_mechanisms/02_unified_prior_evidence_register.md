# 02 — Unified prior evidence register

This register is the foundation for Step 1C. It does not reopen the earlier verdicts. Each row says what was established, under what assumptions, and what Step 1C is allowed to use it for.

Status labels: proved, numerically checked, disproved, conditional, conjectural, incomplete, illustrative, not supported by the Transformer equations.

## Phase 1

Source directory: `docs/research_audits/cor_phase1/`. Letter: `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`. `PHASE_2 = NO GO`. `OBSTRUCTION_READY = NO`.

| statement | where | assumptions | status | use in Step 1C |
|---|---|---|---|---|
| `O=(ds)^\top W(ds)` and `D=\min_{g\in\ker d}\lVert s-g\rVert_M^2=(ds)^\top(d M^{-1}d^\top)^+(ds)` when `M\succ 0` | Phase 1 verdict, §1 | `ρ` linear so that `ds` is defined; `M` positive definite for the closed form | proved as algebra | `O` and `D` are different functions. Error 2 is the observation that `M` occurs only in `D` |
| `λ_min^+ D \le O \le λ_max D` | MVP Theorem 1, cited in Phase 1 | `W` and the metric operator on the same `ds` | proved spectral sandwich | not a causal bridge |
| `[ds]=0` in `H^1` for a global 0-cochain | Phase 1 | the cochain is a function of one underlying state | proved for that construction | exact agreement can be manufactured by taking both ends from one cochain |
| `D_c\downarrow` does not imply `L_c\uparrow`, and `D_c<D_{c'}` does not imply `L_c>L_{c'}` | Phase 1 counterexamples | unrestricted linear heads and choices of `W`, `M`, `ρ` | disproved as a universal implication | still disproved. Not reopened |
| Update Proposition 2's pointwise Jacobian is not a Lipschitz bound | Phase 1, `L(t)=1-2t^2` | a local derivative used as if it bounded a finite step | disproved | finite interaction is not the mixed partial at the baseline. Error 1 is the same quantifier mistake in a new formula |
| Quadratic repair of an objective that already contains the margin | Phase 1 | the objective includes `-\beta m` | proved that the minimizer uses the margin | circular if offered as a prediction of the margin |
| Historical obstruction comparisons did not establish the mechanism | Phase 1 record; Direction 1 letters | the experiments that were actually run | empirical, unchanged | cited in §21 of this step as constraints, not as a new fit |

## Step 1A

Source: `docs/research_audits/cor_phase15/step1_causal_bridge/`. Verdict: `PHASE15_STEP1_ONLY_GENERIC_BOUNDS_SURVIVE`.

| statement | assumptions | status | use here |
|---|---|---|---|
| Eight families around `O`, `D`, and linear `ρ` do not yield a distinctive signed bridge | the eight families actually written down | conditional negative, scoped to those families | the families are not re-surveyed. The question that stays is whether some other computational relation constrains a response |
| Exact intertwining `P_v\circ f=\rho\circ P_u` implies residual zero on recomputed traces of that fixed `f` | `f` is the map of the intervened argument, not a pre-norm block | proved for that `f` | restated as the recomputed-trace theorem. A pre-norm block is not such an `f` unless the whole block is the chosen map |
| `ρ \approx P_v J P_u^+` transports derivatives, not absolute states | first-order expansion | proved as a Taylor statement | the affine formula in Investigation C keeps the offset `f(h_0)` |
| Descent of `D` has the sign of `-\nabla L·\nabla D` only in the Euclidean metric; the `M`-repair direction differs when `M` is not a multiple of `I` | smooth `L`, nondegenerate metric | proved as a pairing | a metric choice is not a mechanism |
| Factorization `L=q(\Pi s)` makes the inconsistent component idle | `L` ignores the component being repaired | proved | sufficiency is fiber constancy, not a small residual |
| One-point or arbitrary `ρ` can force agreement | `ρ` may depend on the pair | proved by construction | repeated here with the outer product `z_v z_u^\top/(z_u^\top z_u)` |
| Generic Lipschitz and gradient bounds survive | the function satisfies the bound's hypothesis | generic existing results | Investigation D asks whether a relation tightens them. The answer in the examined cases is no, beyond the Jacobian of a restricted family |

## Step 1B

Source: `docs/research_audits/cor_phase15/step1b_mechanism_discovery/`. Verdict: `STEP1B_NECESSARY_CONDITIONS_IDENTIFIED`.

| statement | status after Step 1C | reason |
|---|---|---|
| N1, with the corrected quantifiers: no function of `(Q(h),Q(h+\Delta))` recovers both signs on a domain that contains opposite kernel displacements | proved, kept | rechecked. The domain requirement is essential. One sign on one displacement is not the theorem |
| Nonlinear orientation reversal as a general extension of N1 | conjecture, still unproved | Step 1B already marked it. No proof was added here |
| Joint linear factors can be sufficient when each factor is not. The sufficient object is the tuple | proved existence, kept | a scalar relation of the tuple is a further lossy map |
| Rowspace intervention-sufficiency of a linear `P` | tautology, not a theorem | `ker P` meets the rowspace only at `0` |
| Two systems can share structural residual `0` and have `I=0` and `I=2` | proved for those two maps, kept | this is Error 1's separable square against a nonseparable square. It does not say that every nonlinearity has nonzero `I` |
| A kernel vector of `P_u` leaves the residual invariant | withdrawn | Error 4. The kernel that matters is `ker(P_v A-\rho P_u)` |
| Shared nonlinearity predicts nonzero `I` | withdrawn as a universal sentence | Error 1 |
| Changing `M` changes Euclidean `O` | withdrawn | Error 2 |
| Sign of a fixed linear readout survives LayerNorm | withdrawn | Error 3 |
| H-SUF, H-COORD, H-COMPOSE, H-STAB, H-ABS are existing abstraction, attribution, interaction, or Lipschitz arguments | classification kept | Investigations A–D rederive the Transformer cases and reach the same classification, with the quantifiers fixed |
| H-CORR majority vote | toy, empirically untested in a Transformer | not developed further. It is not a block equation |
| H-OD, the original signed obstruction bridge | disproved as a universal implication | unchanged |
| "Architecture predicts `I\neq 0` for addends that meet in `σ`" | illustrative and, as a universal claim, false | the rectangle theorem replaces it |

## What was only an example

Majority-of-bits, a two-node truncation calculator, attention with logits supplied independently of `Q` and `K`, and a single-vector residual stand-in are calculators in the Step 1B harness. They check arithmetic. They are not derivations of a Transformer block. Step 1C replaces the attention calculator with `Q=XW_Q`, `K=XW_K`, `V=XW_V` and a causal mask, and replaces the LayerNorm sentence with the closed form and the sign counterexample.

## What remains unsupported by a trained model

No row in this register is a measurement on Othello-GPT or on any other frozen network. Direction 1's letters are empirical facts about a different question, probe sufficiency, and are used only as constraints in `09` and `23`. They were not refit.
