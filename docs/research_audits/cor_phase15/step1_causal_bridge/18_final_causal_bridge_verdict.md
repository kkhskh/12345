# 18 — Final adjudication

**Verdict:** `PHASE15_STEP1_ONLY_GENERIC_BOUNDS_SURVIVE`

No Transformer was run. No obstruction score was fit. No confirmation split was opened. Direction 1's manuscript was not edited. Phase 1's files were not edited.

`PHASE_2 = NO GO`

`OBSTRUCTION_READY = NO`

Preserved, unchanged: `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`; `08R2_VALIDATION_COMPLETE — NEGATIVE`; `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`; `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`; `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`; `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`; `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`; `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED`; `0A = C / PARTIAL`; `SECTION_5_3 = PARTIAL`.

This verdict does not say that every theory anyone might write later must fail. It says that every bridge examined in this step either fails or reduces to a consistency identity, a Lipschitz bound, a distance to a subspace, or an output gradient. None of those is a distinctive COR prediction. File 16 names no surviving candidate on purpose.

## What was proved

- An affine readout can take any value at a fixed cochain, so `O` and `D` do not determine `L`. Negating the final head reverses a recomputed response while an exact intertwiner keeps the residual at zero (file 02).
- If `P_v f = ρ P_u` as maps, the residual is zero on every recomputed trace of that `f` (Theorem Z). A pre-norm Transformer block is not such an `f`.
- A minimized shared-linear residual on natural pairs is zero exactly when the child stalk is a linear function of the parent stalk on those pairs (Theorem App).
- `|L(s)-L(Πs)| ≤ K √D(s)` if `L` is Lipschitz in the `M` metric. The bound has no sign (Theorem Lip, CE-15).
- If `L = q(Π s)`, the move to the nearest consistent state does not change `L` (Theorem Idle).
- `D(g+r)=D(g-r)` for `r` orthogonal to `ker d`, while a linear readout can flip (Theorem Sign).
- The first-order sign along Euclidean `-∇D` is `-∇L·∇D`. The `M`-repair direction has the same dependence on `∇L` (Theorem Descent, as corrected).
- For a genuine increment bound `S`, the sign is stable when `|L| > S`, and `τ O` is unnecessary slack. The printed pointwise-Jacobian form of Update Proposition 2 is false (`L(t)=1-2t^2`).
- On non-nested clamps, `y=(u+w)^2` has interaction `2` and aggregation residual `0`.

## What was disproved

- `D_c < D_{c'} ⇒ L_c > L_{c'}`, and `ΔD<0 ⇒ ΔL>0`, without an extra assumption.
- Exact computational agreement as a ranking of decisions, on any map that actually intertwines.
- Nonzero linear residual as, by itself, a candidate-specific conflict.
- Commutation as causal use, and failure of a proposed `ρ` as causal disuse.
- Factorization through `ker d` as a reason that lower obstruction helps.
- An analyst-assigned energy as evidence that the forward pass descends on it.
- Pairwise residuals, duplicated edges, and Euclidean coordinates as intrinsic to the computation.
- The printed stability proposition as a proved role for obstruction.
- Any of Families A–H as a signed predictor that is not the readout, the interaction contrast, or the assumption that writes the sign.

## What stays conditional

Theorem Lip needs a Lipschitz constant of `L`. Theorem Z needs an intertwiner, which a Transformer block does not supply. The corrected stability inequality needs an increment bound that the pointwise Jacobian does not provide. None of these conditions is a license to choose `ρ` or `τ` after seeing `Δm`.

## The thirteen points the assignment asked for

1. **Proved.** The list above.
2. **Disproved.** The list above.
3. **Conditional.** Lip, Z, and the corrected stability inequality.
4. **Unjustified assumptions.** That a Transformer implements descent on `D`. That `L` is monotone in `D`. That a constant linear `ρ` is the block. That the nearest global section is attainable. That `τ` and `β` are architectural.
5. **Families that failed.** A fails as use. C idles `D`. D assumes the conclusion. E is attribution. F's printed form is false and its corrected form does not use `O`. G is either Z, for a fixed linear sum, or approximation error. H is an interaction, undefined when targets nest, and not equal to `D` when it is defined.
6. **What survived.** Unsigned distance and Lipschitz bounds, the zero-residual identity on maps that intertwine, and output-gradient arithmetic.
7. **New or established.** Established. File 11, after the novelty review's cuts.
8. **Independent identification.** A residual can be frozen. The sign of `R_c` cannot be recovered from that frozen residual.
9. **Direction or magnitude.** Magnitude only, and only given a constant from `L`. The square discards sign.
10. **Stronger alternatives.** The output gradient and the Jacobian-vector product sign the same first-order questions and do not need `D`. On the audited numbers they already beat the obstruction score. Those numbers were not refit.
11. **Falsification.** There is no surviving signed hypothesis whose future failure would discipline COR. A declared Lipschitz bound can fail. That failure would not be a COR result.
12. **Steps 2–6.** This step does not authorize them. Any later step whose job is to build or identify an obstruction score for a decision is blocked with Phase 2.
13. **Phase 2.** Remains `NO GO`.

## Questions

**Q1.** Internal consistency affects a decision only if some further assumption ties `L` to the residual. No such assumption is derived from Definitions 2–4.

**Q2.** If `ρ` intertwines the actual map, the residual on a recomputed trace is zero. It is not always zero in a Transformer, because a constant linear `ρ` does not intertwine a block. That is a fact about the surrogate, not a supply of nonzero decision conflict.

**Q3.** After a shared linear fit, a leftover residual is the part of the child stalk the parent stalk does not determine, or the kink of a nonlinear map, or an omitted parent in a fixed linear graph. No second functional of `(P,ρ,s)` separates those from a candidate conflict.

**Q4.** A score can be computed without the response once `(P,ρ,M,W,K)` is frozen. The score is not thereby about the response.

**Q5.** Nothing in the definitions. `P=0` and an input-specific outer-product `ρ` are legal. A shared frozen `ρ` blocks those two and does not create a sign.

**Q6.** Yes. Negating the final head preserves a zero residual under an intertwiner and reverses the recomputed response.

**Q7.** Yes. `(1,0) → (0,0)` on the identity edge lowers `D` from `1/2` to `0` and lowers a readout of the first coordinate from `1` to `0`.

**Q8.** Either restrict `L` to a monotone function of `D`, or use `∇L`. The first is the desired conclusion. The second is attribution.

**Q9.** The monotone function cannot be checked without `L`. The gradient can be computed; computing it leaves the obstruction term with nothing to add at first order.

**Q10.** The monotone function is not a Transformer computation. The gradient is a property of any differentiable readout, including a Transformer. The intertwiner hypothesis is not a property of a pre-norm block.

**Q11.** A fixed linear sum of several parents falls under Theorem Z and then has residual zero. Attention weights are not fixed linear coefficients. A shared linear residual on a softmax edge is approximation error. Nonlinear heads produce interactions while that residual stays zero (`I=2`, `D=0` for `y=(u+w)^2`).

**Q12.** The squared score does not determine a sign. Theorem Sign.

**Q13.** The absolute bound is a Lipschitz bound on distance to a subspace. It is not stronger than ordinary sensitivity of `L`.

**Q14.** The sign along the obstruction step is an inner product with `∇L`. The gradient baseline is that inner product with `∇L` itself, and it does not need `D`.

**Q15.** A path effect is `w·Jδ` or the finite recomputed response. The residual is zero when `ρ=J` and the map is linear. An interaction is a second difference of intervened outputs. Neither is `||ds||^2`.

**Q16.** Causal-abstraction faithfulness is agreement of intervened mechanisms. Commutation is not that agreement, and a zero linear residual is neither necessary nor sufficient for it.

**Q17.** Yes. `(2,2)` has `D=0` and need not lie in a natural image whose only point is `(1,1)`. The Euclidean repair of `(1,0)` is `(0.5,0.5)`, which is the same kind of off-image point. `D=0` can also be the wrong decision (CE-04).

**Q18.** Sufficiency would be an empirical claim that interventions in the complement of the stalk do not move `L`, checked by those interventions. Factorization through `ker d`, if it held, would say that the component `D` measures does not move `L`.

**Q19.** Not by a small natural residual. A `ρ` fit to natural pairs makes that residual small because of the fit. The causal test uses held-out interventions. Using the evaluation interventions to choose the stalk is the circularity.

**Q20.** A Euclidean score changes under a rescaling that leaves the network and `L` fixed, unless `W` is transformed by the full congruence. There is no canonical metric. CE-07 reverses a ranking by changing `M` with `s` held fixed.

**Q21.** No. The architecture does not fix the stalk, the shared `ρ`, the metric, or the edge set of this score.

**Q22.** The examined bridges that conclude "lower obstruction improves the decision" assume a monotone relationship or an anti-alignment that already uses `∇L`. The bridges that do not assume it do not get the sign.

**Q23.** Evidence would be a derivation that the forward pass iterates descent on that energy. The block computes a map. No source supplies the iteration. Writing the energy down is not that evidence.

**Q24.** A multi-parent linear residual either matches the sum and vanishes on recomputed traces, or it is a different graph. An interaction contrast is a different object and is undefined until nested targets are given a truncation rule. Neither repair turns `D` into a cause.

**Q25.** No. Mathematical survivors are the generic bounds and the zero-residual identity. They fail the identifiability-to-a-sign test.

**Q26.** No.

**Q27.** The audited obstruction-only association with margin change, Spearman about `0.095` against about `0.905` for attribution, is the empirical form of Theorem Descent: the sign lives in `∇L`. The exact pair `0.936` versus `0.141` is not in the sources and is not used. Othello's positive probe-invisible letters are about a different question and are not evidence for this bridge.

**Q28.** There is no surviving signed bridge to falsify. A proposed future bridge would be falsified by a pre-registered intervention set on which its frozen score ranks responses no better than `∇L`, with the score and the baseline denied access to the evaluation responses. This step does not authorize that experiment.

**Q29.** The residual's inputs can be frozen. The sign cannot be frozen out of them, because it is not in them.

**Q30.** No. The formulation that remains after forbidding a favorable outcome from shaping it is a residual, a Lipschitz inequality, and an output gradient. That is not a hypothesis about obstruction as a cause of decisions.
