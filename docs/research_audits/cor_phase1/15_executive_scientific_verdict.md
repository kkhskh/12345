# 15 — Executive scientific verdict

Phase 1 theoretical audit only. Previous experimental letters are unchanged. `OBSTRUCTION_READY = NO`.

Top-level label:

**`COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`**

The algebra of a graph residual is well-defined. The original scientific theory is not identifiable, not causally derived, and not distinctive. Phase 2: **NO GO**.

---

## 1. What is mathematically correct?

- The coboundary `(ds)(u→v)=ρ s_u - s_v` is type-correct when `ρ: F(u)→F(v)`.
- If `W≻0`, `O=‖ds‖_W^2=0 ⇔ ds=0`.
- If `M≻0`, `D=min_{g∈ker d}‖s-g‖_M^2` exists, is unique, and equals `(ds)^T (d M^{-1} d^T)^+ (ds)`.
- `λ_min^+ D ≤ O ≤ λ_max D` (MVP Theorem 1).
- `W=(d M^{-1} d^T)^+` implies `O=D`.
- `[ds]=0` in `H^1` for a globally defined 0-cochain.
- The quadratic-program minimizer of `O - β m + …` is `-A^{-1}(∇O - β ∇m)` when `A≻0`.
- Ordinary Taylor expansion of `O` and `m` in a `C^2` ball.

## 2. What is mathematically wrong?

- “`O` is equivalent to `D`” as a general statement (sandwich, not equality).
- “`O` is a cohomological / gluing obstruction” for computed full 0-cochains.
- “`D_c < D_{c'} ⇒ L_c > L_{c'}`” (Attack A).
- “Reducing `D` increases target logit / probability / accuracy” (Attack B).
- Euclidean `O` is representation-invariant (Attack D).
- Candidate ranking by `O` or `D` is intrinsic to the model (Attack C).
- Nonemptiness of `ker d` is a meaningful global decision representation (Attack H).
- Learned `ρ` is the Transformer’s computation.

## 3. What is mathematically underspecified?

- Whether `P` is a projector, its inner product, and its rank.
- Composition of `ρ`.
- Whether `ρ` depends on `x`.
- The principle that fixes `M`, `W`, `K`, and edge duplication.
- Any map from `D` to `F_θ` other than putting `-β m` into `S` by hand.
- Sufficiency, minimality, uniqueness of “decision germs.”
- Proofs of Update Propositions 2 and 3.

## 4. What is not identifiable?

- `P`, `ρ` under the written definitions (Attacks E, F).
- `M`, `W`, graph granularity (Attacks C, G).
- `α,β,γ` in `S` if tuned on evaluation `Δm`.
- A canonical probe-complement “germ” from the 08c SVD split.
- Unique semantic features under superposition (Update Prop. 7; not repaired).

## 5. What has no established causal connection?

- `O` and `D` as such.
- Distance to `ker d` and candidate selection.
- Obstruction reduction and behavioral repair.
- Sheaf language and Transformer decisions.
- The response functional, except as a mixture that can recover attribution.

## 6. What does our existing empirical work genuinely support?

Probe-invisible causal information and donor-free steering in Othello-GPT (`TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`, `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`, `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`). A Pythia-1B low-rank residual energy that **monitors** factual conflict (2B) and whose local Taylor map predicts `ΔO` (2D). IOI as a documented negative control.

These are not COR obstruction.

## 7. What evidence contradicts our proposed mechanisms?

- IOI: attribution-positive / obstruction-negative. Prompt-cited Spearman 0.936 vs 0.141 was not found as those exact floats; documented 2D analogue is attribution repair AUC 0.979 vs obstruction-using 0.955, obstruction-only Spearman with `Δm` ≈ 0.095 vs attribution ≈ 0.905.
- 08R2 validation NEGATIVE.
- `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED` (both-positive 13/32).
- `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED` (`I_T=+0.000808`, `I_W=+0.000972`, both CIs include 0 / fail +0.005).
- 2D: combining obstruction with attribution does not improve steering.

## 8. What survives of COR?

A frozen linear graph-consistency residual. Not a decision theory. Not a topological obstruction. Not a compatibility map in the Othello sense. Not an obstruction-ready empirical program.

## 9. Is there a distinctive prospective prediction?

**NO.**

## 10. Should we proceed to Phase 2?

**NO GO.**

Gates B, C, D, E, G, H fail. Algebraic well-definedness (A) is not scientific readiness. Repair would have to freeze identification rules, drop sheaf overclaim, drop the unproved `D→L` implication, and then ask whether anything remains that is not residual energy or interchange. That is a new theory, not a license to run another Othello or Pythia obstruction experiment.

---

## Q1–Q20

**Q1.** Why should distance to a linear consistency subspace have anything to do with Transformer decisions?  
It should not, unless an extra assumption couples `L_c` to `ker d`. No such assumption is proved. Attacks A and B.

**Q2.** What prevents compatibility maps from manufacturing obstruction values?  
Nothing in the definition. Train-only shared ridge constrains them statistically, not causally. Attack E.

**Q3.** What makes candidate-specific projections about decisions?  
Only the analyst’s label `c` and the fitting target. Def. 2 allows `P=0`. Attack F.

**Q4.** Is global obstruction different from a weighted local residual?  
No. `D` is `(ds)` in the metric `(d M^{-1} d^T)^+`. File 03.

**Q5.** Nontrivial local-to-global / cohomological obstruction?  
No. Graph consistency errors. `[ds]=0` in `H^1`. File 05.

**Q6.** Meaningful under hidden-state coordinate changes?  
Not the Euclidean version used in code. Yes if `(M,W,ρ)` transform. Attack D.

**Q7.** Can functionally equivalent representations have different obstruction?  
Yes. Attack J and gauge changes.

**Q8.** Can low obstruction coexist with an incorrect decision?  
Yes. Attack A, H.

**Q9.** Can obstruction decrease while target probability decreases?  
Yes. Attack B.

**Q10.** What is a decision germ vs a steering vector?  
As defined, a linear image of an activation. No sufficiency or uniqueness. A steering vector is a special case. File 07.

**Q11.** Can COR identify causal objects before the intervention response?  
Only if `P,ρ,M,W,K` and the score are frozen without that response. `S` with tuned `β` cannot.

**Q12.** What assumption connects local compatibility to downstream causal sensitivity?  
None in the theorems. Proposed B1/B4/B5 are extra (file 06).

**Q13.** Is that assumption independently testable or tautological?  
B1 is testable as interchange. B6 (`S` includes `Δm`) is tautological if `β` is fit on repair. B4 is testable (`corr(∇O,∇m)`) and empirically weak for obstruction-only.

**Q14.** What did donor-free Othello transfer establish for COR?  
A compact map in the probe-nullspace can move Y without a donor. It did not establish germs, `ρ`, `O`, `D`, or obstruction. NST preregistration said so.

**Q15.** What did the failed visible/null factorial say?  
Under frozen `ρ=ν=3.5` and train-only `r̂`, independently setting visible and invisible target identity did not produce gated `I_T`,`I_W`. Compatibility-as-matched-subspaces failed. It does not reverse NST/SPEC.

**Q16.** Why did attribution outperform the tested obstruction score?  
**Known:** on IOI and 2D, scores using `∇m` predict `Δm`; scores using `∇O` predict `ΔO`; mixing them did not win repair ranking. **Hypothesis, not fact:** `O` is a fitted reconstruction residual, while `m` is the actual decision coordinate, so first-order prediction of `Δm` is attribution. Do not promote that hypothesis to a theorem.

**Q17.** Is there a mathematically justified obstruction score that can be frozen without evaluation outcomes?  
Yes as a residual: freeze `(K,P,ρ,M,W)` and compute `O` or `D`. That score is not justified as a decision mechanism.

**Q18.** Can that score predict something different from geometry, sensitivity, or attribution?  
Not on present assumptions. File 12: distinctive prediction NO.

**Q19.** What would falsify the minimum surviving hypothesis?  
The minimum surviving object is “a frozen residual can be computed.” That is not a scientific hypothesis. The monitor hypothesis “this residual tracks conflict beyond margin” was already falsified in 2A/IOI scalar form and supported in one 2B vector form. Further negatives on new ranks/metrics would not be a COR test unless frozen in advance without escape clauses.

**Q20.** If we were forbidden to modify the theory after another negative experiment, would the present formulation be strong enough to submit?  
**NO.** Unconstrained `P,ρ,M,W,K,β`, unproved `D→L`, Euclidean gauge dependence, and sheaf overclaim leave too many post-hoc moves. A frozen test of original COR would not be interpretable, and a frozen test of the residual toolkit would not be a test of COR’s mechanism.

---

Do not treat this audit as permission to retrain maps, retune `ρ,ν`, or run 08R3. The scientific record already contains the relevant negatives.
