# Open COR foundational questions after Investigation XII

The program remains in **Phase 1.5, Step 1**, at a provisional ~25% of the foundational territory. This is a judgment,
not a measurement. Nothing here closes Step 1, selects a thesis, or validates the original COR framework.

## A. The T4 relational computation: what is now settled, and what is not

**Settled, at the level stated in the other documents:**
- The conjunction is formed in layer 1. This is fresh-cohort replication P1; see the confirmation document for its
  verdict.
- Its removal moves decisions to the "group ineligible" counterfactual under natural-activation interventions as
  well as hybrid ones.
- It is mediated by a low-dimensional subspace whose *value* matters (scrubbing).
- It is created by softmax-normalised attention over additive score shifts, plus MLP1 amplification. It is not a
  thresholded AND.

**Open:**
1. **Layer 2.** How is (S − e_g) reweighting of the other groups' evidence implemented? Is the readout "I1-like"
   (pooled S) or "I2-like" (per-group m)? The VS linear-map result (non-projective, partial) is suggestive. It is
   not decisive.
2. **Sub-head redundancy.** Why can either attention routing or value content carry perfectness? Is there one
   upstream feature read twice, or two features? A feature-level (SAE or probe-with-intervention) analysis would be
   needed.
3. **Residual information outside E.** The 442-dim complement restores 6–64% of the decision effect. What is it, and
   is it the cross-group-equality information the C = 2 repair used?
4. **The A2 → B step in learning.** The conjunction is used from emergence. What is calibrated later, between the
   threshold-4 behaviour (A2) and Bayes? A candidate is the suppression of n = 4 contributions in the other groups'
   evidence weights, outside the factorial's h group. Untested.

## B. Causal-abstraction mathematics

5. **A non-product-support version of Proposition A2**, with a proof. When e is a function of (W, L), interchange
   constrains natural states, but the exact characterization is not written.
6. **Admissibility.** Classes I–V (math §4) are a taxonomy, not a theory. A principled account of which
   off-image interventions are *informative* about a network would require a model of the network's off-manifold
   extrapolation. That is an open empirical-theoretical question.
7. **Uniqueness of mediators.** No positive identifiability theorem exists for these diagnostics. Counterexamples
   (redundant encodings, bijective re-encodings) show none can be unconditional.

## C. Generalization

8. **Shared versus independent copying.** The exact two-colluder posterior needs a *between-group* relation (J_H)
   that the per-group eligibility statistic cannot express. The adapted readouts reached exact Bayes without E.
   Which representation do they actually read? A causal test of cross-group-equality features is needed.
9. **Off-support behaviour (S = 0 under partial copying)**: still model-specific and unexplained (XI).
10. **Other aggregation degrees.** G ≥ 4 or C ≥ 3 need a new token layout. Retraining cost: one new tokenization plus
    a full pretraining cycle, about 1 h per parent on this hardware. Not attempted.

## D. Across COR families (T4, GATE, TRANS, adaptive sources, unknown groups)

11. **GATE.** Earlier investigations (`step1_learned_mediator_identification`) found that GATE "computes its relation
    by attention", with relation variables at source tokens and readout attention. The T4 conjunction found here also
    reads keys/values at source tokens, and its queries are irrelevant.
    - Defensible statement: both are attention-read relations at source tokens.
    - Not established: a common mathematical abstraction. GATE is a per-question order-2 selector; T4 is a
      cross-question aggregate (perfectness) conjoined with a per-question relation.
    - They should not yet be placed in one class.
12. **TRANS, adaptive sources, unknown groups, broader architectures.** No XII experiment touched them. They remain
    on the map.
13. **The original obstruction/discrepancy (O/D) framework.** **No connection was found or tested.** None is claimed.

## E. Evidence that would address the most valuable questions

| question | decisive evidence |
|---|---|
| layer-2 algorithm (I1 vs I2) | path patching inside layer 2 with natural donors that differ only in S vs only in m |
| sub-head redundancy | feature-level dictionary of layer-1 keys/values, with interventions on single features |
| cross-group relation under C = 2 | a scrubbing test of the adapted readout with donors matched on e but differing in cross-group equality |
| population-level claims | ≥ 10 fresh parents (≈ 10 h of pretraining at current speed) |
| generality beyond T4 | the same factorial / natural-path / scrubbing protocol on GATE and TRANS models |

## F. Added after the independent final review

14. **The subspace-patching illusion.** Do the scrubbing and restoration results use a natural mediator or a
    dormant steering direction? A test is needed that does not pass through the DAS-fitted subspace, for example
    feature-level or input-level mediation.
15. **Residual eligibility without layer 1.** Mean ablation of layer 1 leaves pref_e0 at 0.06–0.18. Is there an
    eligibility route outside layer 1?
16. **Heterogeneous allocation.** P156 forms the conjunction in MLP1, not in attention. What determines the
    allocation, and is it related to pretraining completeness?
17. **The calibration lag** (A1 → A2 → B, 1–5 checkpoints after sensitivity onset) is unexplained.
