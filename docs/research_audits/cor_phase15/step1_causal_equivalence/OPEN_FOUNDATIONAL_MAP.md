# Open foundational map (after Investigation X)

## Where Investigation X leaves the main question

The repaired native models share a causally interchangeable, distributed eligibility abstraction. They also share a
behavioural algorithm (the Bayes eligibility rule) and an algorithmic learning order. They do not share token subsets,
and their head-level organization is inherited from the pretrained parent. The final status per parent is in
`FROZEN_PREDICTIONS_AND_RESULTS.md` Part II.

## Highest-value unresolved questions, ranked

1. **How is E computed inside the fine-tuned layers 0–1?**
   - *What we know:* total-effect profiles locate a candidate (the layer-0 MLP and a parent-specific layer-0 head)
     but do not isolate the computation of "agreement at q × perfectness".
   - *Next step:* path patching restricted to head → q7-token edges, plus per-head DAS of the agreement-at-q bit.
   - *What it decides:* whether the implementations differ only in *which* heads do the same operation, or in the
     operation itself.
2. **Is there also an S variable?**
   - "Set S" interchange is partial (0.55–0.58) where "set E" is near-exact.
   - Theorem X1 says this is the per-group signature, but the identification caveat (gradient search missed an exact
     S-alignment in a constructed model) leaves it open.
   - *Next step:* train DAS for S with initialization from the E-subspace (S = Σ e is linear in E).
3. **Cross-model mapped interchange.**
   - Fit a linear map between two models' q7 states. Interchange model A's mapped source state into model B's E-subspace.
   - High mapped IIA would turn "same abstraction" into "same encoding up to a linear map".
   - Not run.
4. **Why A1 first?** All runs learn perfectness-based eligibility before q-agreement integration.
   - *Hypothesis:* perfectness is already linearly present in pretrained states (Investigation IX: counts decode at
     1.0), while agreement-at-q × perfectness must be newly computed.
   - *Test:* vary pretraining so that agreement-at-q is pre-encoded, then predict an earlier switch.
5. **The C1 false-positive mode of DAS.** Moderate interchange accuracy (0.85–0.89) appeared in models whose subspace
   is non-selective. A general criterion is needed: selectivity under defining edits, used as a validity check for
   interchange abstractions.
6. **The B3 follow-ups.** Status is in `FROZEN_PREDICTIONS_AND_RESULTS.md` Part II:
   - frozen-state readers with Bayes targets at 32k steps;
   - longer pretraining of incomplete parents.
7. **Generalization.** No new generalization runs were made in Investigation X. The native models are tied to G = 3.
   Investigation IX's identity-free readers generalized to held-out pairs but not to more groups.
   - *Discriminating test for the E abstraction:* a model implementing per-group E should transfer to C = 2 colluders
     differently from one implementing pooled S (Theorem 1-C). This needs a variable-colluder tokenisation.

## Connections to earlier COR work (supported vs unsupported)

| connection | status |
|---|---|
| T4 reliability routing (order 1) vs eligibility (order 2) | **supported** mathematically (Investigation IX, Theorem 2). Behaviourally, the original model sits on order-1-plus-perfectness rules (A1/A2) |
| GATE compatibility gating | **unsupported as a shared mechanism**. GATE's gate is a within-question relation; its routing was causally unnecessary. No interchange abstraction was tested for GATE |
| Investigation IV causal equivalence | refined: behavioural and abstraction-level equivalence can hold without implementation equivalence. This is now shown on *independently pretrained* models, with selectivity controls |
| TRANS, adaptive sources, unknown groups | untouched; open |
| original O/D (obstruction/discrepancy) | **no connection found**. E is a Bayesian sufficient-statistic component, not an O/D quantity |
