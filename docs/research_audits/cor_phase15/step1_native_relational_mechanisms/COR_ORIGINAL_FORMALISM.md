# The original COR obstruction/discrepancy (O/D) formalism (Workstream G)

**Short answer: Investigation IX provides no new evidence for the original O/D operators, and none is claimed.**

## The checklist (mandate §10)

| required connection | found? | explanation |
|---|---|---|
| A restricted consistency-to-decision theorem | **no** | Theorems 1–3 relate a *Bayesian* evidence statistic (pooled eligibility S, per-group weights) to the decision. They are derived from the generative model by Bayes' rule. They make no use of obstruction or discrepancy operators, and no O/D quantity appears in them |
| An independently identifiable relational variable | partly, but not an O/D variable | S, e_g and m_g = S − e_g are identifiable from the data distribution and causally located in native models (`CAUSAL_MECHANISM_RECONSTRUCTION.md`). They are generator-defined Bayesian quantities |
| A candidate-specific causal prediction | yes, but not from O/D | NP2: patching other groups' q-tokens gives positive R1 credit. The prediction follows from Theorem 1 and the development mechanism |
| A nontrivial comparison against exact Bayes | yes, everywhere | all metrics are distances to exact Bayes |
| A result not implied by generic nonlinear evidence aggregation | **largely no** | the mathematics is Bayesian model averaging with an explicit sufficient statistic. The mechanistic results concern how Transformers implement it |

## Historical failure records, preserved and unchanged

- Unjustified projection and map choices.
- Outcome-dependent identification.
- Unfavourable candidate-response cases.
- Discrepancy/decision mismatches.
- The absence of a general causal bridge.

These remain open; see the Phase 1 fatality audit and Steps 1A–1E. Nothing in Investigation IX resolves them.

## Two new observations relevant to O/D, both negative or neutral

1. **The decision-relevant relation is not an answer discrepancy.**
   - In T4-with-colluder, the cross-group operation that decides R1 is a *count of all-agreeing groups* (S). That is a
     relation among groups' agreement *patterns across questions*, not a discrepancy between candidate answers.
   - In the adaptive-echo variant it is a cross-group echo count.
   - In GATE (earlier work) the gate is a relation among *non-answer* sources.

   An operator built only on answer discrepancies cannot express any of these three without additional structure. This
   restates, for the newer families, the earlier finding that answer-discrepancy O/D cannot express gating by non-answer
   sources.
2. **Interaction order is a candidate structural invariant, but it is not an O/D notion.** The evidence function's
   interaction order across groups (1 + C, Theorem 2) is implementation-independent and predicts which aggregation
   architectures can represent the rule. If a future COR formalism is to be "obstruction-like", one could ask whether an
   obstruction degree corresponds to interaction order. **No such correspondence is shown or proposed as established.**

## What would be needed

A definition of O/D on these generators, fixed *before* looking at outcomes, that:
- (i) reproduces the exact Bayes sign pattern on R1/R2/R3;
- (ii) says something Theorem 1 does not;
- (iii) predicts a native-model intervention outcome that the Bayesian account does not.

None exists at present.
