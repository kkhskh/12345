# Causal identification (Problem E)

## 1. Levels of causal claim, and what each requires

| level | claim | evidence that identifies it here | status |
|---|---|---|---|
| L0 input-level | the model's decision depends on the cross-group relation in the right direction | R1/R2/R3 edit responses: same-group count edits whose Bayes effect differs only through other groups' eligibility | identified for every model (behavioural) |
| L1 necessity of a pathway | removing a computational pathway removes the relational effect and spares unrelated effects | attention ablations in the reader: noperf removes R1 credit, F34 unchanged | identified for the attention readers (dev; P3 on new seeds) |
| L2 sufficiency of a pathway | the pathway's state alone carries the effect | m-subspace interchange (share 1.84 / 0.51 vs random) | partial: share ≠ 1; other directions also contribute or cancel |
| L3 specificity | matched controls do not produce the effect | random subspace; self-only attention (regime-independent); generic adapter (gadapt ≈ orig) | identified |
| L4 cross-model causal equivalence | two models implement "the same" relational computation | not identified; see §3 | open |

## 2. Necessity and sufficiency of the relation itself (model-free)

These statements are about the *function*, not about any network:

- **Necessity [P, Theorem 1 of Inv. VII + Prop. A/C here].** No group-separable rule gives R1 a sign opposite to R2/R3.
  The same holds for any rule that is log-additive over cross-group terms, or additive over groups in its logits. Any
  model that gets the R1/R2/R3 pattern right must therefore contain a non-separable cross-group interaction.
- **Sufficiency [P, constructive].** Counts + q-reports are sufficient for the exact posterior. Within the mixture class,
  the gating sum of Proposition B, multiplied into RHO^{n_g}, suffices.
- These two statements locate *what* must be computed. They do not identify *where* a given network computes it. L1–L3
  are needed for that.

## 3. Cross-model causal-equivalence transport

**Question.** Is the relational component found in one successful model the same component, causally, as in another? For
example, does the m-subspace of attn seed 0 play the same role as that of attn2, or as the adapter in a repaired P-count
model?

**Operational test considered.** Train a linear map between two models' group states on natural decisions. Then
interchange: replace model B's m-subspace with the mapped value from model A on the counterfactual input, and check
whether B's R1 response follows.

**Status: not run.** Two reasons:

1. Compute was prioritised for the frozen confirmation.
2. The within-model interchange share already departs from 1 (1.84, 0.51). A cross-model transport score would be hard to
   interpret before the within-model account is complete.

What *is* shown is weaker. Different successful architectures share the necessary ingredient: their behaviour satisfies the
same non-separability constraint. In the attention readers, the ingredient sits in cross-group attention to eligible
partners. "Same function at the behavioural level, mechanism identified in one architecture family" is the accurate
summary.

## 4. Threats and how they were handled

- **Interventions defined with generator-side eligibility.** The noperf and noother masks use the true eligibility, which
  the model never sees. A model that computed a *different* but correlated relation would still be disrupted by them. The
  self-only result (exact regime independence) does not depend on this choice.
- **Off-distribution states.** Masking attention creates internal states not seen in training. The F34 invariance under
  noperf shows the network does not fail globally.
- **Ties.** Argmax is never used for R2/R3 (math doc §5).
- **Model-level inference.** Seeds are the unit; nothing is pooled over decisions across models for a significance claim.
