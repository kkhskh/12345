# Investigation XI — Main research synthesis

**The causal construction of relational computation: mathematical validity, native mechanisms, cross-model transfer,
and generalization.**

- Branch: `cor-phase15-step1-native-causal-construction`.
- Directory: `docs/research_audits/cor_phase15/step1_native_causal_construction/`.
- Status: Step 1 is **not closed**; the program is at roughly 25%. No thesis is selected.

## Summary

1. **Theorem X1 is corrected and partially retracted.**
   - The I2-VS proof sketch was invalid: it assumed equal m with different S, but S = Σm/2.
   - Corrected statement: VS is realizable in I2 by a linear map (½11ᵀ) but by **no** projection, orthogonal or
     oblique. This is proved.
   - I1-VE and I3 are impossible for **any** function of the two states, which is stronger than X claimed.
   - VS is an inconsistent high-level intervention, so X's F3 interpretation is downgraded.
   - New: an exact characterization of projection-realizability as a direct-sum decomposition (Proposition A2).
2. **Where the conjunction is formed.** Perfectness ∧ agreement-at-q is formed in **layer 1** (attention + MLP1),
   not layer 0.
   - Layer 0 only *propagates* agreement-at-q to the decision tokens.
   - Removing layer 1's conjunction makes the model decide exactly as if the group were ineligible.
   - Restoring only the 6-dim E-subspace restores the decision.
   - The conjunction appears **before** the behavioural switch to the Bayes rule.
3. **Mapped cross-model interchange transfers the variable** between independently pretrained parents (gain
   0.89–1.02). From C1 models it does not (0.49–0.71).
4. **Interchange accuracy produced a confirmed false positive.** A subspace trained for the perfect-only rule A1
   reaches IIA 0.83–0.93 in Bayes-behaving models. The **preservation** diagnostic exposes it.
5. **Generalization.**
   - With two colluders, models apply the trained C = 1 combination rule unchanged and fall to chance, while still
     computing the eligibility variable correctly.
   - With partial copying they treat copying as exact.
   - Off-support behaviour is model-specific.
6. **Confirmation.** All 6 frozen predictions were confirmed on 4 pre-existing held-out parents (67/72 individual
   checks).

## Answers to the final scientific questions

### 1. What survives of Investigation X's mathematics?

Survives:
- the three exact implementations and their equality;
- I1-VS and I2-VE realizability (by construction);
- the qualitative claim that equivalent algorithms realize different interventions;
- the caveat that DAS search failure is weak evidence.

Retracted:
- the I2-VS proof argument;
- the claim "I2-VS impossible by every linear interchange". It is possible by a linear map, and impossible only by
  projection, which is now proved.

Strengthened:
- I1-VE: from "no linear interchange" to "no function";
- I3: from numerical near-failure to proved impossibility.

Qualified:
- F3, because VS violates S = Σe. In I2 it targets non-integer eligibility for 49% of pairs.

Details are in `MATHEMATICAL_AUDIT_AND_RESULTS.md`.

### 2. Under what restrictions can equivalent algorithms support different interchange abstractions?

**Proposition A2.** On a product support, setting V by projection interchange is possible **iff** the state is a direct
sum a(V) + b(R) aligned with the projection.

Projection-realizability is:
- invariant to linear re-mixing and to within-block nonlinearities;
- *not* invariant to nonlinear cross-block mixing. Normalization is one such mixing, and so is storing B = Σ e U.

Two implementations of one function differ in realizable interventions exactly when they differ in this
decomposition. The R_proj ⊂ R_lin ⊂ R_any hierarchy is strict in our example:
- I2-VS ∈ R_lin \ R_proj;
- I1-VE ∉ R_any.

### 3. How is the eligibility computation constructed inside the fine-tuned Transformer?

The causal chain, confirmed on held-out models for steps (ii)–(iii):

1. **Layer 0** (attention + MLP0) carries each group's agreement-at-q bit to the q7 tokens. This is propagation: no
   interaction.
2. **Layer 1** attention gathers perfectness, an aggregate over the other five questions, and forms the conjunction
   with the agreement bit. MLP1 completes it. Both are jointly necessary.
3. The conjunction is written into a 6-dim **E-subspace** at the q7 tokens.
4. **Layer 2** reads that subspace and reweights the *other* groups' candidate evidence by (S − e_g).

Weight-level detail (QK/OV features, AND versus product in MLP1) is not established.

### 4. Which native operations causally combine perfectness and agreement?

Layer-1 multi-head attention and MLP1:
- the double-difference interaction is ≈ 0 before layer 1 and large after it;
- additive-completion removal in layer 1 eliminates it (0.00 remaining) and moves the decision to the e_h = 0
  counterfactual;
- the same removal in layer 0 changes nothing.

The split between attention and MLP1 varies by model:
- on held-out models, attention-only removal already eliminates 88–94%;
- on several development models, MLP1 dominated.

### 5. Same abstract computation through different pathways, or only similar behaviour?

**The same abstract computation through different componentwise pathways**, at the level we can test.

Shared:
- the stage at which the conjunction forms (layer 1);
- the causal role (necessary; mediated by a transferable 6-dim subspace);
- the learning order (mechanism before behaviour).

Different:
- the allocation across heads and MLP1. Cross-parent distance is 0.62 after best permutation, against 0.40 for
  same-parent seeds.

The off-support (S = 0) behaviour differs across parents (best-fitting β 0.5–0.8). Their computations therefore
coincide on the trained support but are **not** identical functions.

### 6. Does mapped cross-model interchange support transferable abstraction?

**Yes, between models that compute the variable; no from models that do not.**
- Maps fitted only on natural states transfer the source-specific part of the effect: 0.89–1.02 of the within-model
  gain between independently pretrained parents.
- Controls: a shuffled-pairing map gives ≤ 0.05; random subspaces give no transfer.
- C1→full transfer is partial (0.49–0.71), although C1 states are equally well linearly predictable (R² 0.95–0.97).

Caveat: plain IIA is inflated by a source-free mean-patch effect (IIA_disc 0.56–0.85 without any source). Transfer must
be measured relative to that control.

### 7. Natural causal mediator versus a learned subspace that merely reproduces counterfactuals?

Three properties separate them in our data:
- **Preservation.** Patching between episodes with equal variable value must not change outputs. True E: preserve_KL
  0.006–0.021, near the random floor (0.001–0.008). Decoys: 0.11–0.21.
- **Selectivity.** Responding to the input changes that define the variable and not to their controls: E D2c/D2
  0.21–0.44, against 0.39–0.52 for decoys.
- **Causal restoration under upstream removal.** Restoring the subspace after removing the upstream computation
  restores the decision.

Interchange accuracy alone does **not** separate them. The A1 decoy reached 0.83–0.93.

Limitation: a decoy equal to E on equal-value pairs would pass preservation. Preservation is necessary evidence, not
sufficient.

### 8. Can mechanism-based predictions survive independent confirmation?

Yes, with the stated limits. All six frozen predictions were confirmed on wP140–wP143: mechanism locus, necessity,
mapped transfer, false-positive diagnostic, generalization, and learning order. 67 of 72 checks passed. Three of the
five failures are in the parent with the weakest prior E fit.

Limits:
- the confirmation models are **pre-existing** Investigation X models, not newly trained;
- the pass rule is 3 of 4;
- there are no bootstrap CIs.

### 9. Which mathematical properties explain successes and failures under new source configurations?

The trained computation is the C = 1 rule p ∝ Σ_g (S − e_g) W_g L_g, with e = perfect ∧ agree(q).

- **Two colluders (C = 2).** The variable is still computed correctly (read-out ≥ 0.94), but (S − e_g) gives the
  colluders' shared value weight. The C = 1 rule therefore predicts the model's output almost exactly (KL ≤ 0.027),
  while the true C = 2 posterior differs by KL ≥ 1.35. Accuracy falls to chance (0.33) against Bayes 0.75. **The
  variable generalizes; the combination rule does not.**
- **Partial copying (β = 0.7), S ≥ 1.** The β = 1 rule predicts the model (KL ≤ 0.008), against the true β = 0.7
  posterior at ≥ 0.155. The model treats copying as exact.
- **S = 0.** The rule is undefined (0/0). Model behaviour is not fixed by the training computation. No exact-β
  posterior fits within KL 0.15, and parents differ.

The β → 1 limit of the exact posterior (the "fewest-disagreements colluder" rule) is the unique continuous extension of
the trained rule. The models do **not** follow it.

### 10. What generalizable principles, theorems, counterexamples and limitations are established?

Theorems and counterexamples:
- Proposition A2 (characterization) and Corollary A3 (reparametrization invariance);
- I2-VS ∈ R_lin \ R_proj;
- I1-VE and I3 ∉ R_any, with explicit witnesses.

Empirical principles, in this setting:
- decodability ≠ computation, since AND is linearly separable and e is decodable in C1, which lacks the conjunction;
- interchange accuracy ≠ identification, because of the false positive, which preservation and selectivity expose;
- mapped transfer requires the variable in linearly recoverable form, not just predictable states;
- representational conjunction precedes behavioural use.

Limitations:
- one task, small models;
- pre-existing confirmation models;
- no weight-level circuit;
- compensation not tested;
- no CIs.

### 11. Contribution to COR's foundations

- **T4.** A causal account of how T4's collusion-eligibility test is constructed, and of why the trained combination
  rule fails under C = 2.
- **GATE.** The layer-1 conjunction is the compatibility-conditioned computation. A 6-dim site is causally sufficient
  for its effect on the decision.
- **Accessibility.** Linear accessibility is stage-specific. The conjunction is not available before layer 1, and
  decodability overstates computation.
- **Causal abstraction.** Corrected realizability theory (A2/A3); the inadmissibility of VS; a validated diagnostic
  set (preservation, selectivity, mean-patch-relative transfer).
- **O/D framework.** **No connection was established or tested.** COR's original obstruction/discrepancy framework
  remains unvalidated by these results.

### 12. Most valuable unresolved questions

1. The weight-level layer-1 circuit: how attention computes a perfectness-conditioned function of agreement.
2. Why behaviour lags the representational conjunction (readout learning versus magnitude).
3. An identifiability result: when do preservation + selectivity + mapped transfer imply a unique mediator?
4. Admissible high-level intervention sets: a criterion excluding VS-like inconsistent interventions.
5. Off-support inductive bias (S = 0): which mechanism property determines it?
6. Newly trained confirmation cohorts, and a second relational task.

Full list: `OPEN_FOUNDATIONAL_QUESTIONS.md`.

## Failed and weakened hypotheses

- Investigation X's I2-VS proof (retracted) and its F3 reading (weakened).
- "IIA on discriminating counterfactuals identifies the algorithm". This was already falsified as X's F4; XI explains
  why (source-free mean-patch inflation, false positives).
- Partial failures in confirmation: wP142 on P4a, P4b and P2-C1; wP143→wP142 on P3c; wP143 on P5b.
- Expected but not found: any interaction formed in layer 0; any conjunction in C1 models.

## Unfinished experiments

- Same-parent versus cross-parent mapped transfer on held-out models: there are no wP seed-1 models.
- Weight-level QK/OV analysis.
- Compensation and redundancy under joint layer-2 ablation.
- Bootstrap CIs for borderline arms.
- Newly trained confirmation cohort.
- A direct empirical false-negative test.
- Comparison of alternative ablation completions (mean ablation).

## Where to look

| document | content |
|---|---|
| `MATHEMATICAL_AUDIT_AND_RESULTS.md` | X1 audit, proofs, Propositions A1–A3 |
| `NATIVE_CAUSAL_MECHANISMS.md` | decodability, factorial, ablation, restoration, cross-parent comparison, dynamics |
| `FROZEN_PREDICTIONS_AND_CONFIRMATION.md` | P1–P6, freeze record, results |
| `REVIEWS_AND_ADJUDICATION.md` | same-agent adversarial reviews |
| `OPEN_FOUNDATIONAL_QUESTIONS.md` | open problems |
| `REPRODUCTION.md` | dependency inventory, scripts, clean-extraction procedure and result |

Cross-model interchange and false-positive (decoy) results are tabulated in `FROZEN_PREDICTIONS_AND_CONFIRMATION.md`
Part II and summarized in `NATIVE_CAUSAL_MECHANISMS.md` §8.
