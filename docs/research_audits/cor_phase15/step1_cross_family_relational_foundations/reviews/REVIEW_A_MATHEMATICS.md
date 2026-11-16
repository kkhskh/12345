# Review A: mathematics (COR Investigation XIV, cross-family relational foundations)

Independent adversarial agent review, not human peer review.

**Scope.**
- CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md (CFMS);
- logs/FREEZE_X.md and scripts/score_X.py (read only, not run);
- the scripts and logs g_rules_exact, g_order_ceilings, t4_order1 and math_verify;
- the generator code (tasks_ext.sample_gate / bayes_gate / bayes_gate_no_relation; gen_family.py; latent_reliability bayes.Env / sample_fast / episode_posterior; genlib.posterior_closed);
- T4_FOUNDATIONAL_EVIDENCE_LEDGER.md, spot-checked against the cited sibling documents.

**What was not touched.** I did not open, load or list any file under checkpoints/gate/s3, checkpoints/gate/s4, logs/unit_metrics_I*.jsonl or logs/score_*.json. I trained no models, ran neither scorer, and modified no existing file.

**Reviewer computations.** All are exact enumerations over the 3^6 GATE inputs with the exact P(x), and they took a few CPU-seconds to minutes. I wrote them as throw-away scripts outside the package (/tmp/revA/v1–v5.py). Each one is described below in enough detail to rerun.
- **(R1)** Population A and G for reference rules, using the same definitions as the scorer: centred log-probabilities; G with flips weighted as in g_order_ceilings; A on the mask max|ΔGF| > 0.05.
- **(R2)** The same quantities for random (non-optimal) order-1 and order-2 log-linear rules.
- **(R3)** A scale path. The rules ∝ PB^t and ∝ O1^(1−t)·PB^t, with O1 the order-1 optimum.
- **(R4)** The exact interaction order of the gate-free rule: the best order-k evidence-only fit to GF.
- **(R5)** A partial-regression version of A.
- **(R6)** The finite-sample noise of G at n = 18,000.
- **(R7)** genlib.posterior_closed against bayes.episode_posterior on 3,000 native T4 episodes.

---

## Findings

### F1 [major]: X1a cannot separate the prediction from its stated competitor, because G and A are scale-dependent and the accuracy threshold puts no upper limit on gate use

**Evidence (R3).**
- Take a model that learns *jointly* (relation and evidence together) at a small overall logit scale: P ∝ PB^t, the Bayes logits shrunk by t. Its exact values are G = t, A = 0.71·t and accuracy 0.900 for every t > 0.
- For t ≤ 0.1, that gives G ≤ 0.1, A ≤ 0.07 and accuracy 0.90 ≥ 0.75. So it meets every clause of X1a: acc ≥ 0.75, A ≤ 0.3, G ≤ 0.1.
- So the stated competing explanation ("relational and evidential components are learned jointly, so A or G is already large when accuracy first reaches 0.75") does *not* follow from joint learning. A joint learner whose logit scale grows over 75+ steps passes X1a.
- Second loophole: O1^(0.9)·PB^(0.1), an order-1 rule with 10 % Bayes mixed in, has G = 0.100, A = 0.071 and accuracy **0.891**.
  - The reason is that 22.2 % of the input mass gives the order-1 optimum an exact argmax tie (x₂ = x₃ ≠ x₄ = x₅ and similar patterns).
  - Any gate use breaks those ties correctly.
  - So "accuracy ≥ 0.75" admits models that already get 90 % of Bayes accuracy from the gate.

**Consequence.**
- As written, X1a is mostly a test of "G and A stay small for 100 steps once accuracy is ≥ 0.75". That cannot tell a real order-1 phase from slow growth of the overall scale.
- The development seed (accuracy 0.80, NLL 0.46–0.49 on steps 150–400) does show a real order-1 plateau, but the scorer does not check that signature.
- Because the dev seed already shows the pattern, X1a under the identical recipe is close to a replication, not a risky prediction.

**Correction (to be frozen as an addendum before any confirmation output is read; otherwise report the result as "X1a as frozen, with this caveat").**
- Add a fit clause that identifies the order-1 solution: NLL ≤ 0.50 (the order-1 ceiling is 0.462; uniform → Bayes at G = 0.1 has NLL 0.80). Alternatively, use KL(order-1 optimum ‖ model) ≤ 0.05.
- And/or add an accuracy band: 0.75 ≤ acc ≤ 0.83, since any systematic gate use moves accuracy to about 0.89.
- And/or report scale-normalised indices: G/s and A/s, where s is the overall slope of the model's centred log-probabilities on Bayes's. The order-1 optimum has s = 0.586.
- Restate the competitor as "the model's logits are, up to scale, already Bayes-like at the first competent snapshot". Check it with G/s and the NLL clause.

### F2 [major]: Proposition 4's account of T4 ("trust needs order-12 relations") is a domain artefact; in the log domain T4's trust evidence is low order

**Evidence.**
- With ρ = 1 the colluder weight for pair h is ∏_q [a_{q,h} = 1]·K^{−Q}. Its logarithm is Σ_q log[a_{q,h} = 1] + const, a sum of per-question order-2 terms that take values in {0, −∞}.
- The reliable and noise factors are also per-question functions of the within-pair equality only (genlib/bayes q_lik; checked: P(x_{q'} | Z) depends only on the three equality indicators).
- So log P(Z | X_{−q}) is **additive over questions in order-2 features**, exactly the "agreement counts" of Theorem S. The ANOVA order is defined on logits, so the "conjunction of 12 sources" says nothing about logit order.
- The high order of the T4 *decision* comes from marginalising over the uncertain Z (a log-sum-exp over 6 configurations). It does not come from a high-order trust relation.
- The same mechanism shows up inside GATE. By (R4), the gate-free mixture ½L₂₃ + ½L₄₅ is **exactly order 4**: the best order-2 fit leaves KL 9.7e−4, order 3 leaves 4.1e−4, and order 4 leaves 6e−14. That is higher than the Bayes rule's order of 3.
- So "decision interaction order" is not monotone in how relational or how hard a rule is. If X1b holds, GATE learners acquire an order-4 rule before an order-3 one.
- Minor errors in the same proposition:
  - "the conjunction of 2Q equality relations" should read "Q equality relations over 2Q sources";
  - "it cannot represent the zero": the decision P(y | X) is never zero. Only P(Z | X) has zeros.
  - The order of the T4 decision is asserted, not proved.

**Correction.**
- Rewrite Proposition 4. The T4 trust log-evidence is order 2 per question and additive across questions, with hard constraints. The decision is a mixture over an uncertain latent, which inflates the interaction order.
- Drop "order-12 trust relation" as the contrast with GATE. The true structural contrast is: the latent is *deterministic and local* (GATE) versus *uncertain and accumulated over questions* (T4).
- Remove, or flag as speculation contradicted by the source (see F9), the link from "order" to "T4 not learned from scratch".

### F3 [major]: the "common principle" (§6) is mostly definitional; its one non-trivial prediction needs an extra assumption, and the T4 analogue cited for it measures something else

**Evidence.**
- Items 1–2 restate A1 ("no single-source component, hence only relations identify it"). Given A1, item 2 is a tautology.
- Item 3, "decision order ≥ trust-relation order + 1", is proved only for GATE (Propositions 2–3).
  - It needs a non-degeneracy condition: if both branches of the gate gave the same evidence function, the order would collapse.
  - It is ill-posed for T4 (F2): the trust relation is order 2 in the log domain, and the decision order is high because of marginalisation, not "trust order + 1".
- Consequence (a), source-wise ceilings below Bayes, follows from A1 plus "Bayes depends on Z". The principle does not predict its size.
- Consequence (b) is a sufficiency/invariance fact, and as worded it is too broad (F8).
- Consequence (c), the only dynamic prediction, is labelled [spec]. It rests on an external simplicity-bias hypothesis, not on the principle.
- The cited T4 counterpart (ledger N12, Investigation VIII) does not measure source-level interaction order. Investigation VIII's readers were given *per-group agreement counts and target reports* as inputs (MAIN_RESEARCH_RESULT.md §1). Their "separable phase" is additive-over-groups in features that are already relational at source level. It is not an order-1 (source-wise) phase.

**Correction.**
- Do not present §6 as a common principle with cross-family predictive content.
- What survives:
  - (i) a shared symmetry, A1, verified in both generators;
  - (ii) an exact diagnostic calculus for GATE (order ceilings and the G/A reference values);
  - (iii) a GATE-specific staircase hypothesis, X1.
- Remove the claim that X1 tests N12 (FREEZE_X §4 and ledger N12 "history").
- **Verdict on the principle: rejected as a principle. It is retained as a description of a shared symmetry.**

### F4 [moderate]: X1b does not follow from the mathematics plus "simplicity bias" without a further assumption about magnitude; the development data are weakly against it

**Evidence.**
- The staircase or leap results this alludes to (degree-ordered learning) depend on the size of each component's signal as well as its degree.
- Here the order-2 increment is small (NLL 0.462 → 0.431, a gain of 0.031 nats). The order-3 increment is large (0.431 → 0.182, a gain of 0.25 nats).
- By Proposition 3, the gate has *no* lower-order projection: it is a pure "leap". A degree-staircase account therefore makes no clear prediction about whether the weak order-2 term or the strong leap comes first.
- The dev seed had G = 0.04–0.12 while A rose only from 0.03 to 0.15. By (R1), Bayes itself has A = 0.71. A model moving along O1 → Bayes has A ≈ 0.71·G, so G = 0.12 implies A ≈ 0.085.
- So the dev rise in A is largely or entirely accounted for by gate use. FREEZE_X's statement that X1b "is not visible in the development data" understates this: at step 400 the dev seed was already at the G threshold, with little agreement use.

**Correction.** Present X1b as a hypothesis that needs "order dominates magnitude", not as a consequence. Disclose the A ≈ 0.71·G accounting of the dev seed.

### F5 [moderate]: the A reference table is wrong for Bayes and inexact for order 2, and A is contaminated by gate use

**Evidence (R1).** Population values on the scorer's mask (44.4 % of the mass):

| rule | A | G |
|---|---|---|
| order-1 optimum | 0 | 0 |
| order-2 optimum | 0.962 | 0 |
| gate-free | 1 | 0 |
| order-3 optimum / Bayes | **0.712** | 1 |
| pair23 | 0 | 0 |
| x₃-only | 0 | 0 |

- CFMS §3's table gives "order 2: A = 1; Bayes: —". The correct values are 0.96 and 0.71.
- So "A ≥ 0.5" is also reached by a gate-only learner, and the scorer's reported t_A_half is not a stage marker.
- X1b's joint clause G ≤ 0.1 bounds this contamination at about 0.07, so X1b is still meaningful. X1a's A ≤ 0.3 and the t_A_half reporting are affected.

**Correction (R5).** Use a partial index: regress the model's swap response jointly on ΔGF_swap and ΔBayes_swap (population correlation 0.31).
- The coefficient on ΔGF is exactly 0 (order 1), 0.962 (order 2), 0 (Bayes) and 1 (gate-free).
- This isolates agreement use from gate use. Report it alongside the frozen A, and fix the table.

### F6 [moderate]: Proposition 3 claims "every optimum" but proves only "some optimum"; the equivariance step is loosely stated

**Evidence.**
- The proof shows that a particular rule, Q* extended by zeros, meets the moment conditions, and is therefore *an* optimum. "Can be chosen equivariant" also only gives existence.
- "Every" needs uniqueness of the optimal *predictive* function. This holds:
  - every x has P(x) > 0;
  - every y has positive posterior, so the objective is coercive modulo per-x logit shifts;
  - log-sum-exp is strictly convex modulo those shifts;
  - so the optimal logits are unique up to per-x constants, and the unique optimum is automatically equivariant.
- Proposition 2's "not ≤ 2" relies on this uniqueness.
- "Both the generator and the optimum are equivariant under relabelling the candidates" is false if the relabelling also acts on (x₀, x₁): g is not invariant under, for example, 0 ↔ 1. The relabelling must act on (y, x_E) only. That is all step 4 needs.
- Checked and correct:
  - x₁ ⊥ (g, Y, x_E): P(x₁ | g) is uniform for both g, and (Y, x_E) depend on (x₀, x₁) only through g;
  - x₀ ⊥ (g, Y, x_E);
  - the moment computations in steps 2–4;
  - the list of order-≤ 2 features is complete.

**Correction.**
- Add the uniqueness argument and restrict the equivariance to (y, x_E).
- Better, add the stronger result (R2), which gives Proposition 2's lower bound without any optimisation argument: **every** order-≤ 2 rule, not just the optimum, has population gate-flip slope exactly 0.
  - Sketch: dm decomposes into terms w(x₁, y), w(x₀, x₁, y) and w(x₁, x_s, y).
  - Given g, x_s and Y are independent of (x₀, x₁, x₁′).
  - Both x₁ and its redraw x₁′ are marginally uniform.
  - E[dB_y | x₀, x₁] is constant in y, so each term's covariance with dB vanishes.
- Numerically: random order-2 rules give |G| < 1e−17 in population. With n = 18,000 the sample SD is 0.002 (R6).

### F7 [minor]: Corollary 3 gives an insufficient reason

**Evidence.**
- "All four evidence sources have identical marginal laws, so the optimal order-1 rule is symmetric" does not follow. The order-1 optimum depends on the joint law, and the joint law is *not* invariant under the swap x₃ ↔ x₅ (the pairing changes).
- A correct reason:
  - by Proposition 3 the optimum is evidence-only;
  - the evidence law (a ½/½ mixture over g) is invariant under (2 3), (4 5) and (2 4)(3 5), which generate a dihedral group of order 8 acting transitively on {2, 3, 4, 5};
  - by uniqueness, averaging over the group gives a common table f(x_s, y);
  - so the rule depends only on the multiset of evidence values.
- The conclusion is right. Stronger still, by (R2) **every** order-1 rule has population A = 0, so the reference value A = 0 holds for the whole class, not just the optimum.

**Correction.** Replace the justification as above.

### F8 [minor]: consequence (b) of the principle is over-broad, and X3 cites a non-instance

**Evidence.**
- (b) "An edit that changes values but preserves every trust relation cannot [change the decision]" is false for the target's own evidence sources. Changing x₂ in GATE, or x_q in T4, changes Bayes and preserves g.
- The verified statements (math_verify: max |Δ| = 0 for both) concern only *non-evidence* sources: (x₀, x₁) in GATE, and non-target questions in T4.
- X3 says the continuation's "biased errors" family is an instance of A1 failure. It is not:
  - roles_engine.rel sends errors to (y + 1) mod K, a circulant (doubly stochastic) kernel;
  - so with uniform Y every single-source marginal is still uniform and A1 holds;
  - and the od_bridge "biased_errors_0.8" spec has three reliable groups, so there is no trust latent at all.
- math_verify's docstring promises a "per role" chi-square. It actually tests per source position, pooled over roles. The exact calculation covers the claim; the empirical check does not test it per role.

**Correction.** Restrict (b) to sources that carry no answer evidence for the target. Delete the biased-errors example, or replace it with "biased noise / non-uniform answer prior". Fix the math_verify docstring.

### F9 [moderate]: CFMS §3 and ledger N2 omit a source fact that cuts against the order account

**Evidence.** step1_latent_reliability/MAIN_ANSWER.md (T4 section) says three things the ledger leaves out:
- "DeepSets learns T4 to Bayes in 3/3 seeds";
- "the failure is an optimisation barrier at this budget and initialisation, not inexpressibility";
- the source's own hypothesis: "agreement ⇒ reliable is actively wrong in T4, so there is no easy partial solution".

Ledger N2 states only "T4 from scratch was not learned (0/3)". CFMS Proposition 4 [spec] then attributes the GATE/T4 learnability difference to trust-relation order.

**Correction.** Add the DeepSets 3/3 result and the source's hypothesis (the non-monotone agreement map, M3) to N2, and to Proposition 4 as a competing explanation.

### F10 [moderate]: ledger entries stated more favourably than the latest corrected source

1. **N20.** The heading keeps "different combination operator".
   - XIII-B §2 explicitly says the reading as a "different combination operator" is *not unique*: it is algebraically identical to tempered evidence plus re-sharpening, a lapse floor moves ρ by about 0.1, and ρ explains only about 15 % of the early class's misfit. What is supported is "different intervention-response laws".
   - CFMS §5 inherits the stronger wording ("combine ... with different nested power means"; "mixture-versus-log-linear distinction").
   - Correction: use the source's supported wording in both places.
2. **N3.** Status "CAUSAL (seeds 0, 1)" is attached to "low-dimensional and relational (a group-indexed function of agreement counts; R² ≤ 0.04)".
   - In the source, the 1-D direction is [R] for seeds 0–1.
   - The "group-indexed function of agreement counts" content, including R² ≤ 0.04, is [PH].
   - Correction: split the claim, giving the content part EXPL/PH.
3. **N19.** Status "CAUSAL (dev + conf ...)". The source labels item 1 "[dev; partly conf]", and only the layer-0 start (0.84–0.95) is [conf]. Correction: say which parts are conf.
4. **N24.** "Matched noise ... does not change the route class (0/6)" is accurate, but the entry omits the source's §3: strong early noise flipped the mixture base (2/2), and "some general disturbances can also redirect". Correction: add that.
5. **N12** (see F3). The "history" note "tested by XIV X1 as a cross-family prediction" is a mis-classification. X1 does not test N12's feature-level separable phase.
6. **N21.** The confirmation n = 2 is not stated. A uniform row reaches R² 0.86–0.88, which weakens "act causally as mixture weights". Minor.

**Checked and consistent with source** (no change needed):
- M1, M2, M3 (0.636/0.743; 0.359);
- M4 (5.5 %, 37 %, ≥ 729);
- M6 (bijective on the support);
- M7 (531,441; 3.3e−16);
- N1 (IIA 0.84–0.85, 3/3);
- N2 (4/4, 3/4, 3/4, 2/4);
- N5 (88 %);
- N8 (Δ 0.149 vs 0.45);
- N9 (+0.19 / −0.51 / +0.70, p = 7.8e−5, 16 parents);
- N10 (0.72 → 0.30, 8/8, 7/8);
- N11 / N13 (6/6; 0.99 vs 0.83–0.87);
- N14 / N15 (4/4);
- N16 (0.11–0.27; IIA 0.85–0.97);
- N23 (p = 0.006; per-parent 0.086 / 0.018; homogeneity caveat).

### F11 [minor]: details of the scorer and freeze

1. **Verdict rule ambiguity.** FREEZE_X says "FALSIFIED if any competent model fails" and also "INCONCLUSIVE if fewer than 2 models are competent". With exactly one competent model that fails, the text is contradictory. The code returns INCONCLUSIVE. State the precedence.
2. **Hash coverage.** FREEZE_X_HASHES covers score_X.py, g_train.py, gatelib.py and the two reference logs. It does not cover:
   - tasks_ext.py, which defines sample_gate, bayes_gate and the gate-free rule the scorer uses;
   - fwd3.py, train_tasks.py and ckpt.py, the forward pass and loader;
   - FREEZE_X.md itself.

   "Fail-closed against changed frozen files" is therefore partial. Hash these files too, or record their hashes in the report.
3. **Run length.** "4 consecutive snapshots (100 steps)": at 25-step spacing, 4 snapshots span 75 steps.
4. **Threshold.** G ≤ 0.1 lies inside the dev plateau range (0.04–0.12). The threshold is not fitted, but it is fragile: one snapshot at 0.11 breaks a run. Report the longest run at G ≤ 0.15 as a sensitivity check (not a verdict).
5. **What is sound.**
   - The reference values hold exactly in population for the *whole* classes, not only the optima: G = 0 for every order-≤ 2 rule; A = 0 for every order-1 rule; A = 1 for the gate-free rule by construction. Sampling noise of G at n = 18,000 is about 0.002 (R6).
   - X1b's G ≤ 0.1 clause protects it from gate contamination.
   - X1b is not implied by accuracy: order 1, order 2 and gate-free all have accuracy 0.8008.
   - I found no indexing or scoring error in score_X.py: the run counting, the mask, the flip construction and the competence rule match the text.

### F12 [minor]: the numerical scripts are correct, with small caveats

- **g_rules_exact.**
  - The pgate factor P(g)·P(x₀)·P(x₁ | x₀, g) = ½·⅓·(1 or ½) is correct: it sums to 1 and the assertion passes.
  - The variable `pg` and its comment are wrong but unused; delete them.
  - The gate-free rule (½·L₂₃·K^{−2} + ½·L₄₅·K^{−2}) is exactly P(y | x_E).
  - "sourcewise" is naive Bayes with the correct marginal m = ½β + ½/K.
  - The MI values are reproduced.
- **g_order_ceilings.**
  - The design (one-hot tables for every subset with |A| ≤ k) and the exact-P(x) objective are correct. The gradient is analytic.
  - Accuracy uses the first argmax with no tie-splitting. That is harmless here by symmetry: 0.8008 equals the tie-split value. But 22 % of the mass is exactly tied for order 1 and order 2 (see F1).
  - The flip and swap slopes match the scorer's definitions.
- **t4_order1.**
  - It is a sampled logistic fit, labelled as such, and adequate.
  - genlib.posterior_closed (ENV1 = Env(3, 1), β = 1) equals the native bayes.episode_posterior to 2e−15 on 3,000 episodes (R7).
- **math_verify.** Correct, apart from the docstring point in F8.
- **Generator descriptions in CFMS §1.** All match the code:
  - GATE: x₁ law, β = 0.9, e = (1 − β)/2, role swap;
  - T4: Fam(pc = 1, ρ = 1), configurations uniform over the 6 ordered pairs, the colluder sharing c ~ U(K) independent of Y, α = 0.8.
  - bayes.Env's `alpha5 = 0.5` is used only by task T5 and is irrelevant to T4. The native "beta" (copying) is CFMS's ρ.
  - Note the name clash: GATE's β is a reliability, the native T4 β is a copy probability, and §5's ρ is a power-mean exponent.

### F13 [minor]: Proposition 6 and X2 are correct but near-trivial

- **Proposition 6.** Non-identifiability holds for the *Bayes* rule, where the weights are a point mass. A trained model with an imperfect gate has non-degenerate weights, so ρ is behaviourally identifiable off the optimum. "Cannot even be measured in GATE" should read "is not determined by Bayes-optimal behaviour in GATE".
- **X2.** [x₀ + x₁ ≡ 0] and [x₁ = −x₀] are the same function. The point that implementation descriptions are not behaviourally identifiable is true but trivial.

---

## Claims that survive

1. **A1 in both generators** (Proposition 1). The single-source law is uniform under every role; exact MI I(g; x_s) = 0; in T4 the report sequences are i.i.d. uniform.
2. **GATE's Bayes rule has interaction order exactly 3** (Proposition 2). The upper bound is as given. The lower bound holds, most cleanly via the population result that *every* order-≤ 2 rule has G = 0 (F6).
3. **Proposition 3** (the expected-NLL-optimal order-≤ 2 rule ignores the gate). It is unique, given the uniqueness argument in F6. The steps that x₀ and x₁ are each independent of (g, Y, x_E) are correct.
4. **Exact ceilings.** Order 1: NLL 0.462, accuracy 0.801. Order 2: NLL 0.431, accuracy 0.801. Gate-free: NLL 0.4305. Bayes: NLL 0.182, accuracy 0.900.
5. **T4.** The order-1 ceiling is about 0.872 NLL, and adding other-question sources at order 1 adds nothing.
6. **Corollary 3** (order-1 rules are swap-invariant), with the corrected justification. A = 0 for every order-1 rule.
7. **Screening-off** (Proposition 5) and the invariance results (GATE joint shift; T4 non-target relabelling that preserves equality), restricted to non-evidence sources.
8. **Proposition 6**, as a statement about Bayes-optimal behaviour.
9. **Counterexamples X1, X2, X4 and X5.** X3 survives in its "biased noise / non-uniform prior" form, but not with the biased-errors citation.
10. **The scorer's measure definitions and exact reference values.** G: 0 for all order ≤ 2, 1 for Bayes. A: 0 for all order 1, 1 for gate-free. The corrections are Bayes A = 0.71 and order-2 A = 0.96.

## Claims that do not survive as stated

- **The "common principle" as a principle with cross-family predictive content** (F3): rejected.
- **Proposition 4's "order-12 trust relation"** and its link to T4 learnability (F2, F9).
- **"X1 tests N12"** (F3).
- **X1a as a discriminating test** of staircase learning against joint learning (F1).
- **X1b as a consequence** of mathematics plus simplicity bias (F4). It is a hypothesis.
- **The stage table entries** for order-2 A and Bayes A (F5).
- **Ledger N3, N19 and N20 as worded**, and the N12 history note (F10).
