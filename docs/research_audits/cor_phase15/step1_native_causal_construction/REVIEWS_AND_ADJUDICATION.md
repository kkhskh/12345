# Adversarial reviews and adjudication (Investigation XI)

**Disclosure.** These reviews were written by the same agent that ran the investigation, deliberately taking an
adversarial stance after the confirmation results were in. They are **not independent peer review**. Where a criticism
could be tested cheaply, it was tested; where it could not, it is recorded as an open limitation.

## R1 — Mathematics (Theorem X1)

**Criticism 1.1.** "The correction of X1 is itself a numerical claim dressed up as a proof."
- *Adjudication: rejected.*
- The I2-VS non-projectivity proof is algebraic. Three on-support Δm vectors have determinant 2, which forces P = ½11ᵀ,
  and that P is not idempotent. The U-block argument is exact.
- The numerical checks (C4, E1) only corroborate.
- I1-VE and I3 impossibility rest on explicit witness pairs (C5, C6), which is a complete proof of non-existence of any
  realizing function.

**Criticism 1.2.** "Proposition A2 assumes a product support of (V, R). Natural pairs are not a product set, so the
characterization may not apply to the networks."
- *Adjudication: accepted as a limitation.*
- The (⇒) direction needs enough combinations of V_s with R_b to force the decomposition. On restricted supports, a
  weaker "decomposition on the support" holds.
- The proposition is used only to explain *why* realizability depends on representation format. It is not applied
  quantitatively to the networks.

**Criticism 1.3.** "Calling VS 'illegitimate' is a choice of high-level semantics."
- *Adjudication: partly accepted.* VS is a well-defined intervention on a structural model with S as a node. It is
  inconsistent with the deterministic constraint S = Σe, and it maps I2 off its input image for 49% of pairs.
- We downgrade VS from "invalid" to "not a discriminating test": its failure cannot falsify a pooled implementation.
  That is the claim that matters for F3.

## R2 — Mechanism versus transmission

**Criticism 2.1.** "The factorial interaction at resid2 could be transmitted, not formed. Layer 1 might just copy an
interaction computed elsewhere, for example by layer-0 attention softmax nonlinearities."
- *Adjudication: rejected by the data.* The interaction is ≈ 0 at mid0 and resid1 (≤ 0.014 rel., conf).
- Removing layer 0's own interaction changes nothing (Δ = 0.000, P2c).
- Layer 1 is the first place it is non-zero, and removing it there eliminates it (P2a).
- Within layer 1 we cannot fully separate "attention forms, MLP1 amplifies" from joint formation. MLP1-only removal
  leaves 0.31–0.62, attention-only removal 0.06–0.12.

**Criticism 2.2.** "The additive-completion ablation creates activations no input produces. The decision change may be
generic off-distribution breakage."
- *Adjudication: rejected, with a caveat.* If the change were breakage, there would be no reason for the output to land
  on one specific Bayes counterfactual, cf_e0 (KL 0.03–0.04, against 0.53–0.58 from Bayes, with 67–72% of pairs closer
  to cf_e0).
- Restoring only 6 E-subspace dimensions restores the decision, while 6 random dimensions do not. The ablated state
  therefore differs from the intact one in a *specific, interpretable* way.
- The caveat stands: the ablated activation is off-distribution, and other completions (for example mean-ablation)
  were not compared.

**Criticism 2.3.** "The factorial cells may be off-support."
- *Adjudication: rejected.* Edits touch only group h, which disagrees at q, so h is not the colluder. Another group is
  eligible in all four cells, so S ≥ 1 throughout and every cell has positive probability under T4.
- Sanity check: ΔS = 0, 0, 0, 1 (TRUE) and 0, 0, 0, 0 (CTRL), exactly.

## R3 — Intervention validity and alignment leakage (Workstream C)

**Criticism 3.1.** "Mapped interchange is trivial. The q7 states of any two models are linearly related (held-out
R² 0.92–0.99), so mapping then projecting must work."
- *Adjudication: rejected.* The C1→full maps have the same R² (0.95–0.97) yet transfer only 0.49–0.71 of the
  source-specific gain, against 0.89–1.02 for full→full. High R² does not imply transfer of E.
- What the control shows is that a model's states must *contain the variable in a linearly recoverable form*. C1 does
  not: its e is decodable by a classifier but not linearly mappable onto the full model's E coordinates. This is
  consistent with C1 lacking the conjunction (P1c).

**Criticism 3.2.** "The map could leak counterfactual information."
- *Adjudication: rejected by construction.* The maps are fitted by ridge regression on natural episodes only (stream
  63_000_000), with no pairs, counterfactuals or outcomes.
- B's subspace comes from B's own DAS, fitted in Investigation X on stream 62_500_000.

**Criticism 3.3.** "IIA on discriminating pairs is the standard metric. Why trust the 'source-specific gain' instead?"
- *Adjudication: our central methodological point.* A source-free mean patch reaches IIA_disc 0.56–0.85 (base 0.19).
  Most of what IIA measures here is "move the base to a typical eligibility state".
- Gain relative to the mean-patch control isolates information that comes from the source. This is why Investigation
  X's F4 (C1 vs full IIA_disc) was not diagnostic.

**Criticism 3.4.** "No same-parent seed pairs among the held-out models."
- *Adjudication: accepted.* The same-parent vs cross-parent comparison is development-only. Seed pairs had gains of
  1.00 / 1.00 / 0.99; cross-parent pairs 0.92–1.00, so no difference was detected.
- No wP seed-1 models exist. This is recorded as unfinished.

## R4 — Redundancy

**Criticism 4.1.** "There may be an alternative route that activates only when layer 1 is removed (compensation)."
- *Adjudication: not tested; open.*
- What we know: removing the layer-1 conjunction drives the E-interaction to exactly 0.00 and the decision to cf_e0.
  No compensating route is *visible* at resid2 or in the decision.
- But layer 2 was not ablated jointly, and alternative routes in other token positions were not searched.

## R5 — False positives and negatives (Workstream C2)

**Criticism 5.1.** "Preservation might just penalize decoys because they were trained on harder targets."
- *Adjudication: partly accepted.* Decoys have higher training loss. However:
  - the random (untrained) subspace has *lower* preserve_KL (0.001–0.008 in full models) than every trained decoy;
  - the decoys' high preserve_KL is therefore created by training toward a wrong variable, not by the patching
    operation itself.
- The true-E subspace stays near the random floor (0.006–0.021, excluding wP142 at 0.060) while matching the
  counterfactuals. This is the signature of a variable that changes outputs *only* when its value changes.

**Criticism 5.2.** "Only three decoy rules were tested. A cleverer decoy could pass preservation."
- *Adjudication: accepted.* Preservation is necessary, not sufficient, evidence. A decoy that equals E on the
  equal-value pairs would pass by construction.
- The defensible statement: interchange accuracy produced a confirmed false positive (the A1 rule, IIA 0.83–0.93 in
  Bayes models), and preservation plus selectivity rejected it.

**Criticism 5.3.** "False negatives were not examined."
- *Adjudication: partially addressed.*
- At the theory level: Proposition A2 and Corollary A3 give conditions under which a genuinely computed variable fails
  projection interchange (format mismatch, cross-block nonlinear mixing).
- Empirically, the S-variable failure (Investigation X F3) is our main candidate false negative, but it is
  uninterpretable because VS is inconsistent (math audit §4).
- No direct empirical false-negative test was run.

## R6 — Statistics

**Criticism 6.1.** "3/4 parents is a lenient pass rule; 5 of 72 checks failed."
- *Adjudication: disclosed.* The rule was frozen before confirmation. All failures are reported.
- Three of the five failures are in wP142, the parent with the weakest prior E fit. This suggests a model-quality
  dependence of the diagnostics (P4 in particular), not a failure of the mechanism claims (P1, P2 and P6 hold 4/4 on
  every mechanism arm).

**Criticism 6.2.** "No confidence intervals."
- *Adjudication: accepted.* Per-model estimates use 400–6000 pairs.
- Where reported, standard errors are small (logit-interaction SE ≈ 0.03).
- The effect sizes in the decisive tests are far from their thresholds: interaction ratios 5–7 vs 2; remaining
  interaction 0.00 vs 0.05; KL gaps by more than 10×.
- Bootstrap CIs were not computed and would be needed for borderline arms (P4b wP142, P3c, P5b).

**Criticism 6.3.** "The confirmation models are not new."
- *Adjudication: accepted and disclosed.* wP140–143 were trained and evaluated in Investigation X. XI never analysed
  them before the freeze, and every XI quantity is new.
- Known X properties (F1–F7) constrain them, but do not imply the XI predictions. For example, F7's trajectories do not
  determine when the factorial interaction appears.

## R7 — COR scope

**Criticism 7.1.** "These results are being used to claim COR is validated."
- *Adjudication: no such claim is made.* The results characterize how one Bayesian relational computation (T4
  eligibility) is constructed in small fine-tuned Transformers.
- No connection to the original O/D framework was found or tested. Step 1 is not closed, and no thesis is selected.
