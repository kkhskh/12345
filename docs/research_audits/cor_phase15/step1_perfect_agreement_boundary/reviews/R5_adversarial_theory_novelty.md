# R5 — Adversarial theory and novelty review (pre-review commit 9389496)

**Scope.** I read the mandate, the required documents, CAUSAL_EVIDENCE_UPDATE_MECHANISM, `scripts/cohort6_train.py` and
the logs. No package file was edited. My checks are new files in `reviews/checks/` (`R5_pa_discount_fit.py` with its
`.json`, `R5_regime_tradeoff.py`, `R5_response_weighted_mix.py`). They use the 16 cohort models at step 4000, 2500 conf
episodes, and the same targets as `fitted_operators.py`.

## Findings

**1. [MAJOR] Frozen-feature transfer is missing from the headline scope.**
*Location:* MAIN_RESULT answer paragraph and §5 "Not claimed".
*Problem:* `cohort6_train.py` freezes layers 0–1 and the embeddings. It re-initialises layer 2 and the readout, then trains
only those on T4 with the same objective and schedule in both arms. Because of this design, the only things that differ
between arms are the frozen pretrained weights and the seeds. Any reproducible arm difference must therefore come from what
the frozen representation makes cheaply available to one freshly trained layer. That is a known kind of limit on
frozen-feature transfer. The answer paragraph never says "frozen", and §5 does not list this confound.
*Correction:* Add "under frozen-feature transfer (layers 0–1 and embeddings frozen, layer 2 retrained)" to the answer, §5
and Outcome A; nothing is claimed for full fine-tuning.

**2. [MAJOR] The word "interpreted" conflicts with the package's own causal reading.**
*Location:* MAIN_RESULT §5 "Claimed", OBSERVABLE §4, and the mandate §16 distinction between representation and
interpretation.
*Problem:* CAUSAL §3.3 says reading (i) is "more plausible": P-count features encode 5/5 in a form that layer 2 cannot
condition on the other groups. That reading puts the difference on the **representation-accessibility** side of the mandate's
distinction. The downstream reader (layer 2, same objective) is the same kind of learner in both arms. The headline "perfect
agreement is *interpreted* differently" asserts the side its own evidence does not favour.
*Correction:* Say "the decision response differs"; encoding vs downstream reading is unresolved, (i) favoured but not shown.

**3. [MAJOR] "Grows while KL improves contradicts an approximation-error account" is overstated.**
*Location:* MAIN_RESULT §2 "Training dynamics", and TRAINING §2.1.
*Problem:* It compares B45-R1 with B34-R1, which share no trigger. The relevant competitors are B45-R2/R3, which share "g
reaches 5/5". Over P-count transfer:
- B45-R3 falls from 1.92 to 0.10 and B45-R2 from 2.27 to 0.66, while B45-R1 falls from −0.34 to −0.74;
- at step 4000, P-count is closer to Bayes in R3 than T1 is (0.10 vs 0.34);
- R1-after-type inputs (two perfect groups voting together at q) are only **2.9 %** of natural decisions (my check).

This is the textbook worst-group-versus-average pattern of shortcut or spurious-feature learning. "5/5 → discount" is right
on the majority subsets (R2/R3, 68 % of oriented F45) and wrong on a small explaining-away subset (R1). A limited ERM
learner improves average loss by fitting the majority while minority error grows (Sagawa et al. 2020; gradient starvation),
which is corrected Proposition 1 at work. The trajectory rules out *uniform* approximation error, **not** a feature-limited
approximator.

Partial evidence against a pure shared trade-off: T1@2000 (R1 −0.04, R3 0.38) dominates P@1000 (−0.62, 0.63). But no T1
snapshot reaches P@4000's R3 level (0.10), so for converged P-count models a different point on a shared frontier is not
excluded.

*Correction:* "Inconsistent with uniform approximation error; consistent with curriculum-dependent worst-group error."
Report R1/R3 trajectories jointly.

**4. [MODERATE] Point (a): the discount is learned during transfer, not carried over from pretraining.**
*Location:* MAIN_RESULT §3 consequence, and CAUSAL §3.3.
*Problem:*
The P-count objective has no discount and layer 2 is re-initialised. The R1 reversal is weak at step 250
(Δm_M ≈ −0.11) and grows during T4 training. So "P-count transfers a pretraining shortcut" is inaccurate: pretraining
supplies a salient count = 5 feature, plausibly without an easily read cross-group vote relation, and the shortcut is
learned on top of it. Mandate §12's "exists immediately after pretraining" is unanswerable here (no step-0 readout).

*Correction:* Say this explicitly, and say that the step-0 question is unanswerable under this design.

**5. [MODERATE] Point (c): alternative operators. The result is sharpened, not undercut.**
*Location:* FITTED_OPERATOR_IDENTIFIABILITY §1.
*Checks:*
- **(i) Gain plus a regime-insensitive 5/5 discount** (−δ on each candidate voted by a 5/5 group): δ has the *wrong* sign
  in 8/8 P-count models, ΔR² ≤ 0.06, predicted B45-R1 +0.11 to +0.41 vs −0.51 to −1.00 observed. Rejected.
- **(ii) Mixture plus δ.** Mean predicted B45-R1 is −0.28, against −0.74 observed.
- **(iii) Mixture fitted mainly to edit responses** (three starts): predicted B45-R1 −0.39 (half), R3 excess over-predicted
  (0.30–1.01 vs −0.32 to 0.90).

The mixture family (which contains the factorised "no-explaining-away" operator as φ_col = const) cannot combine a deep R1
discount with near-Bayes R3; the P-count models do. So "only half" is not just whole-surface weighting, as FITTED §1
suggests. A term specific to two perfect groups voting together would fit trivially but only renames the regime.

*Correction:* Add equivalent checks; restate FITTED §1 as "not reproduced even when fitted to responses; remaining
candidates are regime-indexed descriptions".

**6. [MODERATE] The KL-matched control is advertised beyond what it can show.**
*Location:* the MAIN_RESULT bullet "unchanged at matched KL", and P4.
*Problem:* An error of 0.74 on about 3 % of decisions adds roughly 10⁻³ to KL, so KL matching cannot detect R1 behaviour
(TRAINING §4 concedes this). Listing it as a control the result "survives" implies more than it shows.
*Correction:* Add "average-quality matching is insensitive to R1 by construction (Prop. 1)" next to the bullet.

**7. [MINOR] Several "theorems" are standard results.**
*Location:* THEOREMS T1(a,b), Lemmas B1–B3, T2, T5(a,b).
*Problem:* These are elementary mixture identities, a gain identity, and generic facts (behaviour does not identify
mechanism; redundancy defeats single ablations), which mandate §10 says not to present as new. T4(b) is useful but
elementary.
*Correction:* Label them "standard/elementary".

**8. [MINOR] Wording.** *Location:* MAIN_RESULT "Answer". *Problem/correction:* the finding *is* approximation error —
structured and curriculum-dependent, not "ordinary" uniform error. Say so.

## Points (b), (d), (e)

**(b)** Yes, R1 is low-measure (2.9 % of decisions, 32 % of oriented F45 pairs), and the growth argument does not exclude a
limited approximator (finding 3). What separates the arms is that **T1 features, under the same layer-2 learner, objective
and budget, avoid the error**: a curriculum-dependent frozen-transfer limit, not a generic capacity effect.

**(d)** Outcome A vs B, "not a distinct algorithm" and curriculum vs objective are correctly scoped. Missing: the frozen-feature
scope (1) and the word "interpreted" (2).

**(e)** New relative to Investigation V: a **prospective**, parameter-free, sign-level difference with complete 8/8
separation in independent parents; the exact-Bayes R1/R2/R3 decomposition locating it at an explaining-away subset; the
P-count long-transfer arm; the D-R1 contrast and its opposite-sign gain prediction; the Proposition 3 correction; partial
P6 localisation. Not new: the s ordering (replicated), and the general lesson that Bayes-closeness hides structured errors.

## Overall verdict

**The positive behavioural result stands and is not implied by exact Bayes, global gain, or any of the fitted pooling
families tested (including two adversarial ones added here).** It *is* well described by a known phenomenon: shortcut (worst-group) error on a 3 %-measure explaining-away subset under
frozen-feature transfer, with the curriculum deciding whether the frozen features let it be avoided.

That sharpens Outcome A rather than undercutting it, but requires rescoping to frozen-feature transfer, replacing
"interpreted" with "decision response", and replacing "contradicts an approximation-error account" with "contradicts uniform
approximation error; consistent with structured worst-group error".

Required: findings 1–3. Recommended: 4–6. Editorial: 7–8.
