# Pre-run review A: causal identification, COR Investigation XX (design draft v1)

**Reviewer:** an AI agent reviewer (Claude, Anthropic) acting as the mandate §2 causal-identification reviewer. I am not a human peer reviewer. I did not build this package.
**Date:** 2026-11-20 (UTC).
**Scope:** Track A (every arm and hypothesis), Track B (B1–B3 logic, HB_a–HB_e and the classification), and the eligibility, evaluability and verdict rules. Track C is out of scope except where it touches A/B.
**Status of this review:** this happens before any confirmation compute. Mandate §2 gives this review authority to reject tests. Mandate §2 also requires a re-review of the corrected design.

## What I read and what I checked

**Files read:**
- `protocols/DESIGN_XX.md`
- `cor_research/protocol.py`
- `cor_research/jobs/plan.py`
- `cor_research/scoring/{track_a,track_b,functional,trackb_run}.py`
- `cor_research/analysis/hier.py`
- `cor_research/interventions/graph_audit.py`
- `cor_research/generators/t4.py` (relational pairs and the discriminating set)
- `cor_research/reporting/report.py` (the denominator table)
- `cor_research/jobs/budget.py`
- `legacy/.../scripts/harness_xx.py`
- `legacy/.../scripts/compose_xx.py`
- `dev/DEV_NOTES_XX.md`
- `../MANDATE_XX.txt` §§0–4 and 8–10
- XIX records in `../../step1_training_state_causal_identification/`:
  - `MAIN_RESEARCH_SYNTHESIS.md`
  - `FROZEN_CONFIRMATION_AND_STATISTICS.md`
  - `ROUTING_GRADIENT_CAUSAL_TESTS.md`
  - `NON_TARGETED_OPTIMIZATION_CONTROLS.md`
  - `XVIII_FAILURES_AND_UNRESOLVED_CAUSES.md`
  - `COR_FOUNDATIONAL_RESEARCH_MAP_XIX.md`
  - `reviews/PRE_REVIEW_C.md`
- `../../step1_reroll_and_critical_period_validation/REROLL_VS_STRUCTURE.md`

**Checks I ran (CPU, short; no training; no package file modified):**
- **C1. rsg gradient check.** I ran the harness's own `lossf` with `FX_INTERV=rsg:100` against plain training. Parent uP160, step-0 state, 64 episodes of stream 97 900 000. Script: `/tmp/rv_check/rsg_check.py`.
  - **M base (T = own2, layer 2):**
    - `new/0/WQ` gradient norm is **0.000** under rsg, against 0.212 in plain training. `new/0/WK` is **0.000** against 0.229.
    - `new/0/WV` is unchanged (rel. diff 0.000).
    - `pre/0/*` and `pre/1/*` Q/K/V gradients change by 12–20%. `emb/*` changes by 14–60%.
  - **E base (T = own0, layer 0):**
    - `pre/0/WQ` and `pre/0/WK` gradients change by 44% and 32%. They are not zero.
    - `pre/0/WV` is unchanged.
    - `emb/*` changes by 7–18%.
- **C2. Verdict-rule simulation.** I simulated the frozen pair → parent → cross-parent rules (`track_a.py:84-101`) at the pair level. Script: `/tmp/rv_check/oc.py`. Numbers are under objection B1.
- **C3. phi sensitivity.** I evaluated `track_a.phi` on hand-chosen records that use the dev anchor values (E801). Script: `/tmp/rv_check/phi.py`. Numbers are under objection M5.

---

## Summary verdict: **APPROVE WITH REQUIRED CHANGES. The compute gate stays closed until the five BLOCKERs are fixed and the corrected design is re-reviewed.**

The package's base is sound and much better than XVIII and XIX:
- the parent list is preregistered and every screening seed always runs;
- eligibility is decided from baselines only, before any intervention, and is written once;
- evaluability does not depend on a band (the XVIII lesson);
- provenance checks fail closed;
- anchors are re-entered through the composer and checked bitwise;
- the parent is the statistical unit.

Five problems would still produce verdicts that the experiment cannot support:

1. **The verdict rule (B1).** It reports FAILS_TO_GENERALIZE for real effects with no contradicting pair. It also gets *stricter* as replication grows.
2. **HA8M (B2).** It is presented as a new discriminating routing-gradient test. On M, rsg reduces exactly to XIX's already-confirmed frzp-L2_QK nil arm plus an upstream perturbation, and its predicted outcome does not tell the competing accounts apart.
3. **The Track B classification (B3).** "Same rule" is decided by failing to reject a difference, not by an equivalence test. HB_d is largely implied by how the classes are defined. B2 is therefore close to preordained.
4. **The HB_c construction (B4).** Shared data order biases it toward "no functional difference".
5. **The HB_d interchange (B5).** Its ideal reference is wrong for patch layers ≥ 1, because the donor run contaminates the patch.

---

## BLOCKERS

### B1. The parent and cross-parent verdict rules are non-monotone and report FAILS without contradictions. Severity: BLOCKER.

**Where:** `protocol.py:61-69`, `track_a.py:84-101`.

**Problem:**
- PARENT_SUPPORTS requires **every** evaluable pair to be SUPPORTED_PAIR. One MIDDLE pair is enough to make a parent PARTIAL.
- FAILS_TO_GENERALIZE fires when "a strict majority do not support". PARTIAL parents count as not supporting. So FAILS can be reached with **zero** CONTRA pairs anywhere.
- The "all but one" exception is fixed and does not scale with the number of parents.

Simulated operating characteristics (C2). Each pair is SUPPORTED with probability pS, CONTRA with probability 0.02, and MIDDLE otherwise. "Evaluable" is the probability that a pair's anchors are evaluable. Each cell gives P(GENERALIZES) / P(FAILS) / P(INCONCLUSIVE).

| parents (evaluable unit pool) | n_D | evaluable | pS = 0.95 | pS = 0.90 | pS = 0.80 |
|---|---|---|---|---|---|
| 5 | 4 | 1.0 | 0.76 / 0.05 / 0 | 0.45 / **0.22** / 0 | 0.09 / **0.67** / 0 |
| 5 | 6 (Extended) | 1.0 | 0.60 / 0.12 / 0 | 0.23 / **0.44** / 0 | 0.02 / **0.88** / 0 |
| 8 | 4 | 1.0 | 0.55 / 0.01 / 0 | 0.17 / 0.10 / 0 | 0.01 / **0.57** / 0 |
| 8 | 6 | 1.0 | 0.34 / 0.03 / 0 | 0.05 / **0.29** / 0 | 0.00 / **0.87** / 0 |
| 5 | 4 | 0.6 | 0.14 / 0.00 / **0.84** | 0.10 / 0.01 / **0.85** | |
| 8 | 4 | 0.8 | 0.71 / 0.01 / 0.01 | 0.36 / 0.11 / 0.01 | |

Consequences:
- A real, mostly reproducible effect (90% of pairs inside the band, 2% contradicting) is labelled FAILS_TO_GENERALIZE in 22–44% of campaigns.
- Moving from Standard to Extended (more data orders) or adding parents lowers the probability of GENERALIZES for the same true effect.
- Losing pairs can *raise* that probability: with 8 parents, n_D = 4 and pS = 0.95, P(GENERALIZES) is 0.55 when every pair is evaluable and 0.71 when only 80% are.

This fails design-gate Q7. A FAILS verdict would not weaken the hypothesis in the sense the label claims. It is also a perverse incentive for the post-run reading of censoring.

**Required change, frozen before screening:**
1. **Parent level:**
   - PARENT_SUPPORTS iff SUPPORTED ≥ ⌈0.75·n_ev⌉ and CONTRA ≤ 1.
   - PARENT_CONTRADICTS iff CONTRA > n_ev/2.
   - Otherwise PARENT_PARTIAL.
   - Equivalent rules are acceptable if they pass point 3.
2. **Cross-parent level:**
   - FAILS_TO_GENERALIZE must be driven by contradictions: PARENT_CONTRADICTS ≥ 2, or ≥ 1/3 of evaluable parents.
   - "Strict majority not supporting" without contradictions becomes **NOT_ESTABLISHED**, distinct from FAILS.
   - The GENERALIZES exception count should scale, for example SUPPORTS ≥ max(n_ev − 1, ⌈0.8·n_ev⌉) and CONTRADICTS = 0.
3. **Preregistered operating-characteristic check:** commit a simulation (like `/tmp/rv_check/oc.py`) to `tests/` with pass criteria for every preset's n_D and for 4–12 parents:
   - P(GENERALIZES | pS = 0.9, pC = 0.02, evaluable ≥ 0.8) ≥ 0.8;
   - P(FAILS | that same true effect) ≤ 0.05;
   - P(GENERALIZES | true null, pS = 0.1, pC = 0.7) ≤ 0.01;
   - P(GENERALIZES) does not decrease as n_D grows.
4. **Report** the hierarchical phi mean and CI next to every count verdict. Already computed: `track_a.py:138`.

### B2. HA8M (rsg on M) is not a new discriminating test. Its outcome is largely established, and the outcome it predicts does not discriminate. Severity: BLOCKER for HA8M as frozen.

**Where:** `protocol.py:33, 51, 84-86`, `harness_xx.py:131, 156`, `DESIGN_XX.md:50-51, 61`.

**Q1 (guaranteed by the intervention?).** Layer 2 is the last block. The loss reads only rows 36..41 after it (`harness_xx.py:186-189`). So a stop-gradient on the readout-row logits of layer 2 removes **every** gradient to layer-2 WQ and WK. C1 confirms it numerically: 0.000 against 0.21–0.23.
- On M, rsg is therefore "layer-2 QK held at its fresh init for 100 steps" (AdamW receives zero gradients, so updates are zero apart from weight decay) **plus** removal of the Q/K-path gradient into L0, L1 and the embeddings (12–60% changes).
- It is not an edge-local or target-route routing-gradient intervention.
- The graph audit passes trivially: `graph_audit.py` asserts dL/dbias_T = dL/dbias_complement = 0, which holds for any whole-row stop. It does not report that L2_QK receives zero gradient.

**Q2 (already established?).** Yes, in the project record. XIX's frozen confirmation `M_P5_frzp_nil` (target-layer QK values held, moments update) was SUPPORTED at −0.03 / 0.06 / 0.05 (`FROZEN_CONFIRMATION_AND_STATISTICS.md:74`, `ROUTING_GRADIENT_CAUSAL_TESTS.md:28`): "freezing target-layer routing learning does nothing on M". HA8M's predicted B-like result is that result again, with a perturbation of the upstream gradient added.

**Q3 (do competing accounts predict different outcomes?).** Only partly. rsg also removes the routing gradient on **all complement edges** of the readout rows. Consider an account in which the block works by letting *competitor* routes gain routing signal: when T is blocked, attention mass and gradient move onto the complement. That account also predicts B-like under rsg. So a B-like result does not separate "value learning" from "competitor-routing learning". An I-like result could not be attributed to T's routing gradient either.

**Q6 and Q7.** A positive (B-like) result does not establish the value-learning account over the competitor-routing account. A negative result (I-like) cannot be attributed to T.

**XIX already named the discriminating tests** (`MAIN_RESEARCH_SYNTHESIS.md:265-267`):
- a *projected* routing-gradient stop: only the component of the WQ/WK gradient that comes from T's logits, removed on both softmax sides;
- *a step-100 transplant of only the routing state learned under vsgo on M*.

DESIGN §0 (lines 11-12) lists "a routing-state transplant on M", but **no such arm exists** in `ARMS` (`protocol.py:39-53`).

**Decision:** HA8M as frozen is **REJECTED** as a "new, discriminating" hypothesis.

**Required change, either:**
- (a) relabel HA8M as a replication of XIX `M_P5_frzp_nil` that adds an upstream-gradient removal, with no discriminating claim, and keep it only if the budget allows; **and/or**
- (b) replace it with one or both of:
  - **`BfromVSGO_QK` on M:** compose B's window state with the QK groups (L0_QK, L1_QK, L2_QK) from the matched vsgo window state. `compose_xx.py` already supports this: the vsgo window shares the data order, so the RNG assert holds. Add the complement `BfromVSGO_OV` as a control.
    - Prediction under "routing change mediates the vsgo effect": I-like.
    - Prediction under "value or OV state mediates": B-like.
  - **A T-projected routing stop (`prsg`):** zero the shared T-versus-complement share direction of the readout-row logit gradient, and keep within-T and within-complement relative gradients.
    - Add a graph-audit check that distinguishes it from rsg: dL/dbias_T = dL/dbias_C = 0 **and** a nonzero within-complement relative gradient.
    - Also report, per tensor, which tensors receive exactly zero gradient under every mode.

### B3. The Track B classification makes B2 almost preordained: "same rule" is accepted when a difference is not rejected, and HB_d is largely implied by the class definition. Severity: BLOCKER.

**Where:** `protocol.py:117-123`, `track_b.py:83-89`.

1. **No equivalence test.** "No functional difference" means `not (HB_a or HB_c FUNCTIONAL_DIFF)`. That is true whenever the test is underpowered, for example with 4 parents and wide CIs. B2 and B3 then assert "same rule" from absence of evidence. That fails Q6, because a positive B2 does not establish sameness.
2. **HB_d is close to guaranteed (Q1).**
   - The classes are *defined* by which layer's own-route is necessary: EARLY needs own0 at layer 0, MIXTURE needs own2 at layer 2 (`protocol.py:19-20`).
   - OQI_2 measures whether cross-question evidence is still read from other-question positions at the layer-2 input. That is a measurement of where information flows, which the class label already encodes. The dev difference (0.44–0.49 against 0.20–0.24) is what the class definition leads one to expect.
   - Mandate §4 says: "Do not spend GPU hours merely confirming that two models have different attention patterns" and "differences in attention-head necessity alone must not be mistaken for algorithm differences".
3. **The combined effect.**
   - B1 needs HB_d, which is near-guaranteed, plus HB_a *or* HB_c. HB_a is a rule-mixture **fit**, and mandate B1 says a good fit is not proof of a neural operation. So B1 can be reached from a fitted weight difference plus a route-locus difference. That is exactly the "formula fit plus attention route" inference the mandate forbids.
   - B2 is reached by underpower plus the locus difference.
   - B3 is reachable only if HB_d fails, which is unlikely by construction.
   - Track B therefore cannot discriminate its three outcomes as designed (Q3 and Q4).

**Required change:**
- **"Same rule" (B2 and B3) requires equivalence:**
  - the HB_c parent-level 95% upper bound is < delta_dis;
  - for every rule k, the HB_a parent-level CI lies inside ±delta_w (TOST);
  - otherwise the result is **NON_IDENTIFIABLE (underpowered)**.
  - Freeze the margins now and report the achieved power for them.
- **B1 requires HB_c** (decision-level, calibration-invariant, on-support). HB_a may only corroborate.
- **HB_d may not serve as B1's internal criterion** (see M11). Either add an internal test of *what* is computed, or state in DESIGN that B1 is unreachable by design and that Track B can only separate {B2, B3, NON_IDENTIFIABLE}.
- Re-derive the frozen Track B prediction under the corrected rule.

### B4. HB_c is biased toward "no functional difference": between-class pairs always share the data order and within-class pairs never do. Severity: BLOCKER.

**Where:** `track_b.py:74-75`.
- `between` collects pairs with different class and the **same** D.
- `within` collects pairs with the same class and a **different** D.

Shared data order means shared batches. That lowers decision disagreement in the between-class term only, so the between-minus-within statistic is biased downward. The bias favours B2 and B3, which are the frozen prediction.

The comparison sets are also heterogeneous:
- `between` mixes same-seed pairs (E-fullB with E-fullI) and cross-base, different-seed pairs (E-fullB with M-fullB);
- `within` mixes same-seed and cross-base pairs differently.

**Required change:**
- **Primary:** use D ≠ D′ for **both** terms. For example, between = E-fullB(D1) with E-fullI(D2); within = E-fullB(D1) with E-fullB(D2). Both are available in every parent with one eligible base. Also match the seed structure: same unit seed in both terms.
- **Secondary:** a same-D 2×2 contrast in parents with both bases eligible. Between = E-fullB(D) with M-fullB(D), and M-fullI(D) with E-fullI(D). Within = E-fullB(D) with M-fullI(D), and E-fullI(D) with M-fullB(D). All of these are cross-seed pairs, so the comparison is balanced.
- Add a planted test with a pure shared-D effect and no class effect. It must return no FUNCTIONAL_DIFF and a zero mean.

### B5. The HB_d interchange uses the donor run's residuals, so its ideal reference Bayes(x_h) is wrong for patch layers l ≥ 1. Severity: BLOCKER, because HB_d decides B1, B2 and B3 and mandate §10 requires correct ideal references.

**Where:** `functional.py:99-104`, `t4.py:49-59`.
- OQI_l patches the other-question positions with `rd[key]`, the residual of the run on the **donor** x′ (`functional.py:103`). RS_l uses `rh`, the hybrid run (`functional.py:104`).
- At l ≥ 1, the donor's other-question residuals have already mixed in information from **the donor's own question q** through attention in layers < l. Cross-question relational evidence needs exactly that mixing.
- The hybrid x_h contains the base's q-reports, not the donor's. So the patched state is not the state the reference Bayes(x_h) describes.
- How much donor-q information leaks depends on the organisation: EARLY routes at layer 0. An EARLY-versus-MIXTURE difference in OQI_2 can therefore come from the contamination itself.

The test also lacks the source-matched control the mandate asks for (B4: "source-matched and ideal-reference controls"). And IIA is not normalised for the model's own competence on the hybrid input (`model_follow_on_hybrid_input`).

**Required change:**
- The OQI_l patch source must be the run on **x_h** (`rh`) at the other-question positions. Keep the `rd` version only as a labelled secondary.
- Add a source-matched null: patch from a run on x itself. It must reproduce the base decision, with IIA-on-base = 1 up to numerical error, as a planted check.
- Report IIA_l normalised by OQI_0, or restrict it to pairs where the unpatched model on x_h follows Bayes(x_h).
- Re-run the dev check (n = 2 + 2) with the corrected source before freezing the HB_d prediction.

---

## MAJOR

### M1. Every mediation hypothesis is conditional on the switch (HA1) happening in the same pair. This is post-treatment selection that shares the data order. Severity: MAJOR.

**Where:** `track_a.py:68`.
- A pair is ANCHORS_NOT_EVALUABLE for HA2–HA8 unless the fullI anchor is competent and of the opposite class at the scoring step.
- That is necessary to define phi. But it makes HA2–HA8 *conditional* estimands, and parents where the block does not switch only drop to INCONCLUSIVE; they never count against mediation.

Because arms and the I anchor share D:
- the selection favours data orders on which switching happens;
- that is anti-conservative for the I-like predictions (HA2, HA4a, HA5a/b, HA7);
- it is conservative for the B-like ones.

**Required change:**
1. Label every mediation verdict "conditional on the switch". A mediation hypothesis may be called GENERALIZES only if HA1 GENERALIZES on the same base.
2. For every hypothesis, report the full denominator chain: preregistered → valid → eligible unit → parents with ≥ 3 switch-evaluable pairs → evaluated.
3. Add a preregistered unconditional sensitivity analysis: over every pair whose B anchor is native, report the arm's class and its concordance with the fullI class of the same pair. Include pairs that did not switch.

### M2. Eligibility uses one screening data order, while confirmation uses fresh ones, so baseline class can regress toward the parent's base rate. Severity: MAJOR.

**Where:** `protocol.py:17-20`, `plan.py:80-83`.
- A unit qualifies if *some* of 8 seeds gives the native class on D901, and the lowest such seed is used.
- XVIII/XIX evidence: uP161's EARLY anchor was a rare draw. On re-draws P(early) was 0/32 (`REROLL_VS_STRUCTURE.md:82-101`).
- If a rare-draw seed qualifies on D901, its B anchor on D2001… will often be non-native. That gives ANCHORS_NOT_EVALUABLE, and with n_D = 4 and MIN_PAIRS = 3, PARENT_INCONCLUSIVE. In the simulation, 60% evaluability gives INCONCLUSIVE 84% of the time (B1 table).
- The losses are counted, which is good, but they could quietly empty the E base (Q8).

**Required change:**
- Screen each qualifying seed on a second, fresh data order (for example D902; baseline only, cheap).
- Make the unit seed the lowest seed native on **both** orders. Freeze this before screening.
- Report, per unit, the observed rate at which B anchors are native across confirmation pairs.

### M3. The historical cohort uP161–165 is not blind. Severity: MAJOR.

What earlier investigations already observed on these parents:
- **XVIII:** screened them (1 of 5 bistable: uP163) and ran intervention arms on uP163 with known outcomes: vrg 1 F / 1 P, sha 2/2, gsv "M F, E P" (`XVIII_FAILURES_AND_UNRESOLVED_CAUSES.md:32, 42-46`).
- **The reroll study:** characterised uP161.
- **XIX review C:** set the competence threshold 0.06 after inspecting the KL of uP162/164/165 models (`PRE_REVIEW_C.md:442-461`).
- **XX's validity threshold 0.95:** chosen knowing uP162 = 0.779.

This does not select on the *new* arms' outcomes. But uP163 is not an independent confirmation of HA1, and the eligibility thresholds were informed by these parents.

**Required change:**
- Tag uP161–165 as `previously_observed` in `protocol.py`.
- Report every Track A verdict twice: on all parents, and on **fresh parents only**.
- A headline GENERALIZES requires that the fresh-only verdict is not FAILS/CONTRADICTS. If fewer than 4 fresh parents are evaluable, label the headline "supported only with previously observed parents".
- Consider moving fresh parents first in the Standard list.

### M4. "Parent" is confounded with one selected fresh-layer and readout initialisation. Severity: MAJOR.

**Where:** `plan.py:83, 112-119`.
- Each (parent, base) unit fixes L = W = one seed, so data order is the only replicate within a unit.
- XVIII/XIX found that L and W carry outcome variance. Between-"parent" variance therefore includes between-seed variance, and E and M units of the same parent differ in seed.
- The claim "generalises across independent parents" really means "across independent (parent, selected-init) units".

**Required change, either:**
- run a second qualifying seed for units where one exists, to estimate within-parent seed variance; **or**
- restate the claim and rename `var_comp` "between-unit".

Either way, document it in DESIGN §1.

### M5. phi with the log(x + 0.01) floor is driven by noise in near-zero necessities, and phi and the class condition can disagree. Severity: MAJOR.

**Where:** `track_a.py:26-35, 73-80`.

Using the dev E801 anchors (C3):
- **EARLY-class arm** (nec_E 1.0, nec_M 0.10): phi = 0.267, so MIDDLE rather than B-like.
- **MIXTURE-class arm** (nec_E 0.05, nec_M 0.30): phi = 0.698, so MIDDLE rather than I-like.
- **Anchor sensitivity:** changing the B anchor's nec_M from 0 to 0.05 moves the same arm's phi from 0.27 to 0.10.

Necessity is measured on 1500 episodes, and its sampling noise near zero is not reported. Together with B1, this turns noise into FAILS verdicts.

**Required change:**
- Record a bootstrap SE for nec_E and nec_M in `measure.py`.
- Preregister a floor sensitivity analysis (0.01 and 0.03).
- Report phi_E and phi_M separately.
- Either make the class condition primary, with phi used only to grade MIDDLE, or set the floor to at least 2 × the measured SE.

### M6. The validity check for the non-targeted control (lbl) was dropped, and the norm control does not match the E arm's groups. Severity: MAJOR.

**Where:** `protocol.py:34, 49`; XIX `score19.py:178-181`.
- **lbl:** XIX required the lbl control's step-100 loss excess (KL to Bayes) to be ≥ 0.5 × the block's, and on E it was ≈ 1.0 (`NON_TARGETED_OPTIMIZATION_CONTROLS.md:13`). XX measures no step-100 window state and has no such check, so a B-like HA6b could come from an under-strength disturbance (Q5 and Q7).
- **randnorm:** `randnorm_L0L2` is used on E, but E's sufficiency arm is L0 only (HA4a).

**Required change:**
- Measure the B, I, lbl, vsgo and rsg (or replacement) window states at step 100 on a dedicated stream.
- Preregister CONTROL_INVALID when the lbl excess ratio is < 0.5. HA6b gets a verdict only when the control is valid.
- Add `randnorm_L0` for E (the norm group = the group of the arm it controls), or define randnorm per base.

### M7. There are no state-compatibility or transient-disruption measurements for synthetic hybrid states (mandate A4). Severity: MAJOR.

**Where:** `plan.py:138`. Measurements exist only at steps 1211 and 4000.
- A reverse transplant (IfromB_L0, IfromB_L0L2, and the High QK/OV splits) can come out B-like because the composed state is *incompatible* (a loss shock followed by re-learning along B's default path). That is not necessity of the mediator.
- Mandate A4 requires recording whether a transplant "changes the immediate network … produces only a temporary disruption".
- The continuation runs already save snapshots at 150, 165, 200, 246, 300 and 366, so only measurement jobs are needed.

**Required change:**
- Measure every composed state at step 100 (KL, accuracy, necessity), and every continuation at steps 150 and 300.
- Preregister an INCOMPATIBLE flag: the composed state's KL excess over the matched anchor exceeds a frozen threshold, or competence has not recovered by step 300.
- A B-like result from a flagged reverse transplant is reported as "not evidence of necessity".

### M8. Track B statistics with 4–8 parents give false precision, and HB_a has a multiplicity problem. Severity: MAJOR.

**Where:** `track_b.py:59-60, 63-66`, `hier.py:15-22`.
- `diff()` gives each parent a one-element list. When all parent means share a sign, every bootstrap replicate shares it, so "CI excludes 0" is automatic whatever the magnitude or spread.
- With 4 parents, the smallest possible exact two-sided sign-flip p is 0.125. No valid 5% test exists.
- HB_a declares FUNCTIONAL_DIFF if **any** of 4 rules passes, with no correction.

**Required change:**
- Report exact parent-level sign-flip or permutation p-values.
- Apply Holm correction across the 4 rules.
- Label any Track B classification based on fewer than 6 parents with both classes as "indicative".
- Apply the same treatment to Track A's reported CIs.

### M9. Matched precision confounds class with training time, and censoring can differ by class. Severity: MAJOR.

**Where:** `protocol.py:102-103`, `trackb_run.py:25-32`.
- **Training time:** in dev, matched snapshots were MIXTURE at step 600 and EARLY at step 1211 (`DEV_NOTES_XX.md:22`). They therefore differ in learning-rate phase and training age.
- **Censoring:** EARLY models of other parents end at KL 0.039–0.050 (`PRE_REVIEW_C.md:226, 442-445`), above the band [0.024, 0.036]. Snapshots are also sparse (… 1211, 1500, 1803, 2000, 2686, 3000, 4000). EARLY units may be censored by class, which is selection on the very precision difference at issue.

**Required change:**
- Match per parent: target = a KL that both classes reach within the parent. Match models in pairs; do not use a global 0.030.
- Report the matched-step difference and LR for every comparison, and a sensitivity analysis with the step as a covariate.
- Report censoring by class × parent. If class-specific censoring rates differ by more than 20 points, the matched level is NON_IDENTIFIABLE.
- Require the matched and the final (4000) levels to agree before any classification is headlined.

### M10. In Track B, class is confounded with intervention history. Severity: MAJOR.

- EARLY appears natively in E-fullB and is induced in M-fullI. MIXTURE appears natively in M-fullB and is induced in E-fullI.
- `byp` pools them (`track_b.py:55`), so a "class" difference may be a "was blocked for 100 steps" difference.

**Required change:**
- Estimate the class effect within the base × anchor 2×2.
- Require the same sign in the native-versus-native contrast (E-fullB against M-fullB) and the induced contrast (M-fullI against E-fullI).
- Report the history main effect separately.

### M11. HB_d is a locus test, not an internal test of the computation. Severity: MAJOR. This overlaps B3 and states the decision.

- HB_d is an internal *intervention*, so it is causal about the model. But the between-class comparison only shows where cross-question evidence is causally read.
- "Same statistic computed at a different layer" and "a different statistic" predict the same OQI_2 difference (Q3).
- The mandate's B4 asks whether an internal variable **transfers between organisations**. Nothing in the package does that.

**Decision:** HB_d is **REJECTED as the internal criterion for B1**. It is **ACCEPTED (after B5) as a descriptive locus profile** that supports B2's "different implementation" clause.

**Required change for B1 to be reachable:** add a cross-model interchange. Patch a candidate relational variable (for example, the cross-question reliability content at the readout position) from an EARLY model into a matched MIXTURE model, and the reverse. Score it against Bayes(x_h). Different algorithms predict asymmetric or failed transfer; the same statistic predicts symmetric transfer. Otherwise declare B1 unreachable by design (see B3).

### M12. The stability continuation is close to guaranteed to show stability. Severity: MAJOR, for the claim it could support.

**Where:** `protocol.py:26, 96`, `harness_xx.py:190-194`.
- The continuation warms to a peak learning rate of 1e-4, one-tenth of the original 1e-3, with moments carried over.
- Nothing shows that this continuation can move an organisation at all (Q1 and Q7).
- XA4 is labelled exploratory, which is right, but RESULTS must not call anchors "stable".

**Required change:**
- Add a positive control: the same continuation applied to a HYBRID arm state (for example HA8E's), or one anchor per parent continued at peak 1e-3.
- Report parameter displacement and loss change across the continuation.
- Restrict the wording to "class unchanged under a low-LR continuation" unless the positive control moves.

### M13. HA8E is a low-information prediction and is mis-specified. Severity: MAJOR.

**Where:** `protocol.py:87-89`, `track_a.py:77-78`.
- **Dev basis and horizon:** the prediction comes from n = 1 (uP160, D801), from a run trained from step 0, read at step 1211. It is frozen for step 4000. Hybrids may resolve between 1211 and 4000; XVIII's gsg was hybrid at step 2546 on M.
- **No class condition on "partial":** a MIXTURE-class pair (the I class for E) with phi 0.70 counts as partial support.
- **What "partial" predicts:** "both routes contribute" predicts partial, so SUPPORTED distinguishes little. Only a CONTRA outcome is informative. It also repeats XIX's frozen `E_P5_frzp_partial` (0.54–0.69).
- **What rsg on E actually does:** it removes the complement routing gradient of the layer-0 readout rows and changes the embedding gradient (C1). It is not the projected-T stop combined with vsgo that XIX named for E.

An n = 1 basis for a *frozen* prediction is acceptable only if it is labelled low-confidence, and it already is.

**Required change:**
- "partial" requires class HYBRID, or a class that is neither B's nor I's.
- Evaluate it also at step 1211, the dev horizon, as a co-primary step.
- State in DESIGN that SUPPORTED distinguishes little and a CONTRA result is the informative outcome.
- Preferably add the XIX-named E test: prsg (B2) combined with vsgo.

---

## MINOR

- **m1.** A parent with no `pre.json` is skipped in `eligibility()` (`plan.py:70-71`), so no record is written. `report.py:43` shows "NOT YET SCREENED / pretraining incomplete". After a completed run, a parent whose pretraining job failed must appear as `FAILED_PRETRAINING` in `eligibility.json` and in each verdict's denominator.
- **m2.** HA5c (`not_I_like`) is supported by almost any outcome (`track_a.py:79-80`). It is a weak negative claim. Report its phi distribution and do not headline it.
- **m3.** HA4a is close to tautological (DESIGN says so). RESULTS must list it as a replication anchor, not a finding.
- **m4.** NONCOMPETENT arm pairs are dropped from a parent's evaluable count when they are at most half (`track_a.py:85-88`). Under A4 a transplant that breaks competence is informative. Report the competence-loss rate per arm as an outcome; `stats.competence_rate` exists, so cross-reference it in the hypothesis table.
- **m5.** HA9 (persistence) does not require competence at step 4000 (`track_a.py:121-127`). Add it, or report it.
- **m6.** The High-preset extra arms (IfromB_L0_QK, IfromB_L0_OV, half_L0L2) have **no frozen hypothesis**. Declare them exploratory in `EXPLORATORY_A` or freeze predictions for them.
- **m7.** `var_comp` with 4–8 parents is unstable. Report it as descriptive only.
- **m8.** HB_b's noise null is i.i.d. Gaussian logit noise, which is not a structured-error null. It is fine as descriptive output. It must never feed the classification; currently it does not.
- **m9.** The graph audit should report, per tensor, which tensors receive exactly zero gradient under each mode. That would have exposed B2 automatically.

---

## Design-gate answers per test (mandate §2, Q1–Q8), from my reading of the code

Abbreviations: Y = yes; N = no; P = partly; "cond." = conditional on the switch (M1).

**Track A, retained XIX tests**

| test | Q1 guaranteed? | Q2 established? | Q3 explanations differ? | Q4 metric distinguishes? | Q5 controls in code? | Q6 positive establishes? | Q7 negative weakens? | Q8 inconclusive without silent exclusion? |
|---|---|---|---|---|---|---|---|---|
| HA1 switch (fullI) | N | uP160 only (XIX); uP163 partly seen (XVIII) | Y | Y (class) | Y (B anchor) | Y, parent-level | N as frozen (B1) | P (M2) |
| HA2 paramI_optB | N | uP160 | Y (params vs moments) | P (M5) | Y | Y, cond. | N (B1) | P (M1) |
| HA3 paramB_optI | N | uP160 | Y | P | Y | Y, cond. | N (B1) | P |
| HA4a BfromI_L0 (E) | ≈Y (near-tautological) | uP160 | P | P | P (norm control mismatched, M6) | replication only | N (B1) | P |
| HA4b IfromB_L0 (E) | N | uP160 | P (incompatibility, M7) | P | N (no compatibility check) | P | N (B1) | P |
| HA5a / HA5b BfromI_L0L1 / L0L2 (M) | N | uP160 | Y | P | Y | Y, cond. | N (B1) | P |
| HA5c BfromI_L0 not I-like (M) | N | uP160 | P | weak | Y | weak | weak | P |
| HA5d IfromB_L0L2 (M) | N | uP160 | P (M7) | P | N (M7) | P | N (B1) | P |
| HA6a randnorm_L0L2 | N | uP160 | Y | P | P (group mismatch on E) | Y (nil) | N (B1) | P |
| HA6b lbl | N | uP160 | Y | P | **N** (validity check dropped, M6) | only if valid | N | P |
| HA7 vsgo (M) | N | uP160 (frozen) | Y | P | Y | replication | N (B1) | P |

**Track A, new and remaining tests**

| test | Q1 guaranteed? | Q2 established? | Q3 explanations differ? | Q4 metric distinguishes? | Q5 controls in code? | Q6 positive establishes? | Q7 negative weakens? | Q8 inconclusive without silent exclusion? |
|---|---|---|---|---|---|---|---|---|
| **HA8M rsg (M)** | **Y in effect: L2_QK frozen** (C1) | **Y (XIX M_P5_frzp_nil)** | **N** (competitor-routing account also predicts B-like) | P | audit passes trivially | **N** | **N** (not attributable to T) | P |
| HA8E rsg (E) | N | similar to XIX frzp-E partial | P | P (no class condition) | P | weak | Y (CONTRA is informative) | P |
| HA9 persistence 1211 → 4000 | N (LR still ≈ 0.84e-3 at step 1211) | N | Y | Y | P (no competence at 4000) | Y | Y | P |
| XA1 vsgo (E), exploratory | N | N | n/a | n/a | Y | descriptive | descriptive | Y |
| XA3 all hypotheses at 1211 | n/a | n/a | n/a | n/a | Y | secondary | secondary | Y |
| XA4 stability 4000 → 12000 | ≈Y (LR 1e-4, M12) | N | N without a positive control | N | N | N | Y | Y |
| High arms: QK/OV splits, half_L0L2 | N | N | Y | P | N (M7, m6) | P | P | P |
| Eligibility procedure | n/a | n/a | n/a | n/a | Y (written once) | n/a | n/a | P (M2, M3, m1) |
| Cross-parent verdict rule | n/a | n/a | n/a | **N** (B1) | n/a | Y | **N** | P |

**Track B**

| test | Q1 guaranteed? | Q2 established? | Q3 explanations differ? | Q4 metric distinguishes? | Q5 controls in code? | Q6 positive establishes? | Q7 negative weakens? | Q8 inconclusive without silent exclusion? |
|---|---|---|---|---|---|---|---|---|
| HB_a rule-mixture | N | fits non-identifiable (XVII) | P | P (multiplicity, M8) | P (F0 baseline) | **N** (a fit, mandate B1) | P | P (M9 censoring) |
| HB_b above the noise null | N | N | P | descriptive | P | descriptive | descriptive | Y |
| HB_c decision disagreement | N | N | Y | **N as coded** (B4 shared-D bias) | P | Y after B4 | **N** without equivalence (B3) | P |
| HB_d OQI_2 interchange | **≈Y given the class definition** | route difference known | **N** for B1 (locus only) | **N as coded** (B5 contamination) | **N** (no source-matched null, no normalisation) | **N** for B1; P for B2 | P | P |
| HB_e precision at 4000 | N | XIX "precision difference" | Y | Y | Y | Y | Y | Y |
| Matched-precision procedure | n/a | n/a | n/a | P | P | n/a | n/a | **P** (class-differential censoring, M9) |
| Classification B1 / B2 / B3 | n/a | n/a | n/a | **N** (B3) | n/a | **N** (B2 preordained) | N | P |

---

## Decision table per test

| test / component | decision | required change(s) |
|---|---|---|
| Parent list and pretraining validity | ACCEPT-WITH-CHANGE | M3 (tag historical parents, fresh-only verdicts); m1 |
| Screening and eligibility rule | ACCEPT-WITH-CHANGE | M2 (second screening data order); M4 (claim scope, or a second seed) |
| Pair evaluability (anchor-only) | ACCEPT-WITH-CHANGE | M1 (label as conditional, denominator chain, unconditional sensitivity) |
| phi and class pair outcome | ACCEPT-WITH-CHANGE | M5 |
| Parent and cross-parent verdict rules | **REJECT (as frozen)** | B1: rewrite the rules and preregister the operating-characteristic test |
| Hierarchical statistics (Track A) | ACCEPT-WITH-CHANGE | M8 (exact parent-level p, few-cluster caveat) |
| HA1 | ACCEPT-WITH-CHANGE | B1, M2, M3 |
| HA2, HA3 | ACCEPT-WITH-CHANGE | B1, M1, M5 |
| HA4a | ACCEPT-WITH-CHANGE | m3 (replication anchor); M6 (randnorm_L0 for E) |
| HA4b, HA5d | ACCEPT-WITH-CHANGE | M7 (compatibility flag), B1 |
| HA5a, HA5b | ACCEPT-WITH-CHANGE | B1, M1 |
| HA5c | ACCEPT | m2 (do not headline) |
| HA6a | ACCEPT-WITH-CHANGE | M6 (norm group per base) |
| HA6b | ACCEPT-WITH-CHANGE | M6 (restore the control-validity criterion; step-100 measurements) |
| HA7 | ACCEPT-WITH-CHANGE | B1 (labelled replication) |
| **HA8M (rsg on M)** | **REJECT** | B2: relabel as a frzp replication, and/or replace with BfromVSGO_QK (plus OV) and/or prsg |
| HA8E (rsg on E) | ACCEPT-WITH-CHANGE | M13 (class condition, 1211 co-primary, wording; preferably prsg + vsgo) |
| HA9 | ACCEPT-WITH-CHANGE | m5 |
| XA1, XA3 | ACCEPT | none |
| XA4 stability | ACCEPT-WITH-CHANGE | M12 (positive control, restricted wording) |
| High extra arms | ACCEPT-WITH-CHANGE | m6, M7 |
| Track B matched-precision procedure | ACCEPT-WITH-CHANGE | M9 |
| HB_a | ACCEPT-WITH-CHANGE | B3 (cannot alone give B1; TOST for sameness); M8 (Holm) |
| HB_b | ACCEPT | descriptive only (m8) |
| HB_c | ACCEPT-WITH-CHANGE | B4 (symmetric D and seed matching); B3 (equivalence) |
| HB_d as the internal criterion for B1 | **REJECT** | M11 (cross-model interchange, or declare B1 unreachable) |
| HB_d as a descriptive locus profile | ACCEPT-WITH-CHANGE | B5 (rh source, source-matched null, normalisation) |
| HB_e | ACCEPT | none |
| Track B classification rule | **REJECT (as frozen)** | B3, M10 |
| Frozen Track B prediction | ACCEPT-WITH-CHANGE | re-derive after B3–B5 |

---

## Answers to the specific questions in my brief

**Is the denominator honest?** Mostly.
- Every preregistered parent appears in the report table (`report.py:39-46`). Eligibility is written once. All 8 seeds always run.
- Gaps: a missing `pre.json` is not recorded in `eligibility.json` (m1), and hypothesis verdicts list only evaluable parents (M1).

**Is there selection on outcome?**
- Not at eligibility: it is baseline-only and pre-intervention.
- Yes at evaluability: HA2–HA8 are conditioned on the I anchor switching (M1).
- Eligibility on a single data order invites regression to the base rate (M2).

**Is using uP161–165 a problem?** It is not fatal, but they are not blind (M3). Report fresh-only verdicts. uP163's prior XVIII intervention outcomes make it a non-independent confirmation of HA1.

**Could most pairs be silently excluded?** They are not silent; every pair has a status. But they *can* be excluded en masse:
- by B-anchor regression (M2);
- by no-switch pairs (M1);
- in Track B, by class-differential censoring (M9).
Under the current rule, losing pairs can even raise P(GENERALIZES) (B1).

**Do phi against own-pair anchors and the class condition identify sufficiency and necessity?**
- They identify *sufficiency or necessity within the synthetic composed state*, conditional on the switch.
- They do not identify a natural mediator without compatibility checks (M7). Mandate A4 says this explicitly.
- phi is noise-sensitive near zero (M5).

**Does rsg isolate the routing gradient? Is HA8M/HA8E discriminating?**
- No. On M, rsg is exactly a layer-2 QK freeze plus an upstream perturbation, and it repeats XIX's frzp nil (B2).
- On E, it removes all readout-row routing gradient at layer 0, including the complement, and changes the embedding gradient.
- HA8M is not discriminating (REJECT). HA8E is weakly discriminating (M13).

**Is a prediction from one dev run acceptable?** As a labelled, low-confidence frozen prediction, yes. The problems are the horizon mismatch, the missing class condition, and the low information content of "partial" (M13). The n = 1 basis is not the issue.

**Cross-parent rule (thresholds, minimum 4 parents, "all but one")?**
- Rejected as frozen (B1).
- A minimum of 4 parents is acceptable as a floor. With 4 parents, no exact two-sided 5% parent-level test exists (M8), so 4-parent verdicts should be labelled indicative.

**Hierarchical statistics with the parent as the unit?**
- The structure is right: the bootstrap resamples parents and then pairs.
- With 4–8 clusters the percentile bootstrap is anti-conservative and can be degenerate (M8).

**Stability continuation?** Non-discriminating without a positive control (M12).

**Track B B1, B2 and B3 from a causal standpoint; is HB_d internal and causal or descriptive?**
- HB_d is an internal causal intervention *within* each model. As a between-class comparison it is a locus descriptor that is largely implied by the class definition (B3, M11). As coded, its ideal reference is also wrong for l ≥ 1 (B5).
- B2 is close to preordained (B3).
- HB_c is biased toward the null (B4).
- Class is confounded with intervention history (M10).

**Matched-precision snapshots taken at different training times?** This is a real confound (training age and LR phase), and so is class-differential censoring (M9).

---

*End of review A v1. A corrected design must be re-reviewed (mandate §2) before the protocol is frozen and confirmation compute starts.*
