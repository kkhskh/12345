# Pre-run review A, round 2: causal identification, COR Investigation XX (design v2)

**Reviewer:** an AI agent reviewer (Claude, Anthropic) acting as the mandate §2 causal-identification reviewer for the second review round. I am not a human peer reviewer. I did not build this package and I did not write the round-1 review. I read it, and I read the builder's adjudication.
**Date:** 2026-11-20 (UTC).
**Scope:**
- every round-1 causal-identification BLOCKER, MAJOR and MINOR, checked in the code and not only in the documents;
- the tests that are new or changed in v2;
- new problems that the corrections introduced.

Track C is out of scope.
**Authority:** mandate §2 lets this review reject tests.

## What I read and what I ran

**Read:**
- `protocols/DESIGN_XX.md` (v2)
- `reviews/PRE_RUN_REVIEW_A_causal_identification_v1.md`
- `reviews/ADJUDICATION_v1.md`
- `cor_research/protocol.py`
- `cor_research/jobs/plan.py`
- `cor_research/scoring/{track_a,track_b,functional,trackb_run,measure}.py`
- `cor_research/analysis/hier.py`
- `cor_research/interventions/graph_audit.py`
- `cor_research/reporting/report.py` (the parts that touch A/B verdicts)
- the group handling in `legacy/.../compose_xx.py`
- `tests/test_verdict_oc.py`, `tests/test_scorers_planted.py` and `tests/test_trackb_oc.py`, with their stored results

**Ran (CPU, short; no package file modified; all scripts and outputs are in `/tmp/revA2/`):**

| # | check | result |
|---|---|---|
| R1 | `python3 tests/test_verdict_oc.py` | PASS (exit 0). Numbers are under N3. |
| R2 | `python3 tests/test_scorers_planted.py` | 31/31 PASS |
| R3 | `python3 -m cor_research.interventions.graph_audit /tmp/revA2/audit.json` (uP160, step-0 state) | all checks pass. The gradient lists matter for N1 (below). |
| R4 | `oc2.py`, `oc3.py`, `oc4.py`: extra operating-characteristic scenarios through the real `track_a.parent_verdict` / `cross_parent`, plus two candidate rule variants | weak, unreliable and heterogeneous effects (N3, N4, n1) |
| R5 | `b3.py`: real `track_b.score` on synthetic units whose KL is matched at the matched level and differs at the final level | B3 and B2-LOCUS+B3 can never be the headline (N-B1) |
| R6 | `hist.py`, `hist2.py`: real `track_b.score` on synthetic units with an intervention-history effect and **no** class effect | false FUNCTIONAL_DIFFERENCE, and false B1-FUNCTIONAL with one base (N-B2) |

Details for R3:
- On M under vsgo, the layer-2 (`new/0`) WQ/WK gradients are **unchanged** against plain training.
- The layer-0 and layer-1 (`pre/0`, `pre/1`) WQ/WK gradients **are** changed.
- On M under rsg, the zero-gradient tensors are `new/0/WQ, new/0/WK`. They are disclosed, as v2 promised.

---

## Verdict: **APPROVE WITH REQUIRED CHANGES**

**The compute gate stays closed until:**
- the two new Track B BLOCKERs (N-B1, N-B2) are fixed;
- the HA10 changes (N1, N2) are made, because they add or change arms and must therefore be in the plan before the freeze;
- the planted and operating-characteristic tests listed below demonstrate each fix.

A third full review round is not needed. A targeted check of these specific changes (diff plus test output) is enough.

**What v2 got right:** the round-1 Track A corrections are, with few exceptions, implemented in the code and not only described.
- The new parent and cross-parent rules are monotone in n_D for strong effects. Under the null they never give GENERALIZES.
- Conditional labelling, fresh-only verdicts, two-order screening, the lbl validity rule, per-base randnorm, the HYBRID-class rule for HA8E and the zero-gradient disclosure are all present and correct.
- INCOMPATIBLE and CONTROL_INVALID cannot remove inconvenient pairs. INCOMPATIBLE only ever converts a SUPPORTED pair. CONTROL_INVALID is decided from step-100 window states, before the outcome exists.

**Problems that remain:**
1. **Track B classification (two new BLOCKERs):**
   - B3, and the frozen prediction B2-LOCUS+B3, can never be the headline.
   - Intervention history alone can produce FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED, and with one eligible base it can produce B1-FUNCTIONAL.
2. **HA10a/b, the replacement discriminating test, is mislabelled.** It discriminates *QK-weight state against non-QK-weight state*, not routing state against value state. vsgo changes the L0/L1 QK gradients directly and leaves the target-layer L2 QK gradient unchanged at the start. HA10 also lacks a manipulation check and conditioning on the vsgo effect.
3. **The verdict operating-characteristic test does not enforce the criterion its own docstring states.** At the Standard preset's minimum (4 parents, n_D = 4, 80% evaluability), P(GENERALIZES | strong effect) = **0.42**, with INCONCLUSIVE at 0.56. DESIGN §1 claims "≥ 0.97 at 4–12 parents".

---

## Status of every round-1 item (verified in code)

| id | round-1 issue | v2 status | evidence in code; what remains |
|---|---|---|---|
| **B1** | Verdict rule non-monotone; FAILS without contradictions | **Partially resolved** | `track_a.parent_verdict:90-98` and `cross_parent:101-107` match `protocol.PARENT_RULE/CROSS_RULE`. FAILS now needs CONTRADICTS ≥ max(2, ⌈n/3⌉). NOT_ESTABLISHED exists. P(GEN \| null) = 0. Monotone 4 → 6 D for strong effects (R1). **But:** the OC test enforces the "P(GEN \| pS 0.9, pE 0.8) ≥ 0.8" criterion only at pE = 1.0, and it fails at 4p/4D (0.42) and 5p/4D (0.75) (N3). One null parent is absorbed as the exception (N4). Losing pairs still raises P(GEN) for unreliable effects (n1). |
| **B2** | HA8M not discriminating | **Resolved as relabelled; the replacement has new MAJORs** | HA8M → `EXPLORATORY_A` XA2. `graph_audit` reports zero-gradient tensors and asserts the layer-2 QK disclosure (R3). New arms `BfromVSGO_QK` / `BfromVSGO_nonQK` → HA10a/b. Their identification problems are N1 and N2. prsg is deferred to XXI with a stated reason, which is acceptable. |
| **B3** | B2 preordained; no equivalence; HB_d implied by the class | **Resolved in logic, but the implementation has N-B1 and N-B2** | TOST (`track_b:82-83, 99`). B1 requires HB_c (`:119`). HB_d is descriptive only. The B1 → B1-FUNCTIONAL and B2 → B2-LOCUS labels are honest. B3 is unreachable as a headline (N-B1). |
| **B4** | HB_c shared-D bias | **Resolved** | `track_b:74`: different D in both terms, same unit. The same-D 2×2 secondary is in `:75`. Planted `B_shared_D_null_no_difference` passes. (The balanced primary still confounds class with history: see N-B2. That is round-1 M10.) |
| **B5** | HB_d donor contamination | **Resolved** (one MINOR, n9) | `functional.interchange:107` patches from `rh` (hybrid run). The `rd` version is kept as `OQI_unmatched`. A base-run null is computed (`:110`). `IIA_norm` divides by `model_follow_on_hybrid_input`. The null's maximum difference is reported but not gated (n9). |
| **M1** | Mediation conditional on the switch | **Resolved** | `track_a.score:176-178` gives GENERALIZES_CONDITIONAL unless HA1 GENERALIZES on the same base. Denominator chain at `:172-173`. XA5 unconditional concordance at `:191`. For HA10 the relevant condition is vsgo, not HA1 (N2). |
| **M2** | Single screening order | **Resolved** | `plan.eligibility:96-105`: native and competent on both D901 and D902. The lowest such seed is used. |
| **M3** | Historical cohort not blind | **Resolved** | `PREVIOUSLY_OBSERVED`; fresh parents first in every preset. `score_all:205-211`: a fresh-only FAILS withdraws GENERALIZES, and < 4 fresh parents is annotated. Note: Standard has only 4 fresh parents, so in practice every Standard headline will carry the annotation. That is honest; `claims_can` says so. |
| **M4** | Parent confounded with init seed | **Resolved (documented)** | DESIGN §1 restates the unit. The key is renamed `variance_components_between_unit`. That rename broke `report.py:106` (n5). |
| **M5** | phi noise near zero | **Resolved** | `measure.py` records bootstrap SEs. EPS sensitivity 0.003 / 0.03 in `score_all:212`. `phi_E` / `phi_M` are reported. The floor-versus-SE argument (dev nec_M SE ≈ 0.0008) is acceptable. |
| **M6** | lbl validity; randnorm groups | **Resolved** (MINOR n4) | `flags_for:121-124`. `ARMS.randnorm.groups_by_base`. Step-100 window measurement on its own stream (`plan:151`). |
| **M7** | No compatibility checks | **Partially resolved** | 150/300 measurements and the INCOMPATIBLE flag exist for reverse arms. Not done: the step-100 composed-state measurements (`TP`) are loaded but never used; the "competence not recovered by step 300" criterion is missing; the 0.10 threshold has no calibration; forward transplants, including HA10a/b, are never checked (N5). |
| **M8** | False precision; multiplicity | **Partially resolved** | Track B: t-intervals, sign-flip p, Holm, "indicative" below 6. Track A still uses the percentile two-stage bootstrap (`hier.py`), although the adjudication says "t-intervals … track_a". Its sign-flip p tests phi = 0, which is the wrong null for I-like hypotheses (n6). Not decision-relevant, because counts decide. |
| **M9** | Matched precision: training-time confound and censoring | **Partially resolved** | Per-parent κ_p (`trackb_run.match`). Censoring gap > 0.20 gives NON_IDENTIFIABLE. Headlining requires matched/final agreement. Not done: LR and step covariate are not reported. κ_p = max puts the most precise anchors at the earliest steps, which maximises the age gap (n7). The agreement rule is what creates N-B1. |
| **M10** | Class confounded with intervention history | **Not resolved for HB_c; vacuous for HB_c_dir with one base** | N-B2. |
| **M11** | HB_d is a locus test | **Resolved (accepted, deferred)** | HB_d is descriptive. B1 is labelled "mechanism not identified". |
| **M12** | Stability near-guaranteed | **Partially resolved** | A full-LR (1e-3) continuation is added for anchors, and the wording is restricted. If anchors do not move at 1e-3, there is still no state shown to be movable (n12). Acceptable for an exploratory XA4. |
| **M13** | HA8E mis-specified | **Resolved** (MINOR n11) | `pair_outcome:82-83` requires class HYBRID. 1211 is co-primary (`score_all:213`). The DESIGN wording is fixed. How the two co-primary steps combine is undefined (n11). |
| m1 | FAILED_PRETRAINING | **Resolved**; analogous gap for screening (n8) | `plan.eligibility:88-91` |
| m2, m3, m6 | HA5c weak; HA4a anchor; High arms exploratory | **Resolved** | `role` fields; XA6 |
| m4 | Competence-loss rate | **Resolved** | `stats.competence_rate` |
| m5 | HA9 competence at 4000 | **Resolved** | `score:155` |
| m7 | var_comp descriptive | **Resolved** | renamed |
| m8 | HB_b descriptive | **Resolved** | not in the classification |
| m9 | Zero-gradient report | **Resolved** | R3. The vsgo "routing gradient intact" assertion is commented out (n3). |

**Correction to round 1 (my specialty's error, not the builder's).** Round-1 B1 proposed "SUPPORTS ≥ max(n − 1, ⌈0.8 n⌉)" as a rule that "scales". It does not scale. Because it takes the max, it requires all parents at n = 4 and all but one at every n ≥ 5. The builder implemented it faithfully. It is conservative, so I do not require a change, but DESIGN should describe it truthfully (n2).

---

## New objections

### N-B1. B3 and B2-LOCUS+B3 can never be the headline, so the frozen Track B prediction cannot be confirmed. Severity: **BLOCKER**.

**Where:** `track_b.score:138-139`, `track_b.score_level:111, 121-122`, `trackb_run.match`.

**Problem.**
- The matched level is defined by matching KL (κ_p). HB_e (the precision difference) is therefore ≈ 0 at the matched level by construction, and PRECISION_DIFF there is false.
- At the final level HB_e is expected to be positive (DESIGN §2: "dev dKL ≈ 0.015").
- The headline requires the first token of the two classifications to agree. So:
  - a true B3 gives matched `EQUIVALENT_NO_DIFFERENCE_DETECTED` and final `B3`, which becomes NON_IDENTIFIABLE;
  - the frozen prediction gives matched `B2-LOCUS` and final `B2-LOCUS+B3`, which also becomes NON_IDENTIFIABLE.

**Verified (R5) through the real scorer:**
- B3 truth → "NON_IDENTIFIABLE (matched 'EQUIVALENT_NO_DIFFERENCE_DETECTED' and final 'B3' disagree)".
- B2+B3 truth → "NON_IDENTIFIABLE (matched 'B2-LOCUS' and final 'B2-LOCUS+B3' disagree)".
- B2-only truth → B2-LOCUS (correct).

The planted test `B_B3_precision` passes only because it plants identical KL differences at both levels and checks the matched level alone. A real matched level cannot produce that input.

**Consequences:**
- One of the mandate's three Track B outcomes (B3) is unreachable.
- The most likely outcome under the frozen prediction is silently reported as NON_IDENTIFIABLE.
- That fails design-gate Q4 and Q8 for the classification.

**Required change:**
1. Split the classification into a **functional component** (B1-FUNCTIONAL / FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED / equivalence / NON_IDENTIFIABLE, plus the B2-LOCUS locus flag) and a **precision component** (HB_e).
2. The matched/final agreement rule applies to the functional component only.
3. HB_e is evaluated at the final (common-step) level only. The matched-level HB_e is reported as "≈ 0 by construction".
4. Define: B3 = functional equivalence at both levels AND no locus difference AND HB_e at the final level. Define "+B3" in the same way.
5. Replace `B_B3_precision` with a test that sets equal KL at the matched level and a KL difference at the final level, and asserts on `score()["verdict"]` (the headline), not on the matched level.
6. Add the same headline-level test for B2-LOCUS+B3.

### N-B2. Intervention history is confounded with class in HB_c. A history effect with no class effect produces FUNCTIONAL_DIFFERENCE, and with one eligible base it produces B1-FUNCTIONAL. Severity: **BLOCKER** (round-1 M10 not resolved).

**Where:** `track_b.score_level:74` (primary HB_c), `:96-98` (`same_sign … is not False`), `:119-120`.

**Problem.**
- Within one unit, the two classes are always fullB (unblocked) against fullI (blocked for 100 steps).
- So the primary HB_c *between* term compares models that differ in class **and** history. The *within* term compares models with the same history.
- The primary estimates (class + history). The same-D 2×2 secondary has between pairs of equal history and within pairs of different history, so it estimates (class − history). The secondary is computed but unused.
- In HB_c_dir, the native-versus-induced sign requirement passes when it is `None`. That happens in every parent without both bases eligible, which is plausibly the majority given E-eligibility rates.

**Verified (R6) through the real scorer, 6 parents, no class effect:**

| scenario | HB_c primary | secondary | headline |
|---|---|---|---|
| history effect 0.06 (blocked models share fixed deviations), both bases | +0.058 | −0.058 | **FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED** |
| history effect toward rule `local`, E base only | +0.075 | n/a | **B1-FUNCTIONAL** (`same_sign` = None) |
| control: true class effect 0.06, no history | +0.059 | +0.059 | FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED (correct) |

**Required change:**
1. **Primary class effect** = ½(primary + secondary), per parent, in parents with both bases eligible. Report the **history effect** ½(primary − secondary).
   - FUNCTIONAL_DIFF and EQUIVALENT are decided on this history-balanced estimate.
   - Where fewer than `min_parents_both` parents have both bases, the classification is "CLASS_HISTORY_CONFOUNDED" (a functional difference between {native, induced} organisations; class not separable from history). Equivalence can still be stated, because history cannot mask a difference in the other direction unless effects cancel; label it accordingly.
2. **B1-FUNCTIONAL requires `same_sign is True`**, not "is not False". With `None`, the label is at most CLASS_HISTORY_CONFOUNDED.
3. **Planted tests:**
   - a history-only effect (both bases, and one base) must give neither B1 nor FUNCTIONAL_DIFF;
   - a class-only effect must still be detected.
4. **`test_trackb_oc.py`** must include both bases, because its generator currently creates base E only.
5. **Budget consequence:** the Track B claim needs ≥ 4 parents with **both** bases eligible. State the expected number in COMPUTE_BUDGET and in `claims_can`.

### N1. HA10a/b separates QK weights from non-QK weights, not routing state from value state. As frozen it has no manipulation check. Severity: **MAJOR** (not a rejection, provided the changes are made before the freeze).

**Where:** `protocol.py:40, 56-58, 102-105`, `graph_audit.py`.

**Q1 (guaranteed by the intervention?) and Q3 (do competing accounts predict different outcomes?): the confound.** The graph audit (R3, M base, step-0 state) shows:
- **Under vsgo:** the gradients of **layer-2** `new/0/WQ, WK` (the target layer, where T = own2 lives) are **identical** to plain training. The gradients of **layer-0/1** `pre/0/WQ, WK, pre/1/WQ, WK` are **changed**. The reason is that removing the V/O gradient through T changes the residual-stream gradient upstream.

Consequences:
- The vsgo-specific QK state transplanted by `BfromVSGO_QK` (QK of L0, L1 and L2) is mostly a first-order upstream consequence of removing *value*-path gradient. In layer 2 it is only a second-order drift, through the diverging values. So:
  - an **I-like** HA10a could come from L0/L1 QK changes caused by the value-gradient removal. That is evidence about QK weights, not about T's routing state;
  - a **B-like** HA10a may only mean that L2 QK under vsgo hardly differs from B's. There would be nothing to transplant.
- Conversely, attention routing is a function of QK **and** the residual representations, which come from emb, LN, OV and MLP. `BfromVSGO_nonQK` therefore also transplants routing-relevant state.
- The QK / non-QK split is a parameter-group split. It is informative in those terms only.

**Q2 (already established?).** No. This is new.

**Q5 (controls in code?).** No manipulation check and no compatibility flag. Both arms are forward transplants, so N5 does not cover them.

**Required change:**
1. **Relabel** HA10a/b as "QK-weight state against non-QK-weight state learned under vsgo". The "routing-state against value-state" wording may appear only as an interpretation that is conditional on the manipulation check (point 2).
2. **Manipulation check**, preregistered and computed from data the plan already produces (window and TP states at step 100) plus a cheap analysis job:
   - per-layer ‖QK_vsgo − QK_B‖ / ‖QK_I − QK_B‖;
   - the attention mass on T (own2) at the layer-2 readout rows for B, vsgo, I and both composed states;
   - nec_M of the composed states at step 100 (already measured in `mtp:` and currently unused).
   - Freeze a rule: if layer-2 ‖ΔQK‖ ratio and Δ attention-on-T are both below a frozen floor, HA10a is reported as "NO_QK_CONTRAST (uninformative)", not B-like.
3. **Add the target-layer split** `BfromVSGO_L2QK`, and preferably `BfromVSGO_L0L1QK`, on M. This is two continuations per pair. The discriminating claim about T's routing state must rest on L2QK. If the budget does not allow it, state in DESIGN that HA10 cannot attribute a QK effect to the target layer.
4. **Preregister the 2×2 interpretation table** before the freeze:
   - (a I-like, b B-like) → QK state carries the vsgo effect;
   - (B, I) → non-QK state carries it;
   - (I, I) → redundant;
   - (B, B) → non-additive (both needed);
   - any MIDDLE → unresolved.
5. **Label** the HA10a frozen prediction "no development basis (no dev run of BfromVSGO exists)".

### N2. HA10a/b are conditioned on the wrong event, and HA10b is trivially supported where vsgo had no effect. Severity: **MAJOR**.

**Where:** `track_a.pair_outcome:71` (conditions on the fullI switch); `score:176-178` (GENERALIZES_CONDITIONAL keyed to HA1).

**Problem.**
- HA10 decomposes the **vsgo** effect. In a pair where the vsgo arm itself is B-like:
  - HA10b (expect B-like) is SUPPORTED for free;
  - HA10a is CONTRA for free.
- phi is also measured against the fullI anchor. If vsgo gives phi ≈ 0.7 relative to fullI, no component of vsgo's state can reach the I-like band, even if it carries all of vsgo's effect.

**Required change:**
1. For HA10a/b, a pair is evaluable only if the vsgo arm of that pair is competent and of the I class at the scoring step. Otherwise the status is `ANCHORS_NOT_EVALUABLE (vsgo had no effect)`, and it is counted.
2. Additionally report phi relative to the vsgo arm: phi_vsgo = [l(a) − l(B)] / [l(vsgo) − l(B)].
3. GENERALIZES for HA10 is GENERALIZES_CONDITIONAL unless HA7 (vsgo) GENERALIZES on M.
4. Add a planted case: vsgo B-like in every pair must leave HA10b INCONCLUSIVE, not GENERALIZES.

### N3. The verdict operating-characteristic test does not enforce its own documented criterion, and DESIGN overstates power. At the Standard minimum, a strong effect is INCONCLUSIVE in 56% of campaigns. Severity: **MAJOR**.

**Where:** `tests/test_verdict_oc.py:31-33`, `DESIGN_XX.md:62`, `PRESETS.standard.n_D = 4`.

**Problem.**
- The docstring says P(GEN | pS 0.9, pC 0.02, **pE ≥ 0.8**) ≥ 0.8.
- The code checks `t["GENERALIZES"] >= 0.8` only for pE = 1.0. For pE = 0.8 (`t8`) it checks only FAILS.
- The stored results show the failure:

| setting | P(GEN \| strong effect, pE 0.8) | P(INCONCLUSIVE) |
|---|---|---|
| 4 parents, n_D = 4 | **0.42** | 0.55 |
| 5 parents, n_D = 4 (R4) | 0.74 | |
| 4 parents, n_D = 6 | 0.90 | |

**Mechanism:**
- MIN_PAIRS_PER_PARENT = 3 of 4. So P(a parent has ≥ 3 evaluable pairs) = 0.82.
- With n_ev ≥ 4 required and all 4 parents needed (⌈0.8 · 4⌉ = 4), the result is 0.82⁴ ≈ 0.45.

This is honest, because INCONCLUSIVE is not a false verdict. But DESIGN §1 states "P(GEN | strong effect) ≥ 0.97 at 4–12 parents", which is true only when every pair is evaluable. The E base, where eligibility is rare, is the likely casualty.

**Required change:**
1. Make the test enforce the documented criterion for `t8`, or change the criterion in the docstring and in DESIGN to the true numbers.
2. Choose and freeze one of:
   - Standard n_D = 6, which gives 0.90 at 4 parents;
   - keep n_D = 4 and state in `claims_can` that a 4-parent base is INCONCLUSIVE in about half of campaigns, even for a strong effect with 80% evaluability.
3. Add these scenarios to the test (reported, with frozen thresholds where stated):
   - unreliable effect (pS 0.6, pC 0.25);
   - half near-miss (pS 0.5, pM 0.5);
   - one null parent among 4, 5 and 8;
   - **evaluability monotonicity:** P(GEN) at pE 0.8 must not exceed P(GEN) at pE 1.0 by more than Monte Carlo error for pS ≥ 0.5. It currently does in several cells, for example 60S/25C at 8p/6D: 0.034 → 0.085 (R4).

### N4. A parent with no effect is absorbed as the "one exception" at ≥ 5 parents. Severity: **MAJOR** (heterogeneity across parents is the claim being tested).

**Where:** `track_a.parent_verdict:97` (CONTRADICTS iff C > n/2).

**Problem.**
- Take a parent where the effect is absent (pS 0.1, pC 0.7). With n_ev = 4 it is PARENT_PARTIAL whenever it has only 2 CONTRA, which happens about 35% of the time.
- The cross-parent rule allows one PARTIAL at n ≥ 5.
- R4: 8 parents, one of them null, the rest strong → **P(GENERALIZES) = 0.34** at n_D = 4 and 0.25 at n_D = 6.

**Required change:**
- PARENT_CONTRADICTS iff C ≥ ⌈n_ev/2⌉ **and** C > S.
- R4 (`oc4.py`) shows the effect, with strong-effect power unchanged (0.977 / 0.996) and the null still never GENERALIZES:

| setting | P(GEN), current rule | P(GEN), proposed rule |
|---|---|---|
| 8 parents, one null, n_D = 4 | 0.34 | 0.11 |
| 8 parents, one null, n_D = 6 | 0.25 | 0.08 |

- An equivalent rule is acceptable if it passes the N3 scenario set.

### N5. The compatibility rule (round-1 M7) is incomplete, and its threshold is uncalibrated. Severity: **MAJOR**.

**Where:** `protocol.VALIDITY`, `track_a.flags_for:125-130`, `load_inputs:60` (TP loaded, unused).

**Problem:**
- "KL exceeds both anchors by > 0.10 at step 150 or 300" is an absolute threshold. It is set without any measurement of anchor KL at those steps; the dev notes have none.
- The step-100 composed-state measurement, which is the direct "immediate change" record that mandate A4 asks for, is never used.
- The "competence not recovered by 300" half of the round-1 rule was dropped.
- Forward composed states are never checked: HA4a, HA5a/b, the new HA10a/b, randnorm and half_L0L2. A large forward shock that re-learns along I's default path would count as sufficiency.

**Required change:**
1. Compute and report the flag for **every** composed or perturbed arm.
2. Keep its *consequence* (the pair is not counted as evidence) for reverse arms (B-like outcome) **and** for forward arms with an I-like outcome.
3. Include the step-100 KL excess over the matched B and I window states in the rule.
4. Calibrate the threshold on the pilot (separate material: nP3901, D2901). Freeze it before confirmation, or make it relative, for example > 2 × max(anchor KL) at that step.
5. Report flag rates per arm.

---

## MINOR

- **n1. Leniency against data-order-unreliable effects.**
  - The rule is *not* lenient against weak-magnitude effects. A phi of about 0.5 is CONTRA under every expectation, because MIDDLE is only a near-miss zone of width DELTA = 0.10.
  - It *is* lenient against effects that fail in some data orders. pS 0.6 / pC 0.25 gives GEN 0.44 at 5p/4D (R4).
  - **Required:** print the pooled pair-level SUPPORTED and CONTRA fractions next to every verdict, and label a GENERALIZES with pooled CONTRA > 0.15 as "GENERALIZES (data-order-unreliable)".
- **n2.** The cross rule max(n − 1, ⌈0.8 n⌉) means "all parents at n = 4, all but one at n ≥ 5". It never scales; that was round 1's error, and it is conservative. DESIGN must say this, not "scales".
- **n3.** In `graph_audit.py:62` the assertion `vsgo_routing_gradient_intact` sits inside a `#` comment, so it is never evaluated. The measured dL/dbias_T under vsgo is nonzero (−0.0079 on E, −0.0061 on M), so the property holds, but the gate does not check it. Restore it; HA10 depends on it.
- **n4.** In `flags_for:123`, a missing window record yields CONTROL_INVALID instead of MISSING. Also, if I's step-100 KL excess over B is ≤ 0, the lbl threshold passes trivially. Use MISSING, and require (I − B) > 0, else CONTROL_INVALID.
- **n5.** `report.py:106` indexes `h['variance_components']`, but the scorer now writes `variance_components_between_unit`. The statistical-analysis report raises KeyError as soon as any phi exists. Rename it and add the report path to the smoke test.
- **n6.** Track A phi CIs still come from the percentile two-stage bootstrap, contrary to the adjudication's "t-intervals … track_a". The reported sign-flip p tests mean phi = 0, which is meaningless for I-like hypotheses. Report the t-interval on parent means and a sign-flip p for (phi − band edge), or drop the p.
- **n7.** Matching:
  - report the matched step, the LR at the matched step and the per-parent EARLY − MIXTURE step difference;
  - add the round-1 sensitivity analysis with step as a covariate.
  - κ_p = max means the most precise anchors are matched at their earliest snapshot, which maximises the training-age gap. With N-B1 fixed, the matched/final agreement requirement is an adequate guard.
- **n8.** In `plan.eligibility:97-101`, a parent with a permanently *failed* screening job is skipped and never recorded. Record FAILED_SCREENING, which is the analogue of the m1 fix.
- **n9.** The HB_d base-run null (`null_base_patch_max_logit_diff`) is reported but not gated. Freeze a tolerance (for example 1e-4). Above it, HB_d is INVALID for that model.
- **n10.** DESIGN §1, design-gate Q3 still cites "HA8M/E", and there are no Q1–Q8 answers for HA10a/b. Replace them with the table below.
- **n11.** HA8E's co-primary step 1211 is reported (`co_primary`), but no rule combines it with step 4000. Freeze one; for example, the headline is the 4000 verdict, and a conflicting 1211 verdict is shown in the same row.
- **n12.** XA4's full-LR control is applied to anchors. If no anchor moves at 1e-3, there is still no state shown to be movable. The restricted wording covers this. Add one HYBRID-arm (HA8E) continuation at 1e-3 if Extended runs.
- **n13.** TOST: report the *achieved* equivalence bound per test (the smallest margin at which equivalence would be declared), as round 1 required ("report the achieved power"). The OC simulation alone is not enough.
- **n14.** The Track B planted and OC tests evaluate only the matched level and a single base. This is why N-B1 and N-B2 went undetected. Every Track B test must assert on `score()["verdict"]`, and at least one generator must have both bases.
- **n15.** The negative-results list (`report.py:109`) shows MISSING, NONCOMPETENT and ANCHORS_NOT_EVALUABLE but not CONTROL_INVALID or INCOMPATIBLE. Include them.

---

## Design-gate answers (mandate §2, Q1–Q8): new or changed tests only

Abbreviations: Y = yes; N = no; P = partly; "after Nx" = after the named change.

**Q1** guaranteed? · **Q2** established? · **Q3** explanations differ? · **Q4** metric distinguishes? · **Q5** controls in code? · **Q6** positive establishes? · **Q7** negative weakens? · **Q8** inconclusive without silent exclusion?

| test | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
|---|---|---|---|---|---|---|---|---|
| **HA10a BfromVSGO_QK (M)** | N | N (new) | **P**: QK-weight vs non-QK-weight; not routing vs value, because L0/L1 QK change first-order under vsgo and L2 QK does not (R3) (N1) | P (phi against fullI, not vsgo, N2) | **N** (no manipulation check, no compatibility flag; N1, N5) | P: QK weights suffice; T-routing only after N1 (L2QK split) | P: B-like is uninformative if there is no QK contrast (N1) | P (N2: wrong conditioning event) |
| **HA10b BfromVSGO_nonQK (M)** | N | N | P (as HA10a) | P | N | P | P | **N**: trivially SUPPORTED where vsgo was B-like (N2) |
| HA8E rsg (E), changed | N | partly (XIX frzp-E partial) | P | Y (HYBRID class now required) | P | weak (as DESIGN says) | Y (CONTRA is informative) | Y (combination of co-primary steps undefined, n11) |
| HA6b lbl (validity added) | N | uP160 | Y | P | Y (after n4) | Y (valid pairs only) | Y | Y |
| HA6a randnorm (per-base groups) | N | uP160 | Y | P | Y | Y | Y | Y |
| HA4b / HA5d with INCOMPATIBLE | N | uP160 | Y after N5 | P | P (N5) | P | Y | Y (flag converts only SUPPORTED pairs: conservative) |
| XA2 rsg (M), demoted | **Y** (L2 QK frozen) | Y (XIX frzp-M) | N | n/a | Y (disclosed) | replication only | replication only | Y |
| XA5 unconditional concordance | N | N | Y | Y | Y | descriptive | descriptive | Y |
| Parent/cross verdict rules v2 | n/a | n/a | n/a | P (N3, N4) | Y (OC test, after N3) | Y | Y (FAILS needs contradictions) | P (56% INCONCLUSIVE at 4p/4D/pE 0.8; honest, but must be stated, N3) |
| Conditional label / fresh-only | n/a | n/a | n/a | Y | Y | Y | Y | Y |
| Two-order eligibility | n/a | n/a | n/a | Y | Y | n/a | n/a | P (n8) |
| Track B HB_c balanced primary | N | N | Y | **N as coded** (class ⊗ history, N-B2) | P | Y after N-B2 | Y after N-B2 | Y |
| HB_c_dir | N | N | Y | P | P (`same_sign` None passes, N-B2) | Y after N-B2 | P | Y |
| TOST equivalence | N | N | Y | Y | Y | Y (equivalence within the margins at the analysed level) | Y | Y (NON_IDENTIFIABLE is explicit) |
| HB_d (descriptive locus) | ≈Y given the class definition | P | N (locus only, as labelled) | Y after B5 | Y (null computed; gate it, n9) | descriptive | descriptive | Y |
| Per-parent precision matching | n/a | n/a | n/a | P (n7) | Y | n/a | n/a | Y (censoring gap rule) |
| Classification B1-FUNCTIONAL / B2-LOCUS / B3 | n/a | n/a | n/a | **N** (B3 unreachable, N-B1; history, N-B2) | n/a | B1-FUNCTIONAL honest after N-B2 | P | **N** (the frozen prediction collapses to NON_IDENTIFIABLE, N-B1) |
| XA4 full-LR control | ≈Y (anchors may not move) | N | P | P | P (n12) | N | Y | Y |

## Answers to the specific questions in my brief

**Is the recalibrated parent rule sound, and is it too lenient against weak effects?**
- *Soundness:*
  - Monotone 4 → 6 D for strong effects; P(GEN | null) = 0; FAILS only with contradictions. The builder's reason for rejecting the 0.75 rule (⌈0.75 · 6⌉ = 5 of 6 is a jump) is correct.
  - Residual sawtooth from ⌈0.5 n⌉ at odd n (R4: pS 0.5 at 4 parents gives 0.22 / 0.06 / 0.19 for n_D 4 / 5 / 6). This matters because evaluability losses create odd n_ev.
- *Leniency:*
  - Not lenient against weak-magnitude effects, because MIDDLE is a narrow near-miss band.
  - Moderately lenient against data-order-unreliable effects (n1).
  - Too lenient against a whole null parent at ≥ 5 parents (N4).
  - Under-powered at the Standard minimum, and the test hides this (N3).

**Is BfromVSGO_QK / nonQK a discriminating replacement? Confounds?** It discriminates *parameter groups* (QK weights against the rest), not *routing state against value state*.
- Measured: vsgo changes L0/L1 QK gradients directly and leaves layer-2 QK gradients identical at the start. Routing also depends on non-QK state.
- It also lacks a manipulation check, uses the wrong conditioning event and has no compatibility check.
- Accepted after N1, N2 and N5. Ideally the L2QK split is added.

**Conditional labelling and fresh-only verdicts?**
- Correctly implemented for HA1-conditioned hypotheses.
- HA10 must instead be conditioned on vsgo (N2).
- Fresh-only is correctly implemented. In Standard it will almost always produce the "< 4 fresh parents" annotation, which is honest.

**Could INCOMPATIBLE or CONTROL_INVALID silently remove inconvenient pairs?**
- **No.** INCOMPATIBLE converts only a B-like SUPPORTED pair of a reverse arm. Removing supporting pairs can only push a parent toward PARTIAL or CONTRADICTS.
- CONTROL_INVALID is decided from step-100 window KL, before the outcome, and an under-strength lbl is the case most likely to *support* HA6b. Both are counted in the pair status table.
- The gaps are coverage (N5) and labelling (n4, n15), not selective exclusion.

**Track B: is TOST at parent level with 4–8 parents achievable? Is B1-FUNCTIONAL honest?**
- *Achievability:* it depends on the between-parent SD of HB_c, which is unknown. In the builder's simulation (SD 0.02), equivalence holds in 75% of null campaigns at 4 parents and 100% at 8. With t₃ = 2.35, equivalence at 4 parents needs a parent-level SD below about 0.017 for HB_c, and below about 0.025 for each of the 4 HB_c_dir rules. That is plausible but not assured.
- The NON_IDENTIFIABLE fallback is honest. Report the achieved bound (n13).
- B1-FUNCTIONAL is an honest *functional* label only after N-B2. As coded, a history effect can produce it.

**HB_c balanced design?** The shared-D bias is fixed (B4 resolved). The balanced primary still mixes class with history; the secondary contains the cancelling term (N-B2).

**Per-parent precision matching?**
- Removes class-dependent censoring at κ_p, as claimed.
- Maximises the training-age gap (n7).
- Combined with the agreement rule, it makes B3 unreachable (N-B1).

**New problems introduced by the corrections:**
- N-B1 (matching plus the agreement rule);
- the HA10 confounds (N1, N2);
- N3: the OC test omits its own criterion;
- the `report.py` KeyError from the rename (n5);
- n3: the commented-out vsgo assertion.

---
*End of review A, round 2. Required before the freeze: N-B1, N-B2, N1–N5, each with the planted or OC test named above, and the MINOR items where marked "Required". A targeted check of these changes is sufficient; a third full round is not required.*
