# Adjudication of the independent agent reviews (Investigation XIII-B)

These are three separately initialised agent reviews, not human peer review. Every review could reject the primary interpretation, and
each partly did. The documents carry their corrections in "Corrections" boxes at the top and in rewritten sections. The original
development text is kept below the boxes for the record.

## Review 1: mathematics (`REVIEW_1_MATH.md`, scripts in `review1_scripts/`)

| finding | disposition |
|---|---|
| The Proposition (β = 1) proof had a gap: W vs W′ weights, and the case of several admissible roles | **Accepted.** Proof rewritten in two cases; numerical check max |Δ| = 1e-16 recorded |
| PERGROUP is the independent-copying product rule, not exact Bayes of a censored observation; a shared-copy-aware no-comparison rule reaches 0.671 | **Accepted.** Re-described; deficit attributed mainly to ignoring the shared copy event |
| "At β = 1 the J-pairs are off-support" is false for 31.5% of them, which have zero exact change | **Accepted.** Corrected |
| The POE family is handicapped (unnormalised exponents); the fair nested power-mean family shows class-dependent ρ | **Accepted.** New exploratory analysis (`fit_powermean.py`); post-freeze addendum AQ1 registered before evaluation (it held). Q1a declared non-diagnostic against normalised pooling |
| κ "softer exclusion" is an ε-floor artefact | **Accepted.** Withdrawn |
| Scope ("never needed", "exactly sufficient") must be stated inline; "independent pathway" is downstream on the same path | **Accepted** |

## Review 2: experimental integrity (`REVIEW_2_INTEGRITY.md`)

| finding | disposition |
|---|---|
| Chronology and hashes | **Confirmed by the reviewer.** Freeze pushed 1 s before the lanes started; AQ1 committed before its run |
| A failed arm can be reported as "inconclusive"; filtered units with missing inputs are dropped silently | **Accepted.** RB1 and RB2 (`report_13b.py`), logged before any output was opened. Neither case occurred |
| Diagnostic labels are misleading (Q1a, Q4e, Q5b, Q5c, Q6c) | **Accepted.** RB3 strict flag; non-diagnostic arms named in FROZEN §4.2 |
| Preflight nulls are author-written | **Accepted (disclosure)** |
| Q4's decoy set excludes PERM and MEAN, which "work" by projection | **Accepted.** RB4 sensitivity: Q4c also fails with them included. Review 3 adds that MEAN is off-distribution |
| Thin development coverage for Q5 and Q6; stream and seed reuse (`mix_mass` episodes, jpath seed); Q2 low power; non-independent predictions; AQ1 underspecified | **Accepted (disclosure).** AQ1 aggregation rule added before its output was opened |

## Review 3: mechanism and causality (`REVIEW_3_MECH.md`, scripts and outputs in `review3_scripts/`)

| must-fix | disposition |
|---|---|
| 1. Q5 is not evidence against E: site coverage 0.49–0.57 in early units; target includes the model's own deviation; Bayes-to-Bayes rescoring gives 0.49 / 0.20; das_bal "NOOP" is not a no-op | **Accepted.** Reinterpreted in all documents. The frozen verdict (falsified) stands. The NOOP bug is disclosed; that row is in effect a second random control |
| 2. Q3b and Q4c are one event, within noise; withdraw the "categorical routing law" | **Accepted** |
| 3. Class inheritance overstated: 2/5 parents moved by > 0.3 | **Accepted** |
| 4. ρ is not uniquely a combination operator (tempering reparametrisation, lapse floor, about 15% of misfit); multi-start fitting needed | **Accepted.** Wording now "robust class marker of the input-output law". The reviewer's bootstrap and alternative fits are archived |
| 5. "Mostly MLP" not identified; a route via other-question tokens read by layer-2 attention was missed | **Accepted** |
| 6. TOP decoy not contribution-matched in seed 21; MEAN off-distribution; restoration adds nothing beyond sufficiency | **Accepted** |
| 7. Adaptation: capacity unmatched; "reuses mixture" untested; J entering at resid1 at OS is true by construction; the probe performs the sum; early01 jpath output missing | **Accepted.** Claims narrowed; `logs/jpath_dev_P161.json` regenerated with native / early01 / full |
| 8. "E-based" should be "agreement-at-q-gated" (E not separated from A2 by D1/D2) | **Accepted** |
| 9. Confidence intervals: Q3b and Q1a(uP164) lie within about one SD of threshold | **Accepted (disclosure).** Per-unit values reported; borderline units named. No re-scoring |

**What the reviews did not change:** the frozen verdicts (Q1, Q2, Q6 confirmed; Q3, Q4, Q5 falsified) and the AQ1 outcome. They changed
what those verdicts are taken to mean.
