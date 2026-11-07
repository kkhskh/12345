# Adjudication of the review round

## Reports reviewed

The review covered commit `a899a7c`.

| report | role |
|---|---|
| `reviews/independent_verification/REPORT.md` | independent reimplementation |
| `reviews/R_causal.md` | causal identification |
| `reviews/R_math.md` | mathematics, including a re-review of the earlier package's post-review results |
| `reviews/R_counterexample.md` | counterexamples |

**The independent reimplementation agreed with every checked number.**

Each finding below was accepted, and the package was corrected. One new experiment was added in response, REL-XOR3. It was frozen **before training**, as `PROTOCOL_ADDENDUM_XOR3.md` at commit `c7bcd2c`.

## Findings that changed the headline

| # | finding | source | disposition |
|---|---|---|---|
| 1 | In REL-XNOR and REL-MIS, opposite agreement at matched aggregate always means opposite target. So S_rel is a target-class contrast, and "output-free" is only nominal. In TF2 XNOR, U correlates 0.90–0.996 with the margin and has cosine 0.79–0.97 with the readout direction. | R_causal F1; R_counterexample F1 | **Accepted.** The XNOR mediator is reclassified as a **localised target variable**, not evidence of a relation used as an intermediate. The deciding experiment the reviewer asked for, REL-XOR3, was added and run. |
| 2 | TF2 `resid_mid1@R` is effectively the readout state. Deleting the final MLP leaves accuracy at 0.999–1.0, so this is the trivial case the protocol meant to exclude. The readout direction w passes every gate. | R_causal F2; R_counterexample F1 | **Accepted.** That site's verdicts are labelled "readout-equivalent". |
| 3 | Sufficiency is identical to reverse-direction interchange. They are equal in 150 of 150 logged rows, so T-MED's (H2) adds nothing. | R_math 1; R_causal F3 | **Accepted.** (H2) is merged into (H1). |
| 4 | T-MED Conclusions 2–3 do not follow from (H1)–(H4). (H4) was measured on the grid, not on held-out pairs. The additive part alone already accounts for 34–37% of pair flips in s4 and s8. | R_math 2–3 | **Accepted.** T-MED is restated, and a missing hypothesis is added. |
| 5 | The F1 chain mixes TF1 (T-ATT) with TF2 (T-MED). | R_math 4 | **Accepted.** The chain is restated per model class. |
| 6 | E4 is a reseed, not a distribution shift. Its pairs are still uniform. With genuinely Gaussian pairs, s4 fails necessity (0.305). | R_causal F6; R_counterexample F2 | **Accepted.** E4 is relabelled, and the reviewer's Gaussian-pair result is cited. |
| 7 | S_alt is not equally decodable: it decodes agreement at 0.52–0.73, against 0.999 for U. | R_causal F4 | **Accepted.** S_alt is now described as an "orthogonal, partially decodable" control. The claim "decodability ≠ use" is weakened accordingly. |

## Other findings, all accepted

- **T-ATT omitted the output bias b_O.** It is added; with it the formula matches the forward pass to 3e−7.
- **Correct statements kept:** T-ATT itself is exact (R_math), and normalizer-only freezing drives the interaction share to ≤ 1e−28.
- **Agreement and spread are the same variable on all of ℝ²** for scalar sources, not only at fixed z. Corrected.
- **The cut check passes by construction.** Relabelled as a code check, not evidence.
- **The manifold ratio has a near-zero denominator** (saturated plateaus). The reviewer's within-pair measure is cited: patched states lie 0.7–25% of the pair distance from a natural state.
- **AGG-NL "0 of 500".** Near |z| = 1.5 the model's z-boundary shifts with the spread by up to 0.032, so 66–1546 of 2000 near-boundary pairs differ. Scoped to the default pair distribution, and reported.
- **One REL-GATE verdict flips** on near-boundary pairs (TF2 s8). Reported.
- **Forcing k changes outcomes:** k = 1 fails necessity for s3 and s8; at k = 4 the random controls weaken. Reported.
- **Earlier package, T-RF′:** the additive-gate twin has 0 row reversals but 79 of 435 column reversals, and an interaction share of 0.342. The claim "both twins have 0 reversals" is withdrawn, and the test must check both orientations (`ERRATA_PREVIOUS_PACKAGE.md`).
- **Earlier package, minor items:**
  - the K4 grid size;
  - medians-only in the C2 table (`q_B` reaches 8% at single points);
  - the conditions behind N2;
  - T-MS is a definition, not a theorem.

  All are recorded in the errata.
- **`aggnl_reversals.json` had no committed script.** Now added as `scripts/aggnl_reversals.py`.

## The REL-XOR3 result: the deciding experiment

Run according to the frozen addendum (`XOR3_RESULTS.md`):

| site | addendum verdict | content test |
|---|---|---|
| TF1 `resid_mid0@R` (the 3 models that learned the task) and TF2 `resid_mid1@R` (5/5) | **TARGET_ONLY.** The patched decision follows the source's **target** on 1.000 of decisive pairs, and agreement on 0.000. DAS cannot realise agreement there (0.0–0.48). | In TF2 the subspace is a computed interaction. That interaction is the target, `sign(ab)·p`. |
| TF2-AO layer-0 output | **AGREEMENT_INTERMEDIATE** in 3/5 by the frozen rule (S_rel follows H_AB on 0.92–1.0) | The subspace is **source-additive** (additive R² 0.998–1.000; the additive surrogate preserves every decision). |

In XOR3 the output depends only on `sign(a)·sign(b)·p`. A subspace that carries the two source signs separately therefore satisfies H_AB without computing agreement. **The addendum's rule could not separate these cases. The content test does, and it shows no computed agreement.**

TF1 seeds 3 and 6 did not learn the task (accuracy 0.498 and 0.749).

**Net effect.** In no learned model did agreement appear as a *computed* intermediate distinct from the target. The headline answer changes from "yes, restricted" to **no**, as set out in `MAIN_ANSWER.md`.
