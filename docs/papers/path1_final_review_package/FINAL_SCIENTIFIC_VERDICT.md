# Final scientific verdict for Path 1

**Verdict:** `PATH1_COMPLETED_WITH_EXPLICIT_EXTERNAL_VERIFICATION_LIMITATIONS`

## Why this verdict

The manuscript is complete. It is built from a single source, and every number in it is checked against its source by the verifier. The core evidence is recomputed from saved arrays: the matched-L2 study, the matched-divergence study, the donor-free map, probe invariance relative to the Editor A reference activation, and the FIBER diagnostic. Every recomputed interval matches the original runs to within \(3\times10^{-8}\). The package reproduces from a clean extraction without the research repository.

The verdict is not `PATH1_COMPLETE_READY_FOR_INDEPENDENT_HUMAN_REVIEW`, because some limits cannot be removed from inside this assignment:

1. The feature-ablation, switching and factorial results, which include the two failed completeness tests, rest on verbatim console output. Their arrays were searched for and not found, so their means and intervals are archival, not recomputed. The 56.04% stage share combines archival differences with one recomputed value.
2. No intervention was replayed through the model. The checkpoint and probe are not redistributed, and replay was outside this assignment. Recorded quantities such as the map's reconstruction \(R^2\) are read from the original runs' files.
3. The board-edit validation's pair tables are not in the package; only its letter is reported.
4. The sealed confirmation split was not used. The positive kernel results rest on two reused samples of 48 pairs.
5. The adversarial reviews in `reviews/` are AI-generated simulations. No independent human scientist has reviewed the package. Authorship and the disclosure statement are still to be written by the people responsible.

None of these is a hidden failure. The manuscript states each one where it matters.

The other verdicts were considered and rejected. The evidence is not blocked, because the central claims are recomputed and pass every check. Reproducibility did not fail: the pipeline and tests pass in a clean extraction. The manuscript and figures are complete. Novelty is not claimed, so it cannot fail to be established. The paper presents a scoped controlled combination of existing tools and makes no priority claim; see `audits/novelty_review.md`.

## What the evidence supports

For this frozen probe, this model and this site: kernel edits that leave the probe's 64×3 probabilities at the Editor A reference activation unchanged still shift legal-move probability. They shift it more for the intended target than for wrong-target and random directions, under matched L2 and under matched divergence (\(+0.0266\), interval \([0.0179, 0.0357]\), against wrong targets). A map trained only on training games produces part of the effect without a target activation at evaluation (\(+0.0201\) against wrong targets). Mean scores stay negative, more than half of the map's gain is stage-conditioned, joint switching fails its gate in 13 of 32 triplets, and no visible–kernel interaction is found. The paper's conclusion is that this probe does not exhaust the residual information that affects legal-move outputs. It does not claim a second world model, a transferable board state, or anything about other probes or models.

## Official letters, unchanged

`08R2_VALIDATION_COMPLETE — NEGATIVE`; `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`; `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`; `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`; `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`; `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`; `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`; `SYMMETRIC_TARGET_CENSUS_INFEASIBLE`; `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED`; `0A = C / PARTIAL`; `SECTION_5_3 = PARTIAL`; `OBSTRUCTION_READY = NO`.

This paper does not revise the obstruction program's status and does not use its framing.

## What would change the verdict

- Recovering the SEM, FRESH and VNF arrays and recomputing their results.
- An authorized replay of a sample of interventions through the checkpoint, reproducing the saved logits.
- Review of this package by an independent human scientist.

## Distribution

Private. Not submitted and not public. Do not distribute without the authors' authorization.
