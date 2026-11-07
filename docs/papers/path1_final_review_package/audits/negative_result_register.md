# Negative-result register

Every negative, inconclusive or limiting result from the studies behind this paper, with its official letter where one exists, its evidence level, and where the manuscript reports it. Letters are copied verbatim and none is revised here or in the manuscript. Values quoted are those in the manuscript, which the verifier checks against their sources.

| # | Result | Letter | Evidence | Where in the manuscript |
|---|---|---|---|---|
| 1 | The board-edit validation of Editor A did not produce the behavioral repair its protocol required. | `08R2_VALIDATION_COMPLETE — NEGATIVE` | letter from the original run; pair tables not found | Results, "The original board-edit validation remains negative"; Activations and interventions |
| 2 | The first execution of the equal-JS design did not follow its protocol; a forensic audit recorded the deviation. Its numbers are not used. | `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` | letter and protocol | Results, "Target identity still matters when output divergence is matched" |
| 3 | A symmetric-target design on the remaining validation games was infeasible because no category-B triplets existed. | `SYMMETRIC_TARGET_CENSUS_INFEASIBLE` | archival console excerpt | Pairs and triplets; switching results |
| 4 | Joint target switching failed its preregistered gate: both contrasts were positive in 13 of 32 triplets, below 60%. | `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED` | archival | switching results; Figure 5; Abstract |
| 5 | No visible–kernel interaction: \(I_T\) and \(I_W\) are below the 0.005 floor, and both intervals include zero. | `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED` | archival | factorial results; Figure 5; Abstract |
| 6 | Adding the occupancy one-hots or the legal mask to the other feature set improves the map by less than the preregistered +0.01 floor. | `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY` | archival | stage and legal-mask results |
| 7 | STAGE_ONLY accounts for 56.04% of the learned map's mean gain over Editor A. | none | archival differences combined with one recomputed value | stage and legal-mask results; Table 1 context; Abstract |
| 8 | Mean \(Y\) stays negative after every kernel edit at matched divergence; the learned edit's \(Y\) is positive in only 1 of 48 pairs. | none | recomputed | matched-divergence and NST results; Abstract; Conclusion |
| 9 | At matched divergence the advantage over random directions has an interval that includes zero in category A and in pairs where Editor A did not decode the exact board. | none | recomputed | matched-divergence results; Figure 3 |
| 10 | NST transfer is weaker on pairs where Editor A did not decode the exact target board; both of its contrasts there have intervals that include zero. | none | recomputed | NST results; Figure 4 |
| 11 | At matched L2, random kernel directions barely move the output (mean JS 0.038), so SPEC's correct-minus-random contrast partly compares an effective edit with an ineffective one. | none | recomputed | SPEC results |
| 12 | The learned and oracle directions are different vectors (mean cosine 0.792); the learned direction is closer to the stage-mean direction (0.850). The score correlation of 0.998 reflects baseline \(Y\), not a shared mechanism. | none | recomputed | NST results; Figure 4 |
| 13 | The gradient baseline reached the JS tolerance on only 7 of 48 pairs and is not compared. | none | recorded | NST results |
| 14 | The obstruction program's status is unchanged, and this paper does not use the obstruction framing. | `0A = C / PARTIAL`, `SECTION_5_3 = PARTIAL`, `OBSTRUCTION_READY = NO` | project letters | Reproducibility and data availability (letters listed verbatim) |

## Results that are positive but bounded

The positive letters, `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`, `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED` and `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`, and the FIBER diagnostic letter are reported with their effect sizes on the \(Y\) scale. The manuscript never describes a contrast as large because its interval excludes zero. Matched-divergence effects are a few hundredths of probability mass.

## Things the paper does not do with these results

- It does not treat result 2's run as evidence for either side.
- It does not reinterpret result 1 as rescued by the kernel studies. It states that nullspace edits are a different intervention family.
- It does not read results 4 and 5 as underpowered successes. Positive average discrimination is reported separately from the failed joint gate.
- It does not invoke the obstruction theory to explain any result.
