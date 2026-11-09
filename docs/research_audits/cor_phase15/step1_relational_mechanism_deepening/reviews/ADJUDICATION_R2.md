# Adjudication of review round R2

One independent reviewer (`R2_counterexample_defence.md`; counterexample construction and positive-theory defence)
examined only the claims added or changed after R1. Its null-control script is `reviews/checks/r2_null.py`, output
`logs/review_checks/r2_null_ref_T1p1.out` (600 episodes, 1393 informative pairs). I read the script and the output; the
per-slot score (0.799) and full-source patch (0.833) agree with my own R1 re-run (0.805 / 0.833 at 600 episodes, different
fitting draw), which serves as the consistency check. Corrections are marked **[R2]** in the documents.

| R2 finding | verdict | change |
|---|---|---|
| Capacity-matched nulls (permuted pairing 0.073, wrong groups 0.075, answer label 0.080) fail → per-slot alignment is not trivially expressive | accepted (supports CA1) | added to PCOUNT A3, CAUSAL §4, synthesis Q3 |
| High floor: true encoder + counts from a random other pair → 0.459; above floor, 91 % of the full patch's gain (ref_T1p1) | accepted | raw 0.66–0.82 now reported with the floor; P-count floors not measured (open) |
| ref_T1p2 per-slot 0.657 > full patch 0.563: the patch is not a ceiling for an added fitted direction | accepted | stated in PCOUNT A3; "93–96 % of ceiling" not used |
| "Same abstract variable" over-reaches: own-group count alone fits as well; codes differ (shared vs slot-specific); no swap test | accepted | headline restated as "count variables at different sites and codes"; "site (not code)" withdrawn; ADJUDICATION_R1 §4.1's "same variable / interchangeable" wording is superseded by this row |
| Pre-review statements left in bodies (PCOUNT A4, A5; CAUSAL §4; CLASSES "non-count code"; THEOREMS T1a/T2/T3/Consequence) | accepted | corrected in place with strikethrough |
| Own-group range should be 0.65–0.80 | accepted | fixed |
| Necessity removals > 1.0 include collateral damage | accepted | necessity of the count variable "not isolated" |
| GATE "≥ 86 % survives" is mean-ablation-specific; resample drops to 0.37–0.39 | accepted | operator stated everywhere |
| Q3-D3: parent sharing also applies; A2 ties parent lookup on rank, beats it on MAE only | accepted | added to STRUCTURE and synthesis Q30 |
| TRANS over-corrected: restricted-prior model gets 82 % of the both-maps-dependent accuracy | accepted (not re-run) | synthesis Q18: joint use within training pairs, not generalising composition |
| Theorem repairs (OD2 K = 2, AD2, T3) correct | noted | none |
| Frozen blocks unaltered (`git diff --numstat` additions only) | noted | none |

Rejected: none. No round R3 was run; the R2 changes are restatements and caveats, and the one new number not re-run by me
(TRANS 82 %) is marked as reviewer-computed in this table.
