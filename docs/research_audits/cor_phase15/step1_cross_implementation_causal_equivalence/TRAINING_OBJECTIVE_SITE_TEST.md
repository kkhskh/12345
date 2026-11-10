# Training-objective site test (mandate §8; Prediction V)

> **Post-review [R].** Detectability limit added (ME). Pre-review text: commit 8c6363e; adjudication: REVIEW_ADJUDICATION.md.

**Design.** Arm S ("pcsrc", seeds 5, 6, 7): the continuation's P-count pretraining (`train_tasks.train`, task pcount, 3000 steps,
identical optimiser, schedule, sampler, loss, head) with one change — question q's leave-q count targets are read from the
final-layer state of source token 6q + 2(q mod 3) instead of readout token 36+q. Number of supervised positions (6), loss
normalisation, targets, attention (bidirectional) and budget are identical. Then the identical frozen-feature T4 transfer
(2000 steps). Readout arm: existing P-count seeds P2–P4 (same trainer). `scripts/site_train.py`, `checkpoints/*pcsrc*`,
evaluation `scripts/site_eval.py` → `logs/site/site_eval.json`.

| model | pretraining count accuracy | T4 transfer acc | readout site: IIA (floor, ceiling) | source site T_full: IIA (floor, ceiling) |
|---|---|---|---|---|
| S5 | 0.516 | 0.631 | 0.214 (0.199, 0.326) | 0.279 (0.253, 0.431) |
| S6 | 1.000 | 0.720 | 0.346 (0.278, 0.360) | 0.258 (0.223, 0.231) |
| S7 | 1.000 | 0.621 | 0.367 (0.322, 0.459) | 0.286 (0.261, 0.335) |
| P2 | 1.0 (cont.) | 0.708 | 0.642 (0.411, 0.682) | 0.233 (0.202, 0.236) |
| P3 | | 0.720 | 0.732 (0.406, 0.805) | 0.058 (0.063, 0.081) |
| P4 | | 0.711 | 0.609 (0.392, 0.677) | 0.142 (0.142, 0.140) |

(Reference: P-count readout arm 0.704–0.725, T1 0.738–0.742, own-question 0.577.)

**Verdict: Prediction V FAILED**, and partly not evaluable as frozen: when a site's full patch barely beats its mismatched floor
(S6 source 0.231 vs 0.223; P2/P4 source), the floor-adjusted statistic is numerically meaningless.

**What the raw numbers show.**
1. Moving the supervision to a source token does not move a causally used count variable to the source tokens: in S models both
   sites have low ceilings (full-site patch reproduces only 23–46 % of informative natural changes) and the count variable at
   either site barely beats its floor. In the readout arm the readout site has ceiling 0.68–0.81 and the count variable 0.61–0.73.
2. S models' relational computation at the layer-2 input is **not localised at either candidate site** — it must sit in other
   positions (e.g. other questions' tokens) or be spread so that no single site is decisive. One S seed (S6) transfers as well
   as the readout arm (0.720) with no decisive site; two transfer poorly (0.62–0.63), one of which (S7) solved its pretraining
   objective perfectly.
3. The arms are not matched in optimisation success (S5 reached only 0.516 count accuracy), so even the transfer comparison is
   confounded for S5.

**[R: ME]** The test could not detect a variable that moved but stayed distributed: patching all source tokens plus the readout
reaches 0.82–0.99 in the S models.

**Conclusion.** Under a matched change of the supervised position, the site of the causally used count variable does not
follow the supervision; the readout arm's localisation at the readout is not reproduced at a source token. This does not
explain why T1 places its count tuple at source tokens; that question remains open (COUNTEREXAMPLES_AND_LIMITATIONS.md).
Three seeds; one pretraining failure.
