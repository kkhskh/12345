# FREEZE X amendment: registered after mathematical review A, before any X1 confirmation output was inspected

**Timing.** Seeds 3 and 4 had finished training when this was written: their checkpoint directories contain 70 files each, which was checked by file count only. No checkpoint of these models has been loaded, and neither `score_X.py` nor any other evaluation has been run on them.

**Why.** Review A (`reviews/REVIEW_A_MATHEMATICS.md`) made three relevant findings:
- **F1 (major).** X1a cannot separate its prediction from a competitor: joint learning of relation and evidence at a small logit scale. Bayes logits scaled by t ≤ 0.1 give G = t, A ≈ 0.71·t and accuracy 0.90, which satisfies every X1a clause.
- **F4/F5.** A reaches 0.71 under pure gate use, so X1b, as frozen, mixes agreement use with gate use.

**The frozen X1a and X1b and their scorer are not changed.** They are scored and reported exactly as frozen, with the review's criticism attached. This amendment adds two criteria, registered now.

## Criteria

**X1a′.** At least 4 consecutive snapshots at or before step 1500 with:
- 0.75 ≤ accuracy ≤ 0.83;
- NLL ≤ 0.50;
- A_p ≤ 0.3;
- G ≤ 0.1.

The accuracy ceiling excludes the small-scale Bayes competitor, which has the Bayes argmax and accuracy ≈ 0.90. The NLL clause excludes near-uniform outputs. The exact order-1 optimum has accuracy 0.801 and NLL 0.462.

**X1b′.** Some snapshot with G ≤ 0.1 and A_p ≥ 0.5.

**A_p, the partial agreement index.** It is the coefficient on the exact gate-free rule's swap response in the joint least-squares regression of the model's swap response on (gate-free response, Bayes response), over swaps where the gate-free rule changes. Its exact references on this stream (`logs/score_X_amend_refs.json`, `scripts/x_amend_refs.py`):

| rule | A_p |
|---|---|
| order-1 optimum | 1·10⁻¹³ |
| order-2 optimum | 0.962 |
| Bayes | −2·10⁻¹⁵ |
| gate-free rule | 1.000 |
| Bayes scaled by 0.1 | ≈ 0 |

A_p therefore measures pairwise-agreement use separately from gate use.

**Unchanged from FREEZE_X:** models, stream 98_500_000, the competence rule, and the verdict logic (CONFIRMED if both models are competent and satisfy the criterion; FALSIFIED if a competent model fails; INCONCLUSIVE otherwise).

**Disclosed asymmetry.** This amendment was designed after seeing the review, which used development data only (development seed 1, steps ≤ 400). The amended criteria are not tuned to any confirmation result.

**Scorer.** `scripts/score_X_amend.py`, fail-closed. It checks both `FREEZE_X_HASHES.txt` and `FREEZE_X_AMEND_HASHES.txt`. The latter adds `tasks_ext.py`, `train_tasks.py` and `fwd3.py`, following review A's hash-coverage finding, and also `FREEZE_X.md`. Preflight results are in `logs/preflight_X_amend.json`.
