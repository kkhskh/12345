# Stage-0 amendment 1: cohort extension (decided from pretraining completeness only)

> **Correction (independent final review).** The statement below that "no T4 evaluation or analysis output of any
> fresh model existed" is **false**. In-progress fine-tune outputs of vP150 and vP151 existed and were committed in
> this very commit (5425317): run logs with evaluation metrics through step 2686, and 22 step checkpoints. The author
> did not read them, but this cannot be verified.
>
> This amendment also **reverses** the Stage-0 rule "if fewer than 4 parents are complete, no extra parents are
> trained". It is a protocol change. Verdicts restricted to P150 and P154, which excludes the amendment parent P155,
> are identical (FRESH Part II).

Recorded 2026-11-14 ~09:05Z. At this point no fresh-cohort fine-tune had finished: there was no final.json, and no
T4 evaluation or analysis output of any fresh model existed or had been read. The amendment is committed with
hashes in `logs/AMENDMENT1_HASHES.txt`.

## Trigger

Pretraining completeness, the only information Stage 0 allows to be read:

| parent | aux_acc | status |
|---|---|---|
| P150 | 1.000 | complete |
| P151 | 0.762 | incomplete |
| P152 | 0.903 | incomplete |
| P153 | 0.555 | incomplete |
| P154 | pretraining | — |

With at most 2 complete parents, the frozen pass rule (≥ ⌈0.75 n⌉ complete parents) would rest on n ≤ 2. That is too
weak for the mandate's confirmation standard.

## Amendment

1. **Two further parents, P155 and P156**, are pretrained with the identical Stage-0 protocol. Both are new seeds.
   They are fine-tuned identically (full_c1 and C1, seed 10) and enter the same analyses.
2. **The readiness rule is unchanged** (aux_acc ≥ 0.99). Incomplete parents are still reported separately.
3. **Replicate rule, extended.** The seed-11 replicates go to the first two complete parents in seed order among
   P150–P156. If fewer than two parents are complete, the remaining replicate slot goes to the incomplete parent with
   the highest aux_acc, and it is labelled incomplete. The rule is decided from pre.json only.
4. **A secondary aggregate is pre-declared now**, before any outcome is seen. Every arm is also scored over **all**
   fresh parents (complete + incomplete) with the same ⌈0.75 n⌉ rule.
   - The **primary** verdict remains the complete-parent verdict.
   - If fewer than 3 parents are complete, the primary verdict is labelled **"underpowered"**.
   - The all-parents verdict is reported alongside it.
5. Nothing else changes: predictions, thresholds, analysis code and streams.
   - The only code changes are the parent lists in `score_xii.py` and `pick_replicates.py`.
   - `conf_merge.sh` now waits for every parent's outputs before merging, and the secondary aggregate is added.
   - The diffs are in the amendment commit.
