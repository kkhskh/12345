# Stage 0: confirmation parents, frozen before any Investigation X hypothesis was tested

The P-count lineages **P140–P143** were started on 2026-11-13 with `scripts/cohort10_pretrain.py`. It is the
Investigations VIII–IX Stage-0 script with only the seeds changed. Pretraining began before any Investigation X
analysis ran.

**What may be read before the freeze:**
- `pre.json` and the pretraining log: the count-accuracy completeness covariate;
- the fact that a T4 fine-tune finished.

**What may not be read or computed before `FROZEN_PREDICTIONS_AND_RESULTS.md` Part I is committed with hashes:** any T4
evaluation, probe, intervention or mechanism measurement on these lineages.

**Independence.** These are new initialisation seeds and new pretraining data streams. They are independent of
everything else in the program.

**Development material for Investigation X**, whose T4 outcomes are already known:
- the five full_c1 fine-tunes of Investigation IX (xP81, yP130, yP131, yP132, yP134);
- their C1 counterparts;
- any new fine-tunes of those parents.
