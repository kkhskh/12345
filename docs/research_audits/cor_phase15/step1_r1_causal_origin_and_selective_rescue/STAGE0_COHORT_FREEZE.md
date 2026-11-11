# Stage-0 freeze: generation of the confirmation P-count lineages

Committed **before** any Investigation VII hypothesis, repair or analysis was selected, and before any development result of
Investigation VII existed. Its purpose is to let the slow, outcome-independent pretraining of new parents run in parallel with
development. It does this without letting development results influence which parents exist.

## What is frozen here

- **Parents.** 8 new P-count parents, seeds **90–97**. Pretraining is the unchanged original protocol:
  `train_tasks.train("pcount", seed, 3000)`, AdamW warmup-cosine, batch 256, fresh samples.
- **Independence.**
  - No seed is shared with Investigations IV–VI. Development parents use T1 seeds 70–77 and P-count seeds 80–87; Investigation V
    used 60–67; earlier investigations used lower seeds.
  - No parameter is inherited from any development parent.
  - The data streams are seed-specific.
- **Script and lanes.** The script is `scripts/cohort7_pretrain.py`, run in lane order by `scripts/pretrain_lane.sh`. Every
  start and end is logged to `logs/COMPUTE_LOG.txt`.
- **What runs before the full freeze.**
  - The only computation run on these parents before `FROZEN_CONFIRMATION_PROTOCOL.md` is committed is pretraining itself
    and its built-in auxiliary count evaluation (`pre.json`).
  - No T4 transfer, no T4 evaluation, no reader, mechanism or repair analysis is run on them until then.
- **Failed pretraining runs.** A crashed or non-finite pretraining run is kept and reported. The success rule is fixed in the
  later protocol, before any T4 outcome is seen.
- **T1 lineages.** None are created at Stage 0. Whether a T1 comparison panel is added is decided in the full protocol, before
  any evaluation.

This file is hashed in `logs/STAGE0_HASHES.txt` together with the two scripts.
