# Stage-0 freeze: Investigation VIII confirmation lineages

This freeze was committed before any Investigation VIII hypothesis, method or development result existed. It covers only the
pretraining of new lineages, which does not depend on any outcome.

**Lineages.**
- **P-count, seeds 100–107:** `train_tasks.train("pcount", seed, 3000)`.
- **T1 reference panel, seeds 110–113:** `nets.train("T1", "M", seed, 6000, K=3, arch="tf")`.

Both use the original protocols unchanged. No seed is shared with Investigations IV–VII (60–67, 70–77, 80–87, 90–97), and no
parameter is inherited. Script: `scripts/cohort8_pretrain.py`, run by `scripts/pretrain_lane.sh` on a single lane, with every
start and end logged to `logs/COMPUTE_LOG.txt`.

**What may happen before the full protocol is frozen.**
- Pretraining runs, together with its built-in auxiliary evaluation (`pre.json`).
- No T4 transfer, T4 evaluation, reader, probe, mechanism or repair analysis touches these lineages until
  `FROZEN_PROTOCOL_AND_CONFIRMATION.md` is committed.

**Failed runs.** Every failed pretraining run is kept. Inclusion rules are set in the full protocol, before any T4 outcome is seen.
