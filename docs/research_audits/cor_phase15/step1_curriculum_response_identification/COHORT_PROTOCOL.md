# Independent cohort protocol (committed before any cohort training started)

This file fixes the confirmation model set before training. The analysis protocol (estimands, intervention families, tests)
is frozen separately in FROZEN_PREDICTIONS.md **before any cohort output is inspected**; training runs while that protocol is
written, and no cohort evaluation number is read until the freeze commit exists. (Mandate §8 asks for freezing before
inspecting outcomes; the hyper-parameters below are the unchanged original protocols, so nothing is selected from new results.)

- **Arms.** T1 (original T1 parent protocol: `nets.train("T1","M",seed,6000,K=3,arch="tf")`) and P-count (original protocol:
  `train_tasks.train("pcount",seed,3000)`), each followed by the identical Investigation III T4 frozen-feature transfer
  (2000 steps; layers 0-1 + embeddings frozen; layer 2 reset). Code: `scripts/cohort_train.py`, queue `scripts/cohort_lane.sh`.
- **Seeds.** 60, 61, 62, 63, 64, 65, 66, 67 in both arms (16 runs). No earlier package used seeds 60-67 for any model.
  Fresh random initialisation; nothing is loaded from earlier checkpoints.
- **Known protocol difference (kept on purpose, it is part of each curriculum).** Pretraining budget: T1 6000 steps,
  P-count 3000 steps (as in every earlier model of each arm). Both arms then use the same transfer.
- **Success rule (frozen now).** A run is *successful* if its T4 transfer accuracy (20,000 episodes, stream 6_300_000) is
  ≥ 0.6595, the transfer-success threshold used since Investigation II. Primary analyses use successful runs; every run is
  reported, and the primary statistics are also reported on all runs.
- **Replacement rule.** If an arm has fewer than 6 successful runs, seeds 68, 69, … are added in order until 6 succeed or
  compute ends; no run is dropped for any other reason.
- **Development models (not confirmation units).** Investigation IV's P2-P4, T2, T10, T11 (and P0, P1, T0, T1).
- **Execution.** Two lanes, one thread each (`XLA_FLAGS` in the queue script), on 2 CPUs.
