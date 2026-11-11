# Independent-parent replication (mandate §7)

## Design (COHORT_PROTOCOL.md, committed at 981af04 before training)

- 16 fresh runs: T1 parents (`nets.train("T1","M",seed,6000)`) and P-count pretraining (`train_tasks.train("pcount",seed,3000)`),
  seeds 60–67 in each arm, followed by the identical Investigation III T4 transfer (2000 steps). Seeds 60–67 were never used
  before; nothing was loaded from an earlier checkpoint. Original protocols unchanged (pretraining budgets differ by design:
  6000 vs 3000 steps).
- Quality-manipulation variants: for every successful T1 parent, the same transfer with 250 and 500 steps.
- Execution: two single-thread lanes, 2 CPUs, 17:49 → 00:55 UTC (≈ 7 h); timings in `logs/COMPUTE_LOG.txt`. Checkpoints:
  `checkpoints/{pre,ft}_{T1,P}_s{60..67}.npz`, `checkpoints/budget/cT{s}_b{250,500}.npz`.

## Runs

| run | T4 transfer acc | KL(Bayes‖model) | success | run | T4 transfer acc | KL | success |
|---|---|---|---|---|---|---|---|
| cT60 | 0.627 | 0.227 | no | cP60 | 0.652 | 0.228 | no |
| cT61 | 0.739 | 0.010 | yes | cP61 | 0.717 | 0.111 | yes |
| cT62 | 0.741 | 0.008 | yes | cP62 | 0.726 | 0.074 | yes |
| cT63 | 0.740 | 0.021 | yes | cP63 | 0.716 | 0.109 | yes |
| cT64 | 0.742 | 0.012 | yes | cP64 | 0.728 | 0.068 | yes |
| cT65 | 0.741 | 0.014 | yes | cP65 | 0.639 | 0.246 | no |
| cT66 | 0.622 | 0.250 | no | cP66 | 0.671 | 0.188 | yes |
| cT67 | 0.694 | 0.121 | yes | cP67 | 0.725 | 0.090 | yes |

Six successful runs per arm (the frozen minimum; no replacement seeds). Two T1 parents (60, 66) and two P-count runs (60, 65)
failed to transfer; their failure was not anticipated and is part of the record (12/16 successes). Quality is systematically
different between arms: successful T1 runs median KL 0.013, P-count 0.099, overlapping only through cT67.

Budget variants (successful parents): 250 steps KL 0.07–0.28 (median 0.21), 500 steps 0.025–0.23 (median 0.064).

## Replication outcome (FROZEN_PREDICTIONS.md, OUTCOMES)

| statistic | Investigation IV (3 + 3) | development, new episodes (3 + 3) | **cohort (6 + 6)** | cohort all 16 |
|---|---|---|---|---|
| coh_T1 | 0.377 | 0.515 | 0.335 | 0.225 |
| coh_P-count | −0.074 | 0.022 | 0.186 | 0.131 |
| Δ | 0.451 | 0.493 | **0.149** | 0.094 |
| model-level inference | perm p = 0.05 | perm p = 0.05 | perm p = 0.039, CI [−0.047, 0.341] | p = 0.13, CI [−0.07, 0.30] |

- **The primary replication criterion failed**: the direction recurs and the permutation test is below 0.05, but the model-bootstrap
  interval includes 0. The effect is about a third of the original size; the P-count seeds are more mutually coherent (0.19 vs ≈ 0)
  and the T1 parents less (0.34 vs 0.38–0.52) than in Investigation IV.
- Pair structure: within T1, five parents (61–65) correlate 0.29–0.56 with each other; cT67 — the T1 parent with P-count-like
  quality (KL 0.121) — correlates only 0.05–0.19 with them. Within P-count, pairs range −0.10 to 0.39.
- **Quality-matched contrast (P2(b))**: the same T1 parents transferred for 500 steps (median KL 0.064, closer to P-count's 0.099)
  have null-response coherence 0.086, below P-count's 0.186 (Δ_q = −0.10, CI [−0.28, 0.09]). At 250 steps: 0.083. Coherence among
  T1 parents appears only at the full transfer budget, when they approach the exact posterior.

## Precision

With 6 + 6 models the exact permutation distribution has 924 relabellings (resolution 0.001) and the model bootstrap interval for
Δ has half-width ≈ 0.19. An effect of the Investigation IV size (0.45) would have been detected; the observed 0.15 is within the
interval of "no curriculum effect". No population-level statement is made beyond these runs.
