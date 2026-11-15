# XIII-C confirmation population (declared BEFORE any confirmation run is launched; predictions are frozen later, before any output is opened)

Declared 2026-11-15 (UTC) at the time of the commit that adds this file. Training the confirmation runs before the prediction freeze is allowed by the
protocol on one condition: no output of these runs (checkpoints, logs, metrics, loss lines) may be opened or analysed before the freeze commit.
The scorer reads them only after the freeze. The run logs are written to logs/run/<md5>.out and are not read before the freeze.

All runs: ft_factorial.py (full_c1 protocol, 4000 steps, separated randomness), parents from the XIII cohort; none of these (parent, L, W, D)
combinations exists in development.

| id | parent | L2 init | WA init | data | intervention | role |
|---|---|---|---|---|---|---|
| C1 | uP161 | 20 | 20 | 23 | - | R1 crossed cell (diagonals: XIII seed 20 = mixture, XIII-B seed 23 = hybrid) |
| C2 | uP161 | 23 | 23 | 20 | - | R1 crossed cell |
| C3 | uP163 | 21 | 21 | 22 | - | R1 crossed cell (diagonals: seed 21 = hybrid, seed 22 = early) |
| C4 | uP163 | 22 | 22 | 21 | - | R1 crossed cell |
| C5 | uP163 | 22 | 21 | 21 | - | R1b L2-init vs WA-init split |
| C6 | uP163 | 21 | 22 | 22 | - | R1b split |
| C7 | uP160 | 21 | 21 | 21 | ro0 for 600 steps | R2a: early-readout suppression in an early-destined run (reference: XIII-B uP160 seed 21) |
| C8 | uP161 | 21 | 21 | 21 | l2q for 600 steps | R2b: mixture-head suppression in a mixture-destined run (reference: XIII-B uP161 seed 21) |
| C9 | uP160 | 23 | 23 | 23 | l2q for 600 steps | R2b (reference: XIII-B uP160 seed 23) |

Measurement streams for confirmation analyses are fresh (88_xxx_xxx), and no development stream is reused.
