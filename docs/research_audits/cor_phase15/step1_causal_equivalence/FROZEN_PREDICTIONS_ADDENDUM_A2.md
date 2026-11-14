# Frozen predictions, Addendum A2: algorithmic learning order (F7)

Committed **before any trajectory analysis of a confirmation model** was run or any confirmation checkpoint was read.
Confirmation fine-tunes were training at commit time. Parts I and A1 are unchanged.

## Development basis (`scripts/traj_alg.py`, eval_dev, 12 log-spaced checkpoints per run)

Closest rule on D2 (create q-agreement of a perfect group), among B / A1 / A2 / A3 / A0:

| run | steps 50, 74, 111, 165, 246, 366 | first B-closest step (stable thereafter, D1 and D2) |
|---|---|---|
| xP81 full_c1 | A1 ×6 | 813 |
| yP130 full_c1 | A1 ×6 | 1211 |
| yP131 full_c1 | A1 ×6 | 4000 |
| yP132 full_c1 | A1 ×5, then A2 | 1211 |
| yP134 full_c1 | A1 ×6 | 1803 |

Investigation IX's NP4 (separable-first, measured by R1 error) was falsified. R1 cannot separate A1 from B, since their
predicted R1 responses differ by 0.002. The D1/D2 families can (distance 1.05–1.12).

## F7 — "perfect-only eligibility first, q-agreement integration later"

In **at least n − 1** complete confirmation full_c1 trajectories (`TRAJ_EVAL=logs/eval_conf10.npz`):
- (a) the closest rule on D2 at steps 50, 74 and 111 is **A1**;
- (b) the closest rule on D2 at step 4000 is **B**;
- (c) the first step at which D2 is B-closest is ≥ 246.

## Run (appended to the confirmation queue, after all four full_c1 runs exist)

`TRAJ_EVAL=logs/eval_conf10.npz TRAJ_MODELS=wP140:full_c1:0:_eval_conf10,…,wP143:full_c1:0:_eval_conf10 TRAJ_TAG=_conf scripts/traj_alg.py`
