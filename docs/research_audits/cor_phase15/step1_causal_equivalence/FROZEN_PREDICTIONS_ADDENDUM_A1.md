# Frozen predictions, Addendum A1: selectivity of the abstraction subspace (F6)

Committed after Part I and **before any confirmation output was inspected or any confirmation analysis was run**.
- At commit time, confirmation fine-tunes were *training*: `native_train.py` writes its own curve to disk, but none of it
  had been read.
- No DAS or subspace analysis of wP140–143 existed.

Part I is unchanged.

## Development basis (`scripts/subspace_shift.py`, eval_dev)

The measurement is the displacement of the q7 downstream-input state inside the model's learned E-subspace
(L2/q7/k6), under four natural on-support edits:
- **D2:** create q-agreement of a perfect group, so e changes 0 → 1;
- **D2c:** create q-agreement of a non-perfect group, so e is unchanged;
- **CNT:** F45 R2/R3, count 4 → 5 while agreeing, so e changes 0 → 1;
- **CNTc:** F34 R1, count 3 → 4, so e is unchanged.

| model | D2 | D2c | CNT | CNTc | D2c/D2 | CNTc/CNT | D2/CNT |
|---|---|---|---|---|---|---|---|
| xP81 full_c1 | 10.6 | 2.85 | 11.3 | 4.60 | 0.27 | 0.41 | 0.94 |
| yP130 full_c1 | 12.8 | 3.13 | 12.6 | 3.69 | 0.24 | 0.29 | 1.02 |
| yP131 full_c1 | 11.5 | 3.17 | 11.2 | 3.97 | 0.28 | 0.35 | 1.02 |
| xP81 C1 | 3.08 | 2.34 | 2.36 | 2.00 | 0.76 | 0.85 | 1.30 |
| yP130 C1 | 3.04 | 2.99 | 1.88 | 1.86 | 0.98 | 0.99 | 1.62 |
| yP131 C1 | 2.49 | 2.47 | 1.72 | 1.75 | 0.99 | 1.02 | 1.45 |

## F6 — the repaired models' abstraction subspace is selective for eligibility through both input routes; the original's is not

On complete confirmation parents (n as in Part I), with `SHIFT_EVAL=logs/eval_conf10.npz` and subspaces from the frozen
confirmation DAS runs:
- **(a) full_c1:** D2c/D2 ≤ 0.50, CNTc/CNT ≤ 0.60, and 0.70 ≤ D2/CNT ≤ 1.40, in **at least n − 1** parents;
- **(b) C1:** D2c/D2 ≥ 0.60 in **at least n − 1** parents.

## Run (appended to the confirmation queue)

`SHIFT_EVAL=logs/eval_conf10.npz SHIFT_RSUFFIX=_ev62500000 SHIFT_TAG=_conf_<p> scripts/subspace_shift.py <p>:full_c1:0:_eval_conf10 <p>:c1:0:_eval_conf10`,
after both models' DAS runs for parent p.
