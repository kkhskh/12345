# Prospective predictions, Addendum A1: native fine-tuning (frozen before its confirmation runs)

Committed after Part I and before any `full_c1` run on yP130–yP135. At commit time, no `checkpoints/native_train/yP13*_full_c1*` existed.
Part I is unchanged.

## Development basis (xP81, complete parent; eval_dev)

**Training.** Full fine-tuning in the original architecture (`full_c1`: pretrained layers 0–1 trainable, fresh layer 2,
4000 steps, labels).

| run | KL | absE R1 | E R1 | absE R2∪R3 | absE F34 |
|---|---|---|---|---|---|
| full_c1 | 0.001 | 0.157 | −0.108 | 0.127 | 0.096 |
| C1 (frozen 0–1) | 0.042 | 0.584 | −0.550 | 1.225 | 0.339 |

The full_c1 trajectory (12 checkpoints) first deepens the R1 error to E_R1 −0.58 at step 366, while KL falls to 0.07. It
then corrects gradually.

**Patching** (`scripts/mech_native.py`, full-forward token patching at the downstream input). Margin responses on R1
pairs:

| model | quantity | value |
|---|---|---|
| full_c1 | full response | +0.19 |
| full_c1 | patch only the other two groups' source tokens at q | **+0.30** |
| full_c1 | patch everything except those 4 tokens | **−0.57** |
| full_c1 | R2/R3 kept by the complement patch | 64–76 % |
| C1 | patch other groups' source tokens | share 0.08 |

## Predictions (on the complete confirmation parents yP130, yP131, yP132, yP134)

**NP1 (native repair).**
- absE_R1(full_c1) < absE_R1(C1) in **4 of 4**, paired.
- |E_R1| ≤ 0.20 in **at least 3 of 4**.
- KL ≤ 0.005 in **at least 3 of 4**.

**NP2 (causally identified cross-group pathway).** In at least 3 of 4 full_c1 models, all of:
- patching only the other groups' source tokens at q gives a mean R1 margin response > 0;
- patching everything except those 4 tokens gives a mean R1 response < 0, below the full response;
- the n-weighted R2∪R3 share of that complement patch is ≥ 0.5.

**NP3 (no such pathway in the frozen-prefix original).** In at least 3 of 4 C1 models of the same parents, the R1 share of
the others_src patch has |share| < 0.25.

**NP4 (separable first, then cross-group).** In at least 3 of 4 full_c1 trajectories:
- the minimum E_R1 over checkpoints is ≤ −0.30 and occurs at step ≤ 1000;
- the final E_R1 exceeds that minimum by ≥ 0.20.

## Runs

`logs/queue_conf_full.txt`:
- `scripts/native_train.py <p> full_c1 4000 0 label logs/eval_conf9.npz`, then
- `MECH_EVAL=logs/eval_conf9.npz scripts/mech_native.py <p> full_c1 4000 0 _eval_conf9`,

for p ∈ {yP130, yP131, yP132, yP134}.

## Code change disclosure

`scripts/mech_native.py` was modified after the Part I freeze:
- support for fine-tuned prefixes (`full_c1`);
- added patch sets: others_src, others_src_q, all_but_others_src_q.

The c1 code path and the CP5 quantities are unchanged. Its new hash is recorded with this addendum.
