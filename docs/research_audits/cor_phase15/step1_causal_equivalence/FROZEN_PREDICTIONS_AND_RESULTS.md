# Frozen predictions and results (Investigation X)

## Part I — Protocol, frozen before any confirmation model was trained or evaluated

**Freeze statement.** This Part I and `logs/PROTOCOL_HASHES.txt` were committed before any T4 fine-tune, evaluation,
probe or intervention on the confirmation parents wP140–wP143.

- At the freeze, wP140 and wP141 had finished Stage-0 pretraining, and wP142/wP143 were still pretraining.
- Only `pre.json` (pretraining count accuracy) had been read: wP140 = 1.0, wP141 = 1.0.
- The **completeness covariate** for wP142/143 is read from `pre.json` when pretraining finishes, before their T4 runs.
  A parent with pretraining count accuracy < 0.99 is analysed and reported but **excluded from F1–F5 denominators**, as
  pre-specified here, because its information is missing (Investigation IX, CP1/CP2).
- All method choices, sites, variables, dimensions and thresholds below were set on development material only:
  - the five Investigation IX full_c1 models (xP81, yP130, yP131, yP132, yP134) and their C1 counterparts;
  - `eval_dev.npz`;
  - DAS evaluation stream 61_500_000.

### I.1 Material

- **Confirmation parents.** wP140–wP143, new seeds (`STAGE0_COHORT_FREEZE.md`).
- **Evaluation set.** `logs/eval_conf10.npz`, stream **20_900_000**, never used before. DAS confirmation pairs come from
  stream **62_500_000** (`DAS_EVAL_STREAM`), never used before.
- **Models per parent.**
  - `native_train.py <p> full_c1 4000 0 label logs/eval_conf10.npz`: full fine-tuning, the repaired model.
  - `native_train.py <p> c1 4000 0 label logs/eval_conf10.npz`: the original frozen-prefix transfer.
- **Measurements per parent.** All are listed in `logs/queue_conf.txt`; the code is frozen by hash.
  1. `algorithm_discrimination.py logs/eval_conf10.npz <p>:full_c1:0:_eval_conf10 <p>:c1:0:_eval_conf10` (ALG_TAG = _<p>).
  2. `das.py` with DAS_EVAL_STREAM = 62500000:
     - full_c1 E at L2/q7/k6;
     - full_c1 S at L2/q7/k2;
     - full_c1 E at L2/ro/k6;
     - full_c1 E at L2/src6/k6;
     - c1 E at L2/q7/k6.
  3. `das_eval.py` with DAS_EVAL_STREAM = 62500000, for full_c1 and c1 at L2/q7/k6.

### I.2 Predictions (counts over complete confirmation parents; n = number of complete parents, expected 4)

**F1 — functional algorithm identification (behavioural equivalence to Bayes under discriminating on-support edits).**
- For each full_c1 model, among {B, A1, A2, A3, A0}, the closest rule is **B** on all of D1, D2, F45_R2 and F45_R3, and
  the mean distance to B is ≤ 0.40 on D1 and on D2.
- For each C1 model, the closest rule on D2 is **not** B.
- Both hold in **at least n − 1** parents.

*Development basis.* Full_c1 models were B-closest on D1/D2/R2/R3 in 5/5, with D1/D2 distances 0.10–0.27. C1 models
were not B-closest on D2 in 5/5.

**F2 — a shared causal abstraction (eligibility vector E) in independently trained models.** For full_c1, E-interchange at
L2/q7/k6 gives IIA_argmax ≥ 0.85 and KLcf ≤ 0.15, and the random-subspace IIA ≤ 0.60. Holds in **at least n − 1**
parents.

*Development basis.* IIA 0.881–0.967 and KLcf 0.017–0.122 in 5/5; random 0.48–0.50.

**F3 — the abstraction has the per-group, not pooled, signature (Theorem X1).** For full_c1, the S-interchange IIA (L2/q7/k2)
is ≤ the E-interchange IIA − 0.20. Holds in **at least n − 1** parents.

*Development basis.* xP81 S 0.55–0.58 vs E 0.97 (more development S runs were queued; whatever finished before the freeze
is recorded in Part II).

**F4 — discriminating counterfactuals separate repaired from unrepaired models.** Use the pairs where the Bayes and
separable counterfactual decisions differ (`das_eval.py`, "disc").
- full_c1 IIA_B ≥ 0.90;
- for the same parent, C1 IIA_B ≤ full_c1 IIA_B − 0.10;
- KL_B(C1) ≥ 2 × KL_B(full_c1).

All three hold in **at least n − 1** parents.

*Development basis.* full_c1 0.948 / 0.962 / 0.960 vs C1 0.718 / 0.822 / 0.635; KL 0.054 / 0.041 / 0.036 vs
0.304 / 0.181 / 0.396.

**F5 — distributed localization (H2).** For full_c1, the E-interchange IIA restricted to the readout token alone
(L2/ro/k6) and to the 6 source tokens alone (L2/src6/k6) are **each** ≤ the q7 IIA − 0.05. Holds in **at least n − 1**
parents.

*Development basis.* xP81 ro 0.757, src6 0.856, q7 0.967.

### I.3 Rules

- **Intention to treat.** Every listed run is reported. A crashed run is rerun once with its identical command, and the
  crash is disclosed.
- **Primary behavioural metrics** for every model: per-episode absE (R1, R2∪R3, F34), signed E_R1, KL, accuracy.
- **What a failure means.** A failed F2/F3/F5 rejects the tested abstraction or site *under these procedures*. By Theorem
  X1 and its identification caveat, it does not show that no common computation exists.
- **Development-only.** These are not covered by Part I:
  - seed-replicate fine-tunes;
  - checkpoint (learning-trajectory) analyses;
  - head/MLP analyses;
  - the B3 follow-ups.

## Part II — Results

(Filled in after the runs.)
