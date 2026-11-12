# Frozen protocol and confirmation (Investigation VIII)

## Part I — Protocol (frozen before any confirmation outcome was observed)

**Freeze statement.** This Part I was written and committed, together with `logs/PROTOCOL_HASHES.txt`, before any
confirmation-phase training run started. At that time no model had been evaluated on `logs/eval_conf8.npz`. The Stage-0
lineages (P-count seeds 100–107, T1 seeds 110–113) had been pretrained under `STAGE0_COHORT_FREEZE.md` before any
Investigation VIII hypothesis existed. Only their pretraining loss logs had been read. Their representations had not been
probed, and no repair or downstream model had been trained on them. The method choices below were made only on
development material: Investigation VI and VII checkpoints xP80, xP81, xT70, and `eval_dev.npz`. Part I is not edited
after the freeze. Any later correction goes in Part III as an addendum.

### I.1 Material

- **Evaluation set.** `logs/eval_conf8.npz` was built by `scripts/build_eval_conf8.py`. It uses stream **18_900_000**,
  which was never used in development, and Investigation VII's `r1lib.build_eval` unchanged. It has 6000 natural
  episodes plus all edit families (F45 R1 n = 456, R2 n = 88, R3 n = 778; F34; F0/FLOC/FID; S; F54o). Its hash is in
  `logs/PROTOCOL_HASHES.txt`. Exact Bayes evaluated as a model gives max |E| = 0.
- **Confirmation models.**
  - Composition readers: seeds **10, 11, 12**, which are new; development used seeds 0 and 1.
  - Condition Z repairs: Stage-0 lineages `zP100`…`zP107` and `zT1110`…`zT1113`.
- **Code.** `scripts/compose_train.py` and `scripts/mech_reader.py` run with `PHASE=conf`. `scripts/repair_train.py` runs
  with PHASE `conf`. All settings stay at their development values: steps, learning rates, data-stream formula
  `1_000_003·(seed+1)+7919·4+93` for repairs, and `19_000_000+1000·seed` for readers.

### I.2 Runs (intention to treat: every listed run is reported, including failures and crashes)

| block | runs |
|---|---|
| C-A composition | attn R label 16k ×{10,11,12}; relnet R label 16k ×{10,11,12}; mlp R label 16k ×{10,11,12}; attn O label 16k ×{10} |
| C-M mechanism | `mech_reader.py` on the three attn R conf readers |
| C-Z P-count | orig ×{P100…P107}; radapt ×{P100…P107}; gadapt ×{P100…P103}; injm ×{P100, P101} (oracle reference only) |
| C-T T1 | orig ×{T1110…T1113}; radapt ×{T1110…T1113} |

All repairs train for 4000 steps on generator labels. `radapt` is the learned relational adapter: a one-block group
self-attention reader over the raw per-group ingredients (counts, q-reports, group id). It has **no m and no combined
eligibility flag**. Its output is zero-initialised and added at the readout token before the fresh downstream block.
`gadapt` is the matched generic adapter on hashed noise features. `injm` adds Investigation VII's hand-computed m and
serves as the oracle reference; it is not a candidate.

### I.3 Metrics (as in Investigations VI–VII)

- **E** = Δm_M − Δm_B on centred logits, per regime. The headline is mean E and mean |E| on F45 R1. |E| R2∪R3 is the
  n-weighted mean of |E| over R2 and R3.
- **Distance-to-Bayes preservation**: F34 R1 |E|; F54o |E|; KL(Bayes‖model) on natural decisions; F0/FLOC/FID mean |E|.
- **Ties.** Argmax is used for no criterion. Many R2/R3 after-edit states are near-tied.
- **Overshoot rule.** A repair counts as "full" on R1 only if |E_R1| is small. A positive E (overshoot) is an error just
  as a negative one is. The response sign alone is never sufficient.
- **Model-level inference.** The unit is a trained model, not a decision. Paired comparisons use the same parent lineage.
  Sign tests are exact binomial, one-sided in the predicted direction. With n = 8 a sign test needs 7/8 or 8/8 for
  p ≤ 0.035. Bootstrap CIs over decisions are reported descriptively only.

### I.4 Primary predictions

- **P1 (learnability without m, relational readers).** For attn R and relnet R (6 conf models), at 16k steps:
  KL ≤ 0.002, dm_R1 > 0, |E_R1| ≤ 0.25 and |E| R2∪R3 ≤ 0.25 in **at least 5 of 6** models. Also, the across-model mean
  of each quantity meets the same bound.
- **P2 (budget-limited MLP and a separable phase first).**
  - (a) mlp R label 16k has E_R1 ≤ −0.20 in **at least 2 of 3** seeds.
  - (b) At 1k steps, E_R1 ≤ −0.50 in **at least 8 of 9** R readers.
  - (c) attn O seed 10 has E_R1 ≥ −0.20 at 1k steps.
- **P3 (mechanism of the learned attention reader).** In **at least 2 of 3** attn R conf readers, all of the following hold:
  - (a) Removing attention to eligible partner groups ("noperf") makes E_R1 ≤ −1.0 and F54o dm ≥ +1.0, while F34 R1 dm
    changes by less than 0.30 in absolute value.
  - (b) Self-only attention makes dm on both R2 and R3 ≥ −0.5. Base dm on R2/R3 is about −2.
  - (c) An m-subspace interchange on R1 pairs carries a larger share of the effect than a random subspace of equal
    dimension, by at least 0.5.
- **P4 (the original failure persists in new lineages).** On zP100…107, orig E_R1 ≤ −0.20 in **at least 6 of 8** lineages,
  and the mean E_R1 ≤ −0.20.
- **P5 (learned selective repair without hand-computed m).** Paired comparisons on zP100…107:
  - (a) |E_R1|(radapt) < |E_R1|(orig) in **at least 7 of 8** lineages, and mean |E_R1|(radapt) ≤ 0.25. Mean |E| R1
    absolute error (decision-level, mean of absE) for radapt ≤ 0.6 × that of orig.
  - (b) Strict preservation, per mean over lineages, with radapt compared to orig:
    - |E| R2∪R3 ≤ orig + 0.05;
    - F34 R1 |E| ≤ orig + 0.05;
    - KL ≤ orig;
    - each of F0/FLOC/FID mean |E| ≤ orig + 0.05.
  - (c) Specificity: the generic adapter does not repair. Mean |E_R1|(gadapt) ≥ mean |E_R1|(radapt) + 0.20 on P100–103.
- **P6 (transfer of the repair to the T1 panel).** On zT1110…1113: |E_R1|(radapt) ≤ 0.25 in **at least 3 of 4**, with the
  same preservation rules as P5(b) against T1 orig. The sign and size of T1 orig's E_R1 is reported descriptively, with
  no prediction.

### I.4b Development basis for the thresholds (xP80 / xP81, eval_dev, 4000 steps)

| spec | xP80 E_R1 / KL | xP81 E_R1 / KL |
|---|---|---|
| radapt | −0.015 / 0.002 | −0.286 / 0.002 |
| lca (MLP adapter on ingredients) | −0.277 / 0.006 | (deferred) |
| gadapt | −0.889 / 0.102 | (deferred) |
| auxm | −1.097 / 0.118 | (deferred) |
| orig (Investigation VII C1, 12k steps) | −0.698 / 0.072 | — |

The radapt bound of 0.25 was set to cover both development lineages. A bound of 0.15 was considered first and then
abandoned after the xP81 result, before the freeze.

### I.5 Safeguards

- Rare regimes: R2 n = 88 in eval_conf8. R2 and R3 are pooled for the preservation criterion, and R2 is reported
  separately.
- Support: all regimes are drawn from natural episodes of the generator; no off-support edits are added.
- Internal interventions (P3) are reported as interventions on the reader. Each intervention is defined with the
  generator-side eligibility (an analysis tool); none is used by the model.
- Confirmation models are not used to choose any method, threshold or metric. If a run crashes, it is rerun once with an
  identical command and the crash is reported.

### I.6 What a positive P5 would and would not show

It would show that a *learned* module, given the raw evidence and the frozen P-count state, can supply what the one-block
downstream lacks, selectively and without m. It would **not** show that the P-count representation itself contains an
accessible relation. The adapter reads raw ingredients, which are sufficient statistics, so P5 belongs to the
"count+report injection made learnable" family. Development shows the ingredients are not enough unless the module that
reads them is relational: the MLP adapter `lca` reaches E −0.28. The developmental evidence that the frozen state does not
make the relation accessible to the downstream block is that `auxm` (E −1.10) and `orig` fail.

## Part II — Confirmation results

(Filled in after the runs; see below.)

## Part III — Addenda and corrections after freeze

(None yet.)
