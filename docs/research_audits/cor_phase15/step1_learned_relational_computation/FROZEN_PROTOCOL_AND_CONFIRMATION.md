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

**Integrity.** The Part I text above is byte-identical to the frozen commit `4dd9d34`. To verify: `git show
4dd9d34:<path>` and diff up to this heading. Addendum A1 (`FROZEN_PROTOCOL_ADDENDUM_A1.md`) was frozen in `29f7569`. Both
were committed before any confirmation run. The confirmation outputs are `checkpoints/compose_conf/*.json`,
`checkpoints/conf/*.json`, `logs/mech_reader_conf.json` and `logs/predictor_probe_conf.json`. All criteria are computed by
`scripts/conf_evaluate.py`, written before the repair outcomes were seen, and output to `logs/confirmation_results.json`.

**Deviations, all disclosed in `logs/COMPUTE_LOG.txt`.**
1. A scheduling error briefly started the P7 probe before its inputs existed. It crashed on a missing file and produced no
   output.
2. A cloud-environment restart (about 00:35 UTC) killed two in-flight runs, zT1112 orig and zP105 orig, before they wrote
   anything. Both were rerun once with identical commands, as rule I.5 provides.

No run was dropped. Every listed run completed: 10 readers, 1 mechanism analysis, 1 probe and 30 repairs.

| prediction | result | key numbers |
|---|---|---|
| **P1** relational readers learn the relation without m | **confirmed** | 6/6 meet all bounds. Means: KL 0.0008, dm_R1 +0.23, \|E_R1\| 0.099, \|E\| R2∪R3 0.145. Attn under-shoots (E −0.08, −0.12, −0.10); relnet over-shoots (+0.10, +0.13, +0.06) |
| **P2** MLP budget-limited; separable phase first | **confirmed** | (a) mlp 16k E_R1 −0.52, −0.44, −0.39 (3/3 ≤ −0.20). (b) At 1k steps, 9/9 R learners have E_R1 between −1.19 and −1.82. (c) attn O at 1k: −0.03 |
| **P3** attention mechanism | **confirmed** | 3/3 models meet (a), (b) and (c). noperf E_R1 −3.59, −3.70, −3.65; F54o dm +3.35, +3.47, +3.41; ΔF34 −0.003, +0.017, +0.017. Self-only R2/R3 dm ≥ +0.14. m-share 0.93, 1.48, 1.34 vs random −0.04, −0.40, −0.07 |
| **P4** original failure persists | **confirmed** | 8/8 orig E_R1 ≤ −0.20; mean −0.74 (range −0.41 … −1.54) |
| **P5** learned repair without m | **confirmed** | (a) 8/8 paired wins, sign p = 0.004. Mean \|E_R1\| 0.057; decision-level absE ratio 0.22. (b) Every preservation metric improved: \|E\| R2∪R3 1.31 → 0.15; F34 0.42 → 0.10; KL 0.094 → 0.0022; F0/FLOC/FID 0.13/0.32/0.11 → 0.02/0.04/0.02. (c) gadapt mean \|E_R1\| 0.49 vs radapt 0.07 on P100–103 |
| **P6** transfer to T1 | **confirmed** (with a caveat) | 3/4 have \|E_R1\| ≤ 0.25. Preservation passes (R2∪R3 0.30 → 0.15, KL 0.0076 → 0.0021) |
| **P7** decodability does not separate success from failure | **confirmed** | The 3 failed mlps have probe acc 0.990–0.991; successful attn readers 0.83–0.87 |

**Per-lineage repair table (eval_conf8).** Entries are E_R1 / KL / \|E\| R2∪R3.

| lineage | orig | radapt | gadapt | injm (oracle) |
|---|---|---|---|---|
| zP100 | −0.44 / 0.124 / 1.78 | −0.04 / 0.002 / 0.14 | −0.40 / 0.123 | +0.36 / 0.068 |
| zP101 | −0.45 / 0.036 / 0.57 | −0.13 / 0.002 / 0.18 | −0.43 / 0.034 | −0.28 / 0.026 |
| zP102 | −0.67 / 0.111 / 1.53 | −0.03 / 0.002 / 0.16 | −0.67 / 0.111 | |
| zP103 | −0.42 / 0.046 / 0.94 | −0.08 / 0.002 / 0.13 | −0.44 / 0.045 | |
| zP104 | −1.54 / 0.062 / 1.02 | −0.03 / 0.003 / 0.20 | | |
| zP105 | −1.25 / 0.049 / 1.12 | −0.13 / 0.003 / 0.15 | | |
| zP106 | −0.41 / 0.122 / 1.51 | +0.01 / 0.002 / 0.13 | | |
| zP107 | −0.77 / 0.202 / 1.98 | −0.02 / 0.002 / 0.14 | | |
| zT1110 | −0.12 / 0.006 / 0.26 | **−0.34** / 0.002 / 0.15 | | |
| zT1111 | −0.14 / 0.008 / 0.32 | −0.16 / 0.002 / 0.16 | | |
| zT1112 | −0.04 / 0.005 / 0.25 | −0.07 / 0.002 / 0.15 | | |
| zT1113 | −0.12 / 0.011 / 0.39 | −0.01 / 0.002 / 0.14 | | |

**Caveat on P6, which the criterion does not capture.**
- T1 models have only a mild R1 error under the original transfer (mean −0.10).
- radapt does **not** improve R1 there: −0.34 vs −0.12 on zT1110, and about the same on two other lineages.
- radapt does improve R2∪R3 and KL.

P6 asked for |E_R1| ≤ 0.25 plus non-degradation of the *other* families. It was met. A paired R1 comparison with T1 orig
would *not* favour radapt: 1 win in 4. The learned adapter is a large repair where the failure is large (P-count) and
roughly neutral on R1 where it is small (T1).

**Not captured by the criteria for P5 either.** Decision-level |E| on R1 for radapt is still 0.15–0.23. The *mean* R1
error is near zero, but individual R1 responses scatter around Bayes about as much as the composition readers' do.

## Part III — Addenda and corrections after freeze

(None yet.)
