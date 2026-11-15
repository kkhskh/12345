# Experimental-integrity review: XIII-B (step1_relational_decision_computation_followup)

Package: `/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup` (paths below are relative to it).
Freeze d7e74a4, addendum 5c7e67f. The review was read-only. Checks ran in `/tmp/review13b_integrity/`:
- `score_copy.py` is the frozen scorer with its output redirected to /tmp.
- `vals.py` dumps per-arm development values.
- `score_fake.py` and `fakelogs/` test the missing-data path.

I did **not** open any seed-21 or `*_conf13b*` output. I only ran `stat` on them, so their atime record stays intact for later auditors.

## Verdict summary

**Chronology and freeze integrity are clean.**
- No seed-21 output and no confirmation output was read before the freeze.
- The freeze was committed and pushed before the confirmation lanes started.
- AQ1 was committed and pushed before its seed-21 fit was produced.
- No frozen file has changed since d7e74a4.
- `eval_13b.npz` was never read before the freeze.

**Scorer-to-queue wiring is correct.** Every source the scorer reads is produced by the frozen queues, under the file names the scorer expects.

**The reporting and statistics layer has real defects. They must be handled in the results write-up, because the frozen scorer cannot be changed:**
- the "diagnostic" label is close to tautological, or wrong (Q1a);
- arms that pass on no-op nulls are not flagged;
- "inconclusive" masks arm-level falsifications;
- units whose filter data are missing are dropped silently.

**The preflight nulls are hand-written numbers, not measured null models.** They verify the scorer's logic only. They do not show that "random or irrelevant representations do not pass" (mandate §9, L526). The NULLPOE values are unrealistic, given math review 1.

**Development coverage was thin, and the freeze record does not say so.**
- Q5 thresholds were set on two late-class models only.
- Q6 thresholds were set on three models.
- Q4 rests on two models, and its decoy set leaves out the two decoys that pass on development data.

## Confirmed

1. **Seed-21 fine-tune outputs were unread before the freeze.**
   - `checkpoints/native_train/uP160…seed21…/final.json` and `curve.json` (created 01:11:18) and the uP163 pair (01:11:02) existed before the freeze (02:12:50).
   - On this relatime ext4 mount, every `final.json` / `curve.json` has atime < mtime at ns resolution, so neither was read after it was written.
   - The training stdout files (`logs/run/a9c5d4ff6dfb.out`, `b6f02414519d.out`, `689d0ffe3a8a.out`, `d100faa4d497.out`, `a60dd606ee7e.out`) still have atime = birth, so they were never read.
   - The first reads of the `step4000.npz` files came from the confirmation jobs: uP160 at 02:13:09 (K1 class_features, COMPUTE_LOG L79); uP163, uP164 and uP165 at 02:40:20–02:40:57 (DV6 addendum, L105).
   - Caveat: atime is not tamper-proof. The evidence is consistent with the author's statement but does not prove it.
2. **`*_conf13b*` outputs.** All were created after 02:13:59, i.e. after the freeze. All are unread: atime < mtime, including `logs/run/` for K1/K2/K3/DV6.
3. **Freeze hashes.** All 24 entries in `logs/FREEZE_13B_HASHES.txt` match both `git show d7e74a4:<path>` and the working tree. `git diff d7e74a4 -- <files>` is empty. Uncommitted changes touch only `ALTERNATIVE_NATIVE_DECISION_ALGORITHMS.md` (development §3b, post-AQ1) and `COMPUTE_LOG.txt`.
4. **Push timing** (GitHub activity API):
   - d7e74a4 pushed 02:12:53Z. K1/K2 started 02:12:54Z.
   - 5c7e67f pushed 02:39:20Z. The AQ1 seed-21 run started 02:39:26Z (L105), and `fit_powermean_conf13b_addendum.json` was created at 02:41:16. **AQ1 came before its output.**
   - `ADDENDUM_A1.md` and `scripts/fit_powermean.py` match `ADDENDUM_A1_HASHES.txt`.
   - Math review 1's scripts (`reviews/review1_scripts/`) use only development models and streams (75_300_000).
   - The confirmation outputs `fit_family_conf13b_P160` and `P161` existed when AQ1 was written (02:14, 02:22), but they are unread.
5. **`logs/eval_13b.npz`.** Created 00:03:52. Its first atime update after creation is 02:12:50, which is the freeze-commit hashing. It was therefore not used in development. Development algdisc used `eval_conf13.npz` (`build_dev13b.py` L45).
6. **Scorer to producer mapping is correct:**
   - `class_features_conf13b_<P>.json` ← CF_TAG `_conf13b_<P>`.
   - `fit_family…` and `l2_components…` likewise.
   - `das_bal_conf13b/<tag>.json` ← `das_bal.py` L62.
   - `adapt2_conf13b/<tag>_b0.6.json` ← `adapt2.py` L105, where `BETA = 0.6` formats as `0.6`.
   - `algdisc_conf13b.json` ← `algorithm_discrimination.py` L149, then `mv` (K1 L16).
   - Every key path used in `predictions_13b.json` exists in the producer outputs (checked on the development outputs, which come from the same code).
   - The scorer reproduces the preflight's development scoring exactly (`/tmp/review13b_integrity/dev_rescore.txt`).
7. **Scorer against spec:**
   - The pass rule is `k >= ceil(0.75 n)` (score_13b.py L53).
   - `min_units` is per prediction, the `parents` filter is at L46, and `expected_class` comes in through `exp:`.
   - The diagnostic rule (L54) matches its docstring.
   - A missing test value makes the arm inconclusive, never a pass or a fail.
8. **Streams.** The main data streams are new: CF 79.0M, FF 79.1M, L2C 79.2M, DB 79.3M/79.4M, AD 79.5M+60 (eval +500k, J-pairs +700k), FP 79.6M, and eval_13b at 21.4M.
9. **The replicates really are fine-tuning replicates.** `full_c1` takes pretrained layers 0–1 plus embeddings, with a fresh layer 2 and readout (seed+7777); `native_train.py` L21–36 and `ftlib.load_model` L22–31. They are described as fine-tuning replicates, not pretraining replicates (`predictions_13b.json` L3; FROZEN doc L10–11). There is one seed per parent and no unit is double-counted.
10. **Development incidents are logged.**
    - The OOM kills appear in COMPUTE_LOG: L6–7, 24/28, 31/33, 46/64, 50/52/67, 70/72/74/75.
    - The skipped runs are logged: L44 (pre-created claims for adapt2 β=1 on uP163/164/165) and L65 (uP164 β=0.6, uP160 β=1).
    - The stopped development lanes are logged at L81, including "das_bal never completed for early-readout models".
    - Every claimed-but-unlogged entry in `logs/claims` matches one of those disclosed pre-created skips.

## Must-fix

These are fixes in reporting. The frozen scorer must not be changed; apply the corrections as declared post-hoc reporting rules, committed before `score_13b.py` is run on the confirmation units.

**M1. "Inconclusive" overrides "fail".**
- At score_13b.py L59, a prediction is inconclusive whenever any arm is inconclusive, even if another arm has failed.
- The preflight shows it happening:
  - NULLNOOP Q3: a fail, b fail, c `0/0`, verdict "inconclusive".
  - NULLNOOP Q6: a fail, verdict "inconclusive".
- With per-arm unit filters (Q3b/c, Q6b/c), an empty class (n = 0) will hide a falsified arm.
- Fix: any failed arm means "not confirmed, arm X falsified", whatever the status of the other arms.

**M2. Missing filter data drops units silently.**
- At L47–50, a filter that evaluates to `None` excludes the unit. It is not added to `missing` and does not trigger inconclusive.
- I demonstrated this: with uP161's l2_components file emptied, uP161 vanished from Q4 and `missing=[]` (`score_fake.py`).
- The development history includes OOM kills of `l2_components` and `das_bal`, so this can happen in the confirmation run.
- Fix: report every `filter_values: null` unit as missing data, and treat the affected prediction as inconclusive.

**M3. The "diagnostic" flag misleads.**
- **(a) Mirror nulls.** In Q1a/b, Q2a and Q3a/b/c the null is the mirror of the test on the same measurement. Whenever the test passes, the null fails automatically, so "diagnostic" carries almost no information.
- **(b) Q1a's null cannot hold.** The null is `POE.R2 >= 0.97`, but the script's POE family gives R² 0.827 even on exact Bayes (`fit_family_dev.json`). The arm will be labelled "diagnostic" although it cannot tell a mixture from normalised log-linear pooling (REVIEW_1_MATH E4: normalised POE R² 0.963–0.974).
  - Q1a must be reported as non-diagnostic against log-linear pooling.
  - Q1's title phrase "not a product-of-experts pooling" must not be claimed from Q1.
  - AQ1 is exploratory and cannot repair Q1.
- **(c) No-op-passing arms are never flagged.**
  - These arms pass on the NOOP or UNIFORM nulls (`preflight_13b.json`): Q4e, Q5b, Q5c, Q6c.
  - Q4e, Q5b and Q5c have no null, so `diagnostic = None` and the counts treat them as diagnostic (L61, L64).
  - Q6c's mirror-style null will normally mark it diagnostic.
  - Report all four as non-diagnostic "absence" arms, as mandate §9 L538 requires.
  - Correct `confirmed_all_arms_diagnostic` accordingly.

**M4. The preflight nulls are hand-written numbers, and the record does not say so.**
- `build_dev13b.py` L51–68 writes the six null units with values the author chose, already knowing the thresholds. Examples: DECOYALL decoy proj 0.85 against a 0.3 cap; Q4e preservation hard-set to slope 1.0 for every null. No null model was run through any measurement script.
- NULLPOE assumes a pooling model would score POE R² 0.99. Math review 1 shows the POE family cannot reach that even for a true normalised pool.
- The preflight therefore tests the scorer's logic, not whether the metrics discriminate the hypotheses.
- The scripts already compute measured no-op/random controls (`l2_components` `noop`; `das_bal` `NOOP`/`RANDOM`), but no prediction uses them.
- Fix: state this plainly in FROZEN §3 and in the results.

**M5. Q4's decoy set leaves out the decoys that succeed on development data. Disclose this next to the verdict.**
- On development EA data, the best-matched decoys (PERM: same mass, entropy and positions; MEAN: source-free) give:
  - transplant proj 0.53/0.69 (PERM) and 0.52/0.66 (MEAN) for uP161/uP165;
  - restoration 0.60/0.70 against a removed level of 0.31/0.33.
- If PERM or MEAN were in Q4c's "no decoy restoration exceeds removed + 0.15" list, Q4c would **fail** on development data.
- Instead, Q4b/c test only TOP/MAG/LOW, and PERM sits in nulls that cannot realistically be met (`fid_E ≥ 0.6`; restoration `proj ≥ 0.85` and `fid_E ≥ 0.85`).
- The development documents disclose the PERM/MEAN numbers (CAUSAL L56–60, NOTES_DEV L21). The frozen prediction and the eventual verdict do not.
- Margins on uP161 are also tight: RESTORE_TOP 0.437 against a cap of 0.463, and TOP proj 0.221 against 0.3. These are thresholds set on n = 2.
- Report Q4 as "specific relative to other heads' rows (TOP/MAG/LOW)", not "specific under matched decoys". Report the PERM/MEAN proj values for the confirmation units as well.

**M6. Development coverage must go into the results, not only COMPUTE_LOG.**
- **Q5 (min_units 3, all 5 units):** thresholds were set on uP161 and uP165 only, both late class. Development das_bal for uP160, uP163 and uP164 never completed (L31, L52, L81). The preflight scored Q5 as "inconclusive 2/5 missing=3".
- **Q6:**
  - Development rows exist for uP161 and uP163, plus uP160 `full`, which was parsed from the printed log of an OOM-killed run (`build_dev13b.py` L39–43).
  - uP160 `early01` (0.791 / slope 0.965, `logs/run/3c86a9c666b7.out`) was not assembled.
  - uP164 was never run.
  - uP165 arrived only after the freeze (02:26:48).
  - Q6b's threshold of 0.2 sits just under the only early-model development value with a margin (uP163 0.288). Q6a's 0.6 is just under uP163's 0.68.
- **Stale promise:** CAUSAL L113 says early-readout das_bal "development rows added … once the OOM-killed runs were repeated". Those runs were stopped and will not be repeated. Correct the sentence.

## Disclosure issues

- **D1. Two "new stream" exceptions.**
  - `class_features.py` L43 draws the natural episodes for `mix_mass`, `strict_mixture` and `share` from the **fixed** seeds 75_700_001/2. These are exactly the development episodes, so Q2's `mix_mass` is not measured on a new stream. That contradicts FROZEN L13–14 ("new streams 79_000_000+"). There is no outcome leakage, because the models are new, but it should be stated.
  - The development script `jpath.py` L13 used `default_rng(79_100_000)`, the same seed as the confirmation FF_STREAM, through a different sampler (C = 2, β = 0.6, versus the native C = 1 sampler).
  - Auxiliary fixed seeds are shared with development by design: l2_components `design` seed 1304, das_bal key 6 and rng 3/9, adapt2 batch rng 5, and algdisc edit seed 55_000_008, which is also the seed of the development `algdisc_eval_conf13`.
- **D2. Filters exclude the hybrid region.** Q3b/c and Q6b/c filter on the unit's own lik_RO1, with a gap between 0.2 and 0.7. This is not circular: lik_RO1 comes from likelihood pairs, while route_Q and the adapt2 slopes are separate measurements. But the gap removes exactly the hybrid units most likely to break a "categorical" law (uP164 development lik_RO1 = 0.518). With min_units = 1, Q3b/c and Q6b/c can each pass on a single unit. Q3's "differs categorically" wording overstates what the test can show.
- **D3. Q2 has low power against its competing hypothesis.** With n = 4 and a 3/4 rule, one switched replicate still passes. If replicates switched at random with p = 0.5, Q2 would pass with probability 5/16 ≈ 0.31. The preflight tests each null unit alone, so it never covers the realistic "1 of 4 switch" mix.
- **D4. The predictions overlap.**
  - Q1b, Q4d and Q5 all bear on E versus A1.
  - Q2, Q3 (filters), Q4 (filter) and Q6 (filters) all depend on the same lik_RO1 measurement.
  - Report "k of 6 confirmed" with that dependence stated. The six are not independent confirmations.
  - Since all units are fine-tunes of the same five parents used for development, the "law" titles (Q1 "every solution class", Q3 "Routing law") are supported only within these five pretrained parents.
- **D5. AQ1 is incompletely specified.** It has no aggregation rule (all units? ceil(0.75 n)?), no min_units, and no rule for missing lik_RO1. Its late threshold of 0.85 sits close to the development minimum of 0.87. It must stay exploratory. The file name `ADDENDUM_A1` collides with the variable name "A1"; consider renaming.
- **D6. The frozen document is incomplete, and there is no uncertainty treatment.**
  - The frozen FROZEN_CONFIRMATION doc still contains the placeholders `FREEZE_SECTION` / `PREFLIGHT_SECTION`.
  - No prediction specifies how to interpret an inconclusive outcome, which mandate §9 requires.
  - No retry or OOM policy was frozen for the confirmation jobs. Because the jobs are deterministic, reruns are harmless if logged.
  - Thresholds are applied to point estimates with no CI. For example, Q6c's acc ≤ 0.768 (0.758 + 0.01) compares against an evaluation SE of about 0.005.
- **D7. Development jobs kept running after the freeze.** CB uP161 β=1, adapt2_dev2 uP165, and DV4 traj_resp all ran after the freeze (L95, L99–100, L106). Their outputs are development data and must not feed into how confirmation verdicts are interpreted. Label them post-freeze development.

## Minor

- `adapt2.py` `full`/`fresh` mark l0/l1 as trainable (L55). They get zero gradient but AdamW weight decay still changes them. Evaluation uses the frozen resid2, so scores are unaffected, but the saved `full` checkpoints have decayed l0/l1, which matters for any later dissection (C3).
- In `l2_components.py` the docstring describes TOP as "largest direct-logit share". The code uses `share` from `head_stats`, which is the direct-logit share. They are consistent; the wording is fine.
- Some preflight null units are untargeted but still confirm predictions: Q1 by NOOP, SEEDSWITCH and DECOYALL; Q4 by POE. These are expected, because those nulls hold the relevant measurements at normal values, and `preflight_13b.json` reports them.
- Three lanes (K1/K3 sharing a queue file) rely on mkdir claims. This works. `bash -c` inherits the loop's stdin, which is harmless here because no job reads stdin.

## Must-fix list (short)

1. M1: "inconclusive" must not mask a failed arm. Any arm fail means the prediction is not confirmed, with that arm falsified.
2. M2: units whose filter data are missing (`filter_values: null`) must be reported as missing data, not dropped silently. This matters for Q4 and Q6b/c.
3. M3: correct the diagnostic labels:
   - Q1a is non-diagnostic against log-linear pooling, because the POE null cannot be met even by Bayes;
   - Q4e, Q5b, Q5c and Q6c pass on no-op nulls and must be reported as non-diagnostic;
   - mirror nulls are near-tautological.
4. M4: state that the preflight nulls are synthetic, author-chosen values. They verify scorer logic only, and NULLPOE is unrealistic.
5. M5: report Q4 with the PERM/MEAN decoy results. On development data they would fail Q4c, so Q4 is specificity relative to other heads' rows only.
6. M6: put the development coverage into the frozen-results document:
   - Q5 thresholds come from 2 late models only, with no early or hybrid development data;
   - Q6 has 3 partial models, with uP160 early01 not assembled and uP164 absent;
   - correct the stale CAUSAL L113 promise.
