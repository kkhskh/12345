# Review C: experimental integrity (Investigation XIV)

**Reviewer.** An independent agent reviewer. This is not human peer review.
**Scope.** Freeze chronology and hashes; whether runs and measurements are complete; restorations and the container restart; whether the scorers are complete and fail closed, and whether they reproduce; statistical dependence and labelling; execution logs and reproduction tooling; whether the reports quote the score files correctly.
**Method.** I modified no existing file. All tests ran on copies under `/tmp/revc/` (a copy of `logs/`, copies of `checkpoints/gate/s3` and `s4`, and a package copy whose sibling packages are symlinked). I did no training.

## Verified (no finding)

**Hashes.**
- I recomputed every entry of the three freeze hash files at the freeze commits (`git show <commit>:path`) and on the working tree. All 9 + 5 + 7 entries match at `aca3b6c`, `5d3c9dd` and `b019ca9`, and also at the scoring commits `1ec55aa` and `18e0b73`.
- No frozen file changed between its freeze and its scoring.

**Chronology.**

| time (UTC) | event |
|---|---|
| 17:07:44 | FREEZE I commit |
| 17:07:54 | first Stage I training job |
| 18:41:47 | FREEZE X commit |
| 19:06:35 | GATE s3/s4 `hist.json` and step 4000 written |
| 19:33:25 | X amendment commit |
| 19:33:34 | X1 scoring starts |
| 19:46 / 19:59 | `score_X.json` / `score_X_amend.json` mtimes |
| 20:05:49 | commit `1ec55aa` |
| 20:06:04 | exploratory `g_mech` on s3/s4 begins (after X1 scoring) |
| 23:07–23:08 | Stage I sham validity (confirmation stream) |
| 23:41:59 | `score_I` run |
| 23:42:00 | commit `18e0b73` |

- The amendment was committed before any X1 scorer ran.
- No Stage I confirmation measurement predates FREEZE I.
- The preflights ran while training was incomplete, so P0 refused on the real state; they did not score the confirmation data. X preflights ran at 18:36 and 19:27; s3/s4 had no `hist.json` until 19:06; P0 of `preflight_X_amend` uses a synthetic root.

**GATE snapshot integrity.** The 132 s3/s4 snapshots that were accidentally committed in `5d3c9dd` (steps 25–2500) are bitwise identical to the files scored now. Snapshots 3000–4000 have no hash at freeze time.

**Completeness.**
- All 38 runs in `I_design.json` have `stop_done.json` or `final.json`.
- All 178 design measurements occur exactly once in `unit_metrics_I.jsonl`; there are no extras.
- All 93 frozen queue lines (38 train + 55 measure) were executed verbatim. The claim key is the md5 of the line, and each matches its `start` line in COMPUTE_LOG.
- All 93 jobs ended with rc = 0.
- Claims are 1:1 with job ids and with the `logs/run/*.out` files.

**Restart (~22:34Z).**
- Exactly the two interrupted jobs (`694788dad682`, `432535ecadb0`) were started twice, with claim directories recreated at 22:35:07. Each has exactly one `end`, with rc = 0.
- Their `.out` files show "resumed at step 3000" and "resumed at step 2000".

**Restored checkpoints.** All 48 entries of `I_restored_checkpoints.txt` match the files on disk.
- 28 of them (steps 100/300/600, and the four step-2000 files) are byte-identical to the off-repository backup `/home/claude/x13e_excluded_ckpt/checkpoints/ft/`. In that backup they keep their original XIII-E mtimes; the step-2000 files are from 11:27–13:36Z.
- The other 20 (steps 150 and 1211, and the four step-4000 files) match `step1_reroll_and_critical_period_validation/MANIFEST.sha256`.

**Baseline replays.** COMPUTE_LOG reported 5 of 6 checks done. I checked all 6 step-matched baseline replays (`*_stop1146_br546_XIVbref/step600.npz`) against the original step-600 baselines: **6/6 identical** (SHA-256 equal, max |diff| 0.0).

**Scorer reproduction.**

| scorer, on copies | runtime | output | verdicts |
|---|---|---|---|
| `score_I.py` on a copy of `logs/` | — | bitwise identical to `logs/score_I.json` (sha256 `757fb25f…`) | — |
| `score_X.py` on copied s3/s4 → `/tmp` | 13 min | bitwise identical to `logs/score_X.json` | X1a CONFIRMED, X1b FALSIFIED |
| `score_X_amend.py` → `/tmp` | 12 min | bitwise identical to `logs/score_X_amend.json` | X1a′ CONFIRMED, X1b′ FALSIFIED |

**Fail-closed behaviour.** Each of these was refused with rc = 2 and no output written:
- `score_I`: one measurement line removed; a duplicate line; `unit_metrics_I.jsonl` removed; `i_sham_validity.json` removed; an empty checkpoint root (`SCORE_CKPT`).
- `score_X` and `score_X_amend`: one snapshot removed (s4 step 725); `hist.json` removed.

Setting `all_valid = false` gives R-S = SHAM INVALID, as frozen.

**Number checks.**
- Every Stage I table value in `T4_FINAL_CAUSAL_CONTROLS.md` matches `score_I.json` to the quoted precision, except the items in F6. That covers the δ tables, the classes, the KL ranges, the CIs, the PIs, the learning-rate masses, and the sham-validity KL ranges.
- Every X1 trajectory value, interval, run length and verdict in `CROSS_FAMILY_CONFIRMATION.md` matches the score files, except F6.
- The `GATE_CAUSAL_MECHANISMS_AND_TESTS.md` §3b values match `logs/gate_mech/s3.json` and `s4.json`.
- The ledger's §4 / N25 numbers match.

**Labelling.**
- The GATE `g_mech`/`g_das` runs on s0–s2 (18:03–18:40, before FREEZE X) are labelled [expl].
- The s3/s4 mechanisms are labelled "[expl, after X1 was scored]", and the timing confirms this.
- The amendment is labelled as such, and its asymmetry is disclosed.
- The reuse of the XIII-E/XIII-D real-block runs, whose outcome was known before FREEZE I, is disclosed. The new confirmatory content of R-S and R-N is therefore the controls.

**Statistical units.** These are described correctly:
- T4: one parent, n = 6 base × data-order pairs, stated as not independent.
- GATE: n = 2 new models; s0–s4 are separately seeded from-scratch replications.

## Findings

### F1 (Major: disclosure; low impact on verdicts). GATE s3/s4 training outputs existed and were committed before FREEZE X and before the amendment.

**Evidence.**
- `g_train.py` writes a single-batch training loss every 400 steps to `logs/g_train_s{3,4}.out`.
- The step-400 lines were written at about 17:30:00Z (start 17:20:46 + 554 s / 553 s). `logs/FREEZE_X.md` was last modified at 17:41:27Z.
- The FREEZE X commit `5d3c9dd` contains these files up to step 2400. The amendment commit `b019ca9` contains them up to step 4000.

**What they show.** The values bear on X1:
- s3 at step 400: 0.4613, essentially the order-1 ceiling of 0.462;
- s4 at step 400: 0.2404, far below the order-2 optimum of 0.431, which implies gate use by step 400.

**What the documents say.** FREEZE_X.md ("nothing they contain has been loaded"), FREEZE_X_AMEND.md, CROSS_FAMILY_CONFIRMATION.md and MAIN_RESEARCH_SYNTHESIS.md ("before any X1 output was inspected") mention only the checkpoint directories. Training losses are model outputs.

**Impact.** There is no evidence that the lines were read. The impact on the verdicts is small:
- X1a was already strongly indicated by development seeds 0–2 and the seed-1 snapshots;
- X1b and X1b′ were falsified, so the amendment rescued nothing;
- X1a′ is stricter than X1a.

The claim of complete blindness is nonetheless not literally true, and the evidential weight of X1a's confirmation should be stated with this caveat.

**Correction.**
- Replace "before any X1 output was inspected" with: "before any X1 evaluation was run; the training scripts' coarse loss lines (one batch every 400 steps) were being written to `logs/g_train_s3/4.out` and were committed in `5d3c9dd` (to step 2400) and `b019ca9` (to step 4000). Their values were not used, but they were not blinded."
- In future, send confirmation-model training output to a file that is hashed but not viewed, or suppress it until the freeze.

### F2 (Moderate: provenance). The four step-2000 checkpoints that R-P needs were missing at FREEZE I; the frozen measurement jobs skipped them silently.

**Evidence.**
- `unit_metrics.py` keeps only the requested steps that exist (`s in av`).
- The frozen jobs `67b8e1ad8053`, `b2c4ad77d191`, `d791ef8c1a5f` and `de1d6dac805e` (`UM_STEPS=2000,4000` on the D = 801 `br1211_B_*` units) measured only step 4000 and returned rc = 0.
- These step-2000 values are the numerator δ(r = 1900) of the persistence index for both D = 801 pairs, so they are needed for R-P.
- The scorer's completeness check is what caught the gap, which is correct behaviour. The files were restored at 23:40:25 and measured at 23:41:35, outside the lane harness: there is no claim, no `logs/run/*.out` and no rc.
- FREEZE_I.md lists only steps 100–600 (and 1211) as restored.

**Mitigation, which I verified.** The restored bytes equal the XIII-E-era backup files, whose original mtimes are 11:27–13:36Z. The four step-4000 siblings match the XIII-E manifest. The restoration therefore introduces no new information.

**Correction.**
- `T4_FINAL_CAUSAL_CONTROLS.md` and `REPRODUCTION.md` should:
  - name the restoration source (`/home/claude/x13e_excluded_ckpt/...`, a session-local backup of files excluded from the XIII-E archive);
  - state that FREEZE I's restoration list omitted these four files;
  - state that the frozen jobs returned rc = 0 without measuring them;
  - state that the 23:41 measurement used the frozen command line restricted to `UM_STEPS=2000`.
- Give the exact XIII-E commands that regenerate them; "regenerable by replaying XIII-E commands" is not enough.
- For future harnesses, `unit_metrics.py` should exit non-zero when a requested step is absent.

### F3 (Major: reproduction tooling). The preflights and `verify_clean14.sh` fail on the completed package.

**Evidence.**
- P0 of `preflight_I.py` and of `preflight_X.py` runs the scorer on the *real* state and expects a refusal. That holds only while runs are incomplete.
- On a package copy, `preflight_I.py` now returns rc = 1 with `P0_refuse_real_state: false`, and **overwrites `logs/preflight_I.json`** (the record from freeze time). Its sha256 changed from `05bfddf8…` to `1482bb26…`.
- `preflight_X.py` would likewise run a full 13-minute X1 scoring and fail P0.
- `verify_clean14.sh` step 12 runs both preflights, so a clean extraction will print "PREFLIGHT I FAILED" and "PREFLIGHT X FAILED", and will overwrite the preflight records in the extracted tree.

**Other defects in `verify_clean14.sh`:**
- Step 12's `score_I` test checks for `score_I.json.new`, a file that is never written. It also copies the existing `score_I.json` into the temporary directory. The "writes nothing" half of the test is therefore vacuous; only rc is tested.
- `score_X_amend.py` and `preflight_X_amend.py` are not exercised in steps 8, 10 or 12, although X1a′/X1b′ are reported verdicts.
- Steps 2 and 11, and REPRODUCTION.md, refer to `MANIFEST.sha256` and `logs/VERIFY_CLEAN.txt`, which do not yet exist.

**Correction.**
- Make P0 conditional: skip it, or check it against a recorded incomplete state, once the scorer's outputs exist.
- Alternatively, have the verify script run the preflights on a copy and never overwrite archived preflight JSONs.
- Fix the step-12 filename check.
- Add `score_X_amend.py` (reproduction plus a missing-snapshot refusal) and `preflight_X_amend.py`.
- Generate `MANIFEST.sha256` and `VERIFY_CLEAN.txt` before the final commit.

### F4 (Minor: scorer design). `score_I.py` hash-checks `PKG/logs/I_design.json` but reads the design from LOGDIR.

**Evidence.** On a LOGDIR copy whose `I_design.json` measurement list was cut to 10 entries, the scorer accepted it and returned CONFIRMED. The completeness check is therefore only as strong as the LOGDIR design file.

**Impact.** None for the archived result: it was scored from `logs/`, which is hash-covered, and it reproduces bitwise.

**Correction.**
- Read the design from `PKG/logs`, or hash the LOGDIR copy.
- Optionally, check checkpoint hashes rather than existence only. All three scorers verify only that the files exist.

### F5 (Minor/Moderate: statistical scope). R-P is essentially a result for base M.

**Evidence.**
- The 4 valid R-P pairs are M-23, M-801, M-802 and E-802. The E pair's difference is +0.036 (PI 1.151 against 1.115), which is equal persistence.
- The other two E pairs are undefined, because their late block had no effect.
- The bootstrap intervals resample 4 (R-P) or 6 (R-S, R-N) pairs that cluster in 2 bases from one parent. The [0.795, 0.804] interval for R-S describes the spread across data orders, not uncertainty over models or parents.

**Over-statements.**
- T4_FINAL §I-B, point 4 ("early-block changes are more persistent and more consistent than late-block changes") and the ledger's §3 item 10 ("the effect is route-specific and more persistent early") do not separate the bases. For E, persistence is not distinguished: the E late block is mostly *ineffective*, which is a different property.
- "Route-specific" in the ledger is stronger than the frozen meaning, which is "beyond this sham and this TV-matched noise". T4_FINAL itself notes that the shams increase the route's attention share.

**Correction.**
- Add: "R-P's confirmation rests on base M (3 of 4 valid pairs); on E the only defined pair shows equal persistence; the intervals are descriptive over 4–6 dependent pairs."
- Replace "route-specific" in ledger item 10 with "specific relative to an off-route sham and TV-matched noise".

### F6 (Minor: number transcription and wording)

| document | as written | correction |
|---|---|---|
| T4_FINAL | "Late runs: 0.00–0.02 on M" | M late PI values are 0.002, **−0.011** and 0.020. Write "−0.01–0.02". |
| T4_FINAL | "Neither route's own-question attention is reduced" | E layer-2 own mass changes by −0.0007, −0.0003 and +0.001, within the frozen 0.01 tolerance. Write "reduced by at most 0.001". |
| CROSS_FAMILY_CONFIRMATION §3 | order-1 phase "NLL 0.45–0.48" | The frozen X1a runs include step 100: s3 NLL 0.534, s4 0.510. Write "0.45–0.53 (≤ 0.50 in the X1a′ runs)". |
| CROSS_FAMILY_CONFIRMATION §2 | X1a′ s3 (14) | The X1a′ run for s3 is steps 125–450, not 100–425; s4's is 125–275. Give the steps. |
| GATE §4 | column header "shared by s0–s2?" | The rows say "all five" and cite §3b. Rename the header to "shared by s0–s4?". |
| GATE §5 | "all 3 models store it at E" | 5 models (with §3b). |

### F7 (Minor: logs)

- **A retrospective timestamp.** The COMPUTE_LOG entry "22:38:10Z EXPL: g_mech s4 rerun after restart" was written after `logs/gate_mech/s4.json` existed (22:37:57). Mark retrospective entries as such.
- **Runs outside the lane harness.** `g_train.py 3/4`, the dev runs and the 23:41 step-2000 measurement have no rc record. GATE completion is shown only by `hist.json` and step 4000.
- **Pre-restart output lost.** `lane2.sh` truncates `logs/run/<key>.out` (`>`), so the output from before the restart was lost for the two resumed jobs.
- **"Resumed bitwise" is not checked here.** T4_FINAL's claim rests on the XIII-E resume test (`ft_factorial_c.py` docstring, `repro_resume.json`), not on a check of these two runs. Cite that test.
- **Stream note** (informational): "This confirmation stream is unused in development" (FREEZE I) holds within XIV. XIII-E used seed 95_300_000 once, as `RS_TRAIN`, for a different generator (`route_status.py`); this is not a data overlap.

### F8 (Minor: packaging)

**Not yet committed:**
- `T4_FINAL_CAUSAL_CONTROLS.md`, `MAIN_RESEARCH_SYNTHESIS.md` and `logs/DEPENDENCIES.json` are untracked;
- the ledger edit is uncommitted;
- `checkpoints/` is untracked, as disclosed; the final commit must include the 4 restored step-2000 files and the s3/s4 snapshots.

**Other:**
- `logs/gate_mech/s0–s2.json` record absolute `/home/claude/...` checkpoint paths. The verify step-13 scan covers only `scripts/` and `src/`.
- `DEPENDENCIES.json` correctly lists 221 external files across 12 sibling packages, with no missing files. Its record of where the restored files came from is the subject of F2.

## Overall verdict

**The confirmatory results are sound and reproducible.**
- All freeze hashes hold at the freeze and scoring commits.
- The scoring order respects every freeze, including the X amendment.
- Every design run and measurement is present.
- The restorations are byte-verified against XIII-E-era files.
- All three scorers fail closed on missing inputs, and reproduce `score_I.json`, `score_X.json` and `score_X_amend.json` bitwise.
- These verdicts stand: R-S, R-N and R-P CONFIRMED; X1a and X1a′ CONFIRMED; X1b and X1b′ FALSIFIED.

**Required before release:**
- **F1:** disclose that the coarse s3/s4 training losses were present and committed before FREEZE X and before the amendment.
- **F3:** fix the preflight P0 and verify-script defects. As they stand, the clean-extraction check will report failures and overwrite the preflight records.
- **F2 and F5:** add the provenance and statistical-scope corrections.

The remaining findings are wording and logging corrections.
