# Post-results Review E: integrity, provenance, reproduction, raw-output completeness

**Reviewer:** an AI agent (Claude). This is not human peer review.

**Scope:** the mandate's §13 post-confirmation items on frozen-prediction chronology, raw-output completeness and deterministic reproduction, plus checks for missing negative runs. §16 and §17 items 2, 4, 11, 12, 15, 16 and 17 are covered as far as they concern the confirmation record.

**Not in scope:** the scientific interpretation. Reviewers A to D cover it.

**Package and HEAD:** `step1_training_state_causal_identification/` at HEAD `92f8934`. Freeze commit `478a038`; functional-battery freeze `8ca9db7`.

**Reviewer footprint:** I modified no existing file.
- All rescoring went to `/tmp/reviewE/`.
- I created six temporary run directories under `checkpoints/ft/` (names contain `reviewE`) and deleted each one after comparing it. The directory count is back to 353.
- CPU: at most 2 single-threaded processes at a time, `nice`, with the frozen `XLA_FLAGS`.

## Evidence sources used

- **Git:** commit times from `git log` and the reflog.
- **Push times:** the GitHub repository activity API, `gh api repos/<owner>/<repo>/activity`. The local clone has no remote-tracking ref for this branch.
- **Logs:** `logs/COMPUTE_LOG.txt` and `logs/run/<key>.out`.
- **Hashes:** `FREEZE_XIX.json`; `run_config.json`, `provenance.json` and `resume_log.json` in each run; the `measure_c19.jsonl` records.
- **Filesystem:** mtimes and atimes (ext4, `relatime`), and `/proc/stat` btime for the container boot time.

## Findings

### 1. ADVISORY (verified). The design freeze came before the first confirmation run

- **Commit:** `478a038` was committed at 2026-11-19T17:21:39Z.
- **Push:** GitHub's activity API records the branch being created at that commit at **17:21:47Z**.
- **First confirmation job:** `COMPUTE_LOG` line 222, at **17:21:56Z**.
- **Frozen hashes:** all 46 hashes in `FREEZE_XIX.json` match the current files (39 package files and 7 external dependencies).
- **Append-only log:** `git diff 478a038 HEAD -- logs/COMPUTE_LOG.txt` removes no lines.
- **No frozen file changed:** since `478a038`, the only changes under `scripts/` and `src/` are new files: `func_battery19.py`, `score_func19.py`, `conf_lanes19.sh`, `verify19.py` and `build_archive19.py`.

**Correction (wording only).** The chronology table in `FROZEN_CONFIRMATION_AND_STATISTICS.md` (row 1) says "2026-11-19T17:20Z". Replace it with "committed 17:21:39Z, pushed 17:21:47Z, first confirmation job 17:21:56Z".

### 2. MANDATORY. The stage-6 functional battery was designed, calibrated and frozen after confirmation had started

The chronology table lists `8ca9db7` without a time, directly after the design freeze. A reader would take it to precede confirmation. It did not.

- **Code:** `scripts/func_battery19.py` was last modified at 17:22:53Z, one minute after confirmation started.
- **Development calibration:** the outputs were written while the confirmation lanes were running:
  - `logs/dev/func_battery_c18.json` at 17:24:54Z;
  - `func_battery_val.json` and `.out` at 17:33:27Z;
  - `func_battery_degradations.json` at 17:35:28Z.
- **Thresholds:** the SR thresholds 2.2 and 1.6 and the interpretation boundary were therefore fixed during confirmation.
- **Freeze:** `8ca9db7` was committed at 17:35:56Z and pushed at 17:36:01Z, **14 minutes after the first confirmation job**.
- **Outputs already on disk at that time:** 16 of the 54 confirmation step-100 states, by mtime: all 9 D821-E states and 7 of the 9 D822-E states.
- **Unlogged compute:** the battery development ran outside the lanes, on the same 2 CPUs, and does not appear in COMPUTE_LOG.

The claim that no confirmation output had been read rests on self-attestation, backed by the access-time evidence in finding 9. The battery models and streams were XVIII models and 93.1M/93.2M, not confirmation outputs, so I see no direct leak. But F1–F3 are **not** part of the pre-confirmation freeze that reviewers A, B and C approved.

**Correction.**
- In the chronology table, give `8ca9db7` as "committed 17:35:56Z, pushed 17:36:01Z, i.e. 14 min after confirmation start". State that the battery code (17:22:53Z) and its development calibration (17:24–17:35Z) were produced during confirmation.
- Wherever F1–F3 are cited (`FROZEN_CONFIRMATION_AND_STATISTICS.md` §Stage-6, `FUNCTIONAL_VS_CAUSAL_ORGANIZATION.md`, `MAIN_RESEARCH_SYNTHESIS.md`), label them "addendum predictions frozen after confirmation start and before any confirmation output was read (self-attested)". Do not call them part of the reviewed confirmation freeze.

### 3. MANDATORY. The battery scorer was written after the results were read, and it departs from the frozen F3 rule

- **Timing:** `scripts/score_func19.py` was created at 02:25:51Z. That is after `score_c19.json` (02:08:50Z) and after `func_battery_c19.json` (02:22:06Z). The scorer is in no freeze.
- **F3 deviation:** the frozen text in `logs/FUNCTIONAL_BATTERY_FREEZE19.md` defines F3 as "category NEITHER or FUNCTIONAL_ONLY". The script also admits `INCONCLUSIVE(causal_NONE…)`. By contrast, F2 explicitly lists `INCONCLUSIVE(causal_SWITCH)`, so adding the NONE case to F3 is not an obvious reading of the text.
- **Models affected:** 5 M models were added to F3 by this deviation: M821 paramB_optI, M821 IfromB_L0L2, M822 paramB_optI, M822 IfromB_L0L2 and M822 frzp.
- **Unfrozen band:** the script also introduces a PARTIAL band that is not in the frozen text. No verdict landed in it.
- **Effect:**
  - Under the literal frozen rule, M_F3 is **16/16 SUPPORTED**, not 21/21.
  - E_F3 is unchanged at 19/19.
  - No band changes.
  - The 5 extra models all have SR class MIXTURE, which matches their B anchor.

**Correction.** Either change `score_func19.py` to the literal frozen F3 set and re-run it, or keep it and disclose the deviation. In both cases:
- report M_F3 as 16/16 under the frozen rule, and list the 5 INCONCLUSIVE(causal_NONE) models as descriptive;
- state that the battery scorer was written after the results were read.

Apply this in `FROZEN_CONFIRMATION_AND_STATISTICS.md` (the Stage-6 table) and `FUNCTIONAL_VS_CAUSAL_ORGANIZATION.md`.

### 4. ADVISORY (verified). The raw outputs are complete

I matched each queue line to COMPUTE_LOG by the lane's own key, `md5(line + "\n")[:12]`.

- **queue_c1:** 54 lines, 54 distinct keys. Each has exactly one `end … rc=0`.
- **queue_c2:** 99 lines, 99 distinct keys. Each has one `rc=0` end. The two jobs interrupted by the restart (finding 5) each have 2 starts and 1 end.
- **Whole log, development included:** no `rc≠0`, and no job without an end except those two.
- **No extra lane jobs:** after the freeze, COMPUTE_LOG has no lane job outside the two queues.
- **Window states:** 54/54 directories hold `step100.npz`, `opt_step100.pkl`, `resume.pkl`, `run_config.json`, `stop_done.json` and `window_loss.json`. None has a `resume_log.json`, so no pnz or lbl window was resumed.
- **Measurements:** 99 records with 99 unique `run` values, one-to-one with the queue_c2 measurement targets. No duplicates. All have stream 94 700 000, step 1211 and parent uP160. All 99 are competent.
- **Directories:** the 210 confirmation-era directories split exactly into 54 window states, 57 composed `tp_*` states and 99 continuations. There are no orphan or unreported runs.
- **Ordering:** every continuation started after its source's `stop_done.json` was written.
- **Matched data stream:** within each of the 6 pairs, all 9 window states have the identical data-stream RNG state at step 100.

### 5. ADVISORY. The restart is mis-timed in the report, but the resumed continuations are valid (verified bitwise)

**Timing.**
- **Actual time:** the container booted at **23:50:27Z** (`/proc/stat` btime). The last write before the crash was at 23:50:09Z (`…M822_IfromB_L0/step165.npz`).
- **Detection:** the restart was detected, and the lanes restarted, at 00:00:41Z.
- **Report:** `FROZEN_CONFIRMATION_AND_STATISTICS.md` says "≈ 23:55Z".

**The interrupted jobs.** No other confirmation job was running, since both lanes are accounted for.

| key | job | first start | progress before the crash | resume |
|---|---|---|---|---|
| `67316a65ab90` | tp_M822_IfromB_L0 | C1, 23:49:27Z | step 165 | step 165 |
| `ca2146727a93` | tp_M822_BfromI_L0L1 | C2, 23:49:43Z | step 111 | step 111 |

After the restart the jobs swapped lanes. The resume points are recorded in each run's `resume_log.json`.

**Why the resumed runs are valid.**
- On the rerun, `compose19.py` printed "already composed (identical)", so the composed state was not regenerated differently.
- The first attempt's `run_config.json` (start 100) was kept, and its `branch_src` hashes equal the composed `provenance.json` outputs.
- Neither job had written a measurement record before the crash, so the restart created no duplicate record.

**Reproduction.** I re-ran both continuations uninterrupted from step 100 to 1211 in temporary directories. Each was bitwise identical at steps 111, 150, 165, 200, 813, 1000 and 1211 (max |Δ| = 0.0). For both, the step-1211 sha256 equals the `ckpt_sha256` in `measure_c19.jsonl`. So the resumed outputs are exactly what an uninterrupted run produces.

**Correction.** Change "≈ 23:55Z" to "crash ≈ 23:50:10–23:50:27Z (boot 23:50:27Z); detected and lanes restarted 00:00:41Z". Name the two runs and their resume steps (165 and 111), and note that both were reproduced bitwise by uninterrupted replay.

### 6. ADVISORY. The stdout of the first attempts is lost

`scripts/lane2.sh` (frozen) redirects with `>`. The first attempts' stdout in `logs/run/67316a65ab90.out` and `logs/run/ca2146727a93.out` was therefore overwritten, and only the resumed attempt's stdout survives.

**Correction.** Disclose this in the chronology row. In future harnesses, append with `>>` and write an attempt separator.

### 7. ADVISORY. The frozen scorer re-run gives identical bands, but its output file is not byte-reproducible

**What I ran:** `python3 scripts/score19.py logs/conf/design19.json logs/conf/measure_c19.jsonl /tmp/reviewE/x.json` with the frozen `XLA_FLAGS`. It took 44 s, returned rc 0, and was in files mode with all provenance checks.

**What matches:**
- All 35 bands are identical: 27 SUPPORTED, 5 PARTIAL, 3 CONTRADICTED.
- Every per-pair value is identical, and the whole `hypotheses` object is identical.
- `stats19.py` run on my rescored file gives a `stats_c19.json` and `.out` **byte-identical** to the package's.
- `score_func19.py` run on my rescored file gives a `func_score_c19.json` byte-identical to the package's.
- Recomputing the 12 frozen sensitivity variants' band changes from the stored variant files gives exactly the "variants that differ" column of the FROZEN table.

**What differs:** `score_c19.json` itself differs from my output, but only in the order of the `duplicate_composed_states` lists (pairs M822 and E823). The cause is `score19.py` line 157, which iterates `set(c.values())`. String-hash order depends on `PYTHONHASHSEED`, so the file is not byte-reproducible across processes.

**A misleading field.** That duplicate check hashes only `step100.npz`. It therefore always reports (fullI, paramI_optB) and (fullB, paramB_optI), which differ in their moments. The FROZEN report discloses this correctly. Still, the field cannot detect a true duplicate state.

**Correction.**
- Any reproduction claim, here or in `verify19.py`/VERIFY output, should say "bands and values identical; `duplicate_composed_states` order varies". Do not claim byte identity of `score_c19.json`.
- For future scorers: sort that list, and hash parameters plus moments.

### 8. ADVISORY (verified). Deterministic reproduction holds on four further runs

All of these were bitwise identical. Every temporary directory was deleted after comparison.

| run | what I compared | result |
|---|---|---|
| E821 vsgo continuation (branched from a window state) | 100→150, `step150.npz` | sha256 equal, max |Δ| 0.0 |
| M823 randnorm_L0L2 continuation (branched from a composed state) | 100→150, `step150.npz` | sha256 equal, max |Δ| 0.0 |
| M822 pnz window state (own noise RNG), regenerated 0→100 | parameters, Adam moments, data RNG, `window_loss.json` | all bit-identical |
| E823 vsg window state, regenerated 0→100 | the same four items | all bit-identical |

The two interrupted continuations in finding 5 were also reproduced to step 1211. In total that is 6 independent reproductions, including 2 full continuations.

### 9. ADVISORY. Access-time evidence is consistent with no reading before scoring, but cannot prove it

The root filesystem is ext4 mounted `relatime`. Under relatime, the first read after a file's last write updates its atime.

- **Job stdout:** none of the 153 confirmation `logs/run/*.out` files has an atime later than its mtime, except `ca2146727a93.out`, which I read during this review at 02:41:57Z.
- **Measurement file:** `measure_c19.jsonl` (last append 02:03:44Z) was first read at **02:07:38Z**, 4 s after the logged "frozen scoring starts" at 02:07:34Z.
- **Step-100 states:** for all 54, the first read after creation coincides with the queue job that consumes the state (the pnz noise-scale reference, or compose/branch). This includes the 16 states that existed before `8ca9db7`.

**Limits.** Relatime cannot show a read that was followed by a later write, such as a read of the jsonl between two appends. Nor can it show a second read within 24 h. This is corroboration, not proof.

**Correction.** None required. The FROZEN report may cite this evidence with its limits.

### 10. ADVISORY. COMPUTE_LOG does not record the foreground work

**Unlogged:**
- the battery development during confirmation (finding 2);
- everything after 02:07:34Z: scoring, the 12 sensitivity variants, stats, the battery on the confirmation models (stream 93.3M), `s100diag`, `compat` and `score_func`.

**Unsupported claim:** the FROZEN report says "Frozen-file check at scoring … verified before the scorer ran", but there is no logged artefact of that check. `score19.py` itself verifies only the subset listed in `design19.json`. I re-verified all 46 frozen hashes after the fact; they match.

**Correction.** Append dated entries to COMPUTE_LOG for these steps, or state in the chronology that they are unlogged. Either record the full freeze-hash check as a file, or cite this review's check.

### 11. ADVISORY (verified). Measurement records match the checkpoints, and provenance matches the design

- **Checkpoints:** for 99/99 records, `ckpt_sha256` equals the sha256 of the run's `step1211.npz`.
- **Dependency hashes:**
  - `dep_sha256` equals the frozen `dep_sha256_measure`, the `FREEZE_XIX.json` package entries, and the external dependency hashes.
  - `score_code_sha256` equals the frozen `src/xix_score.py`.
  - `refs_sha256` is a single value across all 99 records and equals `design19.json`.
- **Harness:** for all 153 runs, `run_config.json` gives `xix_train.py` = the frozen harness hash.
- **Window states:** all 54 start at 0.
- **Continuations:** all 99 start at 100. Their environments equal the queue lines exactly, and none sets `FX_INTERV`.
- **Branch sources:** `branch_dir` equals the intended source, and `branch_src` hashes equal the current source files.
- **Composed states:** for all 57 `tp_*` states, the `provenance.json` input and output hashes match the current B/I sources and the composed files.

### 12. ADVISORY. The confirmation driver was not in the freeze

`scripts/conf_lanes19.sh` (mtime 17:21:56Z, first committed in `92f8934`) was not in the freeze. It is a 3-line wrapper that runs the frozen `lane2.sh` over the frozen queues, so it carries no scientific risk.

**Correction.** Mention it in REPRODUCTION or the chronology as "unfrozen driver, content shown".

### 13. ADVISORY. No missing negative runs found

- All 99 designed continuations were run, measured and scored.
- All 12 frozen sensitivity variants are present.
- No confirmation job failed or was re-run, apart from the two resumes in finding 5.
- No confirmation-era directory is unaccounted for.
- The 3 CONTRADICTED and 5 PARTIAL hypotheses are reported with their per-pair values.
- No development job in COMPUTE_LOG ended non-zero or lacks an end.

I did not audit whether every development result is discussed in the prose reports. That is outside this review.

## Summary

| check | result |
|---|---|
| Freeze committed and pushed before the first run; frozen hashes | PASS (17:21:39 / 17:21:47 / 17:21:56Z; 46/46 hashes) |
| Functional battery frozen before the confirmation | **FAIL as reported**: frozen 14 min after confirmation start, calibrated during it (finding 2); before outputs were read (self-attested, atime-consistent) |
| Battery scorer frozen and literal | **No**: written after the results; F3 deviation with no band change (finding 3) |
| 54 window states, 99 continuations, 99 records; every queue line rc=0 | PASS |
| Restart handling and resume validity | PASS. Bitwise replay to step 1211 matches the measured hashes. The restart time is misreported (finding 5) |
| Frozen scorer re-run | PASS on bands and values. File not byte-identical, an ordering issue only (finding 7) |
| Determinism spot checks (2 × 100→150, 2 × step-100 regeneration, 2 × 100→1211) | PASS, all bitwise |
| Measurement hashes against checkpoints | PASS, 99/99 |
| Missing negative runs | none found |

VERDICT: APPROVED WITH MANDATORY CORRECTIONS. The integrity, provenance, completeness and deterministic reproduction of the frozen confirmation (score19, 27/5/3) are confirmed. Before the stage-6 functional-battery results (F1–F3) may be presented as frozen confirmation, two corrections are required: the chronology must state that the battery was designed, calibrated and frozen after confirmation began (finding 2), and the post-results F3 scoring deviation must be disclosed or removed, with M_F3 reported as 16/16 under the literal frozen rule (finding 3).
