# Review C: experimental integrity (agent review, not human peer review)

Investigation XIII-E. The reviewer is an independent agent. Branch `cor-phase15-step1-reroll-and-critical-period-validation`, reviewed at HEAD `b10a8f7` plus the working tree of 2026-11-16 at about 14:40Z. At that time `REROLL_VS_STRUCTURE.md`, `EARLY_TRAINING_CAUSALITY.md`, `COR_FOUNDATIONAL_NAVIGATION.md`, the B4 route logs and five lines of `COMPUTE_LOG.txt` were not yet committed.

**What the reviewer did and did not do.**
- Trained nothing and modified no existing file.
- Ran the preflights and scorers only on copies in `/tmp/revC`, because the preflights write `logs/preflight_*.json` inside the package.

## Checks performed

1. **Freeze chronology.**
   - For each freeze commit (A-1 `2063085`, A-2 `a4f66b7`, B `b710b38`, C `87177ad`), every file in `logs/FREEZE_*_HASHES.txt` was hashed with SHA-256 three times: as `git show <commit>:<path>`, as `git show HEAD:<path>`, and in the working tree.
   - The per-commit file lists were compared, and so were the run and checkpoint mtimes against the commit times.
2. **COMPUTE_LOG audit.** 127 lane starts and 127 ends were checked for:
   - non-zero return codes and repeated lane IDs;
   - repeated commands;
   - a `.out` file for every lane ID;
   - every queue line matched to a logged command.
3. **Run completeness.** The run lists in `A_design.json` (64 runs) and `B_design.json` (41 runs plus 4 continuations) were compared with `checkpoints/ft/` (119 directories) and with the metric files.
4. **Preflights.** `preflight_A.py`, `preflight_B.py` and `preflight_C.py` were re-run on a copy.
5. **Scorers.**
   - Re-run on complete copies, with the outputs deleted first.
   - Run on 9 deliberately incomplete copies.
6. **Continuation rule.** The frozen rule was recomputed independently from `unit_metrics_A.jsonl`.
7. **External XIII-D inputs.** The 104 files used here (the C-C1 and C-C2 carrier checkpoints and the B base runs uP160 21/21/21 and 23/23/23) were verified against the XIII-D `MANIFEST.sha256`.
8. **Report numbers.** Every verdict and the main numbers in the three reports were compared with `score_A.json`, `score_B.json` and `score_C.json`.

## Summary of results

| check | result |
|---|---|
| A-1 hashes (12 files): at commit, at HEAD, in working tree | all match |
| A-2 hashes (13 files) | all match |
| B hashes (12 files) | all match |
| C hashes (9 files) | 8 match. `logs/c_audit.json` differs at HEAD and in the working tree (finding 3) |
| Freeze hash files themselves | unchanged since their commits |
| Confirmation output before its freeze | none found (finding 1) |
| Lane return codes | 127/127 rc=0. No OOM, Traceback or Killed in any `logs/run/*.out` |
| Repeated or missing runs | none. 64/64 A, 41/41 B and 14/14 continuations each started exactly once, and each has `stop_done.json` or `final.json`. No checkpoint directory lies outside the design lists |
| Queues | `queue_A` 64/64, `queue_B` 33/33, `queue_Blate` 8/8, `queue_Bcont` 4/4, `queue_Acont` 10/10 and `queue_C` 2/2 lines appear verbatim in the log. `queue_Bmeasure_gated.txt` equals the frozen `queue_Bmeasure.txt` once the gate prefix is removed. The `queue_Ameasure.txt` unit list is exactly the 64 design specs, with the frozen env |
| Preflights | A: P0, P1, N1–N7 OK. B: P0, N1–N7 OK. C: P0, N1–N4 OK. Regenerated `preflight_*.json` files are byte-identical to the frozen ones |
| Scorers on complete copies | `score_A.json`, `score_B.json` and `score_C.json` reproduced **bitwise**. They are identical to HEAD |
| Scorers on incomplete copies | all 9 refused with exit 2 and wrote no output (table below) |
| Continuation set | follows the frozen rule (finding 6) |
| External XIII-D inputs | 104/104 SHA-256 OK |
| Verdicts in the reports | all match the score files (finding 9) |

**Partial-input tests** (copies of `logs/`, with symlinked checkpoint trees where needed):

| test | scorer response |
|---|---|
| A: last line of `unit_metrics_A.jsonl` removed | REFUSED, no 1211 metrics for `uP161:fx:708:21:21` |
| A: one line of `unit_metrics_Acont.jsonl` removed | REFUSED, no 4000 metrics |
| A: checkpoint directory of one run absent | REFUSED, missing `stop_done.json` |
| A: continuation manifest truncated to 9 | REFUSED, "does not follow the frozen rule" |
| B: one line of `unit_metrics_B.jsonl` removed | REFUSED |
| B: last line of `unit_metrics_Bfinal.jsonl` removed | REFUSED, missing anchor baseline |
| B: continuation checkpoint absent | REFUSED, not finished |
| C: C3 block removed | REFUSED |
| C: C4 k = 6 removed | REFUSED |

## Findings

### 1. No confirmation output was produced or read before its freeze. Severity: none (positive finding)

**Evidence.**
- **A.** The earliest A checkpoint file is from 01:09:40Z, after A-1 at 01:09:17Z. Re-roll measurement (lane MA) started at 10:13:18Z, gated on all 64 `stop_done.json` files, and `unit_metrics_A.jsonl` was written at 11:09:51Z. Both are long after A-2 (01:37:19Z).
  - The only computation on A configurations before A-2 is `r2_A.jsonl` (01:36:55Z). `r2_init_stats.py` builds fresh step-0 initialisations and loads no trained checkpoint, as disclosed.
  - `monitor_A.py` reads only the existence of `stop_done.json`.
- **B.** The earliest B checkpoint file is from 02:00:20Z, after the B freeze at 01:59:36Z.
  - `b_calib.json` (01:52Z) uses only the XIII-D base checkpoints, on development stream 95.8 M.
  - The B metrics are from 12:47Z and 14:07Z, and `score_B.json` is from 14:14:59Z.
- **C.** The C-C2 audit (lane CA1, second line) was queued at 03:54:56Z behind `when_file.sh logs/FREEZE_C.flag`.
  - The flag's mtime is 04:06:24Z, after the 87177ad commit at 04:04:25Z, and the flag contains `87177ad`.
  - `when_file.sh` also requires the flag to be unchanged for 60 s before it runs the job.
  - The frozen `c_audit.json` contains only `uP160:s:23`, the development unit.
  - `c1_ideal_refs.json` is pure numpy reference mechanisms, not C-C2 data.
- **Freeze C disclosure.** `FREEZE_C.md` states that XIII-D had already measured SUFF, SRCF, reverse and the trace on C-C2, and that none of those quantities is predicted. That disclosure is appropriate.

**Correction.** None.

### 2. In-progress checkpoints were committed in three freeze commits, not one. The disclosure in COMPUTE_LOG is incomplete. Severity: low

**Evidence.**
- COMPUTE_LOG (04:08:07Z) names only `87177ad`. Two earlier freeze commits also added checkpoint files:

  | commit | checkpoint files added | contents |
  |---|---|---|
  | `a4f66b7` (A-2) | 84 | the 4 finished W' = 601 re-roll runs, complete, including `step1211.npz`, `opt_step1211.pkl`, `stop_done.json` and `resume.pkl` |
  | `b710b38` (B) | 76 | the W' = 602 runs, in progress |
  | `87177ad` (C) | 536 added, 2 modified | A and B runs |

- `acdf0c5` deleted 696 files.
- `resume.pkl` files are therefore in the git history of all three commits. COMPUTE_LOG says checkpoints "are committed, without resume.pkl, with the final package".
- These files hold weights and loss histories only. No outcome metric or class appears in any of them, and nothing shows they were read.

**Correction.**
- Amend the disclosure to name `a4f66b7` and `b710b38` as well, and to state that `resume.pkl` and partial checkpoints remain in the branch history.
- Optionally note the size added to the repository.

### 3. `logs/c_audit.json` is frozen in C but was later extended. Its frozen hash no longer verifies at HEAD. Severity: low (design flaw, no evidence of tampering)

**Evidence.**
- The hash matches at `87177ad` but not at `9dc7b07`, HEAD or the working tree.
- A key-wise comparison shows that the development record `uP160:s:23` is unchanged. The only change is the added key `uP165:s:21`, the C-C2 output.
- The same file serves as the frozen development input and as the confirmation output.

**Correction.**
- Add one sentence to `FREEZE_C.md`, `CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md` or the final verification script: the frozen hash applies to the file as committed in 87177ad, and only the `uP165:s:21` key was added afterwards. Re-hashing the `uP160:s:23` sub-record supports this.
- In future, write held-out output to a separate file.

### 4. `scripts/ft_factorial.py` was edited during A training and then restored. Severity: low (disclosed)

**Evidence.**
- COMPUTE_LOG (01:39:08Z) records that B interventions were added to the frozen A harness at about 01:38Z and removed within about 2 minutes.
- The file's mtime is 01:38, and its hash now equals the A-1 hash at every point checked.
- Two A runs, [9c2c28793002] and [6f206bb35e28], had started at 01:37:10–11Z. Python had already compiled the script before the edit, so both ran the frozen code.
- The next runs started at 01:50:39Z, after the restore.
- The edit time is known only from the log. No per-run hash of the code is recorded.

**Correction.**
- None required for the verdicts.
- In future, record `sha256(harness)` in each run's `stop_done.json` or `final.json`, so this kind of window can be verified rather than asserted.

### 5. Small factual slips in the freeze documents. Severity: low

**Evidence.**
- `FREEZE_A2.md` says "2 of the 64 training runs had finished". COMPUTE_LOG shows 4 ended before the commit: 01:22:39, 01:22:40, 01:37:10 and 01:37:11Z. The last two ended 8–9 s before the commit, so the note was probably written just before them, and `a4f66b7` contains those 4 complete runs.
- `FREEZE_B.md` says the short-window part has "34 runs: 32 conditions and baselines, plus the 2 D=801 sham checks". `B_design.json` and `queue_B.txt` have 33 short runs: 15 on the early base and 18 on the mixture base. 33 short plus 8 late gives the frozen total of 41, which is correct.

**Correction.** Add an erratum line to the reports or the final synthesis. The frozen files must not be edited.

### 6. The continuation set follows the frozen rule, but `score_A.py` checks the rule only weakly. Severity: low

**Evidence.**
- Recomputed independently, the two runs not in the fixed set with `mix_mass@1211` closest to 0.5 are `uP160_fx_L708_W21_D21` (0.4547) and `uP160_fx_L704_W21_D21` (0.5733). These are exactly the extras in `manifest_A_cont.json` and `queue_Acont.txt`. The next candidate is at 0.374, so no tie-break was needed.
- All 10 continuations have `final.json` and 4000-step metrics.
- `score_A.py` checks only that the set is a superset of the fixed 8 and has length 10. It does not re-derive the two extras. A substituted extra would still have had to be trained and measured, so this cannot be exploited silently, but the check is weaker than "follows the frozen rule" implies.
- `score_A.py` and `score_B.py` also build their metric dictionaries by key, so a duplicated metric line would silently overwrite the earlier one. No duplicate lines exist in any metric file.

**Correction.**
- Re-derive the extras in the scorer, or call `select_continuation.py` in a dry-run mode, in any reuse.
- Refuse duplicate `(unit, step)` keys.

### 7. External inputs and code are pinned only by the freeze hash lists and the XIII-D manifest. Severity: informational

**Evidence.**
- None of the scorers verifies hashes at run time. `score_A.py` imports `hashlib` but never uses it.
- `scripts/route_status.py` and `scripts/c_carrier.py` appear in no XIII-E freeze. They are used only for the descriptive B4 route measurements, which are labelled "not frozen", so this is acceptable.
- The external XIII-D checkpoints (parents, base runs and carrier units) are not in the XIII-E hash lists. The reviewer verified the 104 files used here against the XIII-D `MANIFEST.sha256`, and all were OK.

**Correction.** Have the final verification script check the external inputs against the XIII-D manifest. A command-line `sha256sum -c` over the filtered manifest lines is enough.

### 8. Memory and failures. Severity: none

**Evidence.**
- Every lane returned rc=0.
- No OOM, `MemoryError`, `RESOURCE_EXHAUSTED`, Killed or Traceback appears in any of the 127 run outputs.
- No run restarted. Each command was started exactly once, and there is no "resum" line in any run output.
- Peak concurrency reached 6 processes (11:11–12:09Z: TC1, TC2, TBC, TBL, MB and MAC) on 8 GB, with no failure. The per-run peak RSS from the benchmark is 0.77–0.79 GB.

**Correction.** None.

### 9. Reported verdicts and numbers match the score files. Severity: none (positive finding)

**Workstream A** (`REROLL_VS_STRUCTURE.md` against `score_A.json`):
- R1-W: ρ = 0.6548 (per parent 0.548 and 0.762), p = 0.00615, upper bound 0.864, outcome TRANSFER, verdict **FALSIFIED**.
- R1-L: ρ = −0.024, p = 0.541, upper bound 0.522, outcome NO-TRANSFER, verdict **CONFIRMED**.
- Every cell table count, Clopper–Pearson interval and mix_mass list matches.
- The Newcombe differences match: +0.125, +0.25, 0 and 0.
- Homogeneity matches: p = 0.0033 and 0.0015.
- The concordance counts match: 4/3.0, 5/4.0, 2/2.0 and 7/7.0.
- Stability matches: 9 of 10, with L708 going EARLY → hybrid.
- Competence matches: 63/64, the exception being `uP161_fx_L22_W608_D22`.
- The exploratory AUC matches: 0.585.
- The frozen R1 statistic includes the one non-competent run, as the frozen code specifies. The report discloses the 63/64.

**Workstream B** (`EARLY_TRAINING_CAUSALITY.md` against `score_B.json`):
- R3: real100 flipped 3/3 on both bases, dist100 0/3, sham bitwise identical on both bases, verdict **CONFIRMED**.
- R4: f_late = 4 of 8, f_strong = 2 of 2 competent (the third has KL 0.0514 and is excluded), verdict **FALSIFIED**.
- The window dose counts match row by row: 1/6, 6/6 and 6/6. So do the late-run classes and the 4/4 continuation persistence.

**Workstream C** (`CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md` against `score_C.json`):
- R5: k = 1/k = 3 ratio 0.667 against the required 0.85, verdict **FALSIFIED**. Stability clauses: k = 1 at 0.999 and k = 3 at 0.785, both of which pass. The report attributes the failure correctly to the ratio.
- R6: every clause holds (1.11, 0.024, 0.75, 0.068, 7.7, 0.58, 0.066), verdict **CONFIRMED**.
- The C4 table values for C-C2 (0.541 / 0.798 / 0.811) agree with the ratio.
- The COMPUTE_LOG verdict lines at 05:31:43, 14:15:09 and 14:32:09Z agree with the score files.

**Caveat.** `REROLL_VS_STRUCTURE.md` and `EARLY_TRAINING_CAUSALITY.md` were still uncommitted when this review was written. This check applies to their working-tree state at about 14:40Z.

**Correction.** None. After the final commit, re-check that the committed report text is unchanged in its verdict sections.

### 10. Development-to-freeze ordering inside B and C. Severity: none

**Evidence.**
- The B severity matching (the noise σ values and ro0p k = 2) is frozen in `b_calib.json` and `B_design.json`. It was used unchanged in every logged command, for example σ = 2.323, 0.25, 4.0 and 2.06.
- The decision to keep the sham only at D = 801 was made from a 25-step development check before the freeze, and is logged at 01:57:57Z.
- The C-C1 development audit and the C-C2 audit ran the same `c_audit.py`: its mtime is 02:01Z and its hash was frozen in C.
- The `queue_C.txt` gate was added in place at 03:34Z, before the freeze, and the frozen hash covers the gated line.

**Correction.** None.

## Overall integrity verdict

**PASS, with minor disclosure corrections.**

- The freeze chronology holds. Every frozen script, design, queue and preflight file is unchanged at HEAD and in the working tree. The only exception is `c_audit.json`, which was extended by the held-out record as designed (finding 3).
- No confirmation output was produced or read before its freeze.
- The three fail-closed scorers refused every incomplete input tested, and they reproduce the committed score files bitwise.
- All 64 + 41 + 14 runs completed exactly once, with no failures or memory problems.
- The continuation set follows the frozen rule.
- Every verdict reported matches the score files: R1-W FALSIFIED, R1-L CONFIRMED, R3 CONFIRMED, R4 FALSIFIED, R5 FALSIFIED, R6 CONFIRMED.

Corrections needed before the final commit:
- **Finding 2:** extend the checkpoint-commit disclosure to `a4f66b7` and `b710b38`, and to `resume.pkl` remaining in history.
- **Finding 3:** add a note on the post-freeze extension of `c_audit.json`.
- **Finding 5:** add errata for the two count slips.

None of these affects any verdict.
