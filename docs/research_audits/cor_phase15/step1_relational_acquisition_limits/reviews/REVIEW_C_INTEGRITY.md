# Review C: experimental integrity (Investigation XVI)

**Reviewer:** agent Reviewer C, initialised separately from the authoring agent. This is an agent review, not human peer review.
**Mandate:** MANDATE_XVI sections 8, 9 and 11 (Reviewer C).
**State reviewed:** branch `cor-phase15-step1-relational-acquisition-limits`, HEAD `6fc87b4`. During the review the working tree also held uncommitted edits to `LEARNING_SIGNAL_AND_STAGE_VALIDATION.md`, `MAIN_RESEARCH_SYNTHESIS.md` and `logs/expl/expl_summary.json`, plus the untracked files `REPRODUCTION.md` and `scripts/check_reports16.py`. Where a finding concerns those, it says so.
**Method:** read-only. All scratch work was in `/tmp/revC16/`. No existing package file was modified. The CPU work was light: one scorer run plus 13 refusal runs of about 5 s each, a 500-step replay of 2 min 42 s, and `xvi_stats.py`.

Overall, the preregistration survived intact and every frozen verdict reproduces exactly. The blinding claims in the reports do not: confirmation trajectory files were opened during the blind period (finding 1).

---

## Checks that passed

1. **The freeze came before confirmation, and the push is externally timestamped.**
   - Freeze commit `737393b`: 16:19:51Z (git and reflog).
   - GitHub activity API for the branch: `branch_creation 0000000 -> 737393b` at **16:19:57Z**.
   - First confirmation launch: 16:20:02Z (COMPUTE_LOG).
   - Scoring commit `e756bcd` was pushed at 03:18:52Z. Later pushes: 04:07:40Z and 04:10:22Z.
2. **Frozen files are unchanged.**
   - `sha256sum -c logs/FREEZE_XVI_HASHES.txt`: all 32 OK. The list is byte-identical to the one in `737393b`.
   - `git diff 737393b` (HEAD and working tree) is empty for `FROZEN_PROTOCOL_XVI.md`, `scripts/score_XVI.py`, `scripts/xvi_train.py` and `src/`.
   - The only file from the freeze commit modified since is `logs/COMPUTE_LOG.txt`, and only by appending (0 removed lines).
3. **Required runs are complete (18/18).** For every run:
   - `done.json` matches the design: cond, seed, schedule, steps, diagnostic interval and init path;
   - `blind = true`;
   - `traj.jsonl` has exactly one record per diagnostic step, from 0 to STEPS (301 / 81 / 161 lines);
   - every 500-step snapshot and the final snapshot exist;
   - no `resume.pkl` is left over.
   - All 20 lane `.out` files are 0 bytes, so the blind runs printed nothing.
4. **Checkpoint provenance.** For all 6 continuations, the `init_sha256` in `done.json` equals the parameter hash I recomputed from the XV step-4000 snapshot it names. The six XV snapshots also match their file hashes in the freeze list and in `logs/DEPENDENCIES.json`.
5. **Model independence is reported correctly.**
   - The 12 fresh runs use 8 distinct initialisations.
   - Each WB_low/WB_high pair has identical init hashes, as designed (the pair is the unit of analysis).
   - The WA fresh seeds (401–404) and WB seeds (501–504) are disjoint from each other, from XV (201–204) and from development (100, 150).
   - The 6 continuations come from 4 XV initialisations (C_s201/D_s201 and C_s202/D_s202 share an initialisation and a stream). This is disclosed in FROZEN_CONFIRMATION §2.
6. **Fail-closed scoring reproduces byte for byte.** I ran `score_XVI.py` on a `/tmp` copy with sibling symlinks to `../step1_*`. The output is byte-identical to `logs/score_XVI.json` in the working tree, at HEAD and at `e756bcd`: P1 BROAD, P2 CONT_BROAD, P3 INCONCLUSIVE, P5 STAGE_ROBUST. An independent re-implementation of t_A from the raw trajectories matches the scorer in 18/18 runs. The P3 INCONCLUSIVE branch, which preflight never tested, is correct by hand: pair log ratios 0.48, −0.35, −0.13, 0.71 give 2 "speeds", 1 "slows" and 1 "no effect".
7. **Refusals work.** Each of the following exited with code 2 and wrote nothing:
   - a dropped trajectory line;
   - `blind = false`;
   - a leftover `resume.pkl`;
   - a missing snapshot;
   - a missing `done.json`;
   - a changed frozen source file;
   - a config mismatch;
   - an init-hash mismatch;
   - a NaN field;
   - `all_pass = false` in the math verification (caught as a changed frozen file);
   - a missing sibling package;
   - a missing hash list.
8. **Reproducibility is bitwise.**
   - My own replay of WA_const_s401 to step 500 (`xvi_replay.py`) matches `confirm_outputs/WA_const_s401/step500.npz` exactly (max |diff| = 0).
   - The archived replays also match the confirmation snapshots exactly at WB_high_s501@500, WA_cont_C_s201@500, WA_cont_C_s201@1000 and WA_const_s402@1000.
   - `replay_check_dev.txt` (0.0) and `resume_determinism_check.txt` (WB_low_s503@2000, 0.0) are consistent with this.
   - `xvi_stats.py` rerun on the copy gives a byte-identical `logs/xvi_stats.json`.
9. **External dependencies match.** All 16 hashes in `logs/DEPENDENCIES.json` match. The installed versions (Python 3.13.18, jax 0.12.0.1, numpy 2.5.3, optax 0.2.8, scipy 1.18.1, sklearn 1.9.1) match the record.
10. **The resumed runs are sound.**
    - The step-1500 snapshots of WB_low_s503/s504 were written at 01:04:17 and 01:04:32. The lanes were relaunched at 01:08:57, after their claims were cleared.
    - The resume restores parameters, the optimizer state, the generator state and cumulative exposure. Trajectory lines after step 1500 are truncated.
    - Because 500 is a multiple of the diagnostic interval of 25, resuming does not distort `train_loss`.
11. **Dev runs, the WB development decision and the arity-4 rule are declared before the freeze.** The dev runs were stopped at the freeze and committed in `737393b`. The absence of a WB development run and the arity-4 rule are both stated in the frozen protocol, and no arity-4 run exists.
12. **Exploratory analyses did not use confirmation data during the blind period.** By modification and access times:
    - every `logs/expl/*XVI*` file was written after 03:17:41;
    - no confirmation snapshot was read before scoring (0 snapshots with access time after modification time and before 03:17:41);
    - the work during the blind period used XV checkpoints only.
13. **The report numbers I checked match** `score_XVI.json` and `xvi_stats.json`: t_A, exposures, widths, regressions, near-gate-free steps, final KL / F_G, the Clopper–Pearson intervals, FAST_UNLOCK (median 875), the matched exposure (E* = 5.90, 4/6 vs 4/4, Fisher p = 0.47), the P3 ratios and means, the Mann–Whitney and sign-test p values, the stage-sensitivity counts, the acquisition-by-exposure table, and the complexity ratio (4.8, range 1.9–9.1).
14. **Reports: wording.**
    - The synthesis says the reviews are "written by independent agents. They are not human peer review."
    - The synthesis and REPRODUCTION.md state that the archive is not standalone.
    - Post-hoc and exploratory material carries [expl] / [spec] labels, and the post-hoc condition split is explicitly labelled descriptive.

---

## Findings

### 1. [MAJOR] Confirmation trajectory files were read during the blind period; the "first read = scorer" claim is false

**Evidence.** The root filesystem is ext4 mounted `relatime`. A file's access time therefore updates on the first read after a write (and otherwise only once per 24 h). The `traj.jsonl` access times of 10 of the 18 confirmation runs fall **before** the scorer ran (03:17:35–41):

| access time (UTC) | traj.jsonl read | finished at |
|---|---|---|
| 11-17 19:42:31.86 | WA_const_s401, s402 | 19:14 |
| 11-17 23:19:51.35 | WA_const_s403, s404, WA_cont_C_s201, C_s202 | 21:57–22:39 |
| 11-18 01:08:37.69 | WB_low_s501, s502 | 00:47 |
| 11-18 03:04:56.66 | WB_high_s501, s502 | 02:26–02:27 |

- The other 8 trajectories, and **every** `done.json`, show access time 03:17:35, which is the scorer.
- Only `traj.jsonl` files were touched. No snapshot or `done.json` was.
- The reads come right after other session activity: `/usr/bin/uptime` was accessed 36 ms before the 19:42:31 reads, and a new agent session directory was created 0.2 s before the 01:08:37 reads, just after the container restart.
- Reads of files still being written cannot be detected this way, because the next write hides them. The 4 occasions above are therefore a lower bound.

These reads contradict:
- FROZEN_PROTOCOL_XVI §6: "No confirmation output is read before every run completes and `score_XVI.py` runs";
- COMPUTE_LOG 03:17:33: "first read of confirmation outputs = frozen scorer score_XVI.py";
- FROZEN_CONFIRMATION §1: the table row "03:1x first read of any confirmation output: score_XVI.py", and the sentence "No confirmation file was opened."

The same paragraph of §1 also says progress "was monitored only through line counts of the trajectory files". A line count opens and reads the file. The pattern is consistent with that: groups of `*/traj.jsonl` files, including an apparent check of WB_low_* when deciding which jobs to resume after the restart. But file-time evidence cannot tell `wc -l` apart from `tail` or `cat`, and `/usr/bin/tail` was first used in the window at 23:19:45. So I cannot establish whether any outcome values were displayed.

**Impact on verdicts.** None found:
- every frozen file is byte-identical to the freeze, and the protocol is unchanged;
- the verdicts are produced mechanically by the hashed scorer and reproduce byte for byte;
- nothing that could steer a verdict was edited after any of these reads.

The problem is that the blinding and chronology statements are inaccurate, and a frozen integrity rule was breached as written.

**Observation (no finding).** The three runs chosen for the post-scoring transition replays were WA_const_s402, WB_high_s501 and WA_cont_C_s201. `run_replays.sh` was written at 03:18:56, 75 s after scoring. All three are among the runs whose trajectories had been opened earlier. The windows can also be derived from `score_XVI.json` in that time, and the work is labelled exploratory, so I do not count this as a finding.

**Correction.**
- Replace "No confirmation file was opened", "first read of any confirmation output: score_XVI.py" and the COMPUTE_LOG claim with an accurate statement: trajectory files of completed runs were opened by monitoring commands at the 4 times listed. State what was run if a record exists, otherwise say it is unknown. State the relatime limitation.
- Record this as a protocol deviation of the §6 rule in FROZEN_CONFIRMATION §5 and in the synthesis Disclosures.
- Future investigations: monitor progress only through lane-side counters, or through file existence and size (`stat`), never by opening output files.

### 2. [MODERATE] The freeze contains two different P2 expectations

**Evidence.**
- The hashed `scripts/score_XVI.py` docstring says P2 "Frozen expectation: CONT_BROAD".
- `FROZEN_PROTOCOL_XVI.md` §5 says "Expectation: CONT_SELECTIVE, with no FAST_UNLOCK".
- Both are in `737393b`. The scorer was last modified at 16:10 and the protocol at 16:19:23, so the protocol was evidently revised after the D_s100 development result, without updating the scorer.
- EXTENDED §2, FROZEN_CONFIRMATION §3 and the synthesis all call the P2 expectation "falsified". None mentions the conflict. Read against the scorer docstring, the expectation was met.

Verdicts are unaffected, since expectations do not enter the rules.

**Correction.** Disclose the conflict wherever the "falsified" statement appears. State that the protocol is the governing record (it was written later and its rationale is the pre-freeze development run). Qualify "expectation falsified" accordingly.

### 3. [MINOR] Preflight coverage is overstated

**Evidence.**
- FROZEN_PROTOCOL §6 says the two synthetic scenarios cover "all eight verdict branches". The rules have 13 branches (P1 3, P2 3, P3 4, P5 3).
- Preflight covers 8 of them: BROAD, RARE, CONT_BROAD, CONT_RARE, BENEFIT_SPEEDS, NO_EFFECT, STAGE_ROBUST and STAGE_NOT_ROBUST.
- Never exercised: SELECTIVE, CONT_SELECTIVE, BENEFIT_SLOWS, MIXED, and **INCONCLUSIVE**, which is the realised P3 verdict. I checked that branch by hand (passed check 6).
- Preflight also tested only 5 of the 12 refusal paths I exercised.

**Correction.** In FROZEN_CONFIRMATION, without editing the protocol, state "8 of 13 branches" and note that the realised INCONCLUSIVE branch was verified by hand or by a post-hoc test.

### 4. [MINOR] The deviation list misses the second container restart; restart timing is imprecise

**Evidence.**
- FROZEN_CONFIRMATION §5 lists only the ~01:00Z restart. The second restart is mentioned only in the synthesis: boot time 04:06:35; it interrupted `gs2_XVI_cont_C_s201_500`, which was rerun at 04:08:50.
- COMPUTE_LOG records neither the second restart nor any exploratory job.
- The first restart came **after 01:04:32**, when the step-1500 snapshots were written, and before 01:08:22. "About 01:00" is too early.

**Correction.** Add the second restart to FROZEN_CONFIRMATION §5 and COMPUTE_LOG, as an appended, dated entry. Give the first restart as "between 01:04:32 and 01:08:22Z".

### 5. [MINOR] Small chronology inconsistencies around the freeze

**Evidence.**
- FROZEN_CONFIRMATION §1 gives the development stop as 16:19:13. COMPUTE_LOG gives 16:19:23.
- The protocol, the log and the report all say C_const_s150 stopped "at step 2000", but its committed `traj.jsonl` runs to step 2050 (42 records), like D_cont_s100. The F_G maximum of 0.059 is unchanged.

**Correction.** Use the logged time, and state that both development runs stopped at step 2050.

### 6. [MINOR] No archived memory or resource log; resumed-run timings are understated

**Evidence.**
- Memory is reported as "about 0.7–1.0 GB … (from `top`)". No memory or RSS record is archived. `scripts/peakmem.sh` exists but was not used in either queue, and "No OOM events" cannot be checked.
- `done.json` `seconds` counts only the time after resuming (WB_low_s503/s504: 1793 / 1785 s), because `t0` is reset on resume.
- The reported "47–49 min per 4000-step WB run" leaves out WB_low_s501/s502, at 45.0–45.2 min.
- Exploratory XV jobs shared the 2 CPUs with the two training lanes during confirmation. The likely effect is the slower first block (10,454–10,483 s against 9,729–9,756 s). Results are unaffected, because training is deterministic.

**Correction.**
- Say that memory figures are unlogged observations.
- Give the WB range as 45–48 min, and note that `seconds` excludes pre-restart time for the two resumed runs.
- Note the CPU sharing in §6.
- Future: wrap jobs in `peakmem.sh`.

### 7. [MINOR] Some exploratory numbers have no generating script

**Evidence.**
- The EXTENDED §6 optimizer table (plateau and transition gradient and update norms) appears in neither `xvi_stats.json` nor `expl_summary.json`, and no archived script computes it.
- The same is true of the "mean confidence 0.785–0.81 → 0.90–0.91" statement in EXTENDED §5.
- My median recomputation over F_G < 0.1 after warm-up gives similar values, but outside some stated ranges. Examples:
  - fresh plateau update 0.032–0.037, reported 0.033–0.040;
  - censored D plateau update 0.043–0.054, reported 0.049–0.058.
- The definition is therefore unstated.

**Correction.** Add a script and its output (for example in `xvi_expl_summary.py`) that defines plateau and transition windows and reproduces the table and the confidence figures. Otherwise mark these numbers as unreproduced.

### 8. [MINOR] The scorer does not authenticate output values; outputs were committed after the exploratory work

**Evidence.**
- The scorer checks structure, configuration and init hashes, not content. Setting every F_G of WB_low_s501 to 0 was accepted (rc = 0).
- The confirmation outputs were first committed in `4fff13e` (04:07Z), about 50 min after scoring and after the post-scoring exploratory work.
- Mitigation: the committed outputs rescore byte-identically, and the replays reproduce the confirmation snapshots bitwise. There is no sign of tampering.

**Correction.** Disclose. Future: write an output hash manifest at scoring time, and commit it together with `score_XVI.json`.

### 9. [MINOR] The freeze hash list does not include the protocol and some freeze-time artefacts

**Evidence.** These are absent from `FREEZE_XVI_HASHES.txt`:
- `FROZEN_PROTOCOL_XVI.md`
- `scripts/preflight_XVI.py` and `logs/preflight_XVI.json`
- `scripts/lane2.sh`
- `logs/xvi_value_grid.json` and `logs/xvi_math_arity4.json`
- `scripts/xvi_value_search.py`

They are protected by the pushed freeze commit (GitHub timestamp 16:19:57Z) and are unchanged since, so this has no practical effect.

**Correction.** Mention it in the reproduction notes. Future: hash the protocol as well.

### 10. [MINOR] The arity-4 pilot rule depended on investigator discretion

**Evidence.** The second condition ("the session continues for at least 6 h after the queue") was under the author's control, and the choice was made after P1 = BROAD was known. Because no arity-4 data exist, no outcome-selective choice was possible. FROZEN_CONFIRMATION §5.6 and HIGHER_ORDER §5 report the choice openly.

**Correction.** Add one sentence saying the condition was discretionary and was decided after scoring, before any arity-4 data existed.

### 11. [MINOR, uncommitted working-tree edits] Count and wording slips in the learning-signal section

**Evidence.**
- "31 estimates on 25 checkpoints": the 31 `gs2_*` estimates cover 27 distinct checkpoints, 26 excluding the initialisation. The committed text ("30 on 24") has the same kind of error.
- The new row "XVI restart C_s201 (first response 1150 after restart)" conflicts with the frozen t_first = 1200. F_G is 0.099 at step 1150, below the frozen 0.2 threshold.

**Correction.** Fix the count. Use the frozen t_first = 1200, or say "first nonzero F_G at 1150".

---

## Not yet reviewable

The ZIP archive, `MANIFEST.sha256` and `logs/VERIFY_CLEAN.txt` (mandate §13) do not exist yet. `REPRODUCTION.md` (untracked) points to `logs/VERIFY_CLEAN.txt` as if it were already archived. No report claims delivery. The archive and clean-extraction check must be reviewed once they are produced.

---

## Overall verdict

**The preregistration is sound, and all four frozen verdicts stand: P1 BROAD, P2 CONT_BROAD, P3 INCONCLUSIVE, P5 STAGE_ROBUST.**
- The freeze was committed and pushed before the first confirmation run.
- Every frozen artefact is unchanged.
- All 18 runs are complete, blind and correctly configured.
- The continuations start from the hashed XV checkpoints.
- The fail-closed scorer refuses correctly and reproduces its output byte for byte.
- Training replays bitwise.

**The blinding account is not accurate.** Ten confirmation trajectories were opened during the blind period, while the protocol, the compute log and the confirmation report say no confirmation output was read before scoring (finding 1, major). Because the frozen machinery fixed every verdict, I find no effect on any verdict. The reports must still be corrected before acceptance.

Required before acceptance: findings 1 and 2. Recommended: findings 3–11.
