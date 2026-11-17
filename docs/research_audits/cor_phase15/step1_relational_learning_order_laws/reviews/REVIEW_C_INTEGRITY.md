# Review C: experimental integrity of COR Investigation XV

**Reviewer.** An independent agent (Claude). This is an agent review, not human peer review.

**Scope.** I checked:
- freeze chronology;
- blinding;
- run completeness;
- fail-closed scoring;
- the development incident;
- statistical labelling;
- whether the documents report the score files correctly;
- resources and reproducibility.

**What I did not do.**
- I trained no models.
- I modified no existing package file.

**Where the work was done.**
- All scorer runs and tamper tests ran on a copy at `/tmp/revC/cor_phase15/step1_relational_learning_order_laws`. The copy excluded `checkpoints/`. Sibling packages were symlinked so that `../step1_*` resolves.
- Small read-only re-evaluations of saved snapshots (no training) used `/tmp/revC_scripts/evalsteps.py`.

**Documents reviewed** (state at 2026-11-17T09:59Z; first 16 hex characters of SHA-256):

| document | SHA-256 prefix |
|---|---|
| TASK_VARIANT_EXPERIMENTAL_DESIGN | 512f52c3969e7ddd |
| LEARNING_TRANSITIONS_AND_FROZEN_CONFIRMATION | cd5a91a935d15377 |
| OPTIMIZATION_AND_MECHANISM_ANALYSIS | 27fe4da0206aedd3 |
| T4_FROM_SCRATCH_AND_GENERALIZATION | 6a3098f0d581d5f7 |
| COR_FOUNDATIONAL_NAVIGATION_XV | f83924e821bb575a |
| MAIN_RESEARCH_SYNTHESIS | 77ccf297749a88ca |
| REPRODUCTION | 2f36909cabb1b8b7 |

MAIN_RESEARCH_SYNTHESIS and REPRODUCTION were still being written during this review. `logs/VERIFY_CLEAN.txt`, cited in REPRODUCTION, did not exist yet.

**Severity scale.**
- **Critical:** invalidates a frozen verdict.
- **Major:** a frozen-verdict claim is misreported, or integrity is materially compromised.
- **Moderate:** an exploratory or narrative claim is wrong or overstated, or a disclosure is inaccurate.
- **Minor:** a wording, labelling or coverage gap.
- **Verified:** a check that passed, with its evidence.

---

## Part 1: Checks that passed

### V1. The freeze precedes every confirmation run [Verified]

**Freeze commit.** `7957fee` is dated 2026-11-17 05:04:03 +0000.

**Run starts.** The first lane start in `logs/COMPUTE_LOG.txt` is 05:04:17Z (TB, T4ds_s301, and TA, T4ds_s302), 14 s after the commit. All 40 lane starts come later.

**COMPUTE_LOG at the freeze.** At `7957fee` the log contains exactly three lines, the last being "FREEZE XV being committed; confirmation runs launched only after the commit". It has no lane lines.

**File modification times.**
- In each of the 20 `confirm_outputs/*/` directories, every file's modification time lies inside the start/end window of its own logged lane job. The earliest file is 5–19 s after the job start, and `done.json` is within 1 s of the logged end. A scripted check gave 0 violations.
- The earliest confirmation file anywhere is `T4ds_s301/step50.npz` at 05:04:23.
- There were no cross-run writes (no repeat of the development directory collision; see V5).

### V2. Frozen hashes match at the freeze and now [Verified]

All 22 entries of `logs/FREEZE_XV_HASHES.txt` match both:
- the blob at `7957fee` (`git show 7957fee:<path> | sha256sum`);
- the current working-tree file.

This includes the five sibling-package modules. `git diff 7957fee -- scripts src` shows only two new, unfrozen exploratory scripts: `x_mech.py` and `xv_noisy_math.py` (see F8).

### V3. Training was blind [Verified]

**Configuration and output.**
- Every confirmation `done.json` has `"blind": true`.
- No `hist.json` and no leftover `resume.pkl` exists in any `confirm_outputs/` run.
- All 20 training lane outputs (`logs/run/<key>.out`) are 0 bytes.
- The only non-empty `.out` files are the 4 T4 trajectory jobs. They contain only numpy RuntimeWarnings from `bayes.py`.

**No inspection before scoring: access-time evidence.** The filesystem is ext4 mounted `relatime`. Under relatime, the first read after a write sets atime and later reads within 24 h do not move it, so a file's atime records its first read.
- **Snapshots.** In all 20 run directories, the earliest snapshot atime equals the start of that run's *measurement* lane job (09:06:33–09:31:40). No snapshot of any run was opened before its training finished, and none outside the scheduled measurement job.
- **Trajectories.** The atime of all 16 `traj_{A..D}_s*.json` files is 09:37:39.528–09:37:39.545, within 17 ms. That is `score_XV.py`'s validation loop, matching the COMPUTE_LOG scoring line at 09:37:39. The 4 `traj_T4ds_*.json` files were first read at 09:37:46.58–09:37:46.60, which is `score_T4.py`.
- **`done.json`.** The atime of all 20 lies between 09:37:39.53 and 09:37:46.59, so they too were first read by the scorers.
- No other file in the repository or in `/tmp` was written between the freeze and the scoring time except:
  - lane outputs;
  - `__pycache__`;
  - the mathematics-only Workstream F files (05:05);
  - `RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md` (05:06);
  - `x_mech.py` with `dev_mech_C_s100.json` (05:11–05:13, on development seed 100).

This is strong evidence that no confirmation output was inspected before the frozen scorers ran.

### V4. Seed counts and run completeness [Verified]

**Runs.**
- 16 GATE runs: conditions A–D × seeds 201–204.
- 4 DeepSets runs: seeds 301–304.

**Each run's `done.json` matches the design.**
- GATE runs: cond/spec, seed, `stop = 4000`.
- DeepSets runs: seed, `steps = 6000`.
- The scorers recompute the init hash, and it matches.
- By design, the init hash is shared within a seed across conditions (seed pairing). It differs from the development seed 100 hash (`aa7cd6db…`).

**Snapshots.**
- GATE runs: 88 each (80 at 25-step spacing to 2000, plus 8 at 250-step spacing).
- DeepSets runs: 72 each.
- Every trajectory JSON has one entry per snapshot, with the frozen stream and n.

**Jobs.** 40 lane starts and 40 unique claim keys; each key started exactly once (no restarts). There are 40 `rc=0` end lines.

**Measurement integrity.** I recomputed 8 trajectory points from the snapshots and they reproduce the stored values to the printed precision (A_p and G):
- B_s202 at steps 250, 575 and 4000;
- C_s204 at steps 475, 2500 and 4000;
- D_s201 at steps 225 and 4000.

### V5. The scorers reproduce exactly and fail closed [Verified]

**Reproduction.** On the `/tmp` copy, the outputs are byte-identical to the committed files at `180637c` and to the current files:
- `score_XV.py ROOT OUT` gives SHA-256 `0b7b1fb8…`;
- `score_T4.py ROOT OUT` gives SHA-256 `2f25f4cb…`.

**Refusal tests.** Every test below exited with code 2 and wrote no output, unless noted otherwise. After restoring the copy, both scorers again produced byte-identical output.

| # | tamper | result |
|---|---|---|
| T1 | missing `traj_B_s202.json` | "REFUSED: trajectory missing" |
| T2 | missing `B_s203/step1250.npz` | "REFUSED: snapshots missing" |
| T3 | `done.json` with `blind=false` | "REFUSED: config mismatch" |
| T4 | `done.json` with altered init hash | "REFUSED: init hash mismatch" |
| T5 | `done.json` with seed changed to 100 | "REFUSED: config mismatch" |
| T6 | `done.json` with stop changed to 2000 | "REFUSED: config mismatch" |
| T7 | `done.json` deleted | "REFUSED: run not finished" |
| T8 | `done.json` without the `blind` key | KeyError, **rc = 1**, no output (fails closed, but not with the documented exit code 2) |
| T9 | trajectory with one snapshot entry removed | "REFUSED: trajectory mismatch" |
| T10 | trajectory with wrong stream | "REFUSED: trajectory mismatch" |
| T11 | one byte appended to `src/xvtask.py` | both scorers "REFUSED: frozen file changed" |
| T12 | `xv_math_verified.json` with `all_pass=false` | "REFUSED: mathematical verification missing or failed" |
| U1–U6 | T4 scorer: missing trajectory; missing `step3250.npz`; `blind=false`; altered init hash; `steps=3000`; trajectory `n=100` | all REFUSED |

Limits of fail-closed scoring are listed in F7.

### V6. Development and confirmation are separate [Verified]

| | development | confirmation |
|---|---|---|
| GATE seeds | 100 | 201–204 |
| DeepSets seeds | 300 | 301–304 |
| GATE evaluation stream | 99_000_001 | 98_700_000 |
| DeepSets evaluation stream | 99_200_000 | 98_800_000 |
| GATE training data stream | 41_100_000 | 41_201_000–41_204_000 |
| output location | `checkpoints/dev*`, `logs/dev_*` | `confirm_outputs/` only |

No sibling package has `*s201*`/`*s301*` run directories.

### V7. Most document numbers match the score files [Verified]

**LEARNING_TRANSITIONS §1 table.** All 16 rows match `score_XV.json`: t_S, t_I, t_G, width, competent, max A_p before the gate, and final G and KL.

**P1–P3.**
- P1: 8/8; median 1.44; interval [0.99, 1.94]; 6 of 8 censored.
- P2: 5/8; arity-2 ratios 1.44–2.61.
- P3: (+, +, +, +); competent-only calls A and B PRESENT, C and D MIXED.
- The B stage is brief, 150–325 steps (t_G − t_I).

**Sensitivity claims** (`logs/xv_sensitivity.json`). B drops to 1/4 at A_p ≥ 0.6 under every G threshold. A, C and D stay at 4/4 in all 9 cells.

**P6.**
- Source-stage runs of 7–10 snapshots, at 50-step spacing.
- t_G4 700–900; width 150–250.
- Final KL 0.0013–0.0014; G4 0.83–0.84.

**The P5 Spearman values** (−0.80, p = 1.8·10⁻⁴; −0.43; 0.15; 0.46; arity 0.89; gain −0.35; arity-2 −0.26, −0.10, 0.21, 0.02) recompute exactly from `gradsig_summary.json`.

**The mechanism table** values (E2, R2, use E2, use G1) match `logs/expl/mech_*.json`.

---

## Part 2: Findings

### F1 [Moderate]: the development-incident disclosure understates the overwrite, and one retained development trajectory is partly contaminated

**What COMPUTE_LOG (04:22:13Z) and TASK_VARIANT §3 say.**
- The overwrite lasted "~4 min (steps <= ~475 overwritten)".
- The first pair's trajectories "had been evaluated before the overwrite" (A to 1400, C to 1300).

**Evidence that it went further.**
- Both `checkpoints/dev_contaminated/*` directories have modification times of 04:12:11–04:22:02 on every snapshot from step 25 to **step 850**. Step 875 onwards keeps the original 03:43 times.
- The directories also contain a `resume.pkl` written at 04:18 by the killed runs (step 500).
- So the overwrite lasted about 10 min and reached step 850.

**The retained development A trajectory is contaminated.** I re-evaluated the *current* (overwritten) snapshots of the A directory on stream 99_000_001:
- At steps ≤ 475, `logs/dev_traj_contaminated_A.json` equals the overwritten weights exactly. For example, step 200 gives A_p 0.234 and G 0.004 in both; step 450 gives A_p −0.007 and G 0.708 in both.
- At steps 500–850 it differs from them (it holds original-A values).
- The file shows a discontinuity at 475→500: G goes 0.76 → 0.19 and A_p −0.02 → 0.70.
- So entries 25–475 of that file come from the overwriting B-condition run, not from development A.

`dev_traj_contaminated_C.json` differs from the overwritten snapshots at every step tested (100–850), so it is clean.

**Consequence.** TASK_VARIANT §3 says development A had "A_p ≥ 0.5 from step 200". No retained file supports this: the retained file has A_p 0.23 at step 200, and that entry is contaminated. The claim may come from an earlier, unretained evaluation.

**Impact.**
- No impact on the confirmation: different seeds, directories and streams (V1, V6).
- It does affect the provenance of the development evidence from which the RACE hypothesis was formed.

**Correction.** Amend the COMPUTE_LOG and TASK_VARIANT disclosure in a post-hoc note, without editing the frozen text. It should say:
- the overwrite was 04:12–04:22, steps 25–850, and `resume.pkl` was written;
- `dev_traj_contaminated_A.json` entries ≤ 475 are from the overwriting run;
- the "A_p ≥ 0.5 from step 200" development statement has no retained source.

### F2 [Moderate]: the mechanism narrative omits a measured probe site that shows gate information before use

**What `x_mech.py` measures.** Probes at three sites, including `resid_post2@R` (the readout after layer 2).

**What OPTIMIZATION §2 reports.** Only E2 and R2 (`resid_pre2`).

**What the omitted site shows.**
- A_s201 at step 625 (following 0.00, use 0.04 = baseline): post2@R = **0.85**.
- A_s201 at step 775 (following 0.31): post2@R = **0.88**.
- C_s204 at step 2250 (following 0.25): post2@R = 0.70.
- Censored C_s201: post2@R = 0.68 and E2 = 0.62.

**Precision.** The held-out test set is the third of 1500 episodes with x₀ = 2, about 500 cases, so the standard error is about 0.02. The values 0.62–0.70 are therefore clearly above chance.

**Claims this contradicts.** The doc's "not linearly decodable … (0.28–0.65)", "**not** previously stored information becoming usable" (A, B, C) and "never forms a decodable … gate variable" (C_s201) are not supported as written. MAIN_RESEARCH_SYNTHESIS repeats them in Q14, Q16 ("Newly represented in A, B and C") and in "In 3 of 4 analysed models … neither decodable nor usable".

**Correction.**
- Report all measured sites.
- Restate the finding along these lines: "use through the evidence tokens forms during the transition. At the layer-2 readout, g is partly decodable before use in A_s201 (0.85) and D_s203 (0.87), and weakly in C_s204 and C_s201 (0.68–0.70)."
- Drop "newly represented" for A.
- Describe C_s201 as "weakly decodable, not used".

### F3 [Moderate]: the declared P5 analysis was switched to a different headline

**What FREEZE XV §6 declared.** "Spearman correlation of the signal-to-noise ratio **at t_S** with t_G across the 16 models."

**What is reported instead.**
- OPTIMIZATION §1 and MAIN_RESEARCH_SYNTHESIS headline the **initialisation** SNR: ρ = −0.80, p = 2·10⁻⁴.
- The declared statistic is ρ = −0.43 (p ≈ 0.10). It appears only in the table.

**Why the p-value is invalid.** It treats the 16 models as independent. At initialisation, seed-paired A/B and C/D models are the same network: identical init, and the same data stream except the noise law. Their SNRs are near-duplicates (for example 1.544 and 1.540). So there are at most 8 effectively distinct values, and ties at the censored t_G = 4000.

**Correction.**
- Report the declared t_S statistic first.
- Label the initialisation correlation as an additional post-hoc analysis.
- Remove the p-value, or compute it over the 8 distinct initialisations.

The doc's conclusion that "H3 is not distinguished from H1" is unaffected.

### F4 [Moderate]: arity-3 transition widths are limited by snapshot resolution; the "7–30 snapshots" claim is false for them

**What LEARNING_TRANSITIONS Q14 says.** "25-step snapshots resolve every transition into 7–30 snapshots."

**What the data show.**
- Snapshots are every 250 steps after step 2000. C_s204's transition (last G ≤ 0.1 at 2000, first G ≥ 0.9 at 2750) spans **3** snapshot intervals.
- D_s203's transition (1900 → 2500) spans 6.
- C_s204's true width lies anywhere in (250, 750]. That overlaps the arity-2 range of 175–325.

**Consequence.** "The two arity-3 successes were less abrupt, 600–750 steps" (Q13; synthesis Q13) is an upper bound, not an established difference. The synthesis line "16 fresh … Transformers … snapshots every 25 steps" also omits the 250-step spacing after step 2000.

**Correction.**
- Restrict the "7–30 snapshots" statement to arity 2 and DeepSets.
- Report the arity-3 widths as "≤ 600–750 (250-step resolution after step 2000)".
- Withdraw "less abrupt", or mark it as unresolved.

The frozen width definition itself is unchanged.

### F5 [Minor]: the statistical-independence wording is overstated in places

**The design acknowledges the pairing.** TASK_VARIANT §4 rightly states that pairing is variance reduction and that the bootstrap is descriptive. LEARNING_TRANSITIONS §4 repeats that paired seeds are not independent populations.

**Places where the wording reads as independent replication.**
- **P1.** "8/8 seed-paired comparisons" uses 4 independent initialisations twice each: B–C and A–D share seed and init.
- **P3.** "Every one of the 16 models" includes C/D pre-gate trajectories that are near-copies of B/A. t_S and t_I are identical by seed, and parameters differ by at most 0.0056 at step 100. So C's and D's PRESENT calls are largely determined by B's and A's t_I.
- **Synthesis.** "Seed-paired … models … behave identically until the gate matters" is too strong. They are close but not identical: for example B_s201 against C_s201 at step 400 gives A_p 0.50 against 0.57, and accuracy 0.814 against 0.795.

**Correction.**
- State that P1 and P3 rest on 4 independent seeds per arity contrast.
- Say "behave near-identically (identical t_S and t_I at 25-step resolution)".

### F6 [Minor]: post-hoc decompositions appear under frozen headings

**The P2 arity-2 split.** The "4/4 at arity 2" result (LEARNING_TRANSITIONS §2 P2 and Q10 "[frozen; confounded]"; NAVIGATION "Inconclusive"; synthesis P2 and Q10) is a post-hoc split of the frozen 8-pair P2 rule. The pair values come from the frozen scorer, but the subgroup reading is not a frozen test.

**Correction.** Label it [post-hoc subgroup of frozen P2].

**A wording issue in the arity-3 P2 sentence.** "Two ties at 4000, one in each direction" should read "two pairs both censored (r = 0), and one pair in each direction, each involving a censored member".

### F7 [Minor]: limits of fail-closed scoring

The scorers bind configuration, file existence and structure. They do not bind content.

**Gaps.**
1. **Trajectory content.** Trajectory *values* are not tied to snapshots by hash. Test T13: zeroing G in `traj_B_s202.json` was accepted and changed `n_competent`.
2. **Snapshot content.** Snapshot *content* is not checked. Test T15: replacing `B_s202/step500.npz` with C_s202's file was accepted.
3. **Loss history.** A `hist.json` in a blind run is not detected (T14).
4. **Missing key.** A missing `blind` key gives exit code 1, not 2 (T8).
5. **The hash list itself.** `FREEZE_XV_HASHES.txt` is not hashed, and nor is the required `xv_math_verified.json`, which is unchanged since `7957fee`.
6. **Transitive imports.** The transitive imports `fwd.py`, `tasks_ext.py`, `ckpt.py` and `gen_family.py` are not in the hash list. TASK_VARIANT §7 says "Every file the scorer depends on is listed". All four are unchanged in git since `7957fee`.

**Mitigations in this package.**
- Git anchors the freeze.
- My recomputation of 8 trajectory points from the snapshots matched (V4).
- The atime evidence shows the trajectories were untouched between measurement and scoring.

**Correction.** Qualify the TASK_VARIANT §7 sentence. For future freezes, record SHA-256 hashes of the trajectories and snapshots at measurement time, and include the transitive imports.

### F8 [Minor]: material created after the freeze, before scoring

**What was created.**
- `RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md` (05:06). The frozen design cites it.
- `xv_noisy_math.py` with its outputs (05:05).
- `x_mech.py` (05:11).
- `dev_mech_C_s100.json` (05:13). It uses development seed 100 steps 1525–4000 from `dev_contaminated/C…`, which are beyond the overwritten range and so clean.

All of these were written after `7957fee` and first committed in the scoring commit `180637c`.

**Why this is harmless.** None touches confirmation outputs (V3), and none feeds a scorer.

**What is undisclosed.** The mechanism script and its probe sites were fixed after the freeze, though before any output was seen.

**Mechanism checkpoint schedule.** The model-selection deviation (C_s204 and D_s203 replacing seed 201) is disclosed and was forced: they are the only competent arity-3 models. Not disclosed: at 250-step spacing, the declared checkpoint schedule (t_G − 200, t_G − 50, t_G, t_G + 100) collapsed to 2–3 distinct checkpoints for those two models.

**Correction.** Add one disclosure sentence covering both points.

### F9 [Minor]: wrong T4 reference values

T4_FROM_SCRATCH §3 says, "on the confirmation stream", Bayes accuracy 0.751, own-question accuracy 0.588 and KL 0.272.

Those are the **development** stream (99_200_000) values from `dev_t4ds_s300.json`. On the confirmation stream (98_800_000), all 4 trajectory files give:
- Bayes accuracy 0.742;
- own-question accuracy 0.580;
- KL 0.273.

**Correction.** Replace the values. No verdict depends on them.

### F10 [Minor]: resource logging and reproducibility gaps

**Resource logging.**
- There is no peak-memory record for confirmation runs: `peakmem.sh` was used only in development.
- Confirmation runtimes are in `done.json`: 1720–1832 s for GATE runs and 153–155 s for DeepSets runs. They are not summarised in any document. The design states 1966–1970 s.

**Reproducibility.**
- No saved script generated `gradsig_summary.json` or the Spearman table. Both were computed inline; I reproduced the values.
- `confirm_outputs/` (1.1 GB) and `checkpoints/` are git-ignored. Re-scoring requires them, so they must enter the final archive (without `resume.pkl`, which includes the two in `dev_contaminated/`).
- REPRODUCTION cites `logs/VERIFY_CLEAN.txt`, which does not yet exist.

**Correction.**
- Add the runtime summary.
- Save the summary script.
- Confirm that the archive includes `confirm_outputs/`.

---

## Overall verdict

**The frozen confirmation is sound. The narrative needs corrections.**

**Integrity of the frozen verdicts.**
- The freeze precedes every confirmation run.
- All frozen files are unchanged.
- Training was blind.
- Access times show that no snapshot was opened before its measurement job, and no trajectory before the scorer.
- All 20 runs are complete and correctly configured.
- Both scorers reproduce `score_XV.json` and `score_T4.json` byte-for-byte and refuse every structural tamper tested.
- The verdicts **P1 COMPLEXITY_DELAYS, P2 INCONCLUSIVE, P3 H1_COMPLEXITY_FIRST and P6 CONFIRMED** stand.

**What needs correcting.** No finding is Critical or Major. The Moderate findings concern:
- the accuracy of the development-incident disclosure, including a partly contaminated retained development trajectory (F1);
- exploratory mechanism and learning-signal claims that rest on selective site reporting or a switched analysis (F2, F3);
- a resolution-limited claim about arity-3 transition widths (F4).

These should be fixed in a post-review corrections section of MAIN_RESEARCH_SYNTHESIS and the affected documents, without editing the frozen design text.
