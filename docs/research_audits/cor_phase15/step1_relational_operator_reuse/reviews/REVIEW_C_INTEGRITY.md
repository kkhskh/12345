# Review C: experimental integrity and reproducibility (COR Investigation XVII)

This is an agent review, not human peer review. The mandate is `sources/MANDATE_XVII.txt` §10, §11 and §15 (Reviewer C).
The package files were not modified. Scratch work was done in `/tmp/revC17/`. Branch: `cor-phase15-step1-relational-operator-reuse`. Review date: 2026-11-18.

## Checks passed

1. **Freeze chronology (git).** The commit order was checked: `431b1b6` (Stage 1 O/D, 06:26:36Z), `c419773` (FREEZE, 07:24:42Z), `fa02ca0` (development reports, 07:53:27Z), `829dfc1` (scorer run, 13:04:08Z), `007b62e` (reports and outputs, 13:07:15Z).
   - `431b1b6` contains only the mandate, the O/D report, `xvii_od_check.py`/`.json` and `.gitignore`. It has no neural result.
   - `c419773` contains no confirmation output.
   - `fa02ca0` adds only development reports and a development `.out`.
2. **Frozen documents unaltered.** `git diff c419773 -- FROZEN_PROTOCOL_XVII.md COMPETING_NEURAL_MECHANISMS.md scripts src logs/FREEZE_XVII_HASHES.txt logs/XVII_design.json logs/queue_xvii.txt logs/preflight_XVII.json` is empty.
   - Every change in the package since the freeze is an addition (65 604 insertions, 0 deletions in the package diff to HEAD).
   - The 8 modified lines in `007b62e` are in the development reports only.
   - Caveat: the working tree now holds post-review correction banners in some reports. These are not the frozen documents. Of the two frozen documents, COMPETING_NEURAL_MECHANISMS.md now carries such a banner in the working tree. I note it in Finding 6.
3. **Dependency hashes.**
   - All 27 entries of `logs/FREEZE_XVII_HASHES.txt` match on disk (`sha256sum -c`). Each also matches the blob at `c419773`, including the 9 sibling-package files (checked by normalised path).
   - The hash list is identical to the one committed at the freeze.
   - No sibling package has commits after `c419773`.
   - The import closure of the hashed files was traced: `train_tasks`, `tasks_ext`, `gen_family`, `fwd`, `fwd3`, `gatelib`, `nets`, `bayes`, `ckpt`, `xvitask`, `xvidiag`, `xviislot`, `xvi_math_core`. All are hashed.
4. **Blindness: atime and mtime evidence** (ext4, `relatime`). With `relatime`, atime is the first read after the last write. Later reads within 24 h do not update it, so any early peek would have left an earlier atime. Raw record: `/tmp/revC17/stat_record.txt`, taken before any reviewer read.
   - `confirm_outputs/*/traj.jsonl`: first read is exactly the start of each run's frozen analysis job in COMPUTE_LOG, for all 12 runs (for example A3_s801 12:00:22, A2_s701 12:01:11, A2_s708 12:32:54).
   - Snapshot `.npz` files: first read during their analysis (A3_s801 step20 at 12:00:26).
   - `logs/analysis/*.json`: first read for all 12 at 12:39:10. That is the `when_file.sh` release of the learning-arm jobs, 60 s after the last analysis (A2_s707, written at 12:38:09), and is a legitimate frozen-queue read.
   - `confirm_outputs/*/done.json` and `learning/*/slot_traj.json`: first read at 13:03:37–13:03:41, by `score_XVII.py`, which wrote `logs/score_XVII.json` at 13:03:41.
   - `logs/run/*.out` (analysis and learning-arm stdout, which contain F_G values): first read for all at 13:07:14. That is the `git add` for `007b62e`, after scoring.
   - All 12 training `.out` files are 0 bytes (BLIND = 1).
   - The protocol's statement that no output was opened before scoring is supported.
5. **Run completion and exit codes.** COMPUTE_LOG has 48 queue jobs with exactly one start and one end each, all `rc=0`. There are no restarts and no `resume.pkl` (the scorer checks this).
   - 12 training runs, 12 analyses, and 24 learning-arm invocations.
   - 14 learning-arm invocations exited 0 with "not in learning subset". 10 produced arms: s802, s803, s804, s704, s705 × targeted/full.
   - This matches the frozen subset rule: every evaluable arity-3 model plus the first two evaluable arity-2 models by seed. The evaluable arity-2 models are 704, 705 and 707, so the subset takes 704 and 705.
6. **Fail-closed scorer reproduction.** The package was copied to `/tmp/revC17/w/` with symlinks to every sibling `../step1_*`. The two untracked post-hoc files (`logs/DEPENDENCIES.json`, `scripts/deps_inventory17.py`) were removed from the copy. Running `python3 scripts/score_XVII.py . /tmp/revC17/score_copy.json` printed the frozen calls, and the output is **byte-identical** to `logs/score_XVII.json` (`cmp`).
7. **Preflight.** `logs/preflight_XVII.json`, committed at the freeze, records 3 planted scenarios recovered exactly and 5 refusals (rc 2, nothing written). The refusals cover a missing analysis, blind = false, a missing snapshot, a changed frozen file and a missing learning arm.
8. **Seeds and independence.**
   - Design seeds are 801–804 (arity 3) and 701–708 (arity 2).
   - Initialisation is `TT.init("gate", seed)`, and the scorer recomputes and checks the init hash.
   - The data stream is `default_rng(41_000_000 + 1000·seed)`, so the 12 streams are distinct. None collides with the XVI development models (seeds 202, 402, 503).
   - The diagnostic streams (99_700_000 inline, 99_800_000 per slot) are shared evaluation sets, which is appropriate.
   - FROZEN_PROTOCOL §2, COMPETING §4 and FROZEN_CONFIRMATION §2 all say that the replication unit is the model, and that slots and pairs are within-model repeated measures. The scorer counts models, not pairs, and uses exact Clopper–Pearson intervals over models.
9. **Donor and recipient selection applied as frozen** (`scripts/xvii_confirm_analysis.py`, hashed). I checked the recorded outputs against the docstring rule.
   - A = slots with F_G ≥ 0.8 and U = slots with F_G ≤ 0.2 at the selected checkpoint.
   - The selected checkpoint maximises min(|A|, |U|), with ties going to the earliest.
   - Pairs are A × U.
   - The U→U control uses the cyclic next slot in U.
   - The selected steps and A/U sets in `logs/xvii_battery_summary.txt` match the analysis JSONs. The pair counts are |A|·|U| (3, 8, 9, 5, 5, 8).
   - Non-evaluable models are those with no partial checkpoint (701, 702, 703, 706, 708, and 801), as the rule requires.
10. **Question-position indexing.** Slots are indexed 0–5 throughout. The analysis `per_slot[...]["F_G"][q]`, A/U, pairs, and the learning-arm `traj[-1]["F_G"][q]` use the same 6-slot `SlotDiag` ordering. The scorer's `j0 = min(U)` matches the learning runner's targeted loss slot.
11. **Existing (XVI) and fresh models labelled correctly.** Development used C_s202, WB_low_s503 and WA_const_s402 (XVI, replayed), stated as "not confirmation" (FROZEN_PROTOCOL §1). The confirmation cohorts are the 12 fresh runs only.
12. **Report numbers against the logs.**
    - All verdicts, k/n values and CP intervals in FROZEN_CONFIRMATION §3 match `score_XVII.json` (pooled, arity 2 and arity 3).
    - Every cell of the §4 model-level table matches `xvii_battery_summary.txt` and `score_XVII.json` P2_values. This covers [0,2) recipient and control, [0,3) recipient and donor, donor under [2,3) against [0,2), routing L01, and late [0,2) against [0,1).
    - PL targeted_A (0.11–0.29) and full_A (≈1.00) match.
    - Exceptions are listed under Findings.
13. **Labelling and claims.**
    - Reports label evidence [dev], [frozen] and [expl].
    - Agent reviews are explicitly "not human peer review" (MAIN, and every banner).
    - MAIN says the package "is not standalone", and so does REPRODUCTION.md.
    - MAIN and RESEARCH_MAP say that "the O/D operators are not validated".
    - No fictional O/D prediction was frozen (O/D silent, Case D), as §10.2 P6 requires.

## Checks not run

- **Extended refusal battery on the /tmp copy** was not run (permission pending). The command tested 13 tamper cases on the copy: frozen-file edit, sibling dependency edit, missing analysis, blind = false, missing snapshot, missing learning arm, truncated traj, leftover `resume.pkl`, wrong init hash, wrong seed, invalid analysis status, wrong learning-arm checkpoint, missing hash list. The approval request went unanswered, so it was not retried. The evidence for refusals is the preflight record (5 of these cases) plus reading the scorer code, which has an explicit `refuse()` → `sys.exit(2)` before any write for every case above. A reviewer with permission should run the battery.
- **Bitwise reproducibility of `xvii_train.py` snapshots** (short replay of about 100 steps against `step20…step100.npz`) was not run, for the same reason.
- **Independent re-run of the analysis battery** on one model was not run (compute and permission).

## Findings

**F1 (moderate): an undisclosed post-freeze development job ran in parallel with the blind queue.**
- Evidence: `logs/expl/dev_slot_C_s202_full.json` and `.out` have mtime 08:02:10Z. A partial `.out` was committed in `fa02ca0` at 07:53, after the freeze at 07:24:42 and while both lanes were training. The job is not in COMPUTE_LOG, and the FROZEN_CONFIRMATION §1 chronology ends development at 07:24.
- It runs on an XVI model and touches no confirmation output, so blindness is unaffected (F4 evidence). However, it is development compute after the freeze, on a 2-CPU machine also running 2 lanes. Its results feed SLOT_LOCAL_ACQUISITION_RECONSTRUCTION.md.
- Correction: add a chronology row ("07:25–08:02 development per-slot reconstruction of C_s202, XVI model, outside the queue"). State that it could not inform frozen choices, which were already hashed.

**F2 (minor): the wording of "first read" is imprecise.**
- FROZEN_CONFIRMATION §1 and COMPUTE_LOG say "first read of any confirmation output: `score_XVII.py`". In fact the frozen analysis jobs read `traj.jsonl` and snapshots from 12:00:22, and the learning runner read the analysis JSONs at 12:39:10.
- Correction: "first read by the investigating agent (outside the frozen queue): `score_XVII.py` at 13:03:37". The atime record supports that reading.

**F3 (minor): the per-slot acquisition spreads in FROZEN_CONFIRMATION §5 P0 are inconsistent.**
- The report says "spread over 80–160 steps: s802 3700–3760; s803 1360–1440; s804 2580–2680". The listed endpoints, which match `slot_acquisition_step` in the analysis JSONs, give spreads of 60, 80 and 100 steps.
- Correction: "spread over 60–100 steps".

**F4 (minor): the range for the other unacquired slots under targeted training is wrong.**
- FROZEN_CONFIRMATION §5 PL says "0.02–0.19 at 400 steps". From `xvii_battery_summary.txt` (and learning/*_targeted), the slots in U \ {j0} at step 400 are:
  - s705: 0.06, 0.04, 0.01
  - s802: 0.10, 0.08, 0.06, 0.02
  - s804: 0.03, 0.00, 0.02
  - s704 and s803 have |U| = 1.
- The true range is 0.00–0.10. The 0.19 is s804 slot 4, an acquired slot.
- Correction: "0.00–0.10 (3 models with |U| ≥ 2)". The qualitative statement stands.

**F5 (minor): U→U control ranges are quoted inconsistently.**
- §5 says "U→U controls of 0.01–0.12". §7.3 says "U→U control (0.01–0.06)" plus baseline 0.01–0.12.
- The genuine U→U controls (|U| ≥ 2: s802, s804, s705, s707) are 0.04–0.06. The 0.01 (s803) and 0.12 (s704) values are |U| = 1 baselines.
- Correction: "U→U controls 0.04–0.06; single-U baselines 0.01 and 0.12". This does not change P2: every model clears control + 0.4.

**F6 (minor, record-keeping): a frozen document carries a post-review banner, and provenance is not marked.**
- The frozen document COMPETING_NEURAL_MECHANISMS.md now has a post-review banner in the working tree, while the committed freeze version has none. The original text is preserved and withdrawn text is marked, which is good practice.
- Correction: add a line to the banner, "frozen text: `git show c419773:<path>`; banner added post-review". Then nobody can mistake the banner for preregistered content.

**F7 (minor): the P0 denominator and censoring should be stated alongside the verdict.**
- A3_s801 has `t1 = 50` (pooled F_G ≥ 0.05) but no slot acquired, so it is "not acquiring" under the frozen rule and drops out of P0 (n = 11). This is correct per the code, and §7.1 discloses it as censored.
- Make sure the synthesis never cites "12 models" as the P0 or mechanism denominator: P0 uses 11, P1–P5 use 6 and PL uses 5.

**F8 (minor): the trainer docstring is stale.**
- `scripts/xvii_train.py` still says "step<k>.npz: parameters every 500 steps". The code saves every SNAP = 20 steps (also every 500 steps and at STEPS).
- It is hashed, so it should not be edited. Note it in REPRODUCTION.md.

**F9 (informational): the effective sample is small and the tests overlap.**
- The 6/6 calls have a CP lower bound of 0.54, and per-arity results are 3/3 (lower bound 0.29).
- Statistically the reports handle this correctly (model as unit, exact intervals, W5 disclosure that P1 and P3 are not independent of P2).
- No integrity problem was found. The scope of the confirmation is narrow: 6 evaluable models, partial windows of 20–240 steps.

No critical or major findings.

## Verdict

**Integrity: sound. Frozen verdicts: reproduced.**
- The freeze precedes every confirmation run. Frozen code and documents are unchanged and hash-verified, including sibling dependencies.
- Filesystem atime shows that no confirmation output was read outside the frozen queue before `score_XVII.py` ran.
- All 48 jobs completed with rc 0. Selection rules, seeds, indexing and the development/confirmation separation were applied as frozen.
- The scorer reproduces `logs/score_XVII.json` byte for byte from a clean copy.
- Corrections required: F1 (disclose the post-freeze development job) and the number fixes F3–F5. None changes a frozen call.
- Outstanding: the extended refusal battery and the snapshot bitwise replay were not run (permission pending) and should be completed by a reviewer with execution approval.
