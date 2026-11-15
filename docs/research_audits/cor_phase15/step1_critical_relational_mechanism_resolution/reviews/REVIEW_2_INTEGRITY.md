# XIII-C Review 2 — experimental integrity (independent AI-agent review, not human peer review)

Scope: prediction-freeze chronology, development/confirmation separation, leakage, stream reuse, scorer correctness, missing arms, undisclosed
amendments, statistical units (MANDATE_XIIIC §§10, 11, 13). Reviewed at 2026-11-15 ~12:15Z, while confirmation was still running:
C5 and C6 were training; C8 (uP161 21/21/21 l2q) and C9 (uP160 23/23/23 l2q) had not started; lane CA was blocked in `when_ready` on C7/C8.
`logs/score_13c.json` and `logs/bundle_13c.json` did not exist yet, so no verdict was audited. Nothing was trained. The only things I ran
were sha256sum, git, stat, and the scorer/preflight on synthetic bundles **in a scratch copy** (/tmp). No package file was modified apart
from this one being added.

## Summary verdict

**The freeze chronology is sound, and it is better documented than the log alone shows. No must-fix integrity violation was found.**

- 9ec86d6 (population) comes before the first confirmation training job. d6281c6 (freeze) contains no confirmation output. Every file in
  `FREEZE_13C_HASHES.txt` still matches. No analysis or scoring code changed after the freeze.
- Filesystem access times (ext4, `relatime`) give positive evidence that no pre-freeze confirmation checkpoint, metrics file or final
  run-log line was read before 10:21:26Z. At that moment git hashed them, during the freeze commit (details below).
- The scorer implements the written predictions, with a few undisclosed tightenings or loosenings. Several of these let a missing or
  degenerate input pass an arm, or hide a failing arm. They are listed under Should-fix.
- The frozen scorer cannot be changed now. These issues therefore have to be handled in the reporting layer, by disclosure plus extra
  numbers reported beside the frozen verdicts. They must not be handled by re-scoring.

## Must-fix

None at the integrity level, provided the Should-fix items S1–S4 are dealt with in the confirmation report. If the final report states any
verdict **without** the disclosures in S1 (the R3 masking rule) and S2 (no-op passes for "predicted-to-fail" arms), I would upgrade those
items to must-fix.

## Should-fix

**S1. R3: the corr arm can be masked, and it is computed on a subset (score_13c.py l.66–70).**
- (i) When no R3(a) arm is diagnostic, the whole prediction is set to NON-DIAGNOSTIC even if R3(b) fails. I checked this on a scratch
  bundle with corr = +1.0: the verdict was NON-DIAGNOSTIC. The written text ("(b) Across all confirmation units ... corr ≤ −0.7") does not
  say that (b) depends on (a). The code comment ("corr alone cannot confirm") justifies blocking a *confirmation*, not hiding a
  *falsification*.
- (ii) The correlation drops any of the 12 `corr_units` that are missing and is still computed once n ≥ 6. With 6 of 12 units removed, the
  arm still passed. This goes against the RB rule that a missing input fails the arm, and against the text's "all confirmation units and
  references".
- **Reporting fix:** always report the R3(b) arm with its n. If R3 comes out NON-DIAGNOSTIC while R3(b) fails, or if n < 12, say so
  explicitly as a falsified or incomplete arm.

**S2. No-op or degenerate inputs pass the "predicted-failure" arms (R4, R5b).**
- R4 late arms L2, L2plus, L1only and supL2 pass when J-slope < 0.2. An adaptation that did not train at all, for example because of a
  label or optimizer bug that freezes every parameter, passes all four. The scorer does not check that these arms actually learned (acc or
  KL gain over native), and it does not check the claimed parameter matching ("matched to L2"; `ntrain`).
- R5(b) passes when the probe-interchange slopes are ≤ 0.1. The text says "although these quantities are decodable", but the scorer never
  checks decodability (`C2` / `resp_probe_R2` exist in c_causal output and are ignored). A broken probe (span ≈ 0) passes. So does a
  strongly *negative* interchange slope: −0.9 passed on a scratch bundle.
- **Reporting fix:** next to R4, report acc, KL and `ntrain` for every arm. Next to R5(b), report the confirmation decodability R² and the
  signed probe slopes. A "fail" arm whose accuracy did not rise above native, or a probe with R² ≈ 0, should be reported as a non-diagnostic
  pass.

**S3. R6: a missing OTHER arm defaults to R² = 0 (l.106).**
- If `OTHER` is absent, the gap condition reduces to TRUE ≥ 0.4, and the arm passes (checked on a scratch bundle).
- d_rows only writes `OTHER` when a non-mixture head exists. All development units had it, so the risk is low, but this is a missing-input
  pass that the RB rule forbids.
- Also, if d_rows (the script under test) finds no mixture heads in any unit, R6 becomes NON-DIAGNOSTIC rather than failing.
- **Reporting fix:** state per unit whether OTHER was present, and list the mixture heads found.

**S4. Text–code differences that were not stated in the prediction text.** Each is small, but none is disclosed:
- R1(b): a split cell is diagnostic only if its references are ≥ 0.3 apart. The text gives no condition for (b), and (b) has no
  "within 0.25" requirement, unlike (a).
- R2(a)/(b): diagnostic only if the reference really is early, or really is a mixture model. The text says "destined", which implies this.
- R4 and R5 define the "late-readout unit" differently: R4 uses mix ≥ 0.8 and lik ≤ 0.15; R5 uses mix ≥ 0.8 only. R6 "pure" (lik ≤ 0.15
  and ≥ 1 mixture head) also differs from R8 "pure" (lik ≤ 0.15 and mix ≥ 0.8).
- When a unit's metrics are missing, the R4/R5/R7 class gates default to *diagnostic*. This is conservative and fine.
- R7 says "each early-readout confirmation unit", but it scores only the two pre-listed units. A unit that unexpectedly turns out early is
  not tested.
- A partially structured output (e.g. c_causal without `C4`, or d_das without a site or family key) makes the scorer *crash* instead of
  failing the arm. The report should state that a crash counts as FAIL of that prediction.
- Non-diagnostic arms are stored with `passed=True` (R2c intermediate, R6/R8 non-pure). Any arm tally must count only diagnostic arms.

**S5. The freeze record says confirmation outputs "have not been opened". It should also say what happened mechanically.**
- During the freeze commit, git hashed every confirmation file then on disk and wrote each one into the object store:
  - C1 `uP161_fx_L20_W20_D23/*` (all checkpoints, `final.json`, `curve.json`, `resume.pkl`) and its run log `logs/run/68a51b4d4ff5.out`;
  - the partial C3/C4 checkpoints.
- The files were then unstaged. They are not in the d6281c6 tree (verified), but loose blobs remain in the object store:
  - `.git/objects/6c/e69b83…` = C1 `final.json`, written at 10:21:26.758Z;
  - `21/e5d31e…` = the C1 run log, written at 10:21:28.210Z.
- This is harmless: hashing is not inspection. It should still be disclosed, so that a later `git fsck` finding these blobs is not read as
  an amended commit. The reflog shows no amend or reset of a commit.

**S6. A full-run determinism control was descoped, and the documents do not say so.**
- `ft_factorial.py uP161 22 22 22` was "DESCOPED (compute; pre-claimed, not run)" at 07:07Z. It was the planned 4000-step check that
  ft_factorial with L = W = D reproduces the XIII-B diagonal.
- CAUSES §A1 says L = W = D = s reproduces XIII-B "bit for bit". The only evidence is a **50-step** comparison
  (`repro_uP161_22_50.json`, K = 50).
- Every R1 verdict compares new ft_factorial cells with XIII-B diagonals trained by `native_train.py`. The text should say "bitwise for
  the first 50 steps; the full-run control was descoped".

**S7. Statistical-unit wording (§11).** The documents generally comply: CAUSES says "descriptive contrasts, not tests", no CI over parents
is computed, and the bootstrap is episode-level. Remaining points:
- CAUSES A4: "H-A2 data order … rejected in development (data effect ≤ 0.02)" and "H-A3 interaction ≤ 0.02". This is a near-zero effect
  from two n = 1 2×2s in two parents. §11 requires a sensitivity statement: say what data effect could have been detected, given
  episode-level SE and the absence of any replicate cell. Do not write "rejected" or "absent".
- DECISION_LAW B2: "ρ_KL separates the classes … bootstrap intervals do not overlap". These are 8-resample episode bootstraps on 5 units
  from 2 parents. They speak to measurement error, not to separation in a population. The correction box notes the 8 resamples, but the
  separation claim still reads as inferential.
- R3(b) correlates 12 points that are not independent. Intervention runs and their references share initialisation and data, and all 12
  are nested in 3 parents. Three of the points (uP160 s21, uP161 s21, uP160 s23) are **development units** whose ρ_KL and lik_RO1 were
  known when −0.7 was chosen (they are re-measured on fresh streams). Report it as descriptive, with no p-value or CI, and give the value
  without the three references as well.
- `uncertainty_13c.py` (post-freeze) is a within-unit episode bootstrap, which is fine. It must not be presented as uncertainty over runs
  or parents.

**S8. Minor record hygiene.**
- "Lane C1/C2" (training lanes) collides with the unit IDs C1–C9. The lane C1 job at 10:07 is unit **C4**, and the lane C2 job at 10:14 is
  unit **C3**. The freeze line "C1 done; C3, C4 running" is correct by unit ID. Add a lane → unit map.
- The 08:51 DESCOPED lines for `adapt3.py uP160:s:23/s:21` carry the l2q reason, which was copy-pasted.
- The population file says "none of these (parent, L, W, D) combinations exists in development". This is literally untrue for C7–C9: their
  (parent, L, W, D) equal those of the XIII-B reference units, and only the intervention is new. Add "with the intervention".
- "Dev streams 86_xxx_xxx" is not exact. c_causal defaulted to 87_100_000, d_routes to 87_200_000 and adapt3 to 76_000_060 (+offsets).
  None of these overlaps the 88_xxx_xxx confirmation seeds, which I checked, including adapt3's +500k/700k/800k offsets. These sit at
  88_500_060, 88_700_060 and 88_800_060, distinct from the c_causal, d_routes and d_das seeds.

## Confirmed

1. **Population before training.**
   - `CONFIRMATION_POPULATION_13C.md` and `queue_conf.txt` were last modified at 09:11:40 and committed in 9ec86d6 at 09:11:44.
   - Lane C1 started at 09:11:45, and the first C1 checkpoint (step25) is dated 09:12:28.
   - The population (C1–C9, parents, seeds, interventions) and the training queue are unchanged since then (hash OK; no git diff).
   - The trainer (`ft_factorial.py`) is identical in 9ec86d6, d6281c6 and the working tree. All confirmation runs therefore used one fixed
     training code.
2. **Freeze commit contents.**
   - d6281c6 contains none of the 9 confirmation run directories and none of the confirmation run logs (68a51b4d4ff5, 0b1499624279,
     6b56264596fa). I checked by `ls-tree`.
   - The checkpoints it adds are development units: uP160 L23/W23/D21, uP161 L22/W22/D22 ro0, and the dev adapt3 arms. The run logs it adds
     are development jobs.
   - `unit_metrics_ro0peek.jsonl` is the *development* ro0 unit (uP161 22/22/22), despite its name.
3. **No reads before the freeze (positive evidence).**
   - With `relatime`, atime is updated on the first read after a write. Every C1 file has mtime between 09:12 and 10:07 and atime between
     10:21:26 and 10:21:27. So none was read between its last write and git's hashing at the freeze. The same holds for the C1 run log,
     including its final "done acc … KL …" line, and for every pre-freeze C3/C4 checkpoint (atime 10:21:27–28).
   - Lane CA (the first confirmation analysis) started at 10:21:49, after the freeze push.
   - The remote branch head is d6281c6.
4. **Frozen files unchanged.**
   - All 16 entries in `FREEZE_13C_HASHES.txt` verify, including adapt3.py, d_das.py, c_causal.py and fair_fits.py.
   - `git diff d6281c6` on tracked files touches only DECISION_LAW (math-review corrections, labelled), NOTES_MATH_B1 (the same),
     COMPUTE_LOG (appends) and queue_ad.txt (one exploratory dev trajectory job on development units).
   - `PREDICTIONS_13C.md` renders exactly the frozen JSON (text and thresholds).
5. **Pre-freeze code change to adapt3.py (supG/supGFrozen) is legitimate.**
   - It was added at 10:01, before the freeze, after development showed that supL2 fails. It was tested on a development unit
     (`adapt3_dev_supG`, uP161 s21). It is logged in COMPUTE_LOG and disclosed in the arm docstring.
   - It does not touch the training path of the confirmation fine-tunes.
6. **Preflight reproduces.**
   - Re-running `preflight_13c.py` in a scratch copy gives P0 and N1–N8 all OK, and the checks are identical to the committed
     `preflight_13c.json`.
   - The N8 empty bundle gives FALSIFIED for every prediction. Missing top-level inputs therefore fail and are never "inconclusive".
7. **Analysis queue.**
   - Every arm required by R1–R8 is scheduled in `queue_conf_analysis.txt`: unit_metrics on 9 confirmation units plus 7 references;
     adapt3 with all 7 late arms and 2 early arms; c_causal early01; d_das ALL2/k8 on both R7 units; d_rows and d_routes on 6 units each;
     fair_fits on all 12 R3 units.
   - Every arm uses a confirmation stream (88_0/1/3/5/6/7/8xx_xxx).
   - What ran up to 11:59 follows the queue in order, with no skipped or altered lines.
   - The borderline second stream (88_150_000) is pre-declared (NOTES_DEV) and is used only as a reported sensitivity (uncertainty_13c.py).
8. **Descoping and the restart are disclosed and do not bias confirmation.**
   - The container restart at about 07:23Z killed three *development* runs. They were restarted from step 0 after the trainer was made
     resumable, and resume is bitwise (`repro_resume.json`). The A3 end time was recorded manually, and the log says so.
   - Development l2q, the development L2/WA split cells and two dev adapt3 units were descoped. The log marks them, and NOTES_DEV states
     that R1b and R2b are therefore a-priori, risky predictions. This makes the confirmation harder, not easier.
   - No confirmation unit or arm has been descoped so far.
9. **Post-freeze work is kept separate.**
   - The post-freeze exploratory work uses development units only and is labelled [expl]: dense trajectories, and dev adapt3 on uP161 s22,
     later stopped.
   - The documents mark [dev], [conf] and [expl]. They contain no confirmation numbers yet, and they point to a
     FROZEN_CONFIRMATION_AND_STATISTICS.md that does not exist yet.

## Residual risks

- **Agent-controlled timestamps.**
  - Commit dates, COMPUTE_LOG lines and file times are all produced on the same machine by the same agent. No external timestamp exists,
    and the push time of 9ec86d6 is not recorded.
  - The atime evidence is strong but fragile. It would be destroyed by any later reading tool, and the loose git blobs that corroborate it
    disappear at `git gc`. The access record in this review is the only archived copy.
- **Reads during training are not ruled out.**
  - relatime cannot rule out a read of the C1 run log *while it was still being appended* (09:12–10:07). Its content then was training-loss
    lines only. These are low-information about the routing class.
  - `resume.pkl` was overwritten periodically, so reads of it during training leave no trace either.
- **Interim look at confirmation results.**
  - `logs/unit_metrics_conf.jsonl` was read at 12:02:26, after its last write (11:59:01), while confirmation was incomplete. No lane start
    is logged at that time. This is allowed after the freeze, but it opens a door: a decision about the remaining work (C8, C9, d_rows,
    d_routes, fair_fits) could now depend on outcomes.
  - Any later descoping, early stop or re-run of a confirmation job must be treated as missing input, which scores as FAIL, and disclosed.
- **Overlap with development.**
  - Thresholds were set on development units from parents uP161 and uP160. Confirmation reuses uP161 (C1, C2, C8) and uP160 (C7, C9), and
    adds uP163.
  - The confirmation population is therefore 3 parents, 2 of them shared with development. Agreement cannot be read as generalisation
    across pretrained parents. The documents already say that runs are not independent parents, and the final report should repeat it at
    each verdict.
- **Classes of the references were known.**
  - The reference classes (XIII-B diagonals) were known when the diagnostic gates were written. The gates are therefore expected to be met,
    so the risk that gating hides failures is low.
  - The risk shifts to R1: if R1 fails, several later predictions (R4, R5, R7) quietly become NON-DIAGNOSTIC rather than FALSIFIED. The
    report should say so.
- **Cross-script comparison.** The R1 comparisons span two training scripts (S6).
