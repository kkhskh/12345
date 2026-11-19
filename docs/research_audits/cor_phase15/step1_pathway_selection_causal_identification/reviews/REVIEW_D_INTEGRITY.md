# Review D: integrity and reproduction (post-confirmation, XVIII)

**Reviewer.** An agent in the Reviewer D role (integrity and reproduction). This is **not human peer review**.

**Scope.** Branch `cor-phase15-step1-pathway-selection-causal-identification`, HEAD `a81a178`, plus the uncommitted working tree as of 2026-11-19T04:25Z. The freeze commit is `1547ec9`.

**Method.** All scratch work was done in `/tmp/revD18/`, using copies of `scripts/` and `logs/` and a mirror tree with symlinked siblings. The only file written inside the package is this one. The only training was two 150-step replays, run in the mirror.

**Status of the work at review time.** Stage 2b, the stability continuations, was still running: 2 of 37 were finished at 04:22Z, and the rest should end around 09Z. `logs/measure_c18_cont.jsonl` is therefore incomplete, and nothing here covers the stability readout.

---

## Summary

The core integrity claims hold:
- **Chronology.** The design, scorer and predictions were committed before any confirmation run. The freeze commit is at 19:54:40Z and stage 1 started at 19:54:44Z.
- **Frozen files.** Only one frozen file changed after the freeze: the disclosed two-line output-path fix in `scripts/h3_immediate18.py`.
- **The H3 file is genuine.** I recomputed it byte-identically twice: once with the fixed script, and once with the *frozen* script, whose output did land in `./23`.
- **Blinding.** File access times show no read of any confirmation result before the scorer's own read at 00:22:12Z.
- **Completeness.** Every designed run completed with rc = 0 and was measured exactly once on the correct stream.
- **Scorer.** It reproduces its output when re-run, and it refuses the tamper cases it is meant to refuse.
- **Training.** A baseline and an intervention run replay bitwise.

There is **one major finding**: a pre-registered clamp-validity criterion is breached in the confirmation data and not disclosed. There are also several minor accuracy and disclosure points.

---

## Findings by severity

### MAJOR

**D1. The clamp-validity criterion "|β| > 1 on fewer than 10% of window steps" is breached by all six E-base sha runs. The report says the clamp "worked as designed" and does not mention the threshold.**

What the frozen rule says:
- `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md` §10.6, lines 292–295: "|β| must exceed 1 on fewer than 10% of window steps. Otherwise the sha arms are reported as uninterpretable."
- `score_XVIII.py` implements only the `max_err > 1e-3` part of this rule (line 73). It does not implement the β-fraction part.

What the data show (`logs/score_XVIII.json`, `clamp_checks`), as fractions of window steps with |β| > 1:

| run | fraction |
|---|---|
| E\|811 sha_hi_rolled | 0.13 |
| E\|811 sha_lo_real | 0.12 |
| E\|812 sha_hi_rolled | 0.13 |
| E\|812 sha_lo_real | 0.11 |
| E\|813 sha_hi_rolled | 0.11 |
| E\|813 sha_lo_real | 0.10 |
| all 6 M runs | 0.00 |

Since "fewer than 10%" excludes 0.10, **6 of the 12 sha runs fail** the literal rule.

What the reports say:
- `FROZEN_PREDICTIONS_AND_RESULTS.md` lines 68–71: "The share clamp worked as designed … |β| > 1 on 0–13% of window steps; no sha run was marked UNINTERPRETABLE." The 10% threshold is not stated.

Mitigation:
- The same breach was already present in development: E hi-rolled 0.13 and E lo-real 0.12 (design line 377 onward). The design nonetheless declared "Clamp validity rule (§10.6): passed" (line 384), apparently reading the criterion as applying only to the sha-hi-real validity cell.
- So this reading was fixed before the freeze and is not post-hoc tuning. But the written rule does not say that, and the confirmation report should not present the clamp as unconditionally valid.

Consequence: I re-ran the verdict logic with the literal reading, marking the E sha arms UNINTERPRETABLE (`/tmp/revD18/strict_beta.py`):
- **Frozen analysis (2 pairs):** Hm_window goes from CONTRADICTED to INCONCLUSIVE, and so does H1d_plus_Hm. All other verdicts are unchanged.
- **Post-hoc analysis (5 pairs):** every verdict is unchanged. The contradictions are carried by gsg, gsv and vrg on their own.

The claims that rest specifically on the E sha arms would need this caveat:
- PATHWAY_INFORMATION_VS_ATTENTION_COMPETITION.md §4, "sha-hi-rolled keeps the baseline's share and selects, in 5 of 5 pairs" (3 of those 5 are E pairs);
- the Hm_window row in FROZEN §4.

Required:
- State the 10% criterion and the observed 0.10–0.13 on E.
- State which reading was applied and when it was fixed (pre-freeze, design line 384).
- Report the literal-reading sensitivity above.

### MINOR

**D2. "Stream 95_400_000, which had never been used before" (FROZEN line 19) is false.**
- XIII-E used `RS_STREAM=95400000` and `RS_SETS=95500000` on 2026-11-16 (`../step1_reroll_and_critical_period_validation/logs/COMPUTE_LOG.txt` lines 269 and 272, `scripts/route_status.py`).
- `RS_SETS=95500000` calls `MT.sets(95500000)`, which builds the **same metric sets** the XVIII screening measurement uses.
- Those uses were on D801 development checkpoints. No XVIII threshold was calibrated on them, so I see no route for leakage; the impact is negligible.
- The claim should still be corrected, for example to "not previously used on any XVIII-design checkpoint".
- Data orders 811–813 are genuinely fresh: no other package has `_D81[123]_` checkpoints or queue lines.

**D3. Seed and stream independence is narrower than the reports' wording suggests.**
- **Training data depends only on D.** In `xviii_train.py` line 181, `default_rng(1_000_003*(DS+1)+…)`, so the training data stream is set by D alone. As a result:
  - screening at D = 811 sees exactly the training-data sequence of confirmation at D = 811;
  - parent runs at D = 812 see that of confirmation at D = 812.
- **Fresh-head initialisation depends only on L, W.** `fresh = lambda s: N.init(...)`, so the fresh layer-2 and W_A initialisation is set by L and W alone. Therefore the uP163 M base (s = 23, D = 812) has **identical new-layer init, W_A init and training data** to uP160 M|812. Only the pretrained parent differs.
- **The parents themselves are distinct.** `pre.npz` SHA-256 differs for all of uP160–uP165, and `pre.json` records pretraining seeds 160–165.
- **The measurement streams are distinct as stated:** confirmation 95_400_000, screening 95_500_000, parents 95_600_000, development 86_100_000. Every record in each jsonl carries the right stream.
- **Required:** INDEPENDENT_PARENT_GENERALIZATION.md should say that the parent replication shares fresh-head init and data order with uP160 (exactly for the M base, s = 23) and differs only in the pretrained layers. Claims of "independent data" should be scoped accordingly.

**D4. The H3 deviation is real and harmless, but the disclosure is incomplete.**
- **The misdirected confirmation file was not archived.** It was deleted before `de9787b`. The `23` that `de9787b` deletes is the *development* output (10 lines, keys `E|801` and `M|801`) committed pre-freeze in `0bf8dd6`. So the statement "the recomputed file is identical to the misdirected one" (FROZEN line 24) could not be checked from the archive.
- **I verified it independently.** I ran the frozen script (`git show 1547ec9:…/h3_immediate18.py`) in scratch. It wrote `./23`, which is byte-identical to `logs/h3_immediate_c18.json`. The fixed script gives the same bytes.
- **The bug was visible before the freeze.** The development run had already written `./23` and that file was committed in `0bf8dd6`. The scorer preflight bypasses the H3 script by writing a synthetic `h3_immediate_c18.json`, so it never tested that path end to end.
- **Hash record.** `sha256sum -c logs/FREEZE_XVIII_HASHES.txt` now reports FAILED for this file only. The post-fix hash (`e02c5843…77c8`) should be recorded in the deviation entry.

**D5. Denominators are mixed in the post-hoc text.**
- FROZEN §4 Hm_window row: "sha-lo-real is N in 6/6". PATHWAY §4: "6 of 6 pairs". The post-hoc analysis has 5 evaluable pairs, so the 6 includes the non-evaluable M|813.
- FROZEN §4 Hm_release row: "outcome only P in 3/3" on M. Only 2 M pairs are post-hoc evaluable. This row also omits that the gsg contradiction includes E|811 and E|813: band F against mediator P.
- Use the evaluable count, and report non-evaluable pairs separately.

**D6. Arithmetic error in INDEPENDENT_PARENT_GENERALIZATION.md.**
- Line 64 says "Three of the five candidates cannot even be tested".
- Since 1 of 5 is bistable, **four** cannot be tested: uP161, uP162, uP164 and uP165.

**D7. The scorer is robust, with small gaps.**
- **Re-run on a copy (default hash seed).** `logs/score_XVIII.json` is **semantically identical** but not byte-identical. The `counts` dict in `out["arms"][...]` is built from `set(bands)`, so its key order depends on PYTHONHASHSEED. The output is byte-identical with PYTHONHASHSEED = 1, 2 and 3, and differs with 0. A fix is to sort the keys.
- **`score_XVIII.out`** is identical.
- **Preflight re-run.** `logs/preflight_XVIII.json` is byte-identical, with ALL_PASS.
- **Tamper tests on copies** (`/tmp/revD18/tamper/`):
  - correctly refused (rc 2, no output): one record removed; one duplicated; wrong stream; wrong parent; step-100 trajectory missing; H3 file missing.
  - **accepted silently:**
    1. an undesigned extra record on a wrong stream;
    2. an H3 file lacking a key for a non-evaluable pair (for an evaluable pair the script would raise KeyError with rc 1, not a clean refusal);
    3. H3 values set to 9.0;
    4. a baseline KL edited to 0.5.
  - Items 3 and 4 are unavoidable without record hashes.
  - **Mitigation:** `measure_c18.jsonl` has mtime 00:21:51Z, the end of the last run. That is before scoring, and the file is unchanged since its commit in `de9787b`.
- **Preflight coverage gap.** There is no CONSISTENT scenario for H2g_R, H2g_RV or H4.

**D8. Container restart: disclosed and harmless to conclusions, but the bitwise resume is not verifiable for the two interrupted runs.**
- `lane2.sh` redirects with `>`. The resumed jobs therefore overwrote the pre-restart logs `logs/run/5da35601ebe3.out` and `fe55be8d126a.out`. Both now begin "resumed at step 400".
- Their step-400 `resume.pkl` has since been overwritten by the step-1211 one. A bitwise check of the resume would need 546 or more steps, which is beyond this review's budget, so I did not do it.
- Both runs are uP162 screening seeds, s = 22 and s = 23:
  - s22 has a HYBRID structural class, so it could not supply a class whatever its KL (0.050);
  - s23 has KL 0.188.
- uP162 is non-bistable on all six seeds regardless. The parent rule's outcome cannot depend on these two runs.
- **Stage order.** The order changed from run_all18's "2 → 3 → 4" to "3 → 4 → 2b". This is disclosed in COMPUTE_LOG line 165 and FROZEN line 25.
- **Continuation queue.** `queue_cont18.txt` was produced by hand. I regenerated it from the frozen `continue18.py` logic and `score_XVIII.json`, and it is identical (37 lines).
- **Continuations resume correctly.** Both finished continuations log "resumed at step 1211".

**D9. Stale or uncommitted state.**
- FROZEN §5 still reads "(Added when stages 2b, 3 and 4 finish.)", although stages 3 and 4 finished at 02:51Z and 04:07Z.
- `INDEPENDENT_PARENT_GENERALIZATION.md` and `MAIN_RESEARCH_SYNTHESIS.md` are untracked.
- `REVIEWS_AND_ADJUDICATION.md` was first committed only in the post-results commit `de9787b`. Its pre-results status rests on its filesystem mtime alone (2026-11-18T19:56:42Z, before the first confirmation run ended at 20:05Z).

**D10. Wording.**
- INDEPENDENT_PARENT line 48: "vrg 0.82, P (structural class hybrid)". In fact the structural class is MIXTURE and the causal class is HYBRID, so the struct+causal class is HYBRID.
- FROZEN line 42: "nec_M ≤ 0.006". The E-base maximum is 0.0064 (E|812).
- PATHWAY §4 heading: "Excluded [post-hoc; on 2 evaluable pairs, frozen]" runs two evidence levels together. Split it into a frozen list (2 pairs) and a post-hoc list (5 pairs).

---

## Checks that passed (with evidence)

**1. Freeze chronology.**
- `1547ec9` is at 19:54:40Z; COMPUTE_LOG line 63 has "STAGE 1 confirmation start" at 19:54:44Z.
- Every one of the 47 hashed files matches its content at `1547ec9`.
- `git diff 1547ec9 -- <frozen files>` shows exactly one change: `scripts/h3_immediate18.py`, +2/−2, output path only.
- The development lanes U1 and U2 ended at 19:52Z, so no development run overlapped the freeze.
- The post-freeze commit `aa205cd` (19:56:22Z) touches only two non-frozen reports, before any result existed.

**2. Blind protocol.** The mount is ext4 with relatime, so atime records the first read after the last modification.
- **Confirmation run logs.** All 48 confirmation `logs/run/*.out` have their first post-completion read at 00:24:38Z, which is `git add` for `de9787b`.
- **Checkpoint directories.** Across all 1008 files in the 48 confirmation checkpoint directories, **0** reads fall outside the run's own training-and-measurement window before 00:22:12Z.
- **`measure_c18.jsonl`.** Its first read after the last append (00:21:51Z) is at 00:22:12.98Z, the scorer's first attempt.
- **Screening and parent outputs.** The same check on screening (440 files) and parent (252 files) checkpoint directories also finds 0 early reads.
- **Limitation.** Reads *between* successive appends to the jsonl, or of a `.out` while its run was still training, cannot be detected this way.

**3. Completeness.**
- Confirmation: 48/48 designed runs, all rc = 0; the design, queue and measurement sets are equal, with no duplicates.
- Screening: 22 runs, the 16 in `queue_scr.txt` plus 6 extra seeds 24/25 for uP162, uP164 and uP165, as the frozen rule requires.
- Parents: 12 runs on uP163.
- Every one of these has a step-1211 measurement on its own stream. No run failed anywhere in the log.
- Re-running `score_parents18.py` and `posthoc_parents18.py` on copies reproduces their outputs byte for byte.

**4. Bitwise replay.**
- Runs: confirmation E|811 baseline (`FX_SLOG=own0`), and E|811 vrg, each run to step 150 in the scratch mirror.
- Snapshots: the max |Δ| against the archived `step25/50/100/150.npz` is **0.0** for both runs.
- Logs: `slog.json` and `window_log.json` are byte-identical to the archived ones.

**5. Numbers.** I checked every figure in FROZEN §1–4, PATHWAY §3–5 and INDEPENDENT_PARENT §C1–C4 against `score_XVIII.json`, `posthoc_nobehav18.json`, `measure_*.jsonl`, `h3_immediate_c18.json`, `XVIII_parents.json` and `posthoc_parents18.json`. All of them match within the stated rounding, apart from D5, D6 and D10. This includes:
- the evaluable pairs;
- all frozen, per-base and post-hoc verdicts;
- the full φ_nec table and its bands;
- the M|813 accuracies (0.832–0.839 against 0.873);
- the mediator ranges;
- the release jumps;
- the immediate Δloss values;
- the screening classes and KLs;
- the uP163 φ table.

**6. Labelling.**
- Post-hoc material is labelled in all three reports. Both post-hoc scripts and their JSON outputs carry "POST-HOC (not frozen)".
- The unifying "no useful learning signal" account is labelled [interp; post-hoc; not pre-registered].
- Every review file and REVIEWS_AND_ADJUDICATION.md says agent reviews are not human peer review.
- MAIN_RESEARCH_SYNTHESIS.md lines 5–7: O/D "not validated" (O3); the archive is "not standalone".
- None of the three reports in scope claims otherwise.

---

## Required actions

1. **D1:** disclose the β-fraction criterion, the observed 0.10–0.13 on E, the reading applied, and the literal-reading sensitivity (frozen Hm_window and H1d_plus_Hm become INCONCLUSIVE; the post-hoc verdicts are unchanged).
2. **D2, D3:** correct the "never used" claim, and scope the claims about parent and data independence.
3. **D4:** record the post-fix hash of `h3_immediate18.py`. Note that the misdirected confirmation file was deleted and that the identity was established by re-running the frozen script.
4. **D5, D6, D10:** fix the denominators, the "three → four" count and the wording.
5. **D9:** complete or remove FROZEN §5, and commit the untracked reports.

Optional: sort the `counts` keys in the scorer; make the scorer refuse undesigned records and validate the H3 schema; switch `lane2.sh` to `>>`.
