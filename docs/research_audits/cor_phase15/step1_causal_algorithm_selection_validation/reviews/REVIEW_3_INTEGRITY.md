# Review 3: experimental integrity (Investigation XIII-D)

Independent agent review (not human peer review). It covers HEAD `2532e02` on branch
`cor-phase15-step1-causal-algorithm-selection-validation`, plus the working tree as it stood at about 23:45Z on 2026-11-15. At that time an F3A
confirmation job and three exploratory B5 jobs were still running, and `MAIN_RESEARCH_SYNTHESIS.md`, `REPRODUCTION.md` and
`scripts/verify_clean13d.sh` were being written (untracked). I modified no existing file. Scorers and preflights were re-run on copies in
`/tmp/audit13d/`.

## What was checked, and what holds

| check | method | result |
|---|---|---|
| Frozen-file integrity | Recomputed the sha256 of all 47 entries in `logs/FREEZE{1,2,3A,3B}_HASHES.txt` at the freeze commit (`git show <c>:<path>`), at HEAD and in the working tree. Listed every later commit that touches each file. | 46 of 47 are unchanged everywhere. The exception is `logs/queue_f3a.txt` (finding 9). |
| Outputs after their freeze | mtimes of every `*_f1*`, `*_f2*`, `*f3*` output, every confirmation checkpoint directory (first and last file), and lane start lines in COMPUTE_LOG, compared with commit times (F1 17:54:29, F2 18:49:04, F3a 21:16:38, F3b 22:42:03). | Every confirmation measurement postdates its freeze. The first F1 checkpoint is 17:55:17. The first A-C checkpoint is 18:50:46. The first F3a lane start is 21:16:42. The first F3b checkpoint is 22:43:17. The one exception is the C-C1 adaptation checkpoint and its summary (18:38:35), which COMPUTE_LOG discloses as event 18:44:55Z. |
| Preflights | Re-ran `preflight_f1/f2/f3a/f3b.py` in a copy. | All checks OK: P0 + N1–N5, P0 + N1–N10, P0 + N1–N5, P0 + N1–N4. The regenerated `preflight_*.json` files are byte-identical to the frozen ones. |
| Scorers | Re-ran `score_f1/f2/f3a/f3b.py` on a copy of `logs/`. | `score_f1`, `score_f2` and `score_f3b` are byte-identical to the committed files. `score_f3a` has the same verdicts; it differs only because the descriptive `L0attn` arm has since been written (finding 4). |
| Scorer vs written predictions | Read each scorer against its predictions file line by line. | They agree, including the extra P2a condition "g0 < ½g" (stated in the P2a text), the missing-input → FAIL rule (`all()` over `None`), and P10's NON-DIAGNOSTIC rule. |
| Verdict table | Compared every verdict and deciding number in `FROZEN_CONFIRMATION_AND_STATISTICS.md` with the four score files. | All 13 verdicts match (7 confirmed, 6 falsified). The deciding numbers match, except the characterisation of F1b (finding 3). |
| Retries and inert prefixes | Normalised every retry or derivative queue line (`*_RETRY`, `FX_CONF`, removed gate) and compared it with the frozen queue lines. Grepped `scripts/` and `src/` for the prefix variables. | Every retried confirmation command equals a frozen command. No script reads any `*_RETRY` or `FX_CONF` variable, so the prefixes are inert. `adapt4.py` re-initialises each arm identically (`PRNGKey(4040)`, `default_rng(5)`) and skips arms already present, so the partial F3a retry over a subset of arms is equivalent to the full run. |
| Duplicate analysis records | Inspected `route_status_f1_react.jsonl`. | The duplicate F1e record is identical to the first one, so the "last record wins" rule has no effect. |
| Determinism | `logs/determinism_check.json` | 3 of 3 full-length `ft_factorial` replays are bitwise identical at every common checkpoint, with identical loss histories. No confirmation training run needed a mid-run resume: every F1, A-C and F3b trainer ended rc=0 on its first attempt. |
| Dev outputs vs confirmation units | Searched every `logs/*dev*` file for confirmation unit names. | The only overlap is `uP160:s:23` in `d_rows_dev.json` (finding 11). The overlaps in `unit_metrics_b5dev.jsonl` come from post-confirmation exploratory runs. |

No critical integrity failure was found. The freezes are intact, outputs come after their freezes, and the scorers reproduce the verdicts. The
findings below concern disclosure, reporting accuracy and statistics.

## Findings

### 1. Interim scoring of freeze 2 is undisclosed — **major**

**Evidence.** Commit `84d4ec2` contains `logs/interim/score_f2_interim_1915.json` (mtime 19:11:54) and `score_f2_interim_2000.json` (mtime
20:01:57). Their summaries are:

- 19:15: P1, P3, P4 and P6 FALSIFIED (inputs still missing); P7 and P8 CONFIRMED.
- 20:00: P1 and P6 FALSIFIED; P3, P4, P7 and P8 CONFIRMED.

Neither COMPUTE_LOG, `FREEZE2_PROTOCOL.md` nor any report mentions interim scoring (`grep -ri interim` matches only the files themselves).
The 20:00 look showed P3 CONFIRMED while the A-C1..A-C4 trainers were still running. Several later decisions came after these looks:

- at 20:20Z, the A-C trainers were parallelised;
- at 20:02Z, the P5 development scope was reduced;
- at 21:16Z, freeze 3a was frozen. The P5b prediction is "theory-driven ... from the confirmed carrier trace (P3e)", which uses a result known only from an interim look.

The verdicts are unaffected: the scorer is deterministic, and the A-C runs were fixed and fully queued. The protocol record, however, says
outputs were not looked at, and it is silent about these looks.

**Correction.** Add a COMPUTE_LOG erratum and an item in the "Disclosed protocol events" list. It should give both interim times, what was
visible at each, and the statement that freeze 3a's P5b basis drew on the 20:00 interim P3 result.

### 2. The threshold-sensitivity procedure overwrites the frozen score files, contrary to the report — **major**

**Evidence.** In `scripts/uncertainty_13d.py`, `rescore()` copies the scorer to a temp directory and then runs it with
`LOGDIR = LG` (the real `logs/`), so every perturbed run writes `logs/score_{freeze}.json`. The sweep makes 14 + 48 + 14 such writes.
The file is "restored" by a final unperturbed re-run. `FROZEN_CONFIRMATION_AND_STATISTICS.md` §Threshold sensitivity says the opposite:
"using a temporary copy so the frozen files were never touched".

Consequences:

- The mtimes of `score_f1/f2/f3a.json` (23:39:51–23:40:01) record the restore, not the original scoring.
- An interrupted sweep, for example an OOM kill (common here), would have left a perturbed verdict in a frozen output file.
- If a perturbed scorer run crashes, `rescore()` silently reads the stale file.

I re-scored independently, and the committed files are correct.

**Correction.** Pass a temp copy of `logs/` as LOGDIR, and read the score from that copy. Fix the sentence in the report. Optionally,
regenerate `uncertainty_13d.json`, which should not change.

### 3. The report mischaracterises the frozen F1b arm of P2a, and the reports reinterpret falsified predictions as support — **major**

**Evidence.**

(a) The verdict table says "F1b fails substantively (present 0.74, gain 0.063)". The interpretation says the 250-step block "degrades the
unused route itself". But the frozen presence threshold is `min_present 0.6`. F1b's present value of 0.743 **passes** it. F1b fails on
gain (0.063 < 0.10) and gain-removed (0.066 > 0.05) (`score_f1.json`). By the alternatives frozen with P2a, that pattern is "information
present but not usable without retraining deeper layers". It is not "pathway loss (present < 0.6)".

(b) "Interpretation of the failures" states a "corrected statement" as established: "set within the first ~100 steps ... Mixture is the
default organisation; early is fragile ... Data order does not matter". No frozen prediction tested this; it combines P1, P2b and P10
after the fact. In uP164, the "non-early" state that the statement calls the mixture default is a hybrid. Reference s20 at 1211 has
lik 0.615, mix 0.70; at 4000 it has lik 0.52, mix 0.87.

(c) The draft synthesis (untracked `MAIN_RESEARCH_SYNTHESIS.md`) cites the falsified P2a as support for "Commitment of the readout with
the information preserved". It also recasts the falsified P2b, which predicted that a 100-step mask does **not** flip the organisation, as
"critical period < 100 steps (P2b)".

**Correction.**

- For (a): list F1b's failing criteria (gain and gain-removed) and drop "degrades the unused route" as a reading of the frozen criteria.
- For (b) and (c): label every statement built from falsified predictions as a post-hoc, unconfirmed hypothesis. Do not cite a falsified prediction ID as support.

### 4. Freeze 3a was scored and committed while its confirmation queue was still running — **minor**

**Evidence.** Commit `2532e02` (23:41:07) contains `score_f3a.json`, written at 23:40:01. At that time the frozen F3A job
`e4666549b85d` (arms L1only, L0attn, L0mlp; started 23:12:40) had no end line, and `ps` showed it running at 23:43Z. `score_f3a.json`
has `"lr1:L0attn": [null, null]` and `"lr1:L0mlp": [null, null]`. L0attn was written at 23:43:05. All seven *predicted* arms existed at
scoring time (L1only at 23:29), so the P5a–c verdicts are final. Nevertheless:

- `INFORMATION_ACCESS_AND_REPAIR.md` (lines 59 and 88) points readers to `score_f3a.json` for the descriptive arms, and that file contains nulls.
- The draft synthesis already states "L0attn alone suffices (descriptive)", a result that is in no committed scored artefact.

**Correction.** After L0mlp finishes, re-run `score_f3a.py` and commit the result with a COMPUTE_LOG note. Until then, mark the
descriptive arms as pending.

### 5. Effect-size intervals pool development and confirmation units, mostly without labels — **major (statistics reporting)**

**Evidence.** `uncertainty_13d.py` merges `c_carrier_dev.json` with `c_carrier_f2.json`, `d_routes2_dev` with `_f2`, `d_rows_dev` with
`_f2`, and `e4_noninv_dev` with `_f2`. In the report's table, only the first row says "(dev + 2 conf)". The other pooled rows are
unlabelled: routes n = 5 (2 dev), rows n = 4 (2 dev), E4 n = 6 (3 dev). The development units are the ones whose values set the frozen
thresholds. Pooling them with confirmation units inflates the apparent replication, and it narrows percentile intervals that, with n = 3–6,
already span roughly the observed min–max.

Two smaller issues:

- `routes_max_single_open` takes the maximum over all three routes for mixture units, but P8 scores only Q and R for them.
- The factorial summary mixes A-C outcomes measured on stream 92.1M with references from XIII-B/C streams.

**Correction.** Report confirmation-only summaries, with development units in a separate column. Label n per source. Restrict the routes
summary to the routes each prediction scores.

### 6. The stray lane-line disclosure is inaccurate and incomplete — **minor**

**Evidence.** The report says the stray lines "truncated four stdout logs". COMPUTE_LOG contains nine stray start/end pairs with an empty
command: at 20:20:53, 20:38:22, 20:40:35, 20:41:48, 20:44:13, 21:45:18, 22:21:07, 22:40:15 and 22:42:56. They hit at least six
distinct files: `44e05fb2e80e` (twice), `c55bf7ca5fe7` (twice), `c614e77c4df3`, `860474af5974` (twice), `7fa9fdc58618` and `dc9f5e7c1cda`.

The COMPUTE_LOG entry at 20:52:23Z says the 20:44:13 stray in lane AC2 truncated `588e9c4bf46f.out` (A-C3). That is wrong:

- The stray line carries key `44e05fb2e80e`, which was the lane's last computed key after it skipped the claimed lines 3 and 4.
- `588e9c4bf46f.out` is intact (1247 bytes, starting with the step log).
- `44e05fb2e80e.out`, the stdout of A-C4, which was **running** at the time, now begins with null bytes. That is a sparse file, truncated while the process was still writing.

Two of the affected stdouts belong to confirmation analysis jobs: `7fa9fdc58618` (P1 unit metrics, now empty) and `dc9f5e7c1cda`. Their
outputs are in the JSON files and are unaffected.

The mechanism, bash re-reading an edited script at a stale offset, could in principle have run a fragment of a real command. Here it ran
only empty commands.

**Correction.** Replace "four" with the full list, correct the A-C3/A-C4 attribution, and note the residual risk. In future, never edit a
lane script that running lanes are executing; copy it to a new name instead, as `lane2.sh` later did.

### 7. "Declared before launch" is untrue for C-C1 — **minor**

**Evidence.** The commit message of `a931af7` and the report's chronology table (row "population 2 part 1 ... before launch") say the
population was declared before launch. C-C1 (`adapt4.py uP160:s:23`, lane CAD) started at 18:08:41Z, and the declaration was committed
at 18:29:16. `FREEZE2_POPULATION.md` itself correctly says "launched 18:08Z, not analysed". Part 2 is headed "declared ~19:00Z" but was
committed at 18:49:04.

**Correction.** Change the table row to "declares A-C1..A-C4 and C-C2 before launch; C-C1 launched 18:08Z, before declaration" and fix the
Part 2 time.

### 8. P1 references are not matched on the measurement stream — **minor (control matching)**

**Evidence.** The P1 cells are measured with `unit_metrics.py` on stream 92 100 000. The reference vectors are frozen constants from
`known_outcomes.json`, measured on XIII-B/C streams (`src` `unit_metrics_traj_xiiib.jsonl`). The determinism check confirms that the
diagonal fine-tune *is* the natural model, so only the stream differs. Freeze 3b does this correctly: it re-measures its references on the
cells' stream. The verdict is robust: d_W is 0.009 or 0.039 against d_other ≈ 1.08, and 1.09 or 1.02 against d_other ≈ 0.002–0.13.

**Correction.** Disclose the mismatch as a limitation, or add an exploratory re-measurement of the four diagonals on 92.1M.

### 9. A frozen queue changed after freeze 3a, and the hash record is now stale — **minor**

**Evidence.** `logs/queue_f3a.txt` matches `FREEZE3A_HASHES.txt` at `da9e569` but not at HEAD. Commit `dcf511a` appended two
`AD_RETRY=1` lines, which COMPUTE_LOG discloses at 21:37:16Z and 22:40:44Z. The commands are frozen commands, or subsets of their arms,
with inert prefixes (verified above). Other derivative queues that ran confirmation work were never hashed: `queue_f2_retry.txt`,
`queue_f2A_train2.txt`, `queue_carrier_adapt.txt` and `queue_f1_analysis_retry{,2}.txt`. I verified them equivalent.

**Correction.** Add a FREEZE3A addendum that records the post-freeze hash of `queue_f3a.txt` and the hashes of all retry queues, with a
pointer to the COMPUTE_LOG entries.

### 10. The determinism claim is stretched beyond what was tested — **minor**

**Evidence.** The report (event 4) justifies retries with "every run is deterministic, as determinism_check shows". The check covers
`ft_factorial.py` training only. The retried confirmation jobs were `adapt4.py` (C-C2, three attempts; F3a), `d_rows.py`,
`route_status.py` and `unit_metrics.py`, and none of them was replay-tested. The retries themselves are not outcome-dependent: each
killed attempt left no output. What is unsupported is the cited justification.

**Correction.** Narrow the claim to the training scripts that were tested, and note that the analysis retries were full reruns from
scratch after no-output kills.

### 11. Small independence gaps and wording in the frozen files — **minor**

- Freeze 2's `frozen_before` says "No confirmation unit below appears in any development file". But `d_rows_dev.json` analysed natural
  `uP160:s:23`, the parent of carrier unit C-C1. The adapted C-C1 is a different model, so this is a partial overlap, and it is not
  disclosed. The disclosed P6 overlap (uP161 s20 from XIII-C) is handled correctly.
- Freeze 3a calls uP165 s20 a "new unit". It had already been analysed and scored in freeze-2 P6/P8 before freeze 3a, though not adapted.
  The draft synthesis calls the P5 unit "a new parent", but uP165 also supplied C-C2.
- In P10, cell (a) L20/W21/D21 is described as "early diagonal's W on the non-early L, D", but its D is the early diagonal's (D21). With
  respect to the early diagonal, cells (a) and (b) are L-only and W-only changes respectively. This is the reverse of the labels; the
  labels hold only if D is assumed irrelevant. The scorer is unaffected, but the report's prose ("a change of L alone left the outcome
  early") reads as the opposite of its own conclusion "uP164: L decides".
- Erratum: the frozen 3a file cites "COMPUTE_LOG 21:23Z", and the erratum corrects this to 21:16:31Z. The actual entry is 21:16:38Z.
- The report's decision-rule summary says competence is required at 4000 or 1211. Freeze 3a explicitly has no competence exclusion.
- `uncertainty_13d.py` gives freeze 3b no sensitivity analysis. Its thresholds are in prose, so this is acceptable, but the report should say so.

**Correction.** Add a short errata block covering these items. Rewrite the uP164 sentences with respect to a named reference diagonal.

### 12. Development calibration for P5b was stopped by hand — **noted, no defect**

`adapt4_dev_lr3/uP161_s_21_b0.6_lr0.003.json` contains only `native`, so the claim that no arm was produced is true. The run was stopped
28 s after it started (rc=143), at 21:15:42, and freeze 3a followed at 21:16:38. This is disclosed. Readers should know that P5b's
confirmation (supRO 0.894, supQS 0.727) is a theory-based prediction with no calibration, which the frozen file states.

## Overall integrity verdict

**Pass with required corrections. No critical failure.**

All five freezes are intact. Every predicted confirmation measurement was produced after its freeze. The preflights and scorers reproduce
byte-for-byte, and all 13 reported verdicts (7 confirmed, 6 falsified) match the frozen scorers exactly. Retries are equivalent to the
frozen commands, and the development and confirmation units are independent apart from the disclosed and minor overlaps.

The integrity problems are in disclosure and reporting:

- undisclosed interim scoring of freeze 2 (1);
- a false statement that sensitivity analysis never touched frozen files, with an implementation that does overwrite them (2);
- misreporting of the F1b arm, and post-hoc reinterpretation of falsified predictions as positive findings (3);
- effect-size intervals that pool development units with confirmation units (5).

These should be fixed before the package is final. Freeze 3a should be re-scored once its still-running descriptive arms finish (4).
