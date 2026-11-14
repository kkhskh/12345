# Independent experimental-integrity review: Investigation XIII (step1_relational_decision_computation)

Reviewer worked read-only on the package. All checks ran in `/tmp/review_integrity/`: the sandbox scorer and merge are in `sim_tests.py` and `sim_out.txt`, and the preflight reproduction is in `sim/`.
Review time: 2026-11-14 ~18:45Z. At that point uP160's T4 fine-tune was finished and its das_decoy E was running. No cohort analysis output was read.

## Verdict summary

- **Freeze integrity holds.**
  - All 27 hashed files match `git show 47506c3:<path>` byte for byte, and they are unchanged in the working tree.
  - 47506c3 is the remote branch head.
  - No uP16x output existed before the freeze commit.
- **Scorer = spec.** The scorer evaluates the JSON expressions faithfully, with the stated pass rule, min_units, mixture-only filter and gate.json completeness.
- **Queue = protocol.** The queues produce every input the scorer reads, under the expected names, env vars and directories.
  - I ran the verbatim merge code and the frozen scorer on six synthetic `uP16x` units built from real dev outputs. The whole confirmation path worked (tags, `_conf` suffix, `adapt_sel_conf`, gate.json).
- **There are four blocking risks.** None of them changes a frozen prediction, but each can turn a result into a silent *inconclusive*, or hide a crash or a falsifying arm:
  - the mixture-head definition differs between the scorer and the analysis scripts;
  - `lane.sh` always logs rc=0;
  - `conf_merge.sh` has a race and does not stop on errors;
  - the verdict rule lets *inconclusive* override *fail*, and a missing null counts as diagnostic.
- **The preflight is a plumbing check, not a power check.** It is honest about which arms pass on nulls. But the three trivial nulls cannot fail a sane scorer, and the ALTMECH values for P3, P4 and P5 were set by hand.
- **Disclosure gaps:**
  - a post-freeze dev result (vP150 fails P5c and P5d) that bears directly on P5;
  - das_decoy reuses streams 64M/64.5M, contrary to the "74M+ / 73M+" claims;
  - two pre-freeze fixes are not in COMPUTE_LOG;
  - "size-matched" decoy heads are not size-matched;
  - unfrozen post-freeze scripts (`stats_xiii.py`, `transition_predictor.py`);
  - the push timing, which is the external anchor for the freeze.

## Confirmed

1. **Hashes.** For every entry in `logs/FREEZE_XIII_HASHES.txt`, sha256(`git show 47506c3:<pkg>/<path>`) equals the listed hash, and so does the working-tree file.
   - `git diff 47506c3` over `scripts src logs/predictions_xiii.json logs/queue_*` shows changes only in:
     - `logs/queue_dev_ad3.txt` (a dev queue);
     - `scripts/sufficiency_boundary.py` (not used by scorer or queues).
   - Other post-freeze changes:
     - COMPUTE_LOG (append-only);
     - MATHEMATICAL_SUFFICIENCY doc;
     - dev logs (`*_pre.json`, `sufficiency_boundary.json`, `build_eval_conf13.out`);
     - new untracked `stats_xiii.py`, `transition_predictor.py` and other docs.
2. **Timeline.**

   | event | time (UTC) |
   |---|---|
   | frozen files last modified | between 16:31 and 18:03:50 |
   | `FREEZE_XIII_HASHES.txt` | 18:04:03 |
   | commit 47506c3 | 18:04:13 |
   | lanes T1, T2, C1, C2 start (`ps` lstart, COMPUTE_LOG) | 18:04:13 / 18:04:14 |
   | `checkpoints/native_train/uP160_*` created | 18:04:15 |
   | first T4 output, `step50.npz` | 18:05:15 |
   | GitHub activity API: branch created at 47506c3 | 18:04:32 |
   | uP160 `final.json` | 18:41:02 (after `step4000.npz`) |

   No `*uP16*` or `*_conf_P16*` file predates the commit. The external anchor (the push) precedes every T4 output, but it follows the launch by about 18 s.
3. **Scorer logic.**
   - Pass rule: `math.ceil(0.75*n)` (n = 2→2, 3→3, 4→3, 5→4, 6→5).
   - n < min_units, or any missing unit, gives *inconclusive*. Zero eligible units gives *inconclusive*; it cannot pass (sandbox S6, and P6 with n = 0).
   - Mixture-only filter: uses the frozen rule (mass ≥ 0.8 AND share ≥ 0.2) on l2_scan.
   - COMPLETE comes from `checkpoints/zP16x/gate.json["complete"]`. pretrain_gated.sh computes it from the last block's aux_acc ≥ 0.99.
   - Expression substitution is safe. Placeholders are replaced positionally, and eval sees only abs, min and max.
   - Paths: `J(name)` reads `logs/<a>_conf.json`. The adapt path is `logs/adapt_sel_conf/<tag>.json`, where tag `uP160_full_c1_s4000_seed20_eval_conf13` equals `ftlib.load_model` `M['tag']`.
4. **Queue/scorer naming.**

   | script | env vars read | output |
   |---|---|---|
   | l2_scan | `L2_STREAM`, `L2_TAG` | `l2_scan_conf_P16x.json` |
   | l2_mix_causal | `MC_STREAM`, `MC_TAG` | `l2_mix_causal_conf_P16x.json` |
   | l2_mediation | `FA_EVAL=stream:N` (parsed as `EVP[7:]`), `MD_TAG` | `l2_mediation_conf_P16x.json` |
   | score_contrast | `SCC_STREAM`, `SCC_TAG`, `DECOY_DIR`, `XII_RDIR` | `score_contrast_conf_P16x.json` |
   | l1abl_mix | `LA_STREAM`, `LA_TAG`, `DECOY_DIR`, `XII_RDIR` | `l1abl_mix_conf_P16x.json` |
   | decoy_disc | `DD_STREAM`, `DD_TAG` (argv dir `logs/decoy_conf`) | `decoy_disc_conf_P16x.json` |
   | das_decoy | `XI_OUT=decoy_conf`, `DD_EVAL` | `logs/decoy_conf/<tag>_<RULE>_R.npy` |
   | adapt_sel | `AD_OUT`, `AD_STREAM` | `logs/adapt_sel_conf/<tag>.json` |

   - `XII_RDIR=logs/none` resolves relative to the package root (lane.sh cd's there). The file does not exist, so score_contrast and l1abl_mix fall back to `logs/decoy_conf/<tag>_E_R.npy`, the same file das_decoy writes.
   - The merge's `k.startswith("u")` filter correctly drops decoy_disc's scalar keys.
5. **No hang on non-mixture parents.** l2_mix_causal, l2_mediation, score_contrast and l1abl_mix all still write their output file, with `mixture_heads=[]`, when no head qualifies. The merge does not wait on anything class-dependent.
6. **Streams.**

   | analysis | stream(s) |
   |---|---|
   | l2_scan | 74.0M |
   | l2_mix_causal | 74.1M |
   | decoy_disc | 74.2M |
   | l1abl_mix | 74.3M |
   | l2_mediation | 74.4M |
   | score_contrast | 74.5M |
   | adapt_sel | 74.6M, with internal 75.1M and 75.3M |
   | exploratory TPX | 74.8M |

   None collide, and none appear in any dev queue. Exception: das_decoy (see D2).
7. **Preflight reproduces exactly.** Re-running the frozen `preflight_nulls.py` in a sandbox gives a `preflight_nulls_result.json` identical to the committed one.
8. **Sandbox confirmation-path run (S1).** Six complete synthetic parents, each holding xP81 dev outputs, give P1–P5, P7 and P8 confirmed with n = 6, and P6 inconclusive (n = 0).
9. **No crashes hidden so far.** No `Traceback` appears in any `logs/run/*.out`.

## Must-fix

None of these needs a change to frozen predictions. Changing frozen code (scorer, scripts, lane, merge) would itself be a **deviation** under the package's no-amendment policy. So each fix below is a pre-declared reporting or verification procedure, to be added as a logged NOTE **before any cohort analysis output is read**. If code is changed anyway, it must be logged as a deviation, and the affected predictions reported as "inconclusive (deviation)".

**M1. The mixture-head definition differs between the scorer and the analyses. This can silently make P1–P4 inconclusive for the whole cohort.**
- Each script defines its own heads:
  - **Scorer:** a parent is mixture-class iff some head has mass ≥ 0.8 **and** share ≥ 0.2 (l2_scan, 3000 episodes, stream 74.0M).
  - **l2_mix_causal:** heads with mass ≥ 0.8 on its **first 500** episodes (74.1M).
  - **l2_mediation:** heads with mass ≥ 0.8 on the **TRUE-design xAP cells**. These are selected, non-natural episodes.
  - **score_contrast and l1abl_mix:** heads with mass ≥ 0.8 on their own 3000 episodes.
- A parent can be mixture-class in the scorer while a script finds no head. That script then omits `donor`, `TRUE`, `E_mean`, `intact` and so on. The scorer sees `None`, and the "any missing" rule makes **the arm inconclusive for all units**, so the whole prediction is inconclusive.
  - Sandbox S2: P163 is mixture-class by l2_scan, but l2_mix_causal mass is 0.79. Result: every P1 arm is inconclusive at 5/6, with no warning.
- Masses near 0.8 do occur in dev: wP143 has head 0 at 0.776 and head 1 at 0.868, and vP152 has head 3 at 0.894.
- The scripts also patch a different head set from the one that defines class. vP155 patches heads {1, 2, 3}, but only head 1 has share ≥ 0.2.
- Fix, declared now:
  - for each cohort parent, report the scorer class next to each script's `mixture_heads`;
  - if they disagree, label the prediction "inconclusive (head-definition mismatch)";
  - also report, as secondary and non-confirmatory, the arm counts on the units that do have values.
- Do not re-run with harmonised heads unless that is logged as a deviation.

**M2. `scripts/lane.sh` always logs `rc=0`. Crashes are invisible, and a crash hangs the merge forever.**
- The line `bash -c "$line" ...; echo "$(date ...) ... rc=$?"` reports the exit status of the `$(date)` substitution, not of the job. Verified: `bash -c "exit 3"; echo "$(date) rc=$?"` prints `rc=0`.
- Every `rc=` in COMPUTE_LOG, past and future, is therefore uninformative.
- If any confirmation job crashes (OOM is plausible: 8 GB, no swap, two decoy_disc jobs plus training on 2 CPUs), its output file never appears. `conf_merge.sh` then waits forever.
- Fix, without editing the frozen lane.sh:
  - add a COMPUTE_LOG NOTE saying rc values are not meaningful;
  - monitor `logs/run/<key>.out` for `Traceback` or `Killed`, using the key from COMPUTE_LOG;
  - log any crash and re-run as a deviation, with the affected prediction or arm reported per the amendment policy;
  - before scoring, grep every C1/C2 run output for errors and record the result.

**M3. `conf_merge.sh` triggers on file existence, does not stop on errors, and runs the scorer regardless.**
- `while [ ! -f ... ]` fires as soon as `open(...,'w')` creates the file. `json.dump` may still be writing. l2_mix_causal writes about 1 MB per parent, and adapt_sel about 200 kB.
- The race is only against lane C2's files (P161, P163, P165). C1's own jobs ran before the merge in the same lane.
- If a partial JSON is read, the merge's python crashes partway through. Because the script has no `set -e`, `score_xiii.py` still runs on whatever merged files exist. The affected predictions come out inconclusive, or the parent is silently excluded (see M4c).
- Fix:
  - wait until lane C2's last `end` line appears in COMPUTE_LOG;
  - check that each `logs/<a>_conf.json` has exactly six `u…` keys;
  - check that `score_xiii.json` has `units` = 6, the expected complete/incomplete split, and no `None` in `mixture_class`;
  - if any check fails, re-run the identical `bash scripts/conf_merge.sh`, logged as a re-execution of frozen code on complete inputs.

**M4. Verdict and diagnostic edge cases that can misstate a result.**
- **(a) Inconclusive overrides fail.** If any arm is inconclusive, the prediction is *inconclusive* even when another arm **failed**. Sandbox S7 shows the mechanism. A falsifying arm is then reported under "inconclusive", and `counts.falsified` misses it.
  - Fix: declare that any arm with status `fail` is reported explicitly as a failing arm, and the prediction as "not confirmed (arm X failed; arm Y inconclusive)", whatever the scorer's verdict string says.
- **(b) A missing null counts as diagnostic.** A null expression that evaluates to `None` counts as "null fails", so the arm is marked **diagnostic = True** (sandbox S8).
  - Fix: for each arm, check that `null` evaluated to non-None on every unit before calling it diagnostic. Report "diagnosticity unknown" otherwise.
- **(c) A missing l2_scan entry drops the parent silently.** `mixture_class()` returns `None` when a parent's l2_scan entry is missing, and the parent is silently left out of every mixture-only prediction. n shrinks with no flag (sandbox S3).
  - Fix: the M3 check that `mixture_class` has no `None`.

## Disclosure issues

- **D1. A post-freeze dev result bears directly on P5, and the frozen document's dev summary is now stale.**
  - The vP150 adapt_sel run (a complete, pre-assembled parent) ended at 18:14:48, after the freeze. It fails **P5c** (readout KL 0.167 < 0.2) and **P5d** (readout S=3 accuracy 0.541 < 0.7).
  - `logs/score_xiii_pre.json` was regenerated at 18:31 and now shows P5c and P5d at 2 of 3 available units. That overwrote the committed version, where only 2 units were available, both passing.
  - FROZEN_PROTOCOL §5 still says "Every available development unit passes every arm".
  - P5 is **not** mixture-only. The dev complete pool was about 45% pre-assembled (6 mixture / 5 pre-assembled). Sandbox S4: two pre-assembled parents out of six give P5 falsified.
  - This must be disclosed as a post-freeze development observation before the cohort results. The P5 rationale cites only mixture-class dev models, and P8 was never run on a pre-assembled dev model.
- **D2. das_decoy reuses development streams.** das_decoy trains and evaluates its subspaces on **64_000_000 / 64_500_000**, the same streams as XI and XIII development.
  - This contradicts Stage-0 ("analysis streams 73_000_000 and above") and §1 ("new analysis streams (74_000_000 and above), none seen before the freeze").
  - The table's "as in development" only partly discloses it.
  - The E, A1 and PERM subspaces feed P3, P4 and P8. The evaluation streams for those predictions are fresh, so this is reused *training* data, not leakage. It still needs to be stated plainly.
- **D3. The preflight's framing overstates it.**
  - NOOP, UNIFORM and RANDOM are trivial: all-zero effects cannot pass any sane threshold.
  - ALTMECH is the only substantive null, and several of its values were set by hand rather than derived:
    - score_contrast: `complement_mean = 1.0`, `E_mean = 0.05`, `decoy_PERM = 0.02`;
    - l1abl: `layer1_mean_ablated.B = 0.9`;
    - the same decode values in every scenario (bayes_decision complement 0.45 vs E 0.40). These alone make P4c fail on every null;
    - the adaptation profile "learns product rule only".
  - `E_mean = 0.05` under ALTMECH contradicts P3c's own premise that matched-budget mean patches are **not** specific; dev shows A1 and PERM remove 0.88–1.43. So **P3 probably cannot reject the A1 mechanism**. Its preflight protection against ALTMECH rests on hand-set values.
  - The docstring's "values follow from the scenario by construction, not tuned" should be qualified accordingly.
  - The preflight also never exercised the confirmation path (gate.json, `_conf`, `adapt_sel_conf`, the merge), missing values, or head mismatch.
  - The listed arm-level null passes are reported accurately, and so is P7's non-diagnostic status.
- **D4. Not every pre-freeze fix is in COMPUTE_LOG.** §2 says the fixes are "all logged". Fixes 1–3 are in COMPUTE_LOG, but fix 4 (decoy stratification) and fix 5 (P2b mixture NIE ≥ 0.2) are not. Predictions JSON, scorer and preflight were edited at 18:03:26–18:03:39, minutes before the freeze.
- **D5. P2's "size-matched non-mixture heads" are not size-matched.** l2_mediation uses **one** decoy head (lowest mass) against 1–3 mixture heads; vP155 is 3 against 1. This is the same text/code divergence class XII was criticised for. In addition, the DiD "95% CI" bootstraps only the TRUE episodes and treats the CTRL mean as fixed, so it is anti-conservative.
- **D6. Post-freeze math corrections weaken P5's wording.** The post-freeze MATHEMATICAL_SUFFICIENCY changes state that within S=3 only the "not all agree at q" part (0.055 of 0.083) depends on J_H. The product rule is exact on the rest. The frozen P5 wording ("the J_H-dependent S=3 slice") and the adapt_sel docstring therefore overstate it. Scoring is unaffected; disclose the change in interpretation.
- **D7. Unfrozen post-freeze scripts.**
  - `scripts/stats_xiii.py` (18:31) and `scripts/transition_predictor.py` (18:05) are post-freeze and unfrozen.
  - The TPX lane analyses a cohort model. It is logged as exploratory, which is good.
  - Results documents must take verdicts only from `score_xiii.json`, and label stats_xiii as descriptive and written after the freeze.
- **D8. The timing anchors should be stated precisely.**
  - Commit times are self-reported. The external anchor is the GitHub branch creation at 18:04:32Z. That is about 18 s after the cohort fine-tune process launched, but before its first output at 18:05:15.
  - Stage-0 (96c05e8) was **not** pushed before pretraining began. Its only external anchor is the same 18:04:32 push.
  - The Stage-0 commit contains a **0-byte** `logs/eval_conf13.npz`; the real file was written at 16:36:16, 5 s after the commit. The build script is hashed and the file is deterministic, and the real file is hashed in 47506c3. Still, Stage-0 did not freeze the evaluation set's bytes.
- **D9. Outcome blindness is not claimed, and that is honest.** The document correctly says thresholds were chosen after seeing dev results.

## Minor

- `counts` include exploratory P6, for example `meets_thresholds`. Report the primary counts without P6.
- NaN values compare False and count as **fail**, not inconclusive (sandbox S5). This errs on the conservative side.
- l2_mix_causal computes head mass on only 500 episodes. This feeds M1.
- STAGE0 says dev continuations ran "900–1500 of 3000 steps"; COMPUTE_LOG says "~1500–1800".
- adapt_sel runs the unused `fresh` condition and the dead J_H pair construction. This costs compute only.
- About 2 GB peak memory per decoy_disc job, on 8 GB with no swap, plus the exploratory TPX lane competing for 2 CPUs. This raises the M2 crash risk.
- Dev coverage was thin:
  - P8 was never tested on a pre-assembled model;
  - P3c was tested on 2 models;
  - P1c is inconclusive on yP130, whose dev output lacks `donor_EneA1`.
- The development thresholds for P3a used das.py E fits for yP130 and vP156. The confirmation run uses the das_decoy E. Dev `decoy_E_mean` ≈ `E_mean` on xP81 and vP155 (1.375 vs 1.365; 1.466 vs 1.468), so the substitution looks benign.
