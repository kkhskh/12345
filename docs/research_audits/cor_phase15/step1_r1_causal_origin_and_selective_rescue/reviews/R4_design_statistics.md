# Reviewer 4: experimental design and statistics

Scope: commits 76a9d60 (pre-review), a688902 (Stage-0 freeze) and be4a0da (confirmation freeze). No package file was edited. I reran `confirm7_analysis.py` on a copy in /tmp; its output is identical to `logs/confirmatory7.json`.

## Verified (no finding)

- **Hashes.** All entries in `STAGE0_HASHES.txt` and `FREEZE7_HASHES.txt` match the current files. No frozen file was committed after its freeze. The changes to `downstream.py` and `accessibility_probes.py` were made *inside* be4a0da. They are only a `PHASE=conf` path/eval/snapshot switch and a `PROBE_OUT` switch. Seed, stream (`1_000_003(seed+1)+7919·4+93`) and loss are unchanged.
- **Timing.** P90 pretraining started 20:02:06, the same second as a688902. The first confirmation run started 00:02:42, after be4a0da at 00:00:56. All 40 runs finished rc=0 by 04:27:57, and analysis began at 04:31.
- **Independence.** Pretraining uses init seed = seed and stream 31_000_000+1000·seed, transfer init uses seed+7777, and no checkpoint is loaded. None of this collides with dev seeds 70–87. Pretraining before the full freeze was hypothesis-blind and outcome-independent, so it is legitimate.
- **Statistics code.** The sign-flip test is exact: 256 assignments, identity included. ITT imputation is worst-case in the right direction for P1–P7. The intersection–union logic for P1∧P2 is valid.
- **Numbers.** All spot checks match the logs: cohort means, the per-parent table, P1–P7, the selective counts, W_bal, the r values and pretraining.

## Findings

**1. MAJOR: the selective score rewards signed movement, not closeness to Bayes.**
- *Location:* protocol P3 (Δ ≥ 0.20); SELECTIVE_RESCUE §4.1 ("moves R1 to the Bayes level"); MAIN §3.
- *Problem:*
  - **Overshoot counts as success.** yP90–92 overshoot Bayes: E_R1 is +0.32/+0.43/+0.35, and Δm_R1 is +0.55 to +0.65 against a Bayes value of +0.23. The cohort mean of −0.029 is three overshoots cancelling five undershoots.
  - **Pair-level error falls much less.** Mean absE_R1 falls only from 0.82 to 0.51, and the reduction is ≥ 0.20 in only 3/8 parents.
  - **A Bayes-distance criterion would have failed P3.** \|mean E_R1\| falls by ≥ 0.20 in only 5/8 parents (yP91: 0.445 → 0.425), below the ≥ 6/8 bar.
  - **Hidden R3 bias.** In the same three parents, signed R3 bias rises (0.07 → 0.36, 0.24 → 0.46, 0.07 → 0.34). The pooled R2∪R3 \|E\| constraint does not see this.
- The frozen verdict stands; the interpretation does not.
- *Correction:*
  - Report \|E_R1\|, absE_R1 and overshoot per parent.
  - Replace "to the Bayes level" with "shifted toward Bayes; the near-zero mean reflects cancellation; pair-level error falls by about 38%".
  - The directional claim is robust: \|E\| reductions are 8/8 positive, p = 1/256.

**2. MODERATE: P6 is a low-risk prediction and is overread.**
- *Location:* protocol P6; INDEPENDENT_CONFIRMATION §2; MAIN §2.
- *Problem:*
  - **Lenient threshold.** 0.96 sits just above the maximum of the three dev P values (0.912/0.928/0.952), with a majority rate of 0.912.
  - **Insensitive metric.** Raw accuracy on a 91%-majority target barely moves.
  - **H2 is decodable elsewhere.** The frozen output itself shows this: an MLP probe at the q-sources reaches 0.977 (yP96), and a linear probe on readout+qsources reaches 0.996 (yP96) and 0.94–0.95 (yP95/97).
  - **No contrast.** There is no T1 confirmation panel.
- *Correction:* Call P6 a replication of the dev P range. Report balanced accuracy or AUROC across all sites and probes. Restrict H7 prediction 4 to "not linearly decodable at q-sources alone".

**3. MODERATE: there is no power analysis (mandate §10.4).**
- *Problem:*
  - **P1/P2.** With n = 8, the sign-flip test needs essentially ≥ 7/8 concordant deltas.
  - **P3 passes or fails almost by chance.** It passes with probability 0.80 if per-parent success is 0.8, and still with 0.14 if it is 0.5. It has no error-rate guarantee.
  - **Dev data were available.** Effect sizes P80 +1.12 and P81 +0.48 could have been used for a power estimate.
- *Correction:* Add a power and operating-characteristics note. Label P3 as a descriptive criterion, not a test.

**4. MODERATE: the generic-baseline (outcome F) conclusion goes beyond the evidence.**
- *Location:* SELECTIVE_RESCUE §4.3; MAIN §4 row F.
- *Problem:*
  - **One untuned, extreme setting.** W_bal weights the R1 class about 12×, and it was never developed: the dev W_bal runs started at 05:04, after the freeze. injm, in contrast, was selected on three dev parents.
  - **The cited budget control does not exist yet.** The 3× budget control (`xP80 C1 12000`) started at 06:09:51, one minute before the pre-review commit; its log stops at step 1000. MAIN row F nevertheless cites "READOUT_AND_LAYER2_FACTORIAL.md §5", which is not yet written.
- W_bal is correctly labelled descriptive; the inference drawn from it is not.
- *Correction:* Say "one untuned balancing scheme was not selective". Mark outcome F as untested for milder weighting and for extra budget. Remove the dangling reference until the run is reported.

**5. MINOR: the smoke-run disclosure is inaccurate.**
- *Problem:* With STEPS=20 and PHASE=conf, `downstream.py` trains for 20 steps and evaluates on `eval_conf.npz` at step 20. So it was not "untrained", it touched the confirmation set, and it is absent from COMPUTE_LOG. The risk is negligible.
- *Correction:* Reword the disclosure and log the run.

**6. MINOR: the percentile bootstrap is anti-conservative at n = 8.**
- *Problem:* The bootstrap CI is [0.476, 0.880]. The t interval is [0.43, 0.96], and inverting the sign-flip test gives a lower bound of about 0.42. The conclusion is unaffected.
- *Correction:* Report the t or permutation-inversion CI.

**7. MINOR: the preservation constraints have gaps.**
- *Problem:*
  - **R3 dominates the pooled metric.** absE_R23 is 755 of its 856 pairs R3, so R2 and R3 should be constrained separately.
  - **Listed families are unconstrained.** F23, FLOC, FID and F54o are rescue targets in SELECTIVE_RESCUE but are not in P3. I checked them: F23, FLOC and FID improve in 8/8 parents. F54o overshoots in yP90–92.
- *Correction:* Report these families per parent.

**8. MINOR: pretraining heterogeneity has no inclusion rule.**
- *Problem:*
  - **Bimodal auxiliary accuracy.** It is 0.762 for yP90–92 and 1.000 for yP93–97.
  - **Promised rule not delivered.** Stage-0 promised a pretraining failure rule, but the protocol defines only a transfer rule, and the source of its 0.6595 threshold is not stated.
  - ITT means there is no selection bias, but the under-trained lineages are exactly the overshooting ones.
- *Correction:* Disclose this and report results by pretraining class as exploratory.

**9. MINOR: the secondary predictions are high-prior replications.**
- *Problem:*
  - **P5 and P7.** Both are sign-only, and their dev counterparts were 8/8 with large effects.
  - **P2.** It adds little to P1, because injrand is a hash of X with no task information, so Δ ≈ 0 by construction. injn is the informative specificity control.
  - **Headline.** "Seven of seven confirmed" overstates the independent evidence.
- *Correction:* Say this in INDEPENDENT_CONFIRMATION §2.

**10. No mislabelling found.** No development result is labelled confirmatory.

## Verdict

Freeze integrity, lineage independence, timing, ITT handling and the exact tests are sound and reproducible. P1∧P2 validly confirm a **directional** shift of R1 at n = 8.

Two kinds of claim are overstated:
- **"Corrected to Bayes" (Finding 1)** rests on sign cancellation. A Bayes-distance success criterion would have scored 5/8 or 3/8, below the frozen ≥ 6/8 bar.
- **The inaccessibility and generic-baseline conclusions (Findings 2 and 4)** go beyond what P6 and a single untuned W_bal can support.

**Accept with required revisions to Findings 1–4.** Findings 5–9 are recommended.
