# R2 — Statistical design review (Investigation VI)

Reviewer: independent adversarial reviewer (statistical design). Reviewed state: pre-review commit 9389496. No package file was edited. Checks were run on a copy in /tmp.

**Verified without finding a problem**
- **Freeze integrity.** Every hash in `logs/FREEZE_HASHES.txt` (13 files, addendum included) and in `logs/FREEZE_HASHES_P6.txt` (4 files) matches `sha256sum` of the current files.
  - The frozen files have no commits after their freeze commit: 4d42d0a / 8d8e41a for the behavioural protocol, 357a98b for P6.
  - Training began at 04:12:09, the same second as 8d8e41a.
  - The `mech_conf` outputs (12:09) come after the P6 freeze (12:04).
- **Reproducibility.** Rerunning `confirm_analysis.py` on the current files reproduces `logs/confirmatory.json` exactly.
- **Independence.** Seeds are disjoint across arms, and the transfer seeds (seed + 7777) and data RNGs are disjoint too. `independence.json` reports 0 shared arrays. No development checkpoint is a confirmation unit.
- **Permutation test.** It enumerates all C(16,8) = 12,870 assignments exactly. It is one-sided, and ties count as extreme through the `>= obs − 1e-12` comparison. The smallest attainable p is 7.8e-5.
- **Holm and frozen definitions.** Holm is implemented correctly for the two co-primaries. Regimes, orientation (c_g) and pair exclusion are generator-defined, not outcome-defined.
- **Document numbers.** The numbers in OBSERVABLE / INDEPENDENT / TRAINING_STAGE / MAIN that I checked agree with the logs. That covers means, pair counts (1346 / 435 / 89 / 822), KL trajectories, P4 matching and P6.

## Findings

1. **Major — P1(a) "confirmed" depends on an ambiguous frozen ITT rule.** Location: `scripts/cohort_status.py` docstring, `confirm_analysis.py` `contrast()`, MAIN §2, INDEPENDENT §2.
   - *Problem.* The frozen cohort_status docstring says every failure "enters ITT by worst-case imputation". The frozen protocol §4 and the code impute only a parent with *no step-4000 model*. xP87 failed the success rule but has a model, so its observed B45 = +0.407 is used.
   - Under the docstring reading, P87 → max = 0.850. P1(a) then gives diff 0.307, exact p = .066, bootstrap CI [−0.055, 0.643], i.e. **not confirmed**.
   - P1(b) survives that reading: p = 7.0e-4, CI [0.36, 0.79].
   - *Correction.* Report this inconsistency in the frozen materials as a deviation note. Report the alternative-reading result next to P1(a). Describe P1(a) as "confirmed under the protocol §4/code rule; not robust to the stricter reading".

2. **Major — P1(a) is driven by the low-quality T1 parents, which the protocol itself treats as a quality effect.** Location: MAIN §2 table, INDEPENDENT §2 "Power".
   - *Problem.* Without T72 and T74 (KL 0.14 / 0.11; B45 0.85 / 0.65), the pooled difference is 0.19, p = .076, CI [−0.01, 0.36].
   - Protocol §2 says pooled B45 in R2/R3 "tracks KL in both curricula". So the pooled confirmation mainly reflects under-discounting by two weak T1 models. It does not support the mandate's directional hypothesis.
   - *Correction.* Add this caveat wherever P1(a) is called confirmed. State that the hypothesis is supported only through the R1 statistic.

3. **Major — Stage claims leave out frozen P3 contrasts and overstate specificity across stages.** Location: MAIN "present at every transfer stage", MAIN §2 P3 row.
   - *Problem.* The frozen P3 reports B45 and D-R1 at every stage, and these are omitted.
     - Pooled B45 has the *opposite* sign at 250 and 500 (−0.61, p = .998; −0.38, p = .95).
     - D-R1 is −0.04 at step 250 and +0.16 (p = .03) at step 500.
   - The R1 difference is present at every stage, but its *boundary specificity* appears only from about step 1000. At step 250, P models show deficits at both 3→4 (−0.46) and 4→5 (−0.34).
   - *Correction.* Report all frozen P3 contrasts. Qualify "specific … present at every stage" to "the R1 difference at every stage; specificity relative to 3→4 from step 500–1000".

4. **Major — The training-dynamics inference is stronger than the evidence allows.** Location: MAIN §2 "This pattern contradicts an approximation-error account"; TRAINING_STAGE §2.1.
   - *Problem.* This is an exploratory, non-frozen descriptive argument. At step 250 the P arm barely responds at all: the slope of Δm_M on Δm_B is 0.01 and accuracy is .51. Its B45-R1 there is therefore bounded near −Δm_B, and "deepening" partly reflects models starting to respond.
   - Approximation error does not have to shrink monotonically per sub-population, and Proposition 1 (corrected) says so itself.
   - *Correction.* Use "is not what a uniform-shrinkage approximation account predicts" and label the argument exploratory.

5. **Major (reporting) — The P5 rule is nearly vacuous and its expectation was partly missed.** Location: protocol §3 P5, INDEPENDENT §2 ("holds"), MAIN §5 ("survives … controls").
   - *Problem.* The P5 rule is "P1(b) confirmed and prob-space R1 sign positive". That is almost implied by P1(b), and it does not test the frozen regime expectation.
   - That expectation ("not positive in R2 and R3") was not met in R3: +0.23, p = .14.
   - "Not positive" cannot be established from a non-significant test. R2 has only 89 pairs, CI [−0.62, 0.45].
   - The effect's scope is R1 only: 435 of 1346 oriented pairs (32%).
   - *Correction.* Report P5 as "formal rule met; frozen regime expectation partly mismatched (R3) and not testable as stated". State the R1-only scope in MAIN's answer paragraph.

6. **Minor — Selecting the R1 co-primary on development data is legitimate here, but its limits should be stated.**
   - *What is fine.* The selection happened on Investigation IV/V models, was frozen before any cohort model existed, was tested on fresh parents and new streams, and was disclosed. That is a standard exploratory → confirmatory step, and Holm covers the two co-primaries.
   - *Problem.* Development and confirmation share the architecture, generator, trainers and evaluation design, so the replication is across seeds, not across pipelines. Investigation V's development models also had seed coupling, which may have shrunk the development SDs used for power.
   - *Correction.* State "replication across independent seeds of one pipeline" in MAIN.

7. **Minor — Multiplicity across P1–P6, regimes and stages.** More than 20 directional tests are reported, most of them unadjusted by design.
   - Every R1-family result (P1(b), P2, P3, P4, P6b, P6c; p ≤ 1.6e-4) survives Bonferroni over 25 tests.
   - P1(a) (p = .018), P3-B45@2000 (p = .0499) and D-R1@500 (p = .03) do not.
   - *Correction.* Add one sentence saying which results survive family-wide correction.

8. **Minor — ANCOVA (P4(ii)) validity.**
   - Permuting parent labels with snapshots attached is an exact test of the *sharp* null: the arm has no effect on any snapshot's (KL, B45-R1).
   - It is not a test of the "arm effect at fixed KL". KL is a post-treatment variable, and the linear model is misspecified: the arm×KL interaction is −1.0, and the arm coefficient runs from about 0.75 at KL→0 to about 0.55 at KL = 0.2. The P@250 rows (KL 0.35–0.53) lie outside T1 support.
   - My robustness checks support the conclusion: exact enumeration gives p = 7.8e-5, and restricting to the common KL support (51 rows) gives a coefficient of 0.62 with p = 7.8e-5.
   - The code uses 20,000 random draws, giving 1.5e-4, where exact enumeration was available.
   - *Correction.* Report the support-restricted / interaction fit. Phrase the result as "consistent with" rather than "not explained by KL".

9. **Minor — P4(i) design.**
   - "Ranges overlap" is a weak testability criterion, since a single shared point qualifies. Here the matched sets actually agree well: mean KL is 0.079 (T1) vs 0.088 (P).
   - Matching uses step-250 snapshots, which P3 calls "descriptive only".
   - The target includes the failed P87.
   - *Correction.* Report the means and medians of matched KL. Note these points.

10. **Minor — Bootstrap and α.**
    - With n = 8 per arm, the percentile bootstrap is somewhat narrow. A Welch-t check for P1(a) (t = 2.35) agrees.
    - The power analysis uses one-sided α = .025, while the decision rule is p < .05 combined with a 95% CI lower bound > 0 (effectively α ≈ .025).
    - *Correction.* State the effective α.

11. **Minor — Inference conditions on one episode set.** All models share the same 6000 episodes, and the model bootstrap ignores episode-sampling variance. This matters little given the complete separation in R1.
    - *Correction.* Add a two-way (model × episode) bootstrap or state the conditioning.

12. **Minor — Audit trail.**
    - The `logs/conf/*.npz`, `generator.npz` and checkpoint `.npz` files are untracked in git. The probability-space statistic used by P5 depends on them, and `generator.npz` was rewritten at 11:58.
    - `bayes.py` and `ckpt.py`, which `cohort6_train.py` imports, are not hashed. They are unchanged in git since 2026-11-08.
    - `FREEZE_HASHES.txt` keeps two error lines, corrected by the addendum.
    - Interim boundary evaluations of finished parents (10:09) ran while T77, P86 and P87 were still training. No adaptive decision was possible, but it is not disclosed.
    - *Correction.* Commit or hash the npz files, and disclose the interim looks.

13. **Minor — Wording.** INDEPENDENT/MAIN say "Results are identical in direction and size" for ITT vs successful-only. Pooled B45 is 0.36 vs 0.44, so say "same direction; similar size".

## Overall verdict

The freeze is intact, the cohort is independent, and the frozen analysis is correct and reproducible.

**The central claim stands.** The R1 boundary result is a curriculum difference in B45-R1, with complete separation of the 8 + 8 parents. It is specific to the boundary relative to 3→4 at step 4000, and it is robust to:
- the ITT reading;
- successful-only analysis;
- KL matching;
- the ANCOVA support restriction;
- any reasonable multiplicity correction.

**P1(a) should be downgraded** to "confirmed under the literal frozen rule, but driven by low-quality T1 parents and not robust to the alternative ITT reading".

**MAIN needs qualification** in four places: the stage-specificity statement, the approximation-error inference, the P5 "holds" reading, and the R1-only scope. With those corrections the package supports **Outcome A (R1-scoped)**. No critical issues were found.
