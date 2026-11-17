# XV task-variant experimental design: the frozen protocol (FREEZE XV)

This document was written and committed **before any confirmation run was launched**. The confirmation seeds (201–204 for GATE variants, 301–304 for DeepSets T4) have never been trained. Every development observation below comes from development seed 100 (GATE variants) and seed 300 (DeepSets T4).

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25% (unchanged).

## 1. Task family (src/xvtask.py; mathematics in RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md)

**The generator.**
- Six source positions per question; K = 3 values; Q = 6 questions; Y ~ U(K).
- **Gate.** The gate is the relation g = [x₀ + … + x_{a−1} ≡ 0 (mod 3)], with P(g = 1) = ½. The gate has arity a ∈ {2, 3}. For a = 2, position 2 is an irrelevant uniform source.
- **Evidence when g = 1.** The pair (3, 4) reports Y with reliability β = 0.9, and source 5 is noise.
- **Evidence when g = 0.** Source 5 reports Y with reliability β, and the pair (3, 4) is noise that agrees with probability ν.

**What every condition keeps fixed.**
- architecture: the GATE "M" Transformer, 3 layers, 4 heads, d = 64;
- input positions;
- vocabulary;
- objective;
- optimiser and schedule: AdamW, 10⁻³, warm-up 200, cosine to 4000, weight decay 10⁻⁴;
- batch: 256 episodes;
- training budget: 4000 steps;
- evaluation;
- β = 0.9, so **the Bayes NLL (0.288) is identical in all four conditions**.

**Conditions** (exact values from `logs/xv_math_dev.json`, independently verified in `logs/xv_math_verified.json`, where all checks pass):

| condition | gate arity | minimal decision order (verified) | ν | Bayes NLL / acc | order-1 NLL / acc | best rule without the gate (order a) NLL | **gate gain** (usefulness) | intermediate (agreement) gain | accuracy gain from the gate |
|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 3 | 0.05 | 0.288 / 0.900 | 0.504 / 0.872 | 0.428 | **0.140** | 0.076 | 0.028 |
| B | 2 | 3 | 1/3 | 0.288 / 0.900 | 0.590 / 0.791 | 0.565 | **0.277** | 0.025 | 0.109 |
| C | 3 | 4 | 1/3 | 0.288 / 0.900 | 0.590 / 0.791 | 0.565 | **0.277** | 0.025 | 0.109 |
| D | 3 | 4 | 0.05 | 0.288 / 0.900 | 0.504 / 0.872 | 0.427 | **0.139** | 0.077 | 0.028 |

**Matching across arity.** B and C are matched exactly; their gate gains differ by 0.09%. A and D are matched to within 0.9%. The intermediate gain is matched within 1.1% and 1.7% respectively. The pre-registered tolerance was 2% for the gate gain and 10% for the intermediate gain.

**Contrasts.**
- **Complexity at matched usefulness:** B against C, and A against D.
- **Usefulness at fixed complexity:** A against B, and D against C. The gate gain is about 2× larger in B and C.

**Disclosed confound.** Usefulness is varied through ν, the agreement law of the noise pair. Lowering ν lowers the gate's gain *and* raises the agreement (intermediate) gain, because their sum is fixed by β. So the usefulness contrast is also a contrast in intermediate-cue strength. With β fixed, no knob changes one gain without changing the other.

## 2. Measures (src/xvdiag.py; exact references over all 729 inputs)

**G, the gate-flip slope.** The last gate source is redrawn so that g switches; the edit is on-support. G is the model's slope on the Bayes change.
- Every rule of order ≤ a gives G = 0 exactly.
- Bayes gives G = 1.

**A_p, the partial agreement index.** It is the coefficient on the gate-free rule's response to the swap (x₄ ↔ x₅), in a regression of the model's swap response on three references: the gate-free rule's, the Bayes rule's, and the order-1 optimum's responses.

Exact reference values on the confirmation stream:

| rule | A_p |
|---|---|
| order-1 optimum | 0 |
| order-a optimum (the intermediate) | 1.0–1.12 |
| gate-free rule | 1 |
| Bayes | 0 |

**Also recorded:** NLL to the exact posterior, accuracy, and KL from Bayes.

**Evaluation.** Stream 98_700_000, 1500 episodes (9000 questions), at every snapshot.

**Snapshots.** Every 25 steps to step 2000, then every 250 steps to step 4000. This is the same 25-step density as XIV.

## 3. Hypotheses and their predicted patterns

| hypothesis | gate time, complexity contrast (P1) | gate time, usefulness contrast (P2) | intermediate stage in (A, B, C, D) (P3) |
|---|---|---|---|
| H1 complexity-first | later at arity 3 | no effect | (+, +, +, +): the lower-order relation first whenever it is useful |
| H2 decision benefit | no effect (gains matched) | earlier with the larger gain | (+, −, −, +): a stage only where the intermediate payoff is large relative to the gate |
| RACE (formed from development seed 100; H3/H4-type) | later at arity 3 | — | (+, −, +, +): the agreement cue and the gate are learned at their own rates, so a stage appears iff the gate is slower than the cue |

**Development evidence (seed 100; not confirmation).**
- **A:** A_p ≥ 0.5 from step 200; G ≥ 0.5 at about step 650. A stage is present.
- **B:** A_p peaks at 0.41 (step 300); G ≥ 0.5 at about step 450. No stage.
- **C:** A_p ≥ 0.5 from about step 350, rising to 0.92; G ≥ 0.5 at about step 1725, an abrupt change over about 100 steps. A stage is present. The model is competent at 4000.
- **D:** A_p ≥ 0.5 from step 200, rising to 1.0; G = 0.00 throughout 4000 steps. The gate was **not acquired**, and KL stays at 0.142, equal to the gate gain.

These development results already contradict H2's call for C. The RACE pattern was formulated from them. The confirmation runs test all three patterns on fresh seeds.

**Development incident (disclosed).** The first two development runs (A and C) were overwritten at steps ≤ about 475 by the next pair, because of a directory-naming bug. Their trajectories had been evaluated before the overwrite: A to step 1400, C to step 1300. Both directories are kept in `checkpoints/dev_contaminated/`. See COMPUTE_LOG.

## 4. Frozen definitions and verdict rules (scripts/score_XV.py, fail-closed)

**Per-model definitions:**
- **t_G:** the first snapshot with G ≥ 0.5 at it and at the next snapshot. Censored at 4000, which is a lower bound reported as "not acquired within budget".
- **t_I (intermediate stage):** the first snapshot with A_p ≥ 0.5 and G ≤ 0.1, at it and at the next snapshot.
- **t_S (source stage):** the first snapshot with accuracy ≥ (order-1 accuracy − 0.03) and G ≤ 0.1.
- **Transition width:** from the last snapshot with G ≤ 0.1 to the first with G ≥ 0.9.
- **Competent:** final KL ≤ 0.02 and final G ≥ 0.9.

**Statistical unit.** The independently initialised model. The same init seeds are used across conditions; that pairing is a variance reduction, not independence.

**P1, complexity.** For each of the 8 seed-paired comparisons, r = ln(t_G at arity 3 / t_G at arity 2).
- **COMPLEXITY_DELAYS:** r ≥ ln 1.5 in at least 7 of 8. If some arity-3 times are censored, this reads "acquired later, or not within the budget"; the censored count is reported.
- **NO_EFFECT:** |r| < ln 1.25 in at least 6 of 8.
- **INCONCLUSIVE:** otherwise.

**P2, usefulness.** For each of the 8 seed-paired comparisons, r = ln(t_G with the lower gain / t_G with the higher gain).
- **BENEFIT_SPEEDS:** r ≥ ln 1.25 in at least 6 of 8.
- **BENEFIT_SLOWS:** r ≤ −ln 1.25 in at least 6 of 8.
- **NO_EFFECT:** |r| < ln 1.25 in at least 6 of 8.
- **INCONCLUSIVE:** otherwise.

**P3, intermediate-stage pattern.**
- A model shows the stage if t_I exists and precedes t_G, or t_G is censored. This is counted over the models that reached the source stage.
- Each condition is called PRESENT (at least 3 of 4), ABSENT (at most 1 of 4), or MIXED.
- The verdict is the hypothesis whose pattern matches all four calls; otherwise NONE_MATCH.
- A competent-only version is reported as a sensitivity analysis.

**Bootstrap.** The scorer reports median log-ratios with bootstrap intervals over pairs. These are descriptive only, because pairs share seeds.

## 5. Workstream E: T4 from scratch (scripts/t4_ds_train.py, t4_ds_traj.py, score_T4.py)

**History.** The M Transformer failed T4 from scratch (0/3 at 6000 steps; it stayed at the own-question level, 0.578). DeepSets succeeded (3/3, at 0.743).

**Decision.** No new Transformer attempt. The bounded stopping rule is applied in advance: the historical failure, plus a cost of about 50 minutes per run, which would displace the primary design.

**Positive control.** Four fresh DeepSets T4 models (seeds 301–304) with the identical original recipe (6000 steps). Snapshots every 50 steps to 3000, then every 250. Evaluation stream 98_800_000.

**References.**
- **Source stage:** the exact own-question rule P(Y_q | X_q only), with no cross-question information.
- **G4:** the slope on the Bayes change under an other-question agreement edit. The own-question rule gives 0; Bayes gives 1.

**Development seed 300:**
- KL to the own-question rule fell to 0.007 by step 400–600, with G4 = 0.
- G4 then jumped to 0.60 by step 800.
- Final KL 0.001, G4 0.84. G4 is attenuated, so the competence threshold is set at 0.6.

**Trainability gate.** At least 3 of 4 models competent (final KL ≤ 0.02 and G4 ≥ 0.6). Otherwise the verdict is TRAINABILITY_FAILED, with no learning-order conclusion.

**P6.** Among competent models, at least 4 consecutive source-stage snapshots (KL_own ≤ 0.03, G4 ≤ 0.1) occur before t_G4.
- **CONFIRMED** if this holds in every competent model.
- **FALSIFIED** if it fails in 2 or more.
- **INCONCLUSIVE** otherwise.

**Scope.** A different architecture and a different task: this is not a same-architecture T4–GATE comparison.

## 6. Exploratory analyses, declared in advance (labelled exploratory in the results)

- **P5, learning signal (`scripts/xv_gradsig.py`).**
  - **Measure:** the gate-attributable expected gradient, comparing true data with gate-randomised data in which every marginal is preserved.
  - **When:** at initialisation and at t_S, for every confirmation model.
  - **Analysis:** Spearman correlation of the signal-to-noise ratio at t_S with t_G across the 16 models, compared with the condition-level predictors (arity, gate gain). This is descriptive only, not a verdict.
- **Mechanism (Workstream D).**
  - **Models:** seed 201 of each condition that acquires the gate.
  - **Checkpoints:** t_G − 200, t_G − 50, t_G, t_G + 100, and the final step.
  - **Measures:**
    - a linear probe for g at the evidence tokens and at the readout, at the inputs of layers 1 and 2, on held-out x₀ (chance 0.5);
    - full residual-patch transfer at the evidence tokens.
  - **Question:** is g represented before the gate is used?

## 7. Blinding and integrity

- **Blind training.** Confirmation training runs with `BLIND=1`, which prints and saves no loss. Lane logs therefore contain only return codes.
- **Separate outputs.** Confirmation outputs go to `confirm_outputs/`, which git ignores until the final commit.
- **No inspection before scoring.** No trajectory file is read before every run has finished and the scorer runs.
- **Hashes.** Every file the scorer depends on is listed in `logs/FREEZE_XV_HASHES.txt`. That covers the generator, trainer, diagnostics, reference caches, design, queues, scorers, and the sibling-package modules.
- **Preflight.** All checks pass (`logs/preflight_XV.json`):
  - refusal on the empty state;
  - correct verdicts for the race, H2 and no-effect synthetic scenarios;
  - refusal when a trajectory is missing or a seed is wrong;
  - the T4 scorer's refusal, confirmation and falsification branches.

## 8. Compute budget

**Benchmark** (this machine: 2 CPUs, 7 GB):
- one GATE-architecture run of 4000 steps takes 1966–1970 s with two concurrent lanes (2.03 steps/s per lane);
- peak memory 0.71–0.74 GB;
- one DeepSets T4 run takes 160 s.

**Plan.** 16 GATE-variant runs at about 33 minutes each, two at a time: about 4.4 hours. Four DeepSets runs: about 12 minutes. Measurement: about 30 minutes.

**Not run.**
- a 5th and 6th seed per condition (budget);
- the noisy-gate extension (Workstream F), which gets mathematics only, if time permits.

---

## Post-review notes (added after scoring; not part of the freeze)

The text above is the freeze record as committed in `7957fee`, left unchanged. Corrections from the reviews (see `reviews/ADJUDICATION.md`):

1. **Confound rationale (Review A F4).**
   - Delete "because their sum is fixed by β". The sum Δ_gate + Δ_int varies with ν (0.302 against 0.216).
   - The two gains move together only because ν is the generator's single remaining knob.
   - A generator with separate pair and single-source reliabilities can decouple them.
   - "About 2× larger" holds for the Δ_gate ratio (1.98). The full value from the order-1 state gives 1.40, and the accuracy gain 3.9.
2. **G = 0 for order ≤ a (Review A F6).** It is exact in the population, and |G| ≲ 0.01 on the 9000-question stream.
3. **"Matched exactly" (Review A F8).** B and C differ by 0.09%. The match is exact only under the gate-free-mixture definition.
4. **Development incident (Review C F1).**
   - The overwrite reached step 850, over about 10 minutes, and the killed runs left a `resume.pkl`.
   - Steps 25–475 of `logs/dev_traj_contaminated_A.json` match the overwriting run's weights, so that trajectory is partly contaminated.
   - "Development A had A_p ≥ 0.5 from step 200" is not supported by any retained uncontaminated file, and is withdrawn.
   - Confirmation is unaffected.
5. **Post-freeze material (Review C F8).** Some scripts (exploratory, sensitivity, wrappers) and RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md were written after the freeze but before scoring. None of them read confirmation outputs before scoring.
6. **Declared P5 statistic (Review C F3).** The declared statistic was the signal at t_S. The whole signal analysis was later withdrawn (Review B F1).
7. **Auxiliary assumptions (Review A F11).**
   - H1's "no usefulness effect" assumed that complexity alone sets timing.
   - H2's stage pattern assumed that the intermediate payoff, not its ratio to the gate's payoff, decides whether the stage appears.
   - Neither assumption was stated at the freeze.
