# Investigation XVI: frozen protocol (the preregistration record)

This file is committed and hashed before any confirmation run starts. The hash list is `logs/FREEZE_XVI_HASHES.txt` (32 files: code, design, references, sibling modules, and the six XV checkpoints the continuations start from). Anything added after the freeze goes in a separate, clearly marked section at the end.

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%, unchanged.

## 1. What was fixed from mathematics and development before this freeze

**Generator** (`src/xvitask.py`). XV's GATE family is extended with two separate reliabilities:
- b1, the reliability of the agreeing pair when g = 1;
- b0, the reliability of the single source when g = 0;
- ν, the noise-pair agreement, as before.

The generator was verified exactly (`logs/xvi_math_verified.json`, all 7 checks pass):
- With b1 = b0 it reproduces XV bit for bit, including the random-number consumption. So XV's seeds and data streams carry over, and continuations see exactly the data XV would have shown next.
- Its random-number consumption does not depend on (b1, b0, ν).

**Value contrast (Workstream B), arity 2, chosen by exact search** (`logs/xvi_value_grid.json`, 240 conditions):

| | low value `2:0.9:0.55:0.05` | high value `2:0.9:0.9:0.05` (= XV's A) |
|---|---|---|
| gate value (nats) | 0.0802 | 0.1398 (ratio 1.74) |
| agreement law P(agree \| g = 1), P(agree \| g = 0) | 0.815, 0.05 | identical |
| agreement-only benefit | 0.0752 | 0.0757 (0.66% apart) |
| agreement MI with g; cue overlap | 0.345; 0.8825 | identical |
| minimal decision order; proper-subset MI | 3; 0 | 3; 0 |
| **residual:** Bayes NLL / accuracy | 0.591 / 0.725 | 0.288 / 0.900 |
| **residual:** order-1 accuracy | 0.668 | 0.872 |
| **residual:** gate's accuracy gain | 0.020 | 0.028 |

Holding the pair reliability and ν fixed keeps the competing agreement cue's law exactly identical. The price is that the single source's reliability changes how easy the whole task is.

Tolerances were set before the search: gate-value ratio ≥ 1.5, agreement benefit within 2%, and an agreement law equal to 1e-12. Only three parameters exist, so the agreement law, the agreement benefit and the Bayes NLL cannot all be held fixed while the gate's value changes. **The overall-difficulty confound remains and is declared.**

**Arity 4** (S = 7 positions, exact, V5):
- minimal decision order 5;
- zero mutual information between g and every proper subset of the gate sources;
- every order ≤ 4 rule has gate-flip slope 0;
- gate value 0.27702, identical to arity 3 in the same format (difference about 1e-15), with intermediate gain identical as well.

**Diagnostics** (`src/xvidiag.py`, rebuilt after the XV reviews):
- **F_G (gate-flip following)** is decision-level. It is exactly 0 for gate-free rules and 1 for Bayes at temperatures 0.5, 1 and 2 (V6), so sharpening alone cannot produce it.
- **A_conf** is the XV agreement index normalised by the model's own sharpness. It is constant across temperatures for the gate-free optimum, and 0 for the order-1 rule and for Bayes.
- **A decision-level agreement measure was tried in development and rejected.** No pair-single swap changes the gate-free decision without also changing the order-1 decision.

**Development** (unblinded; seeds and streams disjoint from confirmation; both runs were stopped at the freeze to free the two CPUs, and their partial trajectories are kept in `checkpoints/dev/`):
- **Constant learning rate on arity 3** (1e-3 after a 200-step warm-up; seed 150; the C condition):
  - numerically stable through 2000 steps;
  - reached the agreement stage (KL 0.285 against the gate-free 0.280; A_p about 0.7–0.8);
  - **no gate response through step 2000** (F_G ≤ 0.06), at exposure 1.9.
- **Learning-rate restart of XV's development model D_s100** (the D condition, which never acquired the gate in XV):
  - stayed at its gate-free optimum with F_G ≤ 0.002 through 2050 post-restart steps;
  - **so in this one model, restarting did not unlock the gate quickly.**
- **No Workstream B development run.** The high-value condition is XV's condition A, whose four XV models acquired the gate at steps 825–1500. The low-value condition was not piloted. If it fails to acquire within 4000 steps, the P3 rule treats that as a censored lower bound.
- **Throughput measured on this machine (2 CPUs):** 1.4–1.6 steps/s per lane with two lanes and inline diagnostics; about 1.0 GB resident per process.

## 2. Cohorts (logs/XVI_design.json; queue logs/queue_train.txt in this priority order)

| cohort | condition | models | schedule | steps | diagnostics | independence |
|---|---|---|---|---|---|---|
| WA_fresh_const | 3:0.9:0.9:1/3 (XV's C) | seeds 401–404 | warm-up 200, then constant 1e-3 | 15,000 | every 50 | 4 independent initialisations and data streams |
| WA_continuation_restart | XV's C or D | the 6 censored XV models: C_s201, s202, s203; D_s201, s202, s204 | from XV step 4000; new AdamW state; warm-up 200, then constant 1e-3; data stream continued | 4000 | every 50 | 6 distinct XV models. **Not fresh replications; selected as XV's failures** |
| WB_value_low / WB_value_high | 2:0.9:0.55:0.05 / 2:0.9:0.9:0.05 | seeds 501–504 in each | XV cosine (warm-up 200, decay to 0 at 4000) | 4000 | every 25 | 4 seed pairs. Within a pair both conditions share the initialisation and every random draw; the unit for the contrast is the pair |
| XV arity-3 reference | XV's C, D | 8 XV models | XV cosine | 4000 | XV snapshots | existing data (already inspected in XV) |

All runs are blind (BLIND = 1): nothing is printed, and outcomes go only to files that are not read before the scorer runs. Parameter snapshots are saved every 500 steps. Diagnostics run inline, so dense measurement does not need dense checkpoints.

**Schedule conditions required by the mandate.**
- **Condition A (original schedule):** the XV runs.
- **Condition B (extended constant rate):** WA_fresh_const.
- **Condition C (continuation with learning-rate restart):** WA_continuation_restart.
- **Schedule-matched continuation without restart: not run, because it is provably inert [proof].** XV's cosine rate is exactly 0 at step 4000. optax's AdamW multiplies both the Adam step and the weight decay by the learning rate. So continuing on that schedule changes no parameter.
- **Deviation:** XV saved no optimizer state, so continuations start with a fresh AdamW state. "Restart" here means a learning-rate restart plus a moment reset. The two cannot be separated without re-deriving XV's optimizer state by deterministic replay, which would cost about 0.7 lane-hours per model. That is not affordable.

## 3. Budget table (measured; mandate section 10)

| item | steps | lane-hours at 1.4–1.6 steps/s |
|---|---|---|
| WA fresh constant rate, 4 × 15,000 | 60,000 | 10.4–11.9 |
| WA continuation, 6 × 4000 | 24,000 | 4.2–4.8 |
| WB value contrast, 8 × 4000 (diagnostics every 25) | 32,000 | 6.1–6.9 |
| **total frozen queue** | 116,000 | **20.7–23.6 lane-hours ≈ 10.4–11.8 h wall on 2 lanes** |
| exact arity-4 enumeration (orders 1–5, 2187 inputs) | — | 44 s |
| memory | — | about 1.0 GB per process; 2 processes against 7.8 GB total |

**Arity-4 neural decision (prespecified; mandate C3 and section 8).**
- **Arity-4 neural confirmation is not authorised in this session.** There are two reasons:
  1. The frozen A + B queue alone needs about 10.4–11.8 h of wall time.
  2. The architecture has exactly 6 source slots per question, and arity 4 with the pair-and-single evidence needs 7. That changes the token count and the source embeddings. A fair scaling comparison would then need arity-2 and arity-3 baselines retrained in the 7-source format: at least 8 further runs of 15,000 steps, about 20 lane-hours.
- **An exploratory pilot is allowed only if** P1 is BROAD **and** the session continues for at least 6 h after the frozen queue completes. It would be at most 2 runs, labelled exploratory, and not used as scaling evidence.
- This rule is fixed now, before any confirmation outcome exists.

## 4. Frozen definitions (scripts/score_XVI.py is the authority)

- **t_A (sustained acquisition):** the first diagnostic step at which it and the next two points all have F_G ≥ 0.5 and KL ≤ 0.75 · KL(order-a optimum). Otherwise the model is **right-censored** at its last step and is never assigned an acquisition time.
- **t_first:** the first point with F_G ≥ 0.2.
- **E_A:** cumulative learning-rate exposure at t_A. For continuations, XV's exact cosine exposure is added.
- **width:** from the last point with F_G ≤ 0.1 before t_A to the first point with F_G ≥ 0.9.
- **regressions:** the number of points after t_A with F_G < 0.5.
- **time near the gate-free optimum:** steps with F_G < 0.2 and KL within 10% of the gate-free KL.
- **competent at the end:** the mean of the last 5 points has KL ≤ 0.03 (constant-rate runs) or ≤ 0.02 (cosine runs) and F_G ≥ 0.9.
- **Floor-conditioned stage (D2):** at least 4 consecutive points before t_A with F_G ≤ 0.1, A_conf ≥ 0.5 · A_conf(gate-free optimum), and KL ≤ KL₁ − 0.5 · (KL₁ − KL_a). That is, at least half the agreement benefit has been realised.
  - It is evaluated only for models that acquire the gate. A censored model can never confirm a staging claim.

## 5. Frozen predictions

**P1, extended arity-3 acquisition** (4 fresh models, constant rate, 15,000 steps). k = the number acquiring.
- **BROAD** (k ≥ 3) | **SELECTIVE** (k = 2) | **RARE** (k ≤ 1).
- **Expectation: BROAD**, meaning Explanation A (slower acquisition) dominates.
- Alternative: RARE, meaning Explanation B (an optimisation barrier persists despite about 7.5× XV's learning-rate exposure).
- Basis:
  - XV's two arity-3 successes and development seed 100 acquired at exposures of about 1.6–1.9, while the rate was still substantial.
  - The six failures sat at the gate-free optimum as the rate decayed to 0.
- The scorer also reports acquisition times on the step and exposure axes, Kaplan–Meier curves and exact intervals.

**P2, schedule versus duration** (6 learning-rate restarts of the censored XV models, 4000 steps). kc = the number acquiring.
- **CONT_BROAD** (kc ≥ 4) | **CONT_SELECTIVE** (kc = 2, 3) | **CONT_RARE** (kc ≤ 1).
- **Expectation: CONT_SELECTIVE, with no FAST_UNLOCK.** Development model D_s100 did not unlock within 2050 post-restart steps, and the six continued models are XV's failures.
- Alternatives:
  - CONT_BROAD: XV's censoring was mainly learning-rate starvation.
  - CONT_RARE: these seeds sit in a basin that persists under restored optimisation.
- Frozen descriptors:
  - **FAST_UNLOCK:** at least 2 acquire and their median post-restart t_A is ≤ 1000. That would point to a restart effect rather than accumulating exposure.
  - **Matched-exposure comparison with the fresh cohort** at E* (COMPARABLE, CONT_LOWER or CONT_HIGHER, with a Fisher exact p).
- Because the continued models were selected as XV's failures, a difference from the fresh cohort cannot be attributed to the schedule alone.

**P3, decision value at fixed arity, with the agreement law identical** (4 seed pairs). For each pair, r = ln(t_A(low) / t_A(high)); a censored time enters as 4000, a lower bound.
- **BENEFIT_SPEEDS** (r ≥ ln 1.25 in ≥ 3 of 4 pairs) | **BENEFIT_SLOWS** | **NO_EFFECT** (|r| < ln 1.25 in ≥ 3 of 4) | **INCONCLUSIVE**.
- **Expectation: BENEFIT_SPEEDS**, the benefit-driven account.
- Alternative: NO_EFFECT, the competition-driven account. With the competitor's law fixed, timing would then not follow the gate's value.
- **Declared limits:**
  - The low-value condition is also harder overall (Bayes accuracy 0.725 against 0.900), so BENEFIT_SPEEDS cannot separate "gate value" from "overall evidence strength".
  - The gradient-signal account (optimisation signal) is **not** separated by this design.

**P5, stage validity with the corrected diagnostics** (among acquiring models only).
- Per cohort (WA fresh, WB low, WB high): PRESENT if ≥ 75% of acquirers show the floor-conditioned stage, ABSENT if ≤ 25%, otherwise MIXED. NOT_EVALUABLE if fewer than 2 acquire.
- **STAGE_ROBUST** if every evaluable cohort is PRESENT and at least 2 are evaluable; **STAGE_NOT_ROBUST** if any is ABSENT; otherwise MIXED.
- **Expectation: STAGE_ROBUST.**

**Not frozen:**
- **P4 (arity scaling):** not diagnostic or affordable in this session (section 3).
- **P6 (learning signal):** a corrected estimator (unbiased cross-batch ‖μ‖², null and positive controls) will be validated exploratorily. With at most 4 fresh arity-3 models, a frozen signal-predicts-timing test would not discriminate. The mandate asks not to manufacture predictions.

**Threshold sensitivity** for t_A and the stage is reported as exploratory, over a grid fixed in the scorer.

## 6. Integrity rules

- No confirmation output is read before every run completes and `score_XVI.py` runs. The lane logs record only start and end and the exit code.
- The scorer fails closed (`logs/preflight_XVI.json`). It recovers planted outcomes in two synthetic scenarios covering all eight verdict branches, and it refuses (exit 2, no output) on: a missing trajectory line, blind = false, a leftover resume state, a missing snapshot, or a changed frozen file.
- A finding that motivates a new prediction after scoring is labelled exploratory.

---

## Post-freeze notes (added after scoring; not part of the freeze)

The text above is unchanged since freeze commit `737393b`. Review findings that concern it (`reviews/ADJUDICATION.md`):

1. **§5, P2 expectation.** The hashed scorer docstring states CONT_BROAD. This protocol states CONT_SELECTIVE, revised before the freeze in light of the development evidence. The inconsistency was not noticed at the freeze.
2. **§6, preflight.** The preflight covered 8 of the 13 verdict branches, not "all eight".
3. **§6, no reading before scoring.** Confirmation trajectory files were read by line counting (`wc -l`) for progress monitoring before scoring. No content was displayed.
4. **§1, development stop.** Both development runs reached step 2050. They were stopped at 16:19:23.
5. **§1, the decision-level agreement measure.** Its rejection holds for the arity-3 conditions and WB-high. In WB-low a decision-level test was possible (Review A finding 8).
6. **§1 and §5 (P3), cue matching.** The cue's law and log-loss benefit are matched, but its decision-level use is not (Review A finding 1). The P3 alternatives did not discriminate between accounts (Review A finding 2).
