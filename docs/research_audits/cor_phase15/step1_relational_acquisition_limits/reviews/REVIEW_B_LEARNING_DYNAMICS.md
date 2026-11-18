# Review B: learning dynamics and mechanism validity (COR Investigation XVI)

**Reviewer:** an independent agent (Reviewer B, mandate section 11). This is an agent review, not human peer review.

**Scope:**
- acquisition definitions and censoring;
- learning-rate controls and the exposure axis;
- fresh training against continuation;
- cue-conditioned probes, floor-aware stages and calibration confounds;
- snapshot and diagnostic density;
- the rebuilt learning-signal estimator;
- whether late acquisition can be told apart from non-acquisition.

**Method.** I read:
- the scripts: `src/xvidiag.py`, `src/xvitask.py`, `scripts/xvi_train.py`, `score_XVI.py`, `xvi_gradsig2.py`, `xvi_probe.py`, `xvi_replay.py`, `xvi_xv_original.py`, `xvi_stats.py`, `xvi_traj_offline.py`;
- all 18 `confirm_outputs/*/traj.jsonl`;
- `logs/score_XVI.json` and `logs/xvi_stats.json`;
- everything in `logs/expl/`;
- the five reports named in the brief.

All recomputation is in `/tmp/revB16/` (inventory in `README.txt`). It used forward passes, gradients on 12 batches, linear probes and simulations only. No model was trained and no existing package file was modified. Review A's findings 3, 4, 5 and 9 overlap with some points below; I cite them rather than repeat them.

**Reproduced exactly from the raw trajectories:**
- every t_A, t_first, width, regression count, stage run length and near-gate-free duration in `score_XVI.json`;
- all four verdicts;
- the P3 ratios;
- the Kaplan–Meier exposure values;
- the null, planted and label control ranges of the gradient estimator.

---

## Findings

### 1. [MAJOR] Acquisition happens separately in each of the six question slots. Several reported "transitions" are artefacts of pooling over slots and of an F_G threshold that sits exactly at 3 of 6 slots.

**Evidence** (`halfgate.py`, `slots.py`, `fgtraj.py`, inline sweep):

- The architecture reads out 6 questions per episode. F_G pools gate flips over all 6 slots. Decomposing it by slot shows that slots acquire the gate separately:

| checkpoint | pooled F_G | F_G per slot 0–5 |
|---|---|---|
| WA_cont_C_s202 @500 | 0.502 | 1.00, 0.00, 0.00, 1.00, 1.00, 0.00 |
| WA_cont_C_s202 @1000 | 0.499 | 1.00, 0.00, 0.00, 1.00, 0.98, 0.00 |
| WA_cont_C_s202 @2000 | 0.543 | 1.00, 0.01, 0.00, 1.00, 1.00, 0.24 |
| WB_high_s503 @2000 | 0.330 | 0.00, 0.97, 0.00, 0.99, 0.00, 0.00 |
| WB_high_s501 @600 (replay) | 0.111 | 0.00, 0.00, 0.00, 0.00, 0.00, **0.66** |
| WB_high_s501 @650 | 0.163 | 0.00, 0.00, 0.00, 0.00, 0.01, **0.96** |
| WB_high_s501 @700 | 0.484 | 0.00, 0.02, 0.50, 0.77, 0.66, 0.98 |
| WA_const_s402 @1100 (replay) | 0.595 | 0.79, 0.63, **0.00**, 0.74, 0.73, 0.69 |

- The "intermediate F_G plateaus" in the trajectories are integer numbers of acquired slots:
  - C_s202 sits at **0.50187 for 1650 steps (350–2000)**, which is 3 of 6 slots;
  - C_s203 sits at 0.83 (5 of 6) from 600 to 800;
  - WB_high_s503 sits at 0.33 (2 of 6) from 1950 to 2050.
  - Decomposed by gate configuration, flip direction or x₀, the C_s202 state is flat at about 0.50. It is purely a slot effect.
- **C_s202's frozen t_A = 400 rests on F_G = 0.50187 against a threshold of 0.5.** That is about 6 flips out of 3471 decisive flips. Its single "regression" (F_G = 0.4987 at step 1000) is a fluctuation of about 11 flips. Over that period slot 4 moved from 1.00 to 0.98, and no slot changed state.
- **The frozen FAST_UNLOCK descriptor is a knife-edge.** It is TRUE at the frozen F_G ≥ 0.5 and FALSE at every threshold from 0.51 up (sweep below). The P2 *count* (4 of 6) is unchanged at every threshold.

| F_G threshold for t_A | post-restart t_A for C_s201, C_s202, C_s203, D_s204 | median | FAST_UNLOCK |
|---|---|---|---|
| 0.50 (frozen) | 1200, **400**, 550, 1950 | 875 | true |
| 0.51 | 1200, **2000**, 550, 1950 | 1575 | false |
| 0.70 | 1250, 2050, 550, 1950 | 1600 | false |
| 0.90 | 1250, 2050, 850, 2000 | 1625 | false |

**Claims affected:**
- "C_s202 … that transition was not abrupt … the only regression among the 16 acquisitions" (EXTENDED §5; MAIN Q20). It was two abrupt slot-group transitions, at about 250–350 and about 2000–2150, with a stable half-gate between them. There was no regression.
- "FAST_UNLOCK (400–1950 steps after the restart)" and "expectation falsified … fast unlock occurred" (EXTENDED §2; FROZEN §3; MAIN). The descriptor fired only because one model sat at exactly 3 of 6 slots.
- Every reported width is a model-level width. It includes staggering between slots: s501's slot 5 led the others by about 100 steps. Per-slot transitions are at least as sharp as reported, and usually sharper.

**Correction.**
- Report per-slot F_G (cheap: inline or offline from snapshots).
- State that FAST_UNLOCK holds only at the frozen boundary and flips at F_G ≥ 0.51. Report it as non-robust and non-diagnostic (consistent with Review A finding 9).
- Re-describe C_s202 as a half-gate plateau, not a slow or regressing transition.
- Leave the frozen verdicts as they are. The interpretive sentences must change.
- Mechanistically this is the most informative observation in the package, and it is not reported. A network that already computes the gate in three slots fails to extend it to the other three for 1650 steps at the full learning rate. Whatever sets the plateau is therefore largely **slot-local** (it involves position-specific parameters), not the absence of a gate circuit in the network.

### 2. [MAJOR] The learning-signal estimator is unbiased, but "no expected first-order push toward the gate on the plateau" is not justified. The estimator has a non-trivial detection limit, a conservative null, a tautological planted control and too few checkpoints.

**(a) Detection limit (computed).** The z = 2 limit on ‖μ‖ under the reported SE is √(2·se). The calibrated limit, using the empirical null, is √(1.1·se):

| checkpoint (plateau unless noted) | z | ‖μ‖ limit, reported SE | ‖μ‖ limit, calibrated | ‖E∇L‖ | limit / ‖E∇L‖ |
|---|---|---|---|---|---|
| XV C_s201 @1000 | −0.17 | 0.0034 | 0.0025 | 0.262 | 1.3% |
| XV D_s201 @1000 | −0.23 | 0.0038 | 0.0028 | 0.122 | 3.1% |
| XV D_s203 @1000 | 0.34 | 0.0041 | 0.0031 | 0.122 | 3.4% |
| XVI s403 @2500 | 0.43 | 0.0016 | 0.0012 | 0.134 | 1.2% |
| XVI s404 @3000 (F_G = 0) | **1.18** | 0.0012 | 0.0009 | 0.093 | 1.3% |
| XV C_s204 @2000 (F_G = 0) | **1.89** | 0.0035 | 0.0026 | 0.128 | 2.7% |
| XVI cont D_s201 @4000 | −0.01 | 0.0002 | 0.0002 | 0.025 | 0.9% |
| XVI cont D_s202 @4000 | 0.75 | 0.0055 | 0.0041 | 0.052 | **10.5%** |

- On the plateau, "zero" means ‖μ‖ ≲ 0.001–0.004: about 1–3% of the full expected gradient (10% for D_s202). It does not mean zero.
- For comparison: per-pair ‖D‖ is 0.0025 (D_s201 @4000) and 0.012 (s404 @3000); training-gradient noise is 0.09–0.14; the parameter count is 150,848 (`sigma.py`).
- The training process averages over the 1000–3000 batches of a plateau, while the estimator uses 48 pairs. A mean gradient below this limit can still dominate the momentum random walk along its own direction over the plateau. The measurement cannot exclude a slow sub-threshold drift ("hidden progress").

**(b) The null is conservative, and the stated reading of the pre-onset points is wrong.**
- Across the 31 null estimates, z has mean −0.10 and **SD 0.55**. For a degenerate U-statistic the jackknife SE is conservative: in my Gaussian simulation with B = 48 and d_eff = 5–500, the null SD of z is 0.72–0.84 and P(z > 2) ≈ 0.1–0.25%.
- Two of the three pre-onset arity-3 points the report calls "not detectably" non-zero (C_s204 @2000, z = 1.89; s404 @3000, z = 1.18) **exceed all 31 null values**, so their empirical p is below 1/32.
- So the statement "detectable only 150–250 steps before onset in 1 of 4 arity-3 models" understates the evidence. Under a calibrated null, 3 of 4 show a signal 250–500 steps before onset.

**(c) The planted control is an algebraic identity.**
- Exactly, U(D + v) = U(D) + 2 v·D̄ + ‖v‖².
- With ‖v‖² = 10·se and v random (so v·D̄ ≈ 0), the recovery ratio is 1 ± 0.1 by construction. The observed 0.89–1.11 tests only that U is computed correctly. It does not test the gradient pipeline or the regating.
- The natural positive control (labels permuted) has z = 19–139, far from the detection limit.
- There is no control with a known *small, gate-specific* signal. One would be θ_ε = θ_plateau + ε(θ_acquired − θ_plateau), scanning ε to find where z crosses 2.

**(d) Sampling is too sparse to locate onset.**
- Each model has 1–2 pre-onset checkpoints. For D_s203 the signal is somewhere between 1000 (z = 0.34) and 1750 (z = 5.2). For C_s204 it is between 1000 and 2000.
- So "the signal appears only about 150–250 steps before onset" is a statement about checkpoint spacing.
- The XVI arity-3 runs were measured only at 500-step snapshots, although bitwise-exact replay was available.

**(e) The interpretation does not follow.**
- Between 1750 and 2000, ‖μ‖ in D_s203 grows from 0.028 to 1.80, a factor of 64 in 250 steps (e-folding about 60 steps). Extrapolated back exponentially, it would be about 10⁻⁷ at step 1000, far below the 0.004 limit.
- At a strict saddle, deterministic first-order dynamics give exactly this: a gradient along the escape direction proportional to the displacement, growing exponentially from below any fixed detection limit.
- So a gradient indistinguishable from 0 on the plateau does **not** imply that "escape depends on stochastic and higher-order dynamics" (LEARNING_SIGNAL §6.2; MAIN, "the plateau has no expected first-order push"). The estimator measures ‖E∇L‖ at a point. It measures no curvature and no drift.

**Correction.**
- Replace "indistinguishable from zero / no expected push" with "below a detection limit of ‖μ‖ ≈ 0.001–0.004 (1–3% of ‖E∇L‖) at B = 48".
- Recalibrate z against the empirical null, or use a permutation null.
- Relabel the planted control as an arithmetic check.
- Add a near-threshold gate-specific positive control and dense (replayed) pre-onset checkpoints.
- Withdraw the saddle-plus-noise inference as a conclusion; it may remain as one hypothesis among several.
- MAIN Q21's "No, it does not predict acquisition" can stand. Q8's "both kinds show zero first-order signal" should read "both are below the detection limit".

### 3. [MAJOR] For the two censored D continuations, late acquisition cannot be distinguished from non-acquisition. "Finite-budget barrier (Explanation B)" relabels censoring.

**Evidence** (inline hazard calculation; displacement from snapshots):
- Treat the D plateau sitters as an exponential escape process:
  - observed: 1 event (D_s204) in 1950 + 4000 + 4000 = 9950 post-restart plateau steps, or 12,000 with dev D_s100;
  - estimated rate 1/9950 to 1/12,000, 95% CI from 1/1786 to 1/393,000 per step;
  - at that rate, P(still censored after 4000 steps) = 0.67–0.72 per model, and P(both censored) = 0.45–0.51.
- A **homogeneous** slow hazard therefore fits the D outcome (1 of 3 acquired; 0.85–1.0 expected). There is no evidence of between-seed heterogeneity, i.e. a "selective" or seed-specific barrier.
- In C, fresh plateau durations of 800–3200 steps imply roughly a 5–6× higher hazard. "Slower acquisition in D" is as well supported as "barrier in D".
- "Parameters kept moving" does not discriminate:
  - Over 3500 steps the net displacement is 10.7 (D_s201) and 12.8 (D_s202), only 6–7% of the summed update norms.
  - Displacement over √Δt is 0.17–0.22 per √step, the same as the fresh C plateau (s403: 0.10–0.17). That is a momentum-correlated random walk; with β₁ = 0.9 it predicts about 12.
  - The D sitters moved exactly as the C sitters did before they escaped.
- C and D differ in both gate value (0.277 against 0.138) and cue strength (ν), no fresh D models exist, and the D models are selected failures. So nothing about D can be attributed to complexity, value or cue.

**Correction.**
- Describe D_s201 and D_s202 as "right-censored at total exposure 5.9, compatible with slower acquisition at an escape rate about 5× lower than C; a persistent barrier is not demonstrated".
- Drop "Explanation B, selectively" and "A3-like for those seeds" from EXTENDED §3–4 and MAIN Q6, Q8 and Q23–25. Alternatively, keep them only as a hypothesis.
- Report the hazard interval.

### 4. [MAJOR] Restart against constant rate is not identifiable. The "nothing indicates a restart-specific effect" reading depends on the axis chosen and on a selected comparison.

**Evidence:**

| axis | restart C (C_s201, C_s202, C_s203) | fresh C (s401–s404) | reading |
|---|---|---|---|
| total exposure, including XV's 2.0 | 3.10, 2.30*, 2.45 | 1.50, 1.00, 2.90, 3.40 | "comparable" (the report's choice) |
| exposure since the rate restart | 1.10, 0.30*, 0.45 | 1.50, 1.00, 2.90, 3.40 | restart faster |
| steps on the plateau before escape (this run) | 1200, 300*, 550 | 1300, 800, 2800, 3200 | restart faster (exact Mann–Whitney p = 0.11) |

\* C_s202 at the knife-edge, 3 of 6 slots (finding 1). With F_G ≥ 0.9 it becomes 2050 steps, or 2.05/4.05 exposure.

- The "0/3 by exposure 2.0" row for the restarts (EXTENDED §3; `xvi_stats.json`) holds **by construction**: the restarted models were selected because they had not acquired by 2.0. It is not an observation.
- Treating exposure as additive across a cosine tail and a constant rate contradicts the package's own mechanism hypothesis. If escape is noise-driven, its rate depends on the learning rate itself, since gradient-noise temperature scales with lr, and not only on Σ lr.
- The Adam-moment reset is confounded, as declared, and it has a visible signature. Continuations show larger update norms (0.043–0.056 against 0.032–0.037 in the fresh runs) with *smaller* gradient norms (0.11–0.19 against 0.23–0.31). The optimizer state differs, not just the rate.

**Correction.**
- Replace MAIN Q2's "Nothing indicates a restart-specific effect beyond restored exposure" with "not identifiable: on total exposure the routes look comparable; on post-restart exposure or plateau time the restarts look faster (n = 3 against 4, not significant); the restart is also confounded with a moment reset".
- Remove the selection-generated "0/3" cells, or footnote them.

### 5. [MAJOR] The cue-conditioned probes pool over random question slots, so they cannot time "representation against use". The slot-resolved probe gives a different onset at s501, while it strengthens the co-localisation claim.

**Evidence.** `probe_slot.py` is `xvi_probe.py` with the probed slot fixed; everything else is identical.

| model / slot | following | decorrelated after layer 2, evidence / readout | causal use at the evidence tokens (baseline) |
|---|---|---|---|
| WB_high_s501 @600, slot 5 | 0.70 | **0.69 / 0.71** | 0.70 (0.02) |
| WB_high_s501 @600, slot 0 | 0.00 | 0.35 / 0.33 | 0.08 (0.08) |
| WB_high_s501 @600, pooled (reported) | 0.13 | 0.36 / 0.41 | 0.18 (0.06) |
| C_s202 @1000, slot 0 (acquired) | 1.00 | 0.95 / 1.00 | 0.86 (0.00) |
| C_s202 @1000, slots 1 and 2 (not acquired) | 0.00 | 0.46–0.50 / 0.48–0.51 | = baseline (0.29–0.30) |

- **s501.** Gate information beyond the cue, together with causal use, is present in slot 5 at step **600**. The pooled probe misses it, and the report dates its first appearance to 700.
  - The report's own pooled table already shows causal use above baseline at 600–650 (0.18–0.22 against 0.05–0.06).
  - So "gate information beyond the cue first appears at the **same** checkpoint where following and causal use begin (… WB s501 @700)" is wrong as stated.
- **C_s202.** This is a clean natural experiment: one network with the gate in some slots and not in others. In the unacquired slots there is no linearly decodable g and no causal use; in the acquired slot both are present. **At slot level, representation and use co-localise.** This is better evidence for the co-emergence reading than the report's pooled transitions.

**Correction.**
- Slot-resolve the probes, at minimum around the three transitions.
- Restate Q18: "Within a slot, linearly decodable g beyond the cue appears with use. Pooled timings are not interpretable."
- Implement or delete the "keep-g donor" baseline. The docstring of `xvi_probe.py` promises it, but `use_baseline` is never filled in. Without it, a patch of evidence-token residuals from the flipped input could act through raw x_{a−1} content rather than g.

### 6. [MODERATE] Holding x₀ out of the decorrelated probe can mask real g information, at arity 3 as well. The report's "arity-3 shows no artefact" is contradicted by its own data.

**Evidence** (inline simulation; balanced g, as in the regated inputs; logistic probe trained on x₀ ∈ {0, 1} and tested on x₀ = 2):

| representation | added g component, strength 0 / 0.25 / 0.5 / 1 (noise 0.3) |
|---|---|
| arity 2, source-additive one-hot | 0.15 / 0.19 / **0.48** / 0.86 |
| arity 3, source-additive | 0.48 / 0.64 / 0.79 / 0.94 |
| arity 3, with a pairwise (x₁, x₂) code | 0.15 / 0.35 / **0.59** / 0.91 |

- A genuinely decodable g can read as "chance" when it coexists with structured source codes.
- In the data, at **arity 3**, WA_cont_C_s201's decorrelated probe at the gate tokens before layer 2 falls to **0.44 at step 1100 (F_G = 0.00)** and **0.38 at 1150**. It was 0.47–0.49 at 500 and 1000.
  - With about 1000 test items (SE ≈ 0.016), these are 3.8 and 7.5 SE below 0.5.
  - This is the below-chance signature of a newly structured gate-source code, appearing ≥ 50 steps before any following.
- LEARNING_SIGNAL §6.3 says "The arity-3 probes do not show this artifact (0.47–0.53 before onset)". That is false for C_s201.

**Correction.**
- Treat "decorrelated ≈ 0.5" as uninformative, not as "no g".
- Use a non-held-out or cross-fitted probe on regated inputs, with a nonlinear (pairwise-feature) control, or else a probe for partial sums (x₀ + x₁ mod 3).
- Report the C_s201 pre-onset dip as a candidate precursor (exploratory).

### 7. [MODERATE] The uncertainty for P3 is understated.

**Evidence.**
- The log ratios are 0.48, −0.35, −0.13 and 0.71; the mean ratio is 1.20.
- The t-based 95% CI for the paired ratio is **0.54–2.64**.
- An exact paired sign-flip test with 4 pairs cannot go below p = 0.125, so its 95% interval is **unbounded**.
- The picture is the same at F_G ≥ 0.9.

**Assessment.** "Effects of about 1.25–1.5× cannot be excluded" (FROZEN §4; MAIN Q10) implies that larger effects are excluded. They are not: XV's 1.4–2.6× lies inside the t interval. The INCONCLUSIVE verdict is correct.

**Correction.**
- State the 0.54–2.64 interval, and that no exact test with 4 pairs can exclude any effect at α = 0.05.
- Withdraw "the value effect disappears when the cue is held fixed" (MAIN Q23; also Review A finding 1).

### 8. [MODERATE] Under weak cues (ν = 1/3), the floor-conditioned stage is calibration-sensitive and close to circular. The XV-B "stage skipped" correction is not robust.

**Evidence** (tempered order-a optimum, `xvidiag.Diag`). In C the floor is KL ≤ 0.2916, with KL_oa = 0.2800 and a margin of only 0.0116 nats:

| exact agreement rule tempered by T | 0.80 | 0.90 | 1.10 | 1.25 |
|---|---|---|---|---|
| KL | **0.2944** (fails the 50% floor) | 0.2832 | 0.2820 | 0.2912 (fails the 75% floor of 0.2858) |
| A_conf | 1.00 | 1.00 | 1.00 | 1.00 |

- The KL leg in C and XV-B is therefore partly a calibration test, which explains the 2/4 under the 75% floor.
- **XV B_s204:**
  - F_G ≤ 0.02 from step 250 to 525 (12 points);
  - A_conf 0.35–0.57;
  - KL 0.290–0.299, straddling the 0.292 floor.
  - It "lacks the stage" because of the A_conf threshold (0.5) and a KL margin of a few thousandths of a nat, not because no agreement-using phase existed.
- In the same weak-cue regime, C passes and B fails mainly because C's gate plateau is long enough to accumulate ≥ 4 qualifying points. The stage call is then largely a restatement of the gate delay.
- Counting in points also makes the window depend on spacing: 4 points are 100 steps in WB, 200 in WA, and up to 1000 in the late XV snapshots.

**Correction.**
- Present the "corrected staging picture" (an agreement stage only when the cue is strong or the gate slow) as threshold-dependent and exploratory.
- Report time near the floor in steps and with temperature-matched KL references.
- State that STAGE_ROBUST mainly certifies "acquirers spent ≥ 4 points near the gate-free optimum first" (see also Review A finding 3).

### 9. [MODERATE] The frozen record conflicts on the P2 expectation, and the protocol file is not in the hash list.

**Evidence.**
- `logs/FREEZE_XVI_HASHES.txt` lists 32 files, and `FROZEN_PROTOCOL_XVI.md` is **not** among them, despite the protocol's first line. Review A finding 9 calls it hashed.
- The hashed `score_XVI.py` (whose hash matches) states "P2 … Frozen expectation: CONT_BROAD", which is the observed verdict.
- The protocol and every report say CONT_SELECTIVE and call it falsified.

**Correction.** Reviewer C should adjudicate from git history. Until then, "P2 expectation falsified" should not be asserted.

### 10. [MINOR] Confidence cannot produce any t_A. Abruptness survives, with caveats.

**Evidence.**
- t_A requires KL ≤ 0.75·KL_oa. No gate-free rule at any temperature reaches that (finding 8 table: the minimum is KL_oa). So none of the 16 acquisitions can be a confidence change.
- In WB-low, confidence moves only from 0.70 to 0.72 while F_G goes from 0 to 1 and KL halves.
- t_first (F_G ≥ 0.2) is not protected (Review A finding 5). F_G is near-binary, so its widths are lower bounds (Review A finding 4). Model-level widths add staggering between slots (finding 1).

**Correction.** Report per-slot and G-based widths. Qualify "abrupt" by the measure used.

### 11. [MINOR] Diagnostic density and replay are adequate. One verification is unlogged.

- The inline 50-step (WA) and 25-step (WB) diagnostics over the full runs meet D4.
- The 3-point "sustained" window means 100 steps in WA and 50 in WB. It did not exclude a knife-edge state (finding 1).
- I confirmed bitwise identity (max |diff| = 0.0) of the replayed checkpoints used for the probes against the archived snapshots: s402 @1000, s501 @500, and C_s201 @500 and @1000. Replayed F_G also matches the inline trajectories.
- The package logged this check only for the dev run and WB_low_s503. The claim "replays match the archived 500-step snapshots bitwise" is true, but should be logged.

---

## Mandated questions

**1. Is any claimed transition explained entirely by confidence changes?**
- No t_A is (finding 10).
- t_first, the graded G and A_p, and the KL leg of the stage in weak-cue conditions are confidence-sensitive (finding 8; Review A finding 5).

**2. Is any claimed transition explained by inadequate sampling or measurement?** Yes:
- C_s202's acquisition at 400, its "regression" and its "1850-step non-abrupt" transition;
- the FAST_UNLOCK descriptor (finding 1);
- the "signal appears only 150–250 steps before onset" (finding 2d);
- the pooled-probe onset at s501 @700 (finding 5);
- XV-B's "stage skipped" (finding 8);
- XV's arity-3 widths of 500–550 steps, which the package already correctly withdraws.

**3. Can late acquisition be distinguished from non-acquisition for D_s201 and D_s202?** No (finding 3). Under the escape rate estimated from the D data themselves, a 45–51% chance that both remain censored at 4000 steps is fully compatible with eventual acquisition.

**4. The detection limit of the gate-signal estimator.** On the plateau, ‖μ‖ ≈ 0.001–0.004 (calibrated 0.0007–0.003), or 1–3% of ‖E∇L‖. At D_s202 @4000 it is 0.0055 (10.5%).

**5. Is U unbiased?** Yes, exactly, for ‖μ‖² with independent pairs. Its jackknife SE is conservative: null z has SD 0.55 against 1.

**6. Are the controls adequate?**
- The null control is adequate but shows the conservatism.
- The planted control is uninformative (an identity).
- The natural positive control is far above the limit.
- A near-limit, gate-specific positive control is missing.

**7. Is exposure comparable across cosine and constant schedules?** Only as a path-length budget. It is not comparable as an escape budget if escape is noise-driven. The conclusions change with the axis chosen (finding 4).

---

## Claims that survive

1. **The definitions are applied correctly.** I recomputed all acquisition times from the raw trajectories: t_A of 1100, 1600, 3000 and 3500 (fresh); 1200, 400, 550 and 1950 (restart); and the WB times.
   - Censoring is handled correctly: no times are imputed except in P3's frozen lower-bound rule, which was not triggered.
2. **P1 is BROAD, 4 of 4** (95% CI 0.40–1.00). All four models were competent at the end, with no regressions through 15,000 steps.
3. **The P2 count of 4 of 6 is robust** to F_G thresholds from 0.5 to 0.9 and to KL fractions from 0.5 to 0.9.
4. **XV's three censored C models were not incapable:** 3 of 3 acquired after the restart.
5. **The architecture represents the arity-3 gate:** KL from Bayes 0.0003–0.0013 with F_G = 1.00 in 8 XVI models.
6. **Continuing the cosine schedule is provably inert.**
7. **Acquisition is not a confidence artefact** (finding 10). Transitions are abrupt relative to the plateaus, and per slot even sharper.
8. **On probes:**
   - The natural probes of censored models sit at the exact cue baseline (EVID), confirming that XV's D_s203 reading was a cue artefact.
   - The decorrelated probe separates acquired from censored models.
   - At slot level, representation and causal use co-localise (C_s202 @1000; finding 5). This is stronger than the pooled evidence offered.
9. **The rebuilt estimator is unbiased, with zero null mean.**
   - It shows that ‖μ‖ is small (≤ 1–3% of ‖E∇L‖) on the plateaus.
   - It shows that the signal grows by orders of magnitude through each transition.
   - It does not rank plateau durations, so "does not predict acquisition" (MAIN Q21) stands.
10. **P3 is INCONCLUSIVE.** No consistent value effect is observed, but neither is one excluded (finding 7).
11. **Replays are bitwise deterministic**, and so is resume.
12. **New, not reported:** acquisition in this architecture is slot-local, and a network can hold a working gate in some slots and lack it in others for over 1600 steps at the full rate. This should be added as an exploratory finding. It bears directly on mandate questions 22 and 25.
