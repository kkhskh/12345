# Early-training causality (Workstream B)

COR Phase 1.5, Step 1, Investigation XIII-E. This is a training-dynamics result about small transformers. It does not validate COR and makes no claim about the O/D operators.

## Question

XIII-D found that blocking either route for the first 100 steps permanently redirects the final organisation. That result had one run per direction, no sham, no windows shorter than 100 steps, and late interventions whose severity was not matched.

XIII-E asks two questions:
- Is the effect reproducible and specific to the blocked route?
- Is it specific to early training?

## Design

Frozen in freeze B (`b710b38`) before any B run. Harness: `scripts/ft_factorial_b.py`.

**Bases.**
- uP160 21/21/21 (early, "E").
- uP160 23/23/23 (mixture, "M").

**Data orders.** For each base, D ∈ {anchor, 801, 802}. The anchor-D baselines are the XIII-B natural seeds. The D = 801 and D = 802 baselines are new no-mask runs.

**Real route masks.** These are the XIII-D masks:
- ro0 blocks the early route (layer-0 readout→source attention);
- l2q blocks the mixture route (layer-2 readout→own-q attention).

Each was applied to its organisation's base for the first 25, 50 or 100 steps.

**Sham.** The masked code path with a zero bias.
- Development: bitwise identical to plain training over 25 steps.
- Confirmation: bitwise identical over all 1211 steps, for D = 801 on both bases.
- It is therefore a **code-path identity check**: it rules out any effect of the masking code itself. It is *not* the off-route-edge sham the mandate (B1) asked for; that sham was not run (deviation, see reviews/ADJUDICATION.md B-4).

**General-disturbance control.** Gaussian noise added to the residual stream at the layer-2 input for 100 steps.
- The noise uses its own key and consumes no data randomness.
- Its σ is matched to the real mask's immediate loss disruption on development checkpoints (`logs/b_calib.json`): σ = 2.32 on E (matched to ro0) and σ = 0.25 on M (matched to l2q). At steps 0–100, l2q barely moves the loss (Δ ≈ 0.002–0.014).
- A **strong-noise** arm uses σ = 2.32 on M.

**Late, severity-matched.** Windows 546–1146, branched at step 546, run to step 4000. All four conditions are matched to the late-l2q disruption (Δ ≈ 0.46):
- route-specific:
  - partial early-route block ro0p, k = 2 sources (Δ = 0.39), on E;
  - l2q (Δ = 0.46), on M;
- matched noise:
  - σ = 4.0 on E;
  - σ = 2.06 on M.

The full late ro0 (Δ = 1.29, XIII-D F1e) cannot be matched by noise, which saturates at Δ ≈ 0.6–0.8. It was not retested.

**Outcomes.**
- Class at step 1211 for short windows; class at step 4000 for late runs.
- A flip is a class different from the matching data-order baseline.
- Runs that fail the competence check (KL ≤ 0.05 at step 1211, ≤ 0.02 at step 4000) are excluded from flip counts.
- Four short runs were continued to step 4000.

## Results

All 41 runs, the 4 continuations and all metrics are present. `scripts/score_B.py` scored them. Full rows are in `logs/score_B.json`.

### Short windows (class at step 1211; flips out of 3 data orders)

| base | 25 steps | 50 steps | 100 steps | matched noise (100) | strong noise (100) | sham (D = 801) |
|---|---|---|---|---|---|---|
| E (early): real = ro0 | 1/3 (one hybrid) | 3/3 (2 hybrid, 1 mixture) | **3/3 mixture** | 0/3 | — | bitwise identical to baseline |
| M (mixture): real = l2q | 0/3 | 3/3 (all hybrid) | **3/3 early** | 0/3 | **2/2 competent early** (third: early, KL 0.051, excluded) | bitwise identical to baseline |

**Continuous values at step 1211**, from `score_B.json` rows:

| base | 25-step mask | 50-step mask | 100-step mask | matched noise |
|---|---|---|---|---|
| E (baseline early, lik ≈ 0.77, mix ≈ 0.17) | mix 0.22–0.58 | lik 0.10–0.30, mix 0.92–0.95 | lik 0.00, mix 0.95–0.96 | lik 0.78–0.80, mix 0.16–0.17 |
| M | — | lik 0.16–0.25, mix 0.93–0.94 | lik 0.75–0.77, mix 0.15–0.17 | — |

**Persistence after release.** All four continuations kept their step-1211 class at step 4000 (KL ≤ 0.005): the 100-step masks at D = 801 and the D = 801 baselines.

**R3 (early-mask specificity): CONFIRMED.**
- The 100-step real masks flip the organisation in 3 of 3 data orders on both bases.
- Disturbances matched for immediate loss flip 0 of 3.
- The code-path check is exactly neutral.
- *Post-review scope:* the noise is injected at the layer-2 input rather than where ro0 acts, and it favours the early organisation; on M the matched σ is the bottom of the grid and has no measurable effect. Specificity therefore rests on direction on E (ro0 pushes towards mixture, which no noise does) and on M's flip without loss disruption.

**Dose dependence.** The effect grows with window length:
- 25 steps flips 1 of 6 runs, and that one only to hybrid;
- 50 steps changes 6 of 6, mostly to hybrid (two M runs sit 0.006 and 0.025 above the 0.15 threshold and were not continued);
- 100 steps flips 6 of 6, to the opposite organisation.

### Late, severity-matched (class at step 4000)

| base / condition | anchor D | D = 801 |
|---|---|---|
| E: late partial early-route block (ro0p, k = 2) | hybrid (lik 0.36, mix 0.95) | hybrid (0.34, 0.90) |
| E: late matched noise (σ = 4) | early (0.81, 0.17) | early (0.81, 0.16) |
| M: late l2q | hybrid (0.43, 0.91) | hybrid (0.42, 0.93) |
| M: late matched noise (σ = 2.06) | mixture (0.04, 0.98) | mixture (0.03, 0.98) |

**R4 (specifically early versus general disruption): FALSIFIED.** Two of its clauses failed:
- Late route-specific blocks of matched severity changed the final organisation in all 4 cases, to a hybrid at step 4000. Late matched noise did not change the route class.
- Strong early noise on the mixture base flipped it to early (2 of 2 competent runs).

## Interpretation

**1. A route-specific effect exists.** At matched immediate loss disruption, blocking a route redirects the organisation and general noise does not. This holds both early (R3) and late. On the mixture base the early l2q mask barely changes the loss (Δ ≈ 0.01), yet it flips the organisation in 3 of 3 runs. So immediate disruption is not what drives the effect.

**2. It is not confined to an early critical period.**
- A matched late block (546–1146) still leaves a lasting change, but the outcome is a hybrid using both routes, not a full flip. This reproduces XIII-D's F1f, now with 2 data orders and on both bases.
- *Post-review correction.* Measured 65 steps after release (step 1211, as for the short windows), the late l2q block on M gives a **full flip to early** (lik 0.71, mix 0.29 / 0.38), which then partly reverses to hybrid by step 1803 (`logs/unit_metrics_Blate_traj.jsonl`, exploratory). The early 100-step flip does not reverse. On M the early–late difference is therefore one of **persistence/reversibility**, not degree. On E the two arms use different interventions (full ro0 for 100 steps vs ro0p k = 2 for 600 steps) and are not comparable; the late Δmix is as large as the early one.
- What is established is that route-availability sensitivity does not close after an early window; whether early effects are more persistent is supported on M only (2 runs per arm).

**3. Some general disturbances can also redirect.** Strong residual noise at the layer-2 input (σ = 2.32) flips the mixture base to early. This noise is not route-neutral: it degrades exactly what the layer-2 mixture heads read, so it partly acts as a mixture-route disturbance.
- Matched to l2q's tiny early disruption, noise does not change the route class (it does leave other lasting changes, e.g. E's layer-2 MLP share; the strong-noise early runs are degraded, slope_W 0.60–0.67).
- At about 86 times that noise variance (σ 2.32 against 0.25), it acts like a mixture-route block.

**4. The four candidate interpretations.**

| interpretation | what the data show |
|---|---|
| route-specific selection | supported |
| general early sensitivity | partly supported: strong layer-2-input noise also redirects. It is weaker than route blocking at matched severity |
| passive commitment vs. active competition | not distinguished by these experiments |

The unused route's information again persists in pooled form, with group-resolved presence reduced. In the D = 801 runs at step 1211 (`logs/route_status_B.jsonl`):
- early base after a 100-step ro0 block: pooled presence 0.93, group-resolved 0.76;
- early baseline: pooled 0.99, group-resolved 0.97;
- mixture base after l2q, now early-organised: 0.99 and 0.98;
- mixture baseline: 0.95 and 0.85.

**Reactivation** (`logs/route_status_B_react.jsonl`). The run is the D = 801 early base after the 100-step ro0 block, continued to step 4000.
- Retraining only the readout, with the mixture route blocked, raises accuracy from 0.346 to 0.553.
- With the layer-1 readout representation mean-replaced, it rises only from 0.335 to 0.365.

This replicates XIII-D (F1a: 0.333 → 0.592, and 0.32 → 0.37). The unused route is partly readable by a readout-only probe.

**5. Competence.** One strong-noise run failed the competence check and was excluded. Every other run was competent. No run that failed to train is counted as evidence for an alternative algorithm.

## Limits

- One pretrained parent (uP160), two bases, 3 data orders per short condition, and 2 per late condition.
- Severity matching uses immediate loss on development checkpoints, so other disturbance measures could rank the conditions differently.
- The noise control is not perfectly route-neutral (point 3).
