# Frozen predictions and results (XVIII)

**Freeze.**
- Commit `1547ec9`, 2026-11-18T19:54Z.
- Design: PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10.
- Frozen-file hashes: `logs/FREEZE_XVIII_HASHES.txt`.

**Scoring.** `scripts/score_XVIII.py` is fail-closed. Its output is `logs/score_XVIII.json`.

**Labels:**
- **[frozen]**: the frozen scorer's output.
- **[post-hoc]**: an analysis written after the frozen output was seen (`scripts/posthoc_nobehav18.py`).
- **[dev]**: a development run.

## 1. Cohort and what ran

**Within-parent confirmation (uP160).**
- 2 bases × 3 fresh data orders (D = 811, 812, 813) × 8 arms = **48 runs**, all completed with rc = 0.
- Measured at steps 1000 and 1211 on stream 95_400_000, which had never been used before.
- Statistical unit: base × order pair, n = 6, within one parent, so effectively 2 clustered bases.

**Post-freeze deviation (logged in `logs/COMPUTE_LOG.txt`).**
- `scripts/h3_immediate18.py` wrote its JSON to a file named `23` because it overwrote `sys.argv`. The scorer therefore refused, fail-closed.
- I fixed the output path only. The recomputed file is identical to the misdirected one.
- I then ran the scorer by hand. The stability continuations were deferred until after the parent stages.

## 2. The frozen primary result: INCONCLUSIVE, caused by a design weakness

**Only 2 of the 6 pairs are evaluable (E|812 and M|812)** [frozen].

The frozen evaluability rule requires the fresh baseline's class to be unanimous across three diagnostics, including the behavioural gate `cls_behav2`. That gate was calibrated on XIV checkpoints with the development stream:
- EARLY: lapse ≥ 0.017;
- MIXTURE: lapse ≤ 0.017.

On the confirmation stream, 4 of the 6 fresh baselines fall outside the behavioural bands:
- E|811 has lapse 0.010, below the 0.017 an EARLY class needs;
- E|813 has lapse just under 0.017;
- M|811 has lapse 0.018, above the 0.017 a MIXTURE class allows;
- M|813 has ρ_L 0.64, below the 0.68 a MIXTURE class needs.

So their behavioural class is HYBRID, even though their structural and causal classes are unambiguous:
- E baselines: nec_E 1.36–1.58, nec_M ≤ 0.006;
- M baselines: nec_M 0.65, nec_E ≤ 0.04.

The E|811 block is also behaviourally HYBRID (lapse 0.018).

Pre-review Reviewer B (addendum, fix 4) had flagged this classifier as "really a lapse threshold" and fragile. The pre-registered response was a sensitivity analysis without the behavioural gate (§10.8).

**Implementation gap (my error).** In `score_XVIII.py` the sensitivity analysis drops the behavioural gate from the *bands* but not from the *evaluability* test. So the frozen sensitivity output also has only 2 evaluable pairs. Section 4 applies the pre-registered text in full, labelled post-hoc.

**Frozen verdicts** (2 evaluable pairs; symmetric rule: CONSISTENT needs ≥ 5 matches, CONTRADICTED needs ≥ 2 mismatches):

| hypothesis | frozen verdict | frozen sensitivity (bands without behaviour, same 2 pairs) |
|---|---|---|
| H1d (forward content) | CONTRADICTED | CONTRADICTED |
| Hm_window (forward share in the window) | CONTRADICTED | CONTRADICTED |
| Hm_release (latent share at release) | INCONCLUSIVE | INCONCLUSIVE |
| H2g_R (routing gradient) | CONTRADICTED | CONTRADICTED |
| H2g_RV (all learning signal through T) | CONTRADICTED | CONTRADICTED |
| H4 (both content absence and low share required) | CONTRADICTED | CONTRADICTED |
| H1d_plus_Hm | CONTRADICTED | CONTRADICTED |
| H1d_plus_H2g | INCONCLUSIVE | INCONCLUSIVE |

- **Per-base frozen verdicts:** all INCONCLUSIVE, with 1 evaluable pair per base.
- **Arm-level pooled bands:** all INCONCLUSIVE, because fewer than 5 evaluable pairs remain.
- **H3 on E** [frozen]: **not excluded**. The rule needs vrg = F on at least 2 evaluable E pairs, and only E|812 is evaluable.

**The share clamp worked as designed** [frozen]:
- maximum achieved-versus-target error 1.8e-7;
- |β| > 1 on 0–13% of window steps;
- no sha run was marked UNINTERPRETABLE.

## 3. Per-pair results (the data the verdicts rest on)

The table shows φ_nec at step 1211 with the band without the behavioural gate. φ_nec is the log-necessity shift relative to the paired baseline (0) and block (1).

| pair | vrg | gsg | gsv | sha-hi-rolled | sha-lo-real | gof b = 2.3 |
|---|---|---|---|---|---|---|
| E\|811 | 1.02 F | 0.89 F | 0.82 F | 1.02 F | 0.03 N | 1.00 F |
| E\|812 | 1.04 F | 0.65 P | 0.81 F | 1.04 F | −0.03 N | 1.02 F |
| E\|813 | 0.99 F | 0.88 F | 0.85 F | 0.98 F | −0.00 N | 0.99 F |
| M\|811 | 1.00 F | 0.43 P | 1.02 F | 1.00 F | −0.02 N | 0.98 F |
| M\|812 | 1.00 F | 0.41 P | 0.98 F | 1.00 F | −0.00 N | 0.99 F |
| M\|813 | 0.93 (MISSING) | 0.36 P | 0.96 (MISSING) | 0.94 (MISSING) | 0.02 N | 0.98 (MISSING) |

**Competence.**
- Every run has KL ≤ 0.034.
- The four runs marked MISSING on M|813 fail only the frozen accuracy criterion (≥ the baseline's accuracy − 0.03): they score 0.832–0.839 against 0.873.
- The early organisation is systematically less accurate than the mixture organisation, so this criterion penalises M → early flips. That is a rule artefact, reported as such. The runs are otherwise competent, with KL 0.023.

**Mediator: latent own-question share at step 100** (φ_share, where 1 is block-like and 0 is baseline-like):
- vrg: 0.87–0.94.
- sha-hi-rolled: 0.86–0.90.
- gsv: 1.01–1.03.
- gof: 0.99–1.00.
- sha-lo-real: E −0.20 to −0.31; M 0.07–0.08. The latent share rises **above** the baseline's, with release jumps of +0.12 to +0.14 on E and +0.03 on M.
- gsg: E 0.49–0.55; M 0.82–0.85.

**Immediate Δloss** at the baseline's step-100 parameters:
- vrg on E 0.084–0.112, below the 0.18 of XIV's non-flipping E noise; on M 0.011–0.013.
- sha-hi-rolled: the same values.

## 4. [post-hoc] The pre-registered sensitivity analysis, applied in full

Here the baseline and block classes come from structural + causal agreement and the bands drop the behavioural gate. Everything else is frozen (`logs/posthoc_nobehav18.json`).

**Evaluable pairs:** 5 of 6. M|813 is excluded because its block fails the accuracy criterion.

**Pooled verdicts: every one of the eight pre-registered hypotheses is CONTRADICTED.** Per base, all are contradicted on E. On M all are contradicted except H1d_plus_H2g, which is INCONCLUSIVE.

| hypothesis | where it fails |
|---|---|
| H1d | gsv flips in 5/5 pairs with the forward content intact |
| Hm_window | sha-hi-rolled flips 5/5; sha-lo-real is N in 6/6 |
| Hm_release | gsg on M: latent share block-like (φ_share 0.82–0.85), outcome only P in 3/3 |
| H2g_R, H2g_RV | vrg and sha-hi-rolled flip 5/5 while the routing gradient through T is present |
| H4 | gsv flips; sha-hi-rolled flips |
| H1d_plus_Hm | sha arms are F and N, not P |
| H1d_plus_H2g | gsg is F, not P, on 2 of 3 E pairs |

**H3 on E (post-hoc):** vrg is F on 3/3 E pairs, with immediate Δloss 0.08–0.11 < 0.18. **Excluded.**

## 5. Stability readout, and the independent parents

(Added when stages 2b, 3 and 4 finish.)
