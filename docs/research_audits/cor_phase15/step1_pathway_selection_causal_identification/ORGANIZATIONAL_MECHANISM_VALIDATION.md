# Validating that the intervention changes the learned computation, not just an attention statistic (mandate §5)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels:**
- [dev]: development data (XIV checkpoints and XIVIII development runs);
- [frozen]: XVIII confirmation, scored by `score_XVIII.py`.

This report covers the development and frozen parts of Experiment B. The confirmation numbers are in FROZEN_PREDICTIONS_AND_RESULTS.md.

## B1. Behavioural diagnosis, independent of attention

**What was tested.** Two behavioural quantities, both fitted only to the model's output distribution on 3000 natural episodes.
- **XIII-B's nested power-mean exponent ρ.**
  - It separates the two XIV organisations completely: early 0.48–0.58, mixture 0.68–0.81 over 24 runs [dev].
  - Pre-review B (N2) showed that ρ moves strongly with a *lapse*-type error. Exact Bayes mixed with a uniform lapse at KL ≈ 0.016 already gives ρ ≈ 0.76.
- **A lapse-aware fit (ρ_L, plus the lapse ε), adopted for that reason.**
  - It recovers exact references: ρ_L = 1.000 and ε equal to the true lapse for ε = 0, 0.05, 0.10 and 0.15 [dev, `logs/dev/rhoL_xiv.json`].

**Findings on the 24 XIV runs** [dev]:
- ρ_L alone no longer separates the classes cleanly. One early run (E-sham, D = 21) reaches 0.73, against a mixture minimum of 0.69.
- The fitted lapse separates them completely: early 0.020–0.041, mixture 0.000–0.014.
- The behavioural rule `cls_behav2` classifies all 24 correctly:
  - EARLY if lapse ≥ 0.017 and ρ_L ≤ 0.75;
  - MIXTURE if lapse ≤ 0.017 and ρ_L ≥ 0.68.

**What this means.** The early organisation is systematically less accurate, with lapse-like errors, and the mixture organisation is closer to Bayes. That is a real behavioural difference between the two competent organisations. But it is partly a difference in competence level. It is **not** evidence of a different "combination operator": ledger N20 stands, and that reading is not unique.

**How the behavioural class is used.** It is used **only as a gate**. The primary outcome is causal (B2), and every verdict is also reported without the behavioural gate (frozen sensitivity rule).

**Behaviour drifts with training** [dev]:
- ρ_L rises as KL falls: gsg hybrids 0.57 → 0.86 between steps 1211 and 4000.
- The behavioural class is therefore not used at the step-2546 stability readout.

## B2. Causal diagnosis

**Measure.** Test-time route necessity:
- nec_E: rise in KL when the layer-0 readout → sources route is blocked;
- nec_M: rise in KL when the layer-2 readout → own-question-sources route is blocked.

**On the 24 XIV runs** [dev], the two measures give a complete double dissociation:

| class | nec_E | nec_M |
|---|---|---|
| early | 1.18–1.38 | ≤ 0.017 |
| mixture | ≤ 0.10 | 0.57–0.74 |

Every XIV real-block flip follows its new class on both measures.

**Additional necessity checks** [dev]:
- **own0-only.** At test time, blocking only the own-question edges at layer 0 produces most of nec_E in early models (own0 gives 0.81–1.03 of the full-block necessity; 0.98–1.28 against 1.18–1.38; `logs/dev/necessity_xiv.jsonl`). A 100-step training block of own0 alone flips E: nec_E 0.004, nec_M 0.81.
- **rq1** (layer-1 readout pooling). This route is necessary in early models (0.32–0.65 in development runs) and less so in mixture models (0.09–0.15); hybrids 0.08–0.18. It is reported descriptively. It is **not** a "third route": pre-review B addendum.

**Independence (pre-review B §2).** The structural diagnostic (mix_mass, lik_RO1) and the causal diagnostic both read the two manipulated edge sets. They are two measurements of one latent quantity: which route the trained model uses. Their agreement is therefore not independent confirmation.
- This is acceptable here because the interventions end at step 100 and the measurement is taken at step 1211, on the unmodified model.
- The only diagnostic that is not a function of the edges is the behavioural one, and it is weaker (B1).

## B3. Competence

- **Rule.** A run is competent if KL ≤ 0.045 and its accuracy is at least the paired baseline's minus 0.03.
- **Development.** Every development arm is competent at step 1211: KL 0.009–0.031, accuracy 0.83–0.87.
- **Hybrids.** The gsg hybrids are competent and stable to step 4000 (KL 0.002 at 4000) [dev, `logs/dev/gsg_continuation.jsonl`]. A hybrid here is a **competent third organisation**: both routes partly necessary (nec_E ≈ 0.10–0.17, nec_M ≈ 0.38–0.47). It is not failed learning.

## B4. Classification rule

The frozen rule (§10.4 of the design):
- **cls2.** The structural, behavioural and causal diagnostics must all agree, otherwise the class is HYBRID.
- **Bands.** F, P and N are defined by the log-necessity outcome φ_nec relative to the paired baseline and block, together with cls2.
- **Sensitivity.** Every verdict is also given without the behavioural gate.

## B5. Results on the confirmation cohort

(Filled in from `logs/score_XVIII.json` after scoring.)
