# XIII-E freeze A-2: re-roll predictions, analysis and fail-closed scorer

Committed before any measurement of a re-roll run. No unit_metrics job for the re-roll runs has been queued. At commit time, 2 of the 64
training runs had finished, and only their step counts were seen (`scripts/monitor_A.py`).

## Hypotheses (mandate A1) and what this design can separate

- H0 (re-roll): re-drawing W (or L) gives another draw from a distribution that may depend on the parent and on the components held
  fixed. The new seed carries no property that transfers to another context.
- H1 / H3 (structure): a property of the re-drawn tensor, alone or jointly with the fixed components, predicts the outcome.
- Design consequence: the crossed design observes every new seed in two contexts per parent (the early anchor's fixed components and
  the mixture anchor's). A transferable property of the seed makes outcomes concordant across contexts; under H0 they are independent.
  H2 (asymmetric stability) is reported descriptively. It cannot be separated from the context-specific re-roll probabilities with the
  anchors selected on outcome (mandate A4); the anchors' own outcomes are never used as draws.

## Frozen predictions

- **R1-W** (re-roll null for the readout map): no transferable component of a re-drawn W. Statistic, permutation null, bootstrap
  upper bound and decision rule are in `scripts/score_A.py`: TRANSFER if p < 0.05 and rho >= 0.3; NO-TRANSFER if p >= 0.05 and the
  upper bound < 0.6; otherwise INCONCLUSIVE. Verdict: CONFIRMED iff NO-TRANSFER, FALSIFIED iff TRANSFER, otherwise INCONCLUSIVE.
- **R1-L**: the same for a re-drawn layer-2 initialisation L.
- Power (pre-freeze simulation, `logs/A_power.json`):
  - under the null: 67% NO-TRANSFER, 4% TRANSFER;
  - under a latent cross-context correlation of 0.6: 69% TRANSFER, 7% NO-TRANSFER.
  - A NO-TRANSFER verdict therefore bounds the transferable component, roughly c < 0.6. It does not show the component is zero.
- **R2 (structured joint statistic) is NOT frozen.** Development (26 configurations with known outcomes, `logs/r2_dev.jsonl`) found no
  step-0 statistic with a consistent within-parent direction:
  - I_mix AUC 0.65;
  - R_route AUC 0.36, with within-parent AUCs of 1.0 / 0.2 / 0.22 / 0.0;
  - G_ratio AUC 0.56.
- **Exploratory only:** a pre-declared held-out AUC of R_route (direction: negative, from 3 of 4 parents) on the 64 runs. Its inputs
  (`logs/r2_A.jsonl`) were computed from initialisations only, before training outcomes existed.

## Other frozen analyses (descriptive)

`score_A.py` also computes:
- per-cell Clopper–Pearson intervals;
- Newcombe intervals for early-versus-mixture-anchor differences per parent;
- an exact homogeneity test across anchors;
- parent baselines;
- class concordance;
- step-1211 versus step-4000 stability for the continuation set (A5; selection rule frozen in `scripts/select_continuation.py`);
- threshold sensitivity.

## Fail-closed scoring (mandate 7.2)

`score_A.py` refuses (exit 2, writes nothing) unless every run, metric, configuration, continuation manifest and continuation result
listed in `logs/manifest_A.json` is present and consistent. The preflight (`scripts/preflight_A.py` → `logs/preflight_A.json`) passes:
- P0: re-roll synthetic data, no TRANSFER;
- P1: transfer synthetic data, TRANSFER;
- N1–N7: every missing, mismatched or malformed input is refused.
