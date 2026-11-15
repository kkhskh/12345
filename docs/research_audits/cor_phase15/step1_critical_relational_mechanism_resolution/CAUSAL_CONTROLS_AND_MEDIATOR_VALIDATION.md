# Causal controls and mediator validation (Workstream D)

COR Phase 1.5, Step 1 (provisional ≈ 25%). Evidence labels as in the other documents: [dev] means development, [conf] means frozen
confirmation.

Every deficiency the XIII-B review listed is addressed below with an experiment. Scripts:
- `d_rows.py`: rows, decoys, no-op, distribution classes, D7 streams
- `d_routes.py`: D5 parallel routes
- `d_das.py`: D2/D4 DAS with corrected controls
- `metrics.py l2_split`: D6
- XIII-B `l2_decomp_all.py`: 6-component Shapley

## D1. Attention-head decoys rebuilt: contribution- and displacement-matched [dev; conf R6]

The XIII-B TOP head was not contribution-matched. The four layer-2 heads differ strongly in baseline contribution, so no natural head can be
matched on all six criteria. The non-mixture heads' baseline direct-logit shares are 0.00–0.05, against 0.2–0.6 for mixture heads.

We therefore built a **constructed, clearly labelled control, SHUFD**. The decoy keeps the mixture heads themselves and the base rows, and
adds the row displacement (donor − base) of *another* valid relational pair with the same target question, clipped at 0 and renormalised.
By construction this matches four of the criteria and carries the wrong relational content:
- head identity and dimensionality;
- token positions;
- attention mass and baseline contribution (same head);
- the distribution of displacement magnitudes (row L1 displacement 0.27–0.28, against 0.39–0.42 for TRUE; somewhat lower because of
  clipping).

**Specificity is judged by R², the variance of the Bayes change explained, not by slope.** This metric change is itself a finding:
- an unrelated large perturbation can have a positive regression slope on the Bayes change;
- SHUF, PMEAN and SHUFB rows give slopes of 0.3–0.5 with R² ≤ 0.06 and output changes 2–3 times larger than TRUE;
- XIII-B's projection- and slope-based decoy comparisons were therefore not specific tests.

| unit | mixture heads | TRUE slope / R² | SHUFD R² | SHUF R² (slope) | PMEAN R² | SHUFB R² | OTHER heads R² | NOOP max \|Δlogit\| | WHOLE R² |
|---|---|---|---|---|---|---|---|---|---|
| uP161 s21 | 1, 3 | 0.93 / 0.96 | 0.003 | 0.04 (0.52) | 0.05 | 0.03 | 0.01 | 0.0 | 0.97 |
| uP160 s23 | 2, 3 | 0.89 / 0.96 | 0.002 | 0.04 (0.49) | 0.06 | 0.03 | 0.04 | 0.0 | 0.98 |
| uP161 s20 | 3 | 0.71 / 0.90 | 0.002 | 0.04 (0.40) | 0.05 | 0.03 | **0.40** | 0.0 | 0.98 |
| uP165 s21 | 1, 3 | 0.87 / 0.94 | 0.005 | 0.04 (0.50) | 0.06 | 0.03 | 0.08 | 0.0 | 0.96 |
| uP161 s23 (hybrid) | 1 | 0.30 / 0.40 | 0.000 | 0.01 (0.13) | 0.01 | 0.01 | 0.15 | 0.0 | 0.93 |

The rows of the selected mixture heads carry a Bayes-shaped change that no matched or unmatched decoy reproduces. In uP161 s20 a second head
below the selection rule also carries part of it (OTHER R² 0.40).

## D2. Genuine no-op and corrected DAS controls [dev; conf]

**No-op.** The XIII-B "NOOP" was a QR of a tiny random matrix, which is an arbitrary subspace and not a no-op. It is replaced by:
- `d_rows` NOOP: the rows of the base itself;
- `d_das` NOOP: donor := base.

Both reproduce the unmodified forward.
- The DAS no-op is exactly 0 in every unit.
- The row-path no-op is exactly 0 in the development units, but 7e-7 to 9.5e-7 in the confirmation units: float32 reordering, because that
  path recomputes attention. This difference is why R6 was falsified as written.

**DAS controls, all reported separately:**
- same-state replacement (NOOP, = 0);
- random k-subspaces (2 draws);
- source-independent patch (SRCFREE): the trained subspace patched with the state of an unrelated episode with the same q;
- natural whole-site replacement (FULLSITE, k = 64);
- a matched-budget decoy subspace trained to express A2.

**New finding.** SRCFREE transfers a large own-share (0.3–0.7) in every unit. The cause is a base-rate effect: the edits move a group
toward its *typical* state, and an unrelated episode's state is usually typical. Source-dependence must therefore be measured as **excess
over SRCFREE**, not as a raw transfer. This also explains XIII-B's finding that "A1 is reproduced by source-free patches".

## D3. Distribution classes [dev]

Intervention classes used, from least to most artificial:
1. real-input edits (all pair families);
2. whole natural-activation replacement (WHOLE / FULLSITE);
3. component-level natural donor (TRUE rows; site interchange);
4. hybrid states (SHUFD: base + another pair's natural displacement);
5. synthetic or distribution-shifting patches (PMEAN position-matched mean; SHUF/SHUFB rows of unrelated episodes).

The XIII-B MEAN control averaged across target-question positions. It is replaced by PMEAN, the position-matched mean, which is still an
off-distribution average and flagged as such.

**What survives the less artificial classes.** The rows' Bayes-specific effect holds under classes 2–4 (TRUE R² ≥ 0.90 against SHUFD
≤ 0.005). Classes 4–5 produce large non-specific changes (R² ≤ 0.06), so no claim rests on them.

## D4. Eligibility-subspace identification across sites; E vs A1 vs A2 [dev; conf R7]

Families: E5, E3, QF5, QF4 (DECISION_LAW §B6). DAS is trained only to reproduce the model's *own* counterfactual outputs on a balanced
mix, so it is not trained toward any rule. Evaluation is on held-out pairs.

**Site coverage** (FULLSITE own-share, E5/E3/QF5/QF4):

| unit | RO2 | QS2 | OS2 | ROQS2 | ROOS2 | ALL2 | RO1 |
|---|---|---|---|---|---|---|---|
| uP161 s21 (mix) | .79/.28/.25/.01 | .19/.72/.74/.99 | 0/0/0/0 | .99/.99/.98/1.0 | – | – | 0/0/.01/0 |
| uP161 s22 (early) | .41/.37/.67/.95 | .21/.12/.11/.02 | .41/.47/.10/.03 | .62/.49/.80/.97 | .81/.84/.78/.98 | 1/1/1/1 | 0/0/.51/.91 |
| uP160 s21 (early) | .55/.20/.73/.94 | .11/.11/.07/.02 | .35/.68/.11/.04 | .67/.31/.82/.97 | – | 1/1/1/1 | 0/0/.51/.93 |

In the early class the relational eligibility information reaches layer 2 at the **other-question source tokens** (OS) and the readout
token, not at the q tokens. XIII-B searched only the q-token site, and that site covers just 0.02–0.21 here. **XIII-B's Q5a falsification
was a site-coverage artefact.**

**Best covering site, k = 8.** Columns are own-share and R² against E / A1 / A2:

| unit | site | E5 | E3 | QF5 | QF4 | excess over SRCFREE | rand max | A2-decoy subspace (own, R²_A2) on E5 / E3 |
|---|---|---|---|---|---|---|---|---|
| uP161 s21 (mix) | ROQS2 | .93 / .95 / .95 / .61 | .69 / .81 / .79 / .00 | .94 / .97 / .38 / .97 | .91 / .97 / .97 / .18 | **0.34** | 0.14 | .11 / .11 · .10 / .02 |
| uP161 s22 (early) | ALL2 | .83 / .88 / .88 / .62 | .73 / .63 / .63 / .00 | .82 / .89 / .35 / .89 | .82 / .94 / .94 / .22 | **0.33** | 0.12 | .23 / .12 · .17 / .13 |
| uP160 s21 (early) | ALL2 | .85 / .91 / .91 / .70 | .50 / .53 / .52 / .01 | .83 / .91 / .32 / .91 | .82 / .95 / .95 / .21 | **0.31** | 0.16 | .23 / .09 · .10 / .12 |
| uP161 s23 (hybrid) | ROQS2 | .73 / .77 / .78 / .60 | .45 / .40 / .40 / .00 | .76 / .85 / .41 / .85 | .81 / .92 / .92 / .18 | 0.26 | 0.15 | — |

**Reading.**
- At the covering site, an 8-dimensional subspace mediates the eligibility effect in both classes.
- It is source-specific: excess 0.31–0.34 over the source-free patch, and random subspaces give ≤ 0.16.
- It follows E against both alternatives: against A2 on E5 (margin 0.21–0.34), E3 (margin ≥ 0.4) and QF4 (≥ 0.7), and against A1 on QF5
  (margin ≥ 0.54).
- A subspace trained under the same budget to express A2 fails to: R²_A2 0.09–0.13 on E5 and E3. E-specificity is therefore not an
  automatic consequence of the objective.
- **Sensitivity to dimension:** k = 2 reaches only 0.13–0.48 own-share at every site. The mediator is low-dimensional but not
  one-dimensional.
- E3 (the n 4 ↔ 3 change of reliability weight without eligibility change) is the least well captured family (0.45–0.73).
- [conf R7] tests the early-class claim on new runs.

## D5. Parallel routes versus serial transmission [dev; conf R8]

The donor is injected only at its origin, the edited group's layer-0 output at q′. Three downstream routes are cut in every subset:
- **Q**: layer-1 writes into q's source tokens;
- **R**: layer-1 writes into the readout token;
- **O**: everything else reaching layer 2 from the other-question tokens.

Sufficiency is the share transferred with only that route open.

| unit | Q | R | O | redundancy (Σ suff − 1) | interpretation |
|---|---|---|---|---|---|
| uP161 s21 (mix) | 0.40 | 0.60 | 0.01 | +0.01 | two parallel additive routes |
| uP161 s20 (mix) | 0.69 | 0.31 | 0.02 | +0.01 | two parallel additive routes |
| uP160 s23 (mix) | 0.59 | 0.34 | 0.02 | −0.05 | two parallel additive routes |
| uP161 s23 (hybrid) | 0.46 | 0.36 | 0.20 | +0.03 | three additive routes |
| uP161 s22 (early) | 0.17 | 0.40 | 0.44 | +0.01 | R and O dominate |
| uP160 s21 (early) | 0.09 | 0.41 | 0.47 | −0.03 | R and O dominate |

**Answer to Q14.**
- Genuinely separate routes exist, and they are **additive, not redundant**:
  - cutting one route is not compensated by another (necessity ≈ sufficiency);
  - opening a second route only adds its own share.
- No backup pathway restores a removed route. XIII-B's "restoration" was a downstream re-injection on the same chain; the claim is now
  limited to that serial transmission.
- The split between Q and R, not the presence of two routes, is what varies between units. It is init-dependent; see §D7.

## D6. The decision effect outside the dominant attention pathway [dev]

**Two-player Shapley at the readout token** (attention output, all heads / MLP2) on purely relational pairs:

| class | attention | MLP2 | interaction (both − A − M) |
|---|---|---|---|
| mixture (uP161 s20, s21; uP160 s23) | 0.79–0.86 | 0.13–0.20 | −0.22 to −0.27 |
| hybrid (uP161 s23) | 0.62 | 0.25 | |
| early (uP160 s20, s21; uP161 s22) | 0.43–0.54 | 0.39–0.52 | −0.41 to −0.45 |

These are final-checkpoint values from `logs/unit_metrics_traj_xiiib.jsonl`.

**The negative interaction is large, especially in the early class.** Patching attention alone or MLP alone each recovers most of the
effect, while patching both recovers only 0.94–0.98. The two components are partly **redundant** in the early class. Shapley shares are
therefore reference-dependent and should not be read as unique contributions.

**Outside the readout token** (D5): in early units 0.44–0.47 of the relational effect comes through route O, layer-2 attention reading
other-question tokens. In hybrids 0.20 does.

## D7. Borderline routing claims: stable unit property, set by initialisation [dev; conf]

**Stability across streams.** route_Q over four independent relational-pair streams (`d_rows`, D7):

| unit | route_Q streams | mean ± sd |
|---|---|---|
| uP161 s21 | 0.42, 0.38, 0.35, 0.40 | 0.39 ± 0.03 |
| uP160 s23 | 0.66, 0.68, 0.64, 0.67 | 0.66 ± 0.02 |
| uP161 s20 | 0.73, 0.63, 0.66, 0.66 | 0.66 ± 0.04 |
| uP165 s21 | 0.60, 0.65, 0.59, 0.60 | 0.61 ± 0.03 |
| uP161 s23 (hybrid) | 0.43, 0.42, 0.43, 0.34 | 0.41 ± 0.04 |

route_Q is a stable property of a unit, with stream-to-stream sd ≈ 0.03. XIII-B's Q3b/Q4c failure is a real low value, not noise.

**Cause.** The factorial cell uP161 L21/D22 shares its initialisation with s21 but has a different data stream, and gives route_Q = 0.37,
against 0.39 for s21. The routing proportion therefore follows the initialisation (CAUSES §A).

**Threshold sensitivity.** With a routing threshold anywhere in 0.40–0.55, uP161 s21 and its init-sibling fall below it and every other
mixture unit stays above it. No single categorical routing law holds across initialisations.

## Confirmation (frozen)

**R7 confirmed, narrowed by review 3 (M6).** The model's own behaviour already shows the same E-over-A2 margins, and ALL2 includes the edited tokens, so the subspace may partly re-inject the input edit. The claim is that the model's E-following behaviour is carried by a low-dimensional subspace of the full layer-2 input, beyond the source-free and A2-decoy controls. It is not a localized mediator.

 The new early unit uP163 L22/W22/D21, at ALL2 with k = 8, gives:
- source-specific excess 0.36;
- E beats A2 on E5 (0.89 vs 0.67), E3 (0.58 vs 0.00) and QF4 (0.94 vs 0.26);
- E beats A1 on QF5 (0.91 vs 0.35);
- the A2-trained decoy subspace fails (0.06 / 0.02);
- no-op exactly 0.

**R8 confirmed.** Parallel additive Q/R routes in the mixture unit (0.54 / 0.45, redundancy 0.00), and a direct O route in two early units
(0.39, 0.50).

**R6 falsified, on the strict no-op criterion only.**
- In the single pure-mixture confirmation unit, TRUE R² is 0.97, SHUFD 0.010, SHUF/PMEAN/SHUFB ≤ 0.046 and OTHER 0.009.
- The row-patch no-op reproduced the forward pass to 7.2e-7, not to exactly 0. The frozen text demanded exactly 0, which the development
  units had happened to give. The cause is float32 summation order, because the row-patch path recomputes attention through a different
  einsum. The DAS no-op is exactly 0.
- Status: specificity survives in one confirmation unit, plus four development units. **The no-op is valid at tolerance 1e-6 and the frozen
  prediction is falsified as written.**
- In hybrid confirmation units the rows carry less (R² 0.24–0.81), and other heads more (0.13–0.33).
