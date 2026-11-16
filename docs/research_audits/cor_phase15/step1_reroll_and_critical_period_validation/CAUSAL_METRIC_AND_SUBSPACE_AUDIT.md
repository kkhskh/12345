# Causal-metric and subspace audit (Workstream C)

COR Phase 1.5, Step 1, Investigation XIII-E. This report validates measurements on small trained transformers; it says nothing about the COR O/D operators. XIII-D's frozen verdicts are preserved as they were scored. This document changes only what they mean.

**Material.**
- Models: the two XIII-D carrier units. Both are early01-adapted two-colluder models (β = 0.6):
  - C-C1 = uP160 s23;
  - C-C2 = uP165 s21.
- XIII-D's E-workstream outputs.
- No model was trained for this workstream.

**Scripts and outputs.**
- `scripts/c1_ideal_refs.py` → `logs/c1_ideal_refs.json`. Exact-posterior reference mechanisms, pure numpy, adapted from the XIII-D review-1 scripts.
- `scripts/c_audit.py` → `logs/c_audit.json`. This is the XIII-D carrier pipeline with fresh streams (96.2 M pairs; directions fitted on 96.2 M + 3; held-out decoding on 96.2 M + 50), plus the C2–C4 analyses below.

**Prediction protocol.**
- C-C1 served as development.
- R5 and R6 were frozen for C-C2 before its audit ran (freeze C, commit `87177ad`; the C-C2 job was gated on a flag written after the commit).

## C1. Reference values for every XIII-D causal metric

**Reference mechanisms.** In all of them, the "model" is the exact Bayes posterior.
- **Ideal:** the patched subspace carries exactly the variable in question:
  - m (two-colluder);
  - the agreement counts n_g (C = 1, eligibility);
  - the role responsibilities (C = 1, rows).
- **Null:** the patched subspace carries nothing the decision uses.
- **Decoy:** the patched subspace carries a statistic that the exact decision ignores. On m-changing pairs a decoy carrier scores exactly like the null. Null preservation therefore cannot separate an m-carrier from a decoy carrier in an exact-Bayes model.

Uncertainty on the ideal values: bootstrap over pairs, 500 resamples. Observed values are from XIII-D. Only unit-level values were saved, so the observed column gives the range across units, not a sampling interval.

| metric (definition) | ideal | null | observed (XIII-D confirmation units) | what the comparison can identify |
|---|---|---|---|---|
| SUFF: interchange slope on the exact change | 1.000 | 0 | 0.90 / 0.82 | ideal vs null: discriminating |
| SRCF: same subspace and site, unrelated same-q donor | 0.720 [0.678, 0.764] | 0 | 0.59 / 0.59 | observed is near ideal. Not a token-location test, because it shares the −B(X_a) term with the target |
| DAS − SRCF excess | 0.280 | 0 | 0.32 / 0.24 | consistent with an ideal carrier. As a specificity test it is weak: ideal vs null differ by only 0.28 |
| reverse: donor run with the base projection (remaining slope) | 0.000 | 1 | 0.059 / 0.038 | discriminating: survives, close to ideal |
| restore: base mean-replaced, donor intact | 0.462 [0.440, 0.486] | 1 | 0.35 / 0.24 | below ideal by 0.11–0.22. The XIII-D claim of non-separability is retracted; the modest shortfall remains |
| mean-replacement damage KL on natural episodes | 0.347 [0.331, 0.364] | 0 | 0.50 / 0.45 | an ideal carrier must damage, because decisions depend on m. Only the excess of 0.10–0.15 is a candidate off-manifold artefact. P4b is non-diagnostic |
| zero-ablation damage KL | 0.347 | 0 | 0.57 / 0.51 | as above; excess 0.16–0.22 |
| null-pair (J_D0) transfer | 0 | 0 | ≤ 0.09 | cannot separate ideal from null or decoy. Shows only that the patch does not create spurious changes |
| E1 DAS own-share (C = 1 eligibility) | 1.0 | 0 | 0.53–0.89 | ideal vs null: discriminating |
| E1 SRCFREE own-share: E5 / E3 / QF5 / QF4 | 0.50 / 0.40 [0.19, 0.61] / 0.40 / 0.50 | 0 | uP165 s20: 0.55 / 0.92 / 0.61 / 0.38; uP164 s21: 0.56 / 0.50 / 0.67 / 0.56 | Not normalised by patch size: the E3 SRCFREE cosine with the target is only 0.08 in the ideal. P6's "≥ 0.15 excess" criterion is ill-posed; the ideal excess is 0.50–0.60. The uP165 s20 E3 value of 0.92 exceeds the ideal 0.40, but by a metric that rewards magnitude |
| E2 TRUE row interchange R² | 1.0 | 0 | 0.90 / 0.91 | discriminating: survives |
| E2 SHUFD vs base R² | 0.007 [0.000, 0.021] | 0 | 0.014 / 0.014 | ideal and null are both ≈ 0: non-diagnostic |
| E2 SHUFD vs borrowed R² | 0.003 | 0 | 0.003 / 0.001 | no power by construction: non-diagnostic |

Proposition S was re-verified: transplanting m reproduces the donor posterior to 4.9e-15.

## C2. Decoder quality, measured directly

The decoder sets are ridge filters (DEC). The comparison sets are:
- encoding patterns (ENC, Haufe);
- class means (CCD);
- DAS and DAS_DECOY;
- random subspaces (RAND);
- variance-matched random subspaces (VMRAND): directions drawn from N(0, Σ_site), so they lie in high-variance directions.

All use the same site (q-source + readout tokens) and the same J_DM pairs.

| | C-C1 | C-C2 |
|---|---|---|
| token-level ridge decoder, held-out R² for m (its own predictions) | 0.50 | 0.47 |
| held-out R² for m from the 3-d site projection: DEC_m / ENC_m / DAS / DAS_DECOY / RAND0 / VMRAND0 | 0.86 / 0.86 / 0.78 / 0.74 / 0.58 / 0.64 | 0.77 / 0.69 / 0.69 / 0.69 / 0.46 / 0.34 |
| signal variance (variance of E[x \| m]) along DEC_m / ENC_m / DAS / VMRAND | 0.94 / 15.7 / 13.6 / ≈ 4 | 1.74 / 13.4 / 11.4 / ≈ 3.5 |
| noise variance along DEC_m / ENC_m / VMRAND | 0.52 / 39 / ≈ 70 | 1.23 / 48 / ≈ 76 |
| mean patch displacement on J_DM pairs: DEC_m / ENC_m / DAS / VMRAND | 0.88 / 3.85 / 3.56 / ≈ 1.8 | 1.36 / 3.72 / 3.45 / ≈ 1.4 |
| interchange slope: DEC_m / ENC_m / DAS / VMRAND (3 seeds) | −0.00 / 0.84 / 0.91 / 0.07–0.16 | 0.02 / 0.75 / 0.81 / 0.03–0.09 |

**What this shows.**
1. **The decoder directions do decode m.** From the site projection they decode as well as, or better than, the encoding patterns (0.86 vs 0.86; 0.77 vs 0.69). So the information is readable along them.
2. **Interchange along them transfers nothing.** These directions carry very little variance: the signal variance is about 1/10 of ENC and the noise variance about 1/40. The donor–base difference projected onto them is therefore small. That is the correlated-noise geometry review 1 identified: filters cancel noise, while patterns follow the signal.
3. **Variance alone is not what makes a direction work.** Variance-matched random directions have more total variance than ENC but four times less signal variance, and they fail (≤ 0.16). The directions that transmit are the ones aligned with the signal (ENC, CCD, DAS).
4. **The phrase "decodable but not used" is not supported and is withdrawn.**
   - What is established: decodability from a direction says nothing about whether interchange along it moves the decision.
   - What it does not mean: it is not evidence that the model ignores readable information.
   - Use itself is shown by the pattern-direction interventions and by the reverse test.
5. **Caveat.** Site-pooled projections decode m even from random directions (R² 0.34–0.58). Projection decodability is therefore a weak measure of a set's content.

**R6 (frozen for C-C2): CONFIRMED.** Values for C-C2:
- decoder/encoder decoding ratio: 1.11;
- SUFF along DEC_m: 0.024;
- SUFF along ENC_m: 0.75;
- mean SUFF along VMRAND: 0.068;
- signal-variance ratio: 7.7;
- overlap of DAS with ENC_m: 0.58;
- overlap of DAS with DEC_m: 0.066.

## C3. How similar are the successful subspaces?

Overlap = ‖P₁ᵀP₂‖²_F / k (1 = same span; about k/64 ≈ 0.05 for random spans). Euclidean metric:

| pair | C-C1 | C-C2 |
|---|---|---|
| DAS seed 11 vs DAS seed 21 | 0.42 | 0.69 |
| DAS vs ENC_m | 0.40 | 0.58 |
| DAS vs CCD_m | 0.39 | 0.58 |
| DAS vs DAS_DECOY | 0.33 | 0.61 |
| ENC_m vs CCD_m | 0.97 | 0.95 |
| DAS vs DEC_m | 0.05 | 0.07 |
| DAS vs RAND0 | 0.05 | 0.08 |
| DAS vs VMRAND0 | 0.16 | 0.08 |

The Σ-whitened overlaps are in `logs/c_audit.json`. They are higher for the DAS pairs and leave the ordering unchanged.

**Reading.**
- **The DAS spans are only partly shared.** At k = 3, two DAS seeds share only part of their span (0.42 / 0.69). The C4 sweep shows why: the extra dimensions are arbitrary.
- **The successful sets share a common core.** DAS, ENC, CCD and DAS_DECOY overlap substantially (0.33–0.61), far above random, so they share a common signal-aligned core.
- **The decoder span is nearly orthogonal to that core** (0.05–0.07), as the geometry predicts.
- **The decoy-trained subspace shares about as much with DAS as DAS's own second seed** (C-C2: 0.61 vs 0.69). The relational channel is therefore not m-specific; it is the model's general cross-group relation direction at this site.

**Key/query split (answers XIII-D review 2, F9).** The split is a property of the model, not of the particular subspace that DAS happened to find:

| set | C-C1 keys / queries | C-C2 keys / queries |
|---|---|---|
| DAS | 0.89 / 0.02 | 0.19 / 0.73 |
| second DAS seed | 0.90 / 0.02 | 0.18 / 0.74 |
| ENC_m | 0.89 / 0.02 | 0.25 / 0.66 |
| DAS_DECOY | — | 0.17 / 0.73 |

Keys plus queries stay at 0.91–0.96 in every case. The split is stable across every sufficient set within a model and differs between models.

## C4. Dimension sweep, k = 1 to 6

DAS was fitted on the first half of the J_DM pairs and evaluated on the second.

| k | SUFF (C-C1) | SUFF (C-C2) | reverse remaining (C-C1 / C-C2) | unrelated donor (C-C1 / C-C2) | null JD0 rel (C-C1 / C-C2) | decoy-DAS SUFF (C-C1 / C-C2) | seed stability (C-C1 / C-C2) |
|---|---|---|---|---|---|---|---|
| 1 | 0.865 | 0.541 | 0.076 / 0.227 | 0.68 / 0.53 | 0.06 / 0.11 | 0.74 / 0.48 | 0.995 / 0.999 |
| 2 | 0.896 | 0.798 | 0.063 / 0.039 | 0.70 / 0.68 | 0.07 / 0.10 | 0.79 / 0.79 | 0.80 / 0.98 |
| 3 | 0.905 | 0.811 | 0.061 / 0.032 | 0.71 / 0.69 | 0.07 / 0.11 | 0.80 / 0.78 | 0.51 / 0.79 |
| 4 | 0.914 | 0.814 | 0.057 / 0.033 | 0.71 / 0.68 | 0.07 / 0.11 | 0.80 / 0.78 | 0.49 / 0.62 |
| 5 | 0.918 | 0.821 | 0.055 / 0.035 | 0.71 / 0.69 | 0.07 / 0.11 | 0.82 / 0.78 | 0.44 / 0.48 |
| 6 | 0.924 | 0.807 | 0.052 / 0.035 | 0.72 / 0.68 | 0.07 / 0.10 | 0.84 / 0.79 | 0.47 / 0.44 |

**R5 (frozen for C-C2): FALSIFIED.**
- The prediction transferred C-C1's pattern, "one direction carries ≥ 85% of the k = 3 effect and is seed-stable".
- In C-C2 one direction carries only 67%. Two directions are needed (0.80 against 0.81 at k = 3), and those two are seed-stable (0.98).

**Corrected statement.**
- The relational effect is carried by a small, model-specific number of stable directions: one in C-C1, two in C-C2.
- Adding dimensions beyond that gains ≤ 0.03 and adds directions that are not seed-stable.
- "Three-dimensional" was an experimental choice, not a finding. It is withdrawn as a property of the mechanism.

**Across k, the qualitative tests behave the same way:**
- reverse-swap necessity holds for every k ≥ 2;
- null preservation and the unrelated-donor value (near the ideal 0.72) are flat in k;
- the decoy-trained subspace performs about as well as DAS at every k. That non-specificity does not depend on dimension.

## C5. Reassessment of the XIII-D causal claims

| XIII-D claim | status after XIII-E | basis |
|---|---|---|
| P3 sufficiency of a low-dimensional relational channel | **survives** | SUFF 0.82–0.90 vs ideal 1 and null 0, replicated at every k ≥ 2 |
| P3 "3-dimensional carrier" | **contradicted / withdrawn** | C4: 1–2 stable directions; extra dimensions arbitrary (R5 falsified) |
| P3 "m-specific carrier" | **contradicted** | decoy-trained subspaces are equally sufficient at every k; overlap with DAS 0.33–0.61 |
| P3c "decodable ≠ used" | **retracted in that wording**; supported only in the weaker form "filter directions decode but do not transmit, as expected under correlated noise" | C2, R6 confirmed |
| P3e use through layer-2 keys + queries | **survives** | 0.91–0.96 for every sufficient set; K/Q split is a stable model property |
| P4a reverse-swap necessity | **survives** | ideal 0, null 1; observed 0.03–0.06 (k ≥ 2) |
| P4b "synthetic ablation is non-specific" | **non-diagnostic** | the ideal carrier also damages (0.35). Only a 0.10–0.22 excess is attributable |
| P3a DAS excess over source-free control | **non-diagnostic as a specificity test** | the ideal excess is only 0.28; observed values are at that level |
| restore "non-separability" | **retracted**; a modest shortfall against the ideal 0.46 remains | C1 |
| P6 eligibility specificity (falsified as scored) | **non-diagnostic**: the criterion is ill-posed | C1 own-share references; the DAS own-share vs ideal 1 / null 0 survives |
| E2 true row interchange | **survives** | R² 0.90–0.91 vs ideal 1 / null 0 |
| E2 decoy-row controls (base and borrowed) | **non-diagnostic** | ideal ≈ null ≈ 0 |
| null-pair preservation | **survives in a weak form**: no spurious transfer. It cannot separate m-carriers from decoy carriers | C1 |

**Still unresolved:**
- the restore shortfall;
- the 0.1–0.2 excess damage over the ideal;
- why the decoy-trained subspace transmits the m-effect. The likely reason is that the model's relational code is shared across relation types, but this was not tested.
