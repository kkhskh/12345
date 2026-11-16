# Natural relational carrier and necessity (Workstream C)

COR Phase 1.5, Step 1, Investigation XIII-D. This report describes how a few trained transformers work. It does not validate COR and makes no claim about the O/D operators.

## Question

XIII-C found a 3-dimensional DAS subspace that carries the response to the two-colluder joint relation (J_DM pairs) in an early01-adapted model. Two things were still open:

- whether that subspace is what the model actually uses, as opposed to a direction an optimiser can push the readout through;
- whether the effect needs the subspace at all.

XIII-D tests five things:

- **encoding versus decoding:** the directions that *write* the relation compared with the directions from which it can be *read*;
- **content:** what the subspace represents;
- **necessity:** tested with natural donors rather than synthetic ablation;
- **null preservation:** whether the interchange leaves null pairs alone;
- **trace:** the downstream path.

## Setting and site

**Adapted models.** Each model is early01-adapted with adapt4.py: two-colluder generator, β = 0.6, 1000 steps, lr 1e-3, layers 0–1 trainable.

**Matched site.** The q-source tokens and the readout token, read in one shared 64×3 basis. The same site is used for every direction set, so the sets differ only in direction, never in site.

**Pairs:**

- 1200 J_DM pairs: the joint relation m changes, the marginals stay fixed.
- 300 J_D0 pairs: m is unchanged, so Bayes is unchanged.
- 300 DECOY pairs: a non-diagnostic relation changes.

**Scripts and outputs.** `scripts/c_carrier.py`. Development output `logs/c_carrier_dev.json` (uP161 s21). Confirmation output `logs/c_carrier_f2.json`:

- C-C1 = uP160 s23;
- C-C2 = uP165 s21.

Both confirmation models are new early01 adaptations, and their carrier analysis was frozen before it ran (freeze 2, commit `971f07f`).

**Direction sets (all k = 3):**

- **DAS:** trained interchange subspace.
- **DEC_m / DEC_r:** ridge decoders of m and log r.
- **ENC_m / ENC_r:** Haufe encoding patterns, i.e. the covariance-transformed decoders.
- **CCD:** class-conditional difference.
- **RAND0–4:** random subspaces.
- **DAS_DECOY:** DAS trained on decoy pairs.

**Tests:**

- **SUFF:** interchange on J_DM pairs. Slope and R² of the observed versus the exact Bayes change.
- **NULL:** interchange on J_D0 / DECOY pairs, as a fraction of the J_DM patch response.
- **SRCF:** the same subspace and site, but with the donor taken from an *unrelated* episode with the same q. It is an unrelated-donor control, not a token-site ablation (corrected after review 1, F1). Its value for an ideal m-carrier is 0.72 (`reviews/review1_scripts/ideal_carrier.py`).
- **NEC:**
  - ablate: zero the projection;
  - meanrep: replace the projection by its mean;
  - reverse: the donor run is evaluated with the same pair's *base* projection. The ideal value is 0 (corrected description, review 1, F19). Remaining slopes are divided by 1; the unpatched slope is ≈ 0.99 (C-C1) and ≈ 0.85 (C-C2).
  - restore: re-insert the donor's projection after mean-replacement. The donor run is left unmodified, so the ideal value is 0.46, not 1 (review 1, F3).

  For ablate and meanrep, the KL damage on unrelated episodes is also reported.
- **CONTENT:** decoding of m, log r and candidate evidence from the 3-d projection.
- **TRACE:** decomposition of the induced layer-2 attention-score change into keys, queries and values.

## Results

### Sufficiency, nulls, encoding versus decoding

| set | dev uP161 s21 SUFF slope / R² | C-C1 uP160 s23 | C-C2 uP165 s21 | null D0 / DEC (C-C1; C-C2) |
|---|---|---|---|---|
| DAS | 0.850 / 0.82 | 0.903 / 0.87 | 0.821 / 0.67 | 0.056 / 0.026; 0.090 / 0.057 |
| DAS_DECOY | 0.727 / 0.73 | 0.783 / 0.80 | 0.761 / 0.65 | 0.058 / 0.027; 0.085 / 0.058 |
| ENC_m (encoding) | 0.556 / 0.69 | 0.846 / 0.83 | 0.764 / 0.65 | 0.062 / 0.028; 0.103 / 0.061 |
| ENC_r | 0.471 / 0.60 | 0.806 / 0.80 | 0.752 / 0.65 | |
| DEC_m (decoder) | 0.003 / 0.01 | −0.001 / 0.00 | 0.033 / 0.08 | |
| DEC_r | 0.002 / 0.00 | 0.001 / 0.00 | 0.032 / 0.08 | |
| RAND0–4 | ≤ 0.02 | ≤ 0.043 | ≤ 0.030 | |
| DAS source-free (SRCF) | 0.652 | 0.587 | 0.586 | |

**Encoding patterns versus decoder filters (revised after review 2, F7).**

- Interchange along the ridge **decoder filters** of m and log r moves behaviour by essentially nothing.
- Interchange along the **encoding (Haufe) patterns**, or along the class-mean differences (CCD, which are nearly identical), moves it by 0.56–0.85 of the exact change.
- This is replicated in two new models, so the frozen P3c arm passes.

**This is not evidence that the model "writes the relation along directions other than its readable directions."** That would be a property of the model; the result is expected for *any* linear code with correlated noise:

- The decoder filters are fitted near-OLS on raw 64-d token states, so they are dominated by noise-cancelling directions.
- The network is *less* sensitive to them than to random directions: ablation damage KL is 0.000–0.001, against 0.005–0.019 for random.
- An interchange along a filter moves the state only by the donor–base difference projected onto that filter, which is small when the signal is written along the pattern.

The decoders' own held-out decoding accuracy was never measured; the content R² below is from the DAS projection.

What the arm does establish is narrower. The relation is transmitted along signal (pattern) directions, and the filter directions carry no causal effect.

**Two limits on specificity:**

- **DAS_DECOY is nearly as effective as DAS.** The subspace is the model's general relational channel at this site, not a channel specific to m.
- **The carrier is not unique (review 2, F8).** Several differently built 3-d sets each pass the necessity test about equally. Reverse natural-donor remaining slope, C-C1 / C-C2:

  | set | C-C1 | C-C2 |
  |---|---|---|
  | DAS | 0.059 | 0.038 |
  | ENC_m | 0.072 | 0.067 |
  | CCD_m | 0.072 | 0.069 |
  | ENC_r | 0.087 | 0.069 |
  | DAS_DECOY | 0.090 | 0.081 |

  The supported claim is therefore: "a low-dimensional relational channel, shared by DAS, the Haufe patterns and DAS_DECOY, is sufficient and, by natural-donor reversal, necessary at this site". Principal angles between the sets, and necessity tests on their intersection or orthogonal complements, were not computed.
- **The unrelated-donor control gives 0.59–0.65.** That is about what an ideal carrier gives (0.72), because the control shares the −B(X_a) term with the target. It is therefore not evidence that the readout token holds the effect. DAS − SRCF = 0.24–0.32, against an ideal 0.28 (review 1, F1). The site was not split into readout-only and source-only parts.

### Content

| | m linear / quadratic | log r linear / quadratic | candidate evidence linear / quadratic |
|---|---|---|---|
| dev | 0.759 / 0.789 | 0.731 / 0.875 | 0.443 / 0.717 |
| C-C1 | 0.764 / 0.846 | 0.800 / 0.936 | 0.653 / 0.866 |
| C-C2 | 0.698 / 0.775 | 0.679 / 0.829 | 0.453 / 0.704 |

**The projection carries m and log r.** On the full site (448 features), candidate evidence is decodable at R² ≈ 1. That is trivial, since the site includes the source tokens themselves. From the 3-d projection it is only partly decodable, and nonlinearly. So the subspace is a compressed relational code, not a copy of the candidate evidence.

### Necessity: natural donor against synthetic ablation

| | DAS reverse (natural donor) | RAND0 reverse | DAS ablate: remaining / damage KL | DAS meanrep: remaining / damage KL | restore after meanrep |
|---|---|---|---|---|---|
| dev | 0.057 | 0.929 | 0.124 / 0.35 | 0.104 / 0.54 | 0.319 |
| C-C1 | 0.059 | 0.967 | 0.063 / 0.57 | 0.084 / 0.50 | 0.351 |
| C-C2 | 0.038 | 0.790 | 0.013 / 0.51 | 0.012 / 0.45 | 0.239 |

**The reverse test shows the relational channel is necessary.** It is an on-distribution swap: the donor run is evaluated with the base pair's projection. The DAS subspace and the other high-sufficiency sets (above) all pass it. Replacing the base projection with the projection of a matched pair in which m does not change removes 94–96% of the J_DM response. The same operation on a random subspace removes almost nothing (P4a CONFIRMED).

**Synthetic ablation (P4b, corrected after review 1, F2).** Zero-ablation and mean-replacement remove the effect and change behaviour on natural episodes by KL 0.35–0.57. Those episodes' decisions depend on m, so an ideal carrier would also change them, by KL 0.347. Only the excess, ≈ 0.0–0.2, is a candidate off-manifold artefact. **The frozen P4b arm passes, but it is non-informative:** the ideal carrier also meets its bound.

**Restore after mean-replacement: 0.24–0.35, against an ideal 0.46.** The shortfall is real but modest. The earlier claim of non-separability is withdrawn (review 1, F3).

### Downstream trace (layer-2 mixture heads)

| | keys only | queries only | values only | keys + queries | skip only | corr(Δscore, Δlog r) |
|---|---|---|---|---|---|---|
| dev | 0.364 | 0.303 | 0.049 | 0.940 | −0.014 | 0.89, 0.89 |
| C-C1 | 0.886 | 0.021 | 0.030 | 0.964 | −0.019 | 0.93, 0.93 |
| C-C2 | 0.219 | 0.764 | 0.001 | 0.938 | −0.027 | 0.82, 0.83 |

**The carried relation acts on behaviour by changing layer-2 attention scores, through keys and queries jointly (0.94–0.96).** It does not act through values (≤ 0.05) or through the residual skip.

The induced score change tracks the change in the exact log likelihood ratio (corr 0.82–0.93) (P3e CONFIRMED).

**How the work splits between keys and queries is model-specific:**

- keys dominate in C-C1;
- queries dominate in C-C2;
- the two are balanced in development.

The invariant is the joint key–query score, not which side carries it. This was not predicted and is reported as a new observation.

Caveat (review 2, F9): the trace was run only for the DAS subspace, which is trained from one random initialisation. The key/query split may therefore belong to the particular sufficient subspace DAS found, rather than to the model. Only the joint share (0.94–0.96) is claimed as robust.

## Preservation of the XIII-C LOCALQ gap

XIII-D does not re-derive the exact LOCALQ computation. It does preserve the 3.6-point gap reported in XIII-C:

- the XIII-C script and its output are available as a dependency (see `logs/DEPENDENCIES.json`);
- nothing in this workstream contradicts it.

The carrier result is consistent with that gap, but it does not test it. In this carrier, the relation that is actually used enters through the joint key–query score of the layer-2 mixture heads, which combines tokens.

## Verdicts (freeze 2)

- **P3 CONFIRMED.** All five arms pass in both C-C1 and C-C2. Arm (c) is to be read as "pattern ≠ filter", not as "decodable ≠ used" (see above).
- **P4 CONFIRMED.** Both arms pass in both models.

## Limits

- Two confirmation models, both early01-adapted at β = 0.6. The carrier in *naturally trained* two-colluder models is not tested; none exist in this population.
- The subspace is not m-specific (DAS_DECOY). The claim is "the model's relational channel at this site", not "an m-detector".
- k = 3 was fixed by design and never varied, so "3-dimensional" is a design choice, not a finding (review 1, F6). The posterior content row is NaN because of a code defect (a double log). The site has 448 features (7 tokens × 64), not 192 (review 1, F17).
- A training-summary line for C-C1's adaptation (competence only) was seen before freeze 2. It is recorded in COMPUTE_LOG at 18:44:55Z. No carrier measurement was seen.
