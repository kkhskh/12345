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
- **SRCF:** source-free interchange; the q-source tokens are not patched.
- **NEC:**
  - ablate: zero the projection;
  - meanrep: replace the projection by its mean;
  - reverse: replace the base projection by the matched null-partner's projection, a natural donor;
  - restore: re-insert the true projection after meanrep.

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

**The decoder–encoder dissociation is the central finding.** The ridge decoder directions of m and log r read the relation out well; the content section shows m R² ≈ 0.7–0.8 from the site. Yet interchanging along them moves behaviour by essentially nothing. Interchanging along the encoding (Haufe) directions moves it by 0.56–0.85 of the exact change.

So decodable is not the same as used: the model *writes* the relation along directions that differ from the best linear readout directions. This is replicated in two new models (P3c CONFIRMED).

**Two limits on specificity:**

- **DAS_DECOY is nearly as effective as DAS.** The subspace is the model's general relational channel at this site, not a channel specific to m.
- **The source-free interchange already carries 0.59–0.65 of the effect.** The readout token alone holds much of it. DAS beats SRCF by 0.24–0.32, so the q-source tokens add to it.

### Content

| | m linear / quadratic | log r linear / quadratic | candidate evidence linear / quadratic |
|---|---|---|---|
| dev | 0.759 / 0.789 | 0.731 / 0.875 | 0.443 / 0.717 |
| C-C1 | 0.764 / 0.846 | 0.800 / 0.936 | 0.653 / 0.866 |
| C-C2 | 0.698 / 0.775 | 0.679 / 0.829 | 0.453 / 0.704 |

**The projection carries m and log r.** On the full 192-d site, candidate evidence is decodable at R² ≈ 1. From the 3-d projection it is only partly decodable, and nonlinearly. So the subspace is a compressed relational code, not a copy of the candidate evidence.

### Necessity: natural donor against synthetic ablation

| | DAS reverse (natural donor) | RAND0 reverse | DAS ablate: remaining / damage KL | DAS meanrep: remaining / damage KL | restore after meanrep |
|---|---|---|---|---|---|
| dev | 0.057 | 0.929 | 0.124 / 0.35 | 0.104 / 0.54 | 0.319 |
| C-C1 | 0.059 | 0.967 | 0.063 / 0.57 | 0.084 / 0.50 | 0.351 |
| C-C2 | 0.038 | 0.790 | 0.013 / 0.51 | 0.012 / 0.45 | 0.239 |

**The natural-donor test, which stays on-distribution, shows the DAS subspace is necessary.** Replacing the base projection with the projection of a matched pair in which m does not change removes 94–96% of the J_DM response. The same operation on a random subspace removes almost nothing (P4a CONFIRMED).

**Synthetic ablation is not specific.** Zero-ablation and mean-replacement also remove the effect, but they damage unrelated behaviour (KL 0.35–0.57), so they cannot show necessity on their own (P4b CONFIRMED).

**Restoring after mean-replacement recovers only 0.24–0.35 of the slope.** The subspace's contribution is not separable from the rest of the residual stream once that stream has been moved off-distribution.

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

The invariant is the joint key–query score, not which side carries it. This was not predicted. It is a new observation and is reported as one.

## Preservation of the XIII-C LOCALQ gap

XIII-D does not re-derive the exact LOCALQ computation. It does preserve the 3.6-point gap reported in XIII-C:

- the XIII-C script and its output are available as a dependency (see `logs/DEPENDENCIES.json`);
- nothing in this workstream contradicts it.

The carrier result is consistent with that gap, but it does not test it. In this carrier, the relation that is actually used enters through the joint key–query score of the layer-2 mixture heads, which combines tokens.

## Verdicts (freeze 2)

- **P3 CONFIRMED.** All five arms pass in both C-C1 and C-C2.
- **P4 CONFIRMED.** Both arms pass in both models.

## Limits

- Two confirmation models, both early01-adapted at β = 0.6. The carrier in *naturally trained* two-colluder models is not tested; none exist in this population.
- The subspace is not m-specific (DAS_DECOY). The claim is "the model's relational channel at this site", not "an m-detector".
- The source-free control is strong. Much of the effect lives at the readout token.
- A training-summary line for C-C1's adaptation (competence only) was seen before freeze 2. It is recorded in COMPUTE_LOG at 18:44:55Z. No carrier measurement was seen.
