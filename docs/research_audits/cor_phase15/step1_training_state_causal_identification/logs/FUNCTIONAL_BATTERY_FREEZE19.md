# Stage-6 functional battery: frozen rule (addendum, written before any confirmation output was read)

**When this was written.** The confirmation was running and only COMPUTE_LOG and file existence had been looked at. No confirmation checkpoint or measurement had been opened.

**Battery.** `scripts/func_battery19.py`, stream 93 300 000 for the confirmation models; it is new and was used for nothing else.
- Route-free. It is computed only from input–output behaviour on natural T4 episodes, binned by their latent structure at the target question.
- Statistic: split ratio SR = KL(Bayes‖model | reliable pair disagrees at q) / KL(Bayes‖model | reliable pair agrees and colluders agree with it).

**Calibration [dev].** On 12 XVIII reference models (stream 93 100 000):
- EARLY-organised models: 2.60–3.78;
- MIXTURE-organised models: 0.98–1.27.

**Held-out check [dev].** 52 models on stream 93 200 000 (XIV, other parents, gsg hybrids, D801 anchors):
- competent pure EARLY: 2.55–4.02 (n = 16);
- competent pure MIXTURE: 1.03–1.33, with one exception of 1.74 (uP163 L23, another parent);
- hybrids: 1.06–1.91.
- There are no crossings.

**Frozen classification.**
- FE (early-like error structure) if SR ≥ 2.2;
- FM (mixture-like) if SR ≤ 1.6;
- otherwise FAMBIG.

**Interpretation boundary (fixed in advance).** Simple degradations of exact Bayes give:
- uniform lapse: SR 0.16–0.21;
- temperature: SR 0.36–0.53;
- isotropic logit noise: SR 2.9–3.2.

So SR separates the two organisations by **where their errors fall** (low-margin, split-pair cases for EARLY). It does **not** identify a distinct evidence-combination operator: the EARLY profile is consistent with unstructured logit noise around Bayes. A matching SR class is therefore at most evidence for Claim 2 in the weak sense: a different input-dependent approximation to the same Bayes target. It is not evidence for Claim 3.

**Frozen predictions (Stage 6).** These are scored after the frozen score19 result, for every competent confirmation model of the arms below.

| id | prediction |
|---|---|
| F1 | Anchors: on each base, the fullB and fullI SR classes equal their causal classes (FE ↔ EARLY, FM ↔ MIXTURE) |
| F2 | Causal-switch arms (category BOTH or CAUSAL_ONLY or INCONCLUSIVE(causal_SWITCH) in score19): the SR class follows the new causal class |
| F3 | Causal-none arms (category NEITHER or FUNCTIONAL_ONLY): the SR class stays with the B anchor's class |
| F4 | HYBRID arms are reported descriptively (no prediction) |

**Scoring.**
- F2 and F3 are SUPPORTED per base if every eligible model matches, and CONTRADICTED if a strict majority have the opposite class. FAMBIG counts as neither.
- Reported with counts. Eliminates nothing about mechanism.
