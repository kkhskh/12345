# R_counterexample: adversarial robustness review of the TF2 REL-XNOR mediator claim

Package commit a899a7c. Target claim: "in TF2 XNOR models, a 1–2 dim output-free subspace U at `resid_mid1@R` is a causal
mediator (IIA 1.0, necessity, sufficiency) whose non-additive content is ≈ sign(ab)".
All scripts are in `reviews/R_counterexample_scripts/` and use the package's `src/` unchanged (`common.py` holds the helpers).
U is always fitted exactly as the protocol does (800 pairs, seed 5001, 90% energy) unless stated. "nec" is the fraction
of pairs whose decisions still differ after pair-mean ablation (gate ≤ 0.2). Models: TF2 REL-XNOR s2, s3, s4 (historical) and
s7, s8, s9 (new). s3 is the package's source-additive case; the other five are the headline models.

## Findings

### F1 (MAJOR): at this site the mediator is essentially the readout direction, so "output-free" and "content ≈ sign(ab)" are close to tautological
Script `t_g_readout.py`, `t_g2_readout_content.py`, `t_a_fit_k.py`.

- **The site is functionally final.** Replacing `resid_post1` with `resid_mid1` (deleting the final MLP) leaves accuracy at
  0.999–1.000 in all six models (s2 0.9993, s3 0.999, s4 1.0, s7 0.9997, s8 0.9997, s9 1.0). The package's own path split
  already shows MLP-path IIA 0.00 and direct-path IIA 0.99–1.00. The protocol excludes final residual sites because they are
  "readout-trivial", but `resid_mid1@R` in TF2 behaves as one. This also contradicts MECHANISMS "Read by the MLP and the readout"
  for TF2 XNOR.
- **U ≈ readout direction.** Let w = centre(g_f ⊙ (W_U[:,0] − W_U[:,1])). It is defined from the weights alone, with no data
  and no relation labels. The largest principal cosine between U and w is 0.792 / 0.919 / 0.924 / 0.874 / 0.940 / 0.962
  for s2/s3/s4/s7/s8/s9.
- **w passes the protocol's own gates.** Interchanging w gives IIA 0.993–1.000 and sufficiency 0.993–1.000 in all six.
  Necessity: 0.068 (s4), 0.020 (s7), 0.000 (s9) pass; 0.29 (s2), 0.806 (s3), 0.504 (s8) fail.
  w's content test: additive R² 0.006–0.031; interaction R² by sign(ab) 0.998–1.000; the additive surrogate drops
  accuracy to 0.499 (s4), 0.569 (s7) and 0.514 (s9) (exact 0.998–0.999). Random 1-D subspaces give IIA ≤ 0.011.
  So, by the frozen verdict rules, a weight-only direction is a RELATION_COMPUTING_MEDIATOR in s4, s7 and s9.
- **Aggregate matching is not needed in XNOR.** An unmatched difference of means (agree vs conflict on 4000 training-distribution
  samples, no z-matching) has cosine 1.000 / 1.000 / 0.999 / 1.000 / 0.981 / 1.000 with U[:,0]. It gives the same IIA, nec and
  suff as U in s2, s4, s7 and s9 (e.g. s4: 1.0 / 0.102 / 1.0). This holds because in REL-XNOR s ≡ (y > 0): the "output-free"
  relation label *is* the target class label.
- **The content follows from accuracy.** The model's own margin on the 61×61 grid has additive R² 0.000 and interaction R² by
  sign(ab) 0.9999–1.0000 in every seed. Any coordinate that carries the logit at a functionally final site therefore inherits
  "non-additive content ≈ sign(ab)".
- **Scoping correction.** In TF2 REL-XNOR, U at `resid_mid1@R` is (to cosine 0.79–0.96) the logit direction of a model that
  computes sign(ab). Its causal properties and its sign(ab) content follow from the model being accurate and the final MLP
  being inert. They are not independent evidence of a relation representation. "Identified without output information"
  should be qualified: the generator label s equals the target label in this task. The claim that "the matching on the
  aggregate is what makes it identify relation content" is not supported for XNOR, where unmatched contrasts give the same
  subspace. It can only be tested in REL-GATE and REL-MIS, where s ≠ y. MAIN §4.1 (learnability control) concedes part of this
  but does not state that the site is readout-equivalent.

### F2 (MAJOR): the "gauss" replication (E4) does not test the causal gates on Gaussian pairs; under genuine Gaussian pairs s4 fails necessity
Script `t_bcd_shift.py`, `t_a2_s4.py`.

- **What E4 actually changes.** In `run_protocol.analyse`, `dist='gauss'` only changes `ev`, the samples used for accuracy and
  surrogate accuracy. The eval pairs are still drawn by `G.matched_pairs` (uniform z, uniform d) with seed 6102 instead of 6002.
  IIA, necessity and sufficiency under "gauss" are therefore a reseed, not a distribution shift. MAIN §1 says "5/6 under a new
  Gaussian evaluation distribution".
- **Genuine Gaussian matched pairs** were built as follows: agree member (a, b) ~ N(0, 1.2²)² with ab > 0; conflict partner with
  the same z and d ~ N(0, 2·1.2²) conditioned on |d| > |z|; 600 pairs. U kept fixed.
  - IIA 1.0 and suff 1.0 in all six models.
  - nec: s2 0.017, s3 0.095, **s4 0.305**, s7 0.012, s8 0.010, s9 0.000.
- **s4 also fails necessity on other reasonable pair ranges.**
  - |z| ∈ [2.4, 2.6]: 0.302.
  - Near-boundary pairs, with agree |d| ∈ [0.90, 0.98]|z| and conflict |d| ∈ |z| + [0.02, 0.10]: 0.317.
  - Under the protocol's own generator it passes stably: 20 eval seeds give nec 0.132–0.190, and 10 fit seeds give 0.134–0.158.
  - The fit-size sweep at n = 10 (seed 11) gives 0.24.
- **Scoping correction.**
  - Necessity holds robustly for s2, s7, s8 and s9 (≤ 0.08 in every range tested except F3).
  - s4 passes only under the protocol's uniform pair distribution. Under alternative pair distributions the count is 4/6
    relation-computing TF2 XNOR models, not 5/6.
  - Describe E4 as "accuracy and content surrogates under Gaussian inputs; causal gates re-run on a reseeded uniform pair
    sample".

### F3 (MINOR): necessity breaks for pairs within about 0.01 of the agreement boundary
- **Inputs.** Agree |d| = u|z| with u ∈ [0.99, 0.999]; conflict |d| = |z| + e with e ∈ [0.001, 0.01]; |z| ∈ [0.2, 2.5].
- **Results.**
  - nec: s2 0.318, s3 0.225, s4 0.160, s7 0.310, s8 0.075, s9 0.143.
  - IIA and suff: 0.980–1.000.
- **Interpretation.** Here the two members are near-identical inputs, and the pair mean sits on the model's decision boundary,
  so roughly coin-flip outcomes are expected. This is a property of the pair-mean ablation, not a counterexample to mediation.
- **Scoping.** State that necessity is measured on pairs at least 0.02 from the boundary, which is what the generator enforces.

### F4 (MINOR for the headline, notable for the controls): AGG-NL "0 of 500 matched pairs differ" is a pair-sampling artefact
Script `t_e_aggnl.py`, `t_e2_aggnl_protocol.py`, `t_e3_shift.py`.

- **Matched pairs with differing decisions (out of 2000), all 15 AGG-NL models.**
  - |z| ∈ [1.45, 1.55]: 66–344.
  - |z| ∈ [1.49, 1.51]: 378–1222.
  - |z| ∈ [1.499, 1.501]: 792–1546.
  - Even the protocol's own range |z| ∈ [0.2, 2.5] gives 3–14.
  - Inputs in [−5, 5] give 12–287.
  - The maximum within-pair margin difference in the [0.2, 2.5] range is 1.9–5.0 logits; the median |margin| is about 11.
- **The model's z-boundary depends on the spread d by up to 0.032.** For TF1 s1 it is 1.518 at d = 0 versus 1.486 at
  |d| = |z| + 1. Across models it is ±0.01–0.03, and the dependence is not monotone in d: for example TF1 s1 gives 1.518, 1.501,
  1.496, 1.486 and 1.497 for d = 0, 0.9z, z + 0.1, z + 1 and z + 2.
- **The frozen `analyse()` on near-boundary eval pairs.** With |z| ∈ [1.45, 1.55] and 500 pairs, and only the eval-pair
  generator replaced:
  - 9/10 TF1 and TF2 models pass the ≥ 20 differing-pairs gate, and all receive MEDIATOR_NOT_IDENTIFIED.
  - **TF1 s1** passes every S_rel gate (cut 1.0, IIA 1.000, nec 0.000, suff 1.000) and the content test (exact pair_flip 1.0 vs
    additive 0.0). It fails only because the mean S_rand IIA is 0.352 > 0.2. Agreement shifts its margin systematically
    (mA > mC on 82% of pairs; mean +0.96 logits).
  - **TF1 s3** is the same: IIA 0.951, nec 0.049, suff 0.951, interactive. It fails only on S_rand (0.44).
  - TF1 s2 and s4: IIA 0.98–0.99 and nec ≤ 0.013, but the additive surrogate does not lose pair_flip (non-interactive) or
    S_rand > 0.2.
  - Interaction R² by sign(ab) in AGG-NL TF2 is 0.46–0.63. The sign(ab) R² is therefore not specific to relation-trained models;
    it only separates 0.99 from about 0.6.
- **Scoping correction.**
  - The aggregate-only controls show no relation-sensitive behaviour *away from their decision boundary*. Within ±0.05 of
    |z| = 1.5 they show small spread-dependent boundary shifts (≤ 0.032 in z).
  - The protocol rejects them as relation mediators only through the random-subspace gate, because near-boundary pairs are easy
    to flip.
  - The "< 20 differing pairs" exemption is a property of the pair distribution, not of the model.

### F5 (MINOR): REL-GATE source-additive verdict flips for TF2 s8 on near-boundary pairs
- **TF2 s8.** With agree |d| ∈ [0.90, 0.98]|z| and conflict |d| ∈ |z| + [0.02, 0.10], with p random ±1:
  - The additive surrogate loses 0.117 pair_flip relative to exact, which is ≥ 0.10 and so "interactive".
  - The causal gates pass (IIA 1.0, nec 0.07, suff 1.0).
  - The verdict would be RELATION_COMPUTING on that subset.
- **Elsewhere the verdict is robust** (see Survived, item 6).

### F6 (MINOR): forcing k
Forced k uses the protocol SVD basis of the 800 fit pairs. Results are IIA / nec / suff / S_rand mean / S_rand max.

- **k = 1 fails necessity** for s8 (1.0 / 0.468 / 1.0) and s3 (1.0 / 0.852 / 1.0). k = 2 is needed there, as the protocol chose.
- **k = 2, 3 and 4** keep IIA 1.0, suff 1.0 and nec ≤ 0.094 in all six.
- **The random baseline weakens as k grows.** At k = 4 the S_rand max reaches 0.42–0.76 and the mean 0.02–0.19; s9 k = 4 has a
  mean of 0.187, close to the 0.2 gate. The random-subspace control loses power for k ≥ 4 in d = 16.

## No implementation bug found that makes IIA trivially 1 (check g)
- **Formulas.** U from `s_rel` is orthonormal (rows of Vt), with k ≤ 2 of D = 16. The interchange
  `hb + (hs − hb)UUᵀ`, the complement patch `hs + (hb − hs)UUᵀ` and the pair-mean ablation are algebraically correct.
- **Controls behave as they should.** Random 1-D subspaces give IIA 0.000–0.011 and S_alt gives 0.000, so U does not span the
  full state difference. Full-state IIA = 1 holds by construction.
- **IIA is restricted to pairs where the model's decisions differ** (500/500 here).
- **The near-triviality is functional, not a bug** (F1): the site is readout-equivalent and the target is the relation.
- **S_grad's IIA of 0.000** at this site is consistent with final-LN scale invariance: the margin gradient is orthogonal to the
  state's own direction. It should not be read as "output-derived subspaces fail" without that caveat.

## Claims that survived (with search ranges)

1. **U is stable to fit-set size and seed.**
   - Search: n ∈ {10, 25, 100, 800, 3200} fit pairs, 3 pair seeds each.
   - k was never changed; cosine to the protocol U ≥ 0.995; IIA 1.000 and suff 1.000 in every cell.
   - nec ≤ 0.158, except s4 at n = 10 (0.24).
2. **Interchange IIA and sufficiency are robust for all six models (≥ 0.98)** across every range tested:
   - |z| ∈ [0.15, 0.25], [2.4, 2.6] and [0.01, 0.05];
   - near-boundary and extreme-boundary pairs;
   - Gaussian matched pairs;
   - extrapolated pairs in [−5, 5]², including |z| ∈ [3, 4.8] and pairs with some |coordinate| > 3 (n ≈ 2250).
3. **Extrapolation.** Accuracy is 1.000 on [−5, 5]² \ [−3, 3]² and on [−8, 8]² \ [−5, 5]². On pairs in [−5, 5]², IIA is
   1.000 and nec ≤ 0.08 (s3 0.236–0.253; s3 is not a headline model).
4. **Generator transfer.** U fitted on Gaussian matched pairs has cosine 1.000 with the train-generator U, with the same k, and
   transfers to the protocol eval pairs (IIA 1.0, suff 1.0, nec ≤ 0.124).
5. **Necessity** holds for s2, s7, s8 and s9 under every pair distribution tried (≤ 0.08), except extreme-boundary pairs (F3).
6. **REL-GATE source-additive verdict** (TF1 and TF2, s2–4 and s7–9).
   - Search: grids 61, 121 and 241 on [−3, 3], and 121 on [−2, 2]; p = −1 and p = +1 evaluated separately; |z| ≈ 0.2 pairs.
   - (exact − additive) is ≤ 0.011 in accuracy and ≤ 0.021 in pair_flip, far below the 0.05 / 0.10 thresholds.
   - The only exception is F5.
   - TF2 additive R² is only 0.80–0.93, but the non-additive part does not affect decisions.

## Net assessment
The causal facts survive (interchange, sufficiency, and necessity for four of the five headline models). The interpretive
claim needs narrowing: in TF2 REL-XNOR the "relation mediator" is the logit direction at a readout-equivalent site, and
its sign(ab) content is implied by task accuracy (F1). The replication count should be 4–5/6 depending on the pair distribution,
and E4 is not a causal distribution shift (F2). The aggregate-only control negative holds only away from the |z| = 1.5
boundary (F4).
