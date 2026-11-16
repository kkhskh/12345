# Investigation XIII-D: causal algorithm selection and validation — main research synthesis

**Scope.** COR Phase 1.5, Step 1 (about 25% complete). This investigation closes no phase and proposes no thesis. It does not claim that COR is validated, and it makes no claim about the O/D operators. The three reviews in `reviews/` are **agent reviews, not external human peer review**.

**Not standalone.** The package depends on earlier research packages for its pretrained parents, natural fine-tunes, XIII-C runs and modules (`logs/DEPENDENCIES.json`, REPRODUCTION.md).

## Question and setting

> Why does a particular relational decision algorithm emerge during training, what information does it actually use, and under what conditions can that algorithm be changed or reused?

**Models and task.** Small transformers are fine-tuned from six pretrained parents (uP160–uP165) on a latent-reliability generator. Layer 2 and the readout map are freshly initialised from seeds L and W; D sets the data order.

**Two organisations emerge:**
- **mixture:** layer-2 heads gate between per-group answers;
- **early:** the decision is already read out at layer 1.

**What XIII-C claimed.** XIII-C's positive result was that the organisation is set by the readout-map initialisation, through an early, path-dependent race. XIII-D tested that claim directly.

## Principal findings, after adjudication of the reviews

### 1. XIII-C's readout-map cause is falsified, and no simple replacement law was found

Supporting predictions: P1 and P10, both FALSIFIED.

- **W-only cells (P1).** Four W-only cells in two parents all ended in the mixture organisation. Changing W never turned a mixture configuration into early.
- **uP164 (P10).** A W-only change from the early diagonal stayed early, while an L-only change did not.
- **uP163.** W22 gives early on L21 and on L22, but W20 does not.
- **Data order.** D does not change the class in any tested pair: 10 pairs at step 4000 and 1 at step 1211.

**What the evidence supports.** The organisation is a strongly interaction-dependent function of the parent and the fresh (L, W) draw. Base rates show why no simpler law follows:
- natural diagonals: 7 early, 6 mixture, 4 hybrid;
- off-diagonal cells are early 3 of 11 times, against 7 of 17 for diagonals (Fisher p = 0.69).

**Withdrawn.** A first-draft reading, "mixture is the default; early needs the diagonal's W and L", is withdrawn (review 2, F1).

**Development observations (one parent, uP161).**
- A sign flip or column permutation of an early-producing W_A gives mixture, although both preserve its spectrum.
- No step-0 property of W_A alone predicts the outcome across parents.

### 2. Route availability at the very start redirects the organisation; the unused route's information is retained in pooled form

Supporting predictions: P2b FALSIFIED, P2c CONFIRMED, P2a FALSIFIED.

**Early masking flips the organisation permanently.** Blocking either route for only the first 100 of 4000 steps (about 1.2% of the learning-rate mass) permanently gave the other organisation, in both directions (n = 1 each). P2b had predicted recovery.

Two things are untested:
- shorter windows;
- a sham mask.

So the result is "sensitivity to route availability in the first 100 steps", not "a decision made by step 100".

**Late blocks (steps 546–1146) differ in severity.**
- Blocking the early route drives the loss to chance, and the model relearns as mixture.
- Blocking the mixture route is mild, and leaves a stable dual-route hybrid.
- The apparent asymmetry is therefore confounded with severity.

**The unused early route's pooled evidence stays decodable at the readout token** (0.93 at the end vs 0.93 at its peak) while its use falls to zero (P2c, one diagnostic arm). Group-resolved presence is reduced: 0.78 against 0.99 in early-organised siblings.

**Readout-only retraining recovers about half of the gap** (+0.26 / +0.27). After a 250-step block it does not.

**P2a.** The frozen commitment signature was falsified:
- F1a and F1e miss one cap within noise;
- F1b keeps the information (0.74) but is not readout-reactivatable.

### 3. A low-dimensional relational channel is sufficient and necessary for the repaired two-colluder relation, and acts through layer-2 keys and queries jointly

Supporting predictions: P3 and P4, CONFIRMED as scored. Several arms were reinterpreted after review 1.

The tests used two new early01-adapted models.

**Sufficiency and null preservation.**
- DAS interchange slope: 0.90 / 0.82. Random subspaces: ≤ 0.04.
- Null pairs are preserved (≤ 0.09).
- The unrelated-donor control gives 0.59, close to what an ideal carrier gives (0.72). The DAS excess over it (0.24–0.32) matches the ideal excess (0.28).

**Necessity.** The on-distribution reverse swap leaves 0.04–0.06 of the effect; random subspaces leave 0.79–0.97. The channel is not unique: Haufe patterns, class-mean directions and a decoy-trained DAS are about equally sufficient and necessary.

**Content and route.**
- The channel carries m and log r (R² 0.70–0.80).
- It acts through the joint layer-2 key–query score (0.94–0.96), not through values.
- The score change tracks Δ log r (corr 0.82–0.93).

**Reinterpreted.**
- "Decodable ≠ used" is reread as "signal patterns transmit, decoder filters do not". That is expected under correlated noise and is not a property of the model.
- The synthetic-ablation arm (P4b) passes but is non-informative: an ideal carrier causes the same damage.
- "3-d" is a design choice.
- The key/query split was traced only for DAS.

### 4. Supplying the missing statistic enables a repair that otherwise fails; layer-0 attention can compute it

Supporting predictions: P5b and P5c CONFIRMED; P5a FALSIFIED on its budget clause.

**Model:** uP165 s20.

**At lr 3e-3:**

| arm | slope |
|---|---|
| m supplied, layer 2 + readout trainable | 0.906 |
| same parameters, no m | 0.000 |
| readout-token supply | 0.894 |
| source-token supply | 0.727 |

The pathway by which supplied m acts was not traced.

**At lr 1e-3:**

| arm | slope |
|---|---|
| layer 0 alone | 0.832 |
| layer-0 attention alone | 0.818 |
| layer-0 MLP alone | 0.008 |
| layer 1 alone | 0.004 |

A post-review exploratory arm ran layer 1 alone at lr 3e-3; it gave 0.021.

**Why P5a failed.** Supply at lr 1e-3 gave 0.168, not < 0.1. Supply is partly usable at the weaker budget.

**Why layer 0 works: two hypotheses, not discriminated.**
- cross-token access;
- downstream depth.

### 5. Causal-control audit, after review

- **E1 (P6 FALSIFIED as scored).**
  - At sites that exclude the edited tokens, an 8-d subspace transmits the model's own eligibility response. Own-share is 0.53–0.89, and the rule margins mirror the model's behaviour.
  - The specificity criterion that failed is ill-posed: own_share is unnormalised, and an ideal carrier scores 0.40–0.50 on the control.
  - So the falsification says little about specificity.
- **E2.**
  - True attention-row interchange explains 0.90–0.96 of the change.
  - Decoy rows scored against the base explain ≤ 0.014.
  - NOOP is within 1e-5.
  - The borrowed-pair rescoring has no power: an ideal carrier scores 0.003.
- **E3 (P8 CONFIRMED as scored).**
  - Path patching is near-additive.
  - No single route suffices under blocking. This establishes non-redundancy only; the blocking is off-distribution.
- **E4 (P7 CONFIRMED as scored).**
  - Symmetrising the readout rows (SYM1) did not reduce the Bayes-invariant error, at a KL cost of 0.02–0.10.
  - SYM2 reduced it by 5–7%.
  - The test cannot exclude the positional hypothesis, so the error source remains unidentified.
- **E5.** The XIII-C identifiability results are unchanged. Review 1 independently re-verified Proposition S (4.9e-15) and the LOCALQ gap (3.51 ± 0.09 vs 3.6 ± 0.07).

### 6. Reproducibility

Three independent full-length (4000-step) training replays are **bitwise identical** to their originals at every common checkpoint: 24, 12 and 12 checkpoints. The following are also identical:
- the loss history;
- final accuracy and KL;
- the mechanistic classification.

This was tested for the training script only.

### 7. Learning-transition evidence (B5, exploratory, not frozen)

**Setup.** n = 33 runs (12 early), using measures taken at step 246 to predict the final class.

**Result.**
- mix_mass AUC: 0.86. Behavioural baselines:
  - accuracy: 0.71;
  - KL: 0.69;
  - loss: 0.64.
- The advantage over accuracy is +0.15, with a bootstrap 95% interval of [−0.07, +0.37]. Over lik_RO1 it is +0.18 [+0.04, +0.36].
- At step 111 no measure exceeds AUC 0.78.

**Interpretation.** A mechanistic measure separates the classes earlier than loss, but its advantage over behavioural baselines is not established at this n.

## Frozen predictions

**13 predictions in 5 freezes.** Each freeze was committed before the confirmation it governs.
- **Confirmed (7):** P2c, P3, P4, P5b, P5c, P7, P8.
- **Falsified (6):** P1, P2a, P2b, P5a, P6, P10.
- **Non-diagnostic:** 0.

**Reinterpreted after review (verdicts unchanged):**
- P4b is non-informative.
- P7 cannot exclude H_pos.
- P8 establishes non-redundancy only.
- P6 falsifies an ill-posed criterion.

**Threshold sensitivity (±20%).** Two verdicts flip:
- P6 is confirmed at an excess threshold of 0.12;
- P4 is falsified at a random-retention threshold of 0.875.

Details are in FROZEN_CONFIRMATION_AND_STATISTICS.md.

## Answers to the central question (within this setting)

**Why a particular algorithm emerges.**
- It depends on how the freshly initialised layer-2/readout pair interacts with the pretrained parent, not on data order.
- It is sensitive to route availability in the first ~1% of the learning-rate schedule.
- Both routes are learnable, and both kinds of information are computed.
- Beyond "interaction-dependent", no law for which configuration wins was found.

**What information it uses.**
- Early-route evidence is computed and retained (pooled) whether or not it is used.
- In repaired models, a low-dimensional relational channel carrying m and log r is necessary and sufficient. It enters the decision through layer-2 attention scores.

**When it can be changed or reused.** It can be redirected by:
- blocking a route during the first 100 steps;
- partially, by retraining only the readout, after a short block;
- supplying the missing statistic (lr 3e-3);
- retraining layer 0 or its attention (lr 1e-3).

It was not repaired by:
- retraining layer 2 + readout without supply (lr 3e-3);
- retraining layer 1 at lr 1e-3, or at lr 3e-3 (post-review exploratory arm).

## Required final scientific answers (mandate §15)

Evidence labels:

| label | meaning |
|---|---|
| **[proof]** | mathematical proof |
| **[num]** | independent numerical verification |
| **[dev]** | development evidence |
| **[frozen]** | frozen confirmation |
| **[expl]** | exploratory |
| **[spec]** | speculation |

1. **What property of the readout-map initialisation influences the learned implementation?**
   - No W-only property was identified.
   - Changing W alone did not move the organisation as predicted (P1 FALSIFIED [frozen]). In uP164 W was irrelevant (P10 FALSIFIED [frozen]).
   - W's orientation relative to layer 2 matters in uP161: a sign flip or column permutation gives mixture; scaling changes nothing [dev].
   - No step-0 W_A statistic predicts the outcome across parents [dev].

2. **Does the effect replicate under independent manipulation of W, L and D?**
   - No. The outcome depends on parent × L × W interactions, and D never changes the class [frozen + dev].
   - No low-order law was found (base-rate Fisher p = 0.69) [expl].

3. **Active competition, or commitment of the downstream computation?**
   - The unused route's information is retained in pooled form while unused [frozen, P2c, n = 1], and it is partially reactivatable by readout-only retraining [frozen data].
   - The frozen commitment signature (P2a) was FALSIFIED.
   - The locus of lock-in is not isolated. "Commitment" is a compatible reading, not an established one [spec].

4. **Is there a critical period?**
   - Blocking a route for 100 steps permanently redirects the organisation (P2b FALSIFIED, since recovery had been predicted [frozen]).
   - There is no recovery window of ≥ 100 steps. Shorter windows and a sham mask were not tested.
   - A late block on the early run also redirects it, but that block is confounded with severity [frozen data, review].

5. **Does the unused pathway lose its information?**
   - Pooled information is not lost [frozen, P2c].
   - Group-resolved information is reduced, and more so after longer blocks (0.78 / 0.55 vs 0.99) [frozen data].

6. **What does the repair carrier encode?**
   - m and log r (R² 0.70–0.80) [frozen].
   - Candidate evidence, only nonlinearly [frozen].
   - The 3-dimensionality is a design choice.

7. **Necessary, sufficient, or one of several?**
   - Sufficient and necessary under the reverse swap [frozen, P3a / P4a].
   - Not unique: Haufe patterns, class means and a decoy-trained DAS are equally sufficient and necessary. It is a shared low-dimensional channel [frozen data, review 2].

8. **Do encoding-pattern controls remove the subspace-versus-decoder discrepancy?**
   - Yes. Encoding patterns transmit (0.76–0.85) and decoder filters do not (≈ 0) [frozen].
   - This is the expected filter-versus-pattern geometry, not a model property [num: review 1 ideal-carrier checks].

9. **Does the carrier preserve decisions when the joint statistic is unchanged?**
   - Yes: J_D0 and DECOY responses are ≤ 0.09 of the patch response [frozen, P3b].
   - The metric is lax and was not compared with the model's own null error [review 1].

10. **How does the frozen final layer use the repaired representation?**
    - Through the joint layer-2 key–query attention score (0.94–0.96), not through values. The score change tracks Δ log r (corr 0.82–0.93) [frozen, P3e].
    - The key/query split was traced only for DAS.

11. **Why does early-layer repair succeed when final-layer repair fails?**
    - Layer 2 + readout cannot compute m from its input (0.000 at lr 3e-3), but succeeds when m is supplied (0.906) [frozen].
    - Layer 0, and its attention alone, can compute m. Layer 1 cannot (0.004 at lr 1e-3 [frozen]; 0.021 at lr 3e-3 [expl]).
    - Cross-token access versus depth is not discriminated [spec].

12. **Accessibility, optimisation speed, position, or capacity?**
    - **Accessibility:** demonstrated (supply versus no supply at matched parameters) [frozen].
    - **Optimisation speed:** matters for supply (0.168 at lr 1e-3 vs 0.906 at lr 3e-3; P5a FALSIFIED on the budget clause) [frozen].
    - **Position:** readout-token and source-token supply both work (0.89 / 0.73) [frozen].
    - **Capacity:** capacity-matched L0 versus L1 differ, so capacity alone is not the explanation [frozen].

13. **Does changing internal organisation change the ability to use new relational information?**
    - In XIII-C data, layer-2 repair reaches 0.38 in one early model, against 0.00 in mixture models [frozen in XIII-C; re-read here].
    - Not newly tested in XIII-D.

14. **Does eligibility mediation survive excluding the edited tokens?**
    - An 8-d subspace at edited-token-free sites transmits the model's own eligibility response (own-share 0.53–0.89) [frozen].
    - The specificity criterion was ill-posed (P6 FALSIFIED as scored) [review 1].

15. **Are attention-row effects specific under matched controls?**
    - True row interchange: R² 0.90–0.96. Decoy rows scored against the base: ≤ 0.014. NOOP ≤ 1e-6 [frozen].
    - The borrowed-pair rescoring has no power [num: review 1].

16. **Independent, redundant, or components of one mechanism?**
    - Not redundant. Additive under path patching, and non-redundant under blocking [frozen, P8].
    - "Components of one mechanism" is not established, because the blocking is off-distribution [review].

17. **What causes the early class's larger deviation?**
    - Unidentified.
    - Symmetrising the readout rows did not reduce it, but the test cannot exclude the positional hypothesis [frozen, P7, as reinterpreted].

18. **Which claims are non-identifiable from behaviour alone?**
    - ρ (power-mean) versus modified target evidence plus sharpening [proof, XIII-C].
    - Any implementation-level claim in this report, which needs interventions.
    - Decoding-based claims about use (Q8).

19. **What predicts the learning transition?**
    - mix_mass at step 246 has AUC 0.86 against accuracy 0.71 and loss 0.64. Its advantage over behaviour is not established (+0.15 [−0.07, 0.37]) [expl].
    - Training loss does not predict the transition.

20. **What extends to COR and what is specific?**
    - Possibly general, as methodological points:
      - interventions need ideal-model reference values;
      - decoders are not causal directions;
      - which algorithm emerges can hinge on initial conditions of freshly added components, and on early route availability.
    - Specific to this generator and architecture: every numerical result.
    - Nothing here bears on the O/D operators [spec, for the general part].

## Mandatory completion gates

| gate | result |
|---|---|
| Isolated readout-map initialisation effects | W-only and L-only cells in 3 parents; factorial in 4 parents (P1, P10 falsified); D has no class effect |
| Mechanistic explanation of the readout-map influence | No W-only property explains it. The influence is interaction-dependent with L and the parent, and W's orientation matters (development). The full mechanism is open (OPEN #1), and a discriminating design is specified there |
| Competition versus commitment | Pooled information retained while use falls; partial readout reactivation; P2a falsified; locus of lock-in not isolated |
| Release-time / critical-period experiment | Masks of 0–100 and 0–250 steps and a 546–1146 window; no recovery window of ≥ 100 steps (P2b); sham and shorter windows not run |
| Availability and use of the unused route | Pooled presence 0.93–0.98, use ≈ 0, group-resolved presence reduced; reactivatable after a 100-step block, not after 250 |
| 3-d carrier necessity and null pairs | P4a (reverse swap), P3b (nulls); channel not unique; k fixed at 3 |
| Encoding-versus-decoding comparison | Done at a matched site; reread as pattern ≠ filter (review 1, review 2) |
| What the repair representation carries | m and log r |
| Causal use by the frozen final layer | Joint layer-2 key–query score (P3e) |
| Matched supplied-information adaptation | P5a/P5b at matched lr and steps |
| Layer-0-specific repair | P5c; layer-0 attention suffices; depth alternative untested |
| Corrected eligibility-site intervention | ALLX2/ALLXQ2; specificity criterion found ill-posed in review |
| Corrected attention-row controls | NOOP tolerance; the borrowed-pair rescoring was found powerless in review, and base-pair decoy control retained |
| Additivity versus separate pathways | Additive under path patching; non-redundant under blocking |
| Early-class approximation errors | Positional hypothesis not supported, but not excluded; error source unidentified |
| Learning-transition evidence | B5: mechanistic advantage over behaviour not established (n = 33) |

## Post-review corrections

Each change below alters the scientific meaning of a first-draft statement. All are in `reviews/ADJUDICATION.md`.

1. **"Mixture is the default/attractor; early is fragile/conjunctive": withdrawn.** It is replaced by an interaction-dependent description with base rates (review 2, F1 and F3).
2. **"The decision is made within 100 steps" is now "sensitive to route availability in the first 100 steps"** (review 2, F5; review 1, F10).
3. **"Decodable ≠ used" is now "patterns transmit, filters do not",** which is expected under correlated noise (review 1, F5; review 2, F7).
4. **"A source-free interchange carries ~0.6, so the readout token holds the effect": withdrawn.** SRCF is an unrelated-donor control, with an ideal value of 0.72 (review 1, F1).
5. **P4b ("synthetic ablation is non-specific") is non-informative.** An ideal carrier damages to KL 0.347 (review 1, F2).
6. **P6's falsification concerns an ill-posed criterion, and the token-location reading is withdrawn** (review 1, F4).
7. **The E2 borrowed-pair arm has no power, and "no generic signal transfers" is withdrawn** (review 1, F7).
8. **P7 cannot exclude H_pos.** "Never reduces MOVE" was wrong: SYM2 reduces it by 5–7% (review 1, F8).
9. **P8 shows non-redundancy only** (review 1, F9; review 2, F11).
10. **"Layer 1 cannot repair" was first supported at lr 1e-3 only. A post-review arm at lr 3e-3 also fails (0.021). P5b's "query/key side" labels are now readout-token / source-token supply** (review 2, F10).
11. **Integrity.** Interim scoring of freeze 2 was disclosed; P5b's basis drew on it. The sensitivity script was fixed. F1b was re-described. Effect-size tables now use confirmation units only (review 3).

## Post-review exploratory results (not frozen)

**L1only at lr 3e-3 on uP165 s20** (review 2, F10). This used the same settings as the freeze-3a lr-3e-3 arms (`logs/adapt4_postrev/`):

- acc 0.731
- KL 0.169
- J_DM slope **0.021**

So layer-1 retraining fails at the stronger budget too. The "layer 1 cannot" statement therefore holds at both learning rates tested: 0.004 at lr 1e-3 and 0.021 at lr 3e-3.

The depth alternative (layer 1 plus an extra fresh block) remains untested.

## Limitations

- One generator family, six pretrained parents, small transformers.
- Small samples: 1–5 factorial cells per parent; 1–2 models per carrier and access confirmation; n = 1 per direction for each mask arm.
- P10 is post hoc. P5b was frozen from theory, using an interim look. One P6 row unit had been seen in XIII-C.
- The transform screen and B5 are exploratory.
- No per-episode sampling error was computed for the borderline verdicts.
- Out-of-memory kills on the 2-CPU, 8 GB container forced re-runs. All were re-run with unchanged commands and are logged in `logs/COMPUTE_LOG.txt`.
