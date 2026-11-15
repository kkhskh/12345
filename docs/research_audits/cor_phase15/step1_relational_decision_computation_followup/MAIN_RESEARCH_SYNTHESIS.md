# Investigation XIII-B: the unresolved native decision mechanisms

## Main research synthesis

COR Phase 1.5, Step 1 (provisional ≈ 25%). This is not a phase closure. It states no thesis, does not validate COR, and makes no claim
about the O/D operators.

**Evidence labels:**
- **[proof]**: mathematical proof;
- **[num]**: numerical verification;
- **[dev]**: development models (XIII's seed-20 cohort and earlier pools; thresholds were set here);
- **[conf]**: frozen confirmation on new seed-21 *fine-tuning* replicates, not independent pretraining;
- **[add]**: post-freeze addendum registered before evaluation;
- **[expl]**: exploratory.

## 1. Central findings

1. **"Pre-assembled" models are early-readout models, not finished before layer 2.** [dev; partly conf]
   - In uP160 and uP163, candidate evidence reaches the readout token at layer 0 (availability 0.84–0.85).
   - Relational evidence (agreement at other questions) is pooled into the readout token by broad layer-1 heads, and layer 2 (mostly its
     MLP) combines the two there. Freezing layer 2 at the readout token removes 91–97% of a relational edit's effect.
   - Mixture models instead keep candidate evidence at the source tokens, route relational evidence into the q source tokens through layer-1
     attention, and combine through layer-2 attention.
   - The relational effect starts in layer 0 at the edited group's tokens in every model (transfer 0.84–0.95) [conf].
2. **Same weighting variables, different combination operator.** [dev; conf; add]
   - Every model, of either class, responds to on-support eligibility edits like the E-based rule. Distance to B is 0.11–0.28; distance to
     A1 is ≥ 1.03 in 5/5 confirmation units.
   - "E-like" here means agreement-at-q-gated: the D1/D2 edits do not separate E from the threshold-4 rule (review 3).
   - Every model is fit by a reliability-weighted mixture family with R² 0.97–0.995.
   - In a fair nested power-mean family (ρ = 1 mixture, ρ → 0 normalised log-linear pooling, found after math review 1):
     - early-readout models combine with **ρ ≈ 0.54–0.62**;
     - late-readout (mixture) models with **ρ ≈ 0.87–0.98**;
     - hybrids in between;
     - corr(lik_RO1, ρ) = −0.90 over 18 development models and −0.99 over 5 confirmation units (AQ1 held).

   ρ is a **robust class marker of the input-output law**: bootstrap intervals exclude 0 and 1, and it follows each unit's own readout
   signature even where a seed moved the parent. Its reading as a "different combination operator" is **not unique**:
   - the power mean is algebraically identical to tempered per-group evidence plus output re-sharpening;
   - a lapse floor moves ρ by about 0.1;
   - ρ explains only about 15% of the early class's misfit (review 3).

   Supported: the two classes realise **different intervention-response laws** for the same target, with class-correlated errors.
3. **Solution classes form a discrete early-readout cluster plus a continuum, largely but not wholly inherited.** [dev; conf]
   - Over 26 development models, lik_RO1 forms a tight cluster at 0.82–0.83, a cluster near 0, and a band of hybrids.
   - New fine-tuning seeds kept the class in 3/4 predicted parents. **uP163 moved from early to hybrid**, and the unpredicted boundary
     parent uP164 moved from hybrid to early. So 2 of 5 parents moved by more than 0.3 in lik_RO1.
   - **Exploratory seed 22** (post-freeze):
     - uP161, a mixture model under seeds 20 and 21, became **early-readout**: lik_RO1 0.85, no mixture head, ρ 0.56;
     - uP163 and uP164 also landed early.
   - **Exploratory seed 23:** uP160 (early under two seeds) became a **mixture** model (ρ 0.94); uP161 a hybrid.
   - So the class is **not** a property of the pretrained parent. It is set largely by fine-tuning randomness and moves in both
     directions; only uP165 was stable across three seeds. Q2's frozen confirmation (3/4) overstates inheritance.
   - Stable: the structural law relating readout signature to the power-mean exponent, corr(lik_RO1, ρ) = −0.93 over all 29 fine-tunes.
4. **The attention-row mediator is specific relative to other heads, but routing proportions vary.** [dev; conf]
   - With purely relational edits the mixture rows carry 68–88% of the effect (Shapley), against 25–33% in XIII, whose design also
     changed candidate evidence.
   - Contribution- and magnitude-matched decoy heads give about 0 [conf, 2/2]. On the E-vs-A1 discriminating family the rows follow E
     (fid 0.87–0.89) and not A1 (0.04) [conf].
   - Permuted or source-free rows still produce large non-specific changes.
   - The restoration test **could not be carried out in one of two units** (Q4c): in uP161 seed 21 the layer-1 route into the q tokens
     carries less (route_Q 0.38; removal removes only 23%).
   - Q3b failed for the same reason. **Q3b and Q4c are one event**, within stream-to-stream noise (0.42 and 0.45 on two other streams).
   - The rows still passed sufficiency, specificity and E-vs-A1 in that unit.
   - The "categorical routing law" is withdrawn: routing proportions vary with the fine-tuning seed.
5. **Balanced E/A1 identification.** [dev; conf]
   - Training DAS on 50% discriminating pairs learns an E-specific, source-dependent subspace in late and hybrid models (fid_E 0.54–0.77,
     source-free 0.01). In early-readout models it is weak (0.13–0.38), which falsified Q5a.
   - In **all five** units the apparent A1 subspace is matched or beaten by a source-free patch (A1 0.61–0.73 vs source-free 0.78–0.87).
   - Identification of E at the q-token site therefore holds only for late and hybrid models. A1 is not identified anywhere.
   - Q5a's falsification reflects **site coverage** and the target. That site carries only 0.49–0.57 of the relational effect in early
     units, and the target included each model's own deviation from Bayes. It is **not** evidence against E (review 3).
   - A post-hoc rerun at a wider site (readout token plus all source tokens) did not rescue it: 0.42 and 0.18. In early-readout models a
     low-dimensional linear E abstraction at the layer-2 input is weakly identifiable at either site [post-hoc].
6. **The two-colluder repair: which relation, and where.** [proof; num; dev; conf]
   - **At β = 1** (G = 3, C = 2, on-support), the exact decision needs only per-group self-agreement plus the joint values at q; no
     cross-group equality at other questions is needed [proof; corrected after math review 1].
   - **At β = 0.6** that local observation falls 3.6 points short of Bayes. Cross-group equality at the other questions supplies the
     rest [num].
   - Valid isolating J-pairs relabel one group's values outside q. They keep every per-group statistic and the values at q, and change the
     exact decision on 21.5% of pairs; the per-group product rule predicts 0 change.
   - **Confirmed [conf, Q6, 5/5]:**
     - adapting **layers 0–1 with layer 2 frozen** reaches Bayes (0.791–0.798 vs 0.796) and uses the cross-question relation (J-pair slope
       0.87–0.95);
     - a layer-2-only repair uses it only in early-readout units (0.53, 0.63), not in late-readout units (−0.00, below the local ceiling).
   - In the late-readout uP161, 4× the layer-2 training budget still gives slope −0.001 [dev].
   - The frozen C = 1 features *contain* the relation: it is decodable at the source tokens at R² 0.81–0.99 beyond per-group statistics.
     At the readout token it is strong only in early-readout models (0.80–0.87 vs 0.39–0.58).
   - Early adaptation writes it into the tokens that the frozen mixture reads. Capacity is unmatched: two blocks against one.
   - How the frozen layer 2 uses it is untested.
   - **Decodable is not used.** Accessibility at the readout position, set by the class's earlier routing, decides whether a final-layer
     repair can exploit it.

## 2. Predictions

| prediction | verdict | note |
|---|---|---|
| Q1 | confirmed | arm a non-diagnostic against normalised pooling |
| Q2 | confirmed | 3/4; low power; one parent switched |
| Q3 | **falsified** | uP161 seed 21 route_Q 0.38 |
| Q4 | **falsified** | arm c, uP161 removal 0.23; arms a, b, d, e held |
| Q5 | **falsified** | arm a in early-readout units; arms b, c held |
| Q6 | confirmed | arm c alone non-diagnostic |
| AQ1 (addendum) | holds | |

Counts:
- 3 confirmed, of which 1 has every arm strictly diagnostic;
- 3 falsified;
- 0 inconclusive;
- 0 protocol deviations.

The predictions share measurements, so they are not independent.

## 3. Answers to the mandate's final questions

1. **How do pre-assembled Transformers compute?**
   - Early candidate readout: layer-0 readout heads read the values at q.
   - Broad layer-1 pooling of relational agreement features into the readout token.
   - Combination by layer 2, which is serially necessary: freezing it removes 91–97% of the effect.
   - Post-hoc layer-2 Shapley shares in early models:
     - attention 0.38–0.57, including values from other-question tokens;
     - MLP2 0.25–0.36;
     - the readout-token state assembled by layer 1: 0.18–0.28.
   - In mixture models attention is 0.83–0.90.
   - Inputs come both from the readout token and, via layer-2 attention, from other-question source tokens (40–54% of the relational
     effect).
   - The input-output law has a power-mean exponent ρ ≈ 0.55–0.6. The split between attention and MLP inside layer 2 is not identified.
     [dev; conf; add; review 3]
2. **Same abstract computation or distinct algorithms?**
   - The same target function and the same relational variables (E-based eligibility, reliability weighting).
   - A different combination operator (ρ ≈ 0.6 vs ≈ 0.9) and different locations.
   - Equivalent at the level of the variables used (agreement-at-q-gated eligibility, reliability weighting).
   - **Not** equivalent at the level of the intervention-response law (ρ). Whether that is a different combination operator or a different
     evidence-tempering is not identified. Their approximation errors are class-correlated: 0.64–0.72 within the early class [dev].
3. **Does the mixture mediator survive size- and contribution-matched controls?**
   - Yes, relative to other heads' rows, in 2/2 confirmation units.
   - Not relative to permuted or source-free rows judged by projection alone.
4. **Balanced E/A1: variable-specific mediator, or a weaker DAS interpretation?**
   - Both. E is identified, source-dependently, in late and hybrid models.
   - A1 "identification" is a source-free artefact in all models.
   - In early-readout models the q-token site misses most of the E effect, which falsified Q5a.
5. **Can a distinct pathway restore the effect after removal?**
   - Yes where the removal is substantial: uP165 seed 21 (0.38 → 0.97), uP161/uP165 seed 20 [dev], with E-not-A1 restoration.
   - The test was **not achieved** in uP161 seed 21, where removing route Q removes only 23%.
   - The restoration route lies downstream on the same causal path; it is not a parallel route.
6. **What accounts for the remaining effect?**
   - Purely relational edits: MLP2 (8–17%) and other heads (0–17%) in mixture models.
   - Hybrids: other heads, MLP2 and the skip path dominate.
   - XIII's "unexplained 67–75%" was mostly a design confound: its edits also changed candidate evidence.
7. **The joint relation and where it is used.**
   - What is needed: cross-group value equality at questions other than q, only when copying is partial. At exact copying, local equality
     at q suffices [proof].
   - Repaired early layers construct it and route it to the readout and q tokens, where the frozen mixture uses it [dev; conf].
   - The exact constructed feature is not identified.
8. **Layers 0–1 alone with layer 2 frozen?**
   - **Yes at β = 0.6, in 5/5 units [conf].** It outperforms every layer-2-only repair in late-readout models.
   - At β = 1 (uP161, development), both layers-0–1-only (0.801) and layer-2 (0.797) repairs reach Bayes (0.798), so neither is
     necessary.
9. **Stable across seeds; categorical or continuous?**
   - A discrete early-readout attractor plus a continuum of mixture-head and hybrid models.
   - **Not stable across fine-tuning seeds**, in both directions:
     - uP161: late, late, early, hybrid;
     - uP160: early, early, late;
     - only uP165 was stable.
   - Stable: the mapping from readout signature to power-mean exponent ρ (corr −0.93, 29 fine-tunes).
10. **New theorems, counterexamples, restrictions.**
    - The β = 1 local-sufficiency proposition [proof].
    - The LOCALQ/FULL gap at β < 1 [num].
    - Valid isolating J-pairs.
    - Identification restrictions:
      - mean-patch and permuted-row decoys are non-specific;
      - a learned subspace's success must be compared with source-free patches;
      - the site of a subspace search can miss a class's route (Q5);
      - unnormalised product-of-experts comparisons are uninformative.
11. **What connects to wider COR, and what is T4-specific?**
    - *Plausibly general [spec]:*
      - the copying structure decides whether local or cross-question relations are needed;
      - representational accessibility, not presence, limits downstream structural repair;
      - balanced and source-free controls for causal abstraction.
    - *T4-specific:* the particular routes, the class frequencies, and ρ values.
12. **Most valuable open questions.**
    - Why the early class settles on ρ ≈ 0.6, and what in pretraining sets the class.
    - Whether the late-readout repair limit is architectural.
    - What feature early adaptation constructs.
    - See OPEN_COR_FOUNDATIONAL_QUESTIONS.md.

## 4. Limitations

- Confirmation units are fine-tuning replicates: n ≤ 5, with filtered arms on n = 2.
- The predictions share measurements.
- The preflight nulls are author-written.
- Development coverage was thin for Q5 (no early-readout development das_bal, because of OOM) and for Q6.
- The class_features `mix_mass` episodes reuse development seeds.
- Mathematical results are limited to G = 3, C = 2 and this generator family.
- GATE was not run.
- The learning-transition question has a behavioural answer but no internal predictor.

## 5. Reviews

Three independent agent reviews; these are not human peer review.
1. **Math** (`reviews/REVIEW_1_MATH.md`).
   - Proposition proof gap fixed.
   - PERGROUP re-described.
   - J-pairs at β = 1 corrected.
   - POE family shown to be handicapped, which led to the power-mean analysis and addendum AQ1.
   - κ "softer exclusion" claim withdrawn.
2. **Integrity** (`reviews/REVIEW_2_INTEGRITY.md`).
   - Chronology and hashes confirmed.
   - Reporting rules RB1–RB5 adopted before any confirmation output was opened.
   - Disclosures added.
3. **Mechanism** (`reviews/REVIEW_3_MECH.md`).
   - Q5 reinterpreted as a site and target limitation.
   - Q3b and Q4c identified as one event.
   - Class inheritance and ρ wording weakened.
   - "MLP combination", decoy, restoration and adaptation claims narrowed.
   - das_bal's "NOOP" control disclosed as not a no-op.

Adjudication: `reviews/ADJUDICATION.md`.
