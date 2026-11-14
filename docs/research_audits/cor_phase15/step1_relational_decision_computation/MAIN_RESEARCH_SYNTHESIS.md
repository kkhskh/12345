# Investigation XIII: from relational information to candidate decisions

## Main research synthesis

COR Phase 1.5, Step 1 (~25%). This is not a phase closure. It states no thesis, does not claim that COR is validated, and draws no
connection to O/D.

Every result carries a label:

| label | meaning |
|---|---|
| **[proof]** | mathematical proof |
| **[num]** | numerical verification |
| **[dev]** | development evidence: models seen while designing the work |
| **[conf]** | independent confirmation on the fresh cohort, frozen before training |
| **[expl]** | exploratory, not frozen |
| **[spec]** | speculation |

Source documents:
- `NATIVE_DECISION_COMPUTATION.md`
- `CAUSAL_MEDIATORS_AND_CONTROLS.md`
- `MATHEMATICAL_SUFFICIENCY_AND_GENERALIZATION.md`
- `FROZEN_PROTOCOL_AND_CONFIRMATION.md`
- `OPEN_COR_FOUNDATIONAL_MAP.md`
- `reviews/`

## 1. Principal results

1. **The generator's posterior is now exactly known, and the sufficiency boundary is drawn.** [proof] [num]
   - `posterior_shared` equals a brute-force latent-variable sum over all 531,441 inputs of a reduced environment, for C ∈ {1, 2} and
     β ∈ {1, 0.6}. The error is ≤ 3.3e-16, and calibration ECE is ≤ 0.003 at full size.
   - For C = 1, a fixed three-component per-group statistic with a pooled readout is sufficient at every β. The familiar (S − e_g)
     form is its β = 1 special case. The hard rule is undefined on 77% of β = 0.6 episodes.
   - For C = 2 with shared copying, the fixed per-group statistics are insufficient, and a cross-group statistic is needed:
     - at β = 1, exactly on the 5.5% of episodes where S = 3 and the groups do not all agree at q;
     - at β < 1, broadly: the product rule's posterior is off on 85% of episodes and its decision on 37%. Any factorised representation
       needs ≥ 729 features per group.
   - Review 1 found two description errors and both were corrected: a mis-described counterexample, and undefined rows that had been
     scored as answers.

2. **A pooled-versus-per-group factorization question is unidentifiable by realizable inputs** in T4 (C = 1, β = 1). [proof]
   On the support, (e, S) ↔ (e, m) is bijective. XIII therefore makes no behavioural or natural-intervention claim that separates
   pooled S from per-group m_g.

3. **In one class of solutions, layer 2 uses readout attention rows as mixture weights over "which group is reliable".** [dev]
   [conf, n = 2]
   - *Causal part.* Replacing the mixture heads' readout attention row changes the output the way a mixture
     Σ_g w_g L_g(y) with those weights predicts. This holds for natural donor rows (R² 0.915 and 0.933 [conf]; shuffled ≈ 0) and almost
     as well for a uniform row (0.86–0.88; review 3).
   - *Correlational part.* The weights resemble the E-based Bayes responsibilities r_g ∝ W_g(S − e_g)ΣL_g:
     - for the **dominant** head only: corr 0.95–0.98; in uP165 two of three mixture heads reach only ≈ 0.53;
     - in a flattened form: score coefficients 0.57–0.69 on log W, where Bayes would be 1, and a finite S = 1 exclusion gap of about 6
       logits.
   - "Attention ≈ Bayes responsibilities" is therefore an observational match, not a causal identification.
   - *E rather than A1.*
     - On the 93 donors where E and A1 differ, a Bayes(E) prediction fits the transplanted change far better than an A1 prediction
       (0.93–0.94 vs 0.19–0.21 [conf]). This contrast is **observational**: the attention itself already separates E from A1, and all 93
       donors share one configuration.
     - An exploratory interventional check suggested by review 3, within the frozen 400-episode TRUE factorial: the E-mixture predicts
       a transplant change of |Δ| 1.02 and the A1-mixture 0.32. The observed change is 0.97 (uP161) and 0.72 (uP165), with fidelity
       R² 0.96 / 0.76 for E vs 0.60 / 0.50 for A1 [expl].
   - The rows are an eligibility-specific, **partial** mediator: DiD 0.61–0.76, projection on total effect 0.25–0.33 [conf]. The "decoy"
     is the lowest-mass head, not a size-matched one.

4. **The solution class is heterogeneous and graded, and the mixture result is fragile at the class boundary.** [dev] [conf]
   - Fresh complete parents: 2 mixture, 2 pre-assembled (decision largely in resid2 before layer 2), 1 borderline.
   - The borderline uP164 misses the frozen share threshold by 0.001. Including it falsifies P1 and P2.
   - The incomplete uP162 passes P1 but fails P2b.
   - The mechanism is supported for strong-share mixture parents only.

5. **Matched-budget DAS did not recover the variable used.** [dev] [conf]
   - E-trained subspaces preferred E over A1 on E≠A1 pairs in only 1/5 fresh parents, which falsifies P8a.
   - The E-, A1- and PERM-trained subspaces share 4–5 of their 6 directions (review 3).
   - On E≠A1 pairs, every subspace's fit to A1 is within 0.05 of its fit to PERM.
   - A source-free patch fits A1 as well (0.59–0.69).

   So the claim is the weak one: DAS did not identify E here. It is **not** that DAS implements whichever variable it is trained for.
   Mean-patching this shared 6-dim region removes the eligibility score contrast, with overshoot and reversal (155–222% removal), for
   all three subspaces alike.

6. **Layer-1 dependence and redundancy.** [dev] [conf]
   - Under full layer-1 mean ablation, the mixture heads' alignment with r_g collapses (0.95 → ≈ 0). So there is no layer-0-only route to
     the mixture weights. Partial or resample ablations were not run, and the behavioural residual was not computed.
   - Eligibility is decodable at 1.0 both from the E subspace and from its 442-dim complement.
   - The Bayes decision is decodable from the complement (0.82–0.89 in all six fresh parents), not from E (0.40–0.45).

7. **Frozen C = 1 features suffice for an adapted or fresh layer 2 to compute the exact C = 2 posterior.** [dev] [conf]
   - With layers 0–1 frozen, attention-only, MLP-only, full or fresh layer-2 adaptation all reach Bayes in **5/5** fresh complete
     parents (acc 0.79–0.80; S = 3 acc 0.93–0.97).
   - Nothing tested whether changing layers 0–1 would be *necessary* or would also suffice, so "local repair" means sufficiency only.
   - A readout-only adaptation shows a **graded, post-hoc-described** pattern:
     - strong mixture parents (uP161, uP165): poor overall (0.61–0.65), but most S = 3 decisions (0.82–0.91);
     - the others: the reverse.

   This falsifies P5's readout arms. The development result on vP150, disclosed after the freeze, anticipated it. The claim that the
   C = 1 representation already supports most J_H-dependent decisions holds in **2/5** parents only.

8. **Dynamics.** [expl] Attention alignment with Bayes responsibilities appears at or before the behavioural A2 → B separation in all 7
   mixture-forming models. But the KL drop reaches its halfway point **at or before** the alignment onset in all 7. The apparent "lead"
   comes from a late, noisy behavioural gap on a coarse grid; 6 of 10 models cross at the same checkpoint. **No internal predictor of
   the transition is claimed.** Pre-assembled models make the transition without any mixture head.

## 2. Predictions (frozen; FROZEN_PROTOCOL §4 and §6)

Thresholds and predictions were frozen at commit 47506c3 (pushed) before any cohort T4 fine-tune. The scorer is the same file that
defines the predictions.

| | P1 | P2 | P3 | P4 | P5 | P7 | P8 | P6 (expl) |
|---|---|---|---|---|---|---|---|---|
| verdict | confirmed | confirmed | confirmed | confirmed | **falsified** | confirmed | **falsified** | pass |
| n | 2 | 2 | 2 | 2 | 5 | 2 | 5 | 1 |
| type | discriminating | discriminating | qualified | discriminating | – | non-diagnostic (correlational) | – | – |

Counts:

| count | value |
|---|---|
| numeric passes | 5 |
| discriminating passes | 3 |
| qualified passes | 1 |
| non-diagnostic passes | 1 |
| falsifications | 2 |
| inconclusive | 0 |
| protocol deviations | 0 (plus 4 scheduling-only changes) |

Sensitivity: including the borderline uP164 turns P1 and P2 into falsifications.

## 3. Answers to the mandate's final questions

**1. What does layer 2 compute?**
- In strong mixture-class parents, it combines per-group candidate likelihoods with readout attention rows used as mixture weights.
  The weights causally act as weights [conf, n = 2]. That they equal E-based Bayes responsibilities is an observational match: dominant
  head only, flattened [dev] [conf].
- The S = 1 exclusion is approximated by a finite score gap of about 6 logits, where Bayes would use −∞. [dev]
- In other parents the decision is assembled before layer 2, by an unidentified mechanism. [dev] [conf]
- No single "native layer-2 algorithm" holds across all parents.

**2. Pooled, per-group, pairwise or other?**
- For C = 1, the mixture form (a) is supported in mixture-class parents [conf].
- Pooled S versus per-group m_g is **not identifiable** by realizable inputs [proof].
- Pairwise and direct-lookup implementations were **not tested**. The ATT-MIX fidelity R² concerns changes caused by row swaps, not the
  decision, so it does not bound other mechanisms.

**3. Necessary versus redundant.**
- *Necessary* in mixture parents:
  - layer-1 computation: full mean ablation removes the mixture alignment [conf];
  - a 6-dim resid2 region shared by the E-, A1- and PERM-trained subspaces: mean-patching it removes and reverses the score contrast
    (155–222%). This is a synthetic (class III) intervention and is not E-specific [conf].
- *Redundant*: eligibility information in the complement [conf]; decision information in the complement [conf].
- *Partial*: the attention-row pathway carries 25–33% (projection) of the total eligibility effect. The rest goes through value, MLP and
  other routes, which are **not decomposed**.

**4. Is E a genuine natural mediator, or partly an artefact of learned interchange?**
- The subspace evidence does not establish E.
  - E-trained DAS subspaces are often not E-specific [conf; P8 falsified].
  - The trained subspaces largely overlap (4–5 of 6 directions).
  - A source-free patch fits A1 about as well [conf; review 3].
- The evidence that E, not A1, is the variable used comes from the attention-row mediator:
  - observationally, on 93 E≠A1 donors [conf, 2/2];
  - interventionally, in the exploratory 400-episode TRUE-factorial check (E fidelity 0.96 / 0.76 vs A1 0.60 / 0.50) [expl].
- That the six-dimensional E subspace *itself* is a natural mediator is **not established**.

**5. What remains outside E and after layer-1 ablation?**
- Outside E: full eligibility (decodable at 1.0) and most decision information (0.82–0.89) [conf].
- Under full layer-1 mean ablation, layer 2 does not recover the mixture weights from layer-0 features [conf]. The behavioural
  residual, and attention mass after ablation, were not computed.

**6. Is the two-colluder decision explained by a causally identified joint cross-group representation?**
- **No, unresolved.**
- Mathematically, the joint statistic is needed only on the S = 3 not-all-agree slice at β = 1 [proof].
- A linear readout of frozen C = 1 features gets 0.82–0.91 of S = 3 decisions in mixture parents [conf]. That is correlational
  (class V). No representation of J_H was causally identified.

**7. What changes when a downstream block is retrained for C = 2?**
- Adapting either the layer-2 attention or the layer-2 MLP alone is sufficient, given frozen C = 1 features [conf, 5/5]. A fresh layer 2
  also works, so the result concerns the sufficiency of the C = 1 features. Necessity of layer-2 change was not tested (no
  layers-0–1-only adaptation).
- Which components carry the new rule after adaptation, and whether the mixture heads are reused, was **not analysed**.

**8. Which statistics suffice under which generators?** See principal result 1 [proof] [num].

**9. Do fresh models replicate?**
- The mixture mechanism, its E-specificity at the natural mediator, the non-reconstruction and the redundancy replicate in the 2 fresh
  strong-mixture parents.
- The universality of local layer-2 repair replicates in 5/5.
- The predicted readout pattern and the DAS E-specificity do not replicate.

**10. Invariant versus history-dependent.**
- *Invariant across all 6 fresh parents*:
  - the decision is decodable from the complement, not from E;
  - local layer-2 repair;
  - the A1-subspace behaviour;
  - the C = 1 network's chance performance on C = 2.
- *Class-dependent*: whether a mixture head exists; the readout pattern under C = 2; the strength of ATT-MIX.
- *Pretraining*: the incomplete parent uP162 is mixture-class, consistent with development (5/5 incomplete parents were mixture-class),
  but n = 1 [expl]. The gated remedy rescued 3/4 stalled parents.

**11. What generalizes beyond T4?**
- *Mathematical*: sufficiency of per-group statistics depends on the generator's dependence structure, and shared copy events force
  cross-group statistics [proof, within this generator family].
- *Methodological, supported here and plausibly general* [spec beyond T4]:
  - matched-budget DAS can fail to recover the used variable when candidate variables are highly correlated on the training pairs;
  - natural-activation tests with predictions that differ between candidate variables are a necessary complement. Here they separated
    E from A1, observationally and in one exploratory intervention;
  - scorer preflight against null units catches real criterion bugs.
- *Task-specific*: the mixture mechanism and its class frequencies.

**12. Implications for COR.**
- *Additional support*: in a part of the trained population (2 of 5 fresh complete parents, fragile at the class boundary), the network
  combines per-group likelihoods with attention weights that act as mixture weights and resemble the Bayes responsibilities over
  source reliability built on the correct relational variable [conf, n = 2; observational match].
- *Untested*: GATE and other families; adaptive sources; unknown group structure; β < 1 regimes in trained networks.
- *New problems*: see OPEN_COR_FOUNDATIONAL_MAP. They include the pre-assembled class, the graded class boundary, J_H representation
  after repair, and restoration via distinct pathways.

## 4. Limitations

- **Small n.** Mechanism confirmation rests on 2 fresh units, the frozen minimum. With n = 2, parent-level t-intervals are not
  meaningful confidence intervals and are reported only as ranges.
- **Fragility at the class threshold** (the uP164 sensitivity).
- **No outcome blindness.** Thresholds were set on development data.
- **Restoration through a distinct pathway** was not frozen or run.
- **GATE cross-family comparison** was not run.
- **One fine-tune seed per fresh parent.**
- `stats_xiii.py`, `report_xiii.py` and `transition_predictor.py` are post-freeze. R1–R4 were logged before any cohort output was
  opened.
- The archive is **not standalone**: development models live in sibling `cor_phase15` packages (see REPRODUCTION.md).

## 5. Reviews

1. **Math** (`reviews/REVIEW_1_MATH.md`). Core mathematics confirmed. Two description errors, both corrected.
2. **Integrity** (`reviews/REVIEW_2_INTEGRITY.md`).
   - Confirmed: the freeze hashes and the timeline.
   - Found: four reporting and checking hazards, addressed by rules R1–R4 logged before any cohort output was read.
   - Disclosures added.
3. **Mechanism and causality** (`reviews/REVIEW_3_MECH.md`).
   - Confirmed the verdicts and the uP164 sensitivity.
   - Found the causal claims worded more strongly than the data support; ten wording must-fixes, all applied.
   - Suggested an interventional E-vs-A1 check, run as exploratory.

Adjudication: `reviews/ADJUDICATION.md`.
