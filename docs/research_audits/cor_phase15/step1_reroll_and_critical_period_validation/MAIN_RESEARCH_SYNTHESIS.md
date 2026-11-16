# Investigation XIII-E: re-roll or structure? Early-training sensitivity and causal measurement validation

**Status.** This investigation is part of COR Phase 1.5, Step 1. The provisional foundational-exploration estimate stays at about 25%; it has not been raised. No thesis is selected and no phase is closed. The O/D framework is not established, and none of the results below support it.

**Reviews.** The reviews in `reviews/` were carried out by independent agents. They are not external human peer review.

**Dependencies.** This package depends on earlier COR packages (`logs/DEPENDENCIES.json`) and is not standalone.

## Purpose

XIII-D falsified XIII-C's claim that the readout map causes the learned organisation. It left a simpler alternative open: re-drawing a seed may just re-roll the trained solution. XIII-E tests that alternative, checks the early-training sensitivity result against proper controls, and corrects the causal measurements. The full protocol was frozen in four stages, and every scorer was built to fail closed.

## Principal results

### 1. Workstream A — re-roll or structure? Mixed (outcome C)

The design was 64 controlled runs:

- **Anchors:** 4, covering 2 parents. Each parent contributed one early anchor and one mixture anchor.
- **Re-draws:** for each anchor, 8 re-draws of W and 8 re-draws of L. The same new seeds were used for every anchor.

Results:

- **No hidden coupling between "matching" seeds.** W_A and the layer-2 tensors come from different sub-keys. Matched and mismatched seed pairs have identical joint statistics (A2).
- **A within-parent W component (R1-W FALSIFIED the re-roll null).** A newly drawn W pushes the outcome in the same direction in two different L/D contexts of the same parent: mean Spearman ρ 0.65, p = 0.006. This is a pooled result: per parent p = 0.086 (uP160) and 0.018 (uP161), and it is carried mainly by graded hybridness in uP161. It shows that the W draw's value matters across contexts, not an identified structure. Cross-parent transfer was not detected (ρ 0.02, low power), and no W-only spectral statistic captures it.
- **No transfer detected for layer 2's initialisation L (R1-L CONFIRMED).** ρ = −0.02. If transfer were similar in both parents, a latent correlation of about 0.6 or more would be unlikely; if it existed in one parent only, the rule would miss it about 30% of the time. The W − L contrast itself is untested (CI [−0.18, 1.38]).
- **Outcome probabilities depend strongly on the parent.**
  - P(early) after a re-draw is 0.47 [0.29, 0.65] in uP160 and 0/32 [0, 0.11] in uP161.
  - uP161's early anchor was a rare draw. That is the selection effect the mandate warned about.
  - No early-versus-mixture-anchor asymmetry was detected, but the intervals allow up to about ±0.5.
- **Step 1211 predicts step 4000 for clear cases (9 of 10 continuations kept their class).** Borderline runs drifted towards mixture.

### 2. Workstream B — early-training sensitivity is route-specific, not a closing critical period

All runs used one parent, two bases and three data orders.

- **R3 CONFIRMED.**
  - A 100-step block of the used route flipped the organisation in 3 of 3 data orders, on both bases.
  - Noise injected at the layer-2 input, matched to the block's immediate loss disruption, flipped 0 of 6.
  - A masked-code-path identity check was bitwise identical to the unmasked baseline. The off-route-edge sham the mandate asked for was not run (deviation).
  - The noise is injected at the layer-2 input and favours the early organisation, so specificity rests on direction on E and on M's flip without loss disruption.
  - On the mixture base the block barely changes the loss (Δ ≈ 0.01), so disruption is not what drives the flip.
- **The effect grows with block length.**
  - 25 steps: 1 of 6 runs changed, and only to hybrid.
  - 50 steps: 6 of 6 changed, mostly to hybrid; two M runs sit within 0.03 of the class boundary.
  - 100 steps: 6 of 6 flipped fully to the opposite organisation.
- **R4 FALSIFIED.**
  - Blocking a route at steps 546–1146, at matched severity, still changed the final organisation in 4 of 4 runs, to a hybrid at step 4000. Matched late noise did not change the route class.
  - *Post-review exploratory:* on the mixture base the late block first flips fully to early (step 1211) and then partly reverses to hybrid by step 1803. The early 100-step flip does not reverse. So early and late differ in persistence on M; on E the two arms are not comparable.
  - Strong early noise flipped the mixture base to early in 2 of 2 competent runs. That noise lands on the layer-2 input, so it partly blocks the mixture route itself.
- **Conclusion.** Sensitivity to route availability is route-specific and does not close after an early window. Early effects are more persistent on M (2 runs per arm); "largest early" in degree is not established. "Critical period" is not supported.
- **Unused-route information.** The pooled information stays present and the group-resolved information is reduced. Reactivating the readout alone recovers part of it (0.35 → 0.55), and that recovery depends on the RO1 representation. This replicates XIII-D.

### 3. Workstream C — what the causal measurements actually show

- **Ideal-model references (C1).** Several XIII-D metrics turn out to be non-diagnostic:
  - the unrelated-donor control (ideal value 0.72);
  - the excess of DAS over the source-free patch (ideal 0.28);
  - ablation damage (ideal 0.35);
  - the restore test (ideal 0.46, not 1);
  - the E1 SRCFREE own-share (ideal 0.40–0.50; P6's criterion was ill-posed);
  - both decoy-row controls (ideal ≈ null ≈ 0).

  The metrics that do discriminate, and survive, are DAS sufficiency (ideal 1 / null 0), reverse-swap necessity (0 / 1) and true row interchange (1 / 0).
- **Decoder quality (C2; R6 CONFIRMED on the held-out carrier unit).**
  - The decoder filters do decode m, as well as the encoding patterns do (1.11× on C-C2).
  - They carry about a tenth of the signal variance, so interchange along them transfers nothing (0.02).
  - Variance-matched random subspaces also fail (0.07).
  - "Decodable but not used" is withdrawn. It is replaced by "filters decode but do not transmit, because of the correlated-noise geometry".
- **Geometry (C3).** The successful subspaces share a common core:
  - DAS, encoding patterns, class means and decoy-trained DAS overlap by 0.33–0.61, against ≈ 0.05 for random.
  - The decoder span is nearly orthogonal to that core (0.05–0.07).
  - The key/query split is a stable property of each model across all sufficient sets.
  - The decoy-trained subspace is equally sufficient, but decoy pairs carry no posterior change, so m-specificity is **not established** either way (post-review; was "contradicted").
- **Dimension (C4; R5 FALSIFIED).**
  - One stable direction suffices in C-C1, and two in C-C2. Dimensions beyond these are not seed-stable.
  - "Three-dimensional" was a design choice and is withdrawn.

## Frozen predictions

- **6 predictions:** 3 confirmed (R1-L, R3, R6) and 3 falsified (R1-W, R4, R5).
- **R2 was not frozen,** because development found no consistent statistic.
- **Protocol:**
  - no partial scoring, and no scorer ran before its inputs were complete;
  - no out-of-memory kills;
  - 4 minor disclosed events (FROZEN_PROTOCOL_AND_STATISTICS.md).

## Answers to the required questions

Evidence labels: [proof] mathematical proof; [num] numerical verification; [dev] development; [frozen] frozen confirmation; [expl] exploratory; [spec] speculation.

1. **Is changing a fresh initialisation more than a re-roll?**
   - For W, the pooled test says the draw's effect transfers within a parent [frozen R1-W]; per parent it is weak, and no property is identified.
   - For L, no transfer was detected, within a bound that assumes similar parents [frozen R1-L]. The W − L difference is untested.
   - Seed matching carries no coupling [num A2].
2. **What are the probabilities?** See REROLL_VS_STRUCTURE.md [frozen data].
   - uP160: P(early) 0.47, P(mixture) 0.28, P(hybrid) 0.25.
   - uP161: P(early) 0, P(mixture) 0.81, P(hybrid) 0.19.
   - Per-cell P(early) ranges from 0 to 0.62, with interval widths of about 0.6–0.7.
3. **Do they differ for early versus mixture anchors?**
   - No difference was detected: +0.125 and +0.25 in uP160, 0 in uP161.
   - The bound is wide, about ±0.5 [frozen data].
4. **Larger than parent baselines, selection and sampling would predict?**
   - Heterogeneity across anchors is explained by the parent.
   - The early anchor in uP161 is consistent with a rare draw [frozen data; inference].
5. **Does step 1211 predict step 4000?**
   - For clear cases, yes: 9 of 10 continuations kept their class.
   - Borderline runs drift towards mixture [frozen].
6. **Is there a validated joint W–L property?**
   - No. Development found none, and R_route has AUC 0.59 [dev / expl].
   - A pooled within-parent W effect exists but is not identified [frozen + expl].
7. **Does brief masking act beyond a sham disturbance?**
   - Yes: 6 of 6 for the mask against 0 of 6 for matched noise [frozen R3]. Only a code-path identity sham was run, and the noise control is not location-matched, so specificity is directional on E and loss-independent on M [review].
8. **What happens at 25, 50 and 100 steps?**
   - Dose-dependent: 1 of 6 changed (to hybrid), 6 of 6 mostly to hybrid, and 6 of 6 full flips [frozen data].
9. **Can a severity-matched late intervention produce a lasting change?**
   - Yes. A late route block leaves a hybrid at step 4000; late noise does not change the route class [frozen data, R4 falsified]. On M the late flip is initially full and then partly reverses [expl, post-review].
10. **Is there a specifically early critical period?**
    - No. Sensitivity persists late; early effects are more persistent on M. Strong early noise at the layer-2 input also redirects [frozen R4 falsified; expl].
11. **What do the ideal references reveal?**
    - Several XIII-D metrics are non-diagnostic (list above).
    - Sufficiency, reverse necessity and true row interchange are discriminating, and they survive [num C1].
12. **How well do the decoders decode, and what does their causal failure mean?**
    - Projection R² is 0.77–0.86, the same as the encoding patterns.
    - Their failure under interchange reflects low signal variance (filters), not missing information [frozen R6; num].
13. **How similar are the successful subspaces?**
    - They share a core: overlap 0.33–0.61, against 0.05 for random.
    - DAS at k = 3 is not unique (0.42 / 0.69 between seeds) [num C3].
14. **Is three-dimensionality a finding?**
    - No. 1–2 stable directions suffice, and the number is model-specific [frozen R5 falsified; num C4].
15. **Which XIII-D causal claims survive?** [num C5]
    - Low-dimensional relational sufficiency.
    - Reverse-swap necessity.
    - Use through the joint layer-2 key–query score.
    - True attention-row interchange.
16. **What is downgraded or retracted?** [num + frozen]
    - Retracted:
      - "3-d";
      - "m-specific: contradicted" (now "not established");
      - the whitened-overlap ordering claim;
      - "decodable but not used";
      - restore "non-separability";
      - P6's specificity reading.
    - Non-diagnostic:
      - P4b;
      - the source-free excess;
      - both SHUFD controls;
      - null preservation as a test of specificity.
    - The restore shortfall and ablation excess are downgraded to "not interpretable as measured".
    - XIII-D's "early critical period" is downgraded to "sensitivity that persists late, more persistent early on M".
17. **Deeper T4 work or broader testing?**
    - One cheap held-out test of the parent-conditional W effect.
    - Then broader cross-family testing (Investigation XIV), carrying the corrected measurement toolkit [inference; COR_FOUNDATIONAL_NAVIGATION.md].
18. **What remains unresolved?**
    - The identity of the W property.
    - Whether layer-0 repair succeeds because of depth or because of access.
    - The source of the early-class error.
    - Generalisation beyond one generator family and 2 parents (A) or 1 parent (B).
    - Passive commitment versus active competition.
    - The off-route-edge sham, and a location-matched noise control.
    - Whether uP161's W signal reflects convergence speed.
    - The size of the model's response to decoy edits.

## Post-review corrections

Three independent agent reviews (A statistics, B mechanisms, C integrity; not human peer review) are in `reviews/`, adjudicated in `reviews/ADJUDICATION.md`. No frozen verdict changed. Integrity review: PASS (timing, preflights, bitwise scorer reproduction, 9/9 incomplete inputs refused). Corrections that change scientific meaning, all applied above and in the workstream documents:

1. R1-W is pooled and carried mainly by uP161; "identifiable structure" withdrawn; the W − L contrast is untested.
2. R1-L's bound assumes homogeneous transfer; the power script was reconstructed (`scripts/a_power_sim.py`).
3. Late versus early on M: one post-review exploratory measurement (no training) shows the late block's full flip partly reverses; "difference of degree" withdrawn, replaced by persistence. On E the arms are not comparable.
4. The sham was a code-path identity check; the off-route sham was not run (deviation). The noise control is not location-matched or direction-neutral.
5. "m-specific: contradicted" → "not established"; the whitened-overlap ordering claim withdrawn; restore shortfall and ablation excess downgraded; C4 wording corrected.
6. Integrity errata: checkpoints committed in three freeze commits (with `resume.pkl` in history); `c_audit.json` hash note; two count slips in frozen records (left unedited, recorded in COMPUTE_LOG).
