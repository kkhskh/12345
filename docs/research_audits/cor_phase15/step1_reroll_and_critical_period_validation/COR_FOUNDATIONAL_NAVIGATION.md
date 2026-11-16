# COR foundational navigation after XIII-E

**Scope.** This document is about direction only. It marks no phase transition. COR stays at Phase 1.5, Step 1, provisional foundational-exploration estimate about 25%. The estimate has not been raised. No thesis is selected, and the O/D framework is not established.

## Evidence favouring a re-roll model

**L: the fresh layer-2 initialisation (R1-L CONFIRMED, bounded).**
- Re-drawing L behaves like a re-roll: no property of a new L transfers across contexts (ρ = −0.02).
- The frozen power bounds a transferable L component below a latent cross-context correlation of about 0.6. Smaller effects remain possible.

**Seed matching (A2, mathematical and numerical).**
- "Matching seeds" carries no statistical coupling: W_A and the layer-2 tensors use different sub-keys.
- Matched and mismatched seed pairs give the same joint statistics.

**Anchor behaviour.** The anchors' own outcomes behave like single draws from parent-specific distributions. uP161's early anchor is a rare event: 0/32 re-draws were early.

## Evidence for structured initialisation

**W: the readout map (R1-W FALSIFIED the re-roll null).**
- A new W pushes the outcome the same way in two different L/D contexts of the same parent (ρ = 0.65, p = 0.006).
- The effect is parent-conditional: it does not carry over between parents (ρ = 0.02).
- No W-only spectral statistic captures it.
- The development transforms in XIII-D point the same way: a sign flip or column permutation of W changes the outcome. What matters is how W is oriented relative to the parent's frozen layers.

**Decision: outcome C, mixed.** A readout-map effect transfers within a parent (pooled over two parents, carried mainly by uP161). Layer-2 initialisation transfer was not detected. The W − L contrast itself is untested.

## Evidence about early-training sensitivity

- **Route-specific effect beyond controls (R3 CONFIRMED).** Real route blocks redirect the organisation; matched noise does not change the route class, and a code-path identity check is exactly neutral. The noise control is not location-matched or direction-neutral, and the off-route sham was not run, so specificity rests on direction on E and on M's flip without loss disruption.
- **Dose-dependent.** A 25-step block rarely redirects. 50 steps gives a hybrid; 100 steps gives a full flip.
- **No closing critical window (R4 FALSIFIED).** Matched route blocks at steps 546–1146 still change the final organisation, to a hybrid at step 4000. On M the late block first flips fully to early and then partly reverses (post-review exploratory), so early and late differ in persistence there. Strong early noise at the layer-2 input also redirects the mixture base.
- **Supported statement:** "sensitivity to route availability throughout the tested range, largest early". The statement "an early critical period" is not supported.

## Causal measurements that remain trustworthy

- Pattern-direction sufficiency (DAS, encoding patterns).
- Reverse-swap necessity, close to the ideal value of 0.
- The joint layer-2 key–query trace, with a K/Q split that is stable within each model.
- True attention-row interchange.
- The supply-arm contrasts from XIII-D.
- The route-mask interventions, with the control caveats above.

## Measurements that are non-diagnostic or withdrawn

- **Non-diagnostic:**
  - unrelated-donor excess (ideal value only 0.28);
  - ablation damage (the ideal carrier also damages, by 0.35);
  - SRCFREE own-share and P6's excess criterion;
  - SHUFD row controls, scored against either the base or the borrowed pair;
  - null-pair preservation as a test of specificity.
- **Withdrawn:**
  - "3-dimensional carrier": it is 1–2 stable directions, and the number is model-specific;
  - "m-specific carrier: contradicted" is downgraded to "not established" (decoy pairs carry no posterior change);
  - the restore shortfall and ablation excess are not interpretable as measured;
  - "decodable but not used": the decoder filters decode m but carry little signal variance.

## Questions that need different tasks or architectures

- Whether a parent-conditional readout-map effect appears in other generator families (GATE, TRANS, adaptive sources) and in deeper models.
- Whether the subspace that decoy-trained DAS exploits is a relational code or only the site's output-sensitive subspace (needs logged decoy-response sizes).

## Deferred T4 questions (not resolved; carried forward)

**Strictly deferred by the mandate.**
1. Layer-1 depth versus access: the "layer 1 + extra block" experiment.
2. The source of the early-class decision error.

**Raised by XIII-E.**

3. What property of W, relative to the parent, produces the transferable effect? The natural next test is a theory-guided W–parent alignment statistic: for example, how W_A aligns with the pretrained layer-1 readout geometry. It should be developed on the 64 existing runs and validated on newly drawn W seeds and a third parent.
4. A severity-matched late block that also matches the full late ro0 disruption.
5. Larger re-roll samples for uP161. Its early probability is bounded only to ≤ 0.11.

## Broader COR directions

**The evidence favours a limited deepening.**

- **Worth one held-out test:** the parent-conditional W effect is the only structured selection effect found. A single held-out experiment on it (deferred question 3) is cheap and decisive.
- **Not worth further investment:**
  - the early "critical period";
  - the component-level "matching" story.

**Moving outward.** Broader cross-family testing (Investigation XIV) has high value, because every positive mechanistic claim so far comes from one generator family and six parents. A reasonable plan:
1. Run the single held-out W test.
2. Move outward.
3. Carry the corrected measurement toolkit along: ideal references, variance-matched controls, fail-closed scoring.
