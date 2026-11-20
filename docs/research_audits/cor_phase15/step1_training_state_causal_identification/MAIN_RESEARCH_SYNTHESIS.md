# COR Investigation XIX: which learned state changes carry a training-induced switch between relational organisations

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%. No phase closure, no thesis selection. The original O/D framework is not validated. The proposed O/D repair (project doc) is not used as evidence.

**Reviews.** All by AI agents, **not human peer review**:
- pre-experiment: A (identifiability), B (graph/optimizer), C (measurement); three rounds, final classification "Approved";
- post-results: D (causal claims) and E (integrity). Their corrections are applied here (`REVIEWS_AND_ADJUDICATION_FINAL.md`).

**Evidence labels.**
- [proof]: mathematical consequence of the construction.
- [num]: numerical verification.
- [dev]: development data (D801, one pair per base).
- [frozen]: confirmation (D821–823, 3 pairs per base; design frozen in `478a038` and pushed before the first run).
- [frozen-addendum]: the functional battery, frozen in `8ca9db7` after confirmation had started but before any output was read.
- [expl]: exploratory or post hoc, including post-hoc readings of frozen data.
- [falsified]: a frozen prediction that was CONTRADICTED.
- [interp]: interpretation.
- [spec]: speculation.

**Scope of every frozen statement.**
- One parent (uP160).
- One fresh-layer initialisation seed per base (E: L = W = 21; M: L = W = 23).
- 3 data orders per base.

E-versus-M differences are therefore confounded with that initialisation seed and with the target layer.

## 1. Result

**Setting.** A T4 Transformer with two competent organisations:
- **EARLY:** the answer is read from own-question sources at layer 0;
- **MIXTURE:** the answer is pooled at layer 2.

A 100-step block of the route a base would otherwise use switches it to the other organisation. This was XIV's result. The XIX anchors reproduce it on 6/6 fresh pairs [frozen].

**Where the switch is carried at step 100** [frozen, synthetic transplant states].

| | E (early base) | M (mixture base) |
|---|---|---|
| Moments | carry nothing at the full-state level (0.00–0.01) | carry nothing at the full-state level (0.02–0.19) |
| Parameters | carry the switch | carry the switch |
| Sufficient | the layer-0 state (0.97–0.99) | L0+L1 and L0+L2, each (0.97–0.99). L0 alone is not (0.26–0.38) |
| Necessary | the layer-0 state (restoring B's layer 0 gives 0.00–0.02) | restoring L0+L2 together prevents the switch (0.06–0.13). Restoring L0 alone is PARTIAL (0.32, 0.43, 0.75) |
| Caveat | close to tautological: the E route that is measured lives in layer 0. Neither the QK part nor the OV part of layer 0 is necessary alone (0.96–0.98) | necessity of L0 alone, of either partner, and of L1+L2 without L0 is **not established** |

**General optimization disturbance** is eliminated in the forms tested, on both bases [frozen]:
- norm-matched random displacement at step 100;
- a displacement-matched random walk during the window. This one, like the step-100 displacement, leaves step-100 function unchanged;
- label noise whose step-100 loss disturbance matches the block's (E) or is 4.7–12.4× larger (M).

**How the window produces it** [frozen].
- **Training-time content loss with ordinary gradients (vrs) switches fully** on both bases (0.99–1.00).
- **On M**, stopping the gradient into V and WO read through the route, with the forward pass and the per-step routing gradient as in plain training (vsgo), switches fully (0.98–0.99).
  - Stopping only the value-vector gradient (vsg) also switches fully (0.97–0.99). This is a **post-hoc reading of a CONTRADICTED prediction** [expl]: the frozen prediction was a partial switch.
  - These are modified learning rules (synthetic gradients). They are not an ordinary network losing information.
  - Under vsg/vsgo the route's learned attention share still falls by 63–72% of the block's fall by step 100. So a routing-state change that follows from the altered value learning is **not excluded** as the mediator.
- **On E**:
  - vsg is nil on 2 of 3 pairs [falsified: a partial switch was predicted].
  - vsgo gives 0.69, 0.56 and 0.95. That is PARTIAL, predeclared to mean "not established".
  - Freezing layer-0 routing parameters with the moments updating (frzp) gives partial results (0.54–0.69).
  - **What suffices on E, other than content loss and the block, is not established.**

**Function** [frozen-addendum]. A route-free behavioural signature follows the causal class in all 86 non-hybrid confirmation models: 40 switched, 46 not, from 3 data orders per base.
- The signature is where the model errs against exact Bayes. The EARLY organisation errs mainly when the reliable pair disagrees.
- It correlates 0.97 with early-route necessity and 0.94 with KL.
- The EARLY organisation is a ≈ 2× less precise approximation, with errors that look like unstructured logit noise around Bayes [interp].
- So this is best described as **causal implementation reorganisation with a precision difference**. It does **not** establish a different functional rule (Claim 2) or a different algorithm (Claim 3).

**Falsifications.** Three frozen predictions were contradicted: E and M "vsg partial", and the M "vsgo exceeds vsg" contrast, which falls with M vsg. All three replicated single development pairs. The one development pair was not representative for vsg.

## 2. What XIX established that XVIII did not

- **Matched non-targeted control (XVIII confound 5): resolved on both bases.**
  - The tested general-disturbance forms X8a (displacement) and X8b (loss disturbance) are eliminated.
  - **This is the mandate §19 success condition.** These eliminations were predeclared and reviewed before confirmation and were SUPPORTED on both bases.
- **Learned state (confound 7): addressed.**
  - The state is localised at the level of sufficient layer sets.
  - Necessity is established for E layer 0 and for M L0+L2 only.
  - Moments are excluded at the full-state level.
- **Content against value learning (confound 1): partly addressed.**
  - On M, M_P4_vsgo_switch (SUPPORTED) shows that forward-content loss and per-step routing-gradient removal are not *necessary* for the switch.
  - This reading departs from the predeclared joint P4 rule, which failed on both bases.
  - On E the question is not resolved.
- **Routing learning (confound 2): narrowed.**
  - Freezing routing parameters does nothing on M (frz and frzp both nil) [frozen].
  - On E, frzp is partial [frozen].
  - frz is nil on 2 of 3 E pairs and partial on E823 (PARTIAL). The θ-identical frz/frzp contrast is consistent with pre-review B's moment-artefact explanation, but this is not established at frozen level.
- **Scorer failure: repaired.**
  - 6/6 pairs evaluable; all 99 models competent.
  - The verdicts are robust under 12 frozen sensitivity variants. The rare changes are listed in `FROZEN_CONFIRMATION_AND_STATISTICS.md`.

## 3. Answers to the 31 mandated questions

### Causal identification

**Q1. What learned state difference exists after 100 steps?** [num; dev inventory; frozen states hashed]
- Every parameter group and both Adam moments differ; the step counter and data RNG are identical. [dev E801] The largest differences are in layer 0 (0.8–1.1 of B's movement). [dev M801] They are in L2_OV / L2_QK, with large upstream differences too.
- **E, function at step 100** [frozen]: B already uses the early route (nec_E 0.23–0.30) and I does not.
- **M, function at step 100** [frozen]: the states are almost identical (KL 0.425–0.445, nec_M ≈ 0), but the route's attention share differs (B 0.167–0.170, I 0.138–0.139).

**Q2. Which groups differ most, and is that causally relevant?**
- Size is not relevance. [dev M801] L2 differs most and its restoration does nothing.
- [frozen] Relevant: E layer 0 (sufficient and necessary); M sufficient sets L0+L1 and L0+L2, with L0+L2 necessary jointly. [dev] L1 alone and L2 alone are insufficient.

**Q3. Can intervention-trained parameters change a baseline's eventual organisation?** Yes [frozen]:
- E: B←I L0, 0.97–0.99;
- M: B←I L0+L1 and B←I L0+L2, 0.97–0.99;
- both: all parameters with B's moments, 0.98–1.00.

**Q4. Can restoring baseline parameters prevent the change?** [frozen]
- E layer 0: yes (0.00–0.02).
- M L0+L2: yes (0.06–0.13).
- M L0 alone: PARTIAL (0.32, 0.43, 0.75).

**Q5. Does it depend on optimizer history?**
- At the full-state level the moments alone do nothing (E 0.00–0.01, M 0.02–0.19) [frozen]. This is band inclusion with margin 0.25, not a formal equivalence test: on M the descriptive t-interval reaches 0.31.
- Inside composites the moments shift φ by 0.17 [dev].
- Training-time freeze arms (not natural trajectories): frz against frzp, with identical θ_100, gives 0.01 / 0.01 / 0.50 against 0.59 / 0.54 / 0.69 on E [frozen, frz PARTIAL].

**Q6. Individually sufficient, or only in combinations?**
- E: layer 0 is sufficient as a whole. Its tensor classes are individually redundant for necessity [frozen]. No single class or pair of classes is sufficient [dev].
- When layer 0 is incomplete, the non-L0 state of I contributes. I←B L0_QK (0.96–0.98) [frozen] contrasts with B←I L0_OV+MLP+LN (0.35) [dev]. That is an X7 interaction.
- M: only layer pairs [frozen].
- None of this is an identified combination of individually necessary changes.

**Q7. Which results could come from unnatural hybrid states?**
- Natural intervention: vrs only.
- Non-targeted disturbances: pnz, lbl.
- Freezes: frzp; frz (with optimizer manipulation).
- Synthetic learning rules: vsg, vsgo.
- Synthetic states: all transplants. [frozen] The E layer-0 composites are worse than both sources at step 100.

**Q8. A specific mechanism, or a family?**
- A family [interp]. Several sufficient state sets exist (M), and several sufficient window interventions (block, vrs, and on M vsg/vsgo).
- No unique mechanism is identified.

### Forward and backward pathways

**Q9. Forward content changed with value learning controlled?** vrs (content rolled; gradient kept but uninformative) switches fully on both bases [frozen]. It does not separate content loss from loss of informative value learning (pre-review A5).

**Q10. Value-learning gradients changed without the forward change?**
- M: vsgo 0.98–0.99 [frozen, SUPPORTED]. vsg 0.97–0.99 [frozen data; the partial prediction was falsified; sufficiency is a post-hoc reading, expl].
- E: vsg 0.02 / 0.02 / 0.68 [falsified]. vsgo 0.69 / 0.56 / 0.95 [frozen, PARTIAL = not established].

**Q11. Which routing-gradient pathways remain active?** [num, `COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md`]
- gsg keeps the complement-side routing gradient (softmax invariance).
- vsg and vsgo keep the full per-step routing gradient.
- frz and frzp remove all QK learning of the target layer, for every edge.
- Upstream routing learning stays active under all of them.

**Q12. Does a graph-verified gradient intervention reproduce the expected effect?**
- vsgo on M: yes [frozen].
- vsg on M: **no**. The predicted partial was falsified and the outcome was a full switch.
- vsg on E: no [falsified].
- vsgo on E: PARTIAL.
- frzp: as predicted, partial on E and nil on M [frozen].

**Q13. Do shared parameters prevent clean separation?**
- Partly [num]. On M every intervention reaches all upstream layers.
- frz and frzp are not edge-local.
- Downstream readers of T's content cannot be stopped without changing the forward pass [proof].
- The separation achieved is between gradient paths into V, O and QK, not between routes.

**Q14. Can general optimization disturbance explain it?**
- Not in the tested forms [frozen]: (i) norm-matched isotropic displacement, at step 100 or as a random walk, which leaves step-100 function unchanged; (ii) label noise with loss disturbance matched on E and 4.7–12.4× larger on M.
- Disturbances that are functionally comparable but not aligned with I − B are tested only by lbl.

### Learned computation

**Q15. A different decision procedure, or a rerouted similar one?** [frozen-addendum; interp] Causal implementation reorganisation with a precision difference. No different rule or algorithm is established.

**Q16. Do the organisations make distinct functional predictions on valid inputs?** Yes, in error structure [frozen-addendum]. Split ratio on confirmation models: EARLY 2.33–3.50, MIXTURE 1.03–1.52. EARLY models err disproportionately when the reliable pair disagrees at the target question.

**Q17. Are they supported by the trained models?**
- [frozen-addendum] Yes, in all 86 non-hybrid models.
- Hybrids are asymmetric: 12 of 13 are mixture-like even at φ 0.50–0.69, while M823 I←B L0 (φ 0.75) is early-like.

**Q18. Which neural routes are necessary and sufficient?** [frozen anchors] At test time EARLY models need the layer-0 own-question route and MIXTURE models need the layer-2 own-question route. The training-time state components are in Q3–Q6.

**Q19. Competent?** Yes: all 99 confirmation models (KL ≤ 0.035, accuracy ≥ 0.806) [frozen].

### Scientific integrity

**Q20. Independent-stream preflight?** [num]
- Passed: causal class stable on 3 streams for 59 of 60 models; 77 planted checks.
- At confirmation, 6/6 pairs were evaluable [frozen].

**Q21. How many independent units support each conclusion?**
- 3 data-order pairs per base, 1 parent, 1 initialisation seed per base.
- 3/3 in band gives an exact one-sided 95% lower bound of 0.37.
- Within a pair, arms share trajectories. They are not independent.

**Q22–Q24. Which frozen predictions succeeded, failed or stayed inconclusive?** The 35 items are not 35 independent confirmations. Counted by elimination family (post-review D8):

| | E | M |
|---|---|---|
| SUPPORTED elimination families | P1, P2, A4-QK, A4-OV, P3, X8a, X8b, vrs: 8 | P1-L0-alone, P2-L0L2, partner sufficiency, P3, X8a, X8b, vrs, P5: 8 |
| P4 joint reading | not established | failed |
| P5 joint reading | not established (frz PARTIAL) | — (counted above) |
| PARTIAL | — | P2-L0-alone |
| Concordance checks (eliminate nothing) | P7 ×2 (1 SUPPORTED, 1 PARTIAL) | P7 (SUPPORTED) |

- **Item counts:** 27 SUPPORTED, 5 PARTIAL, 3 CONTRADICTED. The 0.70 sensitivity reading duplicates E_P4_vsgo_switch, and the two M P4 contradictions are one observation.
- **[falsified]** E_P4_vsg_partial, M_P4_vsg_partial, M_P4_vsgo_exceeds_vsg.
- **PARTIAL** E_P4_vsgo_switch (and its 0.70 reading), E_P5_frz_nil, E_P7_L0_concordant, M_P2_L0_restore_reduces.
- **Cautions:**
  - E_P4_vsgo_exceeds_vsg passed only because vsg was nil, which is the opposite of the mechanism predicted.
  - The P5 nil on M reflects a nil frz and a nil frzp, not a moment contrast.

**Q25. Which mechanisms did the pre-experiment reviewers reject?**
- gsg as removal of route learning.
- vsg as removal of value learning (WO still learns).
- frz's nil as evidence.
- Same-row controls as non-targeted.
- vrs as a separation of value learning from content.
- RPC as Claim-2 evidence.
- Transplants as mechanism identification.

**Q26. Which claims were weakened or withdrawn after the final reviews?** Withdrawn or relabelled (`REVIEWS_AND_ADJUDICATION_FINAL.md`):
- "value-vector learning is sufficient on M", now a post-hoc reading;
- "V+O carries most of the E effect";
- "frz nil shown to be an optimizer artefact";
- "M identified combination";
- "Claim 2 in the weak sense";
- "86 of 86 switches";
- "stable hybrid";
- the novelty increment;
- "these are not weak controls".

### COR significance

**Q27. What new causal explanation did XIX establish?**
- The selection is carried by step-100 *parameters*, not moments, in specific layer sets.
- Tested general disturbances do not produce it.
- On M, a learning-rule change confined to the value pathway of the blocked route (vsgo) is sufficient, with content and per-step routing gradient intact.
- [interp] The step-100 M states are nearly identical functionally yet lead to different organisations: the selection is held in parameter coordinates before it becomes functional.

**Q28. A contribution beyond known pathway-competition research?** [interp]

In one parent and one initialisation seed per base, XIX provides:
1. a state-level localisation, by synthetic transplants, of where a durable switch between two competent organisations is carried at step 100 (E: layer 0, close to tautological; M: sufficient layer pairs);
2. exclusion of Adam moments at the full-state level;
3. exclusion of tested non-targeted disturbances.

The finding that a gradient-only deficit with the forward pass intact can decide the outcome replicates, in a transformer, the mechanism of Kleinman, Achille & Soatto (ICLR 2024). It is not new. Related precedents:
- critical periods: Achille et al., ICLR 2019;
- gradient starvation: Pezeshki et al., NeurIPS 2021;
- training-time suppression changing the learned strategy: Chen et al., ICLR 2024;
- training-time clamping of transformer components: Singh et al., ICML 2024;
- gradient routing: Cloud et al., 2024;
- competing circuits: Zhong et al., NeurIPS 2023;
- layer re-initialisation: Zhang, Bengio & Singer, JMLR 2022;
- early-phase perturbation and re-initialisation: Frankle, Schwab & Morcos, ICLR 2020;
- basins that correspond to strategies: Juneja et al., ICLR 2023; Lubana et al., ICML 2023.

No new phenomenon is claimed.

**Q29. Anything about the original O/D framework?** No. The appendix of `COR_FOUNDATIONAL_RESEARCH_MAP_XIX.md` contains a labelled speculation only.

**Q30. The most important remaining causal uncertainty?**
- What, besides content loss, suffices on E.
- Whether the routing-state change that follows altered value learning is the M mediator.
- Transport beyond uP160 and its fixed initialisation seeds.

**Q31. What next, if the mechanism cannot be fully identified?**
- The smallest additional interventions:
  - a projected routing-gradient stop (only the component of the WQ/WK gradient from T's logits, on both softmax sides), combined with vsgo on E;
  - a step-100 transplant of only the routing state learned under vsgo on M, to test whether the routing change mediates.
- Then transport to uP163, with varied initialisation seeds.

## 4. Limitations

- One parent, one initialisation seed per base, 3 units per base.
- Synthetic transplant states and synthetic learning rules.
- No readout beyond step 1211, so hybrids are not shown to be stable.
- The E layer-0 localisation is close to tautological.
- Three development-based predictions failed.
- Chronology: the functional battery and its calibration were produced while the confirmation was running. It was frozen 14 minutes after confirmation started and before any output was read (self-attestation, consistent with file access times; reviewer E). It is labelled [frozen-addendum].

## 5. Report index

- `XVIII_FAILURES_AND_UNRESOLVED_CAUSES.md`
- `COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md`
- `PARAMETER_AND_OPTIMIZER_STATE_TRANSPLANTS.md`
- `FORWARD_CONTENT_VS_VALUE_LEARNING.md`
- `ROUTING_GRADIENT_CAUSAL_TESTS.md`
- `NON_TARGETED_OPTIMIZATION_CONTROLS.md`
- `FUNCTIONAL_VS_CAUSAL_ORGANIZATION.md`
- `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`
- `FROZEN_CONFIRMATION_AND_STATISTICS.md`
- `COR_FOUNDATIONAL_RESEARCH_MAP_XIX.md`
- `DESIGN_XIX.md`
- `REVIEWS_AND_ADJUDICATION_FINAL.md`
- `REPRODUCTION.md`
