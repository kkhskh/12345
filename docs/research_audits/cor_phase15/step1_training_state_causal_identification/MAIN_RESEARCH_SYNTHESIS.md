# COR Investigation XIX: which learned state changes cause a training-induced switch between relational organisations

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%. No phase closure and no thesis selection. The original O/D framework is not validated. The proposed O/D repair (project doc) is not used as evidence.

**Reviews.** All reviews are by AI agents and are **not human peer review**:
- pre-experiment: A (identifiability), B (graph/optimizer), C (measurement), three rounds;
- post-results: D and E (`REVIEWS_AND_ADJUDICATION_FINAL.md`).

**Evidence labels.**
- [proof]: mathematical consequence of the construction.
- [num]: numerical verification.
- [dev]: development data (D801, one pair per base).
- [frozen]: confirmation (D821–823, 3 pairs per base, design frozen in `478a038` before any run).
- [expl]: exploratory, post hoc.
- [falsified]: a frozen prediction CONTRADICTED.
- [interp]: interpretation.
- [spec]: speculation.

## 1. Result in one paragraph

**Setting.** A T4 Transformer with parent uP160. The two competent organisations are:
- **EARLY:** the answer is read from own-question sources at layer 0;
- **MIXTURE:** the answer is pooled at layer 2.

A 100-step block of the route that a base would otherwise use switches the base to the other organisation. This was XIV's result, and the XIX anchors reproduce it on 6/6 fresh pairs.

**Where the switch is carried at step 100** [frozen]:
- It is carried by the **parameters, not the Adam moments**.
- On the early base, by the **layer-0 parameter state, sufficient and necessary**. Its QK part and its OV part are each unnecessary alone. [dev] No single tensor class is sufficient.
- On the mixture base, by a **combination of layer 0 with either layer 1 or layer 2**. Neither the blocked layer nor layer 0 is sufficient alone.
- A random parameter displacement of the same size, a matched random walk during training, and a matched or larger loss disturbance from label noise **do not** switch either base. **General optimization disturbance is eliminated** in the forms tested.

**How the window produces it** [frozen]:
- **Mixture base.** Removing only the value-vector gradient through the blocked route switches all three pairs fully (vsg), with the forward computation and the routing gradient exactly as in plain training. Value-vector learning through the route is sufficient there. Freezing the blocked layer's routing parameters does nothing.
- **Early base.** Value-vector removal alone is nil on 2 of 3 pairs. Value-plus-output removal (vsgo) gives large, mostly incomplete switches (0.56–0.95). Freezing layer-0 routing parameters gives a stable hybrid (0.54–0.69). Forward-content loss gives a full switch.

**Function** [frozen]. Every causal switch is accompanied by a change in a route-free behavioural signature (86 of 86 models). The signature is where the model errs against exact Bayes. It is consistent with two different approximations to the same Bayes computation, not with a distinct algorithm.

**Falsified dev expectations.** Three frozen predictions, each a replication of a single development pair, were contradicted: the "vsg is partial" pair and the M vsgo-over-vsg contrast. A one-pair development observation was not representative. That is itself a result.

## 2. What XIX established that XVIII did not

- **XVIII confound 1 (vrg mixed content and value learning): resolved on M.** Value-vector learning alone is sufficient, with content intact. On E: value-pathway learning carries most of the effect; content loss or other learning carries the rest.
- **Confound 5 (no matched non-targeted control): resolved on both bases.** The two forms of general disturbance tested are eliminated.
- **Confound 7 (state never examined): resolved.** The selection-carrying state is localised at step 100 (E: layer 0; M: layer 0 plus a substitutable partner), and the moments are excluded at the full-state level.
- **Routing learning (confound 2).**
  - frz's nil was shown to be an optimizer artefact: frz and frzp have identical θ_100 and different outcomes.
  - With the artefact removed, routing-parameter learning contributes **partially on E and not at all on M**.
- **Scorer failure (XVIII §1): repaired.** 6/6 pairs evaluable, all models competent, verdicts unchanged under 12 frozen sensitivity variants except where noted in `FROZEN_CONFIRMATION_AND_STATISTICS.md`.

The central success condition of mandate §19 is met. The discriminating tests were established in review before confirmation. They eliminated "general optimization disturbance" (displacement and loss forms) on both bases, and "forward-content loss is necessary" and "routing-gradient learning of the blocked layer" on M.

## 3. Answers to the 31 mandated questions

### Causal identification

**Q1. What learned state difference exists after 100 steps?** [num, dev; frozen states hashed]
- Every parameter group and both Adam moments differ, while the step counter and the data RNG are identical.
- On E801 the largest differences are in layer 0, at 0.8–1.1 of B's own movement. On M801 they are in L2_OV / L2_QK, with large upstream differences too.
- Functionally, at step 100 the E states differ (B uses the early route, nec_E 0.23–0.30; I does not). The M states are almost identical in function (KL 0.425–0.445, nec_M ≈ 0) although they lead to different organisations [frozen].

**Q2. Which groups differ most, and are those differences causally relevant?**
- Size does not predict relevance. On M, L2 differs most yet is not necessary [dev], and restoring L0 and L2 together prevents the switch [frozen].
- The causally relevant groups are layer 0 on E [frozen], and layer 0 plus a substitutable partner (L1 or L2) on M [frozen].

**Q3. Can intervention-trained parameters change a baseline's eventual organisation?** Yes [frozen]:
- E: B←I L0, 0.97–0.99;
- M: B←I L0+L1 and L0+L2, 0.97–0.99;
- both: all parameters with B's moments, 0.98–1.00.

**Q4. Can restoring baseline parameters prevent the change?**
- E: restoring layer 0 prevents it fully (0.00–0.02) [frozen].
- M: restoring L0+L2 prevents it (0.06–0.13) [frozen]. Restoring L0 alone reduces it on 2 of 3 pairs only (0.32, 0.43, 0.75), so it is PARTIAL [frozen].

**Q5. Does the effect depend on optimizer history?**
- At the full-state level, no: moments alone give 0.00–0.19 and parameters alone give 0.98–1.00 [frozen, equivalence margin 0.25].
- Moments can matter inside composites ([dev] 0.74 against 0.91).
- They also matter on a natural trajectory: frz against frzp, identical θ_100, gives 0.01 / 0.01 / 0.50 against 0.54–0.69 on E [frozen].

**Q6. Individually sufficient, or only in combinations?**
- E: one layer (layer 0) is sufficient, but within it only the tensor classes acting jointly are [dev].
- M: only combinations, layer 0 plus either partner [frozen].

**Q7. Which results could arise from unnatural hybrid states?**
- Every transplant result is a property of synthetic states [proof L3]. The E layer-0 composites are worse than both sources at step 100 [frozen compat].
- The paramI_optB / paramB_optI pairings are synthetic [L4].
- Natural-trajectory evidence comes only from the training-time arms: vrs, frzp, pnz, lbl, and the synthetic-gradient vsg / vsgo.

**Q8. A specific mechanism, or a family of equivalent ones?**
- A **family**. The state-level carrier is identified: layer 0 (E), or layer 0 plus a substitutable partner (M).
- On M, several sufficient routes exist (L0+L1, L0+L2), and on E several window interventions are sufficient (block, vrs, vrg).
- No unique mechanism is claimed [interp].

### Forward and backward pathways

**Q9. Forward content changed with value learning controlled?**
- vrs, with content rolled and the gradient kept but uninformative, gives a full switch on both bases [frozen].
- That learning is no longer informative, so this does not separate content from value learning (pre-review A5).
- vrg (rolled, gradient stopped) also switches [dev; XVIII].

**Q10. Value-learning gradients changed without the forward change?**
- M: vsg (value vectors only) gives a full switch, 0.97–0.99 [frozen].
- E: vsg is nil on 2 of 3 pairs; vsgo (V+O) gives a large, mostly incomplete switch, 0.56–0.95 [frozen].

**Q11. Which routing-gradient pathways remain active under each intervention?**
- [num] gsg keeps the complement-side routing gradient (softmax invariance).
- vsg and vsgo keep the full routing gradient.
- frz and frzp remove all QK learning of the target layer, for every edge, but keep upstream routing.
- Full table: `COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md`.

**Q12. Does a graph-verified gradient intervention reproduce the expected effect?**
- vsg on M: yes. vsgo on M: yes.
- vsgo on E: partly (PARTIAL).
- frzp: partial on E, nil on M, as predicted [frozen].
- The predicted vsg hybrids were **falsified** [falsified].

**Q13. Do shared parameters prevent clean separation?**
- Partly. On M every intervention touches all upstream layers [num].
- frz and frzp are not edge-local.
- Downstream readers of T's content cannot be stopped without changing the forward pass [proof].
- The separation achieved is between gradient *paths* into V, O and QK, not between routes.

**Q14. Can general optimization disturbance explain the results?**
- No, in the forms tested [frozen].
- Matched-norm random directions, matched random-walk noise, and label noise with matched (E) or 5–12× larger (M) loss disturbance all give nil on every pair.
- A disturbance structured along I − B is not "general" (`NON_TARGETED_OPTIMIZATION_CONTROLS.md`).

### Learned computation

**Q15. A different decision procedure, or a similar procedure rerouted?**
- In between: two different approximations to the same Bayes target, with different input-dependent error structure [frozen battery; interp].
- No distinct operator is identified.

**Q16. Do the organisations make distinct functional predictions on valid inputs?**
- Yes. The EARLY organisation errs disproportionately when the reliable pair splits at the target question (SR 2.6–4.0, against 1.0–1.3) [dev, frozen].

**Q17. Are those predictions supported by the trained models?**
- Yes: 86 of 86 non-hybrid confirmation models, and 12 of 13 hybrids are mixture-like [frozen].

**Q18. Which routes are necessary and sufficient?**
- At test time, EARLY models need the layer-0 own-question route, and MIXTURE models need the layer-2 own-question route [frozen anchors].
- The training-time sufficiency and necessity of state components are answered in Q3–Q6.

**Q19. Do the models remain competent?**
- Yes: all 99 confirmation models (KL ≤ 0.035, accuracy ≥ 0.806) [frozen].

### Scientific integrity

**Q20. Did the measurement system pass independent-stream preflight?**
- Yes. Causal class stable on 3 streams for 59 of 60 models; 0 functional/causal contradictions; 77 planted checks pass [num].
- At confirmation: 6/6 pairs evaluable.

**Q21. How many independent units support each conclusion?**
- 3 data-order pairs per base, 1 parent (uP160).
- The exact lower bound for 3/3 in band is 0.37, which is weak population evidence.
- Development D801 is not counted.

**Q22. Which frozen predictions succeeded?** 27 of 35 (list in `FROZEN_CONFIRMATION_AND_STATISTICS.md`).

**Q23. Which failed?** [falsified]
- E_P4_vsg_partial;
- M_P4_vsg_partial;
- M_P4_vsgo_exceeds_vsg.

**Q24. Which remained inconclusive (PARTIAL)?**
- E_P4_vsgo_switch and its 0.70 reading;
- E_P5_frz_nil;
- E_P7_L0_concordant;
- M_P2_L0_restore_reduces.

**Q25. Which mechanisms did the pre-experiment reviewers reject?**
- gsg as removal of route learning;
- vsg as removal of value learning (WO);
- frz's nil as evidence (moment artefact);
- the same-row controls as non-targeted;
- vrs as a separation of value learning from content;
- the residual-profile (RPC) score as Claim-2 evidence;
- transplants as mechanism identification.

**Q26. Which claims were weakened or withdrawn after the final reviews?** See `REVIEWS_AND_ADJUDICATION_FINAL.md`. The resolutions there are applied to this document.

### COR significance

**Q27. What new causal explanation did XIX establish?**
- A state-level localisation of the selection.
- The elimination of general disturbance.
- On the mixture base, the sufficiency of value-vector learning through the blocked route, with forward and routing unchanged.

**Q28. A contribution beyond known pathway-competition research?** A narrow one [interp]:
- Critical periods (Achille et al., ICLR 2019) and gradient starvation (Pezeshki et al., NeurIPS 2021) already predict that early training deficits can durably change what a network learns.
- Competing circuits for one task (Zhong et al., NeurIPS 2023) and early stabilisation of the optimisation outcome (Frankle et al., ICML 2020) are known.
- XIX's increment is an *identification* in one controlled model: which step-100 parameter coordinates carry a durable switch between two competent relational organisations, that optimizer state and generic disturbance do not, and that on one base the blocked route's value-vector learning alone suffices.
- That is a methodological and empirical contribution for one parent. It is not a new phenomenon.

**Q29. Does the result establish anything about the original O/D framework?**
- No. Nothing here derives from or tests O/D.
- The appendix in `COR_FOUNDATIONAL_RESEARCH_MAP_XIX.md` notes a possible dynamic hypothesis, labelled speculation.

**Q30. The most important remaining causal uncertainty?**
- On the early base, what supplies the remainder of the switch beyond V+O learning: downstream readers of T's content, routing learning (frzp partial), or their interaction.
- Generally: whether any of this transports beyond uP160.

**Q31. The next direction, if the mechanism cannot be fully identified?**
- The smallest additional intervention is a **projected routing-gradient stop**: remove only the component of the WQ/WK gradient that comes from T's logits on both softmax sides, leaving other edges' routing learning intact.
- Combine it with vsgo on E. If vsgo plus the projected stop gives a full switch while each alone is partial, the E mechanism is an identified V+O+routing combination.
- Then test transport on a second bistable parent. Only uP163 qualified in XVIII.

## 4. Limitations

- One parent; 3 units per base; synthetic transplant states.
- Synthetic-gradient interventions (vsg / vsgo) change the learning rule, not the information. Their results are statements about modified optimisation.
- The functional signature is consistent with a noise-like approximation difference.
- The E localisation to layer 0 is close to tautological, because the measured route lives there.
- Three dev-based predictions failed. The single development pair misled on vsg.

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
