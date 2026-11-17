# Investigation XIV: final T4 causal controls, a T4 evidence ledger, and cross-family testing in GATE

**Status.** COR Phase 1.5, Step 1. The provisional foundational-exploration estimate stays at about **25%**. No thesis is selected, no stage is closed, and the original obstruction/discrepancy (O/D) framework is **not** validated.

**Reviews.** The reviews in `reviews/` were written by independent agents. They are not external human peer review.

**Dependencies.** The package depends on 12 earlier COR packages (`logs/DEPENDENCIES.json`) and is not standalone.

**Evidence labels.**
- **[proof]** mathematical proof;
- **[num]** numerical verification;
- **[dev]** development;
- **[frozen]** frozen confirmation;
- **[expl]** exploratory;
- **[spec]** speculation.

## Documents

- T4_FINAL_CAUSAL_CONTROLS.md (Stage I)
- T4_FOUNDATIONAL_EVIDENCE_LEDGER.md (Stage II)
- CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md
- GATE_CAUSAL_MECHANISMS_AND_TESTS.md
- CROSS_FAMILY_CONFIRMATION.md
- COR_FOUNDATIONAL_RESEARCH_MAP.md
- `reviews/` and `reviews/ADJUDICATION.md`

## Principal results

### Stage I: the T4 route-blocking result now has proper controls [frozen; one parent, n = 6 pairs]

**R-S and R-N: CONFIRMED.** A 100-step early block of a decision route flips the learned organisation 6/6. Two comparable interventions flip it 0/6:
- an **off-route sham** that uses the same −1e9 masking mechanism in the same layer and readout queries, validated on all 6 baselines as not interrupting either route;
- **logit noise on exactly the blocked edges**, matched in attention-TV severity.

The paired shift differences are 0.80 [0.795, 0.804] for the sham and 0.76 [0.71, 0.80] for the noise. The flips persist to step 4000.

**R-P: CONFIRMED** (4 valid pairs, 4/4; ΔPI 0.81 [0.30, 1.08]). The same 100-step block was applied late, at steps 546–646, and compared at equal numbers of updates after release:
- On M the late block is **reversed within 50–200 updates** (3/3).
- On E it transiently disrupts the early route in 2/3 runs, which recover within about 200 updates, and leaves a lasting hybrid in 1/3.
- Reframed after review B: a block **during formation** selects the organisation permanently; **after formation** the same block is mostly reversed. R-P rests mainly on M (3 of 4 valid pairs).

**Qualifications.**
- Both controls raise the route's attention share; review B measured this for the noise too. They exclude masking or perturbation of these edges as the cause, but not every reduction of the route's share or access.
- The noise is matched in attention TV, not in loss.
- Learning-rate exposure differs between the early and late blocks.
- There is no closing critical period: longer late blocks (XIII-E) and E-802 leave lasting changes.

### Stage II: the T4 evidence ledger

T4_FOUNDATIONAL_EVIDENCE_LEDGER.md records 11 mathematical, 25 neural/causal and 7 measurement claims, each in its latest corrected form with its status. It also lists the 12 explanations that failed in sequence.
- Review A found 6 entries stated more favourably than their sources (N2, N3, N12, N19, N20, N21); all were corrected.
- `scripts/ledger_check.py` verifies all 18 package pointers and 34 quoted anchors against the source documents.

### Stage III: GATE

**Mathematics** [proof, num]:
1. **Relational invisibility (A1) holds exactly in both generators.** No single source carries information about the trust variable: I(g; x_s) = 0 in GATE, and every T4 source's sequence is i.i.d. uniform under every role. Source-wise ceilings are far below Bayes: GATE accuracy 0.801 against 0.900, KL 0.28; T4 agreement with the Bayes argmax 0.61, KL 0.33.
2. **GATE's Bayes rule has interaction order exactly 3.** Every order-≤ 2 rule has gate-flip response exactly 0, and the order-2 optimum is the gate-free agreement rule (NLL 0.431).
3. **The fitted combination operator that separates T4's two organisations is not identifiable in GATE**, because trust is deterministic there. This is a T4-specific phenomenon.
4. **Review A rejected the proposed "common principle"** as having no cross-family predictive content, and that is adopted. What survives is a shared symmetry (A1), an exact diagnostic calculus for GATE, and a GATE-specific learning hypothesis.

**Neural results** (5 models trained from scratch: 3 earlier and 2 new):
- **Relation used, alternatives excluded** [repro]. All models use the gate relation (G 0.99–1.00), are exactly invariant at the decision level to relation-preserving edits (0 changes on Bayes-unique questions), and are far from the exact order-1 and order-2 alternatives.
- **Decodable but not sufficient alone, in s0 and s1** [repro + expl]. Probes find g at sites whose patching does not transfer the decision. The pre-review s2 example was invalid and is withdrawn.
- **Mediation** [expl]. The gate's effect is mediated by a low-dimensional subspace at the evidence tokens. A *shared* subspace needs k = 2 in s1 and s2 (1.00) and k = 4–8 in s0 (0.87–0.90). Review B's per-token 1-d check gives s1 1.00, s0 0.64 and s2 0.60, so "never one direction" holds only for the shared form. The controls (random, decoy, G-site, R-site) fail but are weak.
- **Layouts differ.** Three layouts appear across five models, so implementations are heterogeneous.

**Frozen cross-family prediction X1** [frozen; 2 new models]:
- **X1a: CONFIRMED** (and X1a′, with the post-review stricter criteria). Both models first sit at the exact order-1 ceiling, with no gate use, for 175–350 steps.
- **X1b: FALSIFIED** (and X1b′). Pairwise agreement is not learned to half strength before the gate. s3 realises about 40% of the order-2 gain (A_p ≤ 0.27) and s4 almost none (≤ 0.09). The gate then arrives abruptly.

## Frozen predictions, all freezes

| prediction | verdict | freeze |
|---|---|---|
| R-S (T4 off-route sham) | CONFIRMED | aca3b6c |
| R-N (T4 location-matched noise) | CONFIRMED | aca3b6c |
| R-P (T4 early more persistent) | CONFIRMED | aca3b6c |
| X1a (GATE order-1 phase first) | CONFIRMED | 5d3c9dd |
| X1b (GATE pairwise before gate) | **FALSIFIED** | 5d3c9dd |
| X1a′ / X1b′ (amendment after review A, before inspection) | CONFIRMED / **FALSIFIED** | b019ca9 |

## Answers to the required questions

1. **Does the T4 route-blocking effect survive an actual off-route sham?**
   - Yes. The real block flips 6/6 and the validated sham 0/6 [frozen R-S].
   - The sham raises the route's attention share, so it is a conservative control for masking artefacts, not for attention-share effects.
2. **Does it survive a location-matched disturbance of comparable severity?**
   - Yes, at matched attention-TV severity: 0/6 flips [frozen R-N].
   - It is not matched in loss: on E the noise's loss disruption is about one third of the block's. On M, the block itself barely disrupts the loss.
3. **Are early-induced changes more persistent than late-induced ones under equivalent recovery conditions?**
   - Yes, in this cohort [frozen R-P, 4/4 valid pairs]. Late blocks on M reverse within 50–200 updates; early ones persist.
   - Remaining confounds: learning-rate exposure; the organisation is not yet formed at early release.
4. **What final T4 causal claims survive after all corrections?** [ledger]
   - The exact sufficiency structure.
   - Route-specific early sensitivity with persistence (Stage I).
   - Low-dimensional relational sufficiency.
   - Reverse-swap necessity.
   - True row interchange.
   - The layer-1 conjunction.
   - Decodability is not use.
5. **Which earlier mechanistic claims must remain retracted or qualified?** [ledger §3]
   - Retracted: "readout map selects the algorithm"; "3-d m-specific carrier"; "decodable but not used" in its original wording; Type-A readout variable; cross-curriculum equivalence; curriculum null-response contrast.
   - Qualified: "critical period", now "greater early persistence, not a closed window"; "different combination operator", now "fitted, non-unique".
6. **What exact relational information is necessary for correct decisions in GATE?** [proof]
   - The gate g = [x₀ + x₁ ≡ 0 (mod 3)], a relation between two non-answer sources, invisible to each one alone, together with the trusted pair's votes. (g, x_{S(g)}) is sufficient.
   - Pairwise agreement is informative marginally but is screened off by g.
7. **How does this differ from T4?** [proof]
   - In T4 trust is episode-level and *uncertain*. It is inferred from within-pair agreement aggregated over questions and must be marginalised, with a non-monotone agreement → trust map.
   - In GATE trust is question-local and *deterministic*.
   - The consequences: an operator split appears in T4 but not in GATE, and screening-off holds in GATE but not in T4.
8. **Does the GATE network actually compute or use the required relations?**
   - Yes, in all five models: G 0.99–1.00, exact decision invariance to relation-preserving edits, and gate mediation through the evidence tokens [repro + expl].
9. **Is the information present, accessible and causally used?**
   - Present at many sites (held-out-x₀ probes up to 1.00; chance 0.5).
   - Sufficient for the decision only at model-specific sites: evidence tokens at layer 1 or 2, then the readout.
   - At other decodable sites in s0 and s1, patching does not transfer the decision [expl].
   - Accessible to interchange through a low-dimensional subspace. Whether one direction per token suffices is model-dependent [expl, review B].
10. **Do T4 and GATE support a nontrivial shared mathematical principle?**
    - Only a shared symmetry (A1) and its source-wise ceilings [proof].
    - The proposed broader principle was rejected after review: its predictive content is GATE-specific, or it restates A1.
11. **Can that principle make a prospective prediction that survives new testing?**
    - Partly. The order calculus derived from A1 and interaction order, combined with a simplicity-bias assumption, predicted an order-1 phase, which was confirmed on new models.
    - It also predicted an order-2 phase before the gate, which was falsified [frozen X1].
    - These are GATE-specific tests, not cross-family confirmations.
12. **Which proposed common mechanisms fail under direct comparison?**
    - Every proposed *mechanism* similarity: none is established causally across families (review B).
    - A fixed site or layout.
    - A shared combination operator.
    - "Agreement means trust" (non-monotone in T4, screened off in GATE).
    - An order-by-order learning sequence.
    - Learnability explained by trust order: refuted by T4 DeepSets (3/3) [expl/proof].
13. **Are the similarities mathematical or causal, rather than architectural or training-related?**
    - A1 and the ceilings are mathematical, designed into both generators.
    - "Decodable ≠ used" is a methodological similarity (H5). No causal cross-family similarity is established.
    - "Trust stored at evidence tokens and read by readout attention" occurs in both, but under the same architecture and similar training, so it is not separable from architecture [spec].
14. **Which findings remain specific to T4?**
    - The two organisations and their fitted operators.
    - Initialisation and route sensitivity (one parent).
    - The perfect-agreement (R1) phenomena.
    - The specific carrier dimensionalities (not comparable across families).
    - Curriculum dependence.
15. **Which findings plausibly extend to multiple relational environments?**
    - Relational invisibility under symmetric marginals.
    - Exact source-wise ceilings.
    - Presence ≠ use.
    - Heterogeneous implementations of a shared behaviour.
    - An initial order-1 learning phase (GATE frozen; T4 untested from scratch) [spec beyond GATE].
16. **What counterexamples limit the emerging explanations?** [proof by construction]
    - A one-source gate (A1 fails).
    - Biased noise marginals.
    - Equivalent relational descriptions (sum against recoded equality; pooled S against per-group m).
    - C = 2 shared copying (per-group statistics insufficient).
    - T4 DeepSets learnability.
    - X1b's leap.
17. **Which foundational directions now have the highest value?** [judgement; research map]
    - GATE variants with gate order 2 to 4 and noisy gates, which test the learning-order leap and whether operator splits appear with trust uncertainty.
    - T4 from scratch with dense snapshots.
    - TRANS presence against use.
18. **Which claims about the original COR framework remain untested?**
    - O/D as a framework.
    - Obstruction operators in neural systems.
    - Any mapping from relational trust computation to O/D quantities.
    - Nothing in XIV tests them.

## Limitations

- **T4:** one parent and two bases.
- **GATE:** one architecture and recipe; 2 new models for X1; mechanisms on the 3 development models plus 2 exploratory new ones; DAS was not run on s3 and s4.
- **No structural training-time intervention in GATE.**
- **The cross-family comparison is mathematical and structural.** No cross-task interchange is possible.

## Post-review corrections

The full record is in `reviews/ADJUDICATION.md`. No frozen verdict changed. The corrections that changed meaning:

1. **The common principle was rejected (review A).** What remains is a shared symmetry plus a GATE calculus. Proposition 4's "order-12" claim and the learnability-by-order link are withdrawn.
2. **X1 was amended before inspection** (review A F1). X1a′ and X1b′ were registered and reproduce the frozen verdicts.
3. **Stage I controls (review B).** Both controls raise the route's share. Late E blocks disrupt the early route transiently. PI was reframed as "during formation against after formation". R-P rests mainly on M.
4. **GATE (review B).**
   - The dimensionality claim is restricted to the shared-subspace form.
   - The s2 "decodable ≠ used" example is withdrawn.
   - The controls are weak.
   - No cross-family mechanism similarity is claimed.
5. **X1 wording (review B).** s3 realised partial order-2 use during its gate-free phase.
6. **Integrity (review C).**
   - The GATE s3 and s4 training-loss logs existed before the freezes, and were partly committed with FREEZE X and the amendment. The investigator read only the final s3 loss line (step 4000, 0.1807) before the amendment.
   - Four checkpoints were restored by hand.
   - The preflight checks and the verification script were fixed.
   - All seven verdicts reproduce exactly.
7. **Ledger (review A).** Six entries were corrected. DeepSets 3/3 was added to N2.
