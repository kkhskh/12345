# Investigation XVII: relational computation, positional access, and reuse

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%, unchanged. No thesis is selected, no phase is closed, and the O/D operators are not validated.

**Reviews.** The reviews in `reviews/` are agent reviews, not human peer review.

**Dependencies.** The archive depends on six earlier COR packages and on XVI's models (`logs/DEPENDENCIES.json`). It is not standalone.

**Evidence labels:**
- [proof] mathematical proof; [num] independent numerical verification;
- [dev] development (existing XVI models); [frozen] frozen confirmation (fresh models);
- [expl] exploratory causal evidence; [fals] falsification; [spec] speculation.

## Summary

**Stage 1** (committed `431b1b6`, before any neural result) established that COR's original O/D mathematics is **silent** on positional reuse. So nothing below is evidence for or against O/D.

**Stage 2** (development on XVI's C_s202, WB_low_s503 and WA_const_s402) found:
- the architecture makes "position" nothing more than a slot code (Theorem P);
- an unacquired slot follows its gate when it carries the acquired slot's label in the first two layers;
- transplanting another slot's content-specific attention does not carry it, though matched routing is necessary (W7);
- slot-targeted training does not rescue a stuck slot, and damages the others. Any single-slot objective does this (W8).

**Freeze `c419773`.**

**Confirmation:** 12 fresh models, 4 at arity 3 and 8 at arity 2.
- Every frozen mechanism verdict came out as predicted in all 6 evaluable models, 3 at each arity.
- **After review, these are not 6 independent confirmations (W5).** P1 is a numerical check of Theorem P, and P3 re-measures P2. The independent tests are P2, P4, P5 and PL, and their interpretation is narrowed (W3, W7–W9).
- **Not identified.** No XVII intervention identifies which of the mandate's H2, H3 or H4 holds (W3, W5).
- **P0.** The prediction that partial-slot states would be resolvable in most acquiring models failed at arity 2 (P0 MIXED). At 20-step resolution, arity 2 is unresolved (W12).

## Frozen verdicts (logs/score_XVII.json)

Frozen calls are unchanged. The last column gives what each call means after review (`reviews/ADJUDICATION.md`).

| | verdict | pooled | arity 3 | arity 2 | post-review reading |
|---|---|---|---|---|---|
| P0 staggering | **MIXED** (expectation STAGGERED **not met**) | 6/11 | 3/3 | 3/8 | arity 2 unresolved at 20-step resolution; not a test of H1 (W6, W12) |
| P1 gate use follows the code | FOLLOWS_CODE | 6/6 | 3/3 | 3/3 | numerical check of Theorem P (W5) |
| P2 early-code transfer | TRANSFER | 6/6 | 3/3 | 3/3 | observation stands; does not separate H2 from H3 (W3) |
| P3 late layer less code-dependent | LATE_LESS_CODE_DEPENDENT | 6/6 | 3/3 | 3/3 | re-measures P2; layer locus withdrawn (W4, W5) |
| P4 routing insufficient | ROUTING_INSUFFICIENT | 6/6 | 3/3 | 3/3 | content-mismatched transplant; matched routing necessary, partly sufficient (W7) |
| P5 code keying persists | CODE_KEYING_PERSISTS | 6/6 | 3/3 | 3/3 | descriptive; does not exclude H1-with-binding (W9) |
| PL learning | INTERFERENCE_NO_ACCELERATION (no-acceleration half at the ceiling) | 5/5 | 3/3 | 2/2 | interference is generic single-slot forgetting (W8) |

## Answers to the 28 questions

**COR mathematics**

1. **Does O/D predict any property of reusable relational computation?** No. Its definitions contain no symmetry, sharing or acquisition structure [proof, ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md].
2. **Position-independent behaviour, position-dependent access, or neither?** Neither. It is silent (Case D), with two limited negative properties (Case C).
3. **Which original assumptions would be required?** None within or near O/D yields a reuse prediction. An equivariance axiom on ρ and P would be new mathematics, and it would only give invariance across a network and its permuted copy (θ, θ_π), not reuse within one trained network [proof; W2].
4. **Do earlier counterexamples undermine the connection?** There is no connection to undermine (Case D). The corrected Proposition S′ adds that every source-level O_c or D_c is an order-≤2 score, and that no order-≤2 score reproduces the GATE Bayes decisions at arity 2 or 3 [proof + exact LP, Review A]. The original Proposition S and its "0 of 729" figure are withdrawn (W1).
5. **Is the resulting prediction distinctive?** No O/D prediction exists. The ideal decision is the same at every slot by exchangeability, a standard Bayesian fact.

**Neural computation**

6. **Is slot-local acquisition reproducible?**
   - Short slot staggering: yes at arity 3 (3/3 fresh); inconsistent at arity 2 (3/8) [frozen P0].
   - The long partial window (C_s202) is rare: 0 of 11 fresh acquiring models, and 1 of 16 XVI acquisitions.
7. **Does relational ability follow embedding, content, position or their interaction?**
   - It follows the question code E_q, in interaction with the slot's own content [proof, Theorem P; num; frozen P1 6/6].
   - "Physical position" does not exist in this model. Content is exchangeable across slots.
8. **Can relational behaviour be transferred from an acquired slot to an unacquired one?** Yes, by giving the unacquired slot the acquired slot's code in layers 0–1 only: 0.65–0.90, against perturbation controls of 0.00–0.07 [frozen P2 6/6; expl, review; dev 0.68–0.96]. KL to Bayes improves as well (B9). Every frozen checkpoint lay 20–220 steps before natural acquisition, so transfer far from acquisition rests on one development model (W10, W11).
9. **Does transfer survive source-matched controls?** Yes, by construction. Each slot keeps its own tokens and evidence, and only the code moves. A U → U exchange gives model means of 0.04–0.06 (pair-level 0.00–0.13) [frozen; corrected W15]. No slot's input content is copied to another slot, and no task-relevant evidence moves (W11).
10. **Does attention-pattern transfer reproduce the behaviour?** Not when the donor's content-specific attention is transplanted: 0.05–0.17 [frozen P4 6/6]. A content-matched transplant gives 0.11–0.47, and forcing baseline attention during the relabel cuts transfer to 0.02–0.32. Routing is therefore necessary and partly sufficient [expl, review; W7].
11. **Are causal activation directions shared across positions?**
    - The causal **site** is not localised. The claim that it is layers 0–1 is withdrawn: derived slot identity persists in the residual, and in A2_s704 a layer-2-only identity swap transfers 0.65–0.70 [expl, review; W4].
    - A shared **direction** was not tested. Codes are near-orthogonal, and no alignment maps were fitted, by design.
12. **Evidence for a common relational computation?** Not identified. P2 shows that an unacquired slot's content is gated correctly when the acquired slot's label drives layers 0–1. Under Theorem P this does not separate a shared computation with code-keyed access from an adoptable code-conditioned implementation (W3).
13. **Evidence for position-dependent access?** Descriptively, yes: gate use depends on which trained label a slot carries in layers 0–1 (donors lose it, 0.02–0.17; recipients gain it), and label mixtures degrade it after acquisition [frozen P2, P5]. "Necessary and sufficient" and the layer locus are withdrawn (W4), and "access" versus "implementation" is not identified (W3).
14. **Evidence for distinct position-conditioned implementations?**
    - Only the straw form (taking another slot's code cannot help) is rejected [frozen P2]. The mandate's H3 is not distinguished from H2 (W3).
    - The slot-specific learning evidence is non-diagnostic (W8).
15. **Which mechanisms cannot be distinguished?** "Shared computation with code-keyed access" and "a code-conditioned implementation that any slot can adopt by taking the code" are observationally equivalent under these interventions [argument]. In this architecture the equivalence is definitional: both describe the same object (Review A F10). H2, the mandate's H3 and H4 are not identified (W5). Untested refinements that could give H2 separate content: a low-dimensional engaging subspace of code space; a decodable but unused gate computation in unacquired slots; a code-independent subcircuit switched by a low-rank code signal.

**Learning**

16. **Does improvement at one position affect learning at the others?**
    - Not positively: under slot-targeted training the other unacquired slots stay at 0.00–0.10 [corrected W15].
    - Negatively: the acquired slots lose gate use, from 0.86–0.96 down to 0.11–0.29 [frozen PL interference + expl]. Training on an already-acquired slot's loss alone causes the same collapse, so this is generic forgetting and not specific to the targeted slot (W8).
17. **Can targeted intervention accelerate one position without training the others?**
    - Not testable here. In fresh models the targeted slot acquired the gate within 100–200 steps, as fast as under full training (a ceiling), and it was already 20–220 steps from natural acquisition. No acceleration is shown (W8).
    - In C_s202's long window, targeted training did not rescue the slot in 1000 steps, and code-only training failed in 1200 steps [dev].
18. **Reusable computation or position-conditioned specialisation?** Not identified (W5; the earlier "H4 mixture" verdict is withdrawn). What is established descriptively:
    - gate use is conferred by carrying an acquired slot's trained label in layers 0–1;
    - matched routing is necessary;
    - training any single slot degrades the others.
19. **Stable across independent fresh models?** The frozen calls: yes, 6/6 (95% CI 0.54–1.00), but they amount to 4 independent mechanism tests plus P0 (W5). Staggering at arity 2 is unresolved at 20-step resolution (W12).
20. **Comparable mechanisms at arity 2 and 3?** Yes, in every evaluable model. Arity 3 staggers more reliably.

**Scientific significance**

21. **Predicted before confirmation:** P1–P5 and PL interference, all met in fresh models. After review, P1 is architectural, P3 duplicates P2, and P0, P3 and PL replicate development regularities rather than test hypotheses (W5, W6).
22. **Predictions that failed:**
    - P0 STAGGERED (MIXED);
    - PL's "no acceleration" was uninformative (ceiling);
    - development's "targeted slot does not acquire" did not generalise to fresh short windows [fals].
23. **Mechanisms ruled out by matched controls:**
    - transfer of task evidence (excluded by the relabel's construction);
    - transplanting another slot's content-specific attention as a sufficient mechanism (P4, qualified by W7);
    - only the straw form of non-transferable per-slot implementations (P2; W3).
    - "Uniform reuse" is **not** ruled out (W9).
24. **Does this explain the earlier T4/GATE implementation heterogeneity?** Not tested. Code-keyed early processing suggests one way different seeds could reach different layouts [spec].
25. **What is genuinely novel relative to mechanistic interpretability?**
    - Position-conditioned computation in shared weights is known (e.g. digit-specific streams in addition).
    - What is distinctive here is a **symmetric** task, an **exactly equivariant** architecture, and **alignment-free, source-matched** interventions. That setting gives a clean, alignment-free demonstration of label-driven transfer between slots [frozen; NOVELTY_AND_PRIOR_EXPLANATIONS.md]. It does **not** separate "the computation exists" from "a position can use it" (W3). The mechanism is ordinary conditional computation.
26. **What XVII established about COR's original theory:** that it is silent on this question, and that its source-level form cannot represent the GATE decision (corrected S′; W1). Nothing in XVII validates O/D.
27. **What remains unproven:**
    - what makes a code engaging (the code acts only as a unique binding label: duplicating it does not transfer);
    - the layer at which an unacquired slot is deficient (model-dependent, W4);
    - transfer far from acquisition in fresh models (W10);
    - why some slots acquire first;
    - whether the long C_s202 window has a special cause;
    - shared activation directions;
    - the architecture control;
    - generality beyond this Transformer and task family.
28. **The next most informative direction** [judgement]:
    - the geometry of code engagement (cheap, on existing checkpoints);
    - then the architecture control with tied codes.

## Disclosures

- **Development models** were selected for being informative: C_s202 is the rare long-window case. All development results are labelled [dev] and were confirmed or refuted only in fresh models.
- **Only 6 of 11 acquiring fresh models were evaluable** (resolvable partial checkpoints), so the mechanism verdicts rest on 6 models.
- **Censoring.** A3_s801 never acquired the gate within 5000 steps.
- **Not run:** the architecture control (Experiment E), for budget reasons.
- **Diagnostic size.** Per-slot diagnostics used 600 episodes in the confirmation analyses (development used 600–1500).
- **Integrity rule.** Monitoring was restricted to the lane log and file existence. Outside the frozen queue, no confirmation output was read before scoring (W14). Reviewer C confirmed this from first-read access times.
- **Concurrent development job (W13).** A development job on the XVI model C_s202 ran from about 07:25 to 08:02, alongside the blind confirmation training. It used XVI checkpoints only and touched no confirmation output, but it was missing from COMPUTE_LOG.txt and from the timeline.

## Post-review corrections

These come from `reviews/ADJUDICATION.md`. All three reviews are agent reviews, not human peer review. The frozen calls in `logs/score_XVII.json` are unchanged. What changed is what they are taken to mean.

- **W1.** Proposition S and the "0 of 729" figure are withdrawn. The gate-flip statistic used only the evidence sources, so 0 was guaranteed. The corrected S′ holds: source-level O/D is an order-≤2 score, and no order-≤2 score reproduces the GATE Bayes decisions.
- **W2.** An equivariance axiom would give only cross-network invariance, not a reuse prediction.
- **W3.** P2 does not separate H2 from the mandate's H3. Under Theorem P a slot is its code, and the code acts only as a unique binding label. The "common relational computation" and "H3 rejected" readings are withdrawn.
- **W4.** The layer locus ("causal site layers 0–1", "late layer code-agnostic", "necessary and sufficient") is withdrawn.
- **W5.** "6/6 mechanism predictions held" overcounts the evidence. P1 is architectural and P3 re-measures P2. The "H4 mixture" verdict is replaced by descriptive claims; H2, H3 and H4 are not identified.
- **W6.** P0, P3 and PL replicate development regularities. They are not hypothesis tests.
- **W7.** Routing: the frozen transplant was content-mismatched. Matched routing is necessary and partly sufficient.
- **W8.** PL interference is generic single-slot forgetting, and "acceleration" was not testable.
- **W9.** P5 does not rule out uniform reuse.
- **W10.** Frozen transfer was measured only 20–220 steps before natural acquisition.
- **W11.** Controls and wording:
  - perturbation controls are 0.00–0.07;
  - U→U control counts were duplicated;
  - "no activation moves" is replaced by "no content or task evidence moves".
- **W12.** P0 at arity 2 is unresolved at 20-step resolution.

- **W13.** A concurrent development job on C_s202 (07:25–08:02) is now disclosed. It touched no confirmation output.
- **W14.** "First read" now means first read outside the frozen queue.
- **W15.** Number corrections:
  - arity-3 acquisition spreads are 60–100 steps;
  - other unacquired slots under targeted training end at 0.00–0.10;
  - U→U controls have model means of 0.04–0.06.
- **W16.** Frozen documents carry post-review banners. The frozen text is at `git show c419773:<path>`.

**Integrity review (C)** found no critical or major problems. The freeze chronology, the 27 freeze hashes, the absence of early reads, run completion and scorer reproduction all checked out.
