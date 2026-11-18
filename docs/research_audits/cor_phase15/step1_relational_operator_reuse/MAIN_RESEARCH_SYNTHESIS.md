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
- an unacquired slot can run the acquired slots' gate computation when given their code in the first two layers;
- attention routing alone does not carry it;
- slot-targeted training does not rescue a stuck slot, and damages the others.

**Freeze `c419773`.**

**Confirmation:** 12 fresh models, 4 at arity 3 and 8 at arity 2.
- Every frozen **mechanism** prediction held in all 6 evaluable models, 3 at each arity.
- The prediction that partial-slot states would be resolvable in most acquiring models failed at arity 2 (P0 MIXED).

## Frozen verdicts (logs/score_XVII.json)

| | verdict | pooled | arity 3 | arity 2 |
|---|---|---|---|---|
| P0 staggering | **MIXED** (expectation STAGGERED **not met**) | 6/11 | 3/3 | 3/8 |
| P1 gate use follows the code | FOLLOWS_CODE | 6/6 | 3/3 | 3/3 |
| P2 early-code transfer | TRANSFER | 6/6 | 3/3 | 3/3 |
| P3 late layer less code-dependent | LATE_LESS_CODE_DEPENDENT | 6/6 | 3/3 | 3/3 |
| P4 routing insufficient | ROUTING_INSUFFICIENT | 6/6 | 3/3 | 3/3 |
| P5 code keying persists | CODE_KEYING_PERSISTS | 6/6 | 3/3 | 3/3 |
| PL learning | INTERFERENCE_NO_ACCELERATION (no-acceleration half at the ceiling) | 5/5 | 3/3 | 2/2 |

## Answers to the 28 questions

**COR mathematics**

1. **Does O/D predict any property of reusable relational computation?** No. Its definitions contain no symmetry, sharing or acquisition structure [proof, ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md].
2. **Position-independent behaviour, position-dependent access, or neither?** Neither. It is silent (Case D), with two limited negative properties (Case C).
3. **Which original assumptions would be required?** An equivariance constraint on the compatibility maps ρ and probes P. That would be new mathematics, not part of O/D.
4. **Do earlier counterexamples undermine the connection?** Yes, and Proposition S adds one: source-level discrepancy is order 1, so it cannot express any gate (0 of 729 inputs change under a gate flip, in every condition) [proof + num].
5. **Is the resulting prediction distinctive?** No O/D prediction exists. The ideal decision is the same at every slot by exchangeability, a standard Bayesian fact.

**Neural computation**

6. **Is slot-local acquisition reproducible?**
   - Short slot staggering: yes at arity 3 (3/3 fresh); inconsistent at arity 2 (3/8) [frozen P0].
   - The long partial window (C_s202) is rare: 0 of 11 fresh acquiring models, and 1 of 16 XVI acquisitions.
7. **Does relational ability follow embedding, content, position or their interaction?**
   - It follows the question code E_q, in interaction with the slot's own content [proof, Theorem P; num; frozen P1 6/6].
   - "Physical position" does not exist in this model. Content is exchangeable across slots.
8. **Can relational behaviour be transferred from an acquired slot to an unacquired one?** Yes, by giving the unacquired slot the acquired slot's code in layers 0–1 only: 0.65–0.90 against controls of 0.01–0.12 [frozen P2 6/6; dev 0.68–0.96].
9. **Does transfer survive source-matched controls?** Yes, by construction. Each slot keeps its own tokens and evidence, and only the code moves. A U → U exchange gives 0.01–0.06 [frozen].
10. **Does attention-pattern transfer reproduce the behaviour?** No: 0.05–0.17 [frozen P4 6/6].
11. **Are causal activation directions shared across positions?**
    - The causal **site** is shared: layers 0–1 for every slot [frozen P3].
    - A shared **direction** was not tested. Codes are near-orthogonal, and no alignment maps were fitted, by design.
12. **Evidence for a common relational computation?** Yes. Every unacquired slot tested could run it with an engaging code [frozen].
13. **Evidence for position-dependent access?** Yes. The code in layers 0–1 is necessary and sufficient for gate use (donors lose it, 0.02–0.17; recipients gain it), and the dependence persists after full acquisition [frozen P2, P3, P5].
14. **Evidence for distinct position-conditioned implementations?**
    - The strong form (an unacquired slot cannot use another's computation) is **rejected** [frozen P2].
    - Learning and maintenance, however, are slot-specific (Q16–18).
15. **Which mechanisms cannot be distinguished?** "Shared computation with code-keyed access" and "a code-conditioned implementation that any slot can adopt by taking the code" are observationally equivalent under these interventions [argument].

**Learning**

16. **Does improvement at one position affect learning at the others?**
    - Not positively: under slot-targeted training the other unacquired slots stay at 0.02–0.07.
    - Negatively: the acquired slots lose gate use, from 0.86–0.96 down to 0.11–0.29 [frozen PL interference + expl].
17. **Can targeted intervention accelerate one position without training the others?**
    - In fresh models the targeted slot acquired the gate within 100–200 steps, as fast as under full training (a ceiling), while the others did not [frozen + expl].
    - In C_s202's long window, targeted training did not rescue the slot in 1000 steps, and code-only training failed in 1200 steps [dev].
18. **Reusable computation or position-conditioned specialisation?** Both, in different senses: a mixture (H4).
    - The computation is reusable: any slot can run it given an engaging code.
    - Acquisition and maintenance of access are slot-specific and driven by each slot's own loss.
19. **Stable across independent fresh models?** The mechanism findings: yes, 6/6 (95% CI 0.54–1.00). Staggering: not at arity 2.
20. **Comparable mechanisms at arity 2 and 3?** Yes, in every evaluable model. Arity 3 staggers more reliably.

**Scientific significance**

21. **Predicted before confirmation:** P1–P5 and PL interference, all met in fresh models.
22. **Predictions that failed:**
    - P0 STAGGERED (MIXED);
    - PL's "no acceleration" was uninformative (ceiling);
    - development's "targeted slot does not acquire" did not generalise to fresh short windows [fals].
23. **Mechanisms ruled out by matched controls:**
    - routing-only access (P4);
    - strong non-transferable per-slot implementations (P2);
    - uniform reuse with no code-specific processing (P5, and slot-specific learning);
    - transfer of evidence (excluded by the relabel's construction).
24. **Does this explain the earlier T4/GATE implementation heterogeneity?** Not tested. Code-keyed early processing suggests one way different seeds could reach different layouts [spec].
25. **What is genuinely novel relative to mechanistic interpretability?**
    - Position-conditioned computation in shared weights is known (e.g. digit-specific streams in addition).
    - What is distinctive here is a **symmetric** task, an **exactly equivariant** architecture, and **alignment-free, source-matched** interventions. In that setting, "the computation exists" and "a position can use it" separate cleanly: transfer by code exchange, coexisting with slot-specific learning [frozen; NOVELTY_AND_PRIOR_EXPLANATIONS.md].
26. **What XVII established about COR's original theory:** that it is silent on this question, and that its source-level form cannot represent relational gates. Nothing in XVII validates O/D.
27. **What remains unproven:**
    - what makes a code engaging;
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
- **Integrity rule.** Monitoring was restricted to the lane log and file existence. No confirmation output was read before scoring.

## Post-review corrections

(Filled in from `reviews/ADJUDICATION.md`.)
