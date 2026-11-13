# Investigation IX — main research synthesis

**Native relational computation, causal mechanism emergence, generalization, and mathematical limits.**

**Evidence levels.**
- **[C]** confirmed under frozen predictions: Part I at commit `1ee69c2`, Addendum A1 at `55eae73`. The independent
  pretrained parents are yP130–135, under the Stage-0 freeze, with a new evaluation stream; the generalization readers
  use seeds 10–12.
- **[F]** frozen prediction falsified.
- **[D]** development only.
- **[P]** proved (elementary).
- **[N]** numerically verified.

## Central findings

1. **The original P-count R1 failure is not an information failure [C, CP1–CP3].**
   - In every completely pretrained parent, linear probes on the frozen states recover every count and every report.
     Fed into the exact rule, they give exact Bayes (KL ≤ 0.0001).
   - Yet the original transfer still fails R1 on 4/4 (E_R1 −0.76 to −1.02).
2. **Two failures had been superimposed in earlier investigations [C, CP2].**
   - P-count pretraining is bimodal: about a third of lineages never learn one group's count.
   - That "incompleteness" perfectly separates the original transfer's KL, which is information loss.
   - It does not explain R1, which fails in complete parents as much or more.
3. **The original block computes a regime-blind routing rule, and its frozen prefix carries no cross-group information
   [C partly; CP5 falsified as frozen].**
   - Attention routing carries ≥ 50 % of the R2/R3 responses on 6/6 parents.
   - The newly perfect group loses attention by the same amount in every regime.
   - Frozen pretrained layers transmit ≈ 0 of an edit to other groups' token states (NP3, 4/4).
   - The quantitative "perfect-group mass < 0.6 × n=4 mass" criterion held in only 3/6.
4. **The original Transformer can learn the cross-group operation natively [C, NP1, 4/4 independent parents].**
   - Full fine-tuning in the original architecture, on raw tokens only with no side channels, cuts per-episode R1
     error 3–5×: 0.21–0.35 vs 0.81–1.06.
   - It cuts R2/R3 error 3–6×, to KL 0.002–0.007.
   - This is **Type III** native learning. Bypass is impossible by construction.
5. **The native repair recurs behaviourally but not mechanistically [F, NP2 0/4].**
   - On the development parent, a clean causal pathway carried the R1 credit: the other groups' source tokens at q,
     necessary and sufficient.
   - On independent parents the credit sits elsewhere, and differently on each: other groups' source states, readout
     states, or distributed.
   - What recurs is dependence on the downstream attention pattern (freeze removes the credit 4/4) and the
     frozen-prefix limitation.
6. **Learning order is not universal in native models [F, NP4 1/4].** "Separable discount first, cross-group credit
   later" is clear on 2 of 5 fine-tuned parents, late on 1, and absent on 2.
7. **Mathematics [P, N].**
   - **Theorem 1:** for the role model with C colluders, p ∝ C(S,C)·A − C(S−1,C−1)·B. It is exact for any G and C. The
     only cross-group quantity is S, the number of eligible groups.
   - **Theorem 2:** the interaction order of the evidence function is exactly 1 + C. Latent reliability (C = 0) is
     additive. The colluder creates the order-2 interaction behind R1.
   - **Theorem 3:** partial copying turns the hard gate into soft per-group likelihood ratios.
   - **Sufficiency:** within-group counts suffice for T4 but **not** for adaptive "echo" sources, which need cross-group
     joint statistics.
   - **Counterexample:** finite-domain sum decompositions are universal, so no general impossibility follows for pooled
     readers.
8. **Generalization [C, GP1–GP4].**
   - Identity-free readers generalize to an unseen eligible pair; the concatenation MLP does not.
   - No learned reader extrapolates to more groups, despite exact representability.
   - Partial copying breaks all readers, as Theorem 3 predicts.
9. **Falsified also [F, CP4].** A native-state MLP reader of frozen states, with labels at 8k steps, does not reliably
   integrate the relation (1/4). It does beat the original on every parent.

## Answers to the 38 final questions

### Native computation

1. **What causes the R1 failure?**
   - The original downstream block, sitting on a frozen prefix that never propagates information across groups, learns
     a regime-blind routing rule: a newly perfect group loses attention whatever the other groups say. By Prop. A of
     Investigation VIII, such a rule must give R1 and R3 the same sign.
   - Incomplete pretraining adds a separate information loss on about a third of lineages.
2. **Does the native representation contain enough information?** Yes, in complete parents: exact Bayes by linear
   decoding [C]. No, in incomplete ones: a group's count is missing [C].
3. **Is it accessible to realistic downstream computations?**
   - Linearly, yes (decode-then-compose).
   - To a learned 2-layer MLP on the 7 relevant tokens at 8k label steps, only partly [F CP4]. Bayes-level with Bayes
     targets and 32k steps on one dev parent [D].
   - To the original one-block downstream, no.
4. **Can the original Transformer learn the operation without engineered inputs?** Yes, with full fine-tuning: 4/4
   independent parents [C].
5. **Which adaptation methods succeed or fail?**
   - Succeed: full fine-tuning [C].
   - Fail: frozen prefix with one or two fresh blocks (C1 [C]; keep3 and deep2 [D]).
   - Partial: Investigation VIII's single-parent unfreeze1, LoRA and deep2.
   - Not run here: LoRA by component, layer-0 unfreezing, curricula, raw-token from-scratch.
6. **How do you tell native computation from a bypass?**
   - By information: native conditions see raw tokens only.
   - By intervention: natural token-state patching, plus parent-state replacement for adapters (Investigation VIII).
   - The Type I–V table is in `NATIVE_RELATIONAL_LEARNING.md` §3.
7. **Which internal pathways transmit the information?**
   - Original model: readout-token perfectness information, routed by the downstream attention [C partial].
   - Fine-tuned models: parent-specific pathways, with a common dependence on downstream attention [C/F].
   - On one dev parent: the other groups' source tokens at q, as a necessary and sufficient set [D; did not recur].

### Learning

8. **When does relational information become available?** For complete parents, before T4 training: counts and
   perfectness are decodable from the pretrained states [C].
9. **When does it begin affecting decisions correctly?**
   - In full fine-tuning, typically after the KL drop: steps ≈ 800–2700 on xP81/yP130, after 1800 on yP131, early on
     yP132/134.
   - Never within 4000 steps with a frozen prefix [C/D].
10. **Why can a decodable feature remain unused?**
    - The downstream can only use what it can route. Per-group perfectness is routed as a per-group weight, which is an
      order-1 computation. The decision needs an order-2 one (Theorem 2).
    - The cross-group term must be carried into every group's evidence, and the frozen prefix provides no such
      carrier [C].
11. **What distinguishes successful attention, pairwise and MLP implementations?**
    - All can represent the rule.
    - Identity-free pooling or attention gives exact permutation invariance and held-out-pair generalization. The
      concatenation MLP memorizes configurations [C].
    - None extrapolates in size [C].
12. **Which conditions change the outcome?**
    - Whether the prefix is trainable: the decisive factor [C].
    - Pretraining completeness, for KL [C].
    - Training objective and budget, for frozen-state readers [D/F].
    - Clean counts make the shortcut stronger [D].
13. **Does the learned computation recur across parents and seeds?**
    - Behaviour: yes [C].
    - Token-level mechanism: no [F].
    - Attention-dependence: yes [C, exploratory measurement].

### Generalization

14. **Unseen group-pair identities?** Yes for identity-free readers; no for the concatenation MLP [C].
15. **Permutation invariance?** Exact by construction for identity-free readers; approximate (0.016–0.027) with ids
    [D].
16. **More groups?** No learned reader extrapolates (KL 0.025–0.053 at G = 4) [C].
17. **Changes in reliability or dependence?** Partial copying breaks all readers [C]. Adaptive echo was analysed
    mathematically only [N].
18. **Within vs off support.**
    - The held-out pair (within support) passes.
    - Semantic shift (within input support, new Bayes rule) fails.
    - Size shift (off the training arity) fails.
19. **Composition with other relational computations?** Not tested. It is the top open item for generalization.

### Mathematics

20. **Sufficient statistics?**
    - Per-group counts plus target reports, or equivalently (S, per-group e_g, W_g, L_g) [P].
    - Under adaptive echo, cross-group echo counts must be added [N].
21. **When do separable computations lose information?**
    - Whenever the evidence function's interaction order exceeds what the aggregation provides. Additive aggregation
      handles order 1 only [P].
    - Caveat: finite-domain universality of sum decompositions with unrestricted ρ [P/K].
22. **Sufficient interaction structures?** Pooled per-group features with a degree-(1 + C) polynomial ρ, for β ≤ 1
    with soft gates [P]. Adaptive echo needs pairwise (joint) features [N].
23. **Mathematically necessary resource restrictions?** Only within restricted classes:
    - evidence-form polynomial ρ of degree < 1 + C is impossible [P];
    - additive logits are impossible (Prop. C).

    No general lower bound is claimed.
24. **Properties that survive alternative implementations?**
    - The behavioural R1/R2/R3 pattern and the Bayes sufficient statistic.
    - The order-2 requirement.
    - Not token-level pathways [C/F].
25. **Common abstraction of native pathways?** Only "attention-dependent cross-group credit" [C]. A causal-abstraction
    test (interchange-trained maps) was not run.
26. **Strongest counterexamples?**
    - NP2: a necessary-and-sufficient pathway on one parent is unnecessary on four others.
    - Exact representability without learned extrapolation (GP3).
27. **Conditional theorems with testable predictions?**
    - Theorem 2 predicts the R1 boundary exists only with a colluder. It predicts order-3 behaviour for C = 2 (not
      tested).
    - Theorem 3 predicts the failure under partial copying, which was confirmed (GP4).

### Broader COR

28. **T4 reliability routing?** It is the order-1 case. R1 eligibility is an order-2 extension that per-group
    reliability routing cannot implement [P]. That is exactly the original block's failure [C].
29. **GATE?** Also an order-2 selector, but per-question, from non-answer sources. GATE's routing was causally
    unnecessary. P-count C1's routing is causally used and wrong for R1. There is no shared mechanism below behaviour.
30. **TRANS?** Nothing new. It remains unresolved and separate.
31. **Which phenomena need distinct classes?**
    - Order 1: reliability.
    - Order 2 pooled: eligibility, partial copying.
    - Order 2 pairwise: adaptive echo.
    - Per-question multiplexing: GATE.
    - TRANS: unclassified.
32. **Any justified link to O/D?** No (`COR_ORIGINAL_FORMALISM.md`).
33. **What is explained by existing methods?**
    - All the mathematics: Bayesian model averaging, sum decompositions, ANOVA.
    - Inductive-bias generalization.
    - Decodable ≠ used.
34. **What genuinely new contribution remains?**
    - The completeness correction.
    - The confirmed separation of information availability from frozen-prefix limitation and downstream shortcut.
    - Native Type III repair on independent parents.
    - Documented non-recurrence of a causal pathway across parents.
    - Elementary but explicit classification theorems.
35. **Confirmed:** CP1, CP2, CP3, GP1–GP4, NP1, NP3.
36. **Falsified:** CP4, CP5 (as frozen), NP2, NP4.
37. **Unexplored territory:**
    - head-level mechanisms;
    - cross-parent causal abstraction;
    - LoRA and curricula on complete parents;
    - mixed-G and mixed-β training;
    - C = 2 colluders;
    - learned adaptive-source models;
    - TRANS;
    - O/D.
38. **Highest-leverage next experiments.**
    1. Interchange-intervention causal-abstraction maps across the 5 fine-tuned parents, for S / m_g.
    2. Head- and MLP-level path patching inside the fine-tuned layers 0–1, along the trajectory.
    3. A budget × objective grid for frozen-state readers, to settle CP4.
    4. Mixed-G training, to test whether size extrapolation can be learned.

## Disclosures

- **Compute.** Two CPU cores. Background jobs are stopped a few minutes after each reply ends, which killed several runs.
  This was first misattributed to environment restarts. All lost confirmation runs were rerun with identical commands.
- **Not run:** the development items listed in `NATIVE_RELATIONAL_LEARNING.md` §1.
- **Reviews:** by the same agent in separate passes (`REVIEWS_AND_ADJUDICATION.md`). They are not independent.
- **Scope.** This investigation does not end Step 1, choose a thesis, or claim a COR law.
