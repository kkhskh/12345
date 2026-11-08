# Open foundational problems

Problems are ordered by mathematical importance and by how much they would change the atlas. Each states what is known, what is missing, and the cheapest decisive test.

**OP-1. What determines off-support extrapolation among on-distribution-equivalent implementations?** (highest leverage)
- **Known.**
  - Ten Bayes-level models obey the same on-support response law (PP1).
  - Off the support they split along a global (configuration-constrained) versus local (factorised per-group) axis (PP2 plus a post hoc reference).
  - DeepSets are consistently global. Transformers vary by seed.
- **Missing.** A predictor of a seed's extrapolation class from its internal structure.
- **Candidate predictors (post hoc, from 7 seeds):**
  - the share of cross-question evidence read through values (seed 12: global);
  - the number of reliability heads (seeds 10 and 11, with three each: local).
- **Decisive test.** Freeze a predictor. Train ≥ 6 new curriculum seeds and classify each one's mechanism on-distribution before it is evaluated off the support.
- **Why it matters.** It is the only result found that Bayes (H1) does not predict. If internal structure predicts extrapolation, a mechanism-level theory has content beyond Bayes.

**OP-2. Learning barrier for T4 from scratch, and the source of the curriculum benefit.**
- **Known.**
  - From scratch: fails at 6k and at 24k steps.
  - From T1: succeeds in 7/7 seeds.
  - From T0: fails.
  - From T1 with the last layer reset: succeeds.
  - Further arms are in `GENERALIZATION_AND_EMERGENCE.md`.
- **Missing.** A mechanistic account of *why* T4 is unreachable from random initialisation. The current hypothesis: the gradient toward any partial relational solution vanishes because "agreement ⇒ reliable" is actively harmful in T4.
- **Decisive test.** Measure the T4 loss gradient's projection onto the T1-learned tag/routing directions at initialisation versus after T1. Add a mixed curriculum (T4 with p_c < 1), where the monotone first step is no longer harmful. The phase diagram predicts where.

**OP-3. When does an explicit reusable relation intermediate form?**
- **Known.**
  - With a latent shared across 6 decisions (R3), a tag forms in most seeds.
  - With the relation consumed once and equal to the target (REL-XOR3, earlier package), no separate intermediate was found.
  - T0 and T5 (no value in the relation): no tag.
- **Missing.** A controlled manipulation of reuse: the number of questions Q, the number of heads and outputs consuming the relation, a multi-output T2.
- **Decisive test.** Vary Q ∈ {2, 4, 8} with matched Bayes gain, and measure tag formation (key-only IIA) prospectively.

**OP-4. Unknown and changing groups.**
- **Known.**
  - The statistical side is settled for pairs: noise re-pairing is unidentifiable and irrelevant (I1). Unknown pairing costs 0.027 accuracy in T1 and ≈ 0 in T4.
  - From scratch, the Transformer fails to learn T1 with permuted sources.
- **Missing.**
  - Whether learning is possible with a curriculum or more capacity.
  - What computation succeeds: clustering-like pairing representations, or on-demand pairwise routing.
  - Statistical theory for groups of size ≥ 3, overlapping groups and hierarchical groups (partition-type statistics).

**OP-5. Natural necessity of the tag representation itself.**
- **Missing.** A removal intervention that does not inject wrong values: mean or conditional-mean ablation of the tag subspace, compared against resample ablation.
- **Decisive test.** Cheap; uses the existing checkpoints.

**OP-6. A representation-free identification method.**
- **Partial result.** The readout attention pattern is basis-free and carries the relation in 6/7 seeds.
- **Missing.** A general operation-level identification that also covers value-path rereading (seed 12), and that is invariant under head splits (Shapley aggregated by pattern class) and Q/K gauge.
- **Theoretical question.** Is there a canonical "routing operation" object, an equivalence class of (pattern, OV-image) pairs, that makes the I5 and I6 transformations act trivially?

**OP-7. Relation families beyond latent reliability.** Not examined with learned models in this step:
- constraint compatibility with reuse;
- multi-hop reconciliation;
- contradictions that must be reconciled;
- semantic and causal compatibility.

  Without these, H7 (task-specificity) cannot be rejected for the *mechanism* level. Only the Bayes-level structure (Theorems S, U, D) is general across the role families examined.

**OP-8. Non-symmetric error models.** When assumption (U) fails (biased sources), counts are not sufficient and values matter. It is unknown whether learned models then keep a count-like tag, which would be suboptimal and a testable bias, or learn value-specific reliability.

**OP-9. Scale.**
- No L-size (d = 128) model, and no width or depth sweep, in this step.
- The mechanism-selection map (tag vs reread vs split) is unknown as a function of capacity.
