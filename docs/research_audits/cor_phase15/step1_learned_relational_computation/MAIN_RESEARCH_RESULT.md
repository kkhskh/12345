# Investigation VIII — main research result

**Learning relational computations: from available evidence to internal causal decision mechanisms.**
T4 generator; base commit Investigation VII `9b4f56f`; protocol frozen at `4dd9d34` (+ addendum `29f7569`) before any
confirmation run. Tags: [P] proved (elementary), [N] numerical, [C] confirmed under the frozen protocol, [D] development
only, [X] exploratory after the freeze.

## Summary of findings

1. **[C] The cross-group eligibility relation is learnable from raw evidence with ordinary labels, given a relational
   reader.** Readers that see only per-group counts and target-question reports, with no m and no combined flag, reach
   Bayes on R1, R2 and R3 together. This holds for group attention and pairwise relation networks at 16k steps: 6/6 new
   seeds, mean |E_R1| 0.10, KL 0.0008 (P1). A concatenation MLP fails at the same budget (E_R1 −0.39 to −0.52; P2a),
   and succeeds at 48k steps or with Bayes soft targets [D]. A linear reader fails even when m is given [D].

2. **[C] Learning goes through a separable phase first.** Every learner without m first adopts the P-count sign: at 1k
   steps E_R1 is between −1.2 and −1.8, in 9/9 new runs. The cross-group gate appears only later. A learner given m does
   not go through this phase (−0.03) (P2b, c). The P-count failure of Investigations VI–VII therefore *looks like* this
   early phase, frozen in place.

3. **[C] In the learned attention readers, the relation is implemented by attention to eligible partner groups, and that
   pathway is identified causally (P3, 3/3 new seeds).**
   - Blocking attention to eligible partners recreates the P-count failure selectively: E_R1 ≈ −3.6, removing a perfect
     partner *raises* the margin (+3.4), and F34 is unchanged (|Δ| ≤ 0.02).
   - Self-only attention makes the response regime-independent, which is exactly what Proposition C [P] says any rule
     additive over groups must do.
   - An m-subspace interchange carries the effect (share 0.93–1.48 vs random −0.40 to −0.04).

4. **[C] Decodable is not the same as used (P7).** Failed MLPs decode m from their last hidden layer at 0.99, *better*
   than the successful attention readers do (0.83–0.87).

5. **[C] Without hand-computed m, a learned relational adapter corrects the original failure on 8/8 new P-count
   lineages (P4, P5).**
   - E_R1 goes from a mean of −0.74 to 0.057 (sign p = 0.004).
   - Every preservation metric improves: R2∪R3 |E| 1.31 → 0.15; KL 0.094 → 0.002.
   - A matched generic adapter does nothing (0.49).
   - The oracle m-injection is erratic (+0.36 / −0.28).
   - On T1 the criterion passes, but R1 is not better than T1's mild original error (P6, with a caveat).

6. **[X] That "repair" bypasses the parent representation.**
   - With the adapter zeroed, the repaired model is at chance (acc 0.33–0.35) on all 8 conf lineages.
   - With the parent's state replaced by its batch mean, nothing changes (KL 0.002).

   The successful learned repair is a learned relational reader of the raw evidence placed beside the frozen P-count
   model, which it ignores. **Whether the P-count representation itself can be made to support the relation is not
   established.** On one development parent (xP80, exploratory), unfreezing its last pretrained layer *partially* helps
   (E_R1 −0.90 → −0.31, KL 0.104 → 0.056). LoRA on that layer (−0.44) and a second fresh block (−0.55) help less;
   see §Native.

7. **[D] Compositional generalization is partial.** One configuration was held out: the two perfect groups being {0,1}.
   - A lookup table collapses there (E_R1 −3.92).
   - The pairwise reader keeps the R1 sign (−0.14), attention loses most of it (−0.42), and the MLP fails (−2.75).
   - Group-permutation and candidate-relabelling invariances hold approximately for the relational readers only.

8. **[D] From ordinary tokens (Condition X), from-scratch Transformers learned nothing relational in 6000 steps.** Both the
   1-block and the 3-block model had dm ≈ 0 on every edit family and KL ≈ 0.28–0.30. The longer runs were stopped for
   compute. This is a budget-limited negative, not evidence of impossibility.

## Answers to the 24 final questions

1. **Can a Transformer learn the relation from ordinary inputs?**
   - From per-group ingredient tokens: yes. Group self-attention, 1–2 blocks [C].
   - From the raw episode token sequence, from scratch: not within 6000 steps [D/budget-limited]. Undetermined at larger
     budgets.
   - The pretrained P-count Transformer, with its original downstream, does not [C, P4].
2. **From ordinary decision supervision alone?** Yes, with labels only (P1). No auxiliary m signal and no soft targets
   were used.
3. **What changes with Bayesian or relational auxiliary supervision?**
   - Bayes soft targets shorten training: the MLP goes from −0.46 to +0.01 at 16k [D]. This is variance reduction, with no
     new information.
   - An auxiliary m head on the frozen-feature transfer made things *worse* (E_R1 −1.10 vs −0.90) [D].
4. **Is the relevant information present in the P-count representations?** Decodably yes (Investigation VII: m is
   probe-decodable). Accessible to a one-block downstream: no. This investigation added no new information-theoretic
   test of the P-count states. The successful repair does not use them (finding 6).
5. **What architecture can access and combine it?** Readers with a multiplicative group-by-group interaction:
   attention across groups, or pairwise φ summed over partners. By Prop. C, additive pooling cannot. A concatenation MLP
   can, given 3× the steps.
6. **Why does the original one-block downstream fail?** Established at two levels:
   - (a) any per-group or additive-over-groups readout cannot produce the sign pattern [P];
   - (b) learners pass through a separable solution first [C].

   Whether the one-block downstream on frozen P-count states fails because the states do not expose per-group
   (n, report) in a form attention can gate on, or because of optimisation, is **not separated**. The partial success of
   unfreezing layer 1 [X] points to a representational component.
7. **Does additional training solve it?**
   - Composition readers: yes for the MLP (48k).
   - Condition Z: Investigation VII found C1 at 12k still at −0.70; here the 4k orig is at −0.90 (xP80). More training of
     the downstream alone does not solve it on the evidence available.
8. **Does selective unfreezing solve it?** Partially, on one development parent [X]:
   - unfreezing the last pretrained layer gives E_R1 −0.90 → −0.31 and KL 0.104 → 0.056;
   - rank-8 LoRA gives −0.44;
   - two fresh blocks give −0.55.

   No variant reaches Bayes within 4000 steps. Not confirmed on new lineages.
9. **Does a generic nonlinear adapter match the hand-computed-m repair?** No. gadapt ≈ orig [C]. The learned *relational*
   adapter beats both: m-injection overshoots or undershoots, while radapt reaches mean |E_R1| 0.06 [C]. It does so by
   bypassing the parent (finding 6).
10. **Does the learned model implement the eligibility operation or an equivalent?** In the attention reader, it
    implements a function of eligibility: attention to partners that are perfect and agree. Its group states encode m
    decodably and causally (interchange). Whether it represents m *as such* or an equivalent re-encoding (for example the
    product RHO^n·m) is not distinguishable: share ≠ 1.
11. **Can the internal computation be identified causally?** At the pathway level, yes (attention to eligible partners:
    necessary, specific, preserving F34) [C]. At the subspace level, partially (the share overshoots).
12. **Necessary, sufficient, redundant, correlated?**
    - Necessary: cross-group attention (self-only removes the regime dependence) and attention to eligible partners
      (noperf).
    - Not sufficient: attention only to eligible partners breaks R2/R3 (noother); other groups' counts are needed.
    - Correlated only: linear decodability of m (P7).
    - Redundant in the repaired model: the entire parent representation (finding 6).
13. **Does a learned repair fix R1 without damaging R2/R3?** Yes [C]. R2∪R3 improves 1.31 → 0.15.
14. **Does it improve distance to Bayes, not just a signed mean?** Yes: the decision-level |E| ratio is 0.22 and KL drops
    40-fold [C]. The residual decision-level R1 |E| is still 0.15–0.23.
15. **New independent parents?** Yes: 8/8 new P-count lineages pretrained under the Stage-0 freeze, plus 4 T1 lineages
    (with the caveat) [C].
16. **Does the relation generalize across group identities and held-out configurations?** Partially. Only the pairwise
    reader keeps the sign on the held-out pair [D, one seed]. An id-free variant was not tested.
17. **Symmetries?** Approximately for the relational readers: group permutation |Δlogit| 0.016–0.024, candidate
    relabelling 0.04–0.06, on a logit scale of about 1.28. Not for the MLP or lookup [D].
18. **Which restrictions explain success and failure?**
    - Architecture: is there a multiplicative cross-group interaction (Prop. C)?
    - Budget, interacting with the separable-first dynamics.
    - Target variance (soft targets).
    - Whether the reader has direct per-group access to (n, report). The frozen P-count states plus one block do not
      provide it in a usable form.
19. **Can it be a theorem?** Only the elementary ones:
    - Prop. C: additive-over-groups logit rules are regime-independent;
    - Prop. A: log-additive aggregation is separable;
    - representability of the closed form by group attention.

    No accessibility theorem relative to (Z, reader, budget) was obtained. Two simple candidates were refuted by
    counterexamples (`FORMAL_RESULTS_AND_COUNTEREXAMPLES.md`).
20. **Common causal computation across implementations?** Behaviourally, every successful model satisfies the same
    non-separable response constraint. Mechanistically, only the attention family was dissected. The relnet (pairwise)
    and the repaired models were not. Cross-model interchange was not run. **Not identified.**
21. **What remains ordinary Bayesian approximation or generic function learning?**
    - The closed-form posterior itself.
    - The fact that relational inductive biases learn relational functions faster.
    - Soft targets reducing variance.
    - The learned "repair", once the bypass is recognised.
22. **What nontrivial causal-decision relationship has been discovered?** Within this task and architecture class, there
    is a confirmed, noncircular chain:
    - a mathematically defined relation (m: eligibility of other groups) is
    - learned from data that does not contain it,
    - into a specific pathway (attention to eligible partners), whose
    - removal causes the *predicted* sign-level decision error (the separable discount) and
    - spares a matched non-relational response (F34).

    Separately, the same separable error appears as a confirmed transient in every learner without m. This is the
    mandate's "noncircular connection between a mathematically defined relation, a learned internal operation, and its
    causal effect on a candidate decision", established for small set-Transformer readers. It is **not** established
    inside the pretrained P-count Transformer.
23. **Evidence still needed for a broader COR theory.**
    - The same chain identified inside a Transformer reading raw tokens (Condition X at a larger budget).
    - Native modification of the P-count model confirmed on new lineages.
    - Cross-implementation causal equivalence (relnet vs attention).
    - Generator variations (K, number of groups).
    - A theorem about the separable-first dynamics.

    No bridge to COR's obstruction or discrepancy operators was found. **None is claimed.**
24. **Corrections, weakenings, strengthenings.**
    - *Corrected:* the mandate's m′ is not the posterior's factor (math §1).
    - *Weakened:*
      - "learned repair of the P-count model" becomes "a learned bypass that does not need m" (finding 6);
      - P6 on T1: no R1 gain;
      - Investigation VII's m-injection repair overshoots or undershoots on new lineages (+0.36 / −0.28).
    - *Strengthened:*
      - Investigation VII's "decodable but not accessible" now has a confirmed learned-model analogue (P7);
      - the P-count sign error is now tied to a confirmed learning-dynamics phase and to a causal pathway whose removal
        reproduces it.

## Native modification of the P-count parent (exploratory, xP80, development eval)

| spec | E_R1 | KL |
|---|---|---|
| orig | −0.90 | 0.104 |
| unfreeze1 | −0.31 | 0.056 |
| lora1 | −0.44 | 0.066 |
| deep2 | −0.55 | 0.061 |
| radapt (bypass, for reference) | −0.02 | 0.002 |

Details are in `SELECTIVE_LEARNED_REPAIR.md`.

## Scientific outcome classification (mandate §19)

- **A** (emerges from ordinary task learning): for relational readers on ingredient tokens, yes, including generalization
  only partially (F). Not for from-scratch token Transformers in the budget run.
- **B** (learnable only with a stronger architecture): yes, for the frozen P-count setting.
- **C** (learned and causally identified): yes, within the reader class.
- **D** (learned repair succeeds on new parents): yes by the criteria, but by bypass.
- **F** (shortcut / partial generalization): partly.
- **G** (bottleneck remains for the P-count representation itself): yes.
- **H** (multiple implementations): suggested (attention under-shoots and relnet over-shoots, systematically), not
  dissected.

**This investigation does not choose a paper thesis and does not assess readiness or close the stage.**
