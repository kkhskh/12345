# Open foundational research map

Ordered by how much uncertainty each item would remove.

1. **Why do native repairs differ in mechanism across parents? Is there an abstraction under which they are
   equivalent?**
   - *Evidence so far.* Behaviour recurs (NP1, 4/4). Token-level pathways do not (NP2, 0/4). Attention-dependence
     recurs.
   - *Next step.* Learn causal-abstraction maps (interchange-intervention training) between parents for a candidate
     high-level variable such as S or m_g. The test is whether a single high-level model is faithful across all 5
     fine-tuned models. This is the most direct test of mandate E5.
2. **Inside the fine-tuned layers 0–1: head-, MLP- and LayerNorm-level decomposition.** None was done. Path patching per
   head, with per-checkpoint tracking along the trajectory, would show whether an S-like variable is formed and where.
3. **Can frozen native states support learned integration cheaply?**
   - CP4 was falsified at 8k label steps, while the development reader reached Bayes with Bayes targets at 32k.
   - *Next step.* A budget × objective grid on independent parents, to distinguish an optimization limit from an
     objective effect.
4. **Learning dynamics.** Separable-first is not universal (NP4, 1/4). Two open questions:
   - What determines the trajectory type? Candidates are pretraining specifics and the initial readout.
   - What happens around the correction? It needs denser checkpoints.
5. **Pretraining completeness.** Why is P-count pretraining bimodal, and does completing it (longer pretraining) change
   the native fine-tuning trajectory or mechanism? It is cheap to test.
6. **Generalization of learned relational operations.**
   - Size extrapolation fails despite exact representability (GP3). Does training on mixed G fix it?
   - Mixed-β training for soft gates.
   - C = 2 colluders: Theorem 2 predicts interaction order 3.
   - Generalization of *native* models: needs a variable-G tokenisation.
7. **Adaptive sources.** Cross-group echo statistics are needed (numerical result). Two open steps:
   - learn a model and locate the pairwise dependence;
   - prove the pooled-vs-pairwise separation (rank of the echo kernel).
8. **TRANS.** No Investigation IX work. It remains a separate, unresolved composition problem.
9. **LoRA, partial unfreezing, extra blocks, raw-token from-scratch, curriculum** on complete parents: not run (see
   `NATIVE_RELATIONAL_LEARNING.md`).
10. **The original O/D formalism.** No connection found (`COR_ORIGINAL_FORMALISM.md`). Progress requires an O/D definition
    on these generators, fixed before outcomes are seen.
