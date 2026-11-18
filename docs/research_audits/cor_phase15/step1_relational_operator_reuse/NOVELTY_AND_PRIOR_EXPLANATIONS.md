# Novelty and prior explanations (bounded audit, mandate §13)

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). The claimed separation of "the computation exists" from "a position can use it" is withdrawn (W3). What remains distinctive is an alignment-free, source-matched demonstration of label-driven transfer in an exactly equivariant model. The mechanism is ordinary conditional computation. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

This is a focused comparison, not a literature review. It names the strongest existing explanations for what XVII observes, and says what would and would not be new.

## 1. The closest prior work

| work | what it shows | relation to XVII |
|---|---|---|
| Quirke & Barez, *Understanding Addition in Transformers*, ICLR 2024 (arXiv:2310.13121) | A one-layer Transformer trained on n-digit addition splits the task into parallel, digit-specific streams, with varied algorithms for different digit positions | **The closest precedent for position-specific computation in a shared-weight model.** Their positions differ in what the task requires (carries), and their model uses positional embeddings. In XVII every slot faces an identical, exchangeable task, so any slot difference is a property of the learned network, not of the task (Theorem P) |
| Olsson et al., *In-context Learning and Induction Heads*, 2022 (arXiv:2209.11895) | Induction heads form in a sharp phase change that coincides with a jump in in-context learning | The precedent for abrupt circuit formation. XVI/XVII show the GATE transition is abrupt **per slot** and staggered across slots that share weights |
| Geiger et al., *Causal Abstractions of Neural Networks*, NeurIPS 2021; Geiger et al. 2023–2025 (interchange interventions, IIA) | Interchange interventions (activation patching) test whether neural representations play the causal role of high-level variables | XVII's relabel intervention is an interchange intervention on the **slot-identity input** within a layer window. It needs no fitted alignment map, because Theorem P makes the identity correspondence between slots canonical |
| Sutter, Minder, Hofmann & Pimentel, NeurIPS 2025 (arXiv:2507.08802), "non-linear representation dilemma" | With unrestricted alignment maps, any network can be aligned to any algorithm | The reason XVII fits no alignment maps. Every cross-slot correspondence is the identity given by the architecture's symmetry |
| McLeish et al., *Transformers Can Do Arithmetic with the Right Embeddings* (Abacus), 2024 (arXiv:2405.17399) | Positional embeddings that align corresponding digits enable length generalisation | Shows that the position code determines whether shared computation is usable at a position. That is consistent with XVII's code-keyed access at the level of whole-model behaviour |
| Conditional computation and superposition (Elhage et al. 2021, 2022) | Shared weights implement input-conditioned computations, and features share directions | **Known phenomenon:** shared parameters can support position-conditioned behaviour. XVII does not claim this as new |

## 2. What is not new

- That shared weights can behave differently at different positions, given different position codes. That is standard conditional computation.
- That circuits form abruptly during training (phase changes).
- That activation and attention interchange can localise causal roles.

## 3. What could be distinctive (pending confirmation)

1. **A clean separation, in a symmetric task, between "the computation exists in the shared weights" and "a position can use it".**
   - The task is identical at every slot, and the architecture is exactly slot-permutation equivariant (Theorem P, a proof).
   - So a partial-acquisition state cannot be explained by different requirements at different slots, which is a key difference from the per-digit addition streams.
   - Development showed that the unacquired slot can run the acquired slot's computation when it is given that slot's code in the first two layers only, with no content moved.
   - Whether that holds in fresh models is P2.
2. **Code keying is not routing alone** (P4, development). Transplanting attention routing does not transfer gate use; exchanging the code does. Access here is a code-dependent transformation, not just an attention choice.
3. **Training on one slot does not speed up that slot, and degrades the others** (PL, development). That is direct evidence that the slots' competences are coupled through the shared weights, yet not interchangeable by local training pressure.

Whether these survive fresh confirmation is recorded in FROZEN_CONFIRMATION_AND_GENERALIZATION.md. If they do, the contribution is methodological and empirical: a symmetric, exactly specified setting in which reuse versus access can be tested with source-matched interventions that need no alignment. They are not a new theory. **None of them bears on the COR O/D framework, which is silent here** (ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md).
