# Prior work and the narrow contribution, if any

**Status.** Written before the confirmation outcomes, as part of the pre-freeze novelty check (mandate §2 Check C and §10). Reviewer B's search (`reviews/PRE_REVIEW_B_DESIGN_NOVELTY.md` §4) and mine overlap. Reviewer B is an agent, not a human peer reviewer. The final paragraph is completed after the results (§4).

## 1. What is already established

| finding | closest work | intervention | does it separate forward information from learning signal? |
|---|---|---|---|
| A temporary early deficit in one of two parallel pathways makes the other pathway win the race for every feature. Early deficits matter, late ones do not, and the effect grows with depth. | Kleinman, Achille & Soatto, *Critical learning periods emerge even in deep linear networks*, ICLR 2024 ([arXiv 2308.12221](https://arxiv.org/html/2308.12221v2)) | `.detach()` on the deprived pathway's output: the **forward signal is kept and learning is blocked** (their App. A.1). The text describes the deficit as also blocking input; the mechanism is the gradient block. | **No.** Gradient-only, in deep linear networks. |
| Briefly suppressing an attention structure (syntactic attention) early in MLM training lets an alternative strategy be learned. Release timing matters; brief suppression can improve the final model. | Chen, Shwartz-Ziv, Cho, Leavitt & Saphra, *Sudden drops in the loss*, ICLR 2024 ([arXiv 2309.07311](https://arxiv.org/abs/2309.07311v6)) | A regulariser on attention between syntactically linked tokens, acting through the gradient. | **No.** |
| Early deficits that change low-level statistics have permanent effects; deficits that preserve them do not. | Achille, Rovere & Soatto, *Critical learning periods in deep networks*, ICLR 2019 ([arXiv 1711.08856](https://arxiv.org/abs/1711.08856v3)) | Input deficits (blur, flips) | **No.** Input-level. |
| Causal manipulation of activations and attention *during* training dissects the formation of induction circuits, and competition plus cooperation between ICL and in-weights strategies. | Singh et al., ICML 2024 ([arXiv 2404.07129](https://arxiv.org/pdf/2404.07129)); Singh et al., ICML 2025 ([arXiv 2503.05631](https://arxiv.org/html/2503.05631v2)) | Clamping activations and weights during training | **No.** Clamping changes both components. |
| Masking gradients during training decides where a computation forms. | Cloud et al., *Gradient routing*, 2024 ([arXiv 2410.04332](https://arxiv.org/abs/2410.04332v2)) | Data-dependent gradient masks | Gradient-only, used as a tool. |
| Networks select among several competent algorithms through initialisation or hyperparameters, and candidate representations compete during training. | Zhong et al., NeurIPS 2023 (clock/pizza, [arXiv 2306.17844](https://arxiv.org/abs/2306.17844)); Ding et al. 2024 ([arXiv 2405.17420](https://arxiv.org/html/2405.17420v1)); Park et al., ICLR 2025 ([arXiv 2412.01003](https://arxiv.org/abs/2412.01003v4)) | Observational, or hyperparameter changes | — |
| A strong learning signal through one feature starves others. | Pezeshki et al., *Gradient starvation*, NeurIPS 2021 ([arXiv 2011.09468](https://arxiv.org/abs/2011.09468v4)) | Theory and experiments | Conceptually the H2g family |
| Attention is not a reliable readout of what a model uses under training-time attention pressure. | Pruthi et al., ACL 2020 ([arXiv 1909.07913](https://arxiv.org/abs/1909.07913v1)) | Attention penalties | — (motivates XVIII's attention-free diagnostics) |

**Consequence.** The broad statements are established in spirit, and XVIII does not claim them as new:
- "temporary restriction of a pathway during training selects between competent computations";
- "early pathway deficits decide a race".

XIV's T4 result is an instance of this known phenomenon in a small transformer on a relational task.

## 2. What XVIII asks that the closest work does not

In a transformer with two competent relational organisations, the question is which component of a training-time attention restriction does the selecting. The candidates, each manipulated alone with the others held fixed, are:
- **(i)** the forward information through the restricted edges;
- **(ii)** the learning signal through them (routing gradient, value gradient, or both);
- **(iii)** the establishment of the route's attention share.

Each candidate is set by an intervention that leaves the others fixed:

| candidate | arm |
|---|---|
| forward content | vrg: content destroyed, mass intact |
| learning signal | gsg and gsv: forward intact, learning signal removed |
| attention share | sha: share clamped, with real or destroyed content |

The outcome is scored by **test-time route necessity** together with a lapse-aware behavioural fit, not by an attention statistic.

**Searches.** A focused search (both reviewers' and mine, November 2026) found no study that decomposes a single training-time attention restriction this way. This is an absence in a focused search, not proof of novelty.

**The sharpest contrast is with Kleinman et al.** Their deficit is gradient-only, with the forward pass kept, and it decides the race in deep linear networks. XVIII's gsv is the like-for-like analogue in a transformer.

## 3. What would not be a contribution

- **fst flipping.** The forward-only block, fst, flips in development, but that is close to guaranteed: its forward pass equals the block's (pre-review A, A2).
- **Graded suppression flipping.** This answers mandate P1 descriptively. By lemma L2 it cannot discriminate between hypotheses.
- **Generality beyond the bistable parents tested.**

## 4. The contribution, after the results

(Completed after scoring; see MAIN_RESEARCH_SYNTHESIS.md.)
