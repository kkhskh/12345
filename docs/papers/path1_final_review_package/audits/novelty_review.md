# Novelty review

This review asks what the paper adds relative to the closest work it could find. It makes no priority claim. The search was not systematic: we checked the works cited in the manuscript, their direct follow-ups that we know of, and targeted web searches on probe-nullspace interventions and Othello-GPT causal studies through late 2026. A systematic search could find closer work. The manuscript's own wording, "We did not find a prior study … That is an observation about the sources we checked, not a priority claim", is the strongest statement this review supports.

## Closest work and the operational difference

| Work | What it does | What differs here |
|---|---|---|
| Li et al. (2023), ICLR | Nonlinear probes recover the Othello-GPT board; interventional edits along the probe change legal-move predictions. | Edits there move the probed representation. Here the probe readout is held fixed and edits lie in its kernel. |
| Nanda, Lee and Wattenberg (2023), BlackboxNLP | A player-relative linear probe reads the board; adding probe directions steers play. | Same model family and probe type. Their edits are in the probe's rowspace; ours are in its nullspace. |
| Hazineh, Zhang and Chiu (2023), arXiv 2310.07582, NeurIPS 2023 SoLaR workshop | Linear board probes across depths and model sizes; layer-wise interventions along the linear representation show causal use in middle layers. | Rowspace interventions at many layers. No kernel edits and no fixed probe readout. Not cited in the manuscript before this review; added to Related work. |
| Jacopin (2026), BlackboxNLP | Linear player-relative board in a masked-diffusion model; board interventions have small, inconsistent effects. | Asks whether a readable board is used under another training objective. No kernel edits. |
| Lin, JackS, Karvonen and Can (2024), Alignment Forum note | Othello-GPT computes legality through many localized heuristics, including stage-sensitive neurons. | A mechanistic account, not a nullspace test. It is consistent with the stage share reported here. Not peer reviewed. |
| Elazar et al. (2021), TACL; Ravfogel et al. (2020), ACL | Remove linearly readable information (INLP) and measure behavior. | They take probe-readable information away. Here it is held fixed and the complement is edited. |
| Ravfogel, Prasad, Linzen and Goldberg (2021), CoNLL (AlterRep) | Builds counterfactual representations by changing the rowspace component of an INLP-found feature subspace while keeping the nullspace component. Measures the change in word predictions. | The closest methodological precedent: an explicit rowspace/nullspace decomposition used for causal counterfactuals. AlterRep alters what the probe reads and keeps the rest. This paper keeps what the probe reads and alters the rest, under target-specific controls. Not cited in the manuscript before this review; added to Related work. |
| Belrose et al. (2023), NeurIPS (LEACE) | Closed-form linear erasure guaranteed against all linear classifiers. | Erasure, not insertion; no target-specific contrast. |
| Geiger et al. (2024, 2025) | Causal abstraction and distributed alignment search. | Fits alignments between causal variables and subspaces. No alignment is fitted here and no faithfulness is claimed. |
| Turner et al. (2023); Zou et al. (2023); Rimsky et al. (2024) | Activation steering with contrastive or learned directions and a strength coefficient. | Here several directions, including wrong-target ones, are matched on output divergence. That is a control, not a new steering method. |
| Hewitt and Liang (2019); Belinkov (2022) | Control tasks for probes; survey of probing limitations. | Methodological background on readout versus use. |

## What may be new, stated without priority

The combination may be new, but its parts are not. The parts are kernel edits of one frozen board probe in Othello-GPT, starting from a probe-edited reference activation, with wrong-target and random kernel controls. They are tested under two different strength controls, matched L2 and matched output divergence. The paper adds a donor-free map from symbolic state into the kernel, evaluated against wrong-target, stage-mean and random directions, and it reports the completeness tests that fail alongside the positive results. Nullspace projection, activation patching, steering, divergence matching and Othello-GPT probes are all established.

## What is not new

- That probes are not causal summaries (Elazar et al., 2021; Belinkov, 2022).
- That Othello-GPT's board features are causally relevant (Li et al., 2023; Nanda et al., 2023; Hazineh et al., 2023).
- That a rowspace/nullspace split can define causal counterfactuals (Ravfogel et al., 2021).
- That a readable world model may be only weakly used (Jacopin, 2026).

## Risks to the contribution

- The effect at matched divergence is a few hundredths of probability mass, and mean \(Y\) stays negative. Readers may judge the finding to be of limited size.
- The edits start from an Editor A activation, which may be off the natural manifold. A reviewer can argue that the kernel effect partly reflects repair of Editor A's damage. The manuscript states this as a limitation.
- One model, one layer and one probe. A nonlinear or later-layer decoder might read the edit.

## Recommendation for the manuscript

Add Ravfogel et al. (2021) and Hazineh et al. (2023) to Related work, with the operational differences above. Keep the existing non-priority sentence. Do not use "first", "novel" or "for the first time" anywhere. The verifier's forbidden-phrase list should not be relied on for this; the scientific sentence audit checks it by hand.
