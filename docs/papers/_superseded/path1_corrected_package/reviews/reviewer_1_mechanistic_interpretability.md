# Simulated review 1 — mechanistic interpretability specialist

This is an internal adversarial simulation, not a conference review.

## Main summary

The paper asks whether a frozen linear Othello board probe is a sufficient statistic for legal-move outputs. The interesting operationalization is adding a residual in \(\ker W\) while keeping the full 64×3 probe probabilities fixed. That is not Nanda-style probe-direction steering, and it is not Li-style board rewriting. Negative switching and factorial results are in the main text, which is the right scientific instinct.

## Strongest positive point

Complete probe-probability invariance plus wrong-target controls is a real distinction from “we steered the board features.” If the Drive arrays confirm invariance, the kernel claim is interpretable.

## Major weaknesses

1. The large SPEC \(\Delta_{\mathrm{specific}}\approx0.39\) is mostly a comparison against random directions that barely change JS. Without the later equal-JS restoration, a specialist would dismiss SPEC as unmatched steering.
2. Donor-free still receives the true legal mask. That is closer to “compile the rules into a residual nudge” than to discovering a latent world model.
3. \(Y\) remaining \(\approx-0.58\) means the model does not behave like the target board. Calling this “transfer” needs constant discipline; the draft mostly has it, the title still might be read as a positive world-model paper.
4. No circuit, no layer causal graph, no check that a different probe would also miss the edit. “Probe-invisible” will be misread.

## Minor weaknesses

Rank 124/388 not locally recomputed. Editor A is a weak baseline (already NEGATIVE). An earlier draft treated Pearson \(r\approx 0.998\) on raw \(Y\) as a direction correlation; the corrected manuscript reports mean cosine 0.792 versus oracle and 0.850 versus mean-state.

## Potentially fatal objection

“Is this more than ordinary activation steering with a different feature representation?”

If the only new piece is “we projected the steering vector onto \(\ker W\),” the paper is a methods note on Nanda. The authors’ reply has to be the invariance of the *entire* probe distribution plus matched-JS wrong-target gaps plus the failed completeness tests. If invariance is not verified from saved 64×3 arrays, the fatal objection stands.

## Required changes

- Recompute probe-probability tensors from NPZ (Level 1).
- Put equal-JS and negative absolute \(Y\) in the first results figure, not only SPEC’s large unmatched number.
- State in one sentence that the legal mask is an input.

## Evidence supporting the criticism

Nanda et al. 2023 already steer Othello-GPT with linear board features. Elazar et al. 2021 already separate probing from causal use. Session record shows random JS 0.038 versus correct 0.279 at equal L2.

## Would the current paper be defensible?

Conditionally. Defensible as a narrow empirical note if Level 1 invariance and JS numbers check out. Not defensible as a new mechanism of Transformer decisions. The draft does not claim that mechanism, which is why this review is not a desk-reject on overclaim—yet the missing arrays are a reproducibility reject risk.
