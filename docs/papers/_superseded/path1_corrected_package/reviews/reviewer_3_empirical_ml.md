# Simulated review 3 — critical empirical ML reviewer

Internal simulation, not a conference review.

## Main summary

A careful Othello-GPT study with small matched-JS effects, a negative switching gate, and a negative factorial. The writing is scoped. The community question is whether that combination is a paper.

## Strongest positive point

The authors did not hide the negatives or convert \(\Delta Y=+0.02\) into a percentage win. The COST split is honest. That is rarer than it should be.

## Major weaknesses

1. **Significance versus interest.** After matching JS, the headline movement is about 0.02–0.03 of probability mass, absolute \(Y\) still \(-0.58\), and the completeness tests fail. A main-conference reader may conclude the method does not work in the sense they care about (make the model play the other board).
2. **Single checkpoint, synthetic game, simulator labels.** No second model, no chess, no language.
3. **Baselines.** Editor A is already a failed editor. Oracle donor is almost identical in L2 to the learned direction, which makes the behavioral gap versus Editor A look more like “nullspace donor beating a bad editor” than a new algorithm beating the right baseline (the natural donor).
4. **Reproducibility in this package.** Tables come from a JSON ledger, not from logits. That would not pass a checklist.
5. **Novelty.** Combination of known tools. Workshop-scale unless the invariance result is rock solid and surprising to Othello specialists.

## Minor weaknesses

No PDF compilation guaranteed. Authors missing. Figure 2 is a blocker card. Secondary JS targets never run.

## Potentially fatal objection

“Why should the broader ML community care, given Li, Nanda, Elazar, and activation steering?”

The only honest answer is: specialists who treat a high-accuracy Othello probe as a complete causal board should update that the kernel of *this* probe still moves \(Y\), a little, in a target-dependent way. That is a workshop correction, not a new capability. If the authors inflate it, reject. If they keep it small, the contribution is real and modest.

## Required changes

Level 1 verification. Lead with matched-JS and negatives, not unmatched SPEC 0.39. Venue: interpretability workshop or technical report, not an implicit NeurIPS claim.

## Evidence supporting the criticism

NST \(Y_{\mathrm{learned}}=-0.582\). FRESH both-positive 13/32. VNF \(I_T,I_W\sim 0.001\). Li ICLR 2023 and Nanda BlackboxNLP 2023 already give causal boards.

## Would the current paper be defensible?

Defensible as a focused workshop paper after array verification. Not defensible as a broad empirical ML conference paper on current evidence. The draft’s own novelty label (`NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS`) matches this review.
