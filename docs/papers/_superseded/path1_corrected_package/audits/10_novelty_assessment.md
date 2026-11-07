# Novelty assessment

**Label:** `NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS`

This is not `NOVELTY_INSUFFICIENT_FOR_CURRENT_PAPER`. It is also not a claim of a new theory of Transformer decisions.

## The skeptical reviewer’s question

> Linear probing is not causally sufficient. Nullspace interventions and activation steering already exist. Othello-GPT causal board-state editing already exists. What is new here?

A precise answer:

Li et al. (2023) showed that Othello-GPT has an internal board that can be read by nonlinear probes and that **changing that board** changes legal-move predictions. Nanda, Lee, and Wattenberg (2023) showed that a **player-relative linear probe** is enough for that reading, and that adding those probe directions steers play. Elazar et al. (2021) and Ravfogel et al. (2020) showed that **removing** linearly readable features is the right causal test of whether a probe tracks used information. Turner-style activation steering adds residual directions without a probe constraint.

Those results already imply that a high-accuracy board probe is not a complete causal account of the network. They do not measure the residual question we actually tested:

**After freezing one linear board probe, do residual directions that leave that probe’s full 64×3 probability vector unchanged still carry target-aligned information about legal-move outputs?**

That is a narrower question. The extra evidence, if the Drive arrays confirm the session record, is:

1. Complete frozen-probe **probability** preservation (not only decoded argmax labels) during the intervention.
2. Different downstream legal-move mass inside that frozen-probe equivalence class.
3. Wrong-target and random-null controls, including a restored protocol that matches **output JS** rather than residual L2, after an earlier COST run that failed its own protocol.
4. A train-only map from **symbolic** target features into the probe nullspace that does not consult \(h_D\) at evaluation, with ablations that attribute much of the Editor A gap to stage-conditioned features and legal-state features.
5. Prespecified failures: joint T/W switching below 60%, near-zero visible/null factorial interaction, and persistently negative absolute \(Y\).

If a reviewer says “this is ordinary activation steering with a different feature representation,” the fair reply is: the representation is defined as the kernel of a frozen board probe, the invariance is the probe’s full probability vector, the target is a different legal-move set, and the negative tests are part of the claim. That combination is not in Li, Nanda, Elazar, or ActAdd as a single result. It is also not a general mechanism.

Two later Othello sources must be in the comparison, without treating them as duplicates.

**Jacopin (BlackboxNLP, October 2026), *Present but Barely Used*.** Reproduces the Li–Nanda linear player-relative board under masked diffusion. The representation is present but weaker, and a Li/Nanda-style board intervention is only barely used (reported mean \(+0.02\), 24% win rate vs a strong AR control). Closest shared idea: readable board \(\neq\) causal use. Difference: Jacopin changes the *training objective* and then applies the original probe-visible board edit. We freeze one autoregressive probe and add a vector in its *kernel* while matching output JS. Not the same experiment.

**Lin, JackS, Karvonen, and Can (2024), *OthelloGPT learned a bag of heuristics*.** Alignment Forum research note, not a peer-reviewed article. Reports localized legality heuristics and stage-tuned neurons. Closest shared idea: legality need not be a single compact board code; stage features can matter. Difference: circuit-level neuron case studies, not a frozen-probe nullspace + matched-JS target-specificity design.

Updated label remains `NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS`. It is **not** priority-proven. The literature review is still not exhaustive of every 2024–2026 Othello note.

## Claim-by-claim

| Claim | Class | Keep as primary? |
|---|---|---|
| A. Probe-probability invariance with output change | POTENTIALLY NOVEL CONTROLLED FINDING | Yes, scoped to this probe |
| B. Target-specific effect at matched JS | POTENTIALLY NOVEL METHOD COMBINATION | Yes, with COST protocol split disclosed |
| C. Donor-free eval-time transfer | POTENTIALLY NOVEL METHOD COMBINATION | Yes, with legal-mask information budget |
| D. Stage/legal limits | descriptive of this map | Yes as a bound, not as a discovery about Transformers |
| E. No reliable complete replacement | negative bound | Yes in main text |

Claim A would collapse into Elazar if we only said “probes are not causal.” The operational difference is **adding** target-conditioned mass in \(\ker W\) while **keeping** the probe output. Claim C would collapse into Nanda if the map used probe directions; it uses the complementary subspace and no donor at eval.

## What is already established

- Othello-GPT has a linear, player-relative, causally relevant board representation (Li 2023; Nanda et al. 2023).
- Probe accuracy is not a causal conclusion (Elazar et al. 2021; Belinkov 2022).
- Nullspace projection can erase linearly readable features (Ravfogel et al. 2020; Belrose et al. 2023).
- Residual steering can change behavior (Turner et al.; Rimsky et al. 2024; Nanda et al. 2023).
- Causal abstraction is a separate, stronger alignment standard (Geiger et al. 2024, 2025). We do not meet it.

## What would make the contribution too weak

If Drive-level recomputation showed that probe probabilities were not actually invariant, or that the equal-JS advantage vanished, the remaining text would be a methods note plus negative results. That would be `NOVELTY_INSUFFICIENT_FOR_CURRENT_PAPER`. Present session letters do not show that collapse. They do show small effects and failed completeness tests.

## Venue implication

A focused mechanistic-interpretability workshop paper or a technical report is the honest default. A full empirical ML conference paper is possible only if reviewers accept a narrow, well-controlled Othello finding with small \(Y\) units and prominent negatives. NeurIPS/ICML/ICLR acceptance is **not** inferred.
