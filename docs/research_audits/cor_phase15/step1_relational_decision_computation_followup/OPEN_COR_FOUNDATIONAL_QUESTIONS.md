# Open COR foundational questions after Investigation XIII-B

Phase 1.5 Step 1 (provisional ≈ 25%). This is not a closure document. It states no thesis and does not validate COR. No connection to the
original obstruction/discrepancy (O/D) operators is drawn: nothing here gives specific mathematical or empirical evidence about them.

## What XIII-B settled, and at what evidential level

These items are summarised. The evidence is in the four topic documents, and confirmation status is in FROZEN_CONFIRMATION_AND_STATISTICS.

- **[proof]** For G = 3, C = 2, β = 1 and on-support inputs, the two-colluder decision needs no cross-group equality at questions other
  than q. Self-agreement bits plus joint values at q are exactly sufficient.
- **[num]** At β = 0.6 that local observation falls short of Bayes by 3.6 accuracy points; cross-group equality at the other questions
  carries the difference.
- **[dev] "Pre-assembled" models are early-readout models.**
  - Candidate evidence reaches the readout token at layer 0, and relational evidence is pooled into it at layer 1. Layer 2 (mostly its MLP)
    then combines both.
  - Mixture models instead write relational evidence into the q source tokens, and combine by layer-2 attention.
- **[dev] Same weighting variables, different combination exponent.** Both classes use E-based eligibility and reliability weighting. The
  early class combines per-group evidence with a power-mean exponent of about 0.6; mixture models stay close to the Bayes mixture
  (ρ ≈ 0.9). Addendum AQ1 tests this.
- **[dev] Within mixture models, the attention-row pathway is specific.**
  - It carries 68–81% of purely relational effects.
  - It is specific relative to magnitude- and contribution-matched heads.
  - It restores the effect after the layer-1 route is removed, following E rather than A1.
  - It is **not** specific relative to permuted or source-free rows judged by projection alone; those produce large non-specific changes.
- **[dev] Balanced DAS identifies E.** The apparent A1 abstraction is reproduced by source-free patches.
- **[dev] Two-colluder repair at β = 0.6.** Adapting layers 0–1 with layer 2 frozen reaches Bayes and uses the cross-question relation.
  Whether a layer-2-only repair can use it depends on the readout class.

## Open problems, ordered by foundational value

1. **Why does the early-readout implementation settle on ρ ≈ 0.6?**
   - Is it a capacity effect of an MLP combining pooled features at one token? An inductive bias of the architecture? A training-dynamics
     effect?
   - Test with architecture-constrained training (e.g. forbidding layer-0 readout attention) and with MLP-width variation.
   - Mathematically: characterise the accuracy cost of a power-mean combination with ρ < 1 relative to the exact mixture, as a function of
     the reliability distribution. Is ρ ≈ 0.6 near a constrained optimum?
2. **Is the class set by pretraining, and how?** All development seed pairs, and the seed-21 replicates (Q2), point to inheritance from
   the parent. The parent's property that decides it is unknown; the P-count pretraining curriculum and group-level incompleteness are
   candidates. Hybrids (uP164, uP162, vP156) are not explained.
3. **Late-readout repair failure: optimisation or capacity?** A 4000-step layer-2 control on uP161 was queued; see FROZEN_CONFIRMATION
   for whether it finished. A capacity answer would make the late-readout representation a genuine **constraint on downstream structural
   adaptation**: useful information present but not accessible to a one-layer repair. That would be an important COR-level phenomenon.
4. **What does the early01 repair construct?** Layers 0–1 write the cross-question relation into the readout and q tokens that the frozen
   mixture reads. The exact feature constructed, and whether it is J_H^β, n4, or another sufficient statistic, is not identified.
5. **Pathways beyond the attention rows.** In mixture models, MLP2 and other heads carry 19–32% of relational effects. In hybrids they
   dominate. These were not decomposed further.
6. **Decoy specificity is metric-dependent.** Permuted and source-free rows produce large changes with moderate projection on the effect.
   A principled specificity metric for natural-activation transplants is still needed. Fidelity to a discriminating prediction worked
   here, but there is no theory.
7. **Learning transition (D4).** XIII's correlational "lead" was withdrawn. The checkpoint-level response analysis
   (`scripts/traj_resp.py`) was repeatedly OOM-killed. Its status is in FROZEN_CONFIRMATION §6. **No internal predictor of the A2 → B
   transition is established.**

## Broader COR families (not examined in XIII-B)

- **GATE:** the optional comparison was **not run**. The mandatory workstreams used the compute budget, with five 4000-step fine-tunes
  and two-colluder adaptation as the dominant costs.
- **TRANS, adaptive sources, unknown group structure:** not examined.
- **T4-specific versus general.**
  - The sufficiency analysis is specific to this generator family. Its qualitative lesson — shared dependence across sources forces
    cross-group statistics, and the degree of copying decides whether *local* or *cross-question* relations are needed — is a hypothesis
    for other families, not a result.
  - The methodological lessons are candidates for general use: balanced identification sets, source-free nulls for learned subspaces, and
    purely relational edit families to avoid confounding candidate evidence.
