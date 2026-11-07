# 07 — Family A: causal sufficiency

**Hypothesis H-SUF.** A specified representation `z = P(h)` determines `L` for a specified intervention family.

**Precise splits.** Predictive sufficiency (a regression from `z` to `L` on natural data) is not causal sufficiency. Interventional equivalence (`L` agrees on all `do` in a family whenever `z` agrees) is the causal-abstraction property. Local sufficiency at one `h` is not global sufficiency. Candidate-specific `P^c` that contains the logit of `c` is sufficiency by construction and is circular as an explanation.

**Constructive result.** Joint sufficiency can hold when each factor fails (proof N2). The joint object is the tuple, not `||z_1 - ρ z_2||`.

**Impossibility nearby.** If `P` drops a coordinate that `w` reads, kernel edits move `L` at constant `z`. That is the probe-preserving edit. Approximate sufficiency then admits a bound `|ΔL| ≤ K ||(I-Π_P)δ||` only after a Lipschitz constant of `L` on the complement, which is ordinary sensitivity on the dropped subspace.

**Does a relation among parts add anything?** In the additive example, `L = z_1 + z_2`, the relation `z_1 - z_2` is independent of `L`. Adding it does not tighten a bound on `L`. In a majority-vote aggregator, the relation among bits is the computation. That aggregator is not a residual stream.

**Circular construction.** Choosing `P` by maximizing interchange accuracy on the same interventions later used as evidence is the DAS objective. It can be a legitimate estimator of an abstraction. It is not an independent forecast of those interventions (Geiger et al., 2024; Sutter et al., 2025).

**Role of the quantity.** `P(h)` is a researcher-defined summary unless an interchange test shows it mediates `L`. Even then the mediator is `z`, not a coboundary on `z`.

**Distinctiveness.** This family is causal abstraction. `REDUNDANT_WITH_EXISTING_THEORY` as a COR bridge. `PROVED_UNDER_RESTRICTED_ASSUMPTIONS` as the joint-versus-single lemma.

**Othello.** A frozen board probe is not sufficient for kernel residual edits. That is a sufficiency failure, used here as a constraint, not as COR validation.
