# 02 — Search-space taxonomy

Eight relationship kinds, kept distinct.

| Kind | Meaning | Example of a false merge |
|---|---|---|
| Representational | Two sites encode related coordinates | Alignment of probes is not causal use (Elazar et al., 2021; Direction 1) |
| Computational | One site is a function of another | Exact `c=f(p)` makes a matching residual vanish (Theorem Z) |
| Causal | `do` on one site changes a descendant | A stitch is not this |
| Predictive | A frozen statistic forecasts a later `R` | Fitting the statistic to `R` is not a forecast |
| Coordination | Several paths jointly determine an output | Vector agreement is not joint causal role |
| Stability | The output stays put under a named perturbation family | Stability is not correctness |
| Abstraction | A lower-dimensional description preserves some `do`-effects | Unrestricted alignments can be vacuous (Sutter et al., 2025) |
| Correctness | Agreement with an external target | `D=0` can be the wrong answer (CE-04) |

A model can be computationally consistent and wrong. Two features can be representationally aligned and causally unused. Two paths can disagree as vectors while the linear readout is stable, because they cancel on `w`. A representation can be sufficient for interventions in its rowspace and insufficient for interventions in its kernel (`test_sufficiency_is_relative_to_an_intervention_family`).

These distinctions are the coverage grid. They are not eight theories.

## Principled coverage matrix (not the full Cartesian product)

Meaningful and Transformer-relevant, treated:

| Structure | Relation | Outcome | Source | Status |
|---|---|---|---|---|
| Linear residual sum | path contribution | `ΔL`, ablation | architecture + `w` | PATH, attribution |
| Residual then nonlinearity | interaction | `I` | architecture | I-LIN / I-SQ |
| Attention mix | approximate transport | leftover | architecture | not constant `ρ` |
| LayerNorm | metric | Euclidean scores | architecture | LN |
| Projection `P` | sufficiency | kernel `ΔL` | frozen `P` | N2, Direction 1 |
| Pairwise `ρ` | equality | residual | analyst | wrong aggregator |
| Discrete majority | correction | Hamming | constructed | not a block primitive |
| Any `Q` invariant on `ker Q` | prediction of `sign ΔL` | sign | unrestricted `L` | N1, impossible |

Ruled out as universal: coboundary residual → decision improvement.

Equivalent: H-SUF with causal abstraction; H-COMPOSE with Hessian/mediation.

Stronger assumptions: `L=q(P)` ; next map linear; majority decoder present.

Distinctive possible prediction that survived as architecture-only: `I=0` when later readout is linear in residual addends. That prediction is also just linearity.

Not filled as separate cells: recurrent iteration, learned multi-head closed forms, synergy information measures as causal proofs. Listed as unexplored in file 27.
