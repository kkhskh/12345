# 04 — Gauge, representation, and identifiability

## 1. Change of hidden-state coordinates

Let `s'_v = A_v s_v` with each `A_v` invertible. For the coboundary residual to represent the same abstract relation `ρ s_u = s_v`, the maps must transform as

```text
ρ'_{u→v} = A_v ρ_{u→v} A_u^{-1}.
```

Then `d' s' = A_1 (d s)` in the appropriate block-diagonal action `A_1` on `C^1`. Euclidean `O(s)=‖ds‖^2` is **not** invariant unless each `A_v` is orthogonal. To preserve numerical `O` and `D`:

```text
W' = A_1^{-T} W A_1^{-1}
M' = A_0^{-T} M A_0^{-1}
```

where `A_0=⊕_v A_v`. Status: PROVEN UNDER STATED ASSUMPTIONS (standard change-of-basis of quadratic forms). Euclidean COR as implemented is **not** invariant.

Verified: `test_attack_d_coordinate_rescaling_without_metric_transform` (scalar two-node; rescaling the head coordinate by 2 multiplies Euclidean `O` by 4; the inverse-square weight restores `O`).

## 2. Candidate-ranking invariance

Even with a well-typed `d_c` for each candidate, `D_{c1}<D_{c2}` is not preserved under independently allowed reparameterizations or metric choices (Attack C). Sufficient extra structure would be: a single ambient inner product on activations, jointly transforming all candidates, with `P` and `ρ` derived from that structure by a frozen rule that does not see evaluation outcomes. The manuscripts do not supply that rule.

Status: PROVEN FALSE BY COUNTEREXAMPLE that ranking is intrinsic.

## 3. Probe coordinate dependence (Othello 08c)

The frozen board probe has rank 124 in 512-D residual space, nullspace 388. That split is:

- invariant under residual automorphisms that preserve the probe’s row space as a measured subspace, i.e. changes of basis that conjugate the contrast matrix `W` as `W ↦ W A^{-1}` so `ker(W)` maps to `ker(W A^{-1})`;
- **not** an intrinsic decomposition of “all causally relevant information.”

The orthogonal `P_R ⊕ P_N` split depends on the Euclidean inner product used to take SVD of the probe contrasts. A different inner product yields a different complementary subspace with the same kernel of `W`. Probe-invisible causal effects (SPEC, NST) show that `ker W` is causally nonempty relative to that measurement. They do **not** produce a canonical COR germ. The later factorial that independently set rowspace and nullspace target identity was negative (`TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED`).

## 4. Identifiability attacks

### Arbitrary projection (Attack F)

Unconstrained `P` may be 0, a single convenient feature, the candidate label, or the observed answer. All are compatible with Definition 2. Constraint that blocks it: an independent, frozen, outcome-blind rule (e.g. train-only ridge to a simulator state). That rule is a **protocol**, not part of the definition.

### Arbitrary compatibility (Attack E)

Given nonzero `s_u` and observed `s_v` of compatible dimensions, an unconstrained linear `ρ` can satisfy `ρ s_u = s_v` whenever `s_u ≠ 0` (rank-1 outer product). Then `ds=0` on that example. If maps may depend on `x`, this can be done per input. Shared linear `ρ` across a spanning set of `s_u` is constrained; if the germs span, `ρ` is determined as a least-squares map, which is exactly “statistical alignment,” not causal transport.

Verified: `test_attack_e_unconstrained_rho_zeros_observed_discrepancy`.

### Outcome leakage (Attack / protocol)

`P`, `ρ`, rank, `M`, `W`, `α,β,γ`, site set, and intervention scale can all be chosen using `Δm` from the same evaluation. The 2D score `S = -ΔO + β Δm - γ Cost` with large `β` is then mostly attribution. `paper_claims_after_2d.md` already records that the selected obstruction-using score had `β=2`.

Minimum split required:

1. freeze `K`, ranks, `P`, `ρ`, `M`, `W` on training-clean data;
2. freeze the obstruction formula;
3. freeze the prediction (including whether it uses `∇m`);
4. evaluate on disjoint interventions/outcomes.

Without (3), including `Δm` in `S` makes “obstruction predicts repair” near-tautological.

### Observational equivalence (Attacks A, J)

Same `(s,ρ,O,D)`, opposite logits: give two systems the same internal germs and different output heads (Attack A). Same logits, different `O`: change `ρ` scale (Attack J). Therefore obstruction does not determine `F_θ`, and `F_θ` does not determine obstruction.

### Trivial graph (Attack G and empty edges)

No edges: `O=D=0` identically (`test_empty_edge_set_makes_O_zero_and_D_zero`). Duplicating an edge doubles Euclidean `O` while `D` is unchanged (`test_attack_g_edge_duplication_changes_O_not_necessarily_D`). Inserting identity maps, merging nodes, or changing granularity changes the score without changing `F_θ`. The theory depends on an arbitrary computation-complex choice.

## 5. What version is invariant?

| Version | Invariant? |
|---|---|
| Euclidean `O` in probe/PCA coordinates | NO |
| `O`,`D` with correctly transformed `(M,W,ρ)` | YES, as abstract quadratic forms of the same constraint residual |
| Candidate ranking under independent per-candidate gauges or metrics | NO |
| Graph refinement | NOT CLAIMED, and empirically false for raw `O` |

The scientifically used version in IOI/factual/Othello code is the non-invariant Euclidean residual.
