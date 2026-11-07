# 18 — Competing explanations

The question was whether a relation among pathways constrains `R` without already being a derivative or a value of `m`. The competitors below are the calculations that already do that job on the families where anything was constrained.

## The total derivative

`Dm·E_{t,δ}` is the first-order response. It uses the weights, the state, and `δ`. Its sign is the linear sign. Seed 16 shows that dropping the `dQ`/`dK` piece can flip that sign. Seed 0 shows that the linear sign does not decide a finite step in a direction orthogonal to `g`. The derivative explains exactly the linear term. It does not need a second name.

## The remainder

The finite gap along the seed-0 direction is a higher-order term. Estimating it is a Hessian-vector product or a second evaluation of `m`. Either budget is larger than the first derivative, not a reduction. Calling the gap a consistency failure adds no equation: the structural equations hold on both the base state and the moved state.

## The frozen-pattern value path

Clamping `P` and moving `V` is a path-specific intervention. Elhage et al. (2021) isolate that path by freezing the pattern. The resulting map is affine in `V` before the second normalization. The bias `b_O` is constant in `V`. After `LN_2` and `tanh` the map is not affine, and it is not the derivative of the recomputed block. Path patching already estimates this intervention by running the patched computation. The closed form at the attention output is that run, differentiated. It is not a residual of a compatibility map.

## Causal abstraction

Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard (JMLR 2025) require a high-level variable to correspond to the low-level intervention on the experimental set. Seed 16 is a projection that fails that correspondence for one intervention. The failure is the definition of "not an abstraction of this response." Distributed alignment search (Geiger, Wu, Potts, Icard, Goodman, CLeaR 2024) learns a subspace. This step does not. Sutter, Minder, Hofmann, and Pimentel (NeurIPS 2025) show that unconstrained alignment maps make the abstraction relation vacuous. The rank-one `ρ` solved from one token is a one-pair instance of fitting the map to the pair. It is not a new proof of their theorem.

## The constant-shift evaluation

For `δ=a1`, `R` equals `a 1^T w` or `0` according to the read token. The computation uses the architecture and the readout. It does not use `W_Q`, `W_K`, `W_V`, `W_O`, or the MLP. That is a real reduction of the budget, and it is the content of per-token mean subtraction plus a residual add. Xiong et al. (ICML 2020) distinguish this placement from post-norm. They analyze gradient scale at initialization. They do not need to be cited for the one-line shift identity, and their warm-up theorem is not a result of this step.

A post-norm block, which normalizes after the add, does not obey the same identity. The identity is architectural. It is not a score that can be attached to an arbitrary compatibility relation.

## What does not compete

`O` and `D` from Phase 1 are not functions of this block until a projection and a quadratic form are chosen. The structural residual of the true equations is identically zero, so it does not compete with `Dm` as a predictor. A nonzero surrogate residual competes only as an approximation error. Row 7 of the catalog is a case where that error changes and the decision does not.

No examined alternative predicts the seed-16 sign from less information than `Dm`, and none predicts the seed-0 finite sign from the first derivative alone.
