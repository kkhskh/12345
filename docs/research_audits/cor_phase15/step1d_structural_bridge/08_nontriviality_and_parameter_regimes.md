# 8. Nontriviality, and the open set

## Diagnostics fixed before reading favorable rows

The output scale `α` multiplies `W_O` and `W2` only. The grid is `{0, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}, 0.1, 1, 10}`. For each row the record contains `rate(α)`, `B_dir`, `B_align`, `B_iso`, `|R|`, `||∇m||`, `∇m·δ`, `||ΔA||_F`, and `||ΔM||_F`, on the unit orthogonal edit at the seed-0 state.

A branch is called *visible* in a row when `||ΔA||_F > 10^{-3}` or `||ΔM||_F > 10^{-3}`. That label is not an exclusion rule. Rows below it stay in the table and in the tests. The label exists so a near-zero branch is not described as a full-size circuit.

## The grid

| `α` | `rate` | `B_dir` | `|R|` | `||∇m||` | `||ΔA||` | `||ΔM||` | `B_dir < ||∇m||` | visible |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.876435 | 0 | 0 | yes | no |
| `10^{-6}` | 0.004623 | 0.004051 | `1.33e-7` | 0.876435 | `2.57e-7` | `4.93e-7` | yes | no |
| `10^{-4}` | 0.480609 | 0.421222 | `1.33e-5` | 0.876441 | `2.57e-5` | `4.93e-5` | yes | no |
| `10^{-3}` | 6.475 | 5.675 | `1.32e-4` | 0.876493 | `2.57e-4` | `4.91e-4` | no | no |
| `10^{-2}` | 231.6 | 203.0 | `1.29e-3` | 0.877011 | `2.57e-3` | `4.76e-3` | no | yes |
| 0.1 | `1.90e4` | `1.67e4` | `9.84e-3` | 0.881938 | 0.0257 | 0.0338 | no | yes |
| 1 | `1.86e6` | `1.63e6` | `1.08e-2` | 0.964532 | 0.2566 | 0.0979 | no | yes |
| 10 | `1.85e8` | `1.63e8` | `8.41e-2` | 4.305 | 2.566 | 1.876 | no | yes |

At unit scale the branches move by tenths, the stacked `Q,K,V` singular value remains `0.789`, and the directional certificate is valid and vastly larger than both `|R|` and `||∇m||`. The favorable comparison with the Cauchy number occupies the rows where the branch increments are below `10^{-4}`. That is a mathematically valid regime and a scientifically thin one. Both descriptions are kept.

## Theorem Persist

Let `rate` be the coefficient in Theorem S, a continuous function of the weights for `ε>0` fixed. Spectral norms are continuous, including at rank changes. `L_1` and `L_2` are continuous in the gains.

On any weights with `rate < 1/2` and `w ≠ 0`, and for every base state `H` and every `δ ⊥ w`,

```text
||∇m(H) - w|| ≤ ||w|| rate,
||∇m(H)|| ≥ ||w|| (1 - rate) > ||w|| rate,
```

so, when `||δ|| > 0`,

```text
B_dir = ||w|| rate ||δ|| < ||∇m(H)|| ||δ||.
```

The derivative bound holds because a Lipschitz constant of the branch readout dominates the operator norm of its derivative, at every state. The set `{rate < 1/2}` is open and nonempty: on the seed-0 ray it is `0 ≤ α < α_*` with

```text
α_* = 1.0387929173812708×10^{-4},
```

the positive root of `rate(α) = 1/2`. The seed-0 point at `α = 10^{-4}` lies in the set and has `σ_min(Q,K,V) = 0.7887 > 0`. The set `{σ_min > 0}` is open. Their intersection is a nonempty open set of weights. On that intersection the strict comparison holds for every `H` and every nonzero readout-orthogonal `δ`, and the edit is nonconstant whenever it is not the zero vector, because `1·w ≠ 0` at seed 0 and the condition `1·w ≠ 0` is itself open.

Twenty multiplicative perturbations of relative size `10^{-3}`, at `α = 10^{-4}`, all stayed inside the inequality. Those twenty points illustrate a box. They are not the proof. Continuity is the proof that some ball exists. The radius of the largest ball was not computed.

The orthogonal family moves with `w`. The quantifiers are "for each weight in the set, for every `δ` orthogonal to that weight's `w`." No continuous selection `δ(w)` has to be fixed in advance.

## What the open set establishes

Local robustness of a strict numerical comparison between a Regime 1 finite bound and the first-order Cauchy number, on a class that includes injective `Q,K,V` maps and nonconstant edits. The comparison is uniform in the base state.

## What it does not establish

It does not establish that trained Transformers lie near this set. An open neighborhood of a nearly zero output-scale model is a small-rate neighborhood. It is not a statement about a training distribution, about density in weight space beyond this sublevel set, or about practical tightness. Failure or success on Gaussian weights is not a theorem about trained circuits. This package runs no trained model and does not pretend the seed-0 draw is one.

It also does not establish that `B_dir` is a better finite-response tool than Regime 4, or than a Regime 2 predictor plus a tight local remainder. At `α = 10^{-6}`, `|∇m · δ|` is already smaller than `|R|`, and `B_dir` is larger than `|R|` by four orders. The open-set win is specifically against `||∇m|| ||δ||`.

## The output-kernel point is a different regime

Theorem Kernel-out gives `|R| = |w·δ|` with order-1 branch spectral norms and visible branch motion (`||ΔA||_F > 0.1`). That point is not an open set. It is the zero locus of `W_O w` and `W2.T w`, which has positive codimension. Its scientific content is that readout-aligned gain, not the full spectral norm, is what enters `R`. A perturbation restores a nonzero branch contribution of size controlled by Theorem Align. Exact cancellation of the branch readout is real, unstable, and still unary: each path misses `w` on its own.
