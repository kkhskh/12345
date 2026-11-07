# 17 — Counterexamples

Each row is a specified map. The executable checks are the named tests. The floats were produced by `numpy.random.default_rng` under NumPy 1.26.4, in the draw order of `make_params`. A floating-point value is an output of that stream and this implementation. It is not a universal constant. Another NumPy build can change the stream and fail a hardcoded comparison without changing the identities.

## 1. The branch misses a shift. The block does not

Seed 1, `ε=10^{-5}`, `a=0.7`, token 1. `H_2` changes by `a1` at that token only, to `10^{-12}`. The readout at token 2 has response `0`. The readout at token 1 would have response `a 1^T w`. Test: `test_shift_is_exact_and_scale_is_not_when_epsilon_is_positive`.

## 2. Scale is not a shift when `ε>0`

Same seed. Replacing token 1 by twice itself moves `H_2` by a vector whose difference from the pure skip has max-norm `0.05220912`. Test: the same function, threshold `10^{-2}`.

## 3. Value-only has the wrong sign

Seed 16, token 1,

\[
δ=(0.01206307677054339,\ -0.8987843131414323,\ 0.00171581962876215,\ -0.22913906964580683).
\]

Total directional derivative `+0.02019538349844552`. Value-only directional derivative `−0.008580691798845158`. Test: `test_value_only_linearization_can_have_the_wrong_sign`.

## 4. First-order cancellation, nonzero finite response

Seed 0, token 1. `g≈(1.93178219, −2.10943103, −1.39327748, 1.57092632)`. Unit `u ∝ (g_1,−g_0,0,0)`. Directional derivative about `−1.1×10^{-16}`. At step `0.2`, `R=−0.629265931`. At step `10^{-3}`, `R/step²=−14.0474795`. Test: `test_zero_first_order_response_need_not_be_a_zero_finite_response`.

This is also the example in which a path decomposition of `Dm` sums to zero and the finite intervention does not. The decomposition is the chain rule. The finite gap is the remainder.

## 5. Same projection, opposite responses

Seed 16. `H'` equals `H` except `H_{0,3}` increased by `2`. Deleting feature 3 leaves the states equal. Token-1 intervention `δ=(0.4,−1.2,0.3,0.5)` has responses `−0.121187908` and `+0.108445402`. Test: `test_same_projection_opposite_intervention_signs`.

## 6. The structural equation holds and the margins differ

Seed 0 margin `0.464472198`. Seed 2 margin `−1.356476385`. On both, recomputed attention matches the stored attention output to `10^{-12}`. The equation's residual is zero in both cases. It does not select the sign of `m`. Test: `test_exact_equations_hold_while_margins_differ_across_inputs`.

## 7. A fitted transport residual moves when the decision does not

Seed 1. Constant shift `0.7·1` at token 1. The two margins agree to `10^{-16}`. The rank-one map built from token 0 of `H` and token 0 of `H_1`,

\[
ρ = H_1[0]\, H[0]^T / \|H[0]\|^2,
\]

has residual `1.25789191` on the base state and `1.79690659` on the shifted state. The residual is the error of `ρ`. The intervention response is zero. Test: `test_rank_one_transport_can_disagree_when_the_margin_does_not`.

## 8. A multi-path identity that is only the derivative

The sum of the skip increment, the attention increment, and the MLP increment is `dH_2`. Setting the readout of that sum to zero is `Dm=0`. Row 4 is a case. No extra constraint appears. Register row R19.

## 9. What each failed statistic is not

On row 5 the projection is constant across the two states and the finite response is not. The quantity that does separate them is the full state, or any measurement that keeps `H_{0,3}`. On row 6 the structural residual is zero on both seeds and the margins have opposite signs. The quantity that carries the sign is `m`. On row 3 the value-only directional derivative has the opposite sign from `Dm`. In none of these rows is the failed statistic equal to `Dm` or to the constant-shift evaluation. Row 5 is a finite recomputation, not a claim that `Dm` equals `R`. Row 6 is a comparison of two inputs, not an intervention response.

## 10. Frozen pattern is a different intervention

Row 3 is the case. With `P` clamped, the attention output is affine in `V` before the second normalization: the homogeneous part is `P V` times `W_O`, and `b_O` does not depend on `V`. After `LN_2` and `tanh`, the map to `m` is not that affine map. The clamp is not the derivative of the recomputed block. The identity is valid inside the attention clamp and false as a description of `R`.

## 11. Both-necessary is not a nonlinear interaction

`b=−1.5`, `c_1=c_2=1`. Both deletions leave `−0.5`. The map `(u,v)↦b+u+v` has interaction `0`. The zero-baseline pair `1+(−0.5)` does not make both deletions necessary. Tests: `test_biased_addition_makes_both_necessary_with_zero_interaction`, `test_zero_bias_addition_cannot_make_both_deletions_necessary`.

## 12. Heads, finite responses

Seed 4, token 1, `δ=(0.2,−0.4,0.1,0.3)`. Joint response `0.162853943`. Single-head responses `0.059199942` and `0.111275110`. The attention directional derivatives at the original `X` do add. The finite leave-one-out responses do not. Test: `test_head_derivatives_add_and_token_edits_need_not`.

## 13. Two token edits interact

Same seed. Edits `(0.3,−0.2,0.4,0.1)` at token 1 and `(−0.2,0.5,0.1,−0.4)` at token 2. `I=−0.062808329`. Same test.

## What was not a counterexample

A constant intervention cannot separate states, because `R` does not depend on `H`. An early search that used `δ=1` and found no gap was this identity. It is not evidence that every projection is sufficient.

A loose singular-value cutoff is not an algebraic counterexample. On `h=(1,0.2,−0.4,0.7)`, `ε=10^{-5}`, tolerance `10^{-3}` reports rank 2 and tolerance `10^{-8}` reports rank 3. The algebraic rank is 3. Test: `test_a_loose_singular_value_cutoff_misreads_small_epsilon`.
