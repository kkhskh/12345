# Fibers, relations, and recomputed traces

## Exact sufficiency

Let `R(h,\delta)` be a response and `z=P(h)`. There is a function `g` with

\[
R(h,\delta)=g(P(h),\delta)
\]

for every `h` in a set `H` and every `\delta` in a family `\Delta`, if and only if `P(h)=P(h')` implies `R(h,\delta)=R(h',\delta)` for every such pair and every `\delta\in\Delta`.

The reverse direction defines `g(z,\delta)` as the common value of `R` on the fiber. The forward direction is substitution. No architecture has been used. The theorem says what sufficiency is. It does not produce a sufficient `P`.

## A projection that drops a key feature

`P` keeps coordinate 0. `W_Q` and `W_K` read coordinate 1. `W_V` reads coordinate 0. The states

\[
X_a=\begin{pmatrix}1&0\\0&0\end{pmatrix},\qquad
X_b=\begin{pmatrix}1&0\\0&4\end{pmatrix}
\]

have the same projected coordinates. Their value vectors agree. Their token-1 patterns do not. Token-1 outputs are `1/2` and `1/(1+e^{16})`. The second value is about `1.125\times 10^{-7}`. The test's absolute tolerance of `10^{-6}` accepts it and does not make it zero. Same measurement, different recomputed attention output.

Checked by `test_same_projected_residual_different_attention`.

## A relation is a function of the tuple

For any `ψ`, the number `ψ(z_u,z_v)` is determined by the pair `(z_u,z_v)`. If two pairs share `ψ` and the response does not, `ψ` is not sufficient for that response. The pairs `(0,0)` and `(1,1)` share the difference `0` and do not share the second coordinate.

Checked by `test_a_relation_collapses_pairs_that_the_tuple_separates`.

Joint sufficiency of `(P_1 h, P_2 h)` is sufficiency of the concatenated measurement. It does not transfer to a scalar function of that concatenation.

## Recomputed structural residual

Define the child by the structural equation after the parent intervention: `h_v := f(h_u)`. The residual `h_v-f(h_u)` is zero for every intervention in this family, by that definition. A quantity that is constant on a family cannot separate two different responses on that family.

`\tanh` recomputed after a parent shift has residual zero at every shift used in the test, and the responses at `+0.5` and `-0.5` differ.

Checked by `test_recomputed_structural_residual_is_identically_zero`.

Clamping the child to a value other than `f(h_u)` is a different intervention. The residual can then be nonzero because the analyst overrode the equation. That nonzero number records the override. It does not record a conflict inside the unclamped computation.

## One-point transport

Given one pair with `z_u\neq 0`, `\rho=z_v z_u^\top/(z_u^\top z_u)` satisfies `\rho z_u=z_v`. The residual at that pair is zero because `\rho` was solved to make it zero. A second target with the same `z_u` is not transported to itself by this `\rho`.

Checked by `test_manufactured_compatibility_zeroes_one_pair`.

Fixing `\rho` independently of the pair removes this particular manufacture. It does not by itself imply anything about a downstream response. That implication is a further theorem, and none is proved here for a residual of this form.
