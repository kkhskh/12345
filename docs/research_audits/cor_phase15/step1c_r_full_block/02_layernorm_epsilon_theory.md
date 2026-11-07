# 02 — LayerNorm at positive epsilon

Let `C = I - 11^T/d` and `z = C h`. Set `v = z^T z / d` and `s = sqrt(v+ε)`. With a gain and a bias,

\[
\operatorname{LN}_ε(h) = γ \odot \frac{z}{s} + β.
\]

`C` is the orthogonal projector onto `1^⊥`. It satisfies `C^2=C`, `C^T=C`, and `C1=0`. The vector `z` is already centered, so `Cz=z` and `z^T 1 = 0`.

## Derivation of the Jacobian

Differentiate `z` and `v` in a direction `δ`:

\[
dz = Cδ, \qquad dv = \frac{2}{d} z^T δ.
\]

The second line uses `z^T C = z^T`. Then `ds = dv/(2s) = z^T δ / (d s)`.

The normalized vector is `u = z/s`. Its derivative is

\begin{align*}
du
&= \frac{s\, dz - z\, ds}{s^2}
= \frac{Cδ}{s} - \frac{z (z^T δ)}{d s^3}.
\end{align*}

The gain is a constant diagonal. Therefore

\[
J_{\operatorname{LN}_ε}(h)
= \operatorname{diag}(γ)
\left(
\frac{C}{s} - \frac{z z^T}{d s^3}
\right).
\]

SymPy simplifies the Jacobian of `z/sqrt(v+ε)` on three symbolic coordinates to this bracket. Central differences match the same matrix at `ε ∈ {0, 10^{-5}, 10^{-2}}` for a five-dimensional vector with a nontrivial gain. The bias does not enter the Jacobian.

## Shift and scale

Replace `h` by `h + a 1`. Then `C(h+a1)=z` and `v` is unchanged, for every `ε≥0` and every `a`. The gain and bias do not disturb that identity. Shift invariance is exact.

Replace `h` by `α h` with `α>0`. Then `z` becomes `α z` and `v` becomes `α^2 v`, so

\[
\frac{α z}{\sqrt{α^2 v + ε}}
= \frac{z}{\sqrt{v + ε/α^2}}.
\]

The identity scale `α=1` leaves both the normalized vector and the full map unchanged for every `ε≥0`. For `d≥2`, some gain nonzero, and `α>0` with `α≠1`, the full map satisfies `LN_ε(αh)=LN_ε(h)` for every admissible `h` if and only if `ε=0`. The normalized vector, which does not involve `γ`, obeys that same biconditional for `d≥2` and `α≠1` even when the gain hypothesis is dropped. At `ε>0` the normalized vector already fails to match for any `h` with `z≠0`. If every gain is zero, `LN_ε` is the bias for every `ε`, so the full map is unchanged by scaling even though the normalized vector is not. For `ε>0` the comparison is approximate when `α^2 v` dominates `ε`. In dimension `1` the centered vector is always `0`: a nonzero gain still leaves the map equal to the bias, and there is no scale to check. The numerical block below rejects scale invariance at `ε=10^{-5}`: doubling one token moves the block output by a vector that is not the skip alone. The gap is larger than `10^{-2}`.

## Kernel and rank

Write `M` for the bracket, and assume `s>0`, which is true whenever `ε>0`.

`M1 = 0`, because both terms kill `1`.

Conversely, suppose `Mδ=0`. Then `Cδ = λ z` with `λ = (z^T δ)/(d s^2)`. Taking the inner product with `z` also gives `λ = (z^T δ)/(d v)` when `v≠0`. If `z^T δ ≠ 0`, the two expressions for `λ` force `ε=0`. For `ε>0` one concludes `z^T δ = 0` and then `Cδ=0`, so `δ` is a multiple of `1`. If `v=0`, then `z=0` and `M=C/s`, whose kernel is again `span{1}`.

So for `ε>0` the kernel of `M` is exactly `span{1}`.

If every entry of `γ` is nonzero, `diag(γ)` is invertible and

\[
\ker J = \operatorname{span}\{1\}, \qquad \operatorname{rank} J = d-1.
\]

For `ε=0` and `v>0`, the same calculation allows `Cδ` to be the projection of `δ` onto `z`. Decomposing `δ = μ 1 + α z + w` with `w` orthogonal to both `1` and `z` forces `w=0`. The kernel is `span{1, z}` and, when `γ` is invertible, the rank is `d-2`.

`M` maps into `1^⊥`, because `C` and `z` do. For `ε>0` the kernel is one-dimensional, so the image is all of `1^⊥`. A vector in `1^⊥` supported on one coordinate is zero. If exactly one gain is zero and the others are nonzero, `diag(γ)` stays injective on that image and the rank remains `d-1`. If exactly two gains are zero and the others are nonzero, the vectors in `1^⊥` supported on those two coordinates are the multiples of `e_i-e_j`, and the rank is `d-2`. On the checked vector of dimension 4, one zero gain leaves numerical rank 3 and two zero gains leave numerical rank 2, at tolerance `10^{-8}`.

## Numerical rank is not the algebraic rank

On `h=(1, 0.2, -0.4, 0.7)` with `ε=10^{-5}` and `γ=1`, the identity `J z = ε z / s^3` holds to `10^{-12}`, and `||J z||` is about `7.1×10^{-5}`. The singular values are about `1.883`, `1.883`, `6.68×10^{-5}`, and `0`. Tolerance `10^{-8}` reports rank 3. Tolerance `10^{-3}` reports rank 2. The second report is the epsilon-zero count applied to a matrix whose radial singular value happens to be small. Step 1C's uniform bound "rank at most `d-2`" is the algebraic fact only at `ε=0`.

## The skip

Let `y = h + A(LN_ε(h))`, and let the attention map depend on `h` only through `LN_ε(h)`. A finite shift `h ↦ h + a 1` leaves the branch exactly unchanged and adds `a 1` through the skip. The perturbation is invisible to the normalization and visible to `y`.

The same holds tokenwise in the block of `04`: LayerNorm is applied per token. Adding `a 1` to one token leaves both normalizations of that token unchanged, leaves every attention and MLP branch unchanged, and adds `a 1` to that token's output. A readout that does not read that token has response `0`. A readout `w` on that token has response `a 1^T w`. No attention or MLP weight is used. The identity is exact at `ε=10^{-5}` in the tests, not merely first-order.

A centered radial step is different. At `ε=0` and `v>0` it is a second invisible direction of the branch, and the skip still carries it. At `ε>0` the branch Jacobian applied to `z` equals `diag(γ) ε z / s^3`. That vector is zero when `z=0`, and when every nonzero coordinate of `z` has gain zero. It is not zero when `z≠0` and `γ` is nonzero on the support of `z`. In that nondegenerate case the radial direction is visible to the branch. The skip contributes the same vector on the residual path. The block derivative along it is the skip plus the branch. Those two terms cancel when the branch contribution equals the negative of the skip. For a constant vector the branch contribution is identically zero, so the block response is exactly the skip. For this radial direction the branch contribution depends on the weights. Invisibility to the normalization is not invisibility to the block.
