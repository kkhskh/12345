# Workstream 4. Geometry, without a privileged obstruction

## The algebra that stays

On a finite-dimensional coboundary `d`, with `M ≻ 0` and `W ≻ 0`,

```text
O(s) = (ds)^T W (ds),
D(s) = (ds)^T (d M^{-1} d^T)^+ (ds).
```

Both are quadratic forms of `ds`. They agree for every `s` when `W = (d M^{-1} d^T)^+`. They do not agree in general. On the identity edge, `s=(1,0,0,0)`, Euclidean `M` and `W`, one has `O=1` and `D=1/2`.

Changing `M` and holding `(d, W, s)` fixed changes `D` and does not change `O`. The diagonal `M=diag(4,4,1,1)` is the check.

`O=0` if and only if `D=0` if and only if `ds=0`, when both metrics are positive definite. The common zero set is `ker d`. It contains `0`. It is not a correct decision: the same `s` with readout `w` and with readout `-w` has two margins and one obstruction.

`D(s')<D(s)` does not imply a larger margin. The zero cochain has `D=0` and a smaller first-coordinate readout than `s=(1,0)`.

`D_c < D_{c'}` is not a theorem about `L_c` and `L_{c'}`. Two linear heads on one cochain reverse the margin and preserve `(O, D)`.

`ds` is `d` applied to a globally defined 0-cochain, so its class in `C^1 / im(d)` is zero. On the two-node operator, least squares reconstructs `ds` from `d` exactly. A functional of the cohomology class is the zero functional. It cannot constrain `R`. Cohomology is removed from the prospective mechanism. No relative cocycle is introduced to keep the word.

## Four operations that are not the same

**Coordinate change of a stalk.** Invertible `P`, with `(ρ, M, W)` transformed by the same change, preserves `O` and `D` as scalars. Leaving the metrics fixed does not.

**LayerNorm.** A map inside the block. For `ε>0` and at most one zero gain, its finite level set is the constants (Theorem F). It is not scale-invariant. Its Jacobian has rank `d-1` when every gain is nonzero, and rank `d-2` at `ε=0` with non-constant input and invertible gain. Those are properties of the map, not of a change of basis of `H`.

**Intervention.** `do(H_t <- H_t+δ)` changes the input of the block. A constant intervention is invisible to both normalizations and visible to the residual skip. A centered intervention is visible to the normalization when the gains on its support are nonzero, and the branch may or may not cancel it.

**Function-preserving reparameterization.** Replacing a bias-free linear factor by an equivalent product changes internal matrices and not the block map. The finite kernel construction is the opposite: it changes the block until one direction is unread. A metric built from `∇m`, for example `q(δ)=(g·δ)^2`, moves when `g` moves and is an even function of the linear response. It inherits the response. Euclidean `O` on a fixed cochain does not move when a readout that was never an input is negated, and therefore does not inherit the sign either. Inheritance and blindness are different failures.

## What is not claimed

Topology adds no distinctive causal theorem on this complex, so it is not a candidate mechanism. The metric `M` is not privileged. A gradient metric is a rewriting of the gradient. Neither repairs `D ↓ ⇒ L ↑`.
