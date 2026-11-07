# 09 — Stability and response constraints

## D1. Outcomes that are not correctness

A relation might constrain the absolute response, the sign, the rank of two margins, the size of a Jacobian, the probability of a sign change under a stated distribution on `\delta`, or an interaction. These are different functions of `F_θ`. A theorem about one is not a theorem about the others. In particular, a bound `|Δm|\le B` does not say which way the margin moves, and a sign theorem does not say the decision matches a label.

Stable and wrong is a normal case: both paths contribute `-1`, the margin is `-2`, and ablating either path leaves the sign negative. Correctness is a comparison to a label. Nothing in the block equations produces a label.

## D2. First order, second order, remainder

Where `m` is `C^2`,

\[
m(h+\delta)-m(h)=\nabla m(h)^\top\delta+\frac12\delta^\top H_m(h)\delta+r_3(\delta).
\]

The first-order piece is reliable when the remainder is small relative to it on the interventions actually used. It fails for a finite step when higher-order terms dominate. The square `h^2` at `h=2`, step `1`, has first-order change `4` and true change `5`. LayerNorm composed with a fixed readout is a further failure mode for any argument that treats `w·h` as `w·LN(h)`: those scalars can have opposite signs, so a first-order story written in the wrong coordinates is not a small error. It is a different function.

A computational relation predicts the remainder only if the relation determines the higher derivatives that were not already computed. In the examined cases it does not. The relation is a function of a projection of `h`. The remainder is a property of `m`. Connecting them requires the fiber condition on that property.

## D3. Bounds

`|\nabla m^\top\delta|\le\lVert\nabla m\rVert\lVert\delta\rVert` is the Cauchy–Schwarz bound on the first-order term. It uses the gradient. It is not a COR identity.

The value-only attention map is exactly linear, so the remainder is zero and the response equals `(w^\top A \Delta V W_O)` without an inequality. That is tighter than a Lipschitz bound that does not know `A` is clamped. The tightening uses `A`, `w`, `W_O`, and `\Delta V`. Those are the Jacobian of this intervention family. A relational residual is not among the inputs. Under comparable information — someone who already has the Jacobian — the residual adds nothing. Under a weaker information budget — someone who knows only that the intervention is value-only and does not know `A` — the architectural fact still yields remainder zero, but not the numerical response. The content of that weaker prediction is linearity, which is the attention equation.

No examined relation produces a stricter numerical bound than the Jacobian of the same family from strictly less information.

## D4. Margin preservation

For `m=c_1+c_2` with a linear readout, the sign survives deletion of path 1 if and only if `c_2` has the same sign as the margin, or is zero and the margin was already zero. The analogous statement holds for path 2. The condition is about the remaining inner product. It is not implied by `\lVert p_1-p_2\rVert` being small: the vectors `(1,100)` and `(1,-100)` are far apart and each inner product with `(1,0)` equals `1`. It is not implied by the vectors being equal: both can be negative.

For attention with a clamped pattern, deleting one value subtracts exactly that value's contribution. The margin is preserved when the remaining convex combination still lies on the same side of the decision threshold. Again this is a calculation in `A` and `V`. A low consistency distance is not a hypothesis of the calculation.

For residual edits of the bias-free attention map, preservation is a statement about that recomputed map. The two states with the same projection and token-1 outputs `1/2` and `1/(1+e^{16})` already show that a preserved projection does not preserve the output, hence does not preserve a margin that reads the output. The second value is not exactly zero.

## D5. Assessment

Computation-grounded relational measurements, in the sense of a residual between projected states, do not yield a distinctive stability prediction in the families examined here.

The stability facts that do follow are: exact linearity of value-only attention, which is the Jacobian; sign survival under ablation, which is the sign of the remaining contribution; and the failure of Euclidean or fixed-readout reasoning across LayerNorm. Each is a standard sensitivity or invariance fact about the map that is actually differentiated. None is a theorem of the form "a smaller coboundary implies a more stable decision."
