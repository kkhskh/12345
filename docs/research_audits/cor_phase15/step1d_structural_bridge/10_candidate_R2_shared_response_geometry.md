# 10. Candidate R2 — shared response geometry

## Unary output subspaces

The attention increment lies in the row space of `W_O`, because `ΔY = Δconcat W_O`. The MLP increment lies in the row space of `W2.T`, because `ΔM = Δhidden W2.T`. These are facts about the factors, checked without `R`. Projecting `w` onto each row space reproduces Theorem Align: the sharp one-path constants are exactly `||W_O w||` and `||W2.T w||`, since

```text
max_{||a||≤ρ} a·(W_O w) = ρ ||W_O w||
```

and likewise for the hidden increment. On seed 0 both output maps are full rank onto `R^4` (`W_O` is `4×4`, `W2` is `4×5`), so the row spaces are the whole readout space and the subspace formulation gives nothing beyond the aligned gains. The aligned gains still improve on `||w|| σ(·)` because `w` is not a top singular vector. That improvement remains unary.

## Orthogonality of the two images does not beat the sum

Suppose the admissible increments `(a, m)` range over the product of two balls, `||a|| ≤ ρ_A` and `||m|| ≤ ρ_M`, with `a` in the attention row space and `m` in the MLP row space. Then

```text
max (w·a + w·m) = ||P_A w|| ρ_A + ||P_M w|| ρ_M,
```

because the maxima are attained independently. If the row spaces are orthogonal, `a` and `m` are orthogonal vectors, but the sum of the scalars `w·a` and `w·m` is still the sum of the separate maxima. Pythagoras lowers `||a+m||`, not `w·a + w·m`, once each piece is already aligned with the projection of `w`.

The concatenated Cauchy–Schwarz form

```text
|a·v_O + h·v_2| ≤ √(ρ_A^2 + ρ_M^2) √(||v_O||^2 + ||v_2||^2)
```

upper-bounds the inner product, and that right-hand side is at least the separate sum `ρ_A ||v_O|| + ρ_M ||v_2||`. The square gap expands to `(ρ_A ||v_2|| - ρ_M ||v_O||)^2 ≥ 0`. The two bounds agree when the radius ratio equals the gain ratio. Otherwise the separate sum is strictly tighter. Packaging the same radii as one Euclidean ball is a weaker certificate.

## Coupling through the residual

The architecture does not let `a` and `m` vary in a full product: `m` is the MLP increment of `H + a`, not an independent vector. The Regime 1 bound already uses that coupling once, in the radius `||ΔH1|| ≤ ||δ|| + ||ΔA||`. That is the chain rule estimate written as a Lipschitz composition. It is inside `B_align` and `B_dir`. It is not an extra relation on top of them.

A further reduction would require excluding some pairs inside that radius without evaluating the perturbed branch or its derivative. No such exclusion was proved. The product-ball maximum remains the certified uniform bound. The sign-flip pair in `09` shows that two realizations with the same radii and the same `B_align` can have different finite responses, so the uniform bound is not sharp on every point. The unrealized slack is not, by itself, a tighter closed-form certificate.

## Bottleneck that looks joint and factors

`w` orthogonal to the sum of the two row spaces means `w` orthogonal to each of them. The finite conclusion `R = 1_{t=r} w·δ` is Theorem Kernel-out. It factors into the two unary kernel conditions. There is no residual joint hypothesis.

## Verdict on R2

Shared row spaces, orthogonal images, and a shared downstream functional `w` were examined. Each either reproduces a per-path bound already in Theorem Align or produces a weaker concatenated estimate. None yields `B_joint < B_align` under Regime 1. R2 does not clear Level 3.
