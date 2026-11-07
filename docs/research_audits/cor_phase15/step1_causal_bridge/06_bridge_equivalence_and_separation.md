# 06 — Equivalence and separation

Eight writeups produced four mathematical objects, not eight mechanisms.

## 1. Objects that remain

**Object 1. Commutation residual.** Family A's `E = P_v f - ρ P_u`, Family B's transport error on linear maps, Family E's `ρ - J`, and Family G's residual against a misspecified parent set are the same kind of quantity: the failure of a chosen linear graph to reproduce a chosen projected computation. On a recomputed trace they are zero exactly when the chosen graph matches the computation (Theorem Z), and they are approximation or omission error otherwise (Theorem App, the residual-skip calculation, the pairwise-versus-joint calculation).

**Object 2. Distance to the kernel.** `D` and, up to the spectral sandwich, `O` are the squared size of Object 1 after it has been assembled into `d s`. Family B's Lipschitz bound is the ordinary distance-to-set bound for this squared norm. Family C says that if `L` factors through the kernel, Object 2 is invisible to `L`.

**Object 3. Output derivative.** Family E's path effect, Family F's stability gap, and the sign analysis in file 09 are statements about `∇L` or `J`. They predict `R_c` without using Object 2, except in the negative way already proved: steepest descent on `D` increases `L` at first order only when `∇L · ∇D < 0`.

**Object 4. Interaction.** Family H's `I(a,b)` is a second difference of intervened outputs. It is not a function of a single cochain's `D`. No other family produces it.

## 2. Implications that fail

- Object 1 exact does not imply Object 3 is aligned with anything (CE-13).
- Object 2 small does not imply Object 3 positive (CE-02, CE-04).
- Object 3 large on the kernel does not imply Object 2 small (Theorem Idle: sensitivity can sit entirely inside `ker d`).
- Object 4 is not determined by Object 2. A linear network has `I=0` and can have any `D` on a stitch that is not a recomputed state.

## 3. Separating examples

| Pair | Shared appearance | Separation |
|---|---|---|
| A and C | Both talk about `ker d` | A can hold with `L` reading a coordinate outside the stalk (CE-13). C forbids that by assumption and then makes `ΔL=0` on the repair |
| B and E | Both bound output change | B bounds `|L(s)-L(Πs)|` with no sign. E gives the signed path effect and does not need `Π` |
| D and F | Both sound like "consistency stabilizes the choice" | D assumes `L` decreases with `D`. F's proved reading of Update Proposition 2 never uses the sign of `∇D` |
| G and A | A bigger graph looks like a richer obstruction | The exact joint graph collapses to A and then to residual 0. The pairwise graph is a different, wrong `d` |
| H and E | Both are causal effects | E's linear term is the cross-derivative set to first order. `I` is the mixed difference. Neither equals `||ds||^2` |

## 4. What is not a further discovery

Proving the Lipschitz bound in obstruction notation is Object 2. Proving that a monotone function of `D` moves `L` is Object 2 plus an assumption that writes the conclusion. Proving that anti-alignment of `∇D` and `∇L` makes descent work is Object 3, with Object 2 supplying a direction that still has to be checked against `∇L`.

No pair among A–H yields a fifth object that predicts the sign of `R_c` from `(P, ρ, M, W)` alone.
