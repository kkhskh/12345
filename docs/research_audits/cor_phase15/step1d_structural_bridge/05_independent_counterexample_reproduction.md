# 5. Reconstruction of the structural-bound example

`INDEPENDENT_OVERNIGHT_AUDIT.md` and `independent_boundary_checks.py` were not in the repository. The construction below meets the stated mathematical conditions and was fixed before the response was computed. It was not fitted to `6.50425e-5` or `7.58e-11`.

## Construction, chosen first

1. Draw seed 0 from `make_params`, NumPy 1.26.4 `default_rng`. This fixes every weight, both LayerNorms, `w`, and `H`.
2. Replace `W_O` by `α W_O` and `W2` by `α W2`. The scale `α = 10^{-6}` was chosen as a small nonzero output scale, not by inverting a target bound. `W_Q`, `W_K`, `W_V`, `W1`, the biases, and the gains stay at the seed-0 values.
3. Let `v_0 = (0, 1, 0, 0)`. Set `δ` to the normalization of the projection of `v_0` onto `w^⊥`.

Consequences, checked after the definition and not used to change it:

- `||δ||_2 = 1`.
- `w · δ = -4.34×10^{-19}`, numerical zero.
- `δ` is not constant. Its coordinates are about `(0.003833, 0.999980, 0.003020, 0.004064)`.
- `1 · w = 1.5010589548912336 ≠ 0`, so every nonzero vector in `w^⊥` is automatically nonconstant.
- The stacked `Q,K,V` map `R^4 → R^{12}` has smallest singular value `0.7886607004096067`. Scaling `W_O` and `W2` does not change that map. The kernel is `{0}`.

## Recomputed values at `α = 10^{-6}`

Two forwards agree, and the analytic gradient agrees with central differences to `10^{-8}` on the unscaled block. At this scale:

| quantity | value |
|---|---|
| directional certificate `B_dir` | `0.004051326443971923` |
| aligned certificate `B_align` | `0.0028008328328102084` |
| isotropic certificate `B_iso` | `0.8804862965926746` |
| `||∇m||_2` | `0.8764350281115764` |
| `∇m · δ` | `-5.192333216467657e-08` |
| `R` | `-1.326797428591320e-07` |
| `||ΔA||_F` | `2.565584280469757e-07` |
| `||ΔM||_F` | `4.927701386349185e-07` |

The audit's Cauchy number `0.876435` matches `||w|| = 0.8764349701487028` to the printed precision. At small output scale the gradient tends to `w`, so that agreement is expected if the audit used this readout. It is not evidence that the branch scale was the same. `B_dir` and `|R|` do not match `6.50425e-5` and `7.58e-11`. Those floats are not claimed.

## Which weights the scale hits

Only the branch output maps `W_O` and `W2`. At `α = 10^{-6}`,

```text
σ(W_O) = 1.285734502700360e-06,   ||W_O w|| = 7.977444801761173e-07,
σ(W2)  = 9.971626566633242e-07,   ||W2.T w|| = 4.683066119672438e-07.
```

`L_A` and `L_M` are proportional to `α`. The LayerNorm constants stay `L_1 = 320.27815449796316` and `L_2 = 316.48913445938666`, from `||γ||_∞ / √ε` at `ε = 10^{-5}`.

## How the bound and the response depend on `α`

For this ray, `L_A(α) = α L_A(1)` and `L_M(α) = α L_M(1)`. The branch rate in Theorem S is

```text
rate(α) = α (L_A(1) L_1 + L_M(1) L_2) + α^2 L_M(1) L_2 L_A(1) L_1.
```

`B_dir = ||w|| rate(α)` when `w·δ = 0` and `||δ|| = 1`. The linear term dominates until `α` is large enough for the quadratic piece to matter. The finite response is not exactly linear in `α`, because the MLP sees `H1 = H + A` and `A` itself scales with `α`, and because LayerNorm and tanh are nonlinear. On this ray `|R|` grows from `0` at `α = 0` through `1.33×10^{-7}` at `10^{-6}`, `1.29×10^{-3}` at `10^{-2}`, and `1.08×10^{-2}` at `α = 1`. The certificate grows much faster, from `4.05×10^{-3}` at `10^{-6}` to `1.63×10^6` at `α = 1`.

The inequality `|R| ≤ B_dir` was checked on the grid `α ∈ {0, 10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}, 0.1, 1, 10}`, on tokens `0, 1, 2`, and on orthogonal, parallel, constant, and mixed edits. That grid is a check of a proof, not the proof. The proof is in `06`.

## What the example disproves

It disproves the universal sentence "no weight-informed quantity bounds a nonconstant finite response more tightly than the Cauchy number `||∇m|| ||δ||`." At `α = 10^{-6}` the Regime 1 certificate `0.004051` is a valid finite bound and is smaller than `0.876435`. The stacked `Q,K,V` map is injective, so the example is inside the class the overnight sentence called generic.

It does not disprove Theorem U. The weights are fixed and small; Theorem U quantifies over unbounded weights.

It does not disprove Theorem L. `B_dir` is not an isotropic Lipschitz constant. The isotropic constant on the same weights is `0.880486`, which is still above `||∇m||`.

It does not produce a COR mechanism. Section `07` splits the numerical gap. The drop from the isotropic bound to `B_dir` is the skip orthogonality identity. The small size of `B_dir` relative to `||w||` is the small output scale. Neither step is a relation between attention and the MLP.

## A second point, so the claim is not one float

At `α = 10^{-4}`, still below the threshold of `08`, `B_dir = 0.4212221296158794`, `||∇m|| = 0.8764408`, and `|R| = 1.326418×10^{-5}`. The certificate remains valid and remains under the Cauchy number, and it is already four orders above `|R|`. At `α = 10^{-3}` the rate exceeds `1/2`, `B_dir = 5.674902`, and the Cauchy comparison reverses, while `|R| ≤ B_dir` still holds. The phenomenon is a scale interval, not a single tuned point.
