# 6. Directional finite-response bounds

Norms: Euclidean on token vectors, Frobenius on token matrices. These agree on a single-token edit: `||E_{t,δ}||_F = ||δ||_2`. Spectral norm means the largest singular value. Throughout, `ε > 0`.

## Lemma 1. LayerNorm radius and Lipschitz constant

For `s = √(||z||^2/d + ε)` one has `||z/s||_2 ≤ √d`. Hence

```text
||LN_{γ,β}(h)||_2 ≤ ||γ||_∞ √d + ||β||_2 =: R_LN.
```

The Jacobian of `h ↦ γ ⊙ (z/s) + β` has operator norm at most `||γ||_∞ / √ε`, because the symmetric bracket has eigenvalues `0` and numbers at most `1/s ≤ 1/√ε`, and `γ` multiplies by at most `||γ||_∞`. The domain is convex and the derivative bound is global, so

```text
||LN(h) - LN(h')||_2 ≤ (||γ||_∞ / √ε) ||h - h'||_2.
```

Write `L_1` and `L_2` for the two LayerNorms. Biases do not appear in the Lipschitz constants. They do appear in `R_LN`. On seed 0, `R_1 = 2.060344826367208`, `L_1 = 320.27815449796316`, `L_2 = 316.48913445938666`.

A sample of 20 random pairs stayed under `L_1`. The largest observed ratio was far below 1. The sample is a check. The proof is the eigenvalue bound.

## Lemma 2. Softmax is globally 1/2-Lipschitz

The Jacobian `diag(p) - pp^T` has quadratic form equal to the variance of the coordinate function under `p`. A Euclidean unit vector has coordinate range at most `√2`, because `(v_i - v_j)^2 ≤ 2(v_i^2 + v_j^2) ≤ 2`. Popoviciu's bound then puts the variance at most `(√2)^2 / 4 = 1/2`. Equality holds at `p = (1/2, 1/2)` and `v = ±(1/√2, -1/√2)`. SymPy evaluates that quadratic form as `1/2`. A causal row is the same map on its legal coordinates, with illegal scores held at `-∞` and contributing derivative zero. The constant `1/2` still upper-bounds the row.

## Lemma 3. Attention on the row ball

Let `B` be the set of token matrices whose rows have Euclidean norm at most `R_1`, the radius of the first LayerNorm. `B` is convex. Every LayerNorm image lies in `B`. Attention is `C^∞` on a neighborhood of `B`: the causal mask fixes some scores at `-∞` and does not depend on `X`.

On `B`, one head with factors `W_Q, W_K, W_V` of widths `d × d_k` satisfies

```text
||d(head)||_F ≤ l ||dX||_F,
l = σ(W_Q) σ(W_K) σ(W_V) R_1^2 T / √d_k + σ(W_V) √T.
```

Derivation. `||dQ||_F ≤ σ(W_Q) ||dX||_F`, and likewise for `K` and `V`. At a point of `B`, `||K_j|| ≤ σ(W_K) R_1` and `||Q_i|| ≤ σ(W_Q) R_1`. The legal score derivative obeys

```text
||dS_i||_2 ≤ (σ(W_K) R_1 / √d_k) (√T ||dQ_i|| + σ(W_Q) ||dX||_F),
```

where future keys, which the mask drops, are kept in the estimate and only loosen it. The softmax row contributes a factor `1/2`. The value term `P dV` has row norm at most `||dV||_F ≤ σ(W_V) ||dX||_F`, because `P` is a probability vector. The pattern term `dP · V` has row norm at most `||dP_i||_2 ||V||_F`, and `||V||_F ≤ σ(W_V) R_1 √T`. Collecting the coefficient of `||dQ_i||` and the coefficient of `||dX||_F`, summing the rows in Frobenius norm, and using `||dQ||_F ≤ σ(W_Q) ||dX||_F` produces the displayed `l`. The two bilinear pieces each carry a factor `1/2` and add to the coefficient `1` in front of `σ_Q σ_K σ_V R_1^2 T / √d_k`.

This is a bound on the derivative everywhere on `B`. The mean value inequality on the convex set `B` upgrades it to a Lipschitz constant. Concatenating heads costs the Euclidean combination `L_concat = √(Σ l_h^2)`. The output map costs one more spectral norm:

```text
L_A = σ(W_O) L_concat.
```

`b_O` cancels in differences. On seed 0 the head constants are `6.300358214660888` and `7.948545021286059`, and `L_A = 13.040788809170579`. Twenty-five random pairs inside the row ball produced a largest ratio `||Δ Attn||_F / ||ΔX||_F` of about `0.0805`, under `L_A`.

The overnight code used a larger head constant, `2 σ_Q σ_K σ_V T R_1^2 / √d_k + √T σ_V`. That expression was not copied. The factor `2` is what one gets by replacing the softmax constant `1/2` with `1` and not cancelling the resulting halves. The smaller constant is the one proved here, and it is the one the tests call.

`L_A` depends on the row-ball radius. Attention is not globally Lipschitz on unbounded inputs, because the score map is bilinear. The hypothesis "inputs lie in the LayerNorm image" is necessary for this `L_A`.

## Lemma 4. MLP

`tanh` is 1-Lipschitz because its derivative `1 - tanh^2` lies in `(0, 1]`. SymPy reduces the derivative identity to `0`. Therefore each token map `U ↦ tanh(U W1.T + b1) W2.T + b2` has Lipschitz constant

```text
L_M = σ(W1) σ(W2),
```

globally, with no restriction on the domain. Biases cancel in differences.

## Theorem S. Directional certificate

For every base state, every token `t`, and every finite `δ`,

```text
|R(H, t, δ)| ≤ |1_{t=r} w·δ| + ||w|| (L_A L_1 + L_M L_2 (1 + L_A L_1)) ||δ||.
```

Proof. Only token `t` changes, so `||ΔX||_F ≤ L_1 ||δ||` and `||ΔA||_F ≤ L_A L_1 ||δ||`. Then `||ΔH1||_F ≤ ||δ|| + ||ΔA||_F`, `||ΔU||_F ≤ L_2 ||ΔH1||_F`, and `||ΔM||_F ≤ L_M ||ΔU||_F`. The read-token rows are at most the Frobenius norms. Hence

```text
||ΔA_r|| + ||ΔM_r|| ≤ (L_A L_1 + L_M L_2 (1 + L_A L_1)) ||δ||.
```

The identity `ΔH2_r = 1_{t=r} δ + ΔA_r + ΔM_r` gives

```text
R = 1_{t=r} w·δ + w·ΔA_r + w·ΔM_r,
```

and Cauchy–Schwarz on the two branch vectors produces the displayed bound. When `t = r` and `w·δ = 0`, the skip term drops and the formula is the audit expression

```text
|R| ≤ ||w|| [L_A L_1 + L_M L_2 (1 + L_A L_1)] ||δ||.
```

The argument never restricts `||δ||`. Both LayerNorm constants and the MLP constant are global, and `L_A` applies to every pair of LayerNorm images. Multi-head attention is inside `L_A`. Biases cancel. The last-token case is sharper than an earlier token only because, under the causal mask, an edit at token `r = T-1` does not change earlier queries' legal keys; the proof does not need that fact, because Frobenius norms already cover the earlier-token case. The tests include tokens `0` and `1`.

## Theorem S-iso. Gap against the isotropic constant

The same estimates without separating the skip give

```text
|R| ≤ ||w|| (1 + L_A L_1)(1 + L_M L_2) ||δ|| =: B_iso.
```

Expanding the product shows `B_iso = ||w|| ||δ|| + B_branch`, while `B_dir = |1_{t=r} w·δ| + B_branch`. The gap is `||w|| ||δ|| - |1_{t=r} w·δ|`.

## Theorem Align. Readout-aligned gains, still separate

`Y = concat(X) W_O + b_O`, so `w · ΔY = Δconcat · (W_O w)` and

```text
|w·ΔA| ≤ ||W_O w|| ||Δconcat||_F ≤ ||W_O w|| L_concat L_1 ||δ||.
```

`M = hidden W2.T + b2`, so `w · ΔM = Δhidden · (W2.T w)` and `||Δhidden|| ≤ σ(W1) ||ΔU||`. The input radius of the MLP still uses the full vector bound `||ΔA|| ≤ L_A L_1 ||δ||`, because the residual `H1` moves by the whole attention vector, not by its inner product with `w`. Therefore

```text
|w·ΔM| ≤ ||W2.T w|| σ(W1) L_2 (1 + L_A L_1) ||δ||.
```

Adding the exact skip inner product yields `B_align`. Since `||W_O w|| ≤ σ(W_O) ||w||` and `||W2.T w|| ≤ σ(W2) ||w||`, one has `B_align ≤ B_dir`. On seed 0 at unit scale the ratios `||W_O w|| / (σ(W_O) ||w||) = 0.7079340668437615` and `||W2.T w|| / (σ(W2) ||w||) = 0.5358516666336607`, and `B_align = 873666.373048331` against `B_dir = 1629250.228383555`. The improvement is per path. It is the operator applied to `w`, not a relation between the paths.

## Theorem Kernel-out. Exact raw identity

If `W_O w = 0` and `W2.T w = 0`, then `w·ΔA = 0` and `w·ΔM = 0` for every finite edit, so

```text
R(H, t, δ) = 1_{t=r} w·δ.
```

This holds whether or not `δ` is constant, and whether or not `Q,K,V` are injective. On seed 0 the projection that enforces the two conditions leaves the stacked singular value unchanged at `0.7886607004096067`. For `δ` equal to `2.5` times the unit orthogonal vector, the recomputed gap against `w·δ` is below `10^{-15}`. The branches still move: `||ΔA||_F > 0.1` and `||ΔM||_F > 0.05`. Spectral norms of the projected maps remain `σ(W_O) = 1.2823531777637214` and `σ(W2) = 0.9960563055080783`. The branch computation is not small. Its readout gain is zero.

The hypothesis is two unary conditions. The separate aligned bound is already `0` under the same hypothesis. The joint statement does not beat the separate one.

The identity is not open. A nonzero `W_O w` produces a nonzero attention contribution to the readout as soon as `Δconcat` meets that direction. The quantitative form is the inequality in Theorem Align, not a neighborhood of exact zeros.

## Final LayerNorm

Theorem S does not apply to `m_f = w · LN(H2[r])`. The outer normalization mixes coordinates before the inner product with `w`, so `w·δ = 0` does not delete a skip term.

Zero branch outputs, orthogonal unit `δ`, seed-0 state: the raw response is `0` and the final-LayerNorm response is `-0.32024254867419905`. The diameter bound of `02` contains it.

Output-kernel weights, same orthogonal `δ`: the raw response is `0` and the final-LayerNorm response is `-0.13186230895395856`. The exact raw identity does not survive the outer normalization.

A constant edit is the opposite transfer. Raw response equals `w·1 = 1.5010589548912336`. Final-LayerNorm response is numerically `0`, because LayerNorm deletes constants. Raw identities do not move to a normalized readout in either direction.
