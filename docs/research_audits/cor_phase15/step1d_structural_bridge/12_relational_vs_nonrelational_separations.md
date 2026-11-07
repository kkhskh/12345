# 12. Did any joint relation beat the separate bound?

## The inequality that was required

A Level 3 result would be a relation `Q`, fixed from the weights and the architecture without using `R` or `∇m`, and a proved bound `B_joint(H, δ; Q)` with

```text
|R(H, δ)| ≤ B_joint(H, δ; Q) < B_align(H, δ)
```

on a nonempty class, where `B_align` is the separate certificate of Theorem Align. `Q` may not be defined as `B_align - B_joint`. The hypothesis may not be the sentence "`B_joint` is smaller."

## Three attempts

**R1.** `Q_1 = J_A + J_M` is smaller than `|J_A| + |J_M|` when the derivatives oppose. It equals the branch part of `∇m`, so the "without using `∇m`" clause fails. Replacing the base-point value by a bound on the whole segment either costs a family of JVPs or a remainder too coarse to beat `B_align`. The step-`2` example has `Q_1 = 0` and `|R| = 0.169`.

**R2.** The product-ball maximum of `w·a + w·m` equals the sum of the separate maxima. Orthogonality of the images does not reduce that sum. The concatenated Cauchy–Schwarz packaging is weakly worse. The residual coupling is already the radius inside `B_align`.

**R3.** `|R_m - R_s| ≤ B_branch` matches the separate branch bound. It does not undercut it.

## A separation that stops at the gradient

The Householder reflection of `W2` preserves `B_dir` and `B_align` and changes both `Q_1` and `R` (numbers in `09`). This shows that the separate norm bound is blind to a real finite-response difference. The observable that sees the difference is the path-derivative sum. Under Regime 2 that observable is part of `∇m`. Under Regime 1 the signed vector `W2.T w` changes, but a bound that uses only its norm does not, and a bound that uses its direction together with `Δhidden` has evaluated the MLP path.

So the separate certificate is incomplete as a predictor of `R`. The completion available in this block is the derivative or the forward pass, not a third static relation between the two factors.

## Sharpness obstruction, scoped

Let `v_O = W_O w` and `v_2 = W2.T w`. For any radii `ρ_c, ρ_h ≥ 0`,

```text
max_{||a||≤ρ_c, ||h||≤ρ_h} |a·v_O + h·v_2| = ρ_c ||v_O|| + ρ_h ||v_2||.
```

No smaller constant is a valid upper bound for every pair in the product of balls. Therefore no Regime 1 certificate that knows the increments only through those radii can beat the separate sum. A stricter certificate has to shrink the set of pairs. The block does shrink the set, because `h` is a function of `a`. Every shrinkage that was turned into a number used either the Lipschitz radius already present in `B_align` or an evaluation of that function. The evaluation is Regime 4, or Regime 2 if only the derivative is taken.

This is not a theorem that no future argument can find a tighter closed form. It is a theorem that the product-ball bound is already optimal for the information it uses, and that the examined ways of adding the architectural coupling do not improve the uniform number.

## Regime reminder

None of the failed improvements is a Theorem U phenomenon. The weights stay fixed. A weight-free impossibility and a weight-informed product-ball ceiling are different sentences.
