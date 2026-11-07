# 11. The baseline that the gap is measured against

The fair baseline on a forward segment, under Theorem S2, is the sum of the exact marginal maxima:

```text
max f + max g = f(ρ) + 0.
```

It uses the same block, the same segment, the same raw branch objective, and the same closed form of `λ`. It does not use the joint values. For the special gain at `ρ = 1` it equals `0.7070997102248`. The joint certificate is `0`.

Three other numbers are easy to mistake for this baseline.

- **Set C, the product of norm balls.** At the same endpoint the ball ceiling is `1.595484248969981`. Beating it is weaker than beating `f(ρ)`, because the ball ceiling is larger.
- **A global Lipschitz constant built from `1/√ε`.** Here `1/√ε = 316.227766016838`. Any bound that multiplies a spectral norm by that factor is orders of magnitude above both the joint value and the marginal sum. Step 1D's directional certificate is of that type. It remains a correct bound for the whole space. It is not the comparison used to claim a relational improvement.
- **The joint Jacobian norm at the base point.** `|R'(0)| ≈ 50157` upper-bounds a linear model, not the finite response. The finite response at step `1` has the opposite sign. The Jacobian is also not independent of the coupled derivative: it is that derivative.

The information in the marginal sum is the separate ranges. The information in the joint bound is those ranges plus the fact that the maximizing steps differ. Both are available from the weight signs and one evaluation of `λ(ρ)` if one wants the numerical size of the baseline. The joint upper bound `0` does not need `λ(ρ)`. Computing `λ(ρ)` is not a perturbed forward of the MLP and is not the target response.
