# Reviewer A — functional and operator analysis

Same-context pass, not an external referee.

The equality criterion in Theorem S1 is correct under the attainment hypothesis, and the segment supplies attainment. The linear gap is the Cauchy defect and the equality case is right.

The place the argument is thinnest is the passage from `tanh(z) > z - z^3/3` to the ratio `2k/3`. That ratio is comfortable, and the integer check `4k > 3` is the right one, but a reader has to reconstruct `k = sqrt(50000)/(2 sqrt(3))` from `s(1/100)^2 = 3/50000`. The note states those fractions. I do not see a hole in the case split: the earlier-token coordinate is the smaller one, and the read-token coordinate only makes `tanh(λ(y1))` larger.

I would not call `R(t) ≤ ρ` a sharp operator bound. It throws away the negative branch instead of integrating it. It is still strictly below the marginal sum, which is the comparison on offer. No objection to the verdict on operator grounds, provided the family stays the segment.
