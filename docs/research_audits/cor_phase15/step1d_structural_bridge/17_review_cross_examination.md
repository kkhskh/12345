# 17. Cross-examination

The prosecution and defense passes are in `reviews/`. Both were written in this context. This note records the points on which they now agree, with the mathematical reason, and the one point that does not need a winner because both descriptions are true.

## Agreed: CE-L-scope kills the universal B1 sentence

Prosecution accepts the witness as valid. Defense refuses to discard it for being small. The overnight sentence quantified over the injective-QKV class and over nonconstant edits, without a lower bound on branch scale. The witness lies in that class. The sentence is withdrawn. Theorem L is untouched, because `B_iso = 0.880486 > ||∇m|| = 0.876435` at the same point.

## Agreed: the numerical gap has two named pieces

On the unit orthogonal edit at `α = 10^{-6}`,

```text
B_iso - B_dir = ||w|| = 0.8764349701487028,
B_dir         = 0.004051326443971923.
```

The first line is orthogonality of `δ` to `w`. The second is the branch majorant at that scale. Prosecution's remark that the comparison with `||∇m|| ||δ||` is mostly `||w||` against a small majorant is correct. Defense's remark that `B_dir < B_iso` holds at `α = 1` as well, with the same gap `||w||`, is also correct. Those are different comparisons. `03` and `07` keep them apart. Neither comparison is a joint-path gain.

## Agreed: R1 is the derivative

Prosecution says `J_A + J_M` is the branch part of `∇m`. Defense does not offer a different identification. The integral form is the fundamental theorem of calculus. The step-2 witness blocks any claim that the base-point equality controls `|R|`. There is no remaining dispute.

## Agreed: the product ball is optimal for the radii it uses

Defense accepts the maximum argument. Prosecution's two-line version and the square-gap identity in `10` are the same fact. The Householder pair shows unrealized slack in the uniform bound. Both passes agree that the slack is visible to `∇m` and not to `B_align`. Using `∇m` to capture it leaves the Regime 1 question. No third object was produced.

## Agreed: final LayerNorm is a different functional

Raw orthogonality, raw output kernels, and the raw constant shift all fail to transfer, in the directions recorded as CE-final-orth, CE-final-kernel, and CE-const-final. The diameter bound is the Regime 1 statement that does transfer, and both passes call it a correction of scope rather than a mechanism.

## The point that is two true descriptions

CE-L-scope is a valid counterexample and a thin regime. `08` states both. The open set `{rate < 1/2}` makes the thin regime topologically stable and does not make it typical. Prosecution's refusal to read the open set as a trained-model statement matches defense's text. Nothing further is owed.

## Not agreed, and not left hanging

There is no item on which one pass asserts a relational Level 3 theorem and the other denies a specific inequality. The Level 3 claim was not made. The argument stops at the product-ball ceiling and the derivative identification.
