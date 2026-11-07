# 1. Overnight claims that went past Theorems U and L

The overnight files are left as they were. This note quotes the overstatements and replaces them with scoped sentences. A quote is not a re-endorsement.

## What Theorem U actually proved

On the raw-readout block, fix `ε>0`, the gains, `w`, `H`, and a nonconstant `δ`. There is a family of branch weights, obtained by scaling an output map aligned with the centered LayerNorm increment, on which `|R|` is an arbitrary multiple of `||w||^2 / T` plus the skip term. No function of the weight-free budget is a finite upper bound on `|R|` for every branch weight.

Theorem U does not say that a fixed weight tuple has an unbounded response. It does not say that a final LayerNorm readout with fixed gain and bias is unbounded. It does not say that a certificate which is allowed to read the weights fails.

## What Theorem L actually proved

If `|m(x) - m(y)| ≤ Λ ||x - y||_2` on a convex neighborhood of a differentiable point, then `Λ ≥ ||∇m||_2`. An isotropic Lipschitz constant cannot undercut the gradient norm. Equivalently, `Λ ||δ|| ≥ ||∇m|| ||δ|| ≥ |∇m · δ|` for every `δ`.

Theorem L does not say that a direction-specific finite bound `B(H, δ)` is at least `||∇m|| ||δ||`. The isotropic constant has to work for every direction. A bound that uses `w · δ = 0`, a factor Lipschitz constant, or a kernel identity is not that constant.

## Four quantities that were treated as one

| symbol | object | what it bounds |
|---|---|---|
| A | `|∇m · δ| ≤ ||∇m|| ||δ||` | the linear term only |
| B | `|m(x+δ) - m(x)| ≤ Λ ||δ||` | the finite response, if `Λ` is a Lipschitz constant on the segment |
| C | `|R| ≤ B(H, δ)` | the finite response, for that `δ`, if the certificate's hypotheses hold |
| D | `R = m(H+δ) - m(H)` | the reference value |

A is not a finite-response bound. Comparing C with `||∇m|| ||δ||` does not show that C is a better finite-response certificate than a gradient method. The gradient method's finite competitor is `|∇m · δ|` plus a proved remainder, or a direct evaluation of D.

## Sentences that overreached

These remain in the overnight package. Each is withdrawn as a claim about this block.

1. `EXECUTIVE_SCIENTIFIC_VERDICT.md`: "Any Lipschitz bound that does use the weights is at least the Cauchy–Schwarz number of the gradient."
   Replacement: any *isotropic* Lipschitz constant is at least the gradient norm. A directional certificate need not be.

2. `README.md`: "every Lipschitz certificate is at least the Cauchy–Schwarz number."
   Same replacement.

3. `final/ADVERSARIAL_MATHEMATICAL_AUDIT.md`: "no quantity in budget B0 or in budget B1 on the open set of injective branch input maps bounds the finite response of a non-constant edit more tightly than the constant-shift identity and Cauchy–Schwarz, without using `R` or `Dm`."
   Replacement: budget B0 has no finite bound on the unbounded-weight class (Theorem U). Budget B1, meaning the weights may be read but the perturbed forward may not be evaluated, does have directional certificates. Theorem S below is one. It undercuts `||∇m|| ||δ||` on a nonempty open set of small output scales and does not undercut it at unit scale. Undercutting that number is not, by itself, a win over a valid finite competitor.

4. `final/PHASE15_FINAL_VERDICT.md`, the verdict paragraph and Q23, Q26, and the closing paragraph "Lipschitz certificates cannot."
   Replacement: isotropic certificates cannot undercut `||∇m||`. Directional ones can undercut the Cauchy *number* `||∇m|| ||δ||` without contradicting Theorem L. They cannot undercut `|∇m · δ|` merely by being Lipschitz in one direction unless a remainder argument is supplied. The overnight Q23 answer is right about B0 and wrong about every weight-informed certificate.

5. `final/CAUSAL_MECHANISMS.md`: "No B0 or generic B1 quantity beats the shift identity and Cauchy–Schwarz on non-constant edits."
   Replacement: no B0 quantity does, on nonconstant edits, in the unbounded-weight class. "Generic B1" was being used for the injective-QKV set. Injectivity of `Q,K,V` does not forbid a small `W_O` or a `W_O` whose rows miss `w`. Those are different matrices.

6. `final/SEARCH_COVERAGE_AND_OPEN_SPACE.md` lists "A Lipschitz certificate that beats Cauchy–Schwarz" as closed.
   Replacement: closed for isotropic constants. Open, and now answered, for directional certificates: they beat the isotropic Lipschitz bound by `||w|| ||δ|| - |w·δ|`, and they beat the Cauchy number only in the small-rate regime described in `08_nontriviality_and_parameter_regimes.md`.

## Same-model trace, replacing the two-seed comparison

The overnight note compared seed 0's margin `+0.4644721976523223` with seed 2's margin. Those are different weights. The structural equations hold for both because both traces are recomputed, so the comparison does not hold the mechanism fixed.

On seed-0 weights alone, the base state has margin `+0.4644721976523223`. Shifting the read token by `-1.158811460185589 w` produces margin `-0.4644721976523225`. Both structural residuals are `(0, 0)`. Satisfaction of the block's own equations does not determine the sign of `m`. It does not show that every relation among paths is uninformative.

The inaccessible audit's symmetric pair was approximately these two numbers. The positive value is the seed-0 margin. The negative value was obtained here by a one-dimensional shift along `w` on that same weight tuple, not by a second seed.
