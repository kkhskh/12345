# Adversarial mathematical audit

The workstream 6 table is the dropped-assumption list. Reviews A–H, same environment, are summarized in `reviews/cross_examination.md`. This note records what those passes did to the deepest claim.

The deepest claim, after the negative theorems, is:

> On this pre-norm block, no quantity in budget B0 or in budget B1 on the open set of injective branch input maps bounds the finite response of a non-constant edit more tightly than the constant-shift identity and Cauchy–Schwarz, without using `R` or `Dm`.

Review A accepted Theorem F's eigenspaces and the finite-level argument, and left the explicit `Λ` loose. Looseness makes Theorem L's negative stronger, not weaker: the certificate one can actually write from weight norms is the one that loses to Cauchy–Schwarz by five orders on seed 0.

Review B required the degenerate scale family to be described as a point of the weight class. It is. A bound advertised for every weight has to include it. The same review separated the raw readout from a final LN logit. The transfer fails by a direct computation, not by an analogy.

Review C marked the manufactured kernel point as existence for a conditional sentence. The sentence is true on that point: branch norm below `10^{-15}`, `R=0`, Cauchy number `0.4555`. Using the point as a general law was the unsupported reading, and it was removed.

Review D refused "every seed has a large branch singular value." The proof uses stacked `QKV` injectivity, which the singular values support above `0.5` on seeds 0–11, together with Theorem F.

Review E classified the positive rows as known structure. That classification is the novelty matrix.

Review F left exotic zeros and larger architectures open. Neither opening supplies a B0 bound or a Lipschitz improvement.

Review G blocked any empirical promotion. No score was fit.

Review H asked whether a thin family could still be the successor. A thin family that is one manufactured vector, and that is empty on the injective set, does not beat the shift identity on the class the previous step asked about.

The claim stands. The open cell is the exotic zero, and it is not a candidate theory.
