# Causal mechanisms

`do(H_t <- H_t+δ)` and `R` are defined in workstream 1 before any of the following is called a mechanism.

## What actually carries a finite response

The skip adds `δ` at the edited token. Both LayerNorms delete the constant part of that addition and, when `ε>0` and at most one gain is zero, delete nothing else (Theorem F). Attention and the MLP then see only the non-constant image. If that image lands in the input kernels, they add nothing and `R` is the skip dot product on the read token. If it does not, the branch contribution depends on the branch weights. Scaling those weights makes the contribution arbitrarily large (Theorem U). The derivative along a nearby direction need not vanish when the finite increment was killed, and a direction killed at first order need not be killed at finite step (Theorem J).

A value-only linearization freezes the pattern and keeps the OV path. That is a different intervention. It can have the opposite sign from the total derivative.

A normalization-invisible direction still passes through the skip. A further LayerNorm on the readout deletes that pass. The raw identity does not transfer.

## What does not carry it

A pairwise compatibility residual. On a recomputed trace it is zero. Off that trace it is surrogate error, a missing parent, a clamp, or an analyst's graph, and the scalar does not say which.

`O` and `D`. They are two quadratic forms of `ds`. They do not see `w` unless the analyst builds the metric from `w` or from `∇m`, in which case the metric has inherited the response.

The cohomology class of `ds`. It is zero for a global 0-cochain.

Information redundancy, causal redundancy, additive necessity, and nonlinear synergy. The pair `c1=1`, `c2=-0.5` fails necessity for both deletions. The pair `b=-1.5`, `c1=c2=1` has both deletions necessary and interaction zero. `u^2+v^2` is nonlinear with interaction zero. None of these predicates is `D`.

## Separate claims

| claim type | status on this block |
|---|---|
| Representation | A projection is sufficient only on the fibers of `R`. Feature 3 fails. Constants make every projection sufficient for a vacuous reason. |
| Computational relationship | The structural equations hold on recomputed traces and do not order `m`. The shift identity is a symmetry of those equations. |
| Causal sufficiency | The kernel hypothesis is sufficient for the skip formula and is empty of non-constant edits when `QKV` is injective. |
| Structural prediction | No B0 or generic B1 quantity beats the shift identity and Cauchy–Schwarz on non-constant edits. |
| Mechanistic explanation | The skip, the LayerNorm level set, and the branch maps are the mechanism. Renaming them as an obstruction does not add a constraint. |
