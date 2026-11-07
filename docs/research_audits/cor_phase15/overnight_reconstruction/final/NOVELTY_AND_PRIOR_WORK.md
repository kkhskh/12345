# Novelty and prior work

The matrix covers statements that were proved in this package. "New here" means new relative to the previous audit's writeup, not new relative to mathematics.

| statement | Geiger et al. JMLR 2025 | Sutter et al. NeurIPS 2025 | Pîslar et al. PMLR 2025 | Olivieri & Pérez Rodríguez, arXiv:2605.25225 | Vaswani / Ba / Xiong / Elhage | class |
|---|---|---|---|---|---|---|
| Finite LN level set at `ε>0`, at most one zero gain | Not their subject | Not their subject | Not their subject | Not their subject | Calculation on Ba, Kiros, Hinton. The Jacobian kernel was already in the previous audit. The finite level set, including the two-gain counterexample, is the scope correction. | Not a COR mechanism |
| No finite B0 bound for non-constant `δ` | An intervention whose mechanism is unread is ordinary | Unrestricted maps are a different vacuity (Theorem 1) | Not a combine operation | Their sensitivity uses the derivative, which B0 forbids | Pre-norm (Xiong et al.) makes `X` independent of a constant; scaling `W_O` then scales a centered `ΔX` | Boundary |
| Lipschitz certificate `≥ \|\|g\|\|` | Not their subject | Not their subject | Not their subject | Their linear regime is this first-order fact in field language | Mean-value inequality on a differentiable readout | Generic |
| Explicit `Λ` from weight norms and `1/sqrt(ε)` | Not their subject | Not their subject | Not their subject | Not a Green operator | Uses Vaswani's softmax, the `1/2` spectral bound proved here via Popoviciu, and Ba's `ε` | Valid and non-competitive |
| Kernel equality `R=w·δ` | Exactness for a hard intervention on an unread coordinate (Defs. 9, 25, 44) | A linear restriction, which their dilemma says abstraction requires, and which is still the block | Not an input-space partition (their Def. 1) | Not a sensitivity field | Skip plus Elhage's residual stream; the OV path is the value-only object, which can have the opposite sign | Conditional, rejected as successor |
| Jacobian kernel versus finite gap | Not interchange accuracy | Not an alignment map | Not faithfulness | Separates their first-order sensitivity from a finite patch | Chain rule versus the finite map | Boundary between orders |
| `O` and `D` do not rank `L` | A residual is not an interchange | A fitted `ρ` can manufacture agreement | A trivial high-level model is faithful and empty | A derivative score is not `O` | Not an attention identity | Preserved negative |
| `[ds]=0` has no causal content | Not a macrovariable | Not an alignment | Not a partition | Not a propagator | Not a circuit | Removed |

No row is a new causal abstraction theorem, a new interchange criterion, or a response field that avoids the derivative.
