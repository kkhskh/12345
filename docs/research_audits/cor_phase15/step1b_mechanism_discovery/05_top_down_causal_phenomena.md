# 05 — Top-down, from causal phenomena

Start from a phenomenon and ask what structure would produce it.

**Robustness to path removal.** Linear case: `|ΔL|` small under ablation of path `i` iff `|w·p_i|` is small. That is a small path-specific attribution, not a small disagreement with other paths.

**Compensation.** Output unchanged after removing `p_i` while `p_i` is large as a vector: then `w ⊥ p_i`, or a nonlinearity saturates the contribution. Distinguishing those two is a Jacobian or a finite ablation, which is mediation.

**Competition / interference.** Two edits that each raise `L` can lower it together if they share a saturating bottleneck. Structure required: a nonlinear map of the sum. Identified from architecture if that map is known to be nonlinear. The value of `I` still needs local curvature or four interventions.

**Causal sufficiency.** `L` constant on level sets of `P`. Structure required: `L = q ∘ P` on the domain of interest. Identification cannot use evaluation `L` to pick `P` if the claim is independent prediction.

**Recovery after disruption.** Requires a map that restores a coordinate from other coordinates, as in majority vote. Structure required: redundancy plus a correcting aggregator. Not implied by a residual skip, which preserves rather than corrects.

**Decision instability.** First-order: small margin relative to `||∇L||`. A relational score tightens this only if it supplies a strictly smaller Lipschitz constant on a checkable subset. No such constant was derived from `O` or `D` that does not already use `∇L`.

**Target specificity of a repair.** A kernel edit that moves `L` while holding a probe fixed is the failure of probe sufficiency, which Direction 1 measured. The structure is "the probe is not the whole causal state," not "the kernel disagrees with the probe."

Top-down and bottom-up meet on three structures: linear additivity of residual paths, shared nonlinear bottlenecks, and restricted sufficiency of a representation. They do not meet on a scalar coboundary.
