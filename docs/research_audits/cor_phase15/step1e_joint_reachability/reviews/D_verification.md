# Reviewer D — robustness verification

Same-context pass, not an external referee.

The linear theorem is a support function that affine arithmetic and zonotope verifiers already compute for affine maps, up to the choice of norm. Citing Comba and Stolfi for the shared-symbol idea, and Singh, Gehr, Mirman, Püschel, and Vechev for DeepZ, is the right reduction. The package does not pretend to have a new abstract domain.

The line theorem is exact, so it is stronger than an enclosure on its family and useless as a template for a ball. A verification reviewer should not score it as a certified-robustness method. There is no over-approximation soundness proof for a set larger than the segment, and the notes say so.

The float tests are not a soundness proof. The proof is the case split. That division is acceptable if the verdict does not say "floating-point certified." The chosen verdict does not say that.

Beating `1/√ε` Lipschitz constants would have been a false win. The baseline is the marginal sum. I agree with that choice.
