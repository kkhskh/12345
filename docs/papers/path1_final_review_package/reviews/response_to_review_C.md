# Response to Review C

Review C is `reviews/review_C_empirical_ml.md`. It was written from a copy taken while the manuscript was still being edited, so its reproduction failure is the stale build it copied. The current package rebuilds the PDF from the current Markdown, and `scripts/verify_paper_claims.py` passes.

## Changed since that copy, including for this review

The PDF and TeX are regenerated from the current manuscript. Figure 1's caption no longer says that every study adds its edit at the Editor A reference. The switching study adds its edit at the natural recipient activation, and both the caption and the setup say so. Figure 4 draws the 0.01 floor only on the learned-minus-wrong row, which is the contrast the floor applies to.

Editor A is described as a full-space editor with the manifold penalty off in the abstract, the introduction, and the conclusion. The short names FIBER, SPEC, NST, SEM, FRESH, and VNF are expanded once, and SEM is distinguished from a standard error. The README includes the FIBER letter and says that the last three letters belong to a separate program.

The verifier now requires six directional sentences to be present, including that the absolute score stayed negative, that switching is below the 60% gate, and that kernel steps barely move the output at a natural donor activation. Reversing one of those sentences fails the verifier. It still does not understand arbitrary rewordings.

The contrast matrix \(W\) is not in the package. The text now says that \(W P_N=0\) is how the projector was built, and that the check available here is the saved probe probabilities.

## Not changed

The 3.4% "fraction of the gap to the full donor" is not reported. The full-donor and probe-visible means are from SPEC, at the matched-L2 dose. The matched-divergence contrasts are at JS 0.01. Dividing one by the other mixes two interventions. Both sets of numbers are in the paper separately.

A control that restores the recipient's own kernel component, other probes, other layers, and a second editor were not run. The archival console excerpts are still the chat record of the original runs. Their hashes match the files in the package. They do not become stronger evidence by being hashed. The release ZIP and the clean-extraction logs are not part of this review response. They are the remaining packaging step.
