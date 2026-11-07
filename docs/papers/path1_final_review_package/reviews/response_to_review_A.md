# Response to Review A

Review A is `reviews/review_A_mechanistic_interpretability.md`. The manuscript was changed where a saved file supported the finding. No new experiment was run, no letter was changed, and the sealed confirmation data were not opened. Findings that need a model replay or a new probe are declined here and stated as limitations.

## Accepted and changed

1. **Natural-donor diagnostic.** The sentence "we did not measure kernel effects starting from the natural recipient" was literally about SPEC, the matched-divergence study, and NST, and those three were still not run from \(h_R\). What the paper omitted is real. The FIBER archive scores kernel steps around the natural **donor** activation, not the recipient. At L2 1 the mean absolute change in \(Y\) is 0.000316 for the donor-kernel step and 0.000173 for a random kernel step. At L2 4 the donor-kernel change is 0.00151 and the mean JS is \(4.76\times10^{-5}\), against JS 0.01 at \(h_E\) for a correct-kernel step of mean L2 5.298. These means are averages of the saved per-row scores. The anchor logits are not in the archive. The abstract, the FIBER results, the discussion, the limitations, and the conclusion now say that the positive kernel results are results at \(h_E\), and that similar steps barely move the output at the natural donor.

2. **Conclusion scope.** The conclusion now limits "not a sufficient causal description" to edits at the Editor A reference, and it states the natural-donor contrast in the same paragraph.

3. **Protocol history, with a correction to the review.** The git record does support a timing problem. The first frozen matched-divergence protocol (commit `63433f2`, 2026-10-18 20:05) matched wrong and random directions to the correct direction's own JS from \(h_E\), and its gate required a mean contrast of at least 0.05. It was not donor-referenced. The inconclusive letter was recorded at 22:04 (`ddceca1`). The fixed-JS-0.01 protocol, with a 0.02 floor, was committed at 22:20 (`987fb77`), on the same 48 pairs. The manuscript now says this. It does **not** say that +0.0266 fails the 0.05 floor. That floor applied to a different matching target, the correct direction's own divergence, which is much larger than 0.01. Comparing the two numbers would be a different error. Both letters stay: `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` on the first run, and `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED` on the follow-up. The follow-up does not replace the first letter.

4. **Random direction as a baseline.** A random kernel direction at JS 0.01 exceeds Editor A by +0.0223 on 34 of 48 pairs. The 56.04% stage share uses Editor A as the baseline and is unchanged as a ratio of means. The text now says that most of the stage-only gain over Editor A is shared with that unstructured edit, and that the learned-minus-random contrast of +0.0314 is the comparison that removes it.

5. **Cells that were in the package.** SPEC, recomputed from logits: the probe-visible part of \(h_D-h_E\) has mean \(Y\) +0.363 and the full donor activation +0.725, against +0.144 for the kernel part. FIBER factorial cells, means of the saved per-cell scores: \(Y\) is −0.689 at \(h_E\), +0.251 with the probe-visible part, +0.404 with the kernel part, and +0.758 with both. The interaction is −0.586. This is not the VNF estimand and VNF's letter is unchanged. The FRESH console marks the anchor-dependence gate as failed. The Editor-A-anchored switching numbers are not in the excerpt, so they are not reported. SEM's reverse switching contrast on the NST pairs is archival: \(A_W=+0.0058\), interval \([−0.0003, 0.0124]\), which includes zero.

6. **Editor A.** The introduction now says full-space, matching the setup, and says the manifold penalty was off (\(\mu=0\) in the SPEC protocol). The activations \(h_E\) and \(h_R\) are not saved, so \(\|P_N(h_E-h_R)\|\) is not reported, and the optimizer is not named because the protocol records the step count, learning rate, and penalties but not the optimizer.

11. **Exact-board split.** It is now labeled as not in the matched-divergence preregistration, and as descriptive because in NST it tracks earlier positions. No new occupancy numbers were added.

12. **Related work.** The "we did not find a prior study" sentence and the word "tighter" are gone. AlterRep (Ravfogel et al., 2021) and Hazineh et al. (2023) were already added. Makelov, Lange, Geiger, and Nanda (2024) is now cited for dormant pathways under subspace interventions. The remaining works named in the review were not added. A long list of uncompared papers would overclaim how carefully each was read.

17. **Clustering.** The statistics section now says that one recipient game per pair makes recipient clustering the same as resampling pairs, that target games are not clustered, and that subgroup intervals are descriptive.

## Declined, because they need a new experiment or weights that are not in the package

- Rerunning SPEC or the matched-divergence study from \(h_R\) or \(h_D\) (finding 1, the "if possible" clause).
- A fresh probe on \(P_N h\), LEACE erasure, path patching, and projections onto legality or unembedding directions (findings 7 and 8). The title was not changed. The text now says the kernel result is at \(h_E\) and does not say the kernel is board-free. Held-out residual \(R^2\) 0.305 was already reported.
- An on-manifold random control (finding 9). Correct-minus-random was already caveated. PCA-residual means were not added, because the column is described in the file as not a manifold test, and quoting it as one would overclaim.
- Activation norms \(\|h_R\|\) and \(\|h_E-h_R\|\) (finding 10). Those activations are not saved. The training-mean distances in the FIBER controls were not promoted to a scale for the SPEC edits, which are a different sample.
- Replacing Figure 2A (finding 16). The figure specification for this package fixes panel A as the maximum probe change and panel B as total variation, pair-aggregated, relative to Editor A. The roundoff is a check of \(W P_N=0\), and the text already treats it as a check rather than a discovery.
- A target-clustered bootstrap (finding 17). In SPEC and NST each target game appears once, so it would not change those intervals. The FIBER target-sharing is already stated.
- The bound \(|\Delta Y|\le 2\,\mathrm{TV}\) and the aligned-fraction numbers (finding 18). Not added in this pass. The total-variation means are already in the paper.
- Renaming the Lin et al. (2024) authors from the forum byline (finding 19). The Alignment Forum post is signed with the display names the paper uses. Replacing them with legal names needs the post's own byline, which was not rechecked in this pass.
- VNF main effects (finding 14). They are not in the excerpts. The text already reports VNF as a failed gate. It does not claim the interaction was shown to be absent beyond that gate.

## Finding 20

`FINAL_SCIENTIFIC_VERDICT.md`, `RELEASE_MANIFEST.json`, and the audits named in the README were mid-build when the review was written. They are part of the package being finished. The three obstruction letters are listed so that a reader can see they were not changed. They are not results of this paper.
