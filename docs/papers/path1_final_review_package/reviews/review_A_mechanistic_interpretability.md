# Review A — Mechanistic interpretability: "Causal Information Outside a Frozen Board Probe in Othello-GPT"

Reviewer A (adversarial; probing, patching, steering, nullspace/erasure, Othello-GPT). Read-only review of `manuscript/paper_master.md`, the saved arrays under `data/raw/`, `analysis/`, `scripts/p1lib.py`, `figures/*.png`, `audits/`, the protocols under `data/protocols/`, and the console excerpts under `data/archival/`. For one finding I also read the project's git history (`git log` / `git show` on the protocol files, read-only). I did not open any `*confirmation_protocol*` file or `../_superseded/`. I did not run the model. Python checks used `/Users/shkh/opt/anaconda3/bin/python` on the saved CSV/NPZ files. Two web searches were used only to check citations.

## Summary

The paper adds vectors in the 388-dimensional kernel of a frozen 124×512 linear board-probe contrast matrix to an activation that a probe-gradient editor (Editor A) has already moved. It reports that donor-derived kernel directions raise a legal-mass score \(Y=G-Q\) more than wrong-target and random kernel directions, at matched L2 and at matched output JS. The manuscript is unusually honest about its negative results, and its arithmetic reproduces from the saved arrays. My main concerns are about interpretation, and some of them are serious. (i) The package contains a diagnostic that the paper leaves out: the FIBER natural-anchor test. In it, kernel perturbations around *natural* activations changed the output by about two orders of magnitude less than comparable-norm kernel edits at \(h_E\). That is exactly the result the off-distribution critique predicts, yet the Limitations section says that natural-anchor kernel effects were not measured. (ii) The matched-JS study that carries the "target identity still matters" claim was set up after an earlier preregistered matched-JS run on the same pairs came back `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`. The new setup has a lower effect floor (0.02 instead of 0.05). The paper describes this as "restoring" a protocol, but the git record supports that only through a post-hoc audit. (iii) Two headline numbers use Editor A as the baseline: the learned-minus-Editor-A contrast (+0.0537) and the "56% stage share". Both are inflated by generic disruption. A random kernel direction at the same JS already gains about 42% of the +0.0537, and measured against a random-direction floor the stage share falls to about 25%. (iv) Several directly relevant cells sitting in the package are not reported. These are the rowspace-only and full-donor cells of SPEC, the strong negative visible×kernel interaction in FIBER, FRESH's failed anchor-dependence gate, and the reverse switching contrast \(A_W\) on the NST pairs. (v) The framing "information outside the probe" does not separate the non-board-information reading from the familiar redundant-encoding reading, which INLP-style work predicts. The NST map shows the kernel content is partly a linear function of board features. A tight empirical paper is possible here, but the current conclusion is stated more generally than the evidence allows.

## Recommendation

- **Interpretability workshop:** borderline. It would be a weak accept if findings 1–5 are addressed, mostly by reporting data the authors already have and by narrowing the claims.
- **Main-track ML venue (NeurIPS/ICML/ICLR):** reject. The core effect is small (about 0.02–0.03 in \(Y\)) and conditional on an off-distribution reference point, and the most informative manifold diagnostic points against the general reading. The significance of "a 124-dimensional probe does not span every output-relevant direction of a 512-dimensional residual stream" is low without a mechanistic account of what the kernel directions encode.

## Findings

### 1. BLOCKING — The natural-anchor kernel-sensitivity diagnostic is omitted, and Limitations says that measurement does not exist

**Location.** Limitations: "Because every edit starts from \(h_E\) rather than \(h_R\), the effects are conditional on Editor A's change to the decoded board. We did not measure kernel effects starting from the natural recipient activation in the matched-L2, matched-divergence, or donor-free studies." Also Discussion: "We have not shown that \(h_E+t\,\delta\) lies on the natural activation manifold."

**Problem.** The FIBER study in this package did measure kernel perturbations around natural activations, and the paper does not report the result. `data/raw/fiber/08r2_fiber_scientific_report.md`, §F–G: "Mean |ΔY| at ε=1 around donor: 0.00031627…; Same-norm random-null |ΔY| at ε=1: 0.00017…". The report's own §J/§K warns: "Off-manifold hybrids (H01, H10) can inflate effects relative to small natural-anchor moves" and "Small natural-anchor effects versus large hybrid H01 effects would favor distribution-shift over a stable nullspace coordinate." I recomputed from `data/raw/fiber/08r2_fiber_natural_anchor.csv` (768 rows). At ε = 4 (L2 = 4) around the natural donor activation, the donor kernel direction gives mean |ΔY| = 0.0015 and mean JS = 4.8×10⁻⁵. A random kernel direction gives mean |ΔY| = 0.0007 and mean JS = 6.7×10⁻⁶. By contrast, in the matched-JS study a correct kernel direction of mean L2 5.298 already reaches JS = 0.01 from \(h_E\), and a random direction of mean L2 12.265 does too (paper, Table 2). Even with generous quadratic extrapolation from L2 4 to 5.3 or 12.3, the output is roughly 100× more sensitive to kernel edits at \(h_E\) than at a natural activation. This is the central manifold concern, and the package's own data support it. With this result left out, a reader sees "the kernel is causally active" without learning that at natural points the kernel looked nearly inert at these doses.

**Verified.** Yes. I ran pandas groupby on `08r2_fiber_natural_anchor.csv` (columns `family`, `eps`, `dY_vs_donor`, `js_vs_donor`) and read the FIBER scientific report.

**Required change.** Report FIBER §F–G in Results, including |ΔY| and JS versus ε for donor and random kernel directions at natural anchors. Delete or correct the "We did not measure…" sentence. State plainly that output sensitivity to this probe's kernel was far lower at natural activations than at \(h_E\). If possible, rerun SPEC and matched-JS from \(h_R\) (and from \(h_D\)) and report the L2 needed to reach JS 0.01 at natural anchors. Until then, every claim should be stated as "at Editor-A-edited activations".

### 2. BLOCKING (main track) / MAJOR (workshop) — The conclusion and the first Discussion sentence go beyond what the evidence supports

**Location.** Conclusion: "The justified conclusion is the limited one: this probe is not a sufficient causal description of the residual information that affects legal-move outputs." Discussion: "For this frozen linear board probe, the residual stream is not a single occupancy code plus unused orthogonal content." Abstract: "this frozen probe does not exhaust the residual information that affects legal-move outputs".

**Problem.** As stated, these sentences are unconditional. The evidence is conditional on three things: (a) the reference point \(h_E\), which Editor A produced with its manifold term switched off (finding 6); (b) large edits, since SPEC's mean L2 is 18.33 while natural activations sit about 21.9 from the training mean (finding 10); and (c) a test at natural anchors that found little kernel sensitivity (finding 1). Read literally, "not a sufficient causal description" also holds trivially for any 124-dimensional readout of a 512-dimensional stream feeding an unembedding (finding 8). The non-trivial claim is the target-specific one, and that holds only at \(h_E\).

**Verified.** Judgment, supported by the numbers in findings 1, 6 and 10.

**Required change.** Rewrite the Abstract, Discussion and Conclusion along these lines: "At activations already edited by a probe-gradient editor, directions in this probe's kernel shift legal-move probability in a target-aligned way. At natural activations, comparable-norm kernel perturbations had negligible effect (FIBER §F–G)." Remove "the residual stream is not a single occupancy code plus unused orthogonal content", or limit it to \(h_E\).

### 3. MAJOR — The history of the matched-JS protocol amounts to a re-run after an INCONCLUSIVE letter, with a lower floor, and the paper does not disclose it adequately

**Location.** Results, "Target identity still matters…": "An earlier execution of this design did not follow that protocol; a forensic audit recorded the deviation, and the letter on that run remains `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`. We do not use its numbers." Statistics: "Coverage gates and effect floors were preregistered before outcomes." Appendix A: "The matched-divergence study required … a mean correct-minus-wrong contrast of at least 0.02".

**Problem.** `data/protocols/08r2_fiber_cost_preregistration.md` is the only matched-JS preregistration in the repo that predates matched-JS outcomes (commit `63433f2`, 2026-10-18 20:05). It specifies donor-referenced JS matching and gate item 3, "Mean \(\Delta_{\mathrm{JS}}\ge 0.05\)". The "fixed-JS restoration" (`08r2_fiber_cost_fixed_js_preregistration.md`, config `behavior_floor: 0.02`) was committed at 22:20 (`987fb77`). That is after commit `ddceca1` (22:04), "Treat incomplete JS matching as a COST scientific outcome", which implies the COST run had already produced its INCONCLUSIVE outcome on the same 48 pairs. The forensic audit (`612067c`) asserts that a "requested" fixed-JS protocol existed. However, it also says the in-repo preregistration "is not independent evidence of a 0.01 protocol", and nothing in the package documents the 0.02 floor before outcomes. The observed contrast (+0.0266, Table 1) would fail the 0.05 floor of the only pre-outcome document. This does not rewrite the INCONCLUSIVE letter, but it lets the restored run stand in for it under a laxer gate. The headline sentence "Target identity still matters when output divergence is matched" rests entirely on that run.

**Verified.** Yes. I read both protocol files and the JSON config in the package, and ran read-only `git log` / `git show` on `docs/experiments/08r2_fiber_cost*_preregistration.md` and `configs/toy_othello_08r2_fiber_cost*.json`.

**Required change.** In the matched-JS section, state that: (1) the first frozen matched-JS protocol used pair-specific donor-referenced JS and a 0.05 floor, and was INCONCLUSIVE; (2) the fixed-JS protocol was committed after that outcome on the same pairs, with a 0.02 floor; (3) the observed +0.0266 does not meet 0.05. Label the matched-JS result post hoc and exploratory, not preregistered, and qualify the sentence "Coverage gates and effect floors were preregistered before outcomes" accordingly. Keep `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` as the matched-JS letter of record next to `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`, and do not present the second as resolving the first.

### 4. MAJOR — Editor A is the wrong baseline: generic disruption inflates "learned − Editor A" and the "56% stage share"

**Location.** Table 1 row "NST (donor-free) | learned − Editor A | +0.0537". Stage section: "STAGE_ONLY therefore accounts for 56.04% of FULL's mean improvement over Editor A." Abstract: "About 56% of the map's mean gain over Editor A was already obtained by a map that sees only player and occupancy." Also "Much of its average gain over Editor A is a stage correction".

**Problem.** Any kernel edit calibrated to JS 0.01 raises \(Y\) above Editor A, because baseline \(Q\) is large (Editor A's \(Q\approx0.6\)–0.8) and spreading probability mass lowers it. From `analysis/nst_pairs.csv`, random kernel direction minus Editor A is +0.0223 (34 of 48 pairs positive), which is 41.5% of FULL − Editor A (0.0537). In the matched-JS study, from `analysis/fixed_js_pairs.csv` merged with the SPEC `editor_A` rows, random minus Editor A is +0.0232 (42/48 positive), wrong minus Editor A is +0.0196, and correct minus Editor A is +0.0462. So half of the correct direction's gain over Editor A is reproduced by a random kernel vector. Taking the random direction as the disruption floor, STAGE_ONLY − random = 0.0301 − 0.0223 = 0.0078 and FULL − random = 0.0314. The stage share above a generic-disruption floor is therefore about 25%, not 56%. The 0.0301 is archival, from `sem_stage_share_subgroups.txt`; the 0.0223 is recomputed from NST arrays on the same pairs and calibration. The 56% figure and the description "stage correction" mix up generic disruption with stage content.

**Verified.** Yes. Arithmetic in Python on `analysis/nst_pairs.csv`, `analysis/fixed_js_pairs.csv` and `data/raw/spec/fiber_spec_interventions.csv`. The STAGE_ONLY value is archival.

**Required change.** In Table 1 and the Abstract, replace or supplement "learned − Editor A" with "learned − random". Report "random − Editor A" explicitly as the generic-disruption gain. Report the stage share relative to the random floor alongside the Editor A version, and remove "a stage correction" unless it survives that floor. Explain in the Response score section that \(Y\) has a positive bias under any disruption when \(Q\) is large at baseline.

### 5. MAJOR — Directly relevant cells and contrasts in the package are not reported

**Location.** SPEC results paragraph, which reports only correct/wrong/random/Editor A/recipient. The VNF paragraph: "Reconstructing the probe-visible state accurately did not produce the hypothesized interaction". The FRESH paragraph.

**Problem.** Four omissions:
- (a) SPEC saved a `correct_rowspace` cell (Editor A plus the rowspace part of \(h_D-h_E\)) and a `full_donor` cell. Mean \(Y\) recomputed from `fiber_spec_interventions.csv` is +0.363 for `correct_rowspace`, +0.725 for `full_donor`, +0.144 for `correct_null` and −0.639 for `editor_A`. Measured from Editor A, the probe-visible donor component moves \(Y\) by +1.00, the kernel component by +0.78, and both together by +1.36. Both components matter and they are sub-additive. A reader needs this 2×2 to judge how much "outside the probe" matters relative to the probe.
- (b) FIBER reports a large negative visible×kernel interaction, "Mean interaction I: −0.5859… ci95 [−0.748, −0.437]" (`08r2_fiber_scientific_report.md` §C–E). The paper reports only VNF's null interaction. Readers should see that one interaction design found a strong interaction and another found none.
- (c) `fresh_full_run.txt` shows the gate `'anchor_dependence_ok': 0` next to `'natural_symmetric_ok': 0`. The design was a three-anchor test ("Natural \(h_R\); Editor A for T; Editor A for W", `08r2_symmetric_target_preregistration.md`). The paper reports only the natural-anchor arm and never mentions the anchor-dependence gate or the Editor-A-anchored switching values.
- (d) `sem_full_run.txt` reports, on the NST pairs with the Editor A anchor, `A_W` = +0.0058 with CI [−0.0003, 0.0124], 62.5% wins. So the reverse switching contrast already failed to exclude zero on the reference-point design the positive studies use. The paper does not report this.

**Verified.** Yes. pandas on `fiber_spec_interventions.csv`; I read the FIBER report and the console excerpts.

**Required change.** Add a 2×2 table (rowspace × kernel, from \(h_E\)) for SPEC. Report the FIBER interaction alongside VNF's. Report what `anchor_dependence_ok` tested and its values, or state that the values are not in the excerpts. Report SEM's `A_W` on the NST pairs. None of this changes any letter. `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED` and `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED` stand as written.

### 6. MAJOR — Editor A is an unregularized probe-gradient edit; what \(h_E\) is needs to be stated precisely

**Location.** Introduction: "Editor A, a probe-space residual editor from earlier work on this model, moves \(h_R\) toward the intended target board as read by the probe". Setup: "Editor A is a full-space residual editor with L2 budget 16, 40 gradient steps, learning rate 0.4, and norm penalty 0.01". Appendix A: "Editor A minimizes a board-decoding loss…".

**Problem.**
- (a) The manuscript calls Editor A both "probe-space" and "full-space".
- (b) The protocol shows the manifold term was disabled. `08r2_fiber_spec_preregistration.md`: "40 steps, lr 0.4, \(\lambda=0.01\), \(\eta=1.0\), \(\mu=0\)", and the COST config has `"mu_manifold": 0.0`. So \(h_E\) is, by construction, an adversarial-style input to the probe with nothing holding it near the data manifold. The paper does not say this.
- (c) For a softmax linear probe, the gradient of cross-entropy with respect to \(h\) lies in the span of the class-contrast rows. The same holds for the norm penalty on \(h-h_R\) started at \(h_R\). If Editor A uses plain gradient descent or projected GD, then \(P_N h_E = P_N h_R\) exactly. In that case \(h_E\) is a hybrid: an edited rowspace attached to the recipient's natural kernel. The "correct" SPEC edit at \(t=1\) is then exactly \(P_R h_E + P_N h_D\), i.e. a swap of the kernel coordinates toward the donor. That reading makes the target-specific effect look like "restoring consistency between kernel and rowspace" rather than "information the probe misses". If Editor A uses Adam or another coordinate-wise optimizer, this identity fails, and the paper must then report how much of Editor A's edit falls in the kernel.
- (d) Editor A is essentially the probe-gradient intervention of Li et al. (2023), applied with a linear probe at a single layer and a single token. The paper should say so, and should discuss why Li et al.'s interventions succeeded while 08R2 was negative. Li et al. intervened at multiple successive layers; here only one layer at the last token is edited, while attention can re-read board information from earlier positions.

**Verified.** (b) verified from the protocol and config. (c) is an analytic argument that I could not test, because \(h_R\) and \(h_E\) are not saved. (a) is quoted text.

**Required change.** Fix the contradictory description. State that μ = 0 and name the optimizer. Report \(\|P_N(h_E-h_R)\|\). If it is zero, rewrite the framing around the hybrid \(P_R h_E + P_N h_D\). Relate Editor A to Li et al.'s intervention and discuss the single-layer limitation.

### 7. MAJOR — "Information outside the probe" is not separated from redundant linear encoding of the board (the INLP/LEACE alternative)

**Location.** Title. Abstract: "this frozen probe does not exhaust the residual information…". Introduction: "The phrase does not mean that no decoder could read the change".

**Problem.** The disclaimer is there, but the title and the conclusion's wording ("information outside") invite the reading that the kernel holds non-board information. The INLP literature (Ravfogel et al., 2020; Elazar et al., 2021) is built on the observation that after one nullspace projection, a fresh linear probe can usually still recover the attribute. Board information is expected to remain linearly decodable from \(P_N h\). The paper's own NST map shows that kernel residuals are linearly predictable from board and legal-mask features: held-out residual \(R^2\) = 0.305, a recorded value given in the NST section. So "outside the probe" mostly means "in directions this particular logistic regression did not use", and the paper does not test the more interesting "non-board" reading. In addition, the FIBER gradient diagnostic (`08r2_fiber_controls.csv`, column `nullspace_grad_energy_frac`, 32 rows, note "mechanistic_sensitivity_not_an_editor") gives a mean of 0.680 for the fraction of output-gradient energy lying in the kernel. That is below the 388/512 = 0.758 expected for a random subspace of the same dimension. On this diagnostic the kernel is not specially output-relevant; it is about what its dimension predicts. The column's exact definition is not documented in the package, so treat this as indicative.

**Verified.** The R² is quoted from the paper and the NST files. The gradient-energy fraction was recomputed from the CSV, but I could not confirm its definition.

**Required change.** (1) Train a fresh linear probe (and LEACE-erase, then re-probe) on \(P_N h\) for the board, the legal mask and the player, and report accuracy. (2) Retitle, e.g. "Kernel directions of a frozen board probe still shift legal-move predictions in Othello-GPT", and say explicitly that the kernel is not board-free. (3) Report the gradient-energy diagnostic against its dimension baseline and define it.

### 8. MAJOR — An untested simpler explanation: the kernel holds legality/flip/last-move features or unembedding-aligned directions

**Location.** Discussion: "The kernel results say that the rowspace of this probe is not the whole causal story." Setup: "The intervention site is the residual stream at `blocks.6.hook_resid_mid`".

**Problem.** At layer 6 resid_mid only MLP6, layer 7 and the unembedding remain. Nanda et al. (2023) showed that "flipped" and related features are linear, and Lin et al. (2024) report mid-to-late MLP neurons that classify legality with high direct logit attribution. The plainest explanation of "kernel edits shift legal-move mass toward the target" is that \(P_N(h_D-h_E)\) carries the donor's partly computed legal-move or logit directions. Some of the effect may even pass directly through the unembedding (after the final LayerNorm), not through any board-like state. SEM's own result, `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`, fits this explanation. Nothing in the paper separates the direct path from the mediated path.

**Verified.** Judgment. Checking it needs weights that are not in the package.

**Required change.** Report the fraction of the (LayerNorm-adjusted) unembedding row space and of legal-move-probe directions that lies in \(P_N\). Decompose each effect into the direct logit path versus the path through MLP6/L7 by path patching. Project \(\delta_{\mathrm{correct}}\) onto known legality/flipped/last-move probe directions and test whether the residual still carries the target-specific effect.

### 9. MAJOR — Random kernel controls are off-manifold while donor directions are not, so correct − random is confounded

**Location.** "A random direction is \(P_N\) applied to a seeded Gaussian vector." Table 1 rows "correct − random". SPEC: "A random nullspace vector of the same length barely moved the output".

**Problem.** The FIBER controls record a PCA-residual diagnostic (`pca_residual_l2`, described in the file as a "Diagnostic distance from training activation stats, not manifold membership"). Recomputed means: natural recipient (`noop`) 14.57; donor-kernel hybrid H01 12.98; wrong-donor kernel 14.43; random kernel at matched L2 20.62; random rowspace 17.44 (`08r2_fiber_controls.csv`, `08r2_fiber_factorial_interventions.csv`). Gaussian kernel vectors leave the data's principal subspace, while donor-derived vectors stay inside it. A random direction therefore differs from the correct one in "manifold-ness" as well as in target content. Wrong-donor directions are the right control. The paper reports correct − random as a headline, with no warning, even though in matched-JS correct − random is positive in only 30/48 pairs and its CI includes zero in category A (+0.0106, [−0.0033, 0.0246]). Wrong minus random at matched JS is −0.0036 (21/48 positive, from `analysis/fixed_js_pairs.csv`): wrong-target donor directions do no better than Gaussian noise.

**Verified.** Yes, pandas on the FIBER and matched-JS CSVs.

**Required change.** Report PCA-residual (or Mahalanobis) diagnostics for every direction class in every study. Add an on-manifold random control, for example kernel components of differences between random natural activations matched on player and occupancy. Demote correct − random from Table 1, or caveat it. Report wrong − random.

### 10. MAJOR — Edit magnitudes are not placed on the scale of natural activations

**Location.** SPEC: "all rescaled to the same L2 norm (mean 18.33)". Matched-JS: "mean L2 was 5.298 for correct directions, 6.368 for wrong directions, and 12.265 for random directions."

**Problem.** The paper never reports \(\|h\|\) or the natural spread of activations. From the FIBER controls, natural recipient activations lie on average 21.85 (L2) from the training mean (`noop` rows, `l2_to_train_mean`), and Editor A's own edit is about 11.8 (SPEC `editor_l2`). A SPEC edit of 18.33 is comparable to an activation's entire deviation from the mean, and the matched-JS random edits are more than half of it. Readers cannot judge "small edit" versus "replace the activation" without this.

**Verified.** Yes, CSV recomputation.

**Required change.** Report \(\|h_R\|\), \(\|h_E-h_R\|\), \(\|\delta\|\) and \(\|\delta\|/\|h\|\) per study, plus the distribution of natural within-split distances between matched positions.

### 11. MAJOR — The "Editor A exact" subgroup is post hoc and confounded with game stage

**Location.** Figure 3 caption and text: "It is clear in category B … and in pairs where Editor A decoded the exact target board". NST: "The transfer was weaker where Editor A failed to decode the exact target board… We treat this split as a limitation of transfer".

**Problem.** The fixed-JS preregistration and scientific report define only A/B subgroups. The exact/nonexact split appears in neither (`08r2_fiber_cost_fixed_js_preregistration.md`; `fixed_js_scientific_report.md` lists only A and B). In NST, exactness is heavily confounded with stage. Exact pairs have mean Hamming 10.75 and mean occupancy 15.0; nonexact pairs have 33.7 and 37.7 (`null_state_scientific_report.md`, "Exact"/"Nonexact" blocks). The split is effectively early-game versus late-game. Ten or more subgroup intervals are reported with no multiplicity adjustment.

**Verified.** Yes, by reading the protocols and reports.

**Required change.** Label the exact/nonexact split as post hoc. Report its confounding with occupancy and Hamming distance. Either drop the causal language ("weaker where Editor A failed") or show the split holds after conditioning on occupancy.

### 12. MAJOR — Related work leaves out the most relevant methodological critiques; one sentence still reads as a priority claim

**Location.** Related work, last paragraph: "We did not find a prior study that edits within the kernel of one frozen board probe under target-specific controls, with and without a target activation at evaluation, and reports the completeness tests that fail. That is an observation about the sources we checked, not a priority claim." Steering paragraph: "which is a tighter control on the alternative that an edit merely disrupts the next-move distribution."

**Problem.** The manuscript never uses "first", "novel" or "for the first time" (`rg` over `paper_master.md`). The "We did not find a prior study…" sentence still works as a priority claim, and the disclaimer after it does not undo that. "Tighter control" is a comparative claim with no comparison behind it. Omitted work that a hostile reviewer will cite:
- Makelov, Lange & Nanda (2024, ICLR), "Is This the Subspace You Are Looking For?". Subspace patching can activate dormant parallel pathways and produce off-distribution effects, which is directly on point for kernel edits at \(h_E\).
- Kumar, Tan & Sharma (2022, NeurIPS), "Probing classifiers are unreliable for concept removal and detection".
- Ravfogel et al. (2021, CoNLL), AlterRep: counterfactual interventions in a probe's rowspace and nullspace and their behavioral effect, the closest methodological precedent.
- Ravfogel et al. (2022), RLACE.
- Hazineh, Zhang & Chiu (2023), linear world models in Othello-GPT.
- Karvonen et al. (2024), dictionary learning on board-game models.
- Zhang & Nanda (2024, ICLR) and Heimersheim & Nanda (2024), best practices and off-distribution pitfalls in activation patching.
- Chan et al. (2022), causal scrubbing: resampling ablations to stay on-distribution.

**Verified.** The absence of these works was checked by reading `references.bib` and the reference list. Their relevance is judgment.

**Required change.** Add these works and position the paper against AlterRep and Makelov et al. in particular. Delete the "We did not find a prior study…" sentence, or replace it with a neutral description of how the design differs from AlterRep and INLP. Delete "tighter".

### 13. MINOR — Abstract and Figure 5 lead with the positive switching means

**Location.** Abstract: "On a fresh cohort, both target-switching contrasts were positive on average, but both were positive together in only 13 of 32 triplets, below a preregistered 60% gate." Figure 5A.

**Problem.** The letter (`SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`) is preserved, but the framing puts the positive means first. \(A_W\) was positive in only 56% of triplets (`fresh_full_run.txt`, `win_frac` 0.5625), close to chance. Figure 5A draws both intervals well clear of the 0.005 floor, and the failing 60% criterion appears only in a text box, so a reader skimming the figure sees success.

**Verified.** Yes, by reading the figure and the console excerpt.

**Required change.** Lead with the gate failure ("Target switching failed its preregistered joint gate: both contrasts were positive in 13/32 triplets (gate 60%); \(A_W\) was positive in 56%"). Add a Figure 5 panel showing the both-positive fraction against the 60% line. Avoid putting panels A and B side by side on x-scales that differ by 10×.

### 14. MINOR — VNF's null interaction is presented as informative without its main effects or a dose argument

**Location.** "Both fall below the preregistered +0.005 floor … Reconstructing the probe-visible state accurately did not produce the hypothesized interaction in \(Y\)."

**Problem.** VNF used fixed component norms of 3.5 (`08r2_visible_null_factorial_preregistration.md`: "rho = nu = 3.5 … Total L2 sqrt(24.5)"). The paper does not state which anchor VNF used. If it was natural, finding 1 implies kernel main effects at this dose should be tiny, so the interaction gate had little room to pass. The letter stands. The paper should not, however, cite VNF as positive evidence that no interaction exists.

**Verified.** Protocol read; the main effects are not in the excerpts.

**Required change.** State the VNF anchor and report its main effects \(d_T(T)\), \(d_T(W)\), etc., or say they are unavailable. Describe VNF as a failed gate, not as evidence of absence.

### 15. MINOR — The gradient baseline is undefined and its exclusion is unexplained

**Location.** "A gradient baseline reached the JS tolerance on only 7 of 48 pairs and is not compared."

**Problem.** The paper never defines the gradient direction. `null_state_scientific_report.md` reports `'gradient_unavailable': 0`, which reads oddly next to "7 of 48". On the 7 calibrated pairs, the gradient edit reached JS 0.01 at mean L2 0.66, against 5.20 for the learned edit (`null_state_interventions.csv`). There its mean \(Y\) was −0.369, against −0.383 learned and −0.378 oracle (`analysis/nst_pairs.csv`). That is a striking sensitivity contrast, and it deserves a sentence.

**Verified.** Yes, CSV recomputation.

**Required change.** Define the gradient baseline. Explain why it failed to calibrate on 41 pairs. Report the 7-pair values with a selection caveat, or remove the mention.

### 16. MINOR — Figure 2A and the probe-invariance numbers present a construction as a result

**Location.** Abstract: "unchanged to within \(2.2\times10^{-14}\)". Figure 2A.

**Problem.** \(WP_N=0\) guarantees invariance; the roundoff check is a unit test, not evidence. A full figure panel on a log axis from 10⁻¹⁷ to 10⁻¹² takes space that could show an informative control, such as how a fresh probe or a nonlinear probe reads the edit.

**Verified.** Judgment.

**Required change.** Move the roundoff numbers to Appendix or Table 6. Replace Figure 2A with an alternative-decoder readout change, or with the natural-anchor sensitivity from finding 1.

### 17. MINOR — Statistics: clustering is nominal and targets are not clustered

**Location.** Statistics: "Intervals are 95% percentile bootstrap intervals clustered on recipient game".

**Problem.** SPEC, matched-JS and NST each have 48 distinct recipients (`spec` prereg: "48 distinct recipients"; `n_clusters: 48` in the reports), so "clustered" equals an i.i.d. bootstrap over pairs. Target games and wrong donors are not clustered. In FIBER, the 24 category-A pairs share 2 targets, and the paper notes this. There is no multiplicity control across many subgroup intervals.

**Verified.** Yes, from the reports' `n_clusters` fields.

**Required change.** State that clustering is equivalent to resampling pairs. Add a target-clustered or two-way bootstrap where targets or donors repeat, and state how subgroup intervals should be read.

### 18. MINOR — Report \(Y\) changes relative to total output movement

**Location.** Response score section.

**Problem.** \(|\Delta Y|\le 2\,\mathrm{TV}\). At matched JS, mean TV is about 0.077–0.080 for every class (Table 2), so the largest possible \(|\Delta Y|\) is about 0.16. The fraction of output movement that is target-aligned, \((Y-Y_E)/(2\,\mathrm{TV})\), is 0.32 for correct directions and 0.15 for random ones (my recomputation from `analysis/fixed_js_pairs.csv` plus SPEC `editor_A` rows). This normalized number is more interpretable than raw 0.02–0.03 differences.

**Verified.** Yes, recomputed.

**Required change.** Report the bound and the aligned fraction per direction class.

### 19. MINOR — Citation error in Lin et al. (2024)

**Location.** References: "Lin, J., JackS, Karvonen, A., & Can. (2024)."

**Problem.** "Can" and "JackS" are forum display names. The authors are Jennifer Lin, Jack Schonbrun, Adam Karvonen and Can Rager (Alignment Forum post page; also cited that way in arXiv:2511.00059). The Jacopin (2026) citation checks out: BlackboxNLP 2026 proceedings table of contents, pp. 986–1006.

**Verified.** Yes, by web search.

**Required change.** Write it as "Lin, J., Schonbrun, J., Karvonen, A., & Rager, C. (2024)."

### 20. MINOR — The package README promises files that are missing; some letters are unexplained

**Location.** `README.md`: "The reasons are in `FINAL_SCIENTIFIC_VERDICT.md`"; layout rows for `reviews/`, `RELEASE_MANIFEST.json`, and "the claim–evidence ledger, provenance inventory, scientific and caption audits, novelty review…". The paper's final list of letters includes `0A = C / PARTIAL` and `SECTION_5_3 = PARTIAL`.

**Problem.** `FINAL_SCIENTIFIC_VERDICT.md`, `RELEASE_MANIFEST.json` and `reviews/` were absent, and `audits/` held only two files (`find` listing). The letters `0A`, `SECTION_5_3` and `OBSTRUCTION_READY` are listed but never explained, so readers cannot tell what they refer to. They are not reinterpreted, only opaque.

**Verified.** Yes, by file listing.

**Required change.** Add the files or remove the references. Add a one-line gloss for each official letter without changing it.

### Letters check (requested)

- `08R2_VALIDATION_COMPLETE — NEGATIVE`: preserved and not rescued. The paper says "Nullspace edits … should not be read as having solved that failure." Good.
- `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`: preserved. The framing is mildly softened (finding 13), and the anchor-dependence gate is omitted (finding 5c).
- `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED`: preserved and not rescued. The paper should add FIBER's opposite-sign interaction and VNF main effects (findings 5b, 14).
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`: preserved in name, but in effect replaced by a later run on the same pairs with a lower floor (finding 3). This is the one place where the letter is functionally rescued.
- `OBSTRUCTION_READY = NO`: preserved. No sentence implies obstruction readiness.

## What the paper does well

- The numbers reproduce. Every recomputed contrast I spot-checked (SPEC, matched-JS and NST pooled values and subgroups) matches `audits/recomputation_report.md` and the original runs. The evidence-level labels (recomputed / recorded / archival) are clear and honest.
- The negative results are given full weight: absolute \(Y\) stays negative (learned \(Y>0\) in 1/48), the switching gate fails, the interaction is null, and the 08R2 negative stands. The sentence "These failures belong in the main argument" is the right attitude.
- The paper consistently states that probe invariance and JS are measured against \(h_E\), not \(h_R\), and it reports that the patched probe readout differs from \(h_R\)'s by ≥ 0.93.
- Wrong-target controls are matched on player, occupancy, legal-set size and prefix length, chosen with frozen seeds, and scored on the intended target's sets. That is the right control for target specificity.
- The SPEC weakness is acknowledged ("the large correct-minus-random contrast partly compares an edit that changes behavior with one that hardly does"), and matching divergence is a sensible second strength control.
- The donor-free map is fit on training games with grouped CV and frozen before the oracle is computed, and the learned-minus-wrong identity is checked to about 10⁻¹⁵.
- The paper explains why the learned–oracle score correlation (r = 0.998) is not evidence of a shared mechanism, which is a good catch.
- There is no "first/novel" language. Lin et al. is correctly flagged as not peer reviewed, and Jacopin (2026) is described accurately.
