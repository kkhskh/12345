# R2 review: causal identification and Transformer architecture

Scope:
- `COMPUTATIONAL_CLASSES_AND_EQUIVALENCE.md` (CCE);
- `REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md` (RCI);
- `scripts/s12_operations.py`, `scripts/abstraction_shift.py`, `scripts/pp2_mechanism.py`, `src/pp2_sig.py`, `scripts/class_shapley.py`;
- `logs/s12/*.json`.

How the checks were run:
- Reviewer scripts are in `/tmp/r2check/` (`chk.py`, `abs.py`, `probe.py`); the outputs are `chk_s12.json` and `chk_s10.json`. They were not added to the package.
- `chk.py` builds the tag basis exactly as `s12_operations.py` does. It uses the same checkpoints and the same evaluation stream (7,700,000, N = 1000).
- Its reproduced numbers match the logs: s12 tag@all36 IIA is 0.474; the mean-ablation KL to own-Bayes is 0.133 for s12 and 0.111 for s10.

## What the hooked computation actually is (reference for the findings)

- `fwd.forward` is pre-LN:
  - `z = LN(x)`; Q, K and V are all computed from z;
  - `head_out = (softmax(QKᵀ/√e) V) W_O`, so head_out is taken after W_O;
  - `x += Σ_h head_out`, then the MLP on `LN(x)`, then the final LN, then `WA` at positions 36–41.
- Attention is unmasked.
- In the last layer, a source token's residual reaches the answer **only through its K and V** in the layer-2 attention. The source tokens' own MLPs and outputs are never read.
- Patching readout q at `resid_pre2` changes q's query, its skip path, its MLP input and its final-LN input all at once.
- The patch `a + (src − a) @ P` with symmetric orthogonal P replaces exactly the P-component. I checked Pt: it is symmetric to 0.0, idempotent to 3e-8, and has trace 2.
- **However, the patched vector passes through LN before Q, K and V.** A "2-D" patch is therefore not a 2-D intervention on the attention inputs (see F4).

## Findings

### F1 (minor, hygiene). The projector-union fix is correct, and every reported number comes from a correct run. Buggy rows are still on disk unlabeled.

**Bug mechanics.** Elementwise `max(0, Pt)` clips about 50% of Pt's entries, which are negative (`Pt_neg_frac = 0.502`). It clips them even for single-projector conditions, because Pm starts at zero.

**Current code is correct.** Lines 46–47 implement a correct nested union. My independent rerun of tag@all36 on the same stream gives 0.474, identical to `ops_s12_tagfix.json`.

**Which rows were affected.**
- Unaffected: the full-residual conditions (`max(0, I) = I`), the head conditions, `meanabl` (it calls `_pm` directly with `Pt`) and `head_meanabl`.
- Every CCE table entry traces either to an unaffected row of `ops_s1x.json` or to `*_tagfix.json`. The head-ablation table comes from the tagfix files; `ops_s12.json` has no `head_meanabl`.

**Remaining problems.**
1. `ops_s12.json` and `ops_s10.json` still hold the buggy rows `tag@all36`, `tag@own6` and `full@other30+tag@own6` (for example, s10 tag@own6 = 0.221) with no marker.
2. `ops_s12_atlasstream.json` is a **buggy-run** file (tag@all36 = 0.237). It is unlabeled and not cited.
3. Consequently, "fixed tag@all36 (0.474) reproduces the atlas (0.455)" compares across streams: 7.7M against 7.6M. The fixed code was never run on the atlas stream, and the `/tmp` atlas re-execution is not kept.
4. Recommendation: rename or annotate the buggy files.

### F2 (major). The seed-12 reconstruction omits a fourth channel: the other readout tokens.

CCE §1 gives head-0 attention as 0.10 own / 0.54 other. **The remaining 0.36 is unreported.** My measurement of seed-12 head 0 at readout q:

| destination | attention mass |
|---|---|
| other readout tokens (36–41, excluding q) | **0.29** |
| itself | 0.05 |
| episode token | 0.02 |

Head 1 puts 0.09 on other readouts. In seed 10 only head 1 does this (0.21); heads 0, 2 and 3 put less than 0.001 there.

The interchange results agree:

| condition | seed 12 IIA | seed 10 IIA |
|---|---|---|
| full@all36 + readout q + episode token | 0.897 | 0.983 |
| full@all36 + all 6 readouts | **0.989** | 0.996 |

So in seed 12 about 0.09 IIA is carried by the *other questions' readout residuals*. Those residuals already aggregate other-question evidence in layers 0–1.

Consequences:
- The "three channels" list is incomplete.
- "Cross-question rereading" is partly a reread of other questions' **readout states**, not only of their source tokens.
- The claim "Class C is therefore not a different operation from A′. It is A′ plus a parallel rereading channel" should be weakened. It rests on n = 1 seed and an incomplete channel decomposition.

### F3 (major). The random-2-D control for tag necessity is not variance-matched. A variance-matched control removes as much accuracy in seed 10.

CCE §3 and RCI §5 say a random 2-D subspace removes ≤ 0.03. But the random subspace holds only 2.8–4.6% of the centred source-token variance, against **22% (s12) and 34% (s10)** for the tag. The tag also holds 42–56% of the uncentred norm.

My control mean-ablates the top-2 principal components orthogonal to the tag (24–29% of variance). Same pooled mean, same positions:

| | tag drop_frac | control drop_frac | KL to own-Bayes, tag / control (clean 0.46–0.47) | entropy, control |
|---|---|---|---|---|
| s12 | 0.93 | 0.18 | 0.13 / 0.29 | — |
| s10 | 0.64 | **0.71** | 0.11 / 0.37 | 0.84 (above own-Bayes 0.82) |

In s10, the control's agreement with own-Bayes on untied cases drops to 0.57.

So in seed 10, accuracy-drop necessity does **not** single out the tag. What does single it out is *how* the model fails:
- tag ablation pulls the output toward the own-question posterior;
- a high-variance control destroys the output.

Implications:
- "Necessity" should be reported against a variance-matched control, and with an own-Bayes-directed metric rather than accuracy drop.
- The sentence "the nearest candidate ... removes most of the relational gain in every Transformer seed ... a random 2-D subspace removes ≤ 0.03" overstates how specific the result is.

### F4 (major). Pre-LN renormalisation mediates a large, seed-dependent part of the measured necessity.

I reran the same mean-ablation with layer-2 LN1 statistics frozen at their clean values for every token. This makes the 2-D edit propagate linearly into Q, K and V.

| seed | drop_frac, live LN | drop_frac, frozen LN | KL to own-Bayes, live / frozen |
|---|---|---|---|
| s12 | 0.93 | **0.63** | 0.13 / 0.22 |
| s10 | 0.64 | **0.71** | — |

About a third of seed 12's "0.91" comes from LN rescaling the *whole* token after the tag component changes norm. LN is part of the model, so this is not illegitimate. But it means:
- the necessity is not "of the information in that subspace" in the linear sense the RCI §2 table implies;
- the 0.91 vs 0.65 contrast between seeds, which drives the reconciliation story, is partly an LN-treatment artefact. With frozen LN it is 0.63 vs 0.71, and the ordering flips.

The same caveat applies to the 2-D *interchanges* (tag@own6 / tag@all36): they are not 2-D interventions on K and V.

### F5 (major). "Makes seed 12 behave like the own-question Bayes reasoner" is not supported at the decision level.

The claim rests on KL(own‖model) falling from 0.46 to 0.13. Decision-level checks do not bear it out:
- **Untied argmax agreement** with own-Bayes (74.5% of readouts are untied):

  | seed | clean | after ablation |
  |---|---|---|
  | s12 | 0.720 | 0.741 |
  | s10 | 0.759 | 0.698 (decreases) |

  The logged `agree_own_bayes` (0.651 → 0.637) shows the same and was not discussed.
- **Entropy:** model entropy rises from 0.54 to 0.76 (s12) and to 0.68 (s10), against own-Bayes 0.82. Much of the KL drop is flattening.

The ablated model is at own-Bayes *accuracy* (0.587 vs 0.5745). It reaches that accuracy with different decisions in about 26% of untied cases. The safe statement is: "the relational gain is removed, and the output distribution moves toward the own-question posterior."

### F6 (major). Calling mean-ablation "natural necessity" is not justified by the package's own table.

RCI §2 lists the support of mean-ablation as "off-manifold activations" and then calls what it identifies "natural necessity". These are inconsistent.

- **Off-manifold, measured:** the pooled mean (over episodes, *and* over 36 positions and 6 questions) differs from the per-position mean.
  - The between-position share of tag-projection variance is 18% in s10 and 4% in s12.
  - In s10, a per-position mean gives drop_frac 0.53 against 0.64 pooled.
- The atlas "conditional mean" conditions on source position and value, not on question (Eq).

The label should be "ablation necessity (mean-replacement)". The pooled vs per-position (or per-question) mean should be reported, as should the LN treatment (F4).

### F7 (major). The "content × tag" account is consistent with the data, but its "direct test" was runnable and does not show a multiplicative interaction. One architectural test partly supports a value path.

**(a) Content-only interchange at other30.** This is the `I − Pt` component, with base tags kept. Seed 12:

| condition | IIA | KL |
|---|---|---|
| tag@other30 | 0.110 | 0.056 |
| content@other30 | 0.123 | 0.075 |
| full@other30 | 0.193 | 0.201 |

- In IIA the two components are *sub-additive* (0.233 > 0.193).
- In median KL they are super-additive (0.131 < 0.201).
- Neither result is a clean multiplicative signature.
- Content carries as much as the tag on its own. That is unsurprising: cond pairs resample the other questions' *reports*, so their value embeddings differ. Part of the "insufficiency" of a tag-only patch is that the counterfactual differs in the content itself, not only in a tag × content interaction.

**(b) Pattern freeze.** I mean-ablated the tag with all layer-2 attention patterns frozen at their clean values:

| seed | drop_frac, frozen patterns | drop_frac, live patterns |
|---|---|---|
| s10 | **0.02** | 0.64 |
| s12 | **0.40** | 0.93 |

- In seed 10 the tag acts entirely through routing (Q·K). That supports the A′ graph.
- In seed 12, 0.40 survives through values or LN. This is the best evidence in favour of a "tag → head values" edge in class C, and it should be reported.
- Pattern(tag) × value(content) is simply the generic QK × OV structure of attention. So "bilinear content × tag" describes attention; it is not an identified variable.

**Conclusion.** RCI §5 erratum 2 should not describe the joint interchange as "the remaining open part" without running (a) and (b), which take under a minute each.

### F8 (minor). The necessity and patchability numbers are on different scales.

The RCI §2 table pairs two incommensurable quantities:
- necessity: accuracy drop fraction against sampled Y, on the atlas stream 7.5M with N = 2000;
- patchability: IIA, which is argmax agreement with the counterfactual on the roughly 40% of informative pairs, on stream 7.7M.

On a common KL scale the s12 tag@all36 recovery is 0.66, not 0.47. The qualitative gap survives, but the numbers should not be laid side by side as if commensurable.

### F9 (minor). The additivity statements about IIA and median KL say nothing about the mechanism.

- Neither quantity is additive. A median of ratios, or an argmax rate, can look super- or sub-additive under a purely linear logit model.
- The seed-10 control shows the same "additivity": head0 0.34 + head2 0.29 against heads02 0.67.
- Super-additivity under ablation (0.50 vs 0.65) is a backup or redundancy signature combined with argmax thresholding, not evidence of "parallel" channels.

### F10. Head mean-ablation is implemented as described.

- Episode-mean per (head, position), replacing the head_out of the chosen heads at all six readouts at once, after W_O.
- Each readout's answer depends only on its own last-layer head outputs, so ablating the readouts jointly is equivalent to ablating them separately.
- Stream-to-stream variation is about 0.03 (atlasstream gives 0.21 / 0.27 / 0.66 / 0.49, against 0.24 / 0.26 / 0.65 / 0.50).

### F11 (major). The PP2 probe "internal signature" is largely a property of the linear probe on agreement-count features. It is not a validated colluder code off support.

I fitted the same logistic-probe pipeline (T4 stream 6.4M; p_c = 0 stream 6.2M plus 2) on raw input statistics instead of model states:

| probe input | train acc | picks top-agreement group | commit |
|---|---|---|---|
| per-group agreement counts | 0.96 | **1.00** | 0.90 |
| per-question agreement indicators | 0.96 | **1.00** | 0.92 |
| "group agrees in all 6" indicator (the T4-sufficient feature) | 0.96 | **0.39** | 0.41 |

- So "picks top = 1.0" (DeepSets) is what *any* representation that linearly carries graded counts produces, with no configuration constraint involved.
- About 0.40 (TF s0, s10) is what the saturated all-agree feature produces.
- The index therefore measures *which input statistic is linearly decodable*. It does not show that "the most-agreeing group must be the colluder" is carried off support.

Further problems:
- RCI §3 (class A) states the 100% result with no caveat.
- The Spearman correlations are not in `logs/pp2_mechanism.json`; the log holds commit vs local, ρ = 0.25. I recomputed them: all 10 models give −0.89, driven by the tied DeepSets cluster; Transformers only give −0.71 with p = 0.07 and n = 7.

### F12 (minor). The uniform-shift test is clean for the scored question. It needs a noise floor, and "factor through relative counts" is too strong.

**What is clean.**
- The edit touches only question k ≠ q = 0. q's tokens are unchanged, so the edited question's own content does not enter the score except through k → q.
- Bayes invariance is exact.
- A second realisation of the uniform shift (`a := b` instead of `b := a`) gives identical model KLs: s0 0.00052 / 0.00052; s12 0.00234 / 0.00231.

**What is missing.**
- **Null edits** with Bayes KL = 0 and an unchanged agreement pattern give model KLs of 1–5·10⁻⁴: s0 3.1e-4 (relabel) and 2.4e-4 (resample); s11 1.5e-4 and 1.1e-4; s12 4.6e-4 and 3.2e-4. So T1 uniform-shift KLs are only about 1.7–3.6× the content-change floor, and s12 is about 5–7×. The ratio should be reported against the floor.
- The uniform edit changes 3 tokens and the single edit changes 1, so they differ in edit size.
- Only the 0 → 1 shift in one question was tested. "Invariant to this uniform shift" is supported. "Factor through relative counts" is not.
- The T4-on-T1 numbers are off-support behaviour; the document says so.

### F13 (minor). Class Shapley.

- The gauge half of R1 is vacuous at this site. The patched object is head_out after W_O, so `W_V G`, `G⁻¹W_O` leave every value of v unchanged, and head-level Shapley is gauge-invariant too (the table shows exactly that).
- Split invariance holds because exact splits produce exactly equal patterns, which then merge. Approximate duplicates arising from training are not covered (L2 partly says this).
- The `class_shapley.py` docstring says "median KL-recovered". The code uses `kl_recovered_mean` (clipped mean). RCI says "mean". The docstring should be fixed.

### F14 (minor). Two interpretation notes on the s12 tables.

- "full@readout_q" (0.27) bundles the query (routing), the direct skip path and layer-2 MLP input of readout q. Calling it "the readout token's own residual accumulated in L0–L1" is accurate. But it does not separate "routing changed" from "direct content", and the reconstruction should not treat it as one channel.
- IIA is computed only on informative pairs (n_inf = 2422 of 6000 for s12). The KL figure is a median over pairs with base KL > 1e-3 and can be negative. Both are used correctly, but the column header "KL" should say "median KL fraction recovered".

## Summary of severity

- Critical: none.
- Major:
  - F2: missing readout channel;
  - F3: non-variance-matched control;
  - F4: LN mediation of necessity;
  - F5: own-Bayes overclaim;
  - F6: "natural necessity" label;
  - F7: content × tag test was runnable, and its result;
  - F11: the probe signature is generic to count features.
- Minor: F1, F8, F9, F12, F13, F14. F10 is a confirmation.
