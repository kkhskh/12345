# Computational classes and equivalence (mandate §5, §6, §12, part)

Status: every analysis here is **post hoc and exploratory**. It is run on models whose classes were assigned in the atlas.

Revision: revised after review round R2 (`reviews/ADJUDICATION_R2.md`); corrected claims are marked [R]. Additional checks are in `scripts/s12_review_checks.py` and `logs/s12/review_checks_s1{0,2}.json`. The tag rows in `logs/s12/ops_s1{0,2}.json` come from the buggy run and are superseded by `*_tagfix.json`.

Scripts:
- `scripts/s12_operations.py` (seeds 12 and 10);
- `scripts/class_shapley.py`;
- `scripts/abstraction_shift.py`.

Logs:
- `logs/s12/ops_s12.json` and `ops_s12_tagfix.json`;
- `logs/s12/ops_s10.json` and `ops_s10_tagfix.json`;
- `logs/class_shapley_s10.json`;
- `logs/abstraction_shift.json`.

**Bug note.** The first run of `s12_operations.py` combined token projectors by an elementwise maximum. That clips negative projector entries and corrupts every "tag" condition. The full-residual and head conditions were unaffected. The tag conditions were rerun after the fix (`*_tagfix.json`). After the fix, tag@all36 for seed 12 is 0.474 on stream 7,700,000. The atlas value is 0.455 on stream 7,600,000. These are different streams, so this is consistency, not exact reproduction. Re-executing the atlas script itself gave 0.455 exactly (`/tmp`, not kept). [R] `ops_s12_atlasstream.json` came from the buggy run and is superseded (CA-F1).

## 1. Seed 12: operation-level reconstruction (§5)

**Setting.**
- Site: the input of the last layer, and the last layer's per-head outputs at the readout token q.
- Counterfactual: cond pairs. X_q is kept and the context is resampled on the generator.
- IIA is argmax agreement with the counterfactual on informative pairs; KL is the median fraction of KL recovered.
- Seed 10 (class A′) is the control.

| patched at the last-layer input / readout | seed 12 IIA (KL) | seed 10 IIA (KL) |
|---|---|---|
| tag (2-D) at the 6 own-question source tokens | 0.395 (0.51) | **0.771 (0.98)** |
| tag at all 36 source tokens | 0.474 (0.66) | 0.772 (0.98) |
| full residual at the 6 own-question tokens | 0.424 (0.56) | 0.835 (1.00) |
| full residual at the 30 other-question tokens | **0.193 (0.20)** | 0.022 (0.00) |
| full@other30 + tag@own6 | **0.637 (0.87)** | 0.777 (0.98) |
| full residual at all 36 source tokens | 0.679 (0.89) | 0.834 (1.00) |
| full residual at readout q only | 0.270 (0.10) | 0.150 (0.02) |
| full@readout_q + all 36 | 0.884 (1.00) | 0.982 (1.00) |
| all 43 positions (sanity) | 1.000 | 1.000 |

| last-layer head outputs at readout q | seed 12 IIA (KL) | seed 10 IIA (KL) |
|---|---|---|
| head 0 | 0.333 (0.28) | 0.342 (0.43) |
| head 1 | 0.091 (0.05) | 0.034 (0.00) |
| head 2 | 0.143 (0.07) | 0.294 (0.42) |
| head 3 | 0.433 (0.42) | 0.288 (0.42) |
| heads 0+3 | **0.763 (0.93)** | 0.675 (0.89) |
| heads 2+3 | 0.520 (0.71) | 0.602 (0.86) |
| all 4 | 0.877 (0.99) | 0.928 (1.00) |

**Attention mass from readout q** (own question / other questions):

| head | seed 12 | seed 10 |
|---|---|---|
| 0 | 0.10 / **0.54** | 0.98 / 0.02 |
| 1 | 0.11 / 0.76 | 0.17 / 0.52 |
| 2 | 0.93 / 0.05 | 0.98 / 0.01 |
| 3 | 0.95 / 0.02 | 0.98 / 0.01 |

**Mean-ablation of last-layer head outputs at the readouts**, as a fraction of the relational gain lost (clean minus own-question Bayes):

| heads ablated | seed 12 | seed 10 |
|---|---|---|
| 0 | 0.24 | 0.11 |
| 3 | 0.26 | 0.10 |
| 0+3 | **0.65** | 0.21 |
| 2+3 | 0.50 | 0.15 |

**Additional checks (after review; `logs/s12/review_checks_s*.json`).**

| condition | seed 12 | seed 10 |
|---|---|---|
| full residual at all 36 sources + readout q + episode token, IIA | 0.897 | 0.983 |
| full residual at all 36 sources + **all 6 readout tokens**, IIA | **0.989** | 0.996 |
| content only (complement of the tag) at other30, IIA | 0.123 | 0.025 |
| tag only at other30, IIA | 0.110 | 0.014 |
| tag mean-ablation, live patterns: drop fraction | 0.93 | 0.64 |
| tag mean-ablation with last-layer **patterns frozen** at clean values: drop fraction | **0.40** | **0.02** |

Head 0 of seed 12 also puts 0.29 of its readout attention on the *other readout tokens* (reviewer CA-F2; seed 10: < 0.001 for heads 0, 2, 3).

**Reconstruction (seed 12), revised [R].** The relational evidence reaches the decision at q through at least four channels:

1. **Own-question tag routing** (head 3; 95% of its attention on q's sources). This is the A′ operation at reduced strength: tag@own6 alone gives 0.395, against 0.771 in seed 10.
2. **Values read from other questions' source tokens** (head 0; 54% of its attention). Content alone at other30 gives 0.123 and the tag alone 0.110.
3. **Other questions' readout tokens**, which already aggregate evidence in layers 0–1. Adding them raises IIA from 0.897 to 0.989.
4. **The readout token's own residual** from layers 0–1. This bundles query, skip path and MLP input, so it is not a single operation (CA-F14).

The strongest evidence for a non-routing edge in class C is the pattern-freeze result:
- in seed 10, freezing the last-layer patterns removes almost all of the tag's effect (drop fraction 0.64 → 0.02). The tag acts through routing;
- in seed 12, 0.40 of the gain still disappears under a tag ablation with frozen routing. Part of the tag's effect goes through values (or LayerNorm).

**Status.** "Class C = A′ plus additional cross-question value channels" is a **hypothesis from one seed** [R: the pre-review text stated it as a conclusion].

**Reconciling mean-ablation sensitivity with interchange failure, revised [R].**

*What survives:*
- Seed 12's tag subspace is mean-replacement-necessary: the drop fraction is 0.91–0.93 with live LayerNorm and patterns.
- It is only partly interchange-sufficient (0.47).
- At least three other channels (2–4 above) carry counterfactual information outside the 2-D source-token tag, so a tag-only interchange leaves them at base values. That alone explains the insufficiency.
- The pattern-freeze result shows that, in seed 12, the tag also acts through values, which seed 10 does not.

*What is withdrawn:*
- **"The tag is necessary for all the relational evidence."**
  - With the last-layer LayerNorm statistics frozen, the drop fraction is 0.63 for seed 12 and 0.71 for seed 10 (reviewer CA-F4, reviewer-reported). So part of the live-LN necessity is renormalisation, and the seed-12 > seed-10 ordering flips.
  - A variance-matched control (the top-2 non-tag principal components) removes 0.71 in seed 10 (reviewer CA-F3). Necessity *by magnitude* therefore does not single out the tag.
  - What does single it out is the *direction* of the damage: tag ablation moves the output distribution toward the own-question posterior (KL to own-Bayes 0.46 → 0.13), whereas the control destroys the output.
- **"Makes the model behave like the own-question Bayes reasoner."** At the decision level argmax agreement with own-Bayes barely moves, and model entropy rises (CA-F5). The safe statement is: the relational gain is removed, and the output distribution moves toward the own-question posterior.
- **"Bilinear content × tag".** Content-only and tag-only interchanges at other30 are sub-additive in IIA (0.123 + 0.110 > 0.193), so there is no multiplicative signature. "Pattern(tag) × value(content)" is the generic QK × OV form of attention, not an identified variable (CA-F7).

*Terminology [R].* Mean-ablation is an off-manifold intervention. It is called **mean-replacement necessity** here, not "natural necessity" (CA-F6). The pooled mean (over episodes, positions and questions) and a per-position mean give different values in seed 10: 0.64 against 0.53 (reviewer-reported).

## 2. Abstraction test: do the models factor through *relative* counts? (§6, §7)

**Test.**
- Under T1 (p_c = 0), Bayes at q is invariant to adding one agreement to *every* group in another question k: the uniform shift. Exact: KL is 4·10⁻¹⁸.
- Bayes changes when one agreement is added to a single group: KL 0.017.
- 11,178 T1 episodes have a question with no within-group agreement at all.

| model | KL under uniform shift | KL under single shift | invariance ratio |
|---|---|---|---|
| Bayes T1 | 0 | 0.0171 | 0 |
| T1 Transformers (s0, 1, 2, 10, 11, 13) | 0.0004–0.0007 | 0.0167–0.0175 | **0.022–0.043** |
| T1 Transformer s12 | 0.0023 | 0.0171 | 0.137 |
| T4 Transformer s10, on T1 data | 0.152 | 0.011 | 13.7 |
| T4 Transformer s12, on T1 data | 0.063 | 0.023 | 2.7 |
| T4 DeepSets s0, on T1 data | 0.012 | 0.031 | 0.40 |

**Findings, revised [R].**
- **Bayes-level fact.** Under T1 the uniform shift leaves Bayes unchanged.
- **Model fact.** The T1 Transformers are approximately invariant to this one shift (ratio 0.02–0.04; seed 12 0.14). Their shift KL is about 2–4× a null-edit floor of 1–5·10⁻⁴ (reviewer CA-F12).
- **What this is not.** It is not evidence of an abstraction property beyond Bayes-closeness. T1 has full support, so KL-closeness on T1 already bounds the shifted-input KL by the density ratio times ε (THEOREMS K2, replacement; reviewers FM-F2 and NV-A3).
- [R] The pre-review "T1 models factor through relative counts" overstated a single-shift test and is withdrawn.
- **T4 models on T1 data** violate the invariance strongly. That is off-support behaviour: a colluder's agreement is absolute under T4 semantics. It is another view of PP2.

## 3. Implementation-class equivalence: what is and is not invariant (§6, §12)

| relation between classes | A (pool) vs A′ (tag-route) | A′ vs B (split) | A′ vs C (reread) |
|---|---|---|---|
| E1, same natural-distribution behaviour (KL < 0.002) | yes | yes | yes |
| E2, same intervention signs on S1 edits | yes (0.996+) | yes | yes |
| E3, same off-support extrapolation | **no** (DeepSets global, local index 0.05–0.16) | partly | **no** (s12 global, 0.31) |
| E4, common ≤ 2-D aligned relation variable at the last-layer input | not applicable (no attention) | yes (key/query split) | **no**: the tag is mean-replacement-necessary but not interchange-sufficient (§1) |
| E5, common channel structure | — | one channel | A′ plus at least three cross-question channels (hypothesis, n = 1) |
| tag acts only through routing (frozen-pattern necessity ≈ 0) | — | s10: yes (0.02) | **no** (0.40) |
| uniform-shift invariance (T1 parents) | — | — | T1 seed 12 slightly weaker (a Bayes-closeness measure, not an abstraction) |

**Strongest equivalence that holds across every observed class.** This is functional equivalence on the generator support, together with sign agreement on on-support interventions wherever |Δm| exceeds the pointwise margin errors. Both are properties of each model's closeness to Bayes; no theorem with the observed constants implies them (PP1 §2).

**No implementation-specific property is shared by all classes.** The nearest candidate is the 2-D source-token tag subspace. In every Transformer seed tested, ablating it moves outputs *toward the own-question posterior* (OP-5; §1). But its necessity *in magnitude* is matched by a variance-matched control in seed 10, and its sufficiency is class-dependent.
