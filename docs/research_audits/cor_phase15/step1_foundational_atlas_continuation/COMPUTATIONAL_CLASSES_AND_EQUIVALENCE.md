# Computational classes and equivalence (mandate §5, §6, §12, part)

Status: every analysis here is **post hoc and exploratory**. It is run on models whose classes were assigned in the atlas.

Scripts:
- `scripts/s12_operations.py` (seeds 12 and 10);
- `scripts/class_shapley.py`;
- `scripts/abstraction_shift.py`.

Logs:
- `logs/s12/ops_s12.json` and `ops_s12_tagfix.json`;
- `logs/s12/ops_s10.json` and `ops_s10_tagfix.json`;
- `logs/class_shapley_s10.json`;
- `logs/abstraction_shift.json`.

**Bug note.** The first run of `s12_operations.py` combined token projectors by an elementwise maximum. That clips negative projector entries and corrupts every "tag" condition. The full-residual and head conditions were unaffected. The tag conditions were rerun after the fix (`*_tagfix.json`). The fixed tag@all36 result for seed 12 (0.474) reproduces the atlas value (0.455). The atlas script was re-executed as a check (`/tmp`, not kept): 0.455 exactly.

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

**Reconstruction (seed 12).** The relational evidence reaches the decision at q through three channels:

1. **Own-question tag routing** (head 3, which puts 95% of its attention on q's sources). This is the A′ operation, at about half strength: tag@own6 alone gives 0.395, against 0.771 in seed 10.
2. **Cross-question rereading** (head 0, which puts 54% of its attention on other questions' sources). Its input is the *full* residual of other-question tokens:
   - full@other30 gives 0.193, against tag@other30 0.109 in the atlas;
   - adding it to the own-question tag raises IIA from 0.395 to 0.637.
3. **The readout token's own residual**, accumulated in layers 0–1. Adding it to all source tokens raises IIA from 0.679 to 0.884. In seed 10 the same addition gives 0.834 → 0.982.

**Additivity.**
- Heads 0 and 3 are approximately additive in IIA: 0.33 + 0.43 = 0.76, and the joint patch also gives 0.76.
- They are super-additive in KL: 0.28 + 0.42 = 0.70 against a joint 0.93.
- Under ablation they are super-additive: 0.24 + 0.26 = 0.50 against a joint 0.65.

**Class C is therefore not a different operation from A′. It is A′ plus a parallel rereading channel.** The rereading channel is what the atlas called the "value path".

**Reconciling mean-ablation sensitivity with interchange failure (the atlas puzzle).**
- **Mean-ablation.** Mean-ablating the 2-D tag at all 36 source tokens makes seed 12 behave like the own-question Bayes reasoner: KL to own-Bayes drops from 0.46 to 0.13. Seed 10 behaves the same way (0.47 → 0.11). So the tag subspace is *necessary* for most of the relational evidence, in both seeds. The OP-5 drop fractions are 0.91 for seed 12 and 0.65 for seed 10.
- **Interchange.** Interchanging the tag is *sufficient* only in seed 10.
- **Account.** Seed 12's channel 2 reads the tag *jointly with* the non-tag content of the other-question tokens.
  - Ablating the tag removes channel 2's input altogether.
  - Interchanging only the tag gives channel 2 counterfactual tags with base contents, which is an inconsistent combination.
  - Supporting measurements: tag@all36 gives 0.474; adding the full other-question residual gives 0.637; the full residual at other30 alone gives 0.193.
- **Status.** This account is consistent with every number above but is still a hypothesis. The direct test would be a joint interchange of tag and content in which only the content is resampled, and it was not run. The bilinear form ("content × tag") is a description, not a localised variable.

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

**Findings.**
- **T1 models factor through relative counts.** This is an abstraction property no input–output accuracy test checks. T1 seed 12 is the weakest (0.14). That is the same seed index as the T4 rereader, but the T4 and T1 replication seeds are separate training runs, so no link is claimed.
- **T4 models violate the invariance badly on T1 data.** This is expected: under T4 semantics "every group agrees in k" is not a relative shift, because a colluder's agreement is absolute. It is another view of the off-support divergence (PP2).

## 3. Implementation-class equivalence: what is and is not invariant (§6, §12)

| relation between classes | A (pool) vs A′ (tag-route) | A′ vs B (split) | A′ vs C (reread) |
|---|---|---|---|
| E1, same natural-distribution behaviour (KL < 0.002) | yes | yes | yes |
| E2, same intervention signs on S1 edits | yes (0.996+) | yes | yes |
| E3, same off-support extrapolation | **no** (DeepSets global, local index 0.05–0.16) | partly | **no** (s12 global, 0.31) |
| E4, common ≤ 2-D aligned relation variable at the last-layer input | not applicable (no attention) | yes (key/query split) | **no**: the tag is necessary but not sufficient (§1) |
| E5, common channel structure | — | one channel | A′ plus a rereading channel |
| uniform-shift invariance (T1 parents) | — | — | T1 seed 12 slightly weaker |

**Strongest equivalence that holds across every observed class.** Functional equivalence on the generator support, plus sign-equivalence of on-support interventions. Both are implied by KL-closeness (PP1, theorem C4). Nothing implementation-specific is shared by all classes. The nearest candidate is a 2-D source-token subspace whose mean-ablation removes most of the relational gain in every Transformer seed tested: 0.65, 0.54, 0.91 and 0.68 for seeds 10, 11, 12 and 13 (OP-5). For comparison, a random 2-D subspace removes ≤ 0.03. The subspace's sufficiency is class-dependent.
