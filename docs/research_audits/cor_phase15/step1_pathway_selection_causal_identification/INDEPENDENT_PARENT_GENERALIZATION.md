# Generalisation across independently pretrained parents (Experiment C)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Labels:** [frozen] = `scripts/score_parents18.py` output; [post-hoc] = `scripts/posthoc_parents18.py`, written after the frozen output was seen.

## C1–C2. The complete candidate denominator

**The screening rule** (frozen, design §10.5):
- seeds s = 20–23, then 24–25 if a class was missing;
- one data order, D = 811;
- classification at step 1211 by structural + causal agreement, with competence KL ≤ 0.045;
- measured on stream 95_500_000.

| parent | s = 20 | 21 | 22 | 23 | 24 | 25 | bistable? |
|---|---|---|---|---|---|---|---|
| uP160 (reference) | — | (E base, XIV) | — | (M base, XIV) | — | — | yes (reference) |
| uP161 | not screened; XIII-E record 0/32 early | | | | | | **no** |
| uP162 | INC (hybrid-like, KL 0.078) | INC | INC (KL 0.050) | INC | INC | INC (early-like, KL 0.164) | **no** (no competent seed) |
| **uP163** | **EARLY** | HYBRID | INC (KL 0.132) | **MIXTURE** | — | — | **yes** |
| uP164 | HYBRID | INC (KL 0.048) | EARLY | EARLY | HYBRID | EARLY | **no** (no mixture) |
| uP165 | MIXTURE | HYBRID | MIXTURE | HYBRID | MIXTURE | MIXTURE | **no** (no early) |

(INC = incompetent; source `logs/XVIII_parents.json`, `logs/measure_scr.jsonl`.)

**1 of 5 independently pretrained candidate parents is bistable** under the recipe (uP163). This agrees with the XIII-D/XIII-E record that the organisation is an interaction of parent × (L, W) and that some parents give only one organisation.

**Side finding** [frozen screening data]. Competent baselines that are **naturally hybrid**, with both routes partly necessary and no intervention, occur in three parents:
- uP163 s21: nec_E 0.58, nec_M 0.34;
- uP164 s20 and s24;
- uP165 s21 and s23.

Competent hybrid-necessity states therefore also occur without intervention in uP163–uP165. Whether they are the *same* state as gsg's uP160 hybrids is not shown (Reviewer C §5.5).

## C3. Paired interventions on uP163 (E: s20, M: s23; D = 812; stream 95_600_000)

**[frozen]: non-evaluable.** Under the frozen evaluability rule, which includes the behavioural gate, the uP163 baselines are not unanimous:
- E baseline: lapse 0.015 against the 0.017 that EARLY requires;
- M baseline: lapse 0.020 and ρ_L 0.676 against the ≤ 0.017 and ≥ 0.68 that MIXTURE requires.

This is the same failure as in the uP160 confirmation (FROZEN_PREDICTIONS_AND_RESULTS.md §2). Frozen P4 class: **non-evaluable**.

**[post-hoc]** With structural + causal classes (behavioural gate removed), both bases are evaluable. Every arm is competent.

| arm | E (φ_nec, band) | M (φ_nec, band) |
|---|---|---|
| block | F (E → MIXTURE) | F (M → EARLY) |
| vrg | 0.82, P (structural class hybrid) | 0.96, F |
| gsv | 0.64, P | 0.94, F |
| sha-hi-rolled | 0.93, F | 0.95, F |
| sha-lo-real | −0.01, N | −0.02, N |

**Frozen per-parent rule, applied to the post-hoc bands** (Reviewer C, C10):

| arm | E | M | uP163 result |
|---|---|---|---|
| block | F | F | **F** |
| sha-hi-rolled | F | F | **F** |
| sha-lo-real | N | N | **N** |
| vrg | P (φ 0.82; causal class hybrid, nec_E 0.150 at the 0.15 bound) | F | **MIXED** |
| gsv | P (φ 0.64; hybrid on both diagnostics) | F | **MIXED** |

**Caveats:**
- **Marginal competence on M:** uP163 M arms have KL 0.039–0.043, against the 0.045 ceiling.
- **Uninformative mediator on M:** the share denominator A_base − A_block is 0.0048, so no mediator statement about uP163 M is possible.
- **Shared seeds (Reviewer D, D3):** the uP163 M base (s23, D = 812) shares its new-layer initialisation and its data order with uP160 M|812. Only the pretrained parent differs, so the contrast is paired by design.

## C4. Scientific outcome

**[sens] In the one bistable independent parent:**
- the block switches both bases;
- sha-hi-rolled switches both;
- sha-lo-real switches neither.

**vrg and gsv are MIXED:** they switch M and are partial on E. On uP160 E, gsv was F (φ 0.81–0.85). On uP163 E it is a hybrid (φ 0.64). "The pattern matches uP160's" is therefore **withdrawn**.

**What generalises:**
- the block effect;
- "content destroyed with the share on the baseline trajectory switches";
- "the forward-share clamp with continued learning does not switch".

**Limits:**
- **One replication.** uP160 and uP163 are the only bistable parents of the six.
- **Scope.** Claims are restricted to bistable parents.
- **Untestable candidates.** **Four** of the five independent candidates cannot be tested: uP161, uP162, uP164 and uP165.
- **No counterexamples.** No bistable parent gave one.
