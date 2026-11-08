# Review R_arch_counterexample: architecture and counterexample review (mandate §5, §8–§12)

**Reviewer stance.** Adversarial and independent. I regenerated `VERDICTS.md` with `scripts/aggregate.py`: frozen answer PARTIAL; secondary NO (scoped).

**Scope.**
- Read: PROTOCOL, ADDENDUM, DEVIATIONS, THEOREM, MATH_VALIDATION, src/, scripts/, all training JSON/logs, logs/posthoc, logs/deepsets and logs/verdicts.json.
- Ran six small CPU experiments (E1–E6). Their scripts are in `/tmp/rev/` (outside the package; not committed). Each used ≤ 3000 episodes, and seeds are given inline.
- **No package file other than this one was modified.**

**Severity scale.**
- **Fatal:** invalidates a headline claim as worded.
- **Major:** the claim must be re-scoped or a test added.
- **Minor:** wording or hygiene.

---

## A. Architecture review

### A1. "T4 is not learned by the M Transformer" is an optimisation and initialisation artifact. It is not a capacity limit. (Major. Fatal if any text reads it as inability.)

**Package evidence.**
- From scratch, all 3 seeds plateau exactly at own-question Bayes:
  - confirmation acc 0.574–0.575, own-question Bayes 0.573;
  - training loss ≈ 0.82, own-question NLL ≈ 0.816.
- The plateau persists at 24,000 steps (`posthoc_long`).
- Only one learning rate (1e-3) and one warmup/cosine family were ever tried.
- The T1 Transformer itself shows a plateau of about 2,000 steps before escaping (T1 s0 loss: 0.80, 0.73, 0.76, then 0.57 at step 2400). A longer T4 plateau is the default expectation, not evidence of inability.

**E5 (new).** I initialised from `T1_M_K3_s0` (a new zero WR head) and fine-tuned on T4 with the same optimiser and a 600-step warmup/cosine schedule.

| data seed | step 135 | step 200 | step 270 | Bayes | own |
|---|---|---|---|---|---|
| 77 | — | 0.661 | — | 0.745 | 0.578 |
| 78 | 0.538 | — | **0.741** | 0.745 | 0.578 |

- Step-200 accuracy for seed 77 is from a run that was cut off at 200 steps. That is 0.661, already at the P0 threshold.
- A 320-step schedule, which decays lr faster, stayed at 0.54. So the result is schedule-sensitive.
- Conclusion: the same architecture and width reach near-Bayes T4 in under 300 steps when started from a T1 solution. The from-scratch failure is a basin or plateau issue.

**Disposition check.**
- PROTOCOL §2 attributes the failure to "Transformer optimisation at that size", and THEOREM says "within the frozen budget". Both are correct and must stay that way.
- Any final text (MAIN_ANSWER.md, not yet present) must not say the Transformer "cannot" use non-monotone coherence.

**Fix.**
- Report E5-style curriculum runs (T1→T4) for 3 seeds, labelled post hoc.
- Run an lr sweep {3e-4, 3e-3} × warmup {200, 1000} from scratch.
- Report the auxR run (DEVIATIONS item 4). Note that `checkpoints/posthoc_auxR/` does not exist yet.
- The curriculum model is the only T4-solving Transformer available. Run the tag and route battery on it, labelled "non-spontaneous (curriculum)". This is the only way to get at the decisive T4 question (MATH_VALIDATION §7).

### A2. The baselines are capacity-comparable but not optimisation-fair, and `mlp_full` is too weak to count as evidence about T4. (Major.)

**Parameters:**
- Transformer 151k;
- `mlp_full` 166k;
- `mlp_hist` 143k;
- DeepSets 209k (+38%; not strictly matched).

**Compute is recorded (good) but very unequal.** For 6000 steps, wall clock is:
- Transformer 2000–2260 s;
- DeepSets about 185 s;
- MLPs about 35 s.

**Hyperparameters.** Every model used the same untuned lr. "Same lr" does not mean equally well-tuned.

**`mlp_full` on T4.** It reaches 0.553, *below* own-question Bayes (0.577) and equal to `mlp_hist`. So it did not even learn the within-question colluder structure. "Not learned by full-matrix MLP" therefore says nothing about T4 difficulty.

**Fix.**
- Run a per-architecture lr sweep (3 values) and pick on dev.
- Either match DeepSets parameters (e = 48, w = 224) or state the mismatch.

### A3. DeepSets succeeds because of its inductive bias, and that bias is the Theorem S factorisation itself. (Major for interpretation; the package mostly uses DeepSets correctly, as a method-calibration model.)

- φ sees X_q as a fixed-order concatenation of one-hots, so the pair equality A_{q,g} is a fixed per-question MLP function.
- The mean-pool over questions then *is* n_g / Q.
- That is exactly the sufficient statistic, by architecture.

**E6 (new).** DeepSets *without* the raw X_q skip into ρ still learned T4: 0.677 after 1500 steps (Bayes 0.744, own 0.582). So the pool, not the skip, carries the bias.

**What this means.** The contrast "DeepSets learns T4, Transformer does not" is a statement about inductive bias and optimisation, not about relational computation per se.

**Fix.** Say this explicitly. Do not use DeepSets' success on T4 as evidence that T4 is "easy" for a generic learner.

### A4. The "reusable across questions" claim is close to automatic under weight sharing, and the reuse criteria P6 and K6′ are degenerate. (Major.)

**Weight sharing.** Source tokens and readout tokens share all weights and differ only by the additive E_q.

**E3 (new).** Per-question 1-D source-token contrast (D2) directions at resid_pre2:

| model | mean pairwise |cos| |
|---|---|
| trained | 0.977 (s0), 0.974 (s2) |
| **random-init**, same architecture | **0.835** |

Most of the cross-question alignment is supplied by the architecture.

**The ratio criteria pass trivially when the base effect is absent.** "IIA_q345 ≥ 0.7 × IIA_cond" passes whenever IIA_cond is small:
- K6′ passes 3/3 for the **unlearned** T4 models (cond 0.12–0.18);
- K6′ passes for T1 s2, where K2′ fails (0.288 ≥ 0.7 × 0.311);
- P6 passes 2/3 for T4.

VERDICTS shows these with ✓ marks (also P4 3/3✓ and Distr(lit) 3/3✓ for T4). This is misleading even though they are labelled descriptive in the protocol.

**Fix.**
- In `aggregate.py`, gate K6′ on K2′ and P6 on P2. Mark non-P0 rows "descriptive, n/a".
- Compare reuse against the random-init alignment baseline.
- Call the result "cross-question transfer within shared weights". It is not cross-task reuse (mandate §8, §9.3). Cross-task reuse requires T2 (confidence head) or T3 results, which are still running.

### A5. The token design is a fair test of T1 but hard-codes the group structure. (Minor to major.)

**What is fair.** The design is reasonable and standard:
- bidirectional attention;
- additive value + source + question embeddings;
- one readout token per question and a free episode token;
- no positional confounds.

**The caveat.** The groups {0,1}, {2,3}, {4,5} are fixed across all episodes. Partner pairing can therefore be hard-wired through E_src, and the tag can be written in *group-specific* directions.

**E3.** Per-source D2 directions:
- seed 0: |cos| about 0.99 across all sources, i.e. one shared tag axis;
- seed 2: |cos| 0.44–0.69 between group 1 and the other groups.

So "one tag direction shared by all sources" is seed-specific.

**Fix.** State that the mechanism is identified only for a fixed, known grouping. Optionally, add a variant with grouping that varies per episode (group indicated by an input feature).

### A6. The size claims rest on n = 1 pilots. (Minor to major.)

- "S fails T1" is a single pilot seed (100): acc 0.654, own 0.655. PROTOCOL §3 excludes pilots from all criteria.
- The S seeds 0–2 and L s0 are queued but not run. L s1 was dropped.
- Mandate §5 asks for the 16-dimensional and 128-dimensional models.

**Fix.** Label the S result "pilot, 1 seed". Report L, or report that it was dropped.

### A7. Patching before LayerNorm. (Minor.)

- Sites are resid_pre (pre-LN), and the tag coordinate has a large magnitude (means −9.5 to +0.6 in s0).
- Replacing the 1-D component changes the LN scale of every other coordinate.
- So the patch is not purely "1-D".

**Fix.** Also report the patch with the LN scale frozen to the base value.

---

## B. Counterexample review

### B1. A functionally equivalent network without the localised source-token mediator exists in the package's own seed set. (Major: the headline must be re-scoped.)

**Seed 2** is at Bayes (0.740), but:

| patch at resid_pre2 | IIA |
|---|---|
| full residual of the 6 source tokens | 0.48 |
| 2-D readout-query D2 | 0.47–0.50 (E2 and posthoc_joint) |
| key tag + query jointly | 0.78–0.80 |

- E2: increasing the source-token D2 dimension to k = 5 only reaches 0.41.
- E3: group-specific 1-D tags reach 0.39.
- So the failure is not a projection convention. Reliability really is split between key and query in seed 2.

**Constructive counterexample (an argument, not built).**
1. Layer 0: each source token attends to its partner and writes the within-question agreement bit.
2. Layer 1: readout q attends uniformly to all 36 tokens and accumulates the per-group counts into its own residual.
3. Layer 2: routing score = query(counts) · key(group embedding), with λ·n_{g(s)}.

This reaches Bayes with the mediator at the *readout query* and no cross-question tag on the source tokens. Seed 2 is partway toward it.

**Registered outcome.** TagRoute_D2 holds in **1/3** seeds:
- s1 fails K4′ (0.104 against 0.10);
- s2 fails K2′, K3′, K5′ and R1′.

**Overclaim.** THEOREM Level 3 reports seed 0 numbers only, then states "in the T1 M models the answer is produced by" the tag-and-route mechanism. Seed 2 contradicts it: tag IIA 0.31, route-head log-attention R² 0.57 and 0.31.

**Fix.**
- Report all seeds in the Level 3 table.
- Re-word to "a key-side (2/3 seeds) or a mixed key/query (1/3) reliability signal at the input of the last layer, consumed by routing heads".
- Make the invariant claim the *routing* (Type B 3/3), not the tag location.

### B2. The tag is not "own-group agreement count, affine". It is a group-indexed contrast of all groups' counts, and its encoding varies across seeds. (Major for THEOREM Level 2/3 wording.)

**E3: R² of the shared D2 tag coordinate on the test stream.**

| features | s0 | s2 |
|---|---|---|
| own-group leave-one-out (LOO) count (one-hot) | 0.61 | 0.44 |
| + own-question agreement | 0.66 | 0.44 |
| + the other two groups' LOO counts | **0.87** | **0.68** |
| group-id × (LOO, own agreement, others) | 0.885 | 0.893 |
| group identity alone | 0.015 | 0.07 |

**Seed 1 versus seed 0** (posthoc):
- s1's tag fits own-count better than the Bayes logit (0.888 vs 0.653);
- s0's fits the logit better than own-count (0.788 vs 0.615).

So the tag carries other groups' counts (normalisation), which the Level 2 construction does not need, because the softmax normalises. And the coordinate system differs by seed. This matches MATH_VALIDATION §6 (the encoding is not identified), but the Level 3 table row "τ affine in the agreement count" overstates it.

**Fix.**
- Replace that row with the regression ladder above, for all seeds.
- State that the tag is "a sufficient statistic of the LOO count vector, approximately monotone in own count".

### B3. "Consistency score" versus "source-group statistic": the two are not separable in this generator, and the mandated group-specific aggregation baseline is missing. (Major for any novelty or "consistency theory" claim.)

**E4.** The exact posterior is invariant under random within-group swaps of reports (max |Δ| = 0.0 for T1 and T4). Therefore:
- the per-question **per-group vote histograms** are exactly Bayes-sufficient;
- n_g is just Σ_q 1[the group histogram has a 2].

The "relational" agreement count is literally a group-specific aggregate. Mandate §10 lists "group-specific aggregation" as a required baseline and warns that "a richer non-relational baseline may preserve the necessary source-group information". The K7 and P7 controls use only *global* histograms and own-value one-hots.

**Fix.**
- Add an `mlp_grouphist` baseline. Its Bayes ceiling equals full Bayes, which the package should state.
- Add per-group-histogram features to the K7 content regressions.
- Drop or qualify THEOREM's "Novelty" paragraph. It amounts to "the model computes the group-wise sufficient statistic and uses it as attention keys", i.e. Dawid–Skene-style weighting, as the package partly concedes.

### B4. Dependence on the projection convention: seed 0 is robust, seed 2 is not, and seed 2 is not rescued. (Minor; credit.)

**Sign and scale** are irrelevant, because patches use the projector UUᵀ.

**Seed 0 (E2):**
- PC1 explains 81% of the cond-difference variance (PC2 8%);
- IIA is flat in k: 0.848 (k = 1), 0.855, 0.864, 0.86 (k = 5);
- PC2 alone or PC3 alone gives 0.02;
- a random 1-D direction in the top-3 span (|cos| 0.51 with PC1) gives 0.19.

So the seed 0 mediator is genuinely 1-D and localised.

**Seed 2:**
- PC1 55%, PC2 32%;
- no k, and no per-group basis, exceeds 0.41.

**Remaining issue.** The probe-direction versus D2 dissociation (|cos| 0.22–0.51) means "decodable but inert" is a statement about *ridge-probe directions*, not about the variable. P1 passing and P2 failing at the readout is partly a probe-convention fact.

**Fix.** Say "the ridge-probe direction is inert". Do not say "reliability at the readout is inert": in s2 the readout D2 subspace has IIA 0.50.

### B5. Held-out distributions: the computation is "agreement ⇒ reliable", exactly T1-Bayes, including where that is wrong. (Major for interpretation; it also supports the count mechanism.)

**E1 (new): T1 models on shifted distributions** (3000 episodes per cell; values are s0 / s1 / s2).

| shift | model acc | true Bayes | T1-misspecified Bayes | own |
|---|---|---|---|---|
| Q = 4 (E_q sliced) | 0.717 / 0.736 / 0.718 | 0.721 / 0.736 / 0.723 | — | 0.65 |
| Q = 3 | 0.710 / 0.703 / 0.699 | 0.708 / 0.710 / 0.703 | — | 0.65 |
| T4 data (colluders) | **0.363 / 0.363 / 0.355** | 0.744 | **0.362 / 0.366 / 0.355** | 0.58 |
| biased noise (noise sources pick candidate 0 w.p. 0.6) | 0.687 / 0.670 / 0.691 | 0.783 | 0.678 / 0.671 / 0.684 | — |
| two reliable groups (off-prior) | 0.905 / 0.903 / 0.902 | — | 0.907 / 0.907 / 0.907 | — |

**What this shows.**
- Generalisation to fewer questions supports a count- or sum-based mechanism rather than a lookup table. That is good for the package.
- Under colluders and correlated noise, the models fall *below own-question accuracy*, exactly as a misspecified T1-Bayes does.
- So T1 evidence cannot distinguish "reliability inference" from a monotone "coherent group = trustworthy" rule. That distinction is what MATH_VALIDATION §7 says only T4 can test, and no from-scratch Transformer solved T4 (see A1).

**Fix.**
- Add this table.
- Predict and test that the tag ranks the colluding group as *most* reliable on T4 data. This is a cheap falsifiable check of the tag's semantics.

### B6. Bypass and redundancy are partly addressed. (Minor.)

- Freezing the route heads collapses the tag-patch effect (0.83 → 0.07). Good.
- But patching all of q's tokens (0.96) against the key tag alone (0.84) leaves about 12 IIA points on other paths in s0, and far more in s2.

**Fix.** Report the residual path explicitly (readout query and value path), for each seed.

### B7. Hygiene. (Minor.)

- THEOREM cites `MAIN_ANSWER.md` and `REVIEWS/`. Neither exists; the directory is `reviews/`.
- DEVIATIONS item 4 cites `checkpoints/posthoc_auxR`, which is not present yet.
- The Level 3 table mixes the confirmation stream and the posthoc stream (555000) without saying so for each row.

---

## Summary for disposition

| claim | status after review |
|---|---|
| M/T1 reaches Bayes | **Holds** (3/3), including Q = 3/4 generalisation. |
| Probe-direction reliability at the readout is decodable but inert | **Holds** for the probe direction. The readout is *not* inert in s2. |
| 1-D output-free source-token tag drives last-layer routing | **Holds cleanly in s0**, approximately in s1 (K4′ marginal), **fails in s2** (split key/query). The registered result is 1/3. The invariant across seeds is routing (Type B 3/3), not tag location. |
| Tag = own-group agreement count | **Overstated.** It is a group-indexed function of all LOO counts, with a seed-dependent encoding. It is indistinguishable from a group-specific aggregate (no such baseline exists). |
| Reuse across questions | **Weak.** Largely supplied by weight sharing (random-init alignment 0.835), and the criteria pass trivially. |
| Transformer M does not learn T4 | **Holds only as "not from scratch at lr 1e-3".** From a T1 initialisation it reaches 0.741 (Bayes 0.745) in 270 steps (E5, one replicate). Must not be read as inability. |
| DeepSets learns T4 | **Holds**, by an architecture that hard-codes the per-question-then-pool factorisation (E6). |

**Highest-value next tests:**
1. T1→T4 curriculum Transformers with 3 seeds, then the full tag and route battery on them.
2. An `mlp_grouphist` baseline, plus per-group-histogram content controls.
3. Gate the reuse criteria in `aggregate.py`.
4. Report every seed in the THEOREM Level 3 table.
