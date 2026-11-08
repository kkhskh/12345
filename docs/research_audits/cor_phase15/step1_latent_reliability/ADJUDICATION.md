# Adjudication of the independent reviews (mandate §12)

**Reviews.**
- `reviews/R_math_causal.md`: 21 findings, mathematical and causal.
- `reviews/R_arch_counterexample.md`: A1–A7 (architecture), B1–B7 (counterexamples).

The reviewers' check scripts lived outside the package, in `/tmp/rev`, and were not preserved. Where a disposition relies on one of their numbers, it says "reviewer computation, not reproduced in-package" unless we re-ran it.

## Dispositions

| # | finding (short) | disposition | action |
|---|---|---|---|
| MC1 | Level-1 numbers and proofs correct | — | none |
| MC2 | Corollary T4 sentence false in general: with counts (6,2,2), G0 *must* be the colluder | **Accepted** | `THEOREM.md` restated: non-monotonicity holds *given the other counts*, e.g. (n_0, 6, 2) |
| MC3 | Theorem S has unstated scope assumptions (fixed groups, uniform errors, uniform R prior) | **Accepted** | scope stated in `THEOREM.md` |
| MC4 | Level-2 "routing = Bayes group posterior" needs episode-level counts. With leave-one-out counts, agreement with Bayes is 93.6% vs 97.2% (reviewer computation) | **Accepted** | Level 2 restated |
| MC5 | A logit-linear vote is not a probability mixture. Argmax-level only: 5.4% ties, calibration unexplained | **Accepted** | Level 2 restated as argmax-level and approximate |
| MC6 | βa is unit-free and testable. Rough estimates 0.35–1.29, all below λ = 1.357 | **Accepted** | reported as a measured mismatch: the learned routing is *less* sharp than Bayes. Accuracy is insensitive to it |
| MC7 | `target` pairs are a value-content control, not the count-preserving X_{−q} test. Reviewer's count-preserving resample: model argmax change 3.2–4.0% vs 26–27% for cond | **Accepted** | relabelled. The reviewer's test is reported as support (not reproduced in-package) |
| MC8, A7 | Level 2 needs unstated attention assumptions (pre-LN, other heads' contributions) | **Accepted** | listed |
| MC9, B1 | Level 3 generalised to "the T1 M models", against the package's own verdict (TagRoute_D2 1/3 seeds) | **Accepted. This is the most important correction** | `MAIN_ANSWER.md` separates (a) last-layer reliability routing: frozen Type B, 3/3 seeds, from (b) source-token tag localisation: registered verdict 1/3; post hoc it is seed-dependent, with seed 2 splitting the information between keys and the readout query |
| MC10 | "Decodable but inert at the readout" is trivial at the final site in seeds 0–1, and false at the readout *query* in seed 2 | **Accepted** | restated: the frozen site rule picked the final site (readout-equivalent) in 2/3 seeds. At the input to the last layer, the readout query carries a causal share in seed 2 only (post hoc joint patch) |
| MC11, A4 | Cross-question reuse is near-automatic under weight sharing. P6 and K6′ are degenerate (they pass when the base effect is absent) | **Accepted** | gated versions added (`P6_gated`, `K6'_gated`); the literal ones are still shown. "Reusable" is restated as "the same directions serve every question, which weight sharing makes likely" |
| MC12 | Content-control bug in `posthoc.py`: token-invariant features | **Accepted** | fixed (DEVIATIONS 5) and post hoc rerun. Linear-in-raw-reports R² stays ≤ 0.04 |
| MC13 | Decoder identifies reliability, not the target | — | supportive |
| MC14 | IIA null too weak. A mismatched-source patch reaches IIA 0.45 (s0). Chance-corrected IIA ≈ 0.72 | **Accepted** | kl_recovered (0.99 matched vs 0.16 mismatched, per reviewer) is reported as the more discriminating metric. The thresholds stay frozen; their weakness is stated |
| MC15–16 | Informative-pair selection sound; cond is on-manifold | — | none |
| MC17 | Target-pair specificity threshold (0.10) sits near the random floor (0.03–0.075) | **Accepted** | stated |
| MC18 | drop_frac > 1 means wrong tags were injected, not that information was removed | **Accepted** | necessity is reinterpreted as "the tag is read". Mean-ablation is not run (time) |
| MC19 | The D2 transfer test is an in-distribution replication of its own fitting procedure | **Accepted in part** | the output-free definition stands. The content check against Bayes (K7) and the perm and held-out-α transfers are the independent parts |
| MC20, B6 | The bypass test separates QK from OV only; it does not exclude earlier redundant paths | **Accepted** | stated |
| MC21 | Seed 2's failure is not caused by k_tag = 1 | — | consistent with our post hoc joint patch |
| A1 | "T4 not learned" is an optimisation/initialisation artifact. A T1→T4 fine-tune reaches 0.741 in 270 steps (reviewer, 1 replicate, schedule-sensitive) | **Accepted. Second most important correction** | **post hoc curriculum experiment run in-package on 3 seeds** (`checkpoints/posthoc_curriculum`): see `MAIN_ANSWER.md`. No text claims Transformer *inability* |
| A2 | Baselines are not optimisation-fair. `mlp_full` fails even the within-question part of T4, so it says nothing about T4 | **Accepted** | stated. `mlp_full` T4 is not used as evidence about task difficulty |
| A3 | DeepSets succeeds through an inductive bias that *is* the Theorem-S factorisation | **Accepted** | DeepSets is used only as method calibration and a learnability witness |
| A5 | Fixed groups hard-code the pairing | **Accepted** | listed as a scope limit; a held-out group-assignment test is future work |
| A6, MC | Size claims rest on n = 1 S pilot; L was not run | **Accepted** | S grid (3 seeds) reported if completed. No size claim beyond it |
| B2 | The tag is a group-indexed contrast of *all* groups' counts (R² 0.87 / 0.68), not the own count alone, and its encoding varies by seed | **Accepted** | `THEOREM.md` Level 3 restated |
| B3 | "Consistency score" vs "group statistic" cannot be separated: per-group histograms are exactly sufficient. The §10 group-specific aggregation baseline was missing | **Accepted** | `mlp_grouphist` added (post hoc). The novelty claim is withdrawn: the computed variable is a group-indexed reliability statistic, which in this generator equals a within-group consistency statistic |
| B4 | Projection convention: seed 0 robust, seed 2 not rescued | — | consistent |
| B5 | Held-out distributions: T1 models implement "agreement ⇒ reliable" (T1-Bayes) and fall to 0.36 on T4 data | **Accepted** | stated. This is exactly why T4 matters; see the curriculum results |
| B7 | Hygiene (missing files cited) | **Accepted** | fixed |
