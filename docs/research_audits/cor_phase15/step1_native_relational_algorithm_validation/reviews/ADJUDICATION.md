# Adjudication of the independent reviews (Investigation XII)

Both reviews were written by **separately initialized agents**, given the code, the raw results and the documents.
They had no shared working context with the author. **This is not external human review.**

The author agrees with both reviews on every substantive point. No criticism was rejected.

## Review 1: mathematics (`INDEPENDENT_MATH_REVIEW.md`)

| item | verdict | action |
|---|---|---|
| M4/M5 used the wrong pair distribution for A1 | **accepted** (high) | recomputed: δ = 0.040, E changes on 32%; the non-discrimination claim was withdrawn (CAUSAL §2, §5) |
| §4 quantifier ("at least one implementation") | accepted (high) | changed to "every implementation in the hypothesised class" |
| §1.5 restricted A2 statement | accepted (high) | replaced; labelled [stated, not proved] |
| "each condition necessary"; "A1/E not separable" | accepted | restricted to sole-mediator claims; the redundancy counterexample was added |
| XII-1 conditions and consequences | accepted | uniform prior and support conditions added; e_C stated in general |
| β < 1 posterior_brute gross error | accepted | reported; a warning was added to genlib.posterior_brute |
| α ≠ 1/K; hard-coded M3 entry | accepted | the condition was added; M3 is now measured |
| XII-5 overclaim | accepted | qualified; the random-subspace control was run (it also passes, so the criterion is non-diagnostic) |
| novelty | accepted | the results are labelled elementary or task-specific |

## Review 2: final results (`INDEPENDENT_FINAL_REVIEW.md`)

| # | criticism | severity | action |
|---|---|---|---|
| 1 | "every fresh parent" / "invariant across 3 + 4": incomplete parents are heterogeneous, and P156 forms the conjunction in MLP1 | high | **accepted.** NATIVE §7, FRESH Part II and MAIN corrected; routing and allocation are now reported as heterogeneous |
| 2 | P3(b) misread: this is asymmetric partial redundancy | high | **accepted.** Wording corrected |
| 3 | P6-as-coded passed by a null intervention; the VS arm was known to be non-diagnostic pre-confirmation | high | **accepted.** Both labelled non-evidential; 16/94 passes flagged |
| 4 | amendment integrity statement false; it reverses the Stage-0 rule | high | **accepted.** A correction box was added to the amendment and FRESH; a P150 + P154-only verdict is reported (unchanged) |
| 5 | path_inA non-discriminating (cf_e0 ≈ Bayes on xA); path_inA ≡ resamp_A | high | **accepted.** path_inP, e_h = 0 scrub and TRUE-vs-CTRL now lead |
| 6 | pref_e0 criterion biased by entropy asymmetry; mean-ablation pref_e0 unexplained | high | **accepted.** Disclosed; open |
| 7 | scrub, restore and P6 re-test the DAS objective; the patching illusion is not excluded; synthetic at the patched site | high | **accepted.** Caveat added (NATIVE §1, MAIN) |
| 8 | P8 is full-block retraining; "bypasses E" overstated; "entirely downstream" and "linearly available" unsupported | high | **accepted.** Rewritten as a capacity and information result |
| 9 | P7 "no readout lag" holds partly by construction; correct use lags 1–5 checkpoints | high | **accepted.** Claim withdrawn; the narrower statement is kept |
| 10–15 | reference dependence; q7-only queries; MLP1 "amplifies"; CI caveats; pooling; logging; e_h = all-question agreement | medium/low | accepted. Edits in NATIVE, FRESH and MAIN; `ref_dependence_check.py` added; missing COMPUTE_LOG entries added; reframing note added |

**Unresolved after review.**
- Residual eligibility under layer-1 mean ablation.
- The subspace-patching illusion.
- Parent and DAS-fit variance in the CIs.

These are listed in `OPEN_COR_FOUNDATIONAL_QUESTIONS.md`.
