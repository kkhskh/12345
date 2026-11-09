# Adjudication of review round R2 (four independent subagent reviewers)

**Reviews.**
- `R2_formal_math_counterexamples.md` (FM)
- `R2_statistical_inference.md` (ST)
- `R2_causal_architecture.md` (CA)
- `R2_novelty_and_positive_defense.md` (NV)

**Re-runs.** Where a reviewer's number was re-run by the authors, the script and log are given.
- "Reviewer-reported" means the reviewer's own check, kept in `/tmp` and not part of the package. It was not independently re-run.
- Reviewer-reported numbers are cited as such in the documents.

**Original text.** Every finding below led to an edit. Pre-review versions of every document are in git history (commit `9a9de43` and earlier).

## Critical

| id | finding | verified by authors? | disposition |
|---|---|---|---|
| FM-F1 / ST-F1 | "52% of argmax flips at ρ → 1" is a NaN artefact. 78% of posteriors are undefined at ρ = 1 on ρ = 0.6 data, and argmax(NaN) returns 0 | **yes**: `scripts/review_checks_landscape.py`. Defined at ρ = 1: 21.9%; flip rate on defined cells 0.048; interior steps 0.024–0.034 | **Accepted. Retracted** in LANDSCAPE §1.5 and THEOREMS §4: the "singularity / maximal sensitivity" claim is withdrawn. What remains: at ρ = 1 most mixed episodes leave the support, which is a support change, nothing more |
| FM-F2 / NV-A3 | the K2 counterexample is impossible. T1 has full support, so a model KL-exact on T1 equals Bayes everywhere and inherits its invariance. K2 is tautological | yes (argument) | **Accepted.** K2 is replaced by NV-B4's statement: on a full-support generator, KL-closeness bounds the shift response. The uniform-shift test is re-read as a Bayes-closeness check, not an abstraction property (COMPUTATIONAL_CLASSES §2) |
| ST-F2 / FM-F3 / NV-A2 | C4 is valid but **vacuous** on PP1. At ε ≈ 10⁻³ its right-hand side is ≥ 1 for every τ, so "C4 explains the 0.996+ agreement" is false | yes (arithmetic: Wε/τ² ≥ 1 whenever b_τ is small enough to matter) | **Accepted.** PP1 §4–§5 and COR_TEN §1 are rewritten. The explanation is the *empirical* pointwise fact that margin errors at both endpoints are below \|Δm\| in 98.6–99.7% of edits (C1 coverage). C1 itself is an identity, so "never violated" is a code check |
| NV-A1 | OD1's "only if" is false. Unequal accuracies inside a weight band (ratio < 3/2 for 6 sources) give an exact bridge. The correlated-pair case in the package's own log is also 1.000 (FM-F6) | partly: the log value 1.000 is the package's own; the band enumeration is reviewer-reported | **Accepted.** OD1 is restated: "if" is proved; the exact condition is the weighted-voting condition (Nitzan–Paroush; Ben-Yashar–Paroush). "Iff trivial semantics" is withdrawn. COR_TEN F3 status corrected |

## Major

| id | finding | verified? | disposition |
|---|---|---|---|
| ST-F3 / FM-F7 | biased errors *do* move the one-vote boundary; it splits by conflict direction | **yes**: predicted 0.908 / 0.9988; observed jump between 0.905 and 0.91 for v = c + 1, and between 0.995 and 0.999 for v = c + 2 | **Accepted.** V1 is stated for symmetric errors, with the biased-error formula and its check |
| ST-F4 / FM-F4 | W ≈ 1.2 is a coarse lower bound. The pointwise ratio reaches about 1.96 on S1 | reviewer-reported | **Accepted**: "W = O(1) (≥ 1.96 pointwise)". Since C4 is vacuous anyway, nothing else depends on it |
| ST-F5 / FM-F11 | the targeted CX tested the wrong region (low P(x) rather than high dE/dN). A high-ratio family exists ("make the noise group perfect", ratio ≥ 7.7) | **yes**: `scripts/pp1_highratio.py`. Natural P(A_q) = 0.130 | **Accepted. New result.** On the high-ratio family the learned models' unfiltered sign agreement is 0.963–0.980 (clustered CIs ±0.003). With \|Δm\| > 0.5 it is 0.985–0.999. Every failure has \|Δm\| < 1. FM's multi-cell construction (46% of endpoints corrupted, agreement 0.875; reviewer-reported) closes the "open construction" for C7. CX4 (exact except on a 35%-mass region) is *not* a low-KL counterexample (natural KL 0.82), and is reported as such |
| ST-F6 | the \|Δm\| > 0.5 filter removes about half of S1, exactly where failures concentrate | partly: the high-ratio family is now reported by \|Δm\| bin, including (0, 0.5] | **Accepted.** Headline rates are now given with and without the filter where available. For the frozen PP1 population only filtered rates exist; this is stated |
| ST-F7 | pseudo-replication (27,000 inputs × 5 queries); S1 \ S2 comes from 119 base episodes; TF s12 = 1.000 there, misreported as 0.968–0.998 | yes (log) | **Accepted.** Text corrected. The transformer-vs-DeepSets S1 \ S2 "concentration" is withdrawn as a post hoc stratum without multiplicity control |
| ST-F9 | phase minima: at α = 0.8 the accuracy minimum is at 0.55, not at ρ* | yes (log) | **Accepted.** The minima are stated as observed. The symmetry point (where the true-reliable mass is minimal, at 0.5) is distinguished from the accuracy minimum |
| ST-F12 / NV-A7 | P-A is weakly informative: a-priori P(pass) ≈ 0.53 for A1 and ≈ 1 for A2. A2 is a near-proxy of the target. "T4-derived inputs" are off support | reviewer-reported bootstrap | **Accepted.** A post-freeze annotation is appended to PROSPECTIVE_PREDICTIONS. The frozen text is unchanged. The outcome is reported with these caveats |
| ST-F15 / NV-A8 | decision ledger: one model per class; no reference rates; the misleading-relation failure needs a T4-Bayes reference | **yes**: the T4 plug-in disagrees with the truth on 0.587 of perfect-group p_c = 0 questions (models 0.63–0.65) | **Accepted.** COR_TEN §1 is rewritten: the models behave as T4-Bayes (prior mismatch), not as a network-specific failure. Class statements are marked n = 1 per class |
| FM-F5 | C6's "for every generator" is false as quantified. CX1 preserves the full ranking and gets order-swapping edits right | reviewer-reported stratum numbers | **Accepted.** C6 is restated as an existence result for the PP1 generator and edit family |
| FM-F8 | L1's impossibility is a tautology. A nontrivial invariant (mass over value-equivalence classes) may exist | argument | **Accepted.** L1 is reworded. The construction is moved to OPEN_FOUNDATIONAL_MAP OF-5 as a concrete lead |
| FM-F9 | the ρ = 1 statement mixed probability and log-odds scales | — | **Accepted** (moot after the FM-F1 retraction) |
| CA-F2 | seed 12 has a fourth channel: the other readout tokens (head 0 puts 0.29 of its attention there) | **yes**: `scripts/s12_review_checks.py`. full@36 + readout q + episode: 0.897; + all 6 readouts: 0.989 (seed 10: 0.983 → 0.996) | **Accepted.** The reconstruction is amended. "A′ plus a rereading channel" is now marked as a hypothesis (n = 1 seed) |
| CA-F3 | the random-2-D control is not variance-matched. A variance-matched control removes 0.71 in seed 10 | reviewer-reported | **Accepted.** Necessity *specificity* is now claimed only in direction (the output moves toward the own-question posterior), not in magnitude |
| CA-F4 | LayerNorm mediates part of the necessity. With frozen LN: s12 0.63, s10 0.71, so the ordering flips | reviewer-reported | **Accepted.** The seed-12-vs-seed-10 necessity contrast is withdrawn as evidence for the reconciliation |
| CA-F5 | "behaves like own-question Bayes" is not supported at the decision level (argmax agreement barely moves; the output flattens) | partly: logged agree_own 0.651 → 0.637 | **Accepted.** Reworded to: "the relational gain is removed and the output distribution moves toward the own-question posterior" |
| CA-F6 | "natural necessity" contradicts the table's own "off-manifold" | — | **Accepted.** Renamed to "mean-replacement necessity" throughout the continuation. The atlas wording is not edited |
| CA-F7 | content × tag: content-only interchange is 0.123 (tag 0.110; full 0.193), sub-additive in IIA. Pattern-frozen mean-ablation: s10 0.02, s12 0.40 | **yes** (exact match) | **Accepted.** "Bilinear content × tag" is withdrawn as a mechanism claim: it is the generic QK × OV structure. The pattern-freeze result is added as the best evidence for a value-path edge in seed 12 (0.40 of the gain survives frozen routing) |
| CA-F11 | the PP2 "picks the top group" index is reproduced by a probe on raw graded counts (1.00). The all-agree indicator gives 0.39 | **yes**: `scripts/pp2_probe_control.py` (counts 1.00; indicators 1.00; all-agree 0.39) | **Accepted. Re-interpreted.** The index measures *which agreement statistic the representation carries linearly*: graded counts (DeepSets → global) versus a saturated full-agreement indicator (local Transformers). That is a representational statement, not a colluder-code statement |
| NV-A4, A5, A6, A9, A10, A11 | missing prior art: Owen 1977 / Knudsen–Østerdal 2012 / groupShapley; Nitzan–Paroush 1982 / Ben-Yashar–Paroush 2001; Xu et al. 2021 (architecture-determined extrapolation); Gunn et al. 2016 (paradox of unanimity); Fu et al. 2020 / Zhang et al. 2014 (latent-class identifiability); Tian–Pearl 2000 | references flagged [m] by the reviewer are cited as "not verified" | **Accepted.** Added to BASELINES_AND_NOVELTY |
| NV-B2 | the interior support rule is very probably a theorem (T4, edits keeping the group below full agreement) | proof sketch only | **Accepted as a proposition with a sketch**, marked "not independently verified". The empirical 1.000 is kept as the check |

## Minor (all accepted, applied in the text)

**FM:**
- F10: C5 gloss.
- F12: D absolute continuity fails on 3,264 S1 ∩ S2 edits (reviewer-reported).
- F13: U-scope exact at fixed contexts. **Verified**: 45/49 contexts non-quasi-concave for the union; 0/98 for single-role generators.
- F14: I2 is [P].
- F15–F18: V1 test scope; S-without-(O) not new; I1′ sketch incomplete; notation.

**ST:**
- F8: S2 and S3 are artefacts of the design.
- F10: the step label.
- F11: U Monte Carlo.
- F13: the ρ = 0.999 plug-in is part of the target definition.
- F14: Spearman p-values (within Transformers p ≈ 0.07–0.09).
- F16: `set()` ordering, fixed with `sorted`. `phase.json` was **regenerated** with the fixed, NaN-guarded script. Monte Carlo numbers moved slightly (for example, the α = 0.8 accuracy minimum is now at ρ = 0.6), and the documents quote the regenerated values.
- F17: unknown-groups wording.

**CA:**
- F1: buggy rows still on disk. **Labelled**: `ops_s1{0,2}.json` tag rows are superseded by `*_tagfix.json`. The atlas-stream comparison used the buggy run, so the agreement 0.474 vs 0.455 is a cross-stream comparison; the atlas script itself was re-executed and gave 0.455.
- F8, F9: scales and additivity.
- F12: uniform-shift noise floor. Reviewer-reported null-edit KL is 1–5·10⁻⁴, so the T1 invariance is about 2–4× the floor; reported.
- F13: R1's gauge part is vacuous at this site, because the hook sits after W_O. The docstring said median when the code uses the mean.
- F14: wording.

**NV:**
- A12: calibration ≠ ranking, prior art.
- A13: smaller scope issues.
- B1: the OD1 replacement.
- B3: I2.
- B4: the K2 replacement.
- B5–B7: strength of claims.

## Unresolved disagreements

None of the reviewers' factual claims that were re-run disagreed with the re-run. One judgement remains open:
- **NV-A6** argues that the PP2 direction is fully an "instance of architecture-determined extrapolation".
- **The authors' view:** this is right for the DeepSets-vs-Transformer contrast. The within-Transformer spread (local index 0.31–0.87 at near-identical natural KL) is not explained by architecture.
- **Recorded as open**, not as consensus.
