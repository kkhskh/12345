# Independent review of Path 1 Markdown/PDF bundle

**Reviewed:** 2026-11-05. **Bundle SHA-256:** `2a2d0828860b2dca4bc61eeaaf5c52ef5c333f884855515d60b7bb42af86ed74`. **Scope:** All 31 embedded files (25 Markdown and 6 PDF), rendered PDF pages, figure PDFs, historical independent audit documents and, separately, the existing prior `fiber_spec.zip`, `fiber_cost_fixed_js.zip`, and `null_state_transfer.zip` available in this conversation. **No new real-model forward pass.**

**Verdict:** The standalone empirical thesis is coherent and appropriately scoped, and the main fixed-JS and NST results have independent saved-array support. However, this submitted review ZIP is **not** a reproducible, complete, submission-ready paper. Its own label, `PATH1_MANUSCRIPT_COMPLETE_BUT_SCIENTIFIC_BLOCKERS_REMAIN`, should be preserved. A material direction-correlation error must be fixed. The wider novelty review should be reopened narrowly to include additional relevant Othello literature.

## What was actually inspected

- 31 files, including `paper_master.md` (about 5,224 whitespace-delimited words) and `paper_master.pdf` (about 1,785 words, 4 pages). The PDF is a condensed *different text*, roughly 34% of the Markdown's word count. It is not a full PDF rendering of the canonical paper.
- All five separate figure PDFs were rendered. Figure 2 was a blocker card. Figures 3, 4 and 5 plot previously transcribed summary statistics. Figure 1 is schematic. No standalone figures are embedded in the four-page `paper_master.pdf` (zero PDF image objects and none visible on page renders). The canonical Markdown contains no numbered figure references or image embed syntax.
- Three simulated reviewer reports, the revision memo, the abstract/title candidates, appendices and limitations, and the audit set, including provenance, statistical recomputation, split/leakage, negative results, reproducibility levels, novelty, citation verification, writing guidance and final assessment.
- This ZIP contains **no** `.npz`, `.csv`, `.json`, `.py`, `.tex`, or `.bib` files, although the package's reproducibility documentation names scripts, `reported_numbers.json`, `paper_master.tex`, `references.bib`, and plotting utilities that may exist in the original repository. Their absence is an archive-delivery failure, not proof they were never created.

## Independently corroborated claims using other existing uploaded archives

These files are **separate from the review ZIP**, so they strengthen our scientific confidence but do not make the review ZIP self-contained.

| Check | Result independently recomputed from archived outputs |
|---|---|
| SPEC saved interventions | 48 correct-null cells; mean correct-null `Y=+0.14364876`; saved paired correct-minus-wrong `+0.39162660` and correct-minus-random `+0.70858351` |
| Fixed JS=0.01 | 336 directions across 48 pairs, with correct/wrong/random counts 48/144/144 |
| Full probe-probability invariance under fixed JS | max absolute change `1.18794e-14` across 64×3 probability vectors versus each Editor A reference |
| JS reconstruction | maximum disagreement with archived JS `4.41226e-9`; max deviation from .01 `1.23241e-6` |
| Fixed-JS paired contrasts | correct-minus-wrong `+0.02655064`; correct-minus-random `+0.02296572` |
| Fixed-JS actual mean residual L2 | correct `5.29847`; wrong `6.36829`; random `12.26520` |
| NST saved pair-level results | prior independent saved-logit audit reproduced learned-minus-wrong `+0.020057`, learned-minus-Editor A `+0.053713`, and full probe invariance |

These checks do not replay the frozen model checkpoint or establish the provenance of the archived intervention construction independently. FRESH, VNF, and SEM arrays are still not included in this archive; their numbers must continue to be marked as sourced from the cited Colab report unless their arrays are supplied.

**New figure:** `figure2_verified_from_prior_fixed_js.pdf` was independently produced from the previous fixed-JS ZIP and is accompanied by `fixed_js_raw_probe_output_verification.csv`. This can replace the paper's Figure 2 blocker after the provenance and figure labels are integrated properly.

## Serious issue 1 — A mistaken attribution of the 0.998 correlation

The canonical Markdown at **Section 4.4** states that oracle-donor *directions* were "highly correlated with learned directions" with "correlation 0.998". Direct inspection of NST's saved 512-dimensional direction arrays shows:

- Pearson correlation of raw **Y responses** learned vs oracle: **0.997823**.
- Mean cosine similarity of **direction vectors** learned vs oracle: **0.791998**.
- Mean cosine similarity of learned vs **mean-state** directions: **0.849770**.

The manuscript conflates between-game raw outcome correlation with activation-space vector similarity. This is materially misleading, even with the sentence warning that correlations do not prove identical mechanisms.

**Required correction:** Explicitly label 0.998 as the raw behavioral correlation, report the relevant vector cosines, and note that the generic mean-state direction is *more* similar in angle to the learned vector. Also update reviewer and any manuscript sentences that implicitly treat 0.998 as representational recovery. This change does not invalidate donor-free target-aligned response.

## Serious issue 2 — Manuscript/PDF/source mismatch

`paper_master.md` is the actual full narrative. The 4-page PDF is a condensed version with roughly 1,800 words including references. The TeX source used for that PDF is absent from this archive. The PDF excludes the standalone main figures, many methods details, discussion passages, explicit limitations, and appendices. Brightly bordered reference hyperlinks appear in the rendered PDF. It should **not** be described as a checked full-paper PDF.

**Required correction:** Generate the PDF from the canonical source with a single reconciled LaTeX or other publishing pipeline; embed all data-supported figures, citations, tables, appendices and technical details; render and inspect every page. Include `paper_master.tex`, a verified `references.bib`, and build commands, or document why the build is not reproducible from the package.

## Serious issue 3 — Missing verification and reproduction assets

The bundle's `REPRODUCIBILITY.md` instructs readers to execute `reproduce_tables.py`, `reproduce_figures.py`, `verify_paper_claims.py`, and references `reported_numbers.json`; none is in the ZIP. There are no CSV evidence ledgers or pair-level outputs. It is impossible for a recipient to reproduce the archived figures using just this ZIP.

**Required correction:** Supply a private analysis-only release pack that includes read-only raw logit/probe NPZs or durable exact-location manifests for them, legal sets, pair manifests, scoring scripts, verification scripts, plot sources, a figures-to-data ledger, and output hashes. The bundle need not redistribute restricted third-party checkpoint weights, but it must state how an authorized researcher obtains them. Keep sealed confirmation inaccessible.

## Serious issue 4 — Figure presentation and claim anchoring

- Figure 2 was a blocker despite the availability of earlier independent raw fixed-JS arrays in this conversation. A verified replacement is now available.
- Figure 1 draws rowspace information only toward the probe and nullspace information toward model output. That visual can wrongly imply that probe-visible rowspace information does not also affect decisions. Correct it to show both components feed into the *same downstream frozen model*, while only rowspace is read by the frozen probe.
- Figures 3 and 4 show real reported aggregate estimates but are not regenerated from bundled saved data; label provenance or regenerate from the saved per-pair arrays.
- Figure 5 fairly shows two negative-completeness results, but the full manuscript does not refer to the figure by number at all.
- The figure PDFs should appear within the paper, with figure citations in the surrounding paragraphs, not merely as detached files.

## Serious issue 5 — Literature review not yet exhaustive enough for a definitive novelty verdict

The existing review correctly acknowledges Li et al. (2023), Nanda et al. (2023), amnesic probing, INLP/LEACE, activation steering, and causal abstraction. It is reasonable to present the matched-JS target-specific probe-nullspace intervention as a **potentially original controlled combination**.

But the review should also compare, at minimum, **Jacopin (BlackboxNLP, October 2026), _Present but Barely Used_**, which studies whether an Othello world-model representation is causally used under a different training objective, and the **Lin et al. (2024) OthelloGPT bag-of-heuristics research report**, which examines causally relevant move-legality heuristics and game-stage features. Both are related to the central interpretation, even though neither source as described directly duplicates the complete fixed-probe/nullspace/matched-JS design. No claim of exact duplication is made. See sources cited in the response accompanying this report.

**Recommendation:** Treat the novelty verdict as *plausible, narrowly scoped, not priority-proven*. Add the closest missing works to related work and explain the operational difference without "first to" claims.

## Other required clarifications

1. **FRESH board-state overlap:** The report mentions 24 repeated board+player prefixes in the newly generated **400-game cohort**. Determine how many, if any, occur in the *selected 32 scientific triplets*. Do not equate a candidate-cohort overlap with a selected-evaluation leak without evidence. Game-level and state-level overlap remain distinct.
2. **Statistical protocols:** The master text uses a broad "2,000 bootstrap replicates" statement. SPEC confirms 2,000 in its prereg JSON, but check separate saved protocol/runner details before applying that figure to NST, SEM, FRESH, and VNF universally; some prior reports used other resampling counts.
3. **Data availability:** The main PDF describes reproduction by referral to missing repository material; this is not a substitute for the self-contained paper package or a working private artifact manifest.
4. **Figure source:** A scatter from 336 fixed-JS interventions is not 336 independent evaluation games. The 48 recipient/target pairs remain the unit for target-specific inference.
5. **Absolute response:** Preserve `Y_learned≈-0.582`; even the statistically positive paired advantages do not produce the intended target policy.
6. **Calibration:** Equal JS does not equal L2. Avoid calling generic "matched strength" without stating the matching variable.
7. **Human author and ethics disclosures:** Final sign-off is absent; the present author placeholder is not submission-ready.

## Strengths worth preserving

The manuscript is unusually careful about its claims relative to many early drafts. It uses connected academic prose rather than slogan-heavy bullet points; it honestly discloses the COST deviation, lack of sealed confirmation, simulator feature privilege, stage-only contribution, negative Y, incomplete symmetric switching and negative factorial interaction; and it does not claim that COR obstruction was validated. The core empirical question is intelligible and narrower than "linear probes are not causal".

## Recommended stop/go decision

**No submission as-is.** We recommend the following exact order:

1. Correct the 0.998 direction-correlation error and audit every representation-equivalence claim.
2. Include the newly independently verified Figure 2 and 336-row verification evidence from the saved fixed-JS archive; include the NST and SPEC independent audit artifacts.
3. Recover/FRESH/VNF/SEM raw arrays and manifests; perform true Level 1 analysis on them (or keep their results explicitly provisional if unavailable).
4. Assemble the missing code, JSON, figure/table provenance, manuscript source and bibliography into a private reproducibility package.
5. Extend closest-prior-work coverage through late 2026, including the 2026 Othello use-versus-read paper and 2024 heuristics work.
6. Rebuild a single full PDF from canonical manuscript and render-check it, including all verified figures and appendices.
7. Repeat the adversarial review on the **reconciled** evidence package. Then request an independent human scientific review.

**Do not open sealed confirmation, retrain the learned map, edit old verdicts, or substitute speculative COR terminology.** The strongest claim remains probe-relative, target-aligned, and limited to this frozen Othello-GPT study.

## Audit scope caveat

The ZIP's internal links to other project files and original Git history cannot be verified from the ZIP alone. A few earlier experiment ZIPs in this same chat provide genuine saved-array corroboration for SPEC, fixed-JS and NST, but they do not remove missing components of the Path 1 distribution archive. Scientific validity and completeness of a deliverable are separate questions.
