# Stale-document reconciliation

Three earlier drafts of this paper existed under `docs/papers/`: `probe_invisible_causal_information/`, `path1_corrected_package/` (with a ZIP), and `path1_md_pdf_review_bundle/` (with a ZIP). One independent review of the bundle also existed, `~/Downloads/path1_zip_independent_review.md`. None of these drafts was tracked by git, so deleting them could not be undone. They were moved unchanged into `docs/papers/_superseded/`, each with a `SUPERSEDED.md` note that points here. Nothing from them is in this package. This file records what each old document said, whether that is still true, and what replaces it.

## Obsolete Figure 2 variants

| Old file | What it was | Status |
|---|---|---|
| `probe_invisible_causal_information/figures/fig2_probe_preservation.{pdf,png}` | early probe-preservation figure | Removed from use. It was superseded and not regenerated from arrays. |
| `probe_invisible_causal_information/verified_data/figure2_REVISED_from_independent_audit.pdf` | scatter of 336 interventions, drawn by the independent audit from a secondary CSV | Removed from use. It treated 336 interventions as points rather than 48 pairs. |
| `path1_corrected_package/figures/fig2_probe_preservation.pdf`, `path1_corrected_package/verified_data/figure2_REVISED_from_independent_audit.pdf` | copies of the two above | Removed from use. |
| `path1_md_pdf_review_bundle/figures/fig2_probe_preservation_BLOCKED.pdf` | placeholder card saying the figure was blocked | Removed from use. |

The single Figure 2 is `figures/fig2_probe_and_output_by_category.{pdf,png}`. It is drawn by `scripts/reproduce_figures.py` from `analysis/fixed_js_pairs.csv`, which is recomputed from the saved fixed-JS logits and probe probabilities. Panel A is the largest absolute change in any of the 64×3 probe probabilities. Panel B is the total variation distance of the next-move distribution. Both are measured relative to the Editor A reference activation and aggregated to the 48 pairs. The verifier fails if `figures/` holds any figure the manuscript does not use, or if the obsolete file names appear anywhere in the package.

## Old audit documents

| Old document | What it said | Status now | Replaced by |
|---|---|---|---|
| `01_experiment_registry.csv` | registry of the studies | The study list is unchanged. Sample sizes and seeds are now checked against protocols. | `tables/table1_study_overview.csv`, `data/protocols/`, `audits/provenance_inventory.md` |
| `02_artifact_integrity_report.md` | letters read from source constants | Still true; the letters are unchanged. They are now checked verbatim against the raw completion-status files. | `scripts/verify_paper_claims.py` (letter checks) |
| `03_claim_evidence_ledger.csv` | ledger whose values came from `reported_numbers.json`, a typed summary | Stale. A summary file was presented as evidence. | `audits/claim_evidence_ledger.csv`, which is evaluated against in-memory recomputation from raw arrays |
| `04_dataset_split_and_leakage_audit.md` | sample reuse; 24 repeated board-and-player prefixes in the FRESH cohort | The reuse facts are still true. The 24 prefixes are in the 400-game cohort; whether any reach the 32 selected triplets cannot be checked without the triplet arrays. | `audits/leakage_and_reuse_audit.md`; manuscript Discussion and Limitations |
| `05_statistical_recomputation.md` | SPEC, fixed-JS and NST were not recomputed because the arrays were absent | Superseded. The arrays were located in the original result archives and copied with hashes. Every SPEC, fixed-JS and NST number is now recomputed, and each recomputed interval matches the original run to within \(3\times10^{-8}\). | `audits/recomputation_report.md`, `analysis/results.json` |
| `05_statistical_recomputation.md` | "Fixed-JS required 44/48 complete calibrations" | Still true as the preregistered coverage gate; all 48 pairs completed. | `data/protocols/08r2_fiber_cost_fixed_js_preregistration.md` |
| `06_negative_result_register.md` | negative and inconclusive results | The content still holds. It was rewritten against the recomputed and archival sources. | `audits/negative_result_register.md` |
| `07_reproducibility_levels.md` | Level 1 "NOT RUN" for FIBER, SPEC, fixed-JS and NST | Superseded for those four studies, which are now recomputed. Still true for SEM, FRESH, VNF and the board-edit validation. | `REPRODUCIBILITY.md` (evidence-level table) |
| `08_ml_writing_conventions.md`, `12_manuscript_style_rules.md`, `writing_conventions_research.md` | style guidance | Applied. They are not evidence. | `audits/scientific_sentence_audit.md` |
| `09_related_work_matrix.csv`, `10_novelty_assessment.md` | label `NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS` | That label is not used. Novelty is assessed without priority claims. | `audits/novelty_review.md` |
| `11_citation_verification.md` | citation checks | Still valid. The uncited ROME entry was dropped, and the Jacopin DOI was rechecked and resolves as written. | `manuscript/references.bib` and the verifier's bibliography check |
| `PATH1_FINAL_SCIENTIFIC_ASSESSMENT.md` | `PATH1_MANUSCRIPT_COMPLETE_BUT_SCIENTIFIC_BLOCKERS_REMAIN`; Gate B (recomputation) failed | Superseded. Gate B now passes for the four studies with arrays; the remaining limitation is the archival status of SEM, FRESH and VNF. | `FINAL_SCIENTIFIC_VERDICT.md` |
| `REPRODUCIBILITY.md` (old) | ran `verified_data/verify_fixed_js.py` and scripts reading `reported_numbers.json` | Stale. The old scripts imported the research repository, and their paths did not exist in the shipped ZIP. | `REPRODUCIBILITY.md` |
| `final_language_audit.md` | wording corrections | Applied where still relevant; the manuscript was rewritten since. | `audits/scientific_sentence_audit.md` |
| `submission_readiness_report.md` | do not submit; licenses not verified; authorship absent | Still true. | `FINAL_SCIENTIFIC_VERDICT.md` |
| `unresolved_scientific_issues.md` | eleven open issues | Items 1 to 5 (missing arrays for four studies, Figure 2, checkpoint and probe hashes, the truncated interval for NST learned minus Editor A, rank 124) and item 11 (no page inspection) are resolved. The interval is recomputed as \([0.0452, 0.0618]\). The hashes come from the runs' dependency manifests. The rank comes from the saved singular values. Every PDF page was inspected. Items 6 to 10 (sealed confirmation unused, manifold status, secondary JS levels, authorship, other probes) remain open and are stated as limitations. | `audits/page_inspection.md`, manuscript Limitations |
| `reviewer_1/2/3_*.md`, `reviewer_response_and_revisions.md` | simulated reviews of an earlier draft | Superseded. The reviews were repeated on this package. | `reviews/` |
| `abstract_and_title_candidates.md`, `appendix_*.md`, `limitations_and_threats_to_validity.md` | draft fragments outside the master file | Superseded. All text now lives in the single source. | `manuscript/paper_master.md` |
| `paper_master.pdf` in `path1_md_pdf_review_bundle/` | 4-page condensed text, different from the Markdown | Superseded. The PDF is now generated from the Markdown; the build record ties their hashes. | `manuscript/paper_master.pdf`, `manuscript/build_record.json` |

## Errors in the old drafts that were corrected

- JS divergence and probe invariance were sometimes described as measured from the unedited (natural) recipient. Both are measured from the Editor A reference activation \(h_E\). Measured from \(h_R\), the same patched activations have mean JS 0.043 rather than 0.01. The verifier rejects the old phrase.
- An early draft called the Pearson correlation of 0.998 between learned and oracle scores a direction correlation. The direction cosines are 0.792 between learned and oracle and 0.850 between learned and stage-mean. The manuscript reports both and explains the correlation as a reflection of baseline \(Y\).
- Tables and the ledger were populated from typed summary values. They are now generated from recomputation or from verbatim, hashed console excerpts. Every archival value carries that label.
- Scripts imported the research repository and referenced directories absent from the shipped ZIP. No package script imports the repository, and a clean-extraction test runs the pipeline.
- Hard-coded section numbers produced duplicated numbering in the PDF. Headings are now unnumbered in the Markdown and numbered once by LaTeX.

## The independent ZIP review

`~/Downloads/path1_zip_independent_review.md` listed five serious issues and seven clarifications. Each is addressed in this package:

| Issue | Resolution |
|---|---|
| 0.998 misattributed as a direction correlation | Corrected as above; the cosines are recomputed from the saved directions. |
| Markdown, PDF and TeX did not match | Single source; generated TeX and PDF; build record; cross-artifact check. |
| Missing scripts, data and bibliography | Raw arrays, scripts, tests, bibliography and manifest are included; clean-extraction test. |
| Figure 2 blocked; Figure 1 implied that rowspace is causally unused; Figures 3 to 5 typed from reports; Figure 5 unreferenced | Figure 2 is drawn from arrays and pair-aggregated. The Figure 1 caption and drawing state that both components feed the later layers. Figures 3 and 4 are recomputed. Figure 5 is labeled archival. All figures are referenced in the text. |
| Novelty review not exhaustive | Jacopin (2026) and Lin et al. (2024) are discussed in related work; no priority claim; `audits/novelty_review.md`. |
| FRESH overlap of 24 prefixes | Stated as a cohort-level fact that cannot be checked on the selected triplets. |
| Bootstrap replicate counts | Stated per study and checked against the protocols. |
| 336 interventions are not 336 independent units | The 48 pairs are the unit; Figure 2 and the Statistics subsection say so. |
| Mean \(Y_{\mathrm{learned}}\) negative | Reported in the abstract and in the NST results. |
| Equal JS is not equal L2 | Stated in the setup, the results and the limitations. |
| Authorship and disclosure | Withheld pending human confirmation; stated on the title page. |
