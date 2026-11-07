# Review C — Empirical ML: reproducibility, baselines, hygiene, presentation

Paper: *Causal Information Outside a Frozen Board Probe in Othello-GPT* (private review package `path1_final_review_package/`).

Reviewer role: adversarial empirical ML reviewer. I did not run the model, did not open any `*confirmation_protocol*` file, and did not read `../_superseded/`.

**Snapshot reviewed.** I copied the package to a temporary directory at about 10:41 local time and reviewed that copy. Its manuscript hash is `paper_master.md` SHA-256 `5cbbf61c…7b8849`. While I worked, the original package kept changing. At 14:57 its `paper_master.md` had a different hash (`712ac295…`) and modification time 13:08. Its `scripts/p1lib.py` and `data/archival/archival_values.json` also differed from my copy. The newer md adds text that bears on findings 4, 5 and 8. I note where the live text already addresses a point, but my findings refer to the snapshot and to the shipped PDF, which predates both versions of the md.

## Summary

The paper asks whether edits in the kernel of a frozen 64-square linear board probe at `blocks.6.hook_resid_mid` of Othello-GPT can still move legal-move probabilities in a target-specific way. The probe's readout is held fixed to roundoff. Every edit starts from \(h_E\), the activation produced by a board editor ("Editor A") from the natural recipient activation \(h_R\).

The main results are these. At matched L2, correct-target kernel directions beat wrong-target directions by +0.39. At matched output divergence (JS 0.01), the advantage shrinks to +0.027 over wrong-target and +0.023 over random directions. A train-only ridge map from symbolic board features reproduces about +0.020 over its wrong-target control. The completeness tests fail: switching misses its joint gate, the visible–kernel interaction is near zero, and absolute scores stay negative.

The package is unusually careful. Saved logits are re-scored from scratch, bootstrap intervals match the original runs to 3e-8, an independent harness agrees on all 46 comparisons, and numeric mutations are caught.

My main concerns:

1. **The release does not pass its own checks.** The verifier and pytest both fail because the PDF and TeX were built from an older md.
2. **The verifier checks numbers but not claims.** Reversing the direction of a conclusion in the text passes unnoticed.
3. **The matched-divergence floor was lowered after an inconclusive run.** The paper says floors were preregistered.
4. **The headline is conditional on Editor A's output, but the conclusion and title are not.** Shipped raw data show kernel steps barely move the output at natural activations.
5. **Unreported data in the package show a small effect.** The full-donor ceiling and the probe-visible-only patch put the matched-divergence effect in perspective: the correct direction closes 3.4% of the Editor-A-to-donor gap, against 1.4–1.7% for the controls.

## Recommendation

- **Workshop (e.g., a mechanistic-interpretability or BlackboxNLP-style workshop): accept after required fixes.** Rebuild the PDF from the final md and pass the package's own checks. Disclose the floor change. Scope the conclusion to \(h_E\). Report the full-donor and probe-visible ceilings. The careful negative reporting and the artifact are a real contribution at workshop level.
- **Main track (NeurIPS/ICLR/ICML): reject in current form (major revision).** The effect is small and is shown for one probe, one layer, one editor-induced reference state, and evaluation pairs reused across studies. The key control is missing: kernel edits from natural activations. Shipped data suggest the effect may depend on Editor A having pushed the activation into an unusually sensitive region. Several pivotal results (SEM, FRESH, VNF) are archival console values with weak chain of custody.

## Reproduction run

Environment: macOS, `/Users/shkh/opt/anaconda3/bin/python` (Python 3.9.13, numpy 1.26.4, matplotlib 3.8.4, pytest 8.4.2), outside the sandbox. I copied the package with `T=$(mktemp -d /tmp/revC.XXXXXX); cp -R <package> "$T/pkg"`, deleted `__pycache__` in the copy, and ran each command from `$T/pkg`.

| Command | Exit | Summary line |
|---|---|---|
| `env -u PYTHONPATH /Users/shkh/opt/anaconda3/bin/python scripts/verify_paper_claims.py` | **1** | `795 passed, 2 failed`. The failures are `FAIL PDF was built from the current paper_master.md` and `FAIL paper_master.tex header carries the current md hash`. |
| `env -u PYTHONPATH /Users/shkh/opt/anaconda3/bin/python scripts/reproduce_figures.py` | 0 | `wrote 5 figures to …/figures`. All PNG and PDF figures and `figure_provenance.json` were byte-identical to the shipped files afterwards (checked with `cmp`). |
| `env -u PYTHONPATH /Users/shkh/opt/anaconda3/bin/python -m pytest tests/test_paper_reproducibility.py -q` | **1** | `2 failed, 17 passed, 1 warning in 29.75s`. The failures are `test_tex_is_generated_from_current_markdown` and `test_verify_paper_claims_passes`, for the same stale-build reason. |
| `env -u PYTHONPATH /Users/shkh/opt/anaconda3/bin/python verification/independent_harness.py` | 0 | `46 of 46 comparisons agree`; `token list matches saved NST tokens: True`. |

Additional checks on the copy:

- **Stale PDF.** `manuscript/build_record.json` and the TeX header record md hash `bbed03f7…`, but the snapshot md hashes to `5cbbf61c…`. Regenerating TeX from the snapshot md (`scripts/build_pdf.py --tex-only`, copy only) shows the shipped PDF lacks content that is in the md:
  - the AlterRep / Ravfogel et al. (2021) paragraph and reference;
  - the Hazineh et al. (2023) sentence and reference;
  - the statement that clustering on recipient game amounts to resampling pairs, and the wrong-donor dependence caveat;
  - two wording changes.
- **`RELEASE_MANIFEST.json`.** Four files mismatch (`REPRODUCIBILITY.md`, `manuscript/paper_master.md`, `manuscript/references.bib`, `scripts/cross_artifact_check.py`). Two files are not listed (`audits/sentence_audit.csv`, `scripts/sentence_audit.py`).
- **Mutation tests.** Each test ran on a separate copy of my copy. Before these runs I restored the build record that my `--tex-only` run had rewritten. With the restored record the baseline is 2 failures. The first three mutation runs predate the restore, so their counts include two extra build-record failures. The added failures named below are what matter.

| Mutation (copy only) | Verifier result | Caught? |
|---|---|---|
| Abstract: `(\(+0.0266\)` → `(\(+0.0276\)` | 5 failed. Added failure: `every numeric token … covered by the ledger … '0.0276'` | Yes |
| `analysis/fixed_js_pairs.csv`, one `delta_specific` value +0.001 | 6 failed: `analysis/fixed_js_pairs.csv equals recomputation`, plus fig2 and fig3 input-hash failures | Yes |
| `tables/table2_fixed_js_by_category.csv`, `5.298` → `5.299` | 5 failed: table reproducibility and manuscript-table match | Yes |
| "both with intervals that include zero" → "…exclude zero" | No new failure | **No** |
| "below the preregistered 60% gate" → "above the preregistered 60% gate" | No new failure | **No** |
| "The absolute score stayed negative." → "The absolute score became positive." | `795 passed, 2 failed`, identical to baseline | **No** |
| Figure 4 caption: "closer to the stage-mean direction than to the oracle" → reversed | No new failure | **No** |

## Findings

### 1. BLOCKING — The shipped PDF and TeX are stale, and the package fails its own required checks

- **Location:** `README.md`, "Quick check" (*"python scripts/verify_paper_claims.py … python -m pytest …"*). `REPRODUCIBILITY.md`, "The three required commands", and "Clean-extraction test" (*"The logs are `logs/clean_extraction_*.txt`"*). `manuscript/build_record.json`.
- **Problem:** Two of the four required commands exit 1. The PDF a reader would open is not the paper in `paper_master.md`: it lacks the AlterRep and Hazineh related-work text and their references, among other changes. `REPRODUCIBILITY.md` says `logs/` contains the clean-extraction logs and a run in a freshly extracted copy, but `logs/` holds only `integration_check_against_repo.{json,txt}` and `tectonic_build.txt`. No ZIP exists, although the docs refer to "the ZIP". `RELEASE_MANIFEST.json` is stale. The original package was also being edited during my review (md, `p1lib.py`, `archival_values.json`), so no single frozen artifact corresponds to "the paper".
- **VERIFIED:** I ran the commands (exit codes above), compared hashes, regenerated TeX in my copy and diffed it against the shipped TeX, and checked `RELEASE_MANIFEST.json` against the files. I also listed `logs/`.
- **Required change:**
  - Freeze the md.
  - Rebuild TeX and PDF with `scripts/run_all.py --rebuild-pdf`.
  - Regenerate `RELEASE_MANIFEST.json`.
  - Actually perform the clean-extraction run and ship its logs, or delete those sentences.
  - Produce the ZIP the docs describe.
  - Re-run all four commands and record zero failures.

### 2. MAJOR — The verifier certifies numbers, not claims; reversed conclusions pass

- **Location:** `REPRODUCIBILITY.md`, "The three required commands": *"…requires every number in the manuscript text to be covered by a ledger row."* `README.md`: *"checks every number in the text against its source"*. `manuscript/paper_master.md`, "Reproducibility and data availability": *"check every number in this manuscript against its source"*. Also `tests/test_paper_reproducibility.py::test_verifier_rejects_altered_manuscript`.
- **Problem:** The docs describe what the verifier does accurately, but a reader will take "verified" to mean "claims verified". Four mutations that reversed the qualitative meaning of a sentence passed with no new failure: interval excludes vs includes zero, gate passed vs failed, absolute score positive vs negative, and a caption's comparison reversed. Several other features weaken the check:
  - `context`-mode ledger rows certify tokens without evaluating anything (9 rows).
  - `*`-token rows certify every number inside a span with one boolean.
  - Realized counts such as "17 of 32 triplets" are certified only by checking that the protocol text contains the design string "16 A + 16 B" (rows `vnf07`, `vnf12`, `dis03`).
  - The self-test mutates only numbers and a result letter.
  - Smaller hygiene issues: claim IDs `set40` and `set41` are duplicated in the ledger, and `JSSMOOTHDIFF` is defined twice in `verify_paper_claims.py`.
- **VERIFIED:** I ran the mutation runs listed in the table above and read `verify_paper_claims.py` (`ledger()`, `ledger_env()`) and the ledger CSV.
- **Required change:**
  - Say plainly in all three places that the verifier checks numeric tokens and result letters, not the direction of qualitative statements.
  - Add `true`-mode ledger rows for directional claims. Examples: "interval includes zero" → `CI(...,'lo') < 0 < CI(...,'hi')`; "stayed negative" → `R('nst/summary/mean_Y/learned') < 0`; "below the 60% gate" → `A('fresh.both_positive_frac') < 0.6`; "closer to the stage-mean than to the oracle" → a cosine comparison.
  - Add a sign-flip mutation to the test suite.
  - Fix the duplicate IDs and the duplicate function.

### 3. MAJOR — The matched-divergence effect floor was lowered after the first run was inconclusive, but the paper says floors were preregistered before outcomes

- **Location:** `paper_master.md` (snapshot and PDF), "Statistics": *"Coverage gates and effect floors were preregistered before outcomes."* Appendix A, "Preregistered gates": *"The matched-divergence study required … a mean correct-minus-wrong contrast of at least 0.02"*.
- **Problem:** `data/protocols/08r2_fiber_cost_preregistration.md` line 62 required *"Mean \(\Delta_{\mathrm{JS}}\ge 0.05\)"*. The "restoration" protocol `08r2_fiber_cost_fixed_js_preregistration.md` was written after that run was recorded as `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`, and line 62 of it sets *"Mean \(\Delta_{\mathrm{wrong}}\ge 0.02\)"*.
  - The observed +0.0266 passes 0.02 and would fail 0.05.
  - The restoration protocol also preregistered secondary JS targets 0.005 and 0.02 (line 50). They were never run: every row of `fixed_js_calibrations.csv` and `fixed_js_dose_search.csv` has `target_js = 0.01`.
  - Interpolating the dose-search log at JS 0.005 gives correct − wrong = +0.0181 (40/48 wins), below the 0.02 floor.
  - The live md (13:08) now discloses the floor change, but the PDF and snapshot do not, and the secondary targets are still unmentioned.
- **VERIFIED:** I grepped both protocol files, counted `target_js` values in the CSVs, and interpolated the dose-search log in log-JS per direction. The interpolation reproduces +0.0266 exactly at 0.01, which validates the method.
- **Required change:**
  - In the main text, state that the 0.05 floor was replaced by 0.02 after the inconclusive run, and that the result passes the revised floor only.
  - Report that the secondary targets were not run, or run them.
  - Remove "effect floors were preregistered before outcomes" as a blanket statement.

### 4. MAJOR (BLOCKING for a main track) — The conclusion and title generalize beyond the Editor-A reference state, and shipped data cut against generalizing

- **Location:** Title. "Conclusion": *"The justified conclusion is the limited one: this probe is not a sufficient causal description of the residual information that affects legal-move outputs."* "Nullspace edits move the output…": *"FIBER shows that the kernel is causally active."* "Limitations": *"We did not measure kernel effects starting from the natural recipient activation in the matched-L2, matched-divergence, or donor-free studies."*
- **Problem:** Every recomputed result starts from \(h_E\). The manuscript calls Editor A a "full-space residual editor" with an L2 budget of 16 whose own validation was negative, and says it "may produce unnatural activations". The shipped file `data/raw/fiber/08r2_fiber_natural_anchor.csv` holds kernel steps added at the natural donor activation:
  - donor kernel direction: L2 1 → mean JS 9.6e-7, mean |ΔY| 0.00032; L2 4 → mean JS 4.8e-5 (max 6.7e-4), mean |ΔY| 0.0015;
  - random kernel direction: L2 4 → mean JS 6.7e-6.
  
  At \(h_E\), a correct kernel step of mean L2 5.3 reaches JS 0.01. That is about 200× the divergence seen at L2 4 from the natural donor, and far more than L2 scaling (roughly quadratic for JS) would explain. Donor outputs are confident (mean Y ≈ 0.76), so saturation may explain part of this, but it is the only natural-anchor evidence in the package. It suggests the probe's kernel is causally potent mainly at Editor-A-perturbed activations. The snapshot paper does not report it; the live md has since added it to the FIBER subsection. The conclusion "this probe is not a sufficient causal description of the residual information…" is stated without the \(h_E\) qualifier, and so is the title.
- **VERIFIED:** I summarized `08r2_fiber_natural_anchor.csv` by `family` × `eps` (768 rows) and compared it with the fixed-JS L2 values in `analysis/results.json`.
- **Required change:**
  - Scope the title, abstract and conclusion to "at activations produced by a probe-space board editor" or similar.
  - Report the natural-anchor sensitivity contrast in the main text.
  - For a main-track submission, run the missing control: SPEC, fixed-JS and NST with kernel directions \(P_N(h_D-h_R)\) and \(P_N\hat z(b_t)-P_N h_R\) added to \(h_R\), calibrated from \(h_R\). Report recipient-anchored sensitivity, not only donor-anchored.

### 5. MAJOR — Shipped ceilings show the matched-divergence effect closes about 2% more of the gap than controls; random kernel noise alone gets half way

- **Location:** "Target identity still matters when output divergence is matched": *"the effect is a shift of 0.02–0.03 in probability mass, not recovery of the target's move distribution."* Table 1. Abstract: *"At matched divergence the advantage over wrong targets was small"*.
- **Problem:** The paper admits the effect is small but gives no scale. `fiber_spec_full_logits.npz` contains two cells the paper never reports. `full_donor` (whole \(h_D\) patched) gives mean Y = **+0.726** (G 0.726, Q 0.000). `correct_rowspace` (probe-visible part of \(h_D-h_E\) only) gives **+0.363**. Against the Editor A baseline of −0.639, the fraction of the Editor-A-to-donor gap closed at JS 0.01 is:

  | Direction (JS 0.01 from \(h_E\)) | Mean Y | Gap closed |
  |---|---|---|
  | Correct kernel | −0.5925 | 3.4% |
  | Wrong-target kernel | −0.6190 | 1.4% |
  | Random kernel | −0.6154 | 1.7% |

  Other numbers that put the effect in context:
  - SPEC's correct kernel edit at its larger L2 closes 57%; the probe-visible part alone closes 73%.
  - Random kernel directions raise Y by +0.023 over Editor A on their own. Q, the probability on moves that should have become illegal, falls under generic disruption because \(h_E\) still puts about 65% of its mass on the recipient's legal moves. So about half of the correct direction's gain over \(h_E\) is reproduced by isotropic noise.
  - JS from the donor's next-move distribution: Editor A 0.474, correct 0.447, wrong 0.465, random 0.464.
  - The median correct − wrong contrast is 0.0179 (`fixed_js_scientific_report.md`), against a mean of 0.0266. The median is never reported.
- **VERIFIED:** I recomputed Y, G, Q and JS-to-donor from `fiber_spec_full_logits.npz` and `fixed_js_full_logits.npz` with the package's own `p1lib.y_components` and `js_divergence`, for 48 pairs.
- **Required change:** Add the `full_donor` and `correct_rowspace` rows to Table 2 or Table 1. Report each contrast also as a fraction of the \(h_E\)→\(h_D\) gap, and report medians. State in the abstract that random kernel noise of equal divergence already raises Y by about +0.02 over Editor A.

### 6. MAJOR — A key baseline is missing: undoing Editor A's own kernel displacement

- **Location:** "Stage and legal-mask features explain much of the learned advantage": *"Much of its average gain over Editor A is a stage correction"*. "Activations and interventions": *"Editor A is a full-space residual editor"*.
- **Problem:** Editor A edits all 512 dimensions, so it also moves \(P_N h\) away from the natural manifold. Any direction that pulls \(P_N h_E\) back toward typical natural values could raise Y relative to \(h_E\): the stage mean, the learned map, or the oracle. Each of these has the form "natural-looking \(P_N\) minus \(P_N h_E\)". The 56% "stage share" and the +0.054 learned − Editor A gain may therefore measure repair of Editor A's damage rather than target information. The correct − wrong and learned − wrong contrasts cancel this repair component, which is why the specificity claims survive. The gain-over-Editor-A and stage-share claims do not.
- **Judgment:** This needs the control \(P_N(h_R-h_E)\), which restores the recipient's own kernel component, calibrated to JS 0.01. The package has no activations, so I could not compute it.
- **Required change:** Add the \(P_N(h_R-h_E)\) control and a "global training-mean kernel component" control to NST and SEM. Interpret the stage share relative to them, or state this alternative explicitly in the main text.

### 7. MAJOR (main track) — One probe, one layer, one editor: the design cannot separate "this probe is incomplete" from "this linear probe family at this site"

- **Location:** "Model, site, and probe": *"That site was fixed before the studies reported here; we did not search over layers."* "Limitations": *"Another probe, a nonlinear decoder, or a readout at a later layer might detect the edit."*
- **Problem:** The claim concerns one frozen probe. The natural follow-up baselines are all absent:
  1. probes with other seeds or regularization (is the effect in the intersection of their kernels?);
  2. an MLP probe;
  3. the same test at layers 4, 5 and 7;
  4. a later-layer probe, to see whether the kernel edit changes the board read downstream (the most direct test of "information the probe cannot see");
  5. a second editor (e.g., a kernel-preserving rowspace-only editor), so the \(h_E\) dependence can be assessed.
- **Judgment:** These are cheap given the existing infrastructure.
- **Required change:** For a main track, include at least items 1, 3 and 4. For a workshop, list them as limitations in the abstract-level summary.

### 8. MAJOR — Archival evidence has weak chain of custody and selective reporting of gate components

- **Location:** "Evidence levels": *"Archival values come from verbatim console output of the original runs"*. Appendix C: *"The archival values come from eleven verbatim console excerpts, each hashed."* `data/archival/console_excerpts/fresh_full_run.txt`.
- **Problem:**
  - **Chain of custody.** `data/provenance_manifest.json` gives each excerpt's source as *"verbatim Colab console output pasted by the experimenter into the research session transcript"*, with a chat-transcript file and line index. That transcript is not in the package. The excerpt hashes were computed by the packaging script (`scripts/dev_build_provenance.py`), so the verifier's "hash matches provenance manifest" and "value occurs in its excerpt" checks are self-certifying. They prove only that the package is internally consistent. The paper does not disclose that the console text was recovered from a chat transcript.
  - **Unreported gate failure.** The FRESH gate dict shows `'natural_symmetric_ok': 0, 'anchor_dependence_ok': 0`. The paper reports only the 60% joint-positive failure. `anchor_dependence_ok` is not defined in any shipped protocol. `corr_P_C=0.0745` in `fresh_anchor_audit.txt` is also unreported.
  - **No power check for VNF.** Only interaction terms are given; no main effects \(d_T(T)\) and so on. A reader cannot tell whether VNF's null interaction is informative or whether its fixed-coefficient (ρ = ν = 3.5) edits barely moved Y.
  - **Unexplained STAGE_ONLY zeros.** `vnf_stage1.txt` shows STAGE_ONLY A_T = A_W = 0.0. That is expected, because T and W are matched on player and occupancy, so the stage map cancels. It means FRESH's A_T measures the non-stage part of the map, a useful fact the paper does not mention.
- **VERIFIED:** I read all eleven excerpts and the manifest entries, and grepped the protocols for `anchor_dependence`.
- **Required change:**
  - Disclose the transcript provenance.
  - Define and report every failed gate component.
  - Report VNF main effects from the console if they exist, or state that they were not logged.
  - Mention that STAGE_ONLY gives 0 by design on FRESH.
  - Downgrade the language from "verbatim console output of the original runs" to "console output recovered from a session transcript".

### 9. MAJOR — "Recomputed" overstates what can be checked without probe weights and activations

- **Location:** "Model, site, and probe": *"Because \(W P_N=0\), adding \(P_N\delta\) to an activation leaves every probe logit … unchanged"*. `data/README.md`: *"Probe weights … available from the authors on request. The package does not need it"*. Table 1, "Evidence" column.
- **Problem:**
  - The probe weights (about 200k floats) and the activations \(h_R\), \(h_E\), \(h_D\) (48 × 3 × 512 per study) are tiny and are the authors' own artifacts, but they are not shipped. A reader therefore cannot check that \(W P_N = 0\) for the stated probe, that \(\delta_{\text{correct}} = P_N(h_D-h_E)\), or that the saved probe probabilities came from that probe.
  - What is "recomputed" is re-scoring of saved model outputs and saved probe outputs.
  - Legal sets are read from saved JSON rather than regenerated by a simulator.
  - I did verify that every saved SPEC and NST direction lies in the range of the saved \(P_N\): the maximum relative off-kernel norm is 3e-15 for SPEC and 9e-15 for NST. But \(P_N\) itself is unverifiable against the probe.
- **VERIFIED:** I computed \(\lVert(I-P_N)v\rVert/\lVert v\rVert\) for all 336 SPEC and 288 NST directions and read `p1lib.recompute_*`.
- **Required change:** Ship the probe weights and the three anchor activations per pair. Add a check that \(\max\lvert W P_N\rvert \approx 0\) and that \(P_N(h_D-h_E)\) equals the saved correct direction. Alternatively, rename the evidence level "re-scored from saved outputs".

### 10. MINOR — The FRESH study uses a different reference state than Figure 1 says, and the paper does not use that fact

- **Location:** Figure 1 caption: *"Activations and the probe geometry used in every study. … Each experimental edit adds a nullspace direction \(t\,\delta\) to \(h_E\)."* "Reliable target switching…": *"With the edit applied to the natural recipient activation and calibrated to JS 0.01"*.
- **Problem:** FRESH's gate is evaluated at the natural anchor (`08r2_symmetric_target_preregistration.md`: *"Gate (FULL, natural anchor)"*). So "every study" is false. It is also unclear which anchor VNF used. FRESH also partly answers the limitation in finding 4: from \(h_R\), the learned map gives A_T = +0.0285 and A_W = +0.0186. The paper never connects these.
- **VERIFIED:** I read both FRESH protocols and the FRESH console excerpts.
- **Required change:** Correct the Figure 1 caption. State the anchor for each study in Table 1. Discuss FRESH's natural-anchor A_T and A_W as the only recipient-anchored kernel evidence.

### 11. MINOR — Undefined codenames and internal jargon make the paper hard to follow

- **Location:** Throughout.
  - *"(SPEC)"*, *"(NST)"*, *"(FIBER)"*, *"(SEM)"*, *"switching study (FRESH)"*, *"factorial study (VNF)"* are never expanded.
  - *"the 08R2 validation"*.
  - *"Editor A"* (there is no Editor B).
  - The letters *"`0A = C / PARTIAL`; `SECTION_5_3 = PARTIAL`; `OBSTRUCTION_READY = NO`"* in "Conclusion" → "Official result letters".
  - *"class-logit contrasts"*.
  - Pair-manifest hashes in prose (*"`2789accf…1143d3`"*).
- **Problem:** SEM collides with "standard error of the mean". "A" is overloaded three ways: category A, Editor A and \(A_T\). The official-letter block refers to sections of an internal plan the reader never sees. "Kernel" and "nullspace" are used interchangeably without saying so. `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED` appears in the text but not in the "Official result letters" list or the README list. Table 1 says "Matched JS" while the text says "matched-divergence study".
- **Judgment:** I read the manuscript.
- **Required change:**
  - Replace codenames with descriptive names (L2-matched study, donor-free map, feature ablation, switching study, factorial study), or define them in a table.
  - Rename Editor A to "probe-space board editor".
  - Rename the categories (e.g., far and near pairs).
  - Move the official-letter block and the hashes to an appendix, with a one-line gloss for each letter.

### 12. MINOR — Editor A is described inconsistently

- **Location:** Abstract: *"after a probe-space board editor has been applied"*. Introduction: *"Editor A, a probe-space residual editor"*. "Activations and interventions": *"Editor A is a full-space residual editor"*. Conclusion: *"the gain over the probe-space editor"*.
- **Problem:** Whether Editor A changes only the probe rowspace or all 512 dimensions is central to findings 4 and 6. The paper says both. The live md changes the introduction to "full-space" and adds *"Its manifold penalty was off"*, but the abstract and conclusion still say "probe-space".
- **VERIFIED:** I grepped the snapshot md; each phrase occurs once.
- **Required change:** Use one accurate description everywhere: probe-objective, full-space, no manifold penalty.

### 13. MINOR — Figure and table presentation

- **Figure 4A:** the dashed "0.01 floor" runs across all four contrast rows, but it applies only to learned − wrong. Draw it only on that row or label it. There is also no learned − oracle row, although the oracle is the natural comparison.
- **Figure 1:** text overflows its boxes (*"64×3 board probabilities"* touches the border; *"frozen later layers → next-move distribution (61 tokens)"* runs past the right edge). See also finding 10 on the caption.
- **Figure 5:** panels A and B use different x scales (0–0.04 vs 0–0.005) without a note. This is acceptable but should be flagged in the caption.
- **Figures 2 and 3:** captions match the plotted data.
- **Figure 5 inset:** 13 + 13 + 5 + 1 = 32, consistent with 81% and 56%.
- **Table 2 in the PDF:** it floats into the middle of an Appendix B sentence, splitting *"In category A the learned-minus- / wrong contrast"* across pages 16–17.
- **Stage share:** the 56% share has no interval anywhere (Table 5 or text).
- **PDF completeness:** the PDF is otherwise complete. It has 17 pages, sections 1–8, references, Appendices A–C, 5 figures and 2 tables, extracted with PDFKit.
- **VERIFIED:** I viewed every PNG, cross-checked each against `tables/` and `analysis/results.json`, and extracted the PDF text.
- **Required change:** Fix the items above, add a bootstrap interval for the stage share (or state that one cannot be formed because SEM is archival), and pin Table 2 with `[h]` or `[t]`.

### 14. MINOR — Wording that overclaims

- Abstract: *"A 64-component ridge map … produced a similar advantage on 48 new pairs"*. +0.0201 on different pairs is not shown to be "similar" to +0.0266, and no between-sample comparison is made. Use "a smaller positive advantage".
- Related work: *"which is a tighter control on the alternative that an edit merely disrupts the next-move distribution"*. Tighter than what? Divergence matching still lets random directions use 2.3× the L2 (Table 2), and finding 5 shows disruption alone raises Y. Say "a control", or explain the comparison.
- "Nullspace edits move the output…": *"FIBER shows that the kernel is causally active."* Add "at \(h_E\)"; see finding 4.
- Conclusion: *"this probe is not a sufficient causal description of the residual information that affects legal-move outputs"*. Add the \(h_E\) scope.
- Related work, novelty: *"We did not find a prior study that edits within the kernel of one frozen board probe … That is an observation about the sources we checked, not a priority claim."* This is acceptably hedged. However, the snapshot already discusses AlterRep (Ravfogel et al., 2021), which keeps the nullspace and changes the rowspace. "The reverse of AlterRep" is the accurate framing. Also cite Makelov et al. (2024) on dormant pathways activated by subspace interventions (the live md now does). That literature is the main threat to the \(h_E\) results.
- **Required change:** Edit the sentences as indicated.

### 15. MINOR — Package documentation inaccuracies

- `README.md`, Layout: *"`reviews/` | Three adversarial reviews and the responses to them"*. At copy time `reviews/` was empty, and no responses exist.
- `README.md` letter list omits `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED`, which the paper uses.
- `REPRODUCIBILITY.md`: "after the ZIP is extracted", but there is no ZIP.
- `scripts/reproduce_figures.py` rewrites files inside the package (documented, but a "quick check" that mutates the artifact is surprising). Prefer an `--out` option.
- My sandboxed file listing could not see `FINAL_SCIENTIFIC_VERDICT.md`, `RELEASE_MANIFEST.json` or several `audits/` files (a local `.cursorignore` effect). This is harmless but worth knowing for anyone using sandboxed tooling.
- **VERIFIED:** I listed the directories and read the scripts.
- **Required change:** Make the docs match the shipped contents.

## What I checked and found sound

- The analysis library (`scripts/p1lib.py`) contains no hard-coded results. It recomputes Y, G, Q, JS, TV, probe deltas, L2 values and clustered bootstrap intervals from saved logits, probe tensors and directions, and raises on missing cells or hash mismatches. The only hard-coded expectations are regression values in `test_headline_numbers`, which is acceptable.
- `verification/independent_harness.py` is a genuine re-implementation: its own token map, scipy JS with a direct-formula assertion, and `rng.integers` bootstrap draws rather than `rng.choice`. It matched to 1e-9 on all 46 comparisons. It shares seeds and the saved legal sets with the library, so it is independent code, not independent data.
- Every recomputed bootstrap interval matches the original runs to at most 2.7e-8 (`tables/table3`).
- An independent dose-response check from `fixed_js_dose_search.csv` gives correct − wrong = +0.0079, +0.0144 and +0.0266 at JS 0.001, 0.003 and 0.01. Win counts are stable (40–41/48). The 0.01 result is not a lucky operating point within the range searched, though nothing above about 0.03 can be checked.
- The SEM label mapping is correct (console `d_board_inc` = FULL − LEGAL_ONLY = 0.0041), and the stage-share identity closes to 3e-18.

## Strengths

- The negative results are reported in the main argument rather than buried: absolute scores stay negative, switching fails its gate, and the interaction is null. Each official result letter is preserved verbatim.
- The probe-invariance definition is precise and correct (full 64×3 tensor, relative to \(h_E\), at 1e-14). The paper is explicit that \(h_E \ne h_R\) and quantifies how far apart they are.
- Matched-divergence calibration with a frozen grid that never looks at Y is a good design. Reporting both the matched-L2 and matched-divergence results makes the "random barely moves the output" artefact of matched L2 visible.
- The wrong-target control is scored on the intended target's legal sets, which is the right choice and is stated clearly.
- The evidence-level labelling (recomputed / recorded / archival / protocol) is an excellent practice, and the archival values are honestly flagged in the tables and Figure 5.
- The package engineering is strong: hash-checked raw data, byte-identical tables and figures under the pinned matplotlib, a numeric-token ledger that catches every number mutation I tried, and an independent harness.
- The Pearson-r discussion in the donor-free section (learned vs oracle r = 0.998 explained by baseline variation) is exactly the kind of self-skepticism reviewers want to see.
