# Post-results Review D: causal interpretation, statistics, novelty

**Reviewer.** Reviewer D, an AI agent. **This is not human peer review.**

**Scope.** This review checks the reports against the frozen outputs:
- causal-identification claims, including whether each "eliminated" explanation is really eliminated;
- statistical uncertainty, double counting and evidence labels;
- novelty, and the separation of O/D from the results.

Reproduction, compute and archive integrity belong to the other reviewers and were not re-done here, beyond the freeze-hash check below.

**Read.**
- `sources/MANDATE_XIX.txt` §8–§19, `DESIGN_XIX.md` (v2.1), `FREEZE_XIX.json`.
- `logs/conf/{score_c19, stats_c19, sensitivity_summary_c19, func_score_c19, compat_c19(.out), s100diag_*, verdicts_c19}.json`, `logs/FUNCTIONAL_BATTERY_FREEZE19.md`, `logs/DEV_NOTES19.md`.
- All reports named in the brief, plus `COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md`, and the XVIII `PRIOR_WORK_AND_NOVELTY.md`.

**Computations.** Every new number below was computed with read-only Python on the JSON files. No training was run and no existing file was modified.

---

## 0. What was verified and found correct

These checks passed. Nothing in them needs correcting.

- **Freeze integrity.** All 39 files listed in `FREEZE_XIX.json` hash to their frozen SHA-256 values. This includes `DESIGN_XIX.md`, `score19.py`, `func19.py` and `design19.json`.
- **Verdicts.** The 35 primary verdicts in `FROZEN_CONFIRMATION_AND_STATISTICS.md` match `verdicts_c19.json` exactly: 27 SUPPORTED, 5 PARTIAL, 3 CONTRADICTED.
- **Per-pair values.** Every per-pair φ in the frozen table matches `score_c19.json` to two decimals.
- **Sensitivity variants.** The "variants that differ" column matches `sensitivity_summary_c19.json`. Only five hypotheses change:
  - E_P4_vsgo_switch and its 0.70 reading: delta0;
  - E_P4_vsgo_exceeds_vsg: lin;
  - M_P3_moments_alone_nil: thr_pm05 and thr_strict;
  - M_P4_vsgo_exceeds_vsg: delta20.
- **Gates.** pnz ratios are 1.003–1.007. lbl ratios are 0.975–1.018 on E and 4.73–12.38 on M. The M block excess is 0.0076–0.0201, above 0.005, so the ratio rule applied. All gates pass, as reported.
- **Evaluability and competence.** 6/6 pairs are evaluable. All 99 models are competent: maximum KL 0.0348, minimum accuracy 0.806.
- **Duplicate-hash disclosure.** The pairs (fullI, paramI_optB) and (fullB, paramB_optI) are duplicates because the hash covers parameters only. Their continuations differ, which confirms that they are distinct states. This is correctly explained.
- **The central success condition (mandate §19) is met by one route only: X8.** The A7/pnz family (X8a) and the lbl family (X8b) were predeclared to eliminate, and both are SUPPORTED on both bases. The other eliminations claimed in MAIN §2 do not meet the predeclared rule (findings 2 and 4).
- **O/D separation is respected.** No result is derived from or interpreted through O/D. The appendix is labelled speculation. (Advisory A9 covers one wording issue.)

---

## Findings

### 1. MANDATORY: the vsg/vsgo wording reverses the causal direction, and says routing is unchanged when it is not

**The reversal.** The reports say "value-vector learning through the route is sufficient". The experiment shows the opposite direction: *removing* (stop-gradient) value-vector learning through T during the window is sufficient to produce the switch. Learning is not what produces it. The reversed wording appears in:
- MAIN §1 l.34;
- MAIN §2 l.43 ("Value-vector learning alone is sufficient");
- MAIN Q27 ("the sufficiency of value-vector learning through the blocked route");
- FORWARD B3 l.50;
- the MAP table row "Content against value learning".

**"With forward and routing unchanged" is an overclaim.** The graph audit's "routing gradient = plain" is a *per-step, fixed-θ* identity. Over the window the trajectories diverge, and the learned routing state moves. From `s100diag_M82x.json`:

| arm on M | fraction of the block's drop in T attention share at step 100 | nec_M at step 100 |
|---|---|---|
| vsg | 0.63, 0.64, 0.65 | ≈ 0, like I (B: 0.012–0.020) |
| vsgo | 0.69, 0.70, 0.72 | ≈ 0 |

So on M, vsg reaches about two-thirds of the block's routing change by step 100. Endogenous routing adaptation is a live mediator. FORWARD B3 l.52 concedes this, but MAIN §1, §2 and Q27 drop the caveat.

**Required correction.** Replace every instance with this wording, or an equivalent:

> "On M, stopping the gradient into the value vectors read through T during the window (vsg), or into V and WO (vsgo), is sufficient for a full switch (0.97–0.99). This holds although the forward pass and the per-step routing gradient are those of plain training given the current parameters. The learned T attention share still falls by 63–72% of the block's fall by step 100. So a routing-state change that follows from the altered value learning is not excluded as the mediator. This is a statement about a modified learning rule (synthetic gradient), not about an ordinary network losing information."

### 2. MANDATORY: "vsg is sufficient on M" is a post-hoc reading of a CONTRADICTED prediction, and the predeclared P4 joint reading failed on both bases

**The prediction that was frozen.** M_P4_vsg_partial predicted a *hybrid*. It is CONTRADICTED: 0.975, 0.973, 0.988.

**The predeclared reading rule.** DESIGN §5 states: "P4: read only together: vsgo SUPPORTED, vsg PARTIAL and *vsgo − vsg ≥ 0.2* SUPPORTED". Elimination of content loss and routing-gradient removal as necessary causes was conditional on that joint pattern.
- On M, two of the three components are CONTRADICTED.
- On E, vsgo is PARTIAL and vsg is CONTRADICTED.

So the predeclared P4 elimination did not occur on either base.

**What survives.** The logical point survives on M through **M_P4_vsgo_switch (SUPPORTED)** alone: vsgo keeps the forward pass and the per-step routing gradient, and it switches. Using it alone, however, departs from the frozen reading rule. It is not "a test whose ability to distinguish was established before confirmation" in the predeclared form.

**Where the claim is mislabelled.**
- MAIN §1 l.34 and Q10 label "vsg full switch" [frozen].
- MAIN Q12 says "vsg on M: yes" to "does a graph-verified intervention reproduce the *expected* effect?" The expected effect was a partial switch, and it was falsified.
- MAIN §2 l.51 lists "forward-content loss is necessary … on M" among the eliminations that meet the central success condition.

**Required corrections.**
- (a) Label the vsg-sufficiency statement **[frozen data; post-hoc reading of a CONTRADICTED prediction = exploratory]** wherever it appears: MAIN §1, §2, Q10, Q27; FORWARD B3; MAP.
- (b) Change Q12 to: "vsg on M: **no**. The predicted partial switch was falsified, and the outcome was a full switch."
- (c) In MAIN §2 l.51, list the central-success eliminations as X8a and X8b only. Add one sentence: "On M, M_P4_vsgo_switch (SUPPORTED) also shows that forward-content loss and per-step routing-gradient removal are not necessary. This reading departs from the predeclared joint P4 rule, which failed on both bases."
- (d) Rewrite FORWARD B3 "This eliminates the following for M, as necessary causes" so that it rests on M_P4_vsgo_switch, not on vsg, and carries the same disclosure.

### 3. MANDATORY: the E-base "V+O learning carries most of the effect" is an interpretation of a PARTIAL that was predeclared to mean "not established"

**Values.** E vsgo: 0.69, 0.56, 0.95. Mean 0.73, descriptive t95 [0.25, 1.22]. Only 1/3 pairs fall in the band. DESIGN §5 says a PARTIAL here "is predeclared to mean 'not established'".

**Problem.** MAIN §2 l.43 ("On E: value-pathway learning carries most of the effect"), FORWARD B3 l.59 and the MAP row present this as an established attribution. In addition, φ is a log-necessity composite. "0.56–0.69 of φ" is not "most of the effect" in any additive sense, although the lin variant gives similar values (0.62–0.92).

**Required correction.** Replace the statement with:

> "[frozen, PARTIAL = not established] On E, removing V+O learning through T gives large but incomplete switches on 2 of 3 pairs (0.56, 0.69) and a near-full switch on E823 (0.95). Whether V+O learning suffices on E is not established."

Drop "carries most".

### 4. MANDATORY: "frz's nil was shown to be an optimizer artefact" is not established at frozen level, and frz/frzp are mislabelled "natural trajectory"

**The predeclared reading** (DESIGN §5 P5) needed **frz SUPPORTED-nil** and frzp partial. On E, E_P5_frz_nil is **PARTIAL** (0.01, 0.01, 0.50). On M, frz and frzp are both nil, so there is no contrast.

**What the data show.** The artefact reading rests on 2 E pairs. On E823 both arms are partial (0.50 against 0.69). The claim nevertheless appears as established in:
- MAIN §2 l.47 ("shown to be an optimizer artefact: frz and frzp have identical θ_100 and different outcomes"; on M the outcomes are not different);
- ROUTING C2 ("[frozen] … So the frz nil is an optimizer-state artefact");
- MAP ("frz nil shown to be a moment artefact [frozen]").

**The labelling error.** DESIGN §4 classes frz as "a freeze plus an optimizer-state manipulation" and frzp as a parameter freeze. Neither is a natural trajectory. Yet three places call them natural:
- MAIN Q5 ("They also matter on a natural trajectory: frz against frzp");
- TRANSPLANTS A3 ("natural-trajectory moment effect");
- ROUTING C2 ("a natural-trajectory demonstration").

MAIN Q7 lists "vrs, frzp, pnz, lbl, and the synthetic-gradient vsg / vsgo" as natural-trajectory evidence. That contradicts FORWARD B3 l.63 ("vrs is the only natural intervention") and DESIGN §4's B3 categories.

**Required corrections.**
- (a) Write "[frozen, PARTIAL] On 2 of 3 E pairs, frz (nil) and frzp (partial) differ with identical θ_100. This is consistent with the moment-artefact explanation of pre-review B, which is not established at frozen level (E823: both partial). On M there is no contrast."
- (b) Replace "natural trajectory" with "training-time freeze arms (not natural trajectories)" in MAIN Q5, TRANSPLANTS A3 and ROUTING C2.
- (c) Rewrite MAIN Q7 as: "Natural intervention: vrs only. Non-targeted disturbances: pnz, lbl. Freezes: frzp; frz (with optimizer manipulation). Synthetic learning rules: vsg, vsgo. Synthetic states: all transplants."

### 5. MANDATORY: the M "identified combination" is a sufficient set, not an identified combination of necessary changes, and some [frozen] labels cover development-only results

**Development-only results labelled [frozen].** "Neither the blocked layer nor layer 0 is sufficient alone" (MAIN §1 l.30; TRANSPLANTS "What is identified [frozen]" l.142) rests partly on B←I L2. That arm was not run in confirmation; its value, 0.18, is from M801. Restoring L2 alone and the non-necessity of L1 are also development-only.

**Necessity on M was not established at frozen level.**
- Restoring L0 is **PARTIAL** (0.32, 0.43, 0.75). On M823 it barely reduces the switch.
- No restoration of L1, or of L0+L1, was run in confirmation.
- So the "substitutable partner" is established for **sufficiency only** (B←I L0+L1 and B←I L0+L2 both 0.97–0.99).
- B←I L1+L2 (without L0) was never run, so even the claim that L0 is *required* for sufficiency is untested.
- TRANSPLANTS A4–A6 still says "[dev] Layer 0 is necessary in the sense that restoring it removes most of the switch". The frozen result does not replicate this on M823.

**Required corrections.**
- (a) Mark the blocked-layer-alone and L1 statements [dev] in MAIN §1, MAIN Q2 and TRANSPLANTS.
- (b) Replace "carried by a combination of layer 0 with either layer 1 or layer 2" (MAIN §1, Q2, Q6, MAP) with: "[frozen] L0+L1 and L0+L2 are each sufficient, and L0 alone is not (0.26–0.38). Restoring L0+L2 prevents the switch (0.06–0.13). Restoring L0 alone is PARTIAL. The necessity of L0, the necessity of either partner, and whether L1+L2 without L0 suffices are not established."
- (c) Change "Confound 7 … resolved" (MAIN §2 l.45) to "addressed. The state is localised at the level of sufficient layer sets; necessity is established for E layer 0 and for M L0+L2 only."

**The E tensor-class combination is also mis-described** (TRANSPLANTS l.74, "an identified combination", [dev]).
- No single layer-0 class is necessary: restoring QK or OV alone leaves 0.96–0.98 [frozen].
- No single class or pair of classes is sufficient: pairs give 0.35–0.61 [dev].
- B←I (all of L0 except QK, plus *I's non-L0 layers*) = I←B L0_QK gives 0.96–0.98 [frozen]. B←I L0_OV+MLP+LN *with B's non-L0 layers* gives 0.35 [dev].

So I's non-L0 state contributes when L0 is incomplete. That is an X7 interaction with groups outside layer 0, which the design said would be "documented". Write: "[dev + frozen] Layer-0 tensor classes are jointly sufficient and individually redundant for necessity. When layer 0 is incomplete, the non-L0 state interacts with it (X7). This is not an identified combination of individually necessary changes."

### 6. MANDATORY: the functional battery does not support Claim 2, even "in the weak sense", and the 86/86 statement is misdescribed

**Reviewer computation (exploratory), on the 99 confirmation models** (`func_score_c19.json` × `score_c19.json`):
- SR correlates **r = 0.97 with nec_E** and **r = 0.94 with KL to Bayes**. Spearman is 0.77 and 0.76.
  - Pre-review C3 rejected the RPC score as Claim-2 evidence for correlating 0.91 with causal necessity. SR is route-free, which is a real gain. But by the same standard it is a behavioural re-expression of how strongly the early route dominates, together with the competence level.
- The non-hybrid KL ranges do not overlap:
  - EARLY-class: 0.0237–0.0348 (n = 43);
  - MIXTURE-class: 0.0115–0.0166 (n = 43).
  - **A single cut KL > 0.02, chosen post hoc by this reviewer, reproduces all 86/86 SR classes.**
- The frozen interpretation boundary already states that isotropic logit noise around Bayes gives SR 2.9–3.2.

Together, this is what one would expect if both organisations compute the same Bayes rule and the EARLY implementation is about 2× less precise, with noise-like errors that concentrate at low Bayes margin. A noisier implementation of the same rule is **Claim 1 plus a precision/competence difference**. It is not "a different input–output rule or evidence-combination function" (mandate E1, Claim 2).

**Hybrids show the signature is not proportionate to the causal switch.**
- All 7 E hybrids, with φ 0.50–0.69, are already fully FM (SR 1.19–1.47).
- M823 I←B L0, with φ 0.75, is FE.

So FUNCTIONAL E2's "Every intervention that switched the causal route also switched this behavioural signature. Every intervention that did not, did not" holds only for the non-hybrid models.

**Count errors.**
- MAIN §1 l.37 says "Every causal switch is accompanied by a change in a route-free behavioural signature (86 of 86 models)". The 86 are the 6 + 6 anchors (F1), 34 causal-switch arms (F2) and 40 causal-none arms (F3). Only 40 models (34 + 6 fullI) are switches. The 86 models come from 6 pairs, and models within a pair share trajectories. The independent units are 3 per base.
- MAIN Q16 quotes "SR 2.6–4.0, against 1.0–1.3 [dev, frozen]". The **confirmation** ranges are **2.33–3.50 (EARLY) and 1.03–1.52 (MIXTURE)**. The quoted figures are development ranges.

**Required corrections.**
- (a) In FUNCTIONAL "What this shows" and "Conclusion", MAIN §1 and Q15, and FROZEN Stage-6: replace "supports Claim 2 only in the weak sense" and "more than route reorganisation of an identical function (Claim 1 alone)" with this text, or an equivalent:

  > "[frozen] Route-free error structure follows the causal class in all 86 non-hybrid models (40 switched, 46 not; 3 independent data orders per base). SR correlates 0.97 with nec_E and 0.94 with KL. The EARLY organisation is a ≈ 2× less precise approximation whose error pattern matches unstructured logit noise around Bayes. [interp] This is best described as **causal implementation reorganisation with a precision difference** (mandate E4 wording). It does not establish a different functional rule (Claim 2) or a different algorithm (Claim 3)."

- (b) Fix the MAIN §1 count as in (a).
- (c) Use the confirmation SR ranges in MAIN Q16.
- (d) Restrict the FUNCTIONAL E2 "every intervention…" sentence to non-hybrid models, and state the hybrid asymmetry.

### 7. MANDATORY: the "general disturbance eliminated" wording must stay scoped, and two of the three controls are functionally inert at step 100

**Unqualified wording.** MAIN §1 says "in the forms tested", which is correct. MAIN Q27 ("The elimination of general disturbance") and the MAP row ("**no**") drop the qualifier. NON-TARGETED D2 also asserts "These are not weak controls".

**The step-100 data show the X8a controls change nothing functional:**

| control | step-100 KL | other step-100 measures |
|---|---|---|
| pnz, E | 0.451–0.457 (B 0.451–0.456) | T share 0.230–0.240 (B 0.232–0.235); nec_E 0.230–0.291 (B 0.225–0.301); L0 cosine with the block ≈ 0.00 |
| pnz, M | 0.427–0.429 (B 0.428–0.430) | — |
| randnorm (compat), E | 0.449–0.459 | nec_E 0.241–0.289 |
| randnorm (compat), M | 0.429–0.430 | — |

So X8a eliminates only "a Euclidean displacement of matched norm in an isotropic direction". Such a displacement is expected to fall mostly in flat directions and is functionally inert here. Only lbl is a functionally real disturbance. lbl also degrades learning of *all* routes uniformly; it is not a deficit specific to the target route.

**Required corrections.**
- (a) Write in MAIN Q27 and the MAP row: "General disturbance is eliminated in the tested forms: (i) matched-norm isotropic displacement, at step 100 or as a random walk, which leaves step-100 function unchanged; (ii) label-noise loss disturbance, matched on E and 4.7–12.4× larger on M. Disturbances that are functionally comparable but not aligned with I − B are tested only by lbl."
- (b) In NON-TARGETED D2, delete "These are not weak controls" or restrict it to lbl, and add the step-100 numbers above.
- (c) In D4, change "a disturbance that delays the early route's acquisition" to "a disturbance that leaves the early route non-functional at step 100 (nec_E 0.026–0.029), while also slowing the learning of every other route".

### 8. MANDATORY: prediction counting double-counts, so "27 of 35 succeeded" overstates the independent confirmatory evidence

Mandate §11 and §19 forbid counting rescored versions of the same causal observation as independent. Four items break this:
- E_P4_vsgo_switch_sens070 is a sensitivity reading of the same 3 numbers (DESIGN: "never replaces it"), yet it is counted among the 35 and among the 5 PARTIAL.
- M_P4_vsg_partial and M_P4_vsgo_exceeds_vsg are one observation: vsg ≈ vsgo ≈ 1. FROZEN says "contradicted as a consequence", but MAIN §1 and Q23 count three falsifications.
- E_P4_vsgo_exceeds_vsg is SUPPORTED only because vsg was *nil*, which is contrary to the predicted mechanism. It is fragile: lin gives PARTIAL, descriptive t95 [−0.02, 1.01], and E823 is 0.26 against an edge of 0.2. Counting it as a "success" (Q22) is misleading.
- E_A7 and E_P6_pnz form one family (X8a, "counted once per base", DESIGN §5). The two P3 bands form one family per base, the M partner bands one family, and P5 one family. P7 eliminates nothing by design.

**Required correction.** Add a family-level tally to MAIN Q22–Q24 and FROZEN "Totals", and state that the 35-item count is not a count of independent confirmations.

| | E | M |
|---|---|---|
| SUPPORTED elimination families | P1, P2, A4-QK, A4-OV, P3, X8a, X8b, vrs: 8 | P1-L0-alone, P2-L0L2, partner sufficiency, P3, X8a, X8b, vrs, P5: 8 |
| P4 joint reading | not established | failed |
| P5 joint reading | not established (frz PARTIAL) | — (SUPPORTED, counted above) |
| PARTIAL | — | P2-L0-alone |
| Concordance checks, eliminate nothing | P7 ×2 (1 SUPPORTED, 1 PARTIAL) | P7 (SUPPORTED) |

### 9. MANDATORY: statistical statements need scope and must not be read as equivalence tests or as two-base comparisons

**P3 "equivalence margin 0.25" is band inclusion, not an equivalence test.**
- M_P3_moments_alone_nil: mean 0.084, descriptive t95 **[−0.14, 0.31]**. The upper bound exceeds the 0.25 margin.
- Two frozen variants (thr_pm05, thr_strict) give PARTIAL.
- On M822 the moments alone move the early-route component to 0.35, with composite φ 0.19.

MAIN Q5 ("At the full-state level, no … [frozen, equivalence margin 0.25]") should say: "all 3 pairs inside a ±0.25 band; with n = 3 this is not a statistical equivalence demonstration (M t95 upper bound 0.31); M822 shows a moment effect on the early-route component (0.35)."

**The unit population needs to be stated.** Pairs within a base share the parent *and* the fresh-layer initialisation seed: L = W = 21 for every E pair and L = W = 23 for every M pair (DESIGN §4). Only data order varies. The Clopper–Pearson bound (3/3 → 0.37) is therefore over data orders for one (parent, init) combination per base. Because each base is one init seed, every E-against-M contrast compares **n = 1 init seed per side**. This affects:
- FORWARD B3 "The two bases differ. On M, the block of the late route acts through value-vector learning alone. On E, … needs more";
- MAIN §1 l.33–35.

**Required correction.** In MAIN Q21 and the FROZEN "Independent units" section, add: "Within a base the fresh-layer initialisation is fixed (L = W = 21 E, 23 M). Units vary data order only. E-against-M differences are confounded with that single init seed and with the target layer." Qualify FORWARD B3 accordingly.

### 10. MANDATORY: the novelty statement (Q28) omits the closest prior work, including the work XVIII itself identified

Mandate §15 requires reviewing "the closest work identified during XVIII" before any novelty claim. XVIII's `PRIOR_WORK_AND_NOVELTY.md` lists Kleinman, Achille & Soatto (ICLR 2024; arXiv 2308.12221). Their deficit is a `.detach()` on one of two parallel pathways: forward kept, learning blocked. It decides which pathway wins the race, and only early deficits matter.

That is the direct conceptual precedent for XIX's main process-level result: the vsg/vsgo full switch on M. Gradient-only removal with the forward pass intact decides the organisation. Q28 does not cite it. Q28 also omits other work that XVIII listed and that bears on XIX:
- Chen et al., ICLR 2024 (arXiv 2309.07311): brief training-time suppression of an attention structure changes the learned strategy;
- Singh et al., ICML 2024 (arXiv 2404.07129): training-time clamping of transformer components dissects circuit formation;
- Cloud et al. 2024 (arXiv 2410.04332): gradient routing decides where a computation forms.

The transplant, re-initialisation and mode-connectivity literature that mandate §15 names is also missing:
- **Zhang, Bengio & Singer**, *Are All Layers Created Equal?* ([arXiv 1902.01996](https://arxiv.org/abs/1902.01996); JMLR 2022): layer-wise re-initialisation to earlier checkpoints to test which layers carry what was learned;
- **Frankle, Schwab & Morcos**, *The Early Phase of Neural Network Training* ([ICLR 2020](https://iclr.cc/virtual/2020/poster/1934); [arXiv 2002.10365](https://arxiv.org/abs/2002.10365)): perturbation, permutation and re-initialisation of early-training states and their effect on the outcome;
- **Juneja et al.**, *Linear Connectivity Reveals Generalization Strategies* ([ICLR 2023](https://iclr.cc/virtual/2023/poster/12031); [arXiv 2205.12411](https://arxiv.org/abs/2205.12411)): runs from one pretrained parent fall into distinct basins that correspond to distinct competent strategies;
- **Lubana et al.**, *Mechanistic Mode Connectivity* ([ICML 2023](https://proceedings.mlr.press/v202/lubana23a.html); [arXiv 2211.08422](https://arxiv.org/abs/2211.08422)): mechanistically different solutions sit in different basins, and fine-tuning may not move between them.

Juneja et al. and Lubana et al. anticipate the picture of "two competent organisations from one parent, separated by a basin boundary". So does XIX's own interpolation result ([dev]: half the layer-0 I − B difference does nothing).

**Required correction.** Extend Q28 (and the MAP open questions 3–4) with these works. Restate the increment as:

> "[interp] In one parent and one init seed per base: (i) a state-level localisation by synthetic transplants of where a durable switch between two competent organisations is carried at step 100 (E: layer 0, close to tautological; M: sufficient layer pairs); (ii) exclusion of Adam moments at full-state level; (iii) exclusion of tested non-targeted disturbances. The finding that a gradient-only deficit with the forward pass intact can decide the outcome replicates, in a transformer, the mechanism of Kleinman et al. (2024). It is not new. No new phenomenon is claimed."

Remove "on one base the blocked route's value-vector learning alone suffices" from the increment (see findings 1 and 2).

### 11. MANDATORY: the reports cite a final adjudication that does not yet exist

MAIN's header ("post-results: D and E (`REVIEWS_AND_ADJUDICATION_FINAL.md`)") and Q26 ("The resolutions there are applied to this document") present the post-results reviews as adjudicated and applied. At review time `REVIEWS_AND_ADJUDICATION_FINAL.md` does not exist.

**Required correction.** Leave Q26 empty, or mark it pending, until the adjudication is written. Then list each weakened or withdrawn claim explicitly, including the items in this review. Do not state that resolutions are applied before they are.

### 12. MANDATORY: evidence labels on the 31 answers are missing or wrong

Mandate §18 requires an explicit label on every answer.

**Missing labels:**
- Q22–Q24 should be [frozen]/[falsified].
- Q25 should be [record of pre-review].
- Q27 should be mixed: [frozen] for X8, transplants and P3; [frozen data, post-hoc] for vsg.
- Q30 should be [interp].
- Q31 should be [spec].

**Wrong labels:**
- Q10 and Q12, vsg on M (finding 2).
- Q5 and Q7, "natural" (finding 4).
- Q16 ranges, dev presented as frozen (finding 6).
- Q17 hybrids: F4 is descriptive with no prediction, so label it [frozen data, descriptive].
- MAIN §1 "Where the switch is carried [frozen]", which covers the dev-only blocked-layer claim (finding 5).

**Required correction.** Add or fix the labels as listed.

### 13. MANDATORY: "stable hybrid" is unsupported

MAIN §1 l.35 ("Freezing layer-0 routing parameters gives a stable hybrid") and ROUTING C3 ("a stable **partial** switch") say "stable". XIX has no readout beyond step 1211; XVIII's stability readout was at step 2546, for other arms. Also, E823 frzp has causal class MIXTURE (φ 0.69), so not all three runs are hybrids.

**Required correction.** Write "a partial switch at step 1211 (0.54–0.69; HYBRID on 2 of 3 pairs, MIXTURE-class on E823); stability not tested".

### 14. MANDATORY: the E layer-0 headline must carry its own predeclared caveat

DESIGN §3 says the E localisation to layer 0 is "close to tautological … The informative E results are the sub-group restorations, the moments, and the training-time arms". MAIN §1 l.29 and Q27 headline "the layer-0 parameter state, sufficient and necessary" without the caveat, which appears only in Limitations.

The compat data reinforce the point. At step 100, B←I L0 has nec_E 0.021–0.026 (route already non-functional), and I←B L0 has 0.084–0.109 (partly restored).

**Required correction.** Add the caveat in MAIN §1 and Q27 next to the claim.

### 15. MANDATORY: "not yet visible in the step-100 function" on M must be qualified

TRANSPLANTS ("**The M selection is therefore carried by a step-100 state difference that is not yet visible in the step-100 function**"), MAP open question 3 ("functionally indistinguishable") and the MAP appendix need qualifying.
- Step-100 output KL (0.425–0.445) and route necessity are indeed indistinguishable. nec_M is below the stated 0.02 measurement floor in all states.
- The T attention share is not: B 0.167–0.170 against I 0.138–0.139 (`s100diag_M82x.json`), an 18% difference.

**Required correction.** Write "indistinguishable in output KL and in route necessity (below the measurement floor), while the T attention share already differs (0.167–0.170 against 0.138–0.139)".

---

### A1. ADVISORY: development values are quoted inconsistently

- M801 vsg is 0.44 in DESIGN §5 but 0.43 in FROZEN and FORWARD B2.
- E801 frzp is 0.57 in DESIGN §5 but 0.59 in DEV_NOTES.

Quote the DESIGN (measure19) values consistently.

### A2. ADVISORY: one cross-arm pattern on E partly answers the tautology concern

On E, step-100 route function is neither necessary nor sufficient for the switch:
- lbl leaves nec_E at 0.026–0.029, near I, with T-share drop 0.79–0.82 of the block's and L0 cosine 0.76–0.77. It does not switch.
- vsgo leaves nec_E at 0.175–0.263, near B. It switches 0.56–0.95.

This partly answers the tautology concern in finding 14. It is worth stating as [expl] in TRANSPLANTS or FORWARD B4.

### A3. ADVISORY: layer 0 alone has a consistent partial effect on M

M B←I L0 alone gives 0.26, 0.33, 0.38, with t95 [0.17, 0.48], which excludes 0. Layer 0 alone is not inert. State this next to "insufficient".

### A4. ADVISORY: Q31 overstates what its proposed test would show

Q31 says vsgo plus a projected stop would make the E mechanism "an identified V+O+routing combination". It would still be a sufficiency result under a synthetic learning rule. Identification would also need the corresponding necessity tests and a natural-trajectory analogue. Soften the wording.

### A5. ADVISORY: disclose that the confirmatory power came from one development pair

Mandate §11 rejects predictions "already known from development". Every XIX prediction except M_A7 replicates a 1-pair development observation. DESIGN §5 discloses this, and fresh-unit replication is legitimate. MAIN's central-success sentence should still say that the tests' discriminating power was established on one development pair per base, and that this pair misled on 3 predictions.

### A6. ADVISORY: quote the exact lbl ratio range

MAIN Q14 says "5–12×". The values are 4.7–12.4×.

### A7. ADVISORY: Q18 asks for sufficiency, which was not measured

Q18 asks about necessary *and sufficient* routes. Only test-time necessity was measured. Say that test-time route sufficiency was not tested.

### A8. ADVISORY: one contrast passed under the predeclared margin but has a weak lower bound

The E_P4_vsgo_exceeds_vsg descriptive lower bound is −0.02. If the contrast is cited at all, report it with this uncertainty.

### A9. ADVISORY: the O/D appendix generalises an M-only result

The O/D appendix sentence "by a latent, not yet functional state difference" holds for M only, and only with the finding 15 qualifier. Write "on the mixture base". O/D separation is otherwise respected.

---

## Summary of what stands

**Stands as written, within one parent, one init seed per base and 3 data orders:**
- full-state parameter mediation, with moments excluded at band level (finding 9 qualifies this);
- E layer-0 sufficiency and necessity (finding 14 caveat);
- the E QK-only and OV-only restorations do not prevent the switch;
- M L0+L1 and L0+L2 sufficiency, and the L0+L2 restoration;
- the vrs full switch;
- elimination of the tested X8a and X8b forms (finding 7 wording).

**Does not stand as written:**
- "value-vector learning is sufficient" (direction reversed; post-hoc; routing state not unchanged);
- the E V+O attribution;
- "frz artefact shown" and "natural trajectory";
- the M "identified combination";
- the functional battery as Claim-2 evidence;
- "86/86 switches";
- the novelty statement;
- the claim that final adjudication is already applied.

VERDICT: The main claims stand only after corrections. The state-level localisation, the exclusion of optimizer moments and the elimination of tested general disturbance survive with tighter scope. Five must be withdrawn or relabelled as exploratory: the value-learning sufficiency claim on M, the V+O attribution on E, the optimizer-artefact claim, the "identified combination" on M and the Claim-2 functional interpretation. The novelty section must be rewritten to acknowledge Kleinman et al. (2024) and the transplant and mode-connectivity literature.
