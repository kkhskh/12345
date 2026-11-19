# XIX pre-experiment review C: experimental measurement and scoring

**Reviewer.** C (measurement, classification, scoring, statistical design). This is an AI agent review, not human peer review.

**Date.** 2026-11-19 (UTC).

**Scope.**
- Mandate §8, §9, §11, §12 and §13.
- `DESIGN_XIX.md` and `XVIII_FAILURES_AND_UNRESOLVED_CAUSES.md`.
- Code: `src/xix_score.py`, `scripts/measure19.py`, `scripts/score19.py`, `scripts/scorer_validation19.py`, `scripts/scorer_validation_summary19.py`, `scripts/scorer_planted19.py`. I also read `compose19.py`, `conf_setup19.py` and the branch and STOP paths of `xix_train.py`, because the provenance checks depend on them.
- Data: `logs/scorer_validation19.json` and its summary, `logs/scorer_planted19.json`, `logs/conf_spec19_draft.json`, `logs/conf/design19.json`, `logs/dev/dev_table19.json`, `logs/dev/m19_dev.jsonl`, `logs/dev/func_c18_97950000.npz` and `logs/DEV_NOTES19.md`.

**What I ran.**
- `scripts/scorer_planted19.py`: ALL PASS. Its output file was rewritten byte-identical.
- Adversarial planted cases. These call `score19.score()` directly on records built from `scorer_validation19.json`. Scripts are in `/tmp/revC/`, outside the package.
- Numerical checks of φ_nec, of the band rules' operating characteristics, and of the functional score against the causal measure and against the functional distances in `func_c18_97950000.npz`.

I did not run any training, and I did not modify any existing file.

---

## 0. Summary judgement

**What is sound.** The causal-necessity primary measurement is sound and is a real repair of XVIII's failure:
- The class is stable across three new streams (59/60 models; the exception is non-competent).
- The φ variation between streams is ≤ 0.008 for the XVIII gsg hybrids. Measurement noise is therefore negligible against the band widths.
- No fitted behavioural parameter enters evaluability.
- Every sensitivity variant goes through the same `score()` function, so evaluability is recomputed under each variant. The specific XVIII implementation error (bands changed, evaluability not) is not present.
- All hypothesis expressions reference existing arm names. The quoted-name detection of the arms each hypothesis uses finds exactly the referenced arms (checked programmatically for all 17).

**What is not.** The scorer as written still:
- pools bases against the design's own per-base rule;
- contains an unfalsifiable hypothesis construction (P7);
- fails open on the matching gate;
- trusts derived labels in the records;
- has fail-closed checks that miss most of the F4 list.

Separately, the functional score is not independent of the causal measure and is not evidence for Claim 2.

All of these can be fixed before the freeze. None requires new training.

---

## 1. Objections

### 1. MANDATORY: `base: "both"` hypotheses pool the two bases and apply `min_pairs` to the pool, contradicting DESIGN §6 ("min 2 evaluable pairs per base")

**Code.** `score19.score()` puts every pair with `h["base"] == "both"` into one `vals` dict and tests `n < D["min_pairs"]` on the pooled count.

**Demonstrated cases (`/tmp/revC/adv.py`).**
- (a) All three M pairs are non-evaluable and three E pairs are in band. Result: `Hboth SUPPORTED n=3`. The M base contributes nothing, yet the verdict reads as "both".
- (b) E is 3/3 far outside the band and M is 3/3 inside. Result: `PARTIAL`. A base-specific contradiction is hidden.

**Affected hypotheses.** This applies to 9 of the 17: P3 ×2, A7, P6, P4 ×3, P5 ×2, and P7_vrs.

**Correction.**
- Score every "both" hypothesis separately per base, with `min_pairs` applied per base. These per-base results become the primary units (`<id>|E`, `<id>|M`).
- If a joint label is wanted, freeze it as follows:
  - SUPPORTED only if both bases are SUPPORTED;
  - CONTRADICTED if either base is CONTRADICTED;
  - INCONCLUSIVE if either base is INCONCLUSIVE and neither is CONTRADICTED;
  - otherwise PARTIAL.
- Add planted cases (a) and (b) to `scorer_planted19.py`.

### 2. MANDATORY: P7 can never be CONTRADICTED, and it can be SUPPORTED with no causal switch

**Construction.** With `lo=-10, hi=10`, no φ can fall outside [−10.1, 10.1]. A failed functional requirement only blocks SUPPORTED, so the outcome is PARTIAL.

**Demonstrated.**
- Every pair CAUSAL_ONLY (causal switch, functional = B class) gives `P7 PARTIAL`.
- Every pair FUNCTIONAL_ONLY (arm causally identical to the B anchor, functional class flipped) gives `P7 SUPPORTED`. The expression ignores the causal class, yet the hypothesis claims category BOTH.

**Test suite.** `scorer_planted19.py` encodes the defect: its `H_func_req` with a CAUSAL_ONLY arm is expected to be PARTIAL.

**Mandate.** The construction violates §11, which rejects predictions incapable of distinguishing explanations. The outcome that would refute P7 (Claim-1-only, i.e. CAUSAL_ONLY) cannot produce a refutation.

**Correction.** Replace the band construction with a categorical rule over pair categories.

Definitions:
- A pair is *P7-evaluable* for an arm if three conditions hold:
  - (i) the family is evaluable;
  - (ii) the arm's causal sub-label is SWITCH;
  - (iii) both anchors are *functionally* evaluable: B's functional class equals B's causal class, I's functional class equals I's causal class, and |f_I − f_B| ≥ 0.20 (objection 7).
- Below min_pairs P7-evaluable pairs per base, the verdict is INCONCLUSIVE. Report how many pairs failed (ii) and how many failed (iii).

Verdicts:
- SUPPORTED: every P7-evaluable pair is BOTH.
- CONTRADICTED: a strict majority are CAUSAL_ONLY.
- PARTIAL: otherwise. This includes INCONCLUSIVE(causal_SWITCH), i.e. functional AMBIGUOUS.

Alternative (continuous form):
- Use `phif[arm]`. It already exists and is unused. Require band [0.5, 1.5] plus the causal SWITCH category for SUPPORTED. CONTRADICTED is a strict majority with phif < 0.25 − δ.
- Planted tests must show CAUSAL_ONLY×3 → CONTRADICTED and FUNCTIONAL_ONLY×3 → not SUPPORTED.

### 3. MANDATORY: the functional score is not independent of the causal measure and must not be reported as evidence for Claim 2 (functional change)

**Evidence.**
1. **Selected by causal label.** `measure19.refs()` assigns the 12 XVIII c18 references to the EARLY and MIXTURE centroids by their causal class (`XS.causal_class(XS.necessity(...))`). f is therefore "residual-profile resemblance to causally-EARLY minus causally-MIXTURE models".
2. **Redundant with the causal measure.** Over the 52 competent validation models (stream 97 960 000), Pearson(f, log((nec_E+.01)/(nec_M+.01))) = **0.91**, and Spearman = 0.86. The "0 contradictions, 112 agree" result in `scorer_validation_summary19.json` is what a near-monotone function of the causal measure produces. It does not validate independence.
   - 30 of the 112 "agree" model-streams are the references themselves under leave-one-out (10 references × 3 streams; the other 2 references are AMBIGUOUS). They are scored against five same-class siblings from the same parent, bases and intervention.
3. **Functional differences are small.** In `logs/dev/func_c18_97950000.npz` (the same 12 models, natural plus relational-edit probes), the mean symmetric KL is:
   - between *different* causal classes on the same base: 0.065 (range 0.047–0.091);
   - between *same*-class models on the same base: 0.057 (range 0.037–0.083).
   - The ranges overlap almost completely. Relational edits do not increase the separation (0.060 vs 0.067).
   - The between-class predictive difference is therefore about 15% above seed-to-seed functional variation, and of the same size as each model's distance to Bayes (0.02–0.07).
4. **It cannot tell Claim 1 from Claim 2.** Claim 1 (same computation, different components) predicts different *residual* fingerprints, because different implementations make different approximation errors. A residual-correlation score therefore cannot separate Claim 1 from Claim 2. No validation model is a known positive control for CAUSAL_ONLY; the only planted CAUSAL_ONLY is a hand-edited label.
5. **It forces hybrids into binary classes.** `uP164_L20` (causal HYBRID, nec_E 0.89) is called functional MIXTURE. Mandate §8 E4 forbids forcing hybrids into early-versus-mixture classes.

**Correction.**
1. Rename the f-based quantity everywhere, in the code docstring, the design and the outcome categories, to **"residual-profile concordance"**. State in the design: "f is derived from causally labelled references, correlates 0.91 with the causal log-ratio, and is not evidence for Claim 2."
2. In the scorer outputs, rename the categories built on f, for example `BOTH` → `CAUSAL+PROFILE`. Keep F3's "functional change" category reserved for an E2-type test.
3. P7 may stay in the frozen set only as a *profile-concordance* prediction under the corrected rule of objection 2. The synthesis must not cite it for Claim 2.
4. Claim 2 must wait for the Stage-6 battery required by mandate §8 E2:
   - reliability changes;
   - controlled agreement and disagreement;
   - counterfactual edits on which the candidate early and mixture combination rules make *different* predictions;
   - off-support cases labelled as such.
   - The decision statistic must be compared with within-class (seed-to-seed) functional variation. Item 3 above is the benchmark it has to beat.
5. Optional cheap improvement: build the centroids from models whose class is *not* defined by the necessity measure. That is impossible here, so the honest course is item 1.

### 4. MANDATORY: hypothesis bands ignore the causal class, which contradicts the design's own definition of "switch" and "none"

**The mismatch.**
- DESIGN §3 defines "switch" as φ ≥ 0.75 *with the I-anchor causal class* and "none" as φ ≤ 0.25 *with the B class*.
- `score()` uses the class only for the per-arm category. Bands test φ alone.
- This is the same kind of error as XVIII's: the declared rule is not the implemented rule.

**Why it matters (log transform, anchors from dev E801).**
- An arm that stays causally EARLY (nec_E at B's value, nec_M = 0.10) has φ = 0.26. It sits outside the nil band [−0.25, 0.25] while still class-pure B.
- Over the whole EARLY-class region, φ ranges from −0.10 to 0.40. A model that is still pure EARLY can therefore reach the CONTRADICTED zone of a nil prediction (> 0.35).
- Conversely, a HYBRID-class arm can just reach φ ≈ 0.75 if nec_M overshoots the I anchor.

**Correction.** Pick one rule and freeze it in both code and design.

Preferred:
- For switch bands ([0.75, 1.25]), a pair counts in-band only if its causal sub-label is SWITCH.
- For nil bands ([−0.25, 0.25]), a pair counts in-band only if its sub-label is NONE.
- CONTRADICTED stays φ-based (outside the band ± δ). In addition, a pair whose class is opposite to the predicted class counts as outside.
- Partial bands stay φ-only. Report each pair's class alongside its value.

Alternative: amend DESIGN §3 and §6 to say that bands are φ-only, and report the class of every in-band pair.

Either way, add planted cases: a class-pure B arm with φ = 0.30, and a HYBRID arm with φ ≥ 0.75.

### 5. MANDATORY: fail-closed checks do not cover training configurations, seeds, intervention definitions or mathematical reference versions (mandate F4)

**What is covered now.** Stream, parent, the hash of `xix_score.py`, the reference hash, step, exactly one record per run, the step-1211 checkpoint hash, branch-source hashes for training arms, and for transplants `params_from_I`, `opt`, `perturb_seed` and the composed and input hashes.

**What is missing.**
- (a) **Training configuration and seeds.** Continuations run with `FX_STOP=1211`. They exit through `stop_done.json` (stop, hist, seconds) and never write `curve.json` (seeds, interv); the dev run directories confirm this.
  - So no record of L/W/D seeds, `FX_INTERV`, `FX_EDGES`, `FX_B`, `FX_FREEZE`, schedule or weight decay exists for any confirmation run.
  - The scorer checks seeds only through run *names*.
- (b) **Intervention definitions.** The step-100 window states are tied to their intervention only by directory name.
  - `logs/conf/branch_src/<run>.json` is written by a one-line command *before* training. It certifies that a file exists at a path, not what `xix_train.py` loaded.
  - `xix_train.py` silently falls back to `X13E`/`X13D` checkpoint directories if the local branch directory is missing.
  - It also resumes from any stale `resume.pkl` and ignores `FX_BRANCH`.
- (c) **Transplant continuation link.** Nothing records which tp directory a transplant continuation actually branched from. A queue typo (for example `FX_BRANCH=tp_E821_fullB` on the BfromI_L0 line) would pass every check.
  - `alpha` and `foreign_donor` in `provenance.json` are not checked. An interpolated (`--alpha 0.5`) composition with the same groups and opt passes as a full transplant.
- (d) **Mathematical reference versions.** Only `src/xix_score.py` is hashed. Several things that change nec, KL and acc leave that hash unchanged:
  - the Bayes posterior (`GL.posterior_closed`, `RP.ENV1`), the edge sets (`MFW.edges("own0"/"own2")`), the forward pass and the probe sets (`MT.sets`), all in XVIII's `xviiim` tree;
  - `measure19.py` itself.
  - The reference cache `refs19_<stream>.npz` is validated only by checkpoint hashes. Profiles computed by older code would be accepted.
  - `score19.py` is not hashed, and `design_sha256` is recorded but never compared with a frozen value.

**Correction.**
1. `xix_train.py` writes `run_meta.json` into every run directory on every exit path, including STOP. It contains:
   - parent, LSEED, WSEED, DSEED, STEPS, every `FX_*` variable;
   - the resolved branch directory, with the sha256 of the `step{k}.npz` and `opt_step{k}.pkl` actually loaded;
   - whether a resume occurred;
   - the sha256 of `xix_train.py`.
   - Remove the X13E/X13D fallback for XIX, or treat it as a fatal error.
2. The scorer checks:
   - for every run, that `run_meta` matches the design (seeds = pair L/W/D; `FX_*` = `design.window[base][arm]`; no resume from a foreign state);
   - for transplant arms, that the loaded branch hashes equal `provenance.outputs`;
   - for training arms, that the loaded hashes equal the step-100 state hashes;
   - for each step-100 window state, that its own `run_meta` shows the declared `FX_INTERV`/`FX_EDGES`/`FX_B`/`FX_FREEZE` and `FX_STOP=100`;
   - that `alpha is None` and `foreign_donor is False` unless the design declares otherwise.
3. `measure19.py` records:
   - a `math_sha256` manifest of `measure19.py`, `xix_score.py`, `xviiim.py` and every XVIII `src` module it imports;
   - the sha256 of the probe-set arrays (natural X, q and PB; profile Xp, qp and PBp).
   - The design freezes these values and the scorer compares them.
   - The reference cache stores and checks `math_sha256`. Any pre-existing `refs19_94700000.npz` is deleted before the freeze.
4. The scorer compares `sha(design)` and `sha(score19.py)` with `FREEZE_XIX.json` and fails closed on a mismatch.
5. The planted tests must exercise the **file-based** checks. All current planted runs use `--no-files`, so none of the checkpoint, branch, transplant or matching provenance paths has ever been tested.
   - Make the `branch_src` root a scorer argument, so a temporary tree can be used.
   - Plant one defect for each check: checkpoint hash, branch-source hash, composed hash, input hash, wrong groups, alpha set, wrong `run_meta` seed, wrong `FX_EDGES`, wrong `math_sha256`.

### 6. MANDATORY: the matching gate fails open

**Code.** `out["matching"].get(pn, {}).get("ok", True)`: a pair with no matching entry keeps its control.

**Effects.**
- With `--no-files`, `matching` is `{}`, so every control is used. Demonstrated: `P6 SUPPORTED n=3` with no matching computed.
- `matching()` does not hash-check the step-100 states it reads.

**Correction.**
- If `D["matching"]` is set and a pair has no matching result, then either fail closed (PROVENANCE_FAIL) or drop the pair as `"matching unavailable"`.
- `matching()` verifies the hashes of the three step-100 states against the branch-source and compose provenance.
- Add a planted case.

### 7. MANDATORY: the scorer trusts derived labels in the measurement records, and the functional sub-verdict has no anchor-level evaluability

**Trusted labels.** `score()` takes `competent` and `functional` from the record. It does not recompute them from KL, acc and f.
- Demonstrated: a record with `competent=True`, KL 0.30, acc 0.50 is scored and SUPPORTED.

**No functional evaluability for anchors.**
- The functional SWITCH test is `a["functional"] == ci`, where ci is the I anchor's *causal* class.
- The I anchor's own *functional* class is never checked.
- In validation, 2 of the 12 references are functionally AMBIGUOUS on all three streams:
  - E-intervened D813: f ≈ +0.01;
  - M-baseline D813: f ≈ −0.03.
- So roughly a third of fresh pairs can be expected to have an anchor whose functional class does not match its causal class. Arms in those pairs would be called BOTH or CAUSAL_ONLY against a reference the anchor itself fails.

**Correction.**
- Recompute `competent = KL ≤ 0.045 and acc ≥ 0.80`, the causal class from `nec_*`, and the functional class from f inside the scorer. Fail closed if they disagree with the record.
- Add per-family *functional evaluability*: anchors' functional classes equal their causal classes, and |f_I − f_B| ≥ 0.20. The dev values are 0.41 (E801) and 0.34 (M801).
- Report functional evaluability separately from causal evaluability. It must never gate the causal bands.

### 8. MANDATORY: competence gating is the remaining XVIII-type evaluability risk, and no frozen variant addresses it

**Origin of the thresholds.** KL ≤ 0.045 and acc ≥ 0.80 are inherited from XVIII. No XIX document justifies them.

**Margins.**
- The criterion is **class-asymmetric**. Competent EARLY models have KL 0.021–0.036; MIXTURE models have 0.010–0.017.
- The anchors closest to failing are therefore the EARLY ones: E fullB and M fullI. Minimum margin is 0.009 nats.
- Other parents' EARLY models reach 0.039–0.050. `uP164_L21` is a pure EARLY model that is non-competent only because its KL is 0.048–0.050.

**Selection bias at the arm level.** Arms that end EARLY are dropped preferentially:
- on M, the predicted switches;
- on E, the predicted *failures* to switch.
- A dropped arm turns into INCONCLUSIVE rather than PARTIAL or CONTRADICTED.

**Demonstrated.** Two of three E fullB anchors at KL 0.046 make every E hypothesis INCONCLUSIVE under the primary *and under all eight frozen variants*. evalB does not help, because it still requires a competent B anchor.

**Correction.**
1. Freeze a competence sensitivity variant applied to evaluability *and* arm dropping: `comp060` (KL ≤ 0.060, acc ≥ 0.78).
2. Freeze a variant `comp_report` in which competence is reported but does not gate anchors whose causal class is pure. These variants never replace the primary.
3. Pre-declare what happens if a base loses evaluability:
   - the base's hypotheses are INCONCLUSIVE in the primary, and the confirmation is reported as failed for that base;
   - variant results are reported, labelled as sensitivity;
   - no post-hoc rescue.
4. When an arm is dropped for non-competence in a strict majority of a base's pairs, the band label is **NONCOMPETENT**, not INCONCLUSIVE. For a control this is a reportable result under mandate D3.
5. Report KL, acc and φ for non-competent arms in every output.
6. Add planted boundary cases, which F2 requires and which are currently absent: KL 0.044 vs 0.046; acc 0.799 vs 0.801; nec_E 0.29 vs 0.31 on an anchor; nec_E 0.14 vs 0.16 on a MIXTURE anchor.

### 9. MANDATORY: the evalB variant can produce NaN or meaningless φ, and NaN maps silently to PARTIAL

**How it happens.**
- Under evalB the I anchor need not be pure, competent or opposite in class.
- If the intervention fails to switch on a pair, φ is normalised against a non-switched anchor. The denominator can be about 0, giving NaN, or have the wrong sign.
- Demonstrated: `evalB with non-switching I anchor: PARTIAL {E1: nan, E2: nan, E3: nan}`.
- NaN fails both `lo ≤ v ≤ hi` and the outside-band test, so it lands silently in PARTIAL.

**Correction.**
- In `score()`, any non-finite v drops the pair with reason `"phi undefined"`.
- Under evalB, require the I anchor to have the opposite *class*. Only its competence and purity are relaxed. Otherwise there is no I reference.
- **Report the block-replication rate as a primary descriptive outcome**: how many fresh pairs have a pure, opposite fullI. A failure of the block to switch is a scientific result, not just a non-evaluable pair.

### 10. ADVISORY: φ_nec with log(x+0.01) is numerically stable, but its scale is non-linear and route-asymmetric

**Stability.**
- For evaluable pairs the denominator cannot vanish. One route is ≥ 0.30 in one anchor and ≤ 0.15 in the other.
- Variation between streams is ≤ 0.008.
- The eps variants move φ by ≤ 0.04 on the dev and c18 arms.

**Interpretation.**
- In each route, component 0.5 is the geometric mean of the anchors. For E801's E route this is nec ≈ 0.10, i.e. 9% of the I value.
- A component of 0.75 needs a *rising* route to reach about 30% of the I anchor, but a *falling* route to drop to about 2% of the B anchor.
- The two components and their linear counterparts disagree strongly on the hybrid arms:

| arm | log components (E, M) | linear components (E, M) |
|---|---|---|
| E801 vsg | 0.11, 0.58 | 0.41, 0.16 |
| M801 vsg | 0.63, 0.25 | 0.24, 0.60 |
| M801 BfromI_L0 | 0.42, 0.07 | 0.09, 0.23 |

- The scalar averages happen to be similar, so the "hybrid" reading of P4_vsg_partial is robust as a scalar. Which route carries the partial effect, however, depends on the transform.
- Leverage near zero also differs by base, because the denominators differ:
  - E pair: E route 4.64, M route 4.15;
  - M pair: E route 3.72, M route 3.50.
- The B anchors' near-zero values (0.003 on E, 0.021 on M) set the scale of each nil band.

**Correction.**
- Freeze `phi_lin`, a linear normalisation of each route, as an additional sensitivity variant.
- Report both route components, log and linear, for every arm and pair.
- Phrase P4_vsg_partial results as "log-necessity partial". Make no claim about which route carries the partial effect without the components.

### 11. ADVISORY: band rules and power with three pairs per base

The rules are SUPPORTED = all in band; CONTRADICTED = strict majority outside band ± 0.10; otherwise PARTIAL.

**Simulation (`/tmp/revC/power.py`, normal between-pair spread).**
- **Realistic spread.** The XVIII c18 gsg values spread over 0.62–0.85 on E, so a between-pair SD of about 0.12 is realistic.
- **Switch prediction, true mean 1.0, SD 0.12.** SUPPORTED 0.89 with n = 3, and 0.80 when pooled to n = 6. At true mean 0.85, SUPPORTED is 0.51 (n = 3) and 0.26 (n = 6).
- **Half effect, mean 0.60, against a switch band.** CONTRADICTED 0.73 (n = 3) and 0.44 (n = 2).
- **P4_vsg_partial.** At the dev mean of about 0.36 (E sits near the 0.25 edge), P(SUPPORTED) is about 0.55 with n = 3 and 0.31 pooled to n = 6.
- So pooling (objection 1) also lowers power.
- **Nil band, true mean 0.30.** PARTIAL about 0.7 and CONTRADICTED only about 0.27. A "half-way" truth near a nil band is usually not refuted.
- **n = 2 after one dropped pair.** CONTRADICTED needs both pairs outside, and SUPPORTED needs both inside.

**Correction.**
- Keep the rules, which are conservative in the right direction. But declare in the design that PARTIAL is the expected outcome for any true effect within about 0.1 of a band edge, and that PARTIAL is not weak support.
- For predictions centred near a band edge, either justify the band from dev data plus spread, or accept the stated power explicitly. This applies to P4_vsg_partial on E (0.35–0.38 against lower edge 0.25) and to P2_M_L0_necessary.
- On P2_M_L0_necessary: its upper edge of 0.50 calls a pair "necessary" while half the switch remains. Rename it "L0 restoration removes at least half of the switch", or tighten the band.

### 12. MANDATORY: statistical uncertainty reporting is unspecified (mandate §8 E4, §13 "statistical uncertainty")

**Correction.** Freeze the following outputs and attach them to every hypothesis.

1. **Per pair and arm:**
   - φ, both log route components, φ_lin components;
   - raw Δnec_E and Δnec_M versus the B anchor, in nats;
   - KL, acc, f and category.
2. **Measurement uncertainty per pair:**
   - an episode bootstrap of φ (1000 resamples of the 1500 natural episodes, using the same resample for arm and anchors), reporting the 95% interval;
   - *or* re-measurement of anchors and arms on two further frozen streams (measurement only, which is cheap).
   - Validation shows this floor is about 0.01. Reporting it establishes that differences between pairs are not measurement noise.
3. **Between-pair, per base:**
   - all n values, the median and the range;
   - mean ± t₀.₉₇₅,ₙ₋₁·SD/√n. Label it as descriptive. With n = 3, t = 4.30, so the interval is wide, and showing that is the point.
   - No p-values, and no pooled-across-base intervals.
4. **Generalisation bound.**
   - For each verdict, state the exact one-sided 95% lower bound on the proportion of matched pairs (from this parent) that behave as predicted.
   - Values: 3/3 → 0.37; 2/2 → 0.22; 6/6 → 0.61; 2/3 → 0.14.
   - State the single-parent limitation next to it.
5. **Paired contrasts where the mandate asks for "quantitatively distinguishable" results.**
   - P3: per-pair φ(paramI_optB) − 1 and φ(paramB_optI) − 0, compared with the measurement floor.
   - P5: per-pair gsg − frz.
   - Freeze "distinguishable" as |Δ| > max(0.10, 3 × measurement SD) in a strict majority of pairs.
6. **Dependence declaration.**
   - P1_E_L0 and P7_E_L0 share runs, and so do P4_vrs and P7_vrs.
   - P5_frz_nil and P5_gsg_exceeds_frz share frz.
   - The synthesis must not count the shared observations as independent confirmations (§11).

### 13. ADVISORY: sensitivity variants are internally consistent but partly vacuous or incomplete

**Vacuous.**
- `phicut70` and `phicut80` change only per-arm categories. Bands use each hypothesis's own lo/hi, so they cannot change any hypothesis verdict; the planted output shows identical bands.
- Either state this in the docstring, or make them operate on the band edges of switch and nil hypotheses (0.70/0.30 and 0.80/0.20).

**Incomplete.**
- `thr_pm05` shifts both thresholds in the same direction: (+t, +t) and (−t, −t). In each case one inequality becomes stricter and the other looser. Purity under "both stricter" is never tested.
- Use the four sign combinations, or (+t on the dominant route, −t on the other) as the stringent case.

**Missing variants.** Per-base scoring (if objection 1 is not adopted as primary), `phi_lin` (objection 10), and competence (objection 8).

**Planted tests.** Each variant should have a planted case showing that it changes what it claims to change. The current tests show only that each variant runs, and that evalB alone flips P2.

### 14. ADVISORY: outcome categories (F3)

**Gaps.**
- A causal PARTIAL is always labelled HYBRID, which discards its functional sub-label. Keep `HYBRID(functional_X)`.
- The functional classifier has no "hybrid" class, only AMBIGUOUS within ±0.05. Rename it "AMBIGUOUS/HYBRID" and stop calling a model like `uP164_L20` "functional MIXTURE" in prose (see objection 3, item 5).

**Required counts.** Report, per base and arm, the counts of every category, including NONCOMPETENT and non-evaluable pairs. Every frozen output then shows the excluded cases (F3: "do not suppress ambiguous cases").

### 15. ADVISORY: machinery check of the anchors

**Gap.** L1 relies on Φ being deterministic: fullI ≡ the intervention run and fullB ≡ the baseline run. The confirmation queue does not include uninterrupted step-1211 B and I runs for any fresh pair, so this is never re-verified on confirmation units.

**Correction.** Run one uninterrupted B and one uninterrupted I continuation for one fresh pair per base, about 50 minutes in total. Require a bitwise checkpoint match with fullB and fullI, or at minimum identical measurement records. Declare the check as implementation validation.

---

## 2. Answers to the specific challenges

| Question | Answer |
|---|---|
| Competence criteria | Inherited and unjustified. Class-asymmetric: EARLY anchors are 0.009 nats from failing. This is the most likely route to XVIII-type non-evaluability, and no variant covers it (obj. 8). |
| Class thresholds and margins | Adequate for uP160 anchors: EARLY nec_M ≤ 0.012 against 0.10; MIXTURE nec_E ≤ 0.037 against 0.15; dominant routes ≥ 0.58 against 0.30. The validation summary's minimum margin of 0.006 belongs to non-competent models. Dev arms do sit near the boundaries (M801 BfromI_L0 has nec_E 0.139), which matters for categories only (obj. 4, 8). |
| Functional score independence and Claim 2 | Not independent: references are labelled by causal class, and r = 0.91 with the causal log-ratio. Not Claim 2 evidence: between-class functional distance is about 1.15× within-class distance (obj. 3). |
| φ_nec near zero and across M/E anchors | Numerically stable (the denominator cannot vanish for evaluable pairs; stream noise ≤ 0.008), but non-linear and route-asymmetric. The nil-band tolerance in nats depends on the anchors' near-zero values and differs by base (obj. 10, 4). |
| Band rules and power | Conservative. PARTIAL is likely for any effect near a band edge. Pooling "both" lowers power and breaks the per-base rule (obj. 1, 11). |
| Expressions vs arm names | All 17 are correct. The detection is fragile: an expression written with double quotes skips the competence and matching checks (demonstrated). Validate expressions at load time and fail closed on any unknown or undetected arm name. |
| P7 | Cannot be CONTRADICTED, and can be SUPPORTED without a causal switch. Replace it with the categorical rule in obj. 2. |
| Can evaluability fail as in XVIII? | Unlikely through causal purity, plausible through KL competence or through a block that fails to switch. The result would be INCONCLUSIVE under every frozen variant (obj. 8, 9). |
| Sensitivity variants and evaluability | Consistent, because evaluability is recomputed inside `score()` under every variant. But phicut is vacuous, thr_pm05 is incomplete, evalB can produce NaN, and competence and matching are never varied (obj. 9, 13). |
| F4 coverage | Missing: configurations, seeds, intervention definitions, the transplant continuation link, alpha/foreign_donor, math/reference code versions, the reference cache's code version, and the design/scorer freeze hashes. No file-based check has ever been tested (obj. 5). |
| Uncertainty with n = 3 | Per-pair values plus a measurement floor; per-base range and a descriptive t-interval; exact lower bounds on the in-band proportion (0.37 for 3/3); paired contrasts for P3 and P5; no pooling and no p-values (obj. 12). |

**Additional ADVISORY item on expression robustness.**
- At design load, parse each `expr` with `ast`. Collect every subscript of `phi`, `phif`, `f` and `func`.
- Fail closed if any name is not an arm of the family, or if the set differs from `used`.
- Allow only those names and `min`, `max` and `abs`.

---

## 3. Conditions for approval

Before `FREEZE_XIX.json` is written, the following must be implemented in code, reflected in `DESIGN_XIX.md`, and covered by new planted cases that pass (including file-based provenance tests):
- objections 1, 2, 3 (renaming and the Claim-2 restriction), 4, 5, 6, 7, 8, 9 and 12;
- the expression-validation item in §2.

The confirmation runs themselves need no change, except the `run_meta.json` instrumentation in `xix_train.py` (obj. 5) and the optional machinery runs (obj. 15).

**Tied to the Claim-2 restriction.** Until the Stage-6 E2 battery exists, no result of this confirmation may be reported as a change of functional computation. The strongest permitted statement is "causal-route change with concordant residual profile".

CLASSIFICATION: Approved with mandatory corrections

---

## Round 2

**Reviewer.** C. This is an AI agent review, not human peer review.

**Date.** 2026-11-19 (UTC).

**What I read.**
- `DESIGN_XIX.md` v2 and `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.
- `scripts/score19.py` v2, `scripts/measure19.py`, `scripts/stats19.py` and `scripts/scorer_planted19.py`.
- The `run_config` code path in `scripts/xix_train.py`, and `logs/conf/queue_c1.txt` / `queue_c2.txt`.
- `logs/conf_spec19.json`, `logs/conf/design19.json`, `logs/dev/dev_bandcheck19.json`, `logs/dev/matching_gate_test19.json` and `logs/dev/m19_dev*.jsonl`.

**What I ran.**
- `scripts/scorer_planted19.py`: ALL PASS, 69 checks. The output was rewritten with the same check list. The temporary `checkpoints/planted19_world` it creates was removed by the script itself.
- New adversarial cases against v2 (`/tmp/revC/adv2.py`).
- `stats19.py` on a demonstration score.
- Cheap distance and KL checks on dev step-100 states and on the validation log.

**Hashes.** The frozen hashes in `design19.json` match the current files: `xix_score.py`, `measure19.py`, `score19.py`, `compose19.py`, `xix_train.py`, `stats19.py` and `xviiim.py`.

I modified no existing file and ran no training.

### R2.0 Status of the round-1 mandatory items

| # | item | status | evidence |
|---|---|---|---|
| 1 | pooled "both" hypotheses | **Resolved** | `base` must be E or M, otherwise PROVENANCE_FAIL (planted). All 32 hypotheses are per base. Evaluable-pair counts are reported per base. |
| 2 | P7 unfalsifiable | **Partly resolved** | A category hypothesis with functional evaluability exists and can be CONTRADICTED. But `contradict` includes NEITHER (R2-1). |
| 3 | functional score not Claim-2 evidence | **Resolved** | Renamed RPC. DESIGN §1 and §7 forbid any Claim-2 reading. Residue: the `xix_score.py` docstring still says "functional score", and records still carry `functional=` (advisory R2-A1). |
| 4 | bands ignore class | **Resolved in code** | `need_class` on every switch and nil band; a class failure counts as outside. One band's class condition is dev-fragile and conceptually wrong (R2-2). |
| 5 | F4 fail-closed coverage | **Largely resolved** | `run_config` covers argv seeds, the FX_* environment, no fallback, branch directory and branch hashes, and the harness hash. Measurement dependency hashes include the Bayes posterior modules. The reference cache is keyed by code hash. Alpha and the per-pair perturbation seed are checked. fullB is checked bit-identical. Residues: R2-3 and R2-4. |
| 6 | matching gate fails open | **Resolved for missing results** | A missing result or state gives ok = False and the pair is dropped. Two new holes: a gate *exception* is indistinguishable from a genuine gate failure (R2-3), and the kl100 ratio has no denominator guard (R2-5). |
| 7 | trusted record labels | **Resolved** | Labels are recomputed; planted "record labels ignored". |
| 8 | competence / XVIII-type loss | **Partly resolved** | The primary threshold is now 0.06, with comp_strict and comp_loose variants and planted boundary cases. The stated justification is factually wrong, and the NONCOMPETENT band label is still missing (R2-6). |
| 9 | NaN → silent PARTIAL | **Resolved** | NaN drops the pair with a reason (planted). |
| 12 | uncertainty reporting | **Partly resolved** | `stats19.py` gives per-pair values, range, mean, a descriptive t-interval and the Clopper–Pearson bound. The route components promised in its own docstring are not implemented, and there is no measurement floor (R2-7). |

### R2.1 Judgement on the KL ≤ 0.06 primary competence threshold

**Factual check of the justification.** `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md` C8 says "competent uP160 models ≤ 0.042; degraded parents ≥ 0.063". `logs/scorer_validation19.json` contradicts the second half:
- **uP160** models reach at most 0.036. The 0.042 figure belongs to uP164_L25 and uP165_L23.
- **Two other-parent models lie inside the claimed gap:**
  - `uP162_L22`: KL 0.047–0.057, causally HYBRID, f ≈ 0.01–0.02;
  - `uP164_L21`: KL 0.048–0.050, pure EARLY.
  - Both become competent under 0.06 and were non-competent under 0.045.
- So 0.06 is not placed in an empty gap between competent and degraded models. It lies 0.003 above the worst stream of `uP162_L22` and 0.003 below `uP162_L24` (0.063).

**Is this acceptable preflight calibration?**

Yes, on three conditions.

Why it is acceptable:
1. The threshold was chosen before any confirmation-stream (94 700 000) or fresh-pair data existed.
2. The confirmation units are uP160 only. On every validation stream their anchors sit at ≤ 0.036, which is ≥ 0.024 below 0.06. The threshold therefore does not decide confirmation evaluability, which was the purpose.
3. The 0.045 rule is kept as the frozen `comp_strict` variant.

But it was calibrated on the three streams that were meant to be the independent validation streams (F2). It has therefore not been validated on a held-out stream.

**Conditions (MANDATORY, documentation only):**
- Correct the C8 sentence. State: "0.06 was set after inspecting the validation-stream KL distribution (97.96M / 98.96M / 99.96M); two non-uP160 models (uP162_L22 HYBRID, uP164_L21 EARLY) are competent under 0.06 and not under 0.045".
- Say that the threshold's effect on uP160 confirmation evaluability is nil by margin.
- Report comp_strict alongside the primary in the synthesis.

**ADVISORY.** Re-measure the 60 validation models on one more unused stream (measurement only). This would show that no uP160 model approaches 0.06 there.

### R2.2 Remaining objections

**R2-1. MANDATORY: category (P7) hypotheses count NEITHER as a contradiction, which conflates a failed causal switch with RPC discordance**

*The rule.* `contradict: ["CAUSAL_ONLY", "NEITHER"]`. NEITHER means "causal none and RPC none", which is a *concordant* outcome.

*Demonstrated.* An arm that stays causally B on every pair, with a concordant RPC profile, gives `P7 CONTRADICTED`.

*Consequences.*
- E_P7_L0, E_P7_vrs and M_P7_vsgo would then be contradicted by exactly the same observation that contradicts E_P1, E_P4_vrs and M_P4_vsgo.
- That is one causal observation counted twice (§11), and it is labelled a concordance failure.

*Correction.*
- In `score()`, for `kind == "category"`, drop pairs whose arm causal sub-label is not SWITCH, with the reason `"no causal switch"`.
- Set `contradict` to `["CAUSAL_ONLY"]` for all three P7 hypotheses.
- Add the planted case "all NEITHER → INCONCLUSIVE".

*Dev disclosure.* E_P7_vsgo_concordant (dev: INCONCLUSIVE(causal_SWITCH), vsgo f = −0.013) appears in `dev_bandcheck19.json`. It was replaced in the frozen set by E_P7_vrs_concordant, whose dev category is BOTH. The DESIGN table lists E_P7_vrs's dev value as "none (—)" and labels it a replication. Correct the row: dev value BOTH (E801 vrs f −0.12). Also record that the vsgo E variant was dropped after its dev RPC came out ambiguous.

**R2-2. MANDATORY: `need_class: "B"` on M_P1_L0_alone_insufficient is dev-fragile and tests the wrong thing**

*Dev margin.* The dev arm (M801 BfromI_L0, measure19) has nec_E = 0.139 and nec_M = 0.480. It is MIXTURE only because nec_E ≤ 0.15, a margin of 0.011.

*Consequence on fresh pairs.* A fresh pair with nec_E 0.16 and φ ≈ 0.25, i.e. deep inside the band and clearly insufficient, becomes HYBRID. Under the class condition it counts as *outside*. Two such pairs CONTRADICT "L0 alone is insufficient" with data that support it.

*Why the condition is wrong.* Insufficiency is "not a switch". It does not mean "still pure B". The band's upper edge of 0.5 already encodes this.

*Correction.*
- Set `need_class: null` on M_P1_L0_alone_insufficient. Optionally add the condition "class ≠ I-anchor's class".
- Treat M_P2_L0L2_restore_prevents the same way. Its dev nec_E is 0.062, so the class margin is 0.088 and acceptable, but the logic of "prevents" is "not I"; keep `"B"` only if that is the intended claim.
- Add a planted case: a HYBRID arm with φ 0.25 on a [−0.25, 0.5] band must not count as outside.

**R2-3. MANDATORY: residual fail-open and fail-silent paths**

1. *Arm detection.*
   - The arm-validation regex `\['([A-Za-z0-9_+]+)'\]` and the `used` detection both recognise only single-quoted subscripts.
   - Demonstrated: `phi["x"]` with the arm non-competent passes provenance and is SUPPORTED. With `need_class: "I"` and an arm of B class it is also SUPPORTED, because no competence, matching or class check applies when `used` is empty.
   - The frozen expressions are all single-quoted, so this is latent. It is still a fail-open path in a frozen scorer.
   - *Correction:* parse `expr` with `ast`. Collect every `Subscript` of `phi`, `phif`, `f` and `func` with a constant key. Fail closed if any key is not a defined arm on the base, if any other name or call appears, or if the set is empty for a band hypothesis. Derive `used` from that set.
2. *Gate exceptions.*
   - In `matching()`, any exception (for example a crash while loading xviiim for kl100) gives `ok = False`. The pair is then dropped as "control not matched", and per DESIGN §4 the X8 form becomes "unresolved".
   - A code error is thus reported as a scientific non-match.
   - *Correction:* any gate exception other than a declared missing state is a PROVENANCE_FAIL for the whole score.
3. *`--no-files` output.*
   - The output still says `verdict: "SCORED"`.
   - *Correction:* write `verdict: "UNVERIFIED_NO_FILES"` and have `stats19.py` refuse it, so that a frozen report cannot be produced without file provenance (F4: "refuse to issue a frozen verdict").

**R2-4. MANDATORY: the file-mode planted world does not exercise several of the paths the confirmation will use**

*What is covered.* The 19 defects run through a hand-built world, with hand-written `run_config` and `provenance.json`, containing only fullB, fullI and vsg on E.

*Format compatibility.* I checked real artifacts: the harness-written `run_config.json` in dev directories, for example `..._D801_pnzs1T100_stop1211_dev`, has the keys and argv layout the scorer expects, and `compose19.py` writes `alpha` and `perturb_seed`. So format compatibility looks right.

*Paths never executed by any planted case:*
- the per-pair perturbation seed check (randnorm arms);
- the `FX_PNZ_REF` check;
- the kl100 gate inside the scorer;
- a training-arm step-100 source changed after the run started;
- a transplant `branch_src` that differs from `provenance.outputs` while the composed file is intact;
- a composed-from input mismatch;
- a fullB moment/rng mismatch and a fullI parameter mismatch;
- a package `dep_sha256` file mismatch, which is the scorer hashing itself;
- window-state argv or harness mismatch;
- a continuation with `FX_INTERV` set;
- M-base pairs (the world is E only).

*Correction.* Add one planted defect per path to the file world. Include an M pair, a pnz arm and a randnorm arm whose provenance is produced by running the real `compose19.py` on the world's fake states, which is not training.

*Recommended.* Before the freeze, run `provenance()` in files mode against a design restricted to the real D801 dev artifacts that carry `run_config` (vsgo, pnz, frzp). Expected failures should be limited to the known dev differences: harness hash `025d2d…` versus the frozen `67666a…`, and the stream. That confirms the real-artifact path end to end.

**R2-5. MANDATORY: the kl100 gate needs a denominator guard**

*The problem.* The ratio is (KL_C − KL_B)/(KL_I − KL_B). DESIGN §4 notes that on M the block's step-100 excess is only 0.012. On a fresh M pair, KL_I − KL_B can be about 0 or negative:
- a near-zero denominator gives an arbitrary ratio;
- a negative denominator with a below-baseline control gives a large positive ratio, which passes `lo = 0.5` with `hi = 1e9`.

*Correction.*
- Freeze a minimum block excess, for example KL_I − KL_B ≥ 0.005. Below it the gate is "undefined" and fails, with that reason.
- Record KL_B, KL_I and KL_C in the matching output.
- Add a planted case.

**R2-6. MANDATORY: competence documentation and the NONCOMPETENT label (from round-1 objection 8, items 4 and 3)**

1. Correct the C8 justification as stated in R2.1.
2. When a strict majority of a base's pairs are dropped because the arm is non-competent, the hypothesis label must be **NONCOMPETENT**, not INCONCLUSIVE.
   - Demonstrated: an arm non-competent on all three pairs gives INCONCLUSIVE.
   - This matters for M_A7_normmatched_L0L2, a new arm with no dev run. A random L0+L2 displacement of I−B norm may well destroy competence. That is a reportable control result (mandate D3), not an inconclusive one.
3. Pre-declare in DESIGN §6 that a base losing evaluability is reported as a failed confirmation for that base, with variant results labelled as sensitivity. I could not find this sentence in v2.
4. `dev_bandcheck19.json` shows `M801|randnorm_L0L2` as NONCOMPETENT with φ NaN and f 0. There is no such dev run, so this is a placeholder. Label it "missing".

**R2-7. MANDATORY: `stats19.py` does not implement its own specification**

*What is missing.*
- Its docstring promises "for phi of a single arm, its E- and M-route components". These are not computed; demonstrated on a demo score.
- No measurement floor is reported.

*Correction.*
- Compute the route components from the score's `nec_E`/`nec_M` and the anchors, for both the log and the linear transform.
- Report a frozen measurement floor. Either use the stream-to-stream φ range from validation (≤ 0.008 for the c18 gsg arms, which is acceptable if stated as such), or re-measure the anchors on two extra streams.
- Print the declared families (P3 E/M, M partner bands, P5 frz/frzp) and the shared-run dependencies (P7 with P1/P4) next to the verdicts. The synthesis cannot then count them as independent.

### R2.3 Power and dev-fragile band edges (ADVISORY)

These dev values come from `dev_bandcheck19.json`, measure19 on 86 200 000. Probabilities use a between-pair SD of about 0.12 (the XVIII c18 gsg spread), assuming the dev value is the population mean.

| hypothesis | dev | nearest edge | margin | P(all 3 in band) | likely outcome if dev is typical |
|---|---|---|---|---|---|
| E_P4_vsgo_switch | 0.825, MIXTURE | 0.75 | 0.075 | ≈ 0.46 | PARTIAL ≈ 0.5; CONTRADICTED ≈ 0.04 |
| E_P4_vsg_partial | 0.346 | 0.25 | 0.096 | ≈ 0.49 | PARTIAL ≈ 0.5 |
| E_P7_L0_concordant | arm f −0.096 | −0.05 (RPC class edge) | 0.046 | — | AMBIGUOUS arm f → PARTIAL plausible |
| P7 (all) | — | functional evaluability of anchors | 2 of 12 references are RPC-ambiguous | P(≥ 2 of 3 functionally evaluable) ≈ 0.77 | INCONCLUSIVE ≈ 0.23 |
| M_P1_L0_alone_insufficient | 0.245, class margin 0.011 | class edge | — | — | see R2-2 |
| others | — | — | ≥ 0.12 | ≥ 0.7 | — |

**Correction.**
- State these expected outcomes in DESIGN §5 before the freeze, so that a PARTIAL on E_P4_vsgo or E_P4_vsg cannot be presented later as weak support or as a surprise.
- Make no band edge changes after the freeze. If the authors want E_P4_vsgo to have useful power, the edge must be justified now. For example, use 0.70 with the class condition, since the class condition already carries the "switch" meaning. Moving the edge after seeing the confirmation is excluded.

### R2.4 Other advisory items

- **R2-A1.** `src/xix_score.py` still names `functional_score`/`functional_class` and has `competent` at 0.045. As a result, `measure_c19.jsonl` records carry `competent` and `functional` labels that disagree with the scorer's primary rule.
  - The scorer ignores them, so this is harmless for verdicts but misleading in raw data.
  - Changing `xix_score.py` changes the frozen hash, so do it before the freeze or document it in `REPRODUCTION.md`.
- **R2-A2.** The continuation `run_config` records maskfwd and native hashes, but the scorer checks only `xix_train.py`. Compare `code` against a frozen training-dependency set.
- **R2-A3.** fullI is checked bit-identical in parameters only. Its moments and rng should match native I too (`opt follow`, all groups), so add the check.
- **R2-A4.** `design_sha256` is recorded but not compared with `FREEZE_XIX.json`. Add the comparison once the freeze file exists.
- **R2-A5.** The dev runs were produced with harness `025d2d…`, but the frozen harness is `67666a…`. One bit-identity check would show that the harness changes do not alter training semantics: a short branch of one dev state with the frozen harness, compared with the dev checkpoint at the first saved step. The scorer's fullB check at step 100 does not cover this.
- **R2-A6.** `xix_train.py` writes `run_config.json` only if none exists. A relaunch into a directory left by an earlier failed attempt keeps the old config. Write a new `run_config_<start>.json` per launch, and have the scorer require all of them to agree.

### R2.5 Conclusion

The round-1 structural problems are fixed:
- base pooling;
- the unfalsifiable P7 construction;
- trusted labels;
- NaN handling;
- the missing-gate fail-open;
- most of the F4 gaps;
- the Claim-2 over-reach.

What remains are small, local code and documentation corrections:
- R2-1: P7 contradiction set;
- R2-2: one class condition;
- R2-3: expression parsing, gate exceptions and the no-files verdict;
- R2-4: planted coverage of the untested provenance paths;
- R2-5: the kl100 denominator;
- R2-6: competence documentation and the NONCOMPETENT label;
- R2-7: `stats19.py` route components and measurement floor.

None of them requires new training or changes the arms. They must be implemented and covered by passing planted cases before `FREEZE_XIX.json` is written.

CLASSIFICATION: Approved with mandatory corrections

---

## Round 3 (verification of R2-1 to R2-7 only)

**Reviewer.** C. This is an AI agent review, not human peer review.

**Date.** 2026-11-19 (UTC).

**What I checked.**
- The C rows of the Round 2 table in `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.
- `DESIGN_XIX.md` v2.1, sections 4–6.
- `scripts/score19.py`, `scripts/stats19.py` and `scripts/scorer_planted19.py`.
- `logs/conf_spec19.json` and `logs/conf/design19.json`. Their hypothesis lists are identical.
- The frozen hashes of `score19.py` and `stats19.py` in `design19.json`. Both match the current files.

**What I ran.**
- `python3 scripts/scorer_planted19.py`: exit 0, **ALL PASS, 77 checks**. The check list is the same as in the shipped log.
- New adversarial cases against the current scorer: `/tmp/revC/adv3.py`.

I modified no existing file and ran no training.

| item | verdict | evidence |
|---|---|---|
| **R2-1** P7 counts NEITHER as a contradiction | **Resolved** | All three P7 hypotheses now have `contradict: ["CAUSAL_ONLY"]` and `eligible: [BOTH, CAUSAL_ONLY, INCONCLUSIVE(causal_SWITCH)]`, and are labelled "eliminates nothing". Adversarial: NEITHER on all pairs gives INCONCLUSIVE ("not eligible (NEITHER)"); CAUSAL_ONLY on all pairs gives CONTRADICTED. The vsgo→vrs swap is disclosed. |
| **R2-2** class condition on M_P1_L0_alone_insufficient | **Resolved** | `need_class: null` in both the spec and the design. |
| **R2-3** silent paths | **Resolved** | `used` detects both quote styles. Adversarial: `phi["x"]` with a non-competent arm gives NONCOMPETENT, and with a wrong-class arm gives CONTRADICTED; neither is SUPPORTED any more. A gate exception drops the pair with the distinct reason "control gate error (fail closed)", so it can no longer pass as a scientific non-match. No-files mode prints `SCORED_NO_FILES (not a frozen verdict)`. |
| **R2-4** planted coverage of the confirmation paths | **Resolved in substance** | The file world now has M pairs, a perturbation-seed transplant (the 19002 vs 19502 defect is caught), a pnz arm with reference states (a foreign reference is caught), and an M window state trained on the E target. That makes 22 file defects, each failing for its own reason. The kl100 gate runs on real E801 step-100 states: ratio 0.88 passes; block excess 0 → not applicable but usable; a control without excess fails. |
| **R2-5** kl100 denominator | **Resolved** | A control excess ≤ 0 fails. A block excess below `min_block_excess` 0.005 (frozen in the design) switches to the predeclared absolute rule, a control excess of at least 0.005. A negative denominator can no longer give a spurious pass. Tested on real states. DESIGN §4 states the rule. |
| **R2-6** competence text and the NONCOMPETENT label | **Resolved** | DESIGN §6 says the 0.06 threshold was set after inspecting the validation and dev distributions, before any confirmation data, and that the cut is not a clean gap; comp_strict (0.045) is kept. A band is NONCOMPETENT when more than half of its base's pairs drop for arm competence. Adversarial: 2/3 and 3/3 non-competent both give NONCOMPETENT. |
| **R2-7** `stats19.py` | **Resolved** | Per-pair E/M route components are computed for single-arm φ hypotheses. A frozen measurement floor is stated: stream φ shift ≤ 0.008, and differences under about 0.05 are not interpretable. Families and dependencies are declared in DESIGN §5 and in the P7 labels. |

**Residual advisory items (not blocking).**
- **Planted coverage.**
  - These fail-closed branches are still not exercised: a training-arm step-100 source changed after the run started; a transplant `branch_src` that differs from intact composed outputs; a composed-from input mismatch; a fullB moment/rng mismatch; a fullI parameter mismatch; a package `dep_sha256` file mismatch; window-state argv or harness mismatch; a continuation with `FX_INTERV` set.
  - I read each of them. They are direct equality checks that fail closed, so this is a test-coverage gap, not a defect.
- **Arm-name validation in `provenance()`.**
  - The regex still matches only single-quoted names.
  - A double-quoted *unknown* arm would make `eval` crash instead of producing a PROVENANCE_FAIL. A crash still issues no verdict.
  - The frozen expressions are all single-quoted.
- **`stats19.py` and the verdict.**
  - `stats19.py` does not carry the score's `verdict` into its output and does not refuse a `SCORED_NO_FILES` input. Propagate the verdict and refuse non-frozen inputs, or fix the frozen command lines in the FREEZE record.
  - Linear route components come from the `lin` variant run, not from `stats19.py`; state this in REPRODUCTION.
- **Round-2 advisories R2-A1 to R2-A6** stand as advisories.

**Conclusion.** Every round-2 mandatory item from Reviewer C (R2-1 to R2-7) is resolved, and the round-1 mandatory items were resolved or superseded in rounds 2–3. From the measurement and scoring side, the scorer, design and preflight are fit to freeze.

CLASSIFICATION: Approved
