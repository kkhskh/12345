# Review B — Statistics and Causal Inference

**Manuscript:** "Causal Information Outside a Frozen Board Probe in Othello-GPT" (`manuscript/paper_master.md`)
**Reviewer focus:** bootstrap inference, clustering, multiplicity, preregistration, sample reuse, estimands, intervention semantics.
**Method:** I read the manuscript, `scripts/p1lib.py`, the analysis outputs, all protocols in `data/protocols/`, the raw-study preregistrations, and the archival console excerpts. I recomputed the headline quantities from `data/raw/` with my own script, which does not import `p1lib`. I also used read-only `git log`/`git show` on the parent repository to date the protocol files. I did not open any `*confirmation_protocol*` file or `../_superseded/`.

## Summary

The recomputable core is numerically sound. Every headline number I recomputed from the raw logits and legal sets matches the manuscript, including means, win counts and interval endpoints up to Monte Carlo error. The arithmetic of the stage share is also correct. The problems are inferential, not computational.

1. **The matched-JS floor.** The headline matched-divergence result passes a 0.02 floor. That floor comes from a "restoration" protocol committed after the preregistered COST run had returned INCONCLUSIVE on the same 48 pairs. The COST protocol committed before outcomes set a 0.05 floor, and the observed effect (+0.0266) does not meet it. The manuscript's statement that "effect floors were preregistered before outcomes" is therefore not accurate for this study.
2. **The 56% stage share uses the wrong baseline.** The share is measured against Editor A, but a random JS-matched nullspace direction already captures about 42% of the learned map's gain over Editor A. The wrong-target map direction captures about 63%. "Gain over Editor A" is largely a generic effect of perturbing at fixed divergence. It is not evidence of a stage correction specifically.
3. **The subgroups are nested, not independent.** Category B is a subset of the exact-decode subgroup in both matched-JS and NST. The four-subgroup narrative therefore counts one split twice, and some subgroup contrasts are read without testing the difference between subgroups.
4. **Clustering on recipient game does nothing.** Every recipient game occurs once, so the "clustered" bootstrap is an ordinary pair bootstrap. The real cross-pair dependence runs through reused wrong-donor games. It does not change the intervals materially, but the paper should say so.
5. **Preregistered arms are missing.** The SEM crossover arm, the FRESH Editor-A-anchored arms and anchor-dependence gate, VNF's main effects and controls, the secondary JS targets, and SPEC's full-donor and rowspace cells were run or prespecified but are not reported. Some of these bear directly on how the negative results should be read.

## Recommendation

- **Workshop:** accept after required revisions. The data and package support a small, honest, reproducible finding. Finding 1 (disclosing the floor history and reclassifying the matched-JS gate as post hoc) and Finding 2 (reframing the 56% claim) are required before acceptance. The other findings are desirable.
- **Main track:** reject in current form; resubmission encouraged. The positive results all come from sequentially designed studies on reused validation pairs, and one primary floor was set after the outcome. The only fresh-cohort studies are negative and archival-only. A main-track version needs a preregistered rerun of the matched-JS and NST designs on the sealed confirmation split (or another fresh cohort), with protocols frozen and hashed before execution. It also needs recovered or regenerated arrays for SEM, FRESH and VNF, so that their intervals and the stage share can be recomputed with uncertainty.

## Findings

### 1. BLOCKING — The matched-JS effect floor was set after the preregistered version of the study had returned its outcome

**Location:**
- Experimental setup › Statistics: "Coverage gates and effect floors were preregistered before outcomes."
- Results › Target identity still matters…: "An earlier execution of this design did not follow that protocol; a forensic audit recorded the deviation…"
- Appendix A › Preregistered gates: "The matched-divergence study required … a mean correct-minus-wrong contrast of at least 0.02…"

**Problem.** `data/protocols/08r2_fiber_cost_preregistration.md` is the COST protocol committed before any COST outcome (commit `63433f2`, 2026-10-18 20:05). It specifies matching each control to the correct donor's own JS, and gate item 3 is "Mean Δ_JS ≥ 0.05". That run ended `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`; commit `ddceca1` at 22:04 is titled "Treat incomplete JS matching as a COST scientific outcome". A forensic audit followed (`612067c`, 22:12). The "fixed-JS restoration" protocol, with JS target 0.01 and a floor of 0.02 (`08r2_fiber_cost_fixed_js_preregistration.md`, config `"behavior_floor": 0.02`), was committed at 22:20 (`987fb77`).

So the 0.02 floor was written after:
- the SPEC outcomes on the same 48 pairs, and
- the COST outcome on those pairs.

The forensic script says a "requested frozen fixed-JS protocol" existed, but no committed document before `63433f2` contains it. The observed matched-JS effect, +0.0266 with interval [0.0179, 0.0357], **fails** the 0.05 floor of the only protocol verifiably committed before outcomes. It passes only the later 0.02 floor.

The manuscript describes the COST run as a deviation from "that protocol". The package's own COST preregistration shows the run followed its committed protocol. The fixed-JS protocol even describes itself as a "protocol-compliant exploratory follow-up".

**Evidence:** VERIFIED from the protocol text and the commit log.
```
git log --format='%h %ad %s' --date=iso --since='2026-10-18 18:00' --until='2026-10-19 02:00'
63433f2 20:05 Add 08R2-FIBER-COST ...            (floor 0.05, donor-referenced JS)
ddceca1 22:04 Treat incomplete JS matching as a COST scientific outcome ...
612067c 22:12 Add a read-only forensic audit of the COST JS-matching protocol.
987fb77 22:20 Restore the original COST protocol: calibrate all seven directions to JS=0.01.  (floor 0.02)
```

**Required change:**
- State the timeline in the main text: the committed COST protocol, its 0.05 floor and its INCONCLUSIVE outcome, then the fixed-JS protocol written afterward on the same pairs.
- Say that the matched-JS result meets the later 0.02 floor and not the earlier 0.05 floor.
- Reclassify the matched-JS gate as post hoc / exploratory everywhere: Statistics section, Table 1, abstract.
- Remove "preregistered before outcomes" for this study.
- Do not describe the COST run as "not following that protocol" unless an earlier, dated, committed document specifying the fixed-JS design is provided.
- For a main-track submission, rerun the fixed-JS design on fresh or confirmation pairs.

### 2. MAJOR — "Gain over Editor A" is mostly a generic fixed-divergence effect, so the 56% stage share does not show a stage correction

**Location:**
- Abstract: "About 56% of the map's mean gain over Editor A was already obtained by a map that sees only player and occupancy."
- Results › Stage and legal-mask features…: "Much of its average gain over Editor A is a stage correction…"
- Discussion: "Stage-only features already deliver more than half of the gain over Editor A…"

**Problem.** All NST directions are calibrated to the same JS of 0.01 from h_E. At h_E the score is dominated by became-illegal mass (Q is about 0.58). Any perturbation that spreads probability lowers Q and raises Y. The data show that controls with no stage-specific or target-specific content already recover a large part of FULL's gain over Editor A.

| NST direction minus Editor A | mean ΔY | share of FULL's gain | wins |
|---|---|---|---|
| random nullspace | +0.0223 (iid 95% CI [0.0126, 0.0320]) | 41.5% | 34/48 |
| stage-mean direction | +0.0317 ([0.0224, 0.0407]) | 59.0% | 39/48 |
| wrong-target map | +0.0337 ([0.0238, 0.0432]) | 62.7% | 41/48 |
| STAGE_ONLY (archival arithmetic) | +0.0301 | 56.0% | n/a |

The same holds in matched-JS. Relative to Editor A, random directions gain +0.0232 and wrong directions gain +0.0196, against +0.0462 for correct directions; that is 50% and 43% of the correct gain. A map that sees only player and occupancy therefore does somewhat better than a random direction (0.0301 vs 0.0223). On the "beyond random" scale, it accounts for about (0.0301−0.0223)/(0.0537−0.0223) ≈ 25% of the learned gain, not 56%.

No interval is available for either share, because SEM's pair arrays are missing. As an indication of how imprecise such ratios are, the bootstrap 95% interval for the *random* share is [0.22, 0.63].

The "mean-level identity" (0.0041+0.0195 = 0.0056+0.0180 = 0.0236) is tautological. Both sums equal FULL−STAGE_ONLY on the same pairs. It confirms that the console labels are consistent and is not a robustness check. Mixing archival and recomputed values is numerically harmless here: my FULL−EA is 0.0537127 and the console gives 0.053713. The trouble is that it precludes any uncertainty statement.

**Evidence:** VERIFIED. The rows above come from `analysis/nst_pairs.csv` and from SPEC/fixed-JS raw logits via my script. The share arithmetic reproduces 0.56038.

**Required change:**
- Report every share against a random JS-matched baseline as well as against Editor A, and report the random and wrong-target shares alongside the 56%.
- Drop "is a stage correction" unless STAGE_ONLY−random has an interval above zero. Recovering the SEM arrays is required to test that.
- Give the 56% (and the subgroup shares) as point estimates with no interval, at two significant figures rather than "56.04%".
- In Table 1 and the abstract, present the NST learned−Editor A contrast with the warning that a random direction gets 42% of it.

### 3. MAJOR — The subgroups are nested and confounded, and subgroup patterns are over-read without difference tests or multiplicity control

**Location:**
- Results › Target identity…: "It is clear in category B (+0.0477 …) and in pairs where Editor A decoded the exact target board (+0.0395 …). Its interval includes zero in category A … and in pairs where Editor A did not decode…"
- Results › A train-only map…: "The transfer was weaker where Editor A failed to decode the exact target board."
- Figures 3B and 4A.

**Problem.** The cross-tabulation from `analysis/fixed_js_pairs.csv` and `analysis/nst_pairs.csv` shows the subgroups overlap almost completely:

| Study | B pairs that are exact | A pairs that are exact | Non-exact pairs that are A |
|---|---|---|---|
| matched-JS | 16 of 16 | 6 of 32 | 26 of 26 |
| NST | 16 of 16 | 8 of 32 | 24 of 24 |

"Category B" is a strict subset of "exact" and "non-exact" is a subset of "A", so the paper's four subgroups are essentially one split reported twice. Presenting both as separate observations ("clear in B *and* in exact pairs") double-counts.

The NST "weaker where Editor A failed" claim is not supported by a difference test. Exact minus non-exact for learned−wrong is +0.0175, with iid bootstrap 95% CI [−0.0007, 0.0358], which includes zero. For matched-JS correct−random, the A−B difference is −0.0371 [−0.0653, −0.0087] and exact−non-exact is +0.0305 [0.0052, 0.0559]. These exclude zero unadjusted, but they are among many looks.

The manuscript contains 36 interval statements (27 distinct) and makes no multiplicity statement. The pooled primary contrasts do not need correction; their sign-flip p-values are ≤0.0014 and survive Bonferroni across about 10 pooled contrasts. The subgroup readings do need it.

The exact-decode variable is an outcome of Editor A, which precedes the nullspace edit. Subgroups defined on it describe association, not moderation of a causal effect.

**Evidence:** VERIFIED (cross-tabs, difference CIs, sign-flip p-values from my script).

**Required change:**
- State that B ⊂ exact and non-exact ⊂ A in both studies, and collapse to one split, or report the 2×2 cells.
- Report subgroup *difference* estimates with intervals.
- Label all subgroup results exploratory and unadjusted, and state the family size.
- Replace "transfer was weaker where Editor A failed" with "the point estimate was smaller; the difference interval includes zero".

### 4. MAJOR — Preregistered arms bearing on the negative results are omitted, and FRESH changes the anchor

**Location:**
- Results › Reliable target switching…: "With the edit applied to the natural recipient activation and calibrated to JS 0.01…"
- Discussion: "Joint switching fails its 60% gate."
- Results › Stage and legal-mask features (SEM) section.

**Problem.**

(a) **The FRESH anchor differs from NST's.** FRESH applies the edit at h_R; NST, SPEC and matched-JS apply it at h_E. A switching failure at h_R therefore does not directly bound the h_E-anchored NST map that the Discussion and Conclusion generalize about.

The FRESH console gate contains `anchor_dependence_ok: 0` and `natural_symmetric_ok: 0`. The parent protocol (`08r2_symmetric_target_preregistration.md`) lists three anchors: natural h_R, Editor A for T, and Editor A for W. Its commit message reads "test whether target switching depends on the anchor". The Editor-A-anchored arms and the anchor-dependence criterion are not reported or explained.

(b) **The SEM crossover arm is omitted.** SEM's preregistration includes a crossover ("Report A_T, A_W, C, and both-arms-positive fraction"). The console (`sem_full_run.txt`) gives A_T = +0.0201 [0.0107, 0.0297] and A_W = +0.0058 [−0.0003, 0.0124], win fractions 0.8125 and 0.625. This is an in-sample switching test on the NST pairs at the h_E anchor, the anchor that matches the positive results, and its A_W interval includes zero. It is directly relevant and absent from the manuscript.

(c) **VNF reports no main effects or controls.** The manuscript gives no d_T(T), d_W(W) or other main effects, so the reader cannot judge whether a 0.005 interaction was detectable at ρ=ν=3.5. The random and label-shuffled controls are also unreported. The intervals do support a stronger statement than "near zero": the 95% upper bounds of 0.0030 (I_T) and 0.0022 (I_W) are both below the 0.005 floor.

(d) **The FRESH both-positive count is mostly explained by its marginals.** Under independence with the observed marginal rates (26/32 and 18/32), one expects 14.6 both-positive triplets, and 13 were observed. The positivity association is in fact negative (φ = −0.26, Fisher p = 0.20). The A_W marginal of 56% is already below the 60% gate. The failure itself is statistically clear: one-sided binomial test of H0 p ≥ 0.6 gives p = 0.021, with upper 95% bound 0.551.

**Evidence:** VERIFIED (console excerpts, protocol text, my computations for d).

**Required change:**
- Report the FRESH Editor-A-anchored arms and the anchor-dependence result, or explain why they cannot be.
- Restrict the switching conclusion to the h_R anchor.
- Report the SEM crossover (A_T, A_W, C, both-positive) next to FRESH.
- Report VNF main effects and both controls, and state the interaction result as "upper 95% bounds 0.0030 and 0.0022, below the 0.005 floor".
- Note that the FRESH gate failure is driven largely by the A_W marginal.

### 5. MAJOR — The estimand is narrower than some causal language implies, and the correct-vs-random contrast is confounded with restoring natural nullspace content

**Location:**
- Discussion: "For this frozen linear board probe, the residual stream is not a single occupancy code plus unused orthogonal content."
- Conclusion: "this probe is not a sufficient causal description of the residual information that affects legal-move outputs."
- Results › Correct-target nullspace directions…

**Problem.** Every non-random direction has the form P_N(x) − P_N h_E, where x is h_D, h_W or ẑ(b). Editor A is a full-space editor, so P_N h_E already carries Editor A's own nullspace perturbation. Correct, wrong, learned and stage-mean directions all partly undo that perturbation and move the nullspace toward natural content; random directions do not. Two consequences follow:

- Correct−random and "gain over Editor A" mix target information with return toward natural activations. Finding 2 shows the latter is large.
- In SPEC at t=1, the correct direction is an exact nullspace interchange, a natural state. The wrong directions are rescaled to the correct direction's norm, so they are not natural states. The SPEC correct−wrong contrast therefore also mixes naturalness with target identity. Matched-JS removes this, because every direction is rescaled.

Only correct−wrong at matched JS, and learned−wrong in NST, approximately isolate target identity. Even these effects are measured from an edited, possibly off-manifold h_E, which the Limitations section acknowledges. The Discussion's statement about "the residual stream" goes beyond the estimand, which is the effect of a constructed addition at h_E.

The SPEC archive also contains unreported `full_donor` and `correct_rowspace` cells, with mean Y +0.725 and +0.363. These give a natural scale for the effects:
- at matched L2, the correct nullspace edit closes about 57% of the gap from Editor A to the full donor, (0.144+0.639)/(0.725+0.639);
- at matched JS, the correct gain over Editor A (0.046) is about 3.4% of that gap.

**Evidence:** VERIFIED for the cell means and ratios. The confounding argument follows from the direction definitions in the manuscript (judgment).

**Required change:**
- Name correct−wrong (matched JS) and learned−wrong as the primary target-specific estimands.
- Describe correct−random and gain over Editor A as partly generic and partly manifold-restoring.
- Restate the Discussion and Conclusion claims as effects of this intervention at h_E.
- Report the full-donor and rowspace cells as a ceiling or scale reference, and add them to the protocol-deviation list if they were not preregistered.

### 6. MINOR — The recipient-game clustering is vacuous and the real dependence structure is not described (intervals are nevertheless robust)

**Location:**
- Statistics: "Intervals are 95% percentile bootstrap intervals clustered on recipient game…"
- Appendix A › Bootstrap: "The statistic is the mean of cluster means…"

**Problem.**
- **Recipient clustering.** In SPEC, matched-JS and NST, each of the 48 recipient games occurs exactly once, as does each target game, and FRESH/VNF have 32 distinct recipients. The clustered bootstrap is therefore an ordinary pair bootstrap, and "mean of cluster means" equals the reported pair mean. `p1lib.cluster_bootstrap` reports `point = vals.mean()` and centres the bootstrap on cluster means; these would disagree if clusters were ever unequal.
- **Wrong-donor reuse in SPEC.** Cross-pair dependence actually enters through wrong donors. SPEC's 144 wrong-donor slots use 122 games; 20 games are used more than once and 5 activation positions are reused. 18 wrong-donor games are other pairs' recipients and 13 are other pairs' targets. Linking pairs that share any game puts 40 of 48 pairs in one connected component, so component clustering is not feasible.
- **Wrong-target reuse in NST.** NST's 48 wrong-target slots come from 35 games.
- **Sensitivity.** A two-way (pair × donor-game) cluster-robust standard error barely changes anything: matched-JS correct−wrong SE is 0.00461 one-way vs 0.00457 two-way, normal CI [0.0176, 0.0355]. NST learned−wrong SE is 0.00486 vs 0.00483, CI [0.0106, 0.0295].
- **FIBER.** The FIBER pairs are concentrated: one target game (val_00007) appears in 25 of 32 pairs, including 23 of 24 category-A pairs. "Share only two target games" is accurate but understates that one game dominates.

**Evidence:** VERIFIED (counts from the pair manifests; two-way CGM computed by me).

**Required change:**
- State that the bootstrap is effectively a pair bootstrap.
- Describe the wrong-donor reuse and report the two-way sensitivity analysis.
- In FIBER, state that 25 of 32 pairs share one target game.

### 7. MINOR — Precision is overstated for Monte Carlo intervals and ratio estimates

**Location:**
- Evidence levels: "Every recomputed bootstrap interval matches the original run to within 3×10⁻⁸."
- Table 1 (four-decimal bounds) and "56.04%".

**Problem.** Agreement to 3e-8 shows same-seed reproducibility, not Monte Carlo accuracy. With an independent seed and 20,000 iid resamples I obtain:
- SPEC correct−wrong [0.3167, 0.4733] (paper [0.3175, 0.4758])
- matched-JS correct−random [0.0099, 0.0367] (paper [0.0103, 0.0360])

The fourth decimal is therefore not meaningful.

**Evidence:** VERIFIED.

**Required change:** Report bounds to three decimals, or use ≥10,000 resamples and state the Monte Carlo error. Give the stage share to two significant figures.

### 8. MINOR — The correlation discussion is correct but incomplete

**Location:** Results › A train-only map…: "That correlation is not evidence of a shared mechanism … the correlation mostly reflects pair-to-pair differences in baseline Y."

**Problem.** The raw correlations are correctly reported and correctly discounted:
- r(learned, oracle) = 0.9978
- r(learned, wrong) = 0.9914
- r(learned, Editor A) = 0.9938

After removing the Editor A baseline by regression, however, a substantial association remains, and it is specific: partial r(learned, oracle | Editor A) = 0.834, against partial r(learned, wrong | Editor A) = 0.441. Gain-on-gain correlations should not be used as evidence because they are mathematically coupled through the shared baseline term: r(learned−EA, oracle−EA) = 0.846, and r(learned−wrong, oracle−wrong) = 0.881.

**Evidence:** VERIFIED.

**Required change:** Add the baseline-adjusted partial correlations, keep the caution about mechanism, and warn against coupled difference correlations.

### 9. MINOR — Sample reuse is disclosed in Limitations, but its consequences are understated in the abstract and Results

**Location:**
- Abstract (no reuse caveat).
- Limitations: "Evaluation samples are reused…"

**Problem.** Matched-JS reuses the SPEC pairs, and its protocol was designed after seeing SPEC and COST outcomes on them (see Finding 1). SEM reuses the NST pairs, and the stage-share "mean-level correction" was itself a post hoc document. The only fresh cohorts (FRESH, VNF) are negative and archival.

One reassuring fact should be stated: the pair-level correct−wrong contrasts in SPEC and matched-JS correlate at only r = 0.155. Outcome noise is therefore largely not shared, even though the selection of hypotheses is.

**Evidence:** VERIFIED (the correlation); the reuse facts come from the manifests and protocols.

**Required change:** Add one abstract sentence: the positive results come from sequentially designed studies on reused validation pairs, and the independent cohorts were negative and are archival-only. Note the low SPEC/matched-JS correlation.

### 10. MINOR — Unreported preregistered secondaries and the NST tuning boundary

**Location:**
- Experimental setup › Activations and interventions: "Grouped cross-validation on training games chose k and α…"
- No mention of the secondary JS targets.

**Problem.**
- The fixed-JS protocol preregistered secondary JS targets of 0.005 and 0.02. NST Stage 0 records that they "were not in the primary archive", and the manuscript is silent on them.
- The NST cross-validation chose k = 64, the largest value in the grid {8, 16, 32, 64}. CV R² was still rising at the boundary (0.147, 0.201, 0.259, 0.326), so the chosen map is a boundary solution.
- The SEM, FRESH and VNF seeds stated in Appendix A (9482, 10482) cannot be verified from the console excerpts.

**Evidence:** VERIFIED (`data/raw/nst/null_state_cv_table.csv`, protocol text).

**Required change:** Disclose that the secondary JS targets were not run or reported, disclose the k boundary, and mark archival seeds as unverified.

## Independent recomputations

Script: `/tmp/reviewB_recompute.py`, my own code; it does not import `p1lib`. It recomputes the token map and softmax, derives the became-legal and became-illegal sets from the recipient and target legal sets as set differences, and computes Y = G − Q, JS, iid bootstrap (20,000 resamples, independent seeds), sign-flip tests, subgroup difference intervals, and partial correlations. I ran it with `/Users/shkh/opt/anaconda3/bin/python /tmp/reviewB_recompute.py` from the package root. The two-way cluster SE and the extra baseline checks came from short inline scripts.

| Quantity | Manuscript | My recomputation | Match |
|---|---|---|---|
| SPEC correct−wrong mean, wins | +0.3916, 46/48 | +0.39163, 46/48; iid CI [0.3167, 0.4733] | Yes (CI within MC error) |
| SPEC correct−random mean, wins | +0.7086, 48/48 | +0.70858, 48/48; CI [0.6028, 0.8203] | Yes |
| SPEC mean Y correct / wrong / random / Editor A / recipient | +0.144 / −0.248 / −0.565 / −0.639 / −0.707 | +0.1436 / −0.2480 / −0.5649 / −0.6387 / −0.7070 | Yes |
| SPEC became-legal/illegal lists vs set differences | — | 0 mismatches in 48 pairs | Yes |
| Matched-JS correct−wrong mean, median, wins | +0.0266, 40/48 | +0.026551 (median 0.01791), 40/48; CI [0.0179, 0.0360] | Yes |
| Matched-JS correct−random | +0.0230 [0.0103, 0.0360], 30/48 | +0.022966, 30/48; CI [0.0099, 0.0367] | Yes |
| Matched-JS max \|JS − 0.01\| | 1.23e−6 | 1.232e−6 | Yes |
| Matched-JS mean Y correct / wrong / random | −0.592 / −0.619 / −0.615 | −0.5925 / −0.6190 / −0.6154 | Yes |
| Matched-JS subgroup correct−random A / B / exact / non-exact | +0.0106 / +0.0477 / +0.0395 / +0.0090 | +0.0106 / +0.0477 / +0.0395 / +0.0090 | Yes |
| NST learned−wrong mean, wins | +0.0201, 39/48 | +0.020057, 39/48; CI [0.0106, 0.0295] | Yes |
| NST learned−Editor A, wins | +0.0537, 47/48 | +0.053713, 47/48; CI [0.0454, 0.0620] | Yes |
| NST learned−random / learned−stage-mean wins | 35/48, 42/48 | +0.03142, 35/48; +0.02200, 42/48 | Yes |
| NST Y_learned > 0 | 1/48 | 1/48 | Yes |
| NST mean Y learned / oracle / Editor A / recipient | −0.582 / −0.583 / −0.636 / −0.701 | −0.5821 / −0.5827 / −0.6358 / −0.7006 | Yes |
| NST Pearson r learned–oracle / –wrong / –Editor A | 0.998 / 0.991 / 0.994 | 0.9978 / 0.9914 / 0.9938 | Yes |
| NST subgroup learned−wrong exact / non-exact | +0.0288 / +0.0113 | +0.0288 / +0.0113 (non-exact CI [−0.0009, 0.0236]) | Yes |
| Stage share | 56.04% | (0.053713 − 0.023613)/0.053713 = 0.56038 | Yes (arithmetic) |
| FIBER mean / minimum JS | 0.380 / 0.185 | 0.3795 / 0.1853 | Yes |
| FRESH counts (from console cells) | A_T 81%, A_W 56%, both 13/32 | A_T 26/32, A_W 18/32, both 13 | Yes |

Additional computations that the paper does not report: the random, stage-mean and wrong-target shares of gain over Editor A (Finding 2); the subgroup nesting and difference intervals (Finding 3); FRESH independence expectation, φ, Fisher and binomial tests (Finding 4); full-donor and rowspace cell means (Finding 5); two-way cluster SEs and donor-reuse counts (Finding 6); partial correlations (Finding 8); and the SPEC vs matched-JS pair-contrast correlation of 0.155 (Finding 9).

## Strengths

- The package is reproducible to a high standard. Raw logits, probe probabilities, directions, the fitted map and legal sets are shipped with hashes, and every recomputable number I checked matched, from independent code.
- Wrong and random responses are averaged within each pair before inference, so the unit of analysis (pair or triplet) is right and the 336 interventions are not pseudo-replicated.
- Matching output divergence, and reporting how it shrinks the SPEC effect by more than an order of magnitude, is a good control and is presented honestly.
- Negative and failed-gate results (FRESH, VNF, SEM floor, absolute Y < 0 in 47/48) are reported prominently and are allowed to bound the claims.
- The paper correctly separates reconstruction R² from behavioral share, and correctly discounts the r = 0.998 correlation as mostly baseline-driven.
- Evidence levels (recomputed, recorded, archival) are labeled per number, and the claim ledger maps each token to its source.
- Wrong-target scoring on the intended target's legal sets is stated explicitly, and donors are matched on simulator features rather than model outputs.
