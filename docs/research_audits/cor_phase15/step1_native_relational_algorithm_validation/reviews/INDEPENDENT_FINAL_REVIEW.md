# Independent final review: Investigation XII (fresh-cohort confirmation of the native relational algorithm)

Reviewer: a separately initialized agent with no shared working context. I did not do this work and was not
involved in it. I read the five result/protocol documents, the amendment, `logs/COMPUTE_LOG.txt`, the scorer
outputs, the analysis scripts and the raw per-parent JSON outputs. I checked the git chronology with `git show`.

I ran two small checks of my own (`nice -n 19`, under 10 s CPU each, nothing retrained):
- (i) a re-count of `logs/score_xii.json["checks"]`;
- (ii) a numpy recomputation of the Bayes references on the frozen factorial episodes. This used stream 70_000_000
  and the same cell construction and rng 1104 as `alt_ablation.py`, so n = 400 episodes. It compared
  Bayes(xA) with cf_e0(xAP), and computed the entropies of the two reference distributions.

Severity scale:
- **high**: changes a headline claim, or makes a passed check non-evidential;
- **medium**: weakens or mis-states a claim, or a control is missing;
- **low**: a documentation or provenance defect.

Summary verdict:
- Several frozen verdicts are arithmetically correct as scored: P1, P2, P4, P5, P7 and P8 pass, and P3 fails.
- P6-as-coded passes with a null intervention. It should not be called a confirmation.
- Several interpretive claims built on the verdicts are contradicted by the package's own data, especially in the
  "corrected mechanistic account", P8 and P7.

---

## 1. Mechanistic claims

**1.1 [high] The "corrected mechanistic account" and the "invariant" claim are contradicted by the fresh incomplete
parents.**

Where the claims are made:
- `NATIVE_CIRCUIT_AND_CAUSAL_MECHANISMS.md` §7 (lines 207, 211–222) says perfectness reaches the conjunction
  "primarily through attention routing" *in every fresh parent*.
- It says the value route held "in none of the fresh ones".
- It says locus, mediation and amplification are "invariant across all 3 + 4 fresh parents".
- `FRESH_CONFIRMATION_AND_STATISTICS.md` line 279 says: "The incomplete parents behave like the complete ones on
  every mechanism arm."

What the data show:
- `logs/score_xii.json` → `arms.incomplete.P3.b_subhead_redundancy...` = **2/4**. P151 and P152 *pass*, i.e. they
  show the development-style redundancy.
- `logs/alt_ablation_conf.json`, vP151: pattern_A KL_B/KL_e0 = 0.041/0.267. vP152: 0.117/0.221.
- **P156 is value-route dominant**:
  - values_A = 0.183/0.103, pref_e0 0.51, i.e. on the cf_e0 side;
  - pattern_A = 0.139/0.311, on the Bayes side.
- In `logs/circuit_l1_conf.json`, vP156 has an attention share of E-interaction of **0.05**. Its MLP1 share is 0.95,
  and MLP1's *own* nonlinear interaction is 0.33. In P156 the conjunction is formed in MLP1, not by softmax coupling
  in attention.
- `arms.incomplete.P8.d` = 3/4: P156's adapted E-dependence is 0.028.
- Across 7 fresh + 11 development models the routing is heterogeneous: pattern-dominant, redundant, and
  value-dominant all occur. The data support "implementation varies across parents". They do not support any single
  fresh-cohort "rule".

**1.2 [high] The P3(b) interpretation, "not routed redundantly", misreads the 2 × 2 of route removals.**

Complete parents, `alt_ablation_conf.json` (KL_B):

| model | intact | values_A | pattern_A | path_inA |
|---|---|---|---|---|
| P150 | ≈0 | 0.018 | 0.327 | 0.585 |
| P154 | ≈0 | 0.021 | 0.273 | 0.682 |
| P155 | ≈0 | 0.003 | 0.423 | 0.712 |

- With the pattern route removed, the remaining value route still holds the decision about halfway. pattern_A is
  0.40–0.59 of the path_inA KL, and pref_e0 is only 0.35/0.40/0.54.
- Removing the values on top of that completes the shift.
- So the value route is a *partial* backup, masked when patterns are intact. This is asymmetric partial redundancy
  with a strong positive interaction between the routes, not "no redundancy".
- `FRESH...` line 307 says "Removing it from the *values* alone does nothing". That is true only with the patterns
  intact, and it should be stated with that condition.
- The falsification of P3(b) as frozen is correctly scored (0/3). The *interpretation* in FRESH lines 308–311 and
  NATIVE §7 overstates it.

**1.3 [high] The pattern-vs-value caution is correct, but the documents then use the same reference-dependent split
as mechanism.**

- NATIVE §2.3 (lines 96–104) correctly notes that the bilinear split of a double difference depends on the reference
  cell. This is standard path/order dependence of attribution, as in Shapley-order and Integrated-Gradients baseline
  dependence.
- Several claims rest on exactly that kind of split:
  - §2.2 (lines 90–94), "softmax-of-additive-scores term 0.09–0.59 vs full pattern term 0.14–0.65". In
    `circuit_l1.py` lines 67 and 88, PAT is computed with the **x00 reference** (`u["x00"]`).
  - The claim "It is not created by a multiplicative QK feature".
  - The §7 corrected account.
- The value term `a_00·dd(u)` is ≈ 0 *by construction*. A and P edit different tokens, and u_s depends only on token
  s, so dd(u) is non-zero only through layer-0 mixing. `circuit_l1_*.json` shows `value` = 0.00 in all 20 models.
  The finding that "the pattern term carries the interaction" is therefore largely structural.
- The "intervention-level facts are invariant" (line 106) are not reference-free either:
  - pattern_A clamps values at xAP;
  - values_A clamps patterns at xAP.
  These are the two orderings of the same decomposition, carried through the network.
- **Provenance**: the "values fixed at xAP" numbers (0.02 / −0.01 / 0.00, line 101) are produced by no script and
  appear in no log in the package. Only the x00-reference numbers (0.265/0.201/0.097) are reproducible from
  `circuit_l1_dev.json`.

**1.4 [medium] "Keys/values not queries" is only half isolated.**

- In `alt_ablation.py` line 114, `path_inA_keys_only` feeds `L1_in = xA` with only the **q7 queries** reset to xAP.
- So the non-q7 queries and the **MLP1 input residual** (line 72, `xin + oh`) are also from xA.
- This is "path_inA except 7 query rows". That explains why it is "identical to path_inA to 3 decimals" in every model
  (dev and conf, e.g. 0.765 vs 0.766 for xP81).
- Supported claim: *q7-token queries are irrelevant* (`path_inA_queries_only` has no effect).
- Not isolated: a keys/values-only attribution versus the MLP1 input or the non-q7 queries. Doing that would need
  MLP1's residual held at xAP.

**1.5 [medium] "MLP1 amplifies" is a mediation result, not an amplification result.**

- P3(c) compares `attn_addcomp_mlp_recomputed` with `attn_addcomp_mlp_frozen` (`alt_ablation.py` lines 109–110).
  The frozen version keeps MLP1's *intact* xAP output, which already contains MLP1's copy of the interaction (MLP1
  share 0.29–0.95, `circuit_l1_conf.json`).
- Any serial downstream mediator gives total > direct. The test excludes compensation for this pair only. It does
  not show gain > 1.
- "Removing it changes the decision about twice as much" (NATIVE line 216): the observed ratios are 2.1, 1.6 and
  4.6 (P150/P154/P155).

**1.6 [medium] "Smooth, not thresholded" and the count-curvature alternative.**

- e_h = 1[n_h = 5 and agrees at q] is simply "group h agrees at all 6 questions". It does not depend on q
  (CAUSAL XII-1 proof).
- The TRUE (4→6 agreements) vs CTRL (3→5) contrast in `factorial.py` therefore cannot distinguish two hypotheses:
  - a perfectness × agreement-at-q conjunction;
  - any function of the total agreement count whose curvature grows near 6.
- The "smooth GELU curvature" finding is what the second hypothesis predicts.
- The "AND neuron" criterion (`circuit_l1.py` line 113: sign pattern −,−,−,+) is one specific Boolean signature.
  Its low share does not exclude soft-threshold units.
- The "relational conjunction" framing should be stated as "a nonlinearity in the all-question agreement indicator
  that first appears in layer 1".

**1.7 [medium] The "E-interaction remaining" metric is mislabelled for path designs, and the single-head
compensation claim depends on it.**

- In `alt_ablation.py` line 87, rem = ‖E(x2) − E(xA) − E(xP) + E(x00)‖ / I0. For path_inA, x2 = resid1(xAP) +
  L1out(xA), so rem reduces to ‖E(layer-1's own P main effect)‖. It is not the remaining interaction.
- Evidence of the mismatch:
  - path_inA moves the decision fully to cf_e0, yet "remaining" is 0.82–0.92 in xP81, xP81 s1 and yP130 s1
    (`alt_ablation_dev.json`), and 0.77 in P156;
  - scrub_match leaves the decision intact, yet "remaining" is 1.09–1.78.
- The single-head "total ≥ direct, no hydra" result (NATIVE lines 150–153; `alt_ablation.py` line 135) compares a
  projection share with 1 − rem. It inherits this defect.

**1.8 [low] The completeness of the pathway is acknowledged as partial, which is correct.**

- Layer 2 is not dissected (NATIVE §3, §6).
- The complement of E restores 0.19–0.34 of the gap in the fresh complete parents (`restore_conf.json` rec_complement;
  P153 −0.23). The fresh P4 summary (`FRESH` line 315) omits this. It should be reported next to rec_E.

## 2. Causal identification

**2.1 [high] The P2(b) path_inA result does not discriminate the conjunction hypothesis from "layer 1 relays the xA
input".**

- My check (ii) gives KL(Bayes(xA) ‖ cf_e0(xAP)) = **0.018**, with argmax agreement **0.982**, on the frozen
  factorial episodes.
- cf_e0 for xAP is therefore practically the Bayes posterior *of the xA input*.
- "path_inA moves the decision to cf_e0" is then also predicted by the trivial hypothesis that layer 1 is the conduit
  for everything P changes, main effect included.
- path_inP *is* discriminating: KL(Bayes(xP) ‖ cf_e0) = 1.199, and the model goes to cf_e0 (KL_e0 0.06–0.16).
- A specificity control exists but is not reported in P2: CTRL-design path_inA has `CTRL_use_nat` ≈ 0.04–0.05,
  against TRUE 0.55–0.73 (`dyn_mech_conf.json`, final checkpoint).
- Correction: present path_inP and the CTRL contrast as the evidence, and path_inA as non-discriminating.

**2.2 [high] The binary "side" criterion (KL_B < KL_e0) is strongly biased toward the Bayes side.**

- On the frozen episodes, cf_e0 is much sharper than Bayes: entropy 0.21 vs 0.58 nats (K = 3).
- A uniform output gives KL_B = 0.52 against KL_e0 = 0.89.
- An exact 50/50 mixture of Bayes and cf_e0 gives KL_B 0.08 < KL_e0 0.14, with pref_e0 = 0.02.
- So any degraded, diffuse or half-shifted output scores as "Bayes side". Consequences:
  - The cf_e0-side passes (P2 a–d) are conservative, so they are fine.
  - The "stays Bayes-side" readings carry little information:
    - the P3(b) redundancy criterion, and with it the 10/11 development "redundancy";
    - the P2(e) "non-specific" reading.
- Mean ablation of layer 1 gives KL_B 0.12–0.31 and **pref_e0 0.06–0.18** in all fresh models
  (`alt_ablation_conf.json` mean_L1). The decision stays nearer Bayes than cf_e0. NATIVE §1 (line 44) calls this
  "moves the decision away from both references". Without an entropy-matched reference this cannot be told apart from
  partial survival of eligibility under removal of layer 1, which would be a possible compensation route. That
  possibility was not examined.

**2.3 [high] The "natural-activation" label is overstated for scrubbing and restoration.**

- In `alt_ablation.py` lines 118–119 and `restore*.py`, the patched resid2 vector is
  `base + QQᵀ(donor − base)`: a synthetic state at the patched site, even when the donor is natural.
- CAUSAL §4 places scrubbing in class II ("each component state is natural"). For a subspace patch that is not
  accurate.
- path_inA and `resamp_A` are the **same intervention**: both put xA's layer-1 outputs on the xAP residual, and the
  numbers are identical in every model. NATIVE §1 (lines 32–42) counts them as separate designs supporting the
  rejection of H3.
- What the designs avoid is XI's *additive completion inside layer 1*. They do not avoid off-manifold resid2 states.

**2.4 [high] Scrubbing with same-e donors re-tests the DAS objective. It is not independent evidence of native
mediation.**

- `das.py` (lines 30–35, 64–67) trains Q so that swapping E coordinates from any source gives the E-counterfactual.
  When e_src = e_base, that counterfactual is the base posterior.
- P2(d) scrub_match, P2(d) scrub_e0, P4 restoration and P6(a) therefore test generalization of the fitted DAS map to
  new episode distributions (factorial cells, ablated bases, discriminating pairs). Useful, but in-objective.
- The relevant known failure mode is the subspace-patching "interpretability illusion" (Makelov et al., 2023). It is
  not addressed. The random-subspace control only shows that DAS optimization succeeded.
- A stronger control would be a DAS subspace trained for a decoy variable (A1/PERM), or for NULL, in the same model,
  run through the same scrub/restore protocol.
- Same-e donors do move the E coordinates substantially (scrub_match "remaining" 1.09–1.78) without changing the
  decision. That supports "value, not exact coordinates", but only inside an objective-shaped subspace.

**2.5 [medium] P6-as-written: E-identification of the full models is real but thin, and the C1 contrast is weaker
than stated.**

- Only **117** informative pairs per model (`n_inf`, `disc_ident_conf.json`), identical across models. P150's
  IIA_E,inf = 0.744 against a 0.70 threshold, with a binomial SE ≈ 0.04.
- On all discriminating pairs, the patch *lowers* agreement with cf_E relative to no patch:
  - P150: IIA_E 0.745 vs base_E 0.847;
  - P155: 0.869 vs 0.890.
- The C1 models of P150 and P154 have IIA_E,inf **0.86 / 0.82** (`disc_ident_c1conf_A.json`). Their "E" subspace
  passes the E part of the test as well as the repaired models do.
- So "the label E is a false positive in C1" (CAUSAL §Discriminating-pair design) is better put as "the C1 subspace
  moves decisions toward *both* E and A1 targets".

**2.6 [low] Leakage.**

- I found no data leakage. DAS is trained on stream 61_000_000 + seed (61_000_010/011), and every evaluation stream
  is distinct.
- But "analysis streams are all new" (`FRESH` line 35) is not quite true:
  - the scrub/mean-ablation donor pool (rng 77_000_000) is the same as in development;
  - the cell-construction rng (1104) is the same;
  - the factorial base episodes on stream 70_000_000 are shared across P1–P4 and P7.
- These are common random numbers, not leakage, but they should be stated.

**2.7 [medium] Restoration.**

- The fresh P4 used `restore.py`, i.e. XI's *hybrid* additive-completion ablation. The natural-ablation restoration
  (`restore_nat.py`) exists only for 8 development models, post freeze.
- "XI's restoration result is therefore not an artifact of its hybrid ablation" (NATIVE line 63) therefore has no
  fresh-cohort support.
- `restore.py` normalizes by `restore_all` (all q7 coordinates only), not by intact as the docstring says (line 7 vs
  line 52). This is why rec_E is 1.088 > 1 for P150.

## 3. Experimental and statistical integrity

**3.1 [high] The amendment's integrity statement is factually wrong.**

- `STAGE0_AMENDMENT_1.md` lines 3–5 say that at the time "no T4 evaluation or analysis output of any fresh model
  existed or had been read".
- The amendment commit `5425317` (09:04:55Z) itself contains:
  - `logs/run/6591b9ec7d30.out` and `logs/run/f40012d18991.out`: the vP150 and vP151 full_c1 fine-tune logs, with
    per-checkpoint eval_conf12 **KL, accuracy and E-diagnostics through step 2686**;
  - 22 intermediate checkpoints, `checkpoints/native_train/vP15{0,1}_full_c1_.../step*.npz`. These are the very
    checkpoints P7 analyses.
- I cannot tell whether the logs were *read*. But the statement that no T4 evaluation output existed is false.
- Stage 0 itself anticipates that these curves are written ("not read"). The amendment should say: "training-curve
  output existed and was committed; we state it was not inspected".

**3.2 [medium] The amendment reverses an explicit frozen rule, and Part II does not say so.**

- `STAGE0_PROTOCOL_FREEZE.md` line 14: "if fewer than 4 parents are complete, **no extra parents are trained**".
- Amendment 1 trains two extra parents. It also introduces the n ≥ 3 "adequate" label and the secondary aggregate
  mid-run.
- The rationale (n ≤ 2 is too weak) is reasonable, and the trigger (aux_acc) is outcome-blind.
- But FRESH Part II, Deviation 1, presents it as an addition, not a reversal of a pre-registered rule.
- Also, "the only code changes are the parent lists" (amendment lines 36–37) is inaccurate. The same commit changed:
  - `conf_merge.sh`, `conf_transfer.sh` and `make_conf_queues.py`;
  - the `src/genlib.py` docstring;
  - `math_xii.py`, substantively;
  - and it added `restore_nat.py` and `vs_linear_randctrl.py`.
- None of these changes alters a frozen arm. I verified the hashes of all analysis scripts against
  `FREEZE_XII_HASHES.txt`. Only `score_xii.py` differs, as disclosed.

**3.3 [high] P6-as-coded is passed by a null intervention.**

- The coded arm requires IIA_E ≥ 0.70 and IIA_A1 ≤ 0.30 on all discriminating pairs (`score_xii.py` line 67).
- The *unpatched* models already give base_E 0.85–0.89 and base_A1 0.08–0.10 (`disc_ident_conf.json`).
- So a no-op passes in every parent.
- FRESH line 270 reports P6 *as coded* as "confirmed", with "neither preferred". It must instead be labelled
  non-diagnostic.
- The P6(VS) arm is admitted non-diagnostic (FRESH lines 337–343). This was **already known before the confirmation
  outputs**:
  - the development random-subspace control finished 08:57Z and was committed in `5425317`
    (`logs/vs_linear_dev_randomQ.json`);
  - it passes the frozen VS criterion: xP81 LIN 0.854 vs best projection 1.277, ratio 0.67, eigenvalue 5.73.
- Yet CAUSAL lines 196 and 256 still say "A linear-map control on a random 6-dim subspace was not run".

**3.4 [medium] The 94/97 count is arithmetically correct but not evidential.**

- Re-derived from `logs/score_xii.json["checks"]` for groups complete + pairs: 97 checks, 94 passed. The three
  failures are P3(b) × 3. Composition: P1 12, P2 15, P3 12, P4 6, P6 6, P7 9, P8 12, P5 25.
- Of the 94 passes:
  - 3 are P6-coded(a), which a no-op passes;
  - 3 are P6(VS), non-diagnostic;
  - 10 are P5(e), a check that the *metric is inflated*, i.e. a methodological check, not support for any
    hypothesis;
  - the P5 ordered pairs count each direction separately, from 2 replicate parents and 3 cross parents.
- Checks within a parent are not independent.
- Recommendation: drop the headline count, or give it with this breakdown.

**3.5 [medium] The confidence intervals understate uncertainty.**

- `stats_xii.py` bootstraps episodes within one fitted model and one DAS fit. Variance from the DAS fit (random init,
  1200 steps), the fine-tuning seed and the parent is excluded.
- The seed-11 replicates were not used to estimate within-parent seed variance for P1–P4.
- Most P2–P8 numbers in Part II are point values without intervals.
- P5 gains are reported without CIs although `per_pair_KL` exists. Their denominators are small: KL_mean − KL_within
  = 0.10–0.19 nats, e.g. P155 target 0.114 − 0.017.
- On discriminating pairs, the source-free mean patch is sometimes *better* than the mapped source (IIA_disc,
  P155 target: mean 0.97 vs mapped 0.92–0.95).

**3.6 [medium] n = 3 and pooling of development models.**

- With n = 3 the pass rule is 3/3, and the documents rightly disclaim population inference (FRESH lines 367–371).
- Elsewhere the documents pool the threshold-setting development models with confirmation models as evidence:
  - "invariant across all 3 + 4 fresh parents and 11 development models" (NATIVE line 221);
  - "development + fresh, 16 models" (GEN line 135).
- Development models cannot confirm thresholds set on them.
- P5(b) "transfer equal" is an equivalence claim (|Δ| ≤ 0.10) near the ceiling of a residual-relative metric, with no
  interval. It has low power to detect a difference.

**3.7 [medium] Cohort selection.**

- 4/7 parents were incomplete, against 1/4 in X with "the unchanged" pretraining script. The cause was not
  investigated.
- Completeness may moderate mechanism. All 3 complete parents are pattern-dominant, while the incomplete ones are
  mixed (§1.1).
- The incomplete-parent results are therefore not simply "the same pattern". "Complete" is a candidate moderator, and
  calling the complete-parent pattern a "fresh-cohort rule" is selection-sensitive.

**3.8 [low] Freeze defects 1 and 2 are handled acceptably but incompletely.**

- Reporting both P6 versions is right.
- But the C1 runs (defect 2) were made after the full-model P6 outputs and a partial-cohort dry run (12:47Z) had been
  seen.
- Two of the three C1 runs (`disc_ident_c1conf_A.json` 14:31:00Z, `_B.json` 14:31:24Z) have **no entry** in
  `COMPUTE_LOG.txt`. Only `_C` (line 236) is logged.
- "Freeze defect" wording is accurate. "Post hoc" is the right label for P6-text(b).

**3.9 [low] Freeze timing is clean.**

- The prediction freeze `87e2a9f` is at 08:01:28Z. The first fresh fine-tune starts at 08:01:32Z (COMPUTE_LOG
  line 23).
- No fresh-cohort output other than pre.json existed before the freeze.
- The frozen Part I text is unchanged (`git diff 87e2a9f HEAD` shows only the appended Part II).

## 4. Generalization (P8)

**4.1 [high] "Readout-only repair" is a misnomer, and "bypasses E" means only "bypasses a 6-dim DAS subspace".**

- `adapt_c2.py` retrains a **full transformer block** (layer 2: attention + MLP, plus final LN and readout) on 40k
  C = 2 episodes. That is one third of the network.
- A *fresh, randomly initialized* layer 2 reaches the same 0.80 in all 7 fresh models (`adapt_c2_conf/*.json`
  fresh_l2).
- So the result follows from:
  - resid2 retaining the raw report information (token identity is preserved by the residual stream);
  - one block being able to compute the exact posterior from it.
- It says almost nothing about the native E representation, beyond "not necessary".
- The mean-patch removes 6 of 448 dims per q7 block. e remains decodable from the complement (restore_complement
  0.19–0.42) and from other tokens. "Bypasses the eligibility representation" is not shown.
- GEN §1.2 Findings 1 and 2 are overclaims (lines 51–55, 63–68):
  - "the failure is entirely downstream of layer 1": repair sufficiency does not localize the failure;
  - "the required information is **linearly** available in resid2": no linear probe was run, and layer 2 is
    nonlinear;
  - "the obstacle is computational, in the aggregation stage".
- The caveat at GEN lines 122–123 is right and should replace these statements.

**4.2 [low] The exact posterior is used correctly.**

- `xii_bayes.posterior_shared` matches the sampler (shared copied value and copy event; M2).
- The adapted models reach KL_true 0.002–0.005 with final loss 0.414–0.42, which is consistent with the posterior
  being the true one.
- The XI correction (0.75 → 0.80) is sound.
- Minor inconsistencies:
  - "KL 0.049–0.076 / gap 4.6–6.1" (FRESH line 196, CAUSAL line 46) vs "0.047–0.076 / 4.4–6.1" (CAUSAL line 124);
  - exact Bayes accuracy 0.794 / 0.799 / 0.804 on different sets, not always labelled.

**4.3 [medium] P8(a) "applies the C = 1 rule".**

- The C1 rule's own accuracy at C = 2 is 0.336, i.e. chance (`refs.C1rule_acc`).
- "Accuracy ≤ 0.40" is therefore implied by KL(C1 rule) ≤ 0.05 and adds nothing. Only the KL part is informative.
- P8(d) "uses E when available" measures sensitivity to mean-patching a subspace that the initialization already
  reads. It fails in P156 (0.028).

## 5. Learning dynamics (P7)

**5.1 [high] "No readout lag" is close to true by definition of use_hyb, and is contradicted under the natural
reading of readout lag.**

- In `dyn_mech.py`, use_hyb (line 67) is KL(P(xAP) ‖ P(layer 2 on xAP − dd)). Emergence (line 75) is ‖dd‖ ratio ≥ 2.
  Both come from the same dd.
- Once ‖dd‖ jumps (e.g. P150: 1.0 → 6.5 between steps 366 and 546), any readout sensitive to resid2 shows use.
- The CTRL subtraction helps: CTRL dd of norm about 3 gives 0.002–0.05. So the sensitivity is partly
  direction-specific.
- But "use" measures *any* decision change, not *correct* (Bayes-directed) use.
- At use onset, every fresh model is still **A1-closest on D2** (`traj_alg_xii.json`). The D2 → B switch comes 1–5
  checkpoints later: P154 366 → 1211; P153 813 → 2686; P156 366 → 2686.
- In the sense "the representation exists but the decision does not yet use it correctly", a lag is present. The
  documents relabel it "A2 → B calibration, unexplained".
- The checkpoint grid is geometric (×1.5), so "±1 checkpoint" means ±50% of training time.

**5.2 [medium] The "readout gain falls" argument is a saturation artifact.**

- gain = use_hyb / ‖dd‖². KL is bounded and saturates while ‖dd‖ keeps growing (3–15).
- GEN lines 95–97 ("This rules out 'the readout learns to use an already-present representation later'") overstates
  it.
- Also, CTRL use is not "≤ 0.01 throughout" (GEN line 104) in the fresh cohort: vP152 has CTRL_use_hyb 0.047 at
  step 366.

**5.3 [low] Hybrid measure chosen.**

- P7 uses the class-III hybrid (`use_hyb`), although a natural version (`use_nat`) was computed.
- The two agree in onset in the fresh data. Report both.

## 6. Novelty and alternative explanations

**6.1 [medium] These are standard results and should not be presented as new phenomena.**

- Softmax normalization creating interactions from additive scores, and bilinear pattern × value cross terms. This is
  generic to attention.
- Reference-dependence of interaction attributions: Shapley / Integrated-Gradients baselines, order dependence. The
  caution is right but not a finding of this investigation.
- Redundant and compensating routes, and the limits of single ablations: e.g. McGrath et al. 2023, "hydra effect".
- Mean- vs resample- vs path-patching sensitivity: Zhang & Nanda 2023; Heimersheim & Nanda; Wang et al. 2022 (IOI
  path patching).
- Causal scrubbing with value-matched donors: Chan et al. 2022.
- DAS and subspace patching, and their illusions: Geiger et al. 2023/24; Makelov et al. 2023.
- IIA inflation and the dependence of identification on the pair distribution.
- Representation and behaviour emerging together at a phase transition: induction heads (Olsson et al.); progress
  measures (Nanda et al. 2023).
- Retraining the top of a network on frozen features recovers a new task: last-layer retraining / probing
  literature.
- The elementary-symmetric-polynomial form of the colluder posterior: elementary combinatorics.

CAUSAL §5 item 9 already concedes much of this. NATIVE §2.3 ("cautionary methodological finding of this
investigation") and GEN §1.2 should be aligned with it.

**6.2 Task-specific contributions that do stand:**
- the shared-copy posterior correction;
- the measured δ and preservation power;
- the discriminating-pair design, with the caveat in §2.5;
- the CTRL-controlled factorial locus result (P1), with the caveat in §1.6;
- the falsification of sub-head redundancy as a general property;
- the finding that mechanism routing differs across independently pretrained parents.

**6.3 Alternative explanations not ruled out:**
- (a) count-curvature instead of conjunction (§1.6);
- (b) path_inA ≈ "relay the xA input" (§2.1);
- (c) DAS-subspace illusion for the mediation claims (§2.4);
- (d) partial survival of eligibility under layer-1 mean ablation, i.e. a possible backup route, masked by the biased
  side criterion (§2.2);
- (e) P8 repair from raw residual information (§4.1).

---

## Prioritized required corrections to the documents

1. **FRESH line 270 and the verdict table.** Relabel P6-as-coded as "non-diagnostic (passed by a null intervention:
   base_E 0.85–0.89, base_A1 0.08–0.10)". Mark P6(VS) as non-diagnostic, and note that this was known from the
   development control committed in `5425317` before the confirmation outputs.
2. **NATIVE §7 (lines 207, 211–222) and FRESH line 279.** Remove "in every fresh parent", "in none of the fresh ones"
   and "invariant across all 3 + 4 fresh parents". Report:
   - P3(b) is 2/4 in the incomplete parents;
   - P156 is value-route dominant and MLP1-formed (attention share 0.05, own MLP1 nonlinearity 0.33);
   - P8(d) fails in P156.
   State that routing is heterogeneous across parents.
3. **STAGE0_AMENDMENT_1.md lines 3–5 and FRESH Deviation 1.** Correct the statement that no T4 evaluation output
   existed: training-curve logs with eval KL/acc/E-diagnostics, and 22 step checkpoints for vP150/vP151, were
   committed in `5425317`. Also state that the amendment reverses the frozen reserve rule (`STAGE0` line 14), and
   list every file the commit changed.
4. **FRESH lines 305–311.** Reinterpret P3(b): the value route carries about 40–60% of the shift when patterns are
   removed. This is asymmetric partial redundancy, not "not routed redundantly".
5. **GEN §1.2 Findings 1–2 and the Interpretation (lines 51–68), and the P8 wording.** Replace "readout-only" with
   "layer-2 (full block) retraining". Delete "failure entirely downstream of layer 1", "linearly available" and
   "obstacle is computational in the aggregation stage". Say "bypasses the 6-dim DAS subspace", not "bypasses E".
   Lead with the fresh-layer-2 capacity result.
6. **GEN §2 (lines 95–104, 135–137) and FRESH P7.** State that "use" is decision sensitivity to removing dd, not
   correct use. Report that D2 stays A1-closest at use onset, with a 1–5 checkpoint delay to B. Drop "rules out" and
   the gain-falls argument. Correct "CTRL use ≤ 0.01".
7. **NATIVE §1 (lines 32–50) and CAUSAL §4 class II.** State that path_inA ≡ resamp_A (one design, not two). Report
   KL(Bayes(xA) ‖ cf_e0) ≈ 0.02, so path_inA is non-discriminating, and present path_inP and the CTRL path contrast
   (0.04–0.05 vs 0.55–0.73) as the evidence. Classify subspace scrubbing and restoration as synthetic at the patched
   site. Describe the scrub/restore/P6 results as generalization of the DAS objective, not independent mediation
   evidence.
8. **All "Bayes side" readings** (P2(e), P3(b), the NATIVE line 44 mean-ablation sentence). Disclose the reference
   entropy asymmetry: H(cf_e0) 0.21 vs H(B) 0.58, and a 50/50 mixture scores pref_e0 = 0.02. Report mean_L1
   pref_e0 0.06–0.18 and flag possible residual eligibility under layer-1 removal as unexamined.
9. **NATIVE §2.2–2.3 (lines 90–106).** Mark the softmax-coupling and "not multiplicative QK" conclusions as based on
   the x00-reference split, where the value term is ≈ 0 by construction. Remove "intervention-level facts are
   invariant". Provide the script that produced the "values fixed at xAP" numbers, or remove them.
10. **NATIVE §2.1 (lines 67–77) and FRESH P3(a).** Restrict the claim to "q7-token queries are irrelevant". The
    keys-only variant also feeds MLP1 and the non-q7 queries from xA.
11. **The 94/97 headline (FRESH line 261).** Give the breakdown: 16 of the 94 passes are non-evidential (P6-coded a,
    P6 VS, P5 e). Note that the checks are not independent.
12. **CAUSAL lines 196, 218 and 256.** Update the stale statements: the random-subspace control *was* run, and
    showed the VS criterion is non-diagnostic. Within-layer-1 redundancy is development-only and falsified in the
    fresh complete parents.
13. **NATIVE line 63.** Restrict the hybrid-artifact rebuttal to the 8 development models; the fresh P4 used the
    hybrid ablation. Report rec_complement next to rec_E in FRESH P4.
14. **Pooling language** (NATIVE line 221, GEN line 135). Do not pool development models with confirmation models as
    evidence. Add the parent/DAS-fit variance caveat to the bootstrap CIs. Give P5 gains with CIs, and note the
    small denominators and the cases where the mean patch beats the mapped source.
15. **Minor.**
    - Log the `disc_ident_c1conf_A/B` runs in COMPUTE_LOG.
    - Fix the `restore.py` docstring (normalization by `restore_all`).
    - Disclose the reused rngs (77_000_000 donor pool, 1104).
    - Reconcile 0.049/0.047 and 4.6/4.4.
    - Reframe the "relational conjunction" as "a nonlinearity in the all-question agreement indicator", given that
      e_h does not depend on q.
