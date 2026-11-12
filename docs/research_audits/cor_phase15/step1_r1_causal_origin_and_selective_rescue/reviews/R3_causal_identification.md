# Reviewer 3: causal identification (commit 76a9d60)

Read-only checks (scripts in /tmp, not in the package):
- (a) generator statistics of m;
- (b) class-balanced probes on yP90 `pre.npz`;
- (c) a before/after split of the R1 response in the 8 `C1_injm` confirmation models. This reproduces the reported injm
  Δm_R1 (+0.200) exactly.

## Findings

**1. HIGH. Rescue does not show that m was missing.** (SELECTIVE_RESCUE §4.4: "the missing piece is the cross-group
eligibility relation"; MAIN l.10)
- **The n-control gives each readout token all three groups' one-hot counts.** m computed from those counts alone, ignoring
  q-agreement, equals the true m in 97.0% of (episode, q, group) cells. So injn supplies m's ingredients, one nonlinearity
  away, at the same token, yet R1 gets worse (P4).
- **What injm vs injn actually contrasts** is precomputation plus the q-agreement bit. It does not contrast "relation
  available vs not".
- **The oracle reader had explicit counts and still failed R1 at 4000 steps.** Its success at about 16k steps points to how
  hard the interaction is to learn within budget, not to anything P-count-specific.
- **Correction.** Say "precomputed m is sufficient"; injection does not test H7's accessibility clause. Run the dropped
  `xT70 C1_injm` (in devqueue4) plus injm+n and n+q-agreement controls.

**2. HIGH. m is close to an oracle in the subset that decides R1.**
- **Bayes weights are W = RHO^n·m, with n accepted as accessible.** So m is the whole non-trivial factor.
- **In two-perfect decisions (8.1%, where every R1 pair lives),** weighting groups by [m>0] and mixing their q-report
  likelihoods matches the Bayes argmax 97.4% of the time. The separable exclusion rule manages 55.0%.
- **In repaired models the R1 response is carried by the injected feature.** m changes in 100% of R1 pairs. Switching only m
  gives +0.28; switching only z gives −0.08. This split is a hybrid, so it is indicative only.
- **This does not literally violate §9.4,** because m must still be combined with q-reports. But it is the hand-inserted
  Bayes factor.
- **Correction.** Say so explicitly. Classify the result under §9.4's last paragraph: learnability or sufficiency, not the
  original computation.

**3. MEDIUM. Was the cause chosen after seeing what worked?**
- **Sequence from the logs:** injection queue "motivated by oracle-reader result" at 22:29; dev injm results at 23:13 and
  23:27; theory commit `aaa5718` at 23:39; FREEZE `be4a0da` at 00:00:56; first confirmation run at 00:02:42.
- **Confirmation is genuinely prospective,** apart from one disclosed smoke run.
- **The feature itself is not post hoc.** It is the only feature tried, and it is defined by Proposition 1.
- **H7's mechanistic clause was written after the repair worked on two dev parents.** It was then tested only via P6, which
  is weak (see 6).
- **Correction.** Disclose this order.

**4. HIGH. The mandated baselines (§9.3) are mostly absent.**
- **The extra-training run is unfinished.** `checkpoints/dev/xP80_C1_r0_s12000.json` does not exist. The run started at
  06:09:51 and was at step 500 by 06:12, after the document commit at 06:10:53. MAIN's Outcome F row cites
  READOUT_AND_LAYER2_FACTORIAL §5, which does not exist.
- **Other baselines were never run:** a generic or random-site adapter, a low-rank update, logit recalibration, end-to-end
  training, and natural R1 upweighting (W_r1up was dropped).
- **Random injection is a zero-information control.** Because of the mod-7 hash it has at most 7 patterns per episode. It
  controls for an extra path, not for capacity to compute: a zero-initialised map can stay at zero. n (18 inputs against 9)
  is not capacity-starved.
- **Correction.** Downgrade E's "matched generic baselines" clause to outstanding. Remove the §5 citation. Add a
  parameter-matched, input-conditioned adapter on the frozen states.

**5. MEDIUM. W_bal is not "generic training".**
- **Its classes are defined by the relation under study** (H2/agree2), and its weights are extreme: ×11.9 and ×6.0 on the
  two-perfect classes, ×0.36 on the majority.
- **It had no development run before the freeze,** whereas injm was developed on three parents.
- **It raises E_R1 more than injm does** (+0.90 vs +0.70).
- **Correction.** Write "one extreme reweighting is not selective". Do not claim that generic training fails.

**6. MEDIUM. The probes are not evidence of absence.**
- **P6 was nearly guaranteed to pass.** It requires accuracy < 0.96 with a majority rate of 0.912, using unweighted
  logistic regression.
- **Balanced re-probe on yP90.** H2 has AUC 0.73. The per-group perfect indicator reaches AUC 0.995 / 0.997 / 0.925 at the
  q-sources.
- **The package's own MLP probe** decodes agree2 at 0.94 against a majority rate of 0.67.
- **Correction.** Report AUC. Reword to "not linearly precomputed at one token".

**7. MEDIUM. Hybrid patches support a headline claim.**
- **P7 (`conf_mech.py`) is a hybrid:** the after-edit readout token with before-edit other tokens.
- **MAIN uses it without caveat.** It calls P7 "confirmed" and says "contributes −0.74 (8/8)". −0.74 is the development
  Möbius term; the confirmation hybrid mean is −0.535.
- **The split is non-additive.** The readout-only effect exceeds the total in 6/8 parents, and the residuals are 0.04–0.20.
- **Correction.** Caveat both statements in MAIN and do not cite P7 as mediator evidence.

**8. MEDIUM. Redundant pathways remain.**
- **The route is not necessary.** `_rtemb` leaves R1 wrong (acknowledged).
- **The injection site is the route's own site.** No q-source injection was tested.
- **Repaired models reorganise globally.** The frozen-state contribution falls from −0.50 to −0.08. The rescue is not a
  localised intervention on a mediator.

**9. MEDIUM. Overclaims relative to D vs E.**
- MAIN Outcome E says "The rescue is explained by Theorem 1 and the accessibility results". This contradicts §9.5 and the
  row's own disclaimer, and findings 1 and 6 weaken the accessibility leg.
- Outcome A is budget-relative. Write "inaccessible to this learner within 4000 steps".

## Restricted results defended

- **The protocol is sound.** The freeze precedes confirmation, the analysis is intention-to-train, inference is model-level
  by exact sign-flip, and the lineages are new. P1, P2, P4 and P5 hold.
- **P4 (injn worsens R1 while improving KL) is a risky, confirmed prediction.** It is strong evidence that average-quality
  pressure selects a separable discount. Keep it central, framed as a claim about learned solutions.
- **§9.4 is respected literally.** m is a deterministic function of X, and training used labels only.
- **Selectivity against the comparators actually run is real:** 7/8 against 0–1/8, with R2/R3, KL and accuracy improved.
- **"D not reached" and "no unique circuit" are honest.**

## Verdict

**What stands:** a prospectively confirmed, selective **behavioural** rescue. Supplying a hand-computed, generator-derived
Bayes eligibility factor at the readout tokens corrects R1 on 8 independent lineages. It beats a no-information feature, a
separable-count feature and one extreme reweighting.

**What is not established:**
- that the original models failed because m was inaccessible (1, 2, 6);
- that generic baselines fail (4, 5);
- any causal mediator (7, 8).

**Outcomes:** E holds as a behavioural rescue only, with the baseline clause open. D is not reached. G holds at the
mechanism level. F is unresolved. Revise per 1, 2, 4, 5 and 9 before release.
