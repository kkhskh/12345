# R3: novelty and baselines, counterexample construction, learning dynamics (adversarial review)

**Scope.** I read the revised package after `ADJUDICATION.md`: the atlas, GENERALIZATION, BASELINES, PROSPECTIVE, EQUIVALENCE and FAILURE_MATRIX documents; `scripts/run_training.py`, `src/train_ext.py`, `checkpoints/*.json|log`, `logs/pp1_sign_frozen1500.json`, `logs/pp2*.json`, and `logs/mech/{meanabl,alltok,fixr2}_s*.json`. I do not repeat findings from R1 or R2.

**My checks.** Small CPU scripts in `/tmp/r3/` (not preserved; each under 5 minutes):
- `null_pp1.py`: a noisy-Bayes null for PP1, plus an accuracy-preserving warp;
- `drift.py`, `drift2.py`: weight drift in PP3;
- `own_perm.py`, `own_perm_t4.py`: own-question Bayes when the pairing is unknown;
- `u2dec.py`, `u2ex.py`: decision-level search inside the domain of U2.

Numbers I computed are marked [R3].

**Training status at 23:39 UTC.**

| run | status | result |
|---|---|---|
| `pp4_T4perm` | done | acc 0.5528 |
| `pp3_T5` | done | 0.5756, fail |
| `pp3_scratch` | running | — |
| `pp3_T2` | queued | — |

GENERALIZATION §1a is still empty, although the atlas table already lists T5 as failed.

**Severity counts.** Fatal 0, major 7, minor 6.

---

### 1. E2, the "implementation-independent decision law", is implied by near-Bayes log-probabilities. It is not a separate rung or law (major)

**Claim attacked.**
- PP1/E2 is a "genuine implementation-independent causal characterisation".
- It is novelty #1.
- H3 "survives at E2".
- E2 is "strictly finer" than E1.

**Evidence.**
- **Every PP1 model is within 0.0007–0.0019 nats of Bayes NLL.** Transformer curriculum NLL is 0.5412–0.5424 and DeepSets 0.5418–0.5421, against Bayes 0.5405 (checkpoint JSONs).
- **A noisy-Bayes null with no structure reproduces PP1** [R3, 300 base episodes, the PP1 edits and filter]. I took T4-Bayes margins and added iid Gaussian noise at the model's own natural margin-error scale (sd 0.164–0.175).
  - The null scores 0.999 on all scored edits and 1.000 on boundary edits.
  - The observed models score 0.998 / 0.985 (seed 0) and 0.9998 / 0.9986 (seed 12).
  - So any model with margin RMS error ≲ 0.25 passes, whatever its implementation. The |Δm_Bayes| > 0.5 filter does the rest.
- **Accuracy alone does not imply E2: a counterexample** [R3]. Scale each question's Bayes log-probabilities by τ = 1/(m² + 0.05), where m is the Bayes top-two margin.
  - Every argmax is unchanged, so accuracy equals Bayes exactly. That is a stronger E1 than any real model.
  - Yet PP1 sign agreement falls to 0.25 overall, and the model fails PP1.
  - E1 as the ladder defines it (accuracy plus argmax agreement) therefore does not imply E2. E1 measured in KL does.

**Consequence.**
- E2 is "cross-entropy training converged to the posterior predictive on the support". That is a property of the objective, not of the implementations.
- It is the standard prior-fitted-network result: Müller et al., ICLR 2022, "Transformers Can Do Bayesian Inference", [arXiv:2112.10510](https://arxiv.org/abs/2112.10510v4).
- "Tested across heterogeneous circuits and survived" carries no extra information once the NLL gaps are known.

**Fix.**
- Redefine E1 by KL to Bayes.
- State E2 as a corollary: small KL on the edit distribution implies sign agreement for |Δm| > 2ε.
- Report the noisy-Bayes null next to PP1.
- Remove E2 from "what remains new", or keep it only as "verified calibration on the edit neighbourhood".
- **Decisive test that could rescue it:** find a model whose KL is ≤ 0.002 but which fails PP1. If none exists, E2 is not a separate equivalence class.

### 2. E3, "behavioural equivalence does not fix extrapolation", is a known phenomenon, and it does not count against H1 (major)

**Claim attacked.** E3 is novelty #2. It is "the only result H1 does not predict", and it "counts against H1 as a complete account".

**Evidence.**
- **The p_c = 0 shift is off the T4 support.** T1 episodes typically have no fully agreeing group, so they have zero probability under T4. Cross-entropy places no constraint there.
- **Seed and architecture divergence on zero-probability inputs is underspecification.**
  - McCoy, Min & Linzen 2020, ["BERTs of a feather do not generalize together"](https://arxiv.org/abs/1911.02969v2): models with the same in-distribution accuracy but different seeds generalise very differently out of distribution.
  - D'Amour et al. 2020, "Underspecification…" (cited from memory).
  - Architecture-dependent extrapolation: Xu et al., ICLR 2021, "How neural networks extrapolate" (cited from memory).
- **H1 makes no prediction off the support.** What PP2 falsified is "H1-strong": a plug-in extension whose ρ_plug R1-f already called arbitrary. Failing an arbitrary extension is not evidence against H1.

**Fix.**
- Reclassify E3 as "an instance of underspecification, with an organised post hoc index".
- Delete "counts against H1".
- The only novel route is OP-1: predict E3 from internals on held-out seeds, prospectively.

### 3. The novelty audit omits directly relevant prior work (major; mandate §14)

**The reversal at full agreement and its phase boundary.** Gunn et al. 2016, "Too good to be true: when overwhelming evidence fails to convince", Proc. R. Soc. A ([summary](https://sciencedaily.com/releases/2016/01/160112102022.htm)).
- It is the "paradox of unanimity": a small probability of systematic failure makes unanimous agreement *lower* confidence beyond a point.
- That is the qualitative content of the B2 boundary "even p_c = 0.05 reverses". Cite it next to Dong 2009.

**Correlated sources, which the audit says it did not search.** Pochampally, Das Sarma, Dong, Meliou & Srivastava, SIGMOD 2014, ["Fusing data with correlations"](https://lunadong.com/publication/fusionWCorr_sigmod.pdf). Also Clemen & Winkler 1985 on dependent experts (cited from memory).

**E1 and E2.** The PFN result (#1 above).

**I5 (exact head split).** Function-preserving unit splitting is Net2Net's Net2WiderNet (Chen, Goodfellow & Shlens, ICLR 2016; from memory). The construction is not new; only its use against a criterion is.

**Off-manifold subspace patching** (R2-a's finding on the tag interchange). Makelov, Lange & Nanda, ICLR 2024, ["Is this the subspace you are looking for?"](https://arxiv.org/pdf/2311.17030). It should be cited where the tag-interchange IIA is still used as sufficiency evidence.

**The PP3 design** (reset early vs late layers) is the Yosinski et al. 2014 transfer protocol (from memory). The finding that "lower layers carry the transferable features" is the generic outcome of that protocol, so novelty #4 is weak.

**Fix.** Add these papers, and downgrade novelty #1, #2 and #4 accordingly.

### 4. Corollary U2 holds, but the atlas map reads it as a decision-level "help". That reading has counterexamples inside U2's domain (minor)

**Claim attacked.** Atlas §0/§3: "colluders whose agreement rate is at most the honest rate (Corollary U2)" are a regime where relations "help (monotone)".

**Evidence** [R3]. I raised group g's count by one, with g's pair agreeing on candidate c in X_q.
- At α = 0.8, κ = 0.3, ρ = 0.193 (a_coll 0.462 ≤ a_rel 0.762), p_c = 1:
  - P(g reliable) never decreases, so U2 holds;
  - but the margin of g's candidate *decreases* in 0.3% of contexts (min Δm −0.019).
- At ρ = 0.95·ρ* it decreases in 0.5% of contexts.
- **Explicit case.** The contexts are in the `u2ex.py` output.
  - P(g0 reliable) rises from 0.0003 to 0.0017.
  - c's margin falls by 0.021–0.023.
  - Mechanism: a_coll > a_noise, so the count also raises P(g0 colluding), which shifts configuration mass away from configurations in which another group supporting c is the reliable one.
- No violations were found at κ = 0, nor at α = 0.6.

**Fix.**
- State U2 as a role-posterior result only.
- In §3, write "P(rel) is monotone; the decision effect still goes through Theorem D and can reverse slightly (κ > 0)".
- Add the counterexample to the THEOREMS page.

### 5. "The routing operation is the invariant object" names a modal implementation, not an invariant (minor)

**Evidence.**
- It fails in 1 of 7 seeds (seed 12) and does not exist in DeepSets. An object absent from a member of the E2 class is not an invariant of that class.
- R2-j already shows it is not function-level invariant.

**Fix.** Say "the most common relation-carrying site among the Transformer seeds".

**Decisive test.** Is the head-summed readout group-mass the *same function* of the input across seeds? Regress it on π^{−q}(g reliable) per seed and compare the fitted maps across seeds. Better still, predict its presence or absence for held-out seeds from the T1 stage.

### 6. The partner-resample "relation-specific share ≈ 0.21" is not a share (minor; new data from `fixr2_s*.json`)

**Evidence.**
- `acc_partner_resample` is 0.531–0.544, *below* own-question Bayes (0.578).
- The drop (0.204–0.217) exceeds the whole cross-question gain (0.167–0.171), which is 120–127% of the gain.
- So partner resampling injects another episode's confident reliability information: it is a substitution, not a removal. `meanabl` "resample_ablation" shows the same pattern (drop_frac 1.06–1.24).

**Fix.** Report it as "a misleading-context substitution effect" and drop the word "share". Natural necessity should rest on mean or conditional-mean ablation only.

### 7. PP3: arm (iv) is effectively from scratch, so HC-relation is not separated from weaker alternatives. The [C] label is premature (major)

**Evidence** [R3, weight drift relative to the fresh init].
- **Arm (iv).** The fresh layers 0–1 moved only **8–12%** in 3000 steps. In arm (iii), the fresh last layer moved **23–31%**.
  - Arm (iv) therefore did not learn early features at all. It sits at the own-question plateau (0.552 vs 0.577), like T0 (0.575), T5 (0.576), from-scratch T4 (0.578, 6k–24k steps) and every failed arm.
  - (iv) is a re-run of the barrier, not a test of the T1 routing layer. HC-routing's (iv) prediction was a straw man: routing on random keys cannot work under any hypothesis.
- **Arm (iii).** Loss is 0.62 at step 300, against 0.598 for the full T1 curriculum (posthoc s0). The T1 last layer contributes almost nothing. That much is solid.

**Alternatives that are not excluded.** T1's early layers carry at least three things T0 and T5 lack:
- (a) within-question pair-agreement detectors (group parsing);
- (b) cross-question same-group attention, which is a structural scaffold;
- (c) the count or tag itself.

HC-relation as stated is (c). Neither T0 (group 0 hard-coded) nor T5 (majority) needs (a) or (b), so (i) and (ii) are not "matched alternative pretraining tasks" (mandate H1).

**Other limits.**
- n = 1, and T1 s0 is a tag-class seed. Split-class seeds (T1 s2) are untested.
- Arms (v) and (vi) are unfinished, yet the atlas already writes "[C for HC-relation]".

**Decisive tests.**
1. Reset layer 0 only, and layer 1 only.
2. Pretrain on a "T1-local" task (R resampled per question), where (a) is useful but (b) and (c) are not. If it transfers, the benefit is within-question parsing.
3. Run (iii) and (iv) on T1 seeds 1 and 2.
4. Freeze the transferred early layers in (iii). If T4 is still reached, the features themselves, not the init basin, carry the benefit.
5. Run an LR sweep for (iv), because the fresh early layers barely moved.

**Change now.** Label PP3 "[partial; 4/6 arms]" until (v) and (vi) finish.

### 8. Optimiser state and data order are not differential confounds in PP3 (minor, positive)

**Verified.**
- `train_ext.train` builds a fresh AdamW and the same warmup-cosine schedule for every arm.
- Its data RNG formula equals `nets.train` for T4 seed 0, so all arms and the original curriculum see an identical stream. The recipe matches `train_curriculum.py`, apart from the unused WR seed.
- So optimiser-state or data-order artefacts cannot explain (iii) vs (iv).

**Caveats.**
- The one shared LR treats fresh and pretrained layers alike (see #7).
- Arm (iii) shares its stream and its T1 init with the original s0 curriculum, so it is not an independent replicate of that success.

### 9. "The mechanism class survives fine-tuning" is close to trivial under lazy fine-tuning (minor)

**Evidence** [R3].
- In the full T1→T4 curriculum (s0), layer weights move only 8–16% from the T1 weights.
- T1 itself is 18–38% away from its random init.
- Fine-tuning stays in the T1 neighbourhood, so seed 2's split → split persistence is expected.

**Fix.** Report the drift, and do not use the persistence as evidence for HC-relation.

### 10. The T4 barrier hypothesis ("the monotone first step is harmful, so no gradient path builds the count") is contradicted by in-package data (major)

**Evidence.**
- **(a) DeepSets has no barrier.** `main/T4_M_K3_s{0,1,2}_deepsets.json` learn T4 *from scratch*: train loss 0.54–0.59 by step 1200, final 0.7425–0.743. It faces the same task, the same Bayes landscape and the same harmful monotone heuristic, and must also learn the within-question agreement detector in φ.
- **(b) `mlp_grouphist` fails from scratch (0.556) and succeeds with the curriculum.** Its per-group histograms expose within-question agreement almost linearly. The barrier is therefore architecture- or aggregation-dependent, not a property of T4's gradient field.
- **(c) T1 with permuted sources (PP4) also fails**, although there the monotone use of pairwise cross-question agreement *is* rewarded. A gradient path exists, and the model still does not take it (#11).

**Fix.** Withdraw "leading hypothesis" status, and restate the barrier as architecture-specific (attention-based cross-question aggregation without pooling).

**Decisive tests.**
- Train a from-scratch Transformer with explicit per-group leave-one-out count tokens. If it learns, the barrier is feature construction; if not, it is the gate.
- Train DeepSets with a frozen random φ.
- Measure the gradient alignment at init (OP-2).

### 11. PP4 uses the wrong own-question reference. The "regression below own" is an artefact (major)

**Evidence** [R3]. `evaluate()` computes `own` with the *known* pairing on unpermuted data, but the model sees permuted sources.
- **T1.** The correct own-question references when the pairing is unknown:

  | reference | accuracy | NLL |
  |---|---|---|
  | Bayes over the 15 pairs | 0.644–0.646 | 0.786–0.790 |
  | majority vote | 0.638 | — |

  The model scores **0.640**, with train loss ~0.77–0.79. It sits *at* the own-question level for unknown pairing. It is not below it, and it shows no "pure optimisation regression". The registered threshold is also mis-referenced: with the unknown-pairing references it would be 0.68, not 0.70. The verdict E-fail is unchanged.
- **T4 stage 2 (now complete).** Accuracy 0.553, against own-question unknown-pairing Bayes **0.543** (known pairing 0.576), with Bayes 0.743. That is about 0.01 above own: E-fail.

**Consequence.** PP4 shows the same signature as the T4 barrier and every failed PP3 arm: the within-question rule is learned, and no cross-question feature is. That contradicts "a separate phenomenon", and nothing supports "agreement of agreements is harder".

**Fix.**
- Add the unknown-pairing own-question Bayes to `evaluate`.
- Correct the PROSPECTIVE §2 PP4 text, GENERALIZATION §2 and atlas §5.
- Append the stage-2 result.

### 12. Novelty #3 is worded as if a preregistered test succeeded (minor)

**Evidence.** The preregistered PP5 failed (6.9% of argmaxes changed, above the allowed 2%). The exact construction (pp5b) is post hoc, and the split itself is Net2Net-style (#3).

**Fix.** Write "post hoc exact construction against a previously preregistered criterion (R3)".

### 13. Record consistency (minor)

**Issues.**
- The atlas §5 table lists T5 as failed, but GENERALIZATION §1 says "see §1a", and §1a is empty.
- PP3's frozen table has six arms. Two are pending (scratch is running; T2 is queued).

**Fix.** Append the T5 and PP4 stage-2 results now. Keep the HC verdict provisional until (v) and (vi) finish.

---

## Summary

| # | finding | severity |
|---|---|---|
| 1 | E2 is implied by KL ≤ 0.002, with a noisy-Bayes null; accuracy-preserving counterexample to E1 ⇒ E2 | major |
| 2 | E3 is underspecification (McCoy 2020); it is not evidence against H1 | major |
| 3 | missing prior work: Gunn 2016, PFN, Pochampally 2014, Net2Net, Makelov 2024, Yosinski 2014 | major |
| 4 | U2 decision-level counterexample at κ = 0.3 | minor |
| 5 | the routing pattern is modal, not invariant | minor |
| 6 | the partner-resample "share" is 120–127% of the gain (substitution) | minor |
| 7 | PP3 arm (iv) ≈ from scratch (fresh layers moved 8–12%); alternatives (a) and (b) untested; [C] premature | major |
| 8 | PP3 optimiser state and data stream verified identical across arms | minor (positive) |
| 9 | lazy fine-tuning (8–16% drift) makes class persistence trivial | minor |
| 10 | barrier hypothesis contradicted by DeepSets from scratch, `grouphist` and T1-perm | major |
| 11 | PP4 own-question reference wrong; the model sits at unknown-pairing own (0.640 vs 0.645); T4perm 0.553 vs 0.543 | major |
| 12 | novelty #3 wording | minor |
| 13 | §1a not appended; arms pending | minor |

**What survives.** These all stand, re-labelled as above:
- the Bayes-level theorems (S, U, U2 as a role-posterior statement, D, D2, I1, I2);
- PP1 as a calibration check;
- PP3 arm (iii) as evidence that the T1 last layer is unnecessary;
- the honest reporting of the failed predictions.

The package's distinctive contribution, beyond known Bayes, underspecification and transfer results, is currently empty. OP-1, done prospectively, is still the only route to one.
