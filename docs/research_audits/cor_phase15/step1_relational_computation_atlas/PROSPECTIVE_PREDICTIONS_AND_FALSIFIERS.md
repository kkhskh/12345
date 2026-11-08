# Prospective predictions and falsifiers

**Freeze point.** The commit that first adds this file. At that point none of the experiments in §1 had been run on any model, and none of the evaluation streams named below had been generated.

**After the freeze.** Results are appended in §2. The predictions and thresholds in §1 are never edited after the freeze. Failed predictions stay in this file.

**Code.** `src/gen_family.py` is the shared generator and exact-Bayes reference. It was checked against the latent-reliability package's `bayes.py`: the maximum |Δ posterior| is ≤ 1e−14 on T1 and on T4.

## 1. Frozen predictions

### PP1: implementation-independent decision response (J4)

**Question.** Does one formula predict the *direction* of candidate-margin change for natural source edits in every T4-capable model, whatever its internal implementation?

**Models.** Every saved model that learned T4:

| model | implementation | evidence |
|---|---|---|
| Transformer curriculum seeds 0, 1, 2 | post hoc | |
| Transformer curriculum seeds 10, 11, 13 | tag-and-route | confirmed |
| Transformer curriculum seed 12 | direct reread | |
| DeepSets T4 seeds 0, 1, 2 | explicit pool | |

**Data.** T4 generator, stream base 6,100,000, 1500 episodes.

**Edits.** For each base episode, target question q, another question k ≠ q and a group g:
- if the group disagrees in k, set its second source's report to the first source's (this creates agreement);
- otherwise, set its second source to (first + 1) mod K (this breaks agreement).

**Margin.** m = L_{c1} − L_{c2}, where c1 and c2 are the exact-Bayes top-two candidates at q on the base episode. L is the model's log-probability.

**Prediction.** Two levels, for edits with |Δm_Bayes| > 0.5:
- **Overall:** for every model, sign(Δm_model) = sign(Δm_Bayes(T4)) on ≥ 90% of these edits.
- **Boundary edits:** these are edits that move the group's leave-one-out count between 4 and 5, the perfect-agreement boundary. On them:
  - sign agreement with T4-Bayes is ≥ 80%;
  - sign agreement with the *monotone* plug-in (T1-Bayes, which treats agreement as always good) is ≤ 50%.

**What it discriminates.**
- **For H3/H1** (an implementation-independent causal characterization at the level of the Bayes decision function): pass in all models, including seed 12 and DeepSets.
- **Against H3:** a failure in some implementation means the decision-level abstraction does not survive implementation changes.

### PP2: does the learned computation implement *training-distribution* Bayes? (H1 strong form vs an adaptive relational operation)

**Models.** The 7 Transformer T4 curriculum models, and DeepSets T4 seeds 0–2.

**Shifted generators.** Never seen in training; streams 6,200,000 and up, 6000 episodes each:
- (a) partial collusion ρ = 0.9;
- (b) partial collusion ρ = 0.7;
- (c) no colluder at all (pc = 0, which is T1 data).

**References.**
- **True Bayes:** the shifted generator's exact posterior.
- **Plug-in Bayes:** the training generator's posterior, with pc = 1 and ρ = 0.999. The value 0.999 is used instead of 1 so it stays defined.

**Prediction (H1-strong).** On questions where the two references' argmaxes differ, every model's argmax equals the plug-in's in ≥ 70% of cases, in each shifted generator.

**Alternative.**
- **(H2-adaptive):** models side with true Bayes, in ≥ 50% of those cases.
- **Falsifier of H1-strong:** < 70% agreement with the plug-in in any shifted generator, for a majority of models.

### PP3: what the T1 stage contributes to T4 learning (H1 curriculum controls)

**Fine-tuning recipe.** Fine-tune on T4 for 3000 steps, using exactly the `train_curriculum.py` recipe, from each of these starting points. All are M size, one seed, and existing checkpoints unless stated:

| arm | starting point |
|---|---|
| (i) | T0 s0: within-question classifier with a known reliable group; no relation use |
| (ii) | T5 s0: all sources exchangeable |
| (iii) | T1 s0 with the **last layer** re-initialised (attention and MLP of layer 2) |
| (iv) | T1 s0 with **layers 0–1** re-initialised, last layer kept |
| (v) | T2 s0: relation-using, with an extra confidence head |
| (vi) | random initialisation (from scratch), same 3000-step schedule |

**Success.** T4 accuracy ≥ own + 0.5·(Bayes − own) on stream 6,300,000 (20,000 episodes).

**Competing predictions:**

| hypothesis | (i) T0 | (ii) T5 | (iii) T1, last layer reset | (iv) T1, layers 0–1 reset | (v) T2 | (vi) scratch |
|---|---|---|---|---|---|---|
| HC-relation: T1 teaches reusable early-layer relation features | fail | fail | succeed | fail | succeed | fail |
| HC-routing: the T1 last-layer routing scaffold is what matters | fail | fail | fail | succeed | succeed | fail |
| HC-generic: any trained initialisation helps | succeed | succeed | succeed | succeed | succeed | fail |
| HC-classifier: the within-question classifier is what matters | succeed | succeed | ? | ? | succeed | fail |

**Limits.** One seed per arm, so this is a coarse discrimination. Any arm whose outcome is ambiguous (accuracy within 0.02 of the threshold) is reported as ambiguous.

### PP4: unknown source groups (Direction D; exploratory, not confirmatory)

**Bayes level (no model).**
- The pairing of reliable and colluding sources is identifiable from an episode.
- The pairing among noise sources is not identifiable (exact symmetry; see the theorem).
- Exact Bayes accuracy with the pairing unknown is ≥ own-question Bayes and < known-pairing Bayes. The gap is computed exactly.

**Learned level.** Train M Transformers on T1 with a random per-episode source permutation (seed 20, 6000 steps), then T4 by curriculum (3000 steps).

**Competing expectations, registered but not scored as confirmatory:**

| expectation | outcome |
|---|---|
| E-learn | ≥ own + 0.5·gap |
| E-fail | < own + 0.5·gap |
| E-mech-tag | a source-token key tag with the replication protocol's R2 key-only IIA ≥ 0.7 |
| E-mech-reread | key-only IIA < 0.7, with the cross-question effect entering inside the last layer |

### PP5: head-count non-identifiability (F1/J5; constructive)

**Prediction.** Splitting one route head of replication seed 11 into two copies, each with half the OV weight, overwriting one non-route head:
- changes the model's T4 answer distribution by less than the overwritten head's own ablation effect;
- **changes the preregistered R3 outcome** from pass to fail, because the fixed top-2 freeze no longer covers all routing heads.

**What it shows if it passes.** Head-count-dependent criteria are not invariant under near-function-preserving reparameterisation. This is the formal version of the seed-10 failure.

---

## 2. Results (appended after running; §1 unchanged)

### PP1: PASSED (10/10 models)

**Deviation (review R1-c).** The first run used 500 base episodes, while the frozen text specifies 1500. The **frozen-specification re-run** is `logs/pp1_sign_frozen1500.json`:
- 47,005 scored edits; 6,747 scored boundary edits;
- **10/10 pass**;
- boundary sign agreement with T4-Bayes 0.971–0.999, with episode-cluster 95% CIs ≥ 0.96;
- with the T1 plug-in 0.133–0.158;
- unfiltered (no \|Δm\| filter; reported, not a criterion): boundary 0.79–0.93 vs 0.20–0.39 for the monotone plug-in.

The 500-episode numbers below are kept as the first run.

Source: `logs/pp1_sign.json`.

**Scored edits.**
- 15,855 scored edits, with |Δm_Bayes| > 0.5, out of 45,000.
- 13,800 edits are off support (Bayes undefined) and were excluded as stress tests.
- 2,457 scored boundary edits.

| models | sign agreement with T4-Bayes, all scored | at the boundary | at the boundary, with the T1 plug-in |
|---|---|---|---|
| Transformer curriculum seeds 0, 1, 2, 10, 11, 12, 13 | 0.994–1.000 | 0.964–1.000 | 0.131–0.166 |
| DeepSets seeds 0–2 | 0.998–0.999 | 0.993–0.997 | 0.160–0.163 |

- **Reference.** The T1 plug-in itself agrees with T4-Bayes in sign on 77% of all edits but only 16.6% of boundary edits. The boundary subset is therefore the discriminating one.
- **Correlation.** Model Δm correlates with Bayes Δm at r ≥ 0.989.

### PP2: H1-strong FALSIFIED as frozen, for the Transformers, on the p_c = 0 shift only

**Revision (review R1-e).** At ρ = 0.9 and 0.7, the plug-in and true-Bayes disagreements are near-ties (median log-gap 0.012–0.023), so those shifts cannot discriminate the hypotheses. The falsification rests on **p_c = 0**:
- 0/7 Transformers side with the plug-in in ≥ 0.70 of cases (range 0.19–0.64);
- 3/3 DeepSets do (0.75–0.89).

With the frozen "< n/2" reading, 3/10 models pass the plug-in threshold, so H1-strong is falsified for the majority. **ρ_plug = 0.999 is a choice** (review R1-f). The p_c = 0 result comes from the forced-colluder constraint and does not depend on it.

Source: `logs/pp2_shift.json`. The table counts the models (out of 10) that side with the training-distribution plug-in in ≥ 70% of the cases where the plug-in and the true Bayes answer disagree:

| shift | models siding with the plug-in |
|---|---|
| ρ = 0.9 | 0 / 10 |
| ρ = 0.7 | 2 / 10 (DeepSets s1, s2) |
| p_c = 0 (T1 data) | 3 / 10 (all DeepSets) |

- **Transformers.** They side with the plug-in in 19–64% of cases.
- **DeepSets** are much closer to H1-strong: 0.52–0.66 at ρ = 0.9, 0.56–0.72 at ρ = 0.7, and 0.75–0.89 on T1 data.
- **Extra, not frozen.** T1 Transformers side with *their* plug-in in 96–99% of cases under every shift. So H1-strong holds for the T1 models, and fails for the T4 Transformers.

**Post hoc follow-up (exploratory, not a confirmation).** Source: `scripts/pp2b_local_posthoc.py`, `logs/pp2b_local_posthoc.json`. The local factorised reference drops the coupling "exactly one reliable and one colluding group".
- **DeepSets** side with the global plug-in over the local one in 0.67–0.89 of disagreements, in every setting.
- **Transformers are seed-dependent off the support.** On T1 data, seeds 1, 10 and 11 side with the local reference (0.70–0.87), while seed 12 (the direct rereader) sides with the global one (0.61).
- **On the training distribution**, all models side with the global reference (0.69–0.80).

**Status.** The global/local axis is a **post hoc, continuous index** (side-rates 0.30–0.87; review R1-g), not two classes. The on-distribution range is 0.67–0.80, corrected from 0.69–0.80. It has not been confirmed on held-out seeds (open problem OP-1).

### PP3: PARTIAL (one seed per arm; review R3-7)

| arm | starting point | T4 accuracy (threshold 0.6595) | outcome |
|---|---|---|---|
| (i) | T0 | 0.575 | fail |
| (ii) | T5 | 0.576 | fail |
| (iii) | T1, last layer reset | 0.742 | **succeed** |
| (iv) | T1, layers 0–1 reset | 0.552 | fail |
| (vi) | scratch | 0.550 | fail |
| (v) | T2 | see `GENERALIZATION_AND_EMERGENCE.md` §1a | |

**Fit with the frozen hypotheses.**
- **Matches the frozen HC-relation column** in all completed arms.
- **Falsified:** HC-generic and HC-classifier.
- **HC-routing as frozen:** falsified, though by a weak arm, since (iv) effectively re-runs the barrier.

**Not separated.** HC-relation is not separated from "within-question pair parsing" or "cross-question scaffold". Those were not frozen alternatives; R3 raised them afterwards.

### PP4: the unknown-group models FAILED to learn the cross-question task (E-fail); re-baselined after review R3-11

**Correct reference** (`logs/pp4_own_unknown.json`): own-question Bayes with *unknown* pairing is 0.634 (T1) and 0.559 (T4). The models reach 0.640 and 0.553, which is the own-question level. Stage 2 (T4 via curriculum from the stage-1 model) also failed. The earlier text said 'below own-question'; that used the known-pairing baseline and is corrected here.

(Original text, with the wrong baseline, follows.)

| measure | value |
|---|---|
| T1 with permuted sources, M Transformer, seed 20, 6000 steps | 0.640 |
| own-question Bayes | 0.654 |
| exact Bayes, unknown pairing | 0.714 |
| registered threshold, using the known-groups Bayes | 0.700 |

- **Diagnosis.** The model did not even reach the own-question level, a regression that suggests a pure optimisation failure.
- **Stage 2** (T4, permuted) is reported in `GENERALIZATION_AND_EMERGENCE.md`.

### PP5: FAILED as frozen

Source: `logs/pp5_headsplit.json`. The criterion part held but the function-preservation part failed.
- **R3:** flipped from pass to fail (route-frozen IIA 0.21 → 0.58).
- **Function preservation:** missed. Overwriting the "inert" head changed 6.9% of argmaxes, above the 2% the frozen prediction allowed. Accuracy was unchanged (0.7406 vs 0.7405).

**Post hoc exact construction.** Source: `logs/pp5b_exact_split_posthoc.json`. A 5-head split that copies the head into an added head:
- outputs are identical (max |Δp| 2.4e−7);
- R3 flips from pass to fail.

This establishes I5 exactly. The failed near-preserving version remains a failed prediction.
