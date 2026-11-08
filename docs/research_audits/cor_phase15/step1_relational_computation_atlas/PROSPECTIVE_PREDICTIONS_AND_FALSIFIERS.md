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
