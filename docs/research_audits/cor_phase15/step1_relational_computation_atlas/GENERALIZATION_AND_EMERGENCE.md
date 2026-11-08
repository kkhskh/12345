# Generalisation and emergence (mandate H, D, C)

> **Revised after review R3** (see `ADJUDICATION.md`):
> - PP3 is **partial**; arm (iv) is effectively from scratch;
> - the barrier explanation is withdrawn;
> - PP4 is re-baselined.

**Status labels.**
- **[C]** prospective and confirmed.
- **[F]** prospective and failed.
- **[PH]** post hoc.
- **[N]** exact calculation.

## 1. Why the T1→T4 curriculum works (PP3, frozen)

**Setup.**
- Fine-tune on T4 for 3000 steps from each starting point, with the same recipe as `train_curriculum.py`.
- One seed per arm, so each arm is a coarse discrimination.
- Evaluation stream 6,300,000, 20,000 episodes.
- Threshold: own + 0.5·(Bayes − own) = 0.6595.

| arm | starting point | T4 accuracy | outcome | HC-relation | HC-routing | HC-generic | HC-classifier |
|---|---|---|---|---|---|---|---|
| (i) | T0 s0 (within-question classifier, known reliability) | 0.575 | **fail** | fail ✓ | fail ✓ | succeed ✗ | succeed ✗ |
| (iii) | T1 s0, **last layer re-initialised** | 0.742 | **succeed** | succeed ✓ | fail ✗ | succeed ✓ | ? |
| (iv) | T1 s0, **layers 0–1 re-initialised** (last layer kept) | 0.552 | **fail** | fail ✓ | succeed ✗ | succeed ✗ | ? |
| (ii) | T5 s0 (all sources exchangeable) | *see §1a* | | fail | fail | succeed | succeed |
| (v) | T2 s0 (T1 plus a confidence head) | *see §1a* | | succeed | succeed | succeed | succeed |
| (vi) | random init, 3000 steps | *see §1a* | | fail | fail | fail | fail |

**What T1 transfers (revised after R3-7).** The T1 **last layer is unnecessary** (arm iii), and the T1 **early layers are necessary** (arm iv).

Arm (iv)'s fresh early layers barely moved (8–12%, reviewer measurement), so it re-runs the barrier. It does not test the routing layer on learned features.

What the early layers carry is **not separated**. It could be:
- the count/tag;
- within-question pair parsing;
- a cross-question same-group scaffold.

T0 and T5 are not matched controls for the last two.
- With the early layers kept and the routing layer reset, T4 is learned to Bayes level.
- With the routing layer kept and the early layers reset, it is not.

**Hypotheses.**
- **Consistent:** HC-relation. **Falsified:** HC-generic, by arms (i) and (ii).
- **HC-routing as frozen** predicted (iv) succeeds and failed, but that prediction was weak (routing on random features cannot work), as R3 notes.

**Limits.**
- n = 1 per arm.
- The re-initialisation draws fresh random weights for the reset layers, so arm (iv) also tests whether a T1-trained last layer can work on *random* early features. It cannot.

**Mechanistic reading: withdrawn (R3-10).** The earlier explanation, "the monotone first step is harmful, so no gradient path builds the count", is contradicted by in-package data:
- DeepSets learn T4 from scratch;
- `mlp_grouphist` fails from scratch but succeeds with curriculum;
- T1 with permuted sources also fails, although the monotone step is rewarded there.

The barrier is **architecture-dependent**. Its cause is open (OP-2).

### 1a. Remaining arms

*(appended when complete)*

## 2. Unknown source groups (PP4; Direction D)

**Statistical level [N]** (`logs/bayes_boundaries.json`).

| task | known pairing | unknown pairing | own-question |
|---|---|---|---|
| T1 | 0.741 | 0.714 | 0.651 |
| T4 | 0.743 | 0.743 | 0.576 |

- In T4 the colluder's perfect agreement reveals the grouping.
- The noise sources' pairing is unidentifiable and irrelevant (I1).
- T1 and T4 both remain statistically learnable with unknown groups.

**Learned level [F for E-learn]** (re-baselined after R3-11; `logs/pp4_own_unknown.json`).

| stage | model | own-question Bayes with **unknown** pairing | majority vote | full Bayes with unknown pairing |
|---|---|---|---|---|
| T1 with permuted sources, from scratch (6000 steps) | **0.640** | 0.634 | 0.632 | 0.714 |
| T4 with permuted sources, curriculum from that model | **0.553** | 0.559 | 0.558 | 0.743 |

- **Reading.** The model learns the within-question evidence and nothing across questions. That is the same signature as the T4 from-scratch barrier, not a "regression below own" (the earlier claim used the wrong baseline).
- **This is a learnability failure, not a statistical impossibility.** The statistical, neural-learnability, representation and mechanism levels have to be kept separate here (mandate D). Only the first is settled.


**Interpretation.**
- With fixed groups, the source-identity embedding hands the model the pairing for free.
- With permuted sources, the model must discover the pairing *within* the episode from cross-question co-agreement. That is a harder, second-order relation: agreement of agreements.
- That the M model fails at this budget is consistent with the T4 from-scratch barrier, but is a separate phenomenon.

## 3. Mechanism selection: tag vs split vs reread

**Observed** (7 T4-curriculum seeds plus 3 T1 seeds from earlier packages).

| class | T4 seeds | T1 seeds |
|---|---|---|
| key-localised tag | 0, 1, 10, 11, 13 | 0, 1 |
| key/query split | 2 | 2 |
| last-layer rereading | 12 | — |

- **Persistence across fine-tuning [PH].** T4-curriculum seed 2 was initialised from T1 seed 2, which was already split. The mechanism class survives fine-tuning. Fine-tuning moves the weights only 8–16% (R3-9), so this is close to expected.
- **Exception.** Seed 12 was initialised from its own T1 stage, whose mechanism was not analysed. Whether T1 seed 12 was already a rereader is **unknown**.
- **No predictive account yet.** Five tag, one split and one reread is too few to estimate frequencies. The mandate's caution applies: these are not population rates.
- **Cheap prospective test (OP-1/OP-3).** Classify the T1-stage mechanism before fine-tuning, then predict the T4-stage class. This needs no new training for seeds 10–13, whose T1 checkpoints exist, but it was not run in this step.

## 4. Distribution shift, and what each implementation extrapolates

Source: PP2, p_c = 0 shift. This is the only discriminating shift (review R1-e).

| implementation | accuracy on T1 data (Bayes 0.746; global T4 plug-in 0.451; local plug-in 0.705) | sides with the global plug-in |
|---|---|---|
| DeepSets T4 | 0.386–0.433 | 0.72–0.89 |
| Transformer seed 12 (rereader) | 0.500 | 0.61 |
| Transformer seeds 0, 2, 13 | 0.566–0.593 | 0.38–0.46 |
| Transformer seeds 1, 10, 11 | 0.632–0.678 | 0.10–0.27 |

**Reading [PH, continuous index].**
- DeepSets, whose explicit pooled bottleneck is the Theorem-S factorisation, extrapolate like the global Bayes plug-in: they enforce "exactly one colluder" even when none exists.
- Transformers extrapolate more locally. They penalise only groups that individually look like colluders, which on T1 data is accidentally closer to the truth.
- **Practical consequence.** The "better" off-support behaviour of some Transformers is not better inference. It is a different, local inductive bias that happens to suit this shift.

**T1 models under shift.** They follow their own (monotone) plug-in in 96–99% of disagreements under every shift. They implement the training-distribution rule faithfully.

## 5. What was not examined (stated)

- Width, depth and head-count sweeps.
- L-size models.
- Optimiser and normalisation variants.
- Q, K and group-size variation in learned models.
- Multi-output pressure (T2 exists, but no controlled manipulation).
- Mixed-prevalence training (p_c < 1). The phase diagram predicts where the monotone first step stops being harmful; **untested**.
