# Cross-slot learning interventions (Experiment D)

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). **W8:** training on an *already-acquired* slot's loss alone produces the same collapse of the other slots (A3_s804 0.87 → 0.23; A2_s705 0.96 → 0.61) [expl, review B5]. Interference is generic single-slot forgetting under a fresh AdamW restart, and is not evidence of slot-specific maintenance. Acceleration was not testable (ceiling; the frozen checkpoints were 20–220 steps from natural acquisition). **W5:** the H4 reading is withdrawn; H2, H3 and H4 are not identified. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

**Question.** When one slot is pushed to learn, do the others benefit, stay unaffected, or suffer?

## 1. Interventions (`scripts/xvii_slot_train.py`)

All arms start from a partial checkpoint with a fresh AdamW state:
- learning rate 1e-3 after a 50-step warm-up;
- batch 256;
- a new data stream (61_000_000 + seed);
- per-slot F_G every 50–100 steps.

| arm | trainable parameters | loss |
|---|---|---|
| full | all | all six slots |
| targeted | all | one unacquired slot only (other slots weighted 0) |
| code-only | only that slot's code E_q[j] (all shared weights frozen) | all slots |
| code-only control | only an acquired slot's code | all slots |

**Gradients through shared weights.** A slot-restricted loss still updates every shared weight (mandate §8.3). The arms record how much the shared weights and every code move.

## 2. Development results on C_s202, from step 1000 [dev] (A = {0, 3, 4}, U = {1, 2, 5})

| arm | steps | F_G of slot 1 (targeted) | F_G of the acquired slots at the end | parameter change |
|---|---|---|---|---|
| code-only, slot 1 | 1200 | 0.00 throughout | 1.00, 1.00, 1.00 | code moved 1.98 (≈ 46% of its norm); shared weights 0 |
| code-only control, slot 0 | 300 | 0.00 | unchanged | code 0.56 |
| targeted, loss on slot 1 | 1000 | 0.00 throughout | **0.56, 0.23, 0.09** (from 1.00) | shared weights 5.09 |
| full | 1000 | **1.00** at 1000 (slots 1, 2, 5 = 1.00, 0.59, 1.00; slot 5 already 0.37 at 900) | 1.00, 1.00, 1.00 | — |

**Reading [dev]:**
1. **No code-only rescue.** Gradient descent on the unacquired slot's code alone does not find a code that engages the existing computation, even though one exists (an acquired slot's code engages it for that slot: CROSS_SLOT_CAUSAL_TRANSFER.md). That fits XVI's finding that the gate's first-order gradient signal is below detection on the plateau.
2. **No acceleration from targeted pressure.** Training with the loss on the unacquired slot alone did not make it acquire within 1000 steps. Full training made all three unacquired slots acquire after about 900–1000 steps, close to the original run's timing.
3. **Interference.** Targeted training removed gate use from the acquired slots, whose loss was switched off. Their competence is maintained by their own loss through the shared weights; it is not a free consequence of the slot-1 objective.
4. **The three unacquired slots acquire together** under full training (steps 900–1000), as in the original run (about 2000).

**Interpretation [spec].** The slots' competences are coupled through the shared weights, but in a constrained way:
- local pressure on one slot neither builds that slot's access nor preserves the others;
- what extends the computation to new codes is the joint objective.

This is consistent with H2's "shared computation, code-keyed access". It argues against "independent local learners" and against "one slot's success spills over to the others".

## 3. Frozen test in fresh models (PL)

- **Subset:** every evaluable arity-3 model and the first two evaluable arity-2 models.
- **Comparison:** targeted against full, 400 steps.
- **Expectation:** INTERFERENCE_NO_ACCELERATION.

## 4. Fresh-model results [frozen PL: INTERFERENCE_NO_ACCELERATION, 5/5; second half at the ceiling]

**Per slot group over 400 steps** (from `learning/*/slot_traj.json`):

| model | targeted arm: targeted slot / other unacquired / acquired, at step 400 | full arm (same groups) |
|---|---|---|
| A3_s802 | 1.00 / 0.07 / 0.19 (from 0.96) | 1.00 / 1.00 / 1.00 |
| A3_s803 | 1.00 / — / 0.11 (from 0.91) | 1.00 / — / 1.00 |
| A3_s804 | 1.00 / 0.02 / 0.25 (from 0.86) | 1.00 / 1.00 / 1.00 |
| A2_s704 | 1.00 / — / 0.28 (from 0.88) | 1.00 / — / 1.00 |
| A2_s705 | 1.00 / 0.04 / 0.29 (from 0.94) | 1.00 / 1.00 / 1.00 |

**Answers.**
- **Q16 [frozen + expl]:** improving one slot does **not** help the others. Under targeted training, the other unacquired slots stay at 0.02–0.07 while the targeted slot reaches 1.00, and the acquired slots lose most of their gate use.
- **Q17:** *[withdrawn, W8: not testable here, ceiling and imminence]* yes, targeted pressure accelerates one slot without training the others. In fresh models the targeted slot acquired the gate within 100–200 steps, as fast as under full training (a ceiling effect). In the long C_s202 window [dev], it did not acquire in 1000 steps.
- **Q18:** *[withdrawn, W5, W8]* the pattern supports **position-conditioned specialisation of learning and maintenance**, coexisting with a **computation that is reusable by code exchange** (CROSS_SLOT_CAUSAL_TRANSFER.md). That is H4, a mixture. It is neither pure H1 nor pure H3.
