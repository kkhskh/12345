# Cross-family confirmation: the frozen prediction X1 and its outcome

**Freezes.**
- FREEZE X: commit `5d3c9dd`.
- FREEZE X amendment: commit `b019ca9`. It was registered after mathematical review A and before any X1 output was inspected.
- Scoring commit: `1ec55aa`.

**Disclosure (review C, finding 1).** The training scripts' loss logs (`logs/g_train_s3.out`, `logs/g_train_s4.out`) were written during training and partly committed with FREEZE X and the amendment. Before the amendment, the investigator read only the final line of the s3 log (step 4000, loss 0.1807), which bears on competence. The plateau-stage lines were not read. The amendment's phrase "before any X1 output was inspected" should be read with this qualification.

**Scorers.**
- `scripts/score_X.py` (frozen) → `logs/score_X.json`;
- `scripts/score_X_amend.py` → `logs/score_X_amend.json`.

Both are fail-closed. The amendment's exact references are in `logs/score_X_amend_refs.json`.

**Models.** GATE seeds 3 and 4: new models trained from scratch with their own initialisation and data. They were untouched before scoring.
- **Statistical unit:** the model (n = 2).
- **Data:** stream 98_500_000, 3000 episodes (18,000 questions). Episode-bootstrap intervals are reported for G and A.

## 1. What was predicted, and why

**The mathematics** (CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md §§2–3; exact):
- GATE's trust variable is invisible to every single source.
- The Bayes decision has interaction order exactly 3.
- Every rule of order ≤ 2 has gate-flip slope G = 0.
- The order-1 optimum is exactly invariant to the evidence swap x₃ ↔ x₅.
- Exact NLLs: order 1, 0.462; order 2, 0.431; Bayes, 0.182.

**The hypothesis.** Gradient training acquires lower interaction orders first (simplicity bias). This is an external assumption, not a consequence of the mathematics. Review A stressed this point, and the review is right.

**Predictions:**
- **X1a:** an order-1 phase comes first. Competing explanation: relational and evidential components are learned jointly.
- **X1b:** pairwise agreement (order 2) is used before the gate (order 3). Competing explanations: the gate comes first, or the model leaps from order 1 to order 3.

## 2. Outcomes

| | s3 | s4 | verdict |
|---|---|---|---|
| competent (final KL ≤ 0.02, G ≥ 0.9) | yes (KL 0.000, G 0.994) | yes (KL 0.000, G 0.995) | — |
| **X1a** (frozen): ≥ 4 consecutive snapshots with acc ≥ 0.75, A ≤ 0.3, G ≤ 0.1 | yes: 14 snapshots, steps 100–425 | yes: 8 snapshots, steps 100–275 | **CONFIRMED** |
| **X1a′** (amended): the same plus 0.75 ≤ acc ≤ 0.83, NLL ≤ 0.50, A_p ≤ 0.3 | yes (14) | yes (7) | **CONFIRMED** |
| **X1b** (frozen): some snapshot with G ≤ 0.1 and A ≥ 0.5 | no | no | **FALSIFIED** |
| **X1b′** (amended): some snapshot with G ≤ 0.1 and A_p ≥ 0.5 | no (max A_p before the gate 0.27) | no (max 0.09) | **FALSIFIED** |

**Trajectories** (`logs/score_X.json`, `logs/score_X_amend.json`):

| step | s3 acc / NLL / G / A_p | s4 acc / NLL / G / A_p |
|---|---|---|
| 100 | 0.78 / 0.53 / 0.000 / 0.03 | 0.79 / 0.51 / 0.000 / 0.02 |
| 200 | 0.80 / 0.48 / 0.000 / 0.07 | 0.81 / 0.46 / 0.001 / 0.06 |
| 300 | 0.81 / 0.46 / 0.002 / 0.13 | 0.83 / 0.41 / 0.107 / 0.07 |
| 400 | 0.81 / 0.45 / 0.006 / 0.24 | 0.88 / 0.25 / 0.66 / −0.18 |
| 450 | 0.82 / 0.44 / 0.027 / 0.27 | 0.88 / 0.24 / 0.69 / −0.17 |
| 500 | 0.88 / 0.35 / 0.24 / 0.16 | 0.88 / 0.23 / 0.72 / −0.16 |
| 600 | 0.90 / 0.22 / 0.76 / −0.06 | 0.88 / 0.23 / 0.74 / −0.10 |
| 1000 | 0.90 / 0.19 / 0.96 / −0.04 | 0.90 / 0.19 / 0.94 / −0.03 |
| 4000 | 0.90 / 0.18 / 0.99 / 0.00 | 0.90 / 0.18 / 0.995 / 0.00 |

**Intervals.**
- At the first order-1 snapshot (step 100): G = 0.0002 [−0.0007, 0.0010] (s3) and 0.0003 [−0.0003, 0.0009] (s4); A = 0.030 [0.026, 0.035] and 0.018 [0.014, 0.023].
- Final: G = 0.994 [0.994, 0.995] and 0.995 [0.995, 0.995].

## 3. Interpretation

**Supported (frozen confirmation, n = 2 new models; the order-1 phase also occurred in development seed 1).** GATE models trained from scratch first reach the order-1 accuracy level with no use of the gate relation (G ≈ 0, below 0.03) for about 175–350 steps.
- Accuracy 0.80 (exact optimum 0.801).
- **Corrected after review B (D).** s3 does not simply stay at the order-1 ceiling. With G ≤ 0.011, its NLL falls to 0.449, below the exact order-1 optimum of 0.462: it realises about 40% of the order-2 gain, with A_p up to 0.27. s4 is close to a pure order-1 phase (A_p ≤ 0.09; NLL 0.464–0.47).
- The X1a criteria cannot separate order 1 from partial order 2, because both have accuracy 0.801.

**Falsified.** These models do **not** learn the pairwise-agreement (order-2) rule to half strength before the gate. The supported reading is: order-1 accuracy first; agreement use then grows slowly (s3 partly, s4 hardly); then the gate is acquired abruptly, before agreement reaches the order-2 level.
- Agreement use grows slowly during the order-1 phase, to A_p ≤ 0.27 in s3 and 0.09 in s4, measured against the order-2 optimum's 0.96.
- The gate is then acquired as a leap: G goes from 0.03 to 0.76 in about 150 steps in s3, and from 0.11 to 0.66 in 100 steps in s4.
- After the leap A_p is transiently negative and returns to 0, so the models end on the Bayes rule, which uses no agreement beyond the gate.

**What this tells us about the competing explanations:**
- **Joint learning** (relation and evidence together) is rejected for X1a. The amended accuracy band excludes the small-scale-Bayes version of it.
- **An order-2 phase before the gate** is rejected.
- The outcome fits a **leap model** better. The gate has no lower-order projection (Proposition 3), and the order-2 increment is small (0.031 nats against 0.25 for the gate). Learning order is therefore not set by interaction degree alone; the size of each component's contribution matters too. Review A anticipated this (F4); it is an explanation **[spec]**, not a tested law.

**Source of the failure** (mandate §9):
- **Not incorrect mathematics.** All references are exact.
- **Not a model failing to implement the rule.** Both models are competent.
- **The training principle.** "Order-by-order" is too strong as stated.
- **Not a flawed intervention.** G and A_p have exact reference values.
- **Not a cross-family similarity assumption.** X1 is GATE-specific; review A showed that Investigation VIII's separable phase (ledger N12) is not its T4 counterpart.

## 4. What X1 does and does not establish across families

- **It establishes**, for GATE, an exact, measurable learning sequence: order 1 → order 3. This is a prediction derived from mathematics that survived on new models.
- **It does not establish** that the same sequence holds in T4. T4 models in the native line start from pretrained layers that already compute agreement statistics. The T4-from-scratch history is mixed: the M Transformer failed (0/3) and DeepSets succeeded (3/3).
- **The T4 analogue remains untested.** It would need T4 training from scratch with dense snapshots, which XIV did not run.
- **It does not establish a common principle.** Review A rejected the proposed common principle as having no cross-family predictive content. X1 rests on A1, a symmetry shared by both families, together with a GATE-specific order calculus.

## 5. Other prospective tests

Stage I's frozen T4 predictions (R-S, R-N, R-P) are reported in T4_FINAL_CAUSAL_CONTROLS.md. They are T4-local and are not cross-family predictions.
