# Learned synthetic mechanisms (§9–§10)

**Scope.** This program is exploratory. It does **not** validate COR in any pretrained language model.

**Preregistration.** `PREREGISTRATION_LEARNED_MODELS.md`, with these commits:

| commit | content |
|---|---|
| `32121da` | preregistration, before any training |
| `fad7f89` | clarifications D1–D4, before any mechanism metric was computed |
| `c4bc553` | frozen analysis code and dev-seed results, before the held-out seeds were trained |

**Code.**

- `src/learned.py`: training
- `src/learned_analysis.py`: the preregistered analysis
- `src/learned_posthoc.py` and `src/additive_lp.py`: **exploratory**, added after the dev results were seen
- `src/reparam.py`
- `src/transfer.py`: exploratory

**Logs.** `logs/learned_*`, `logs/reparam.txt`, `logs/transfer.txt`.

**Compute.**

| run | wall time |
|---|---|
| dev training (54 Transformers and 60 baselines) | 1,037 s |
| held-out training | ≈ 900 s |
| analyses | under 5 minutes in total |

Everything ran on 2 CPU cores and used well under the 2-hour budget.

## 1. Answer first

1. **Relation computation appears only when the target is a candidate-symmetric agreement rule (T1, XNOR).** In every T1 model, across all seeds and all three architectures:
   - preregistered relation-sensitive units exist;
   - interchange along the agreement probe transfers the decision (1.0, against 0.0 for the random direction);
   - ablating the identified units roughly halves the matched agreement contrast Δagree, and drops accuracy to about 0.75 in 7 of the 10 T1 models that have MLPs.
2. **It does not appear when it is not demanded (T2, T5a).** The margin is a function of the aggregate alone, with `D_sum` between 1e−6 and 3e−5 (binary) and `D_sum^K` ≈ 2e−4 (K = 3). No decision changes on matched-sum pairs.
3. **It does not appear in the tasks *designed* to need agreement-conditional reconciliation (T3, T4, T5b), because those tasks do not need it.**
   - Theorem T-RF (`CONDITIONAL_CAUSAL_THEOREMS.md`) shows that candidate-directed agreement rules are threshold logic on per-source support. The binary T3 rule is AND or OR, selected by the prior; T5b is solved by additive scores.
   - The relation-free baseline B-add (no term reads two sources) reaches:
     - T3: 0.997
     - T5b: 0.991
     - T4: 0.920 (T4 is LP-feasible as an additive rule)
   - The learned T3 and T4 margins are largely additive: `R²_add` ≈ 0.65–0.70, and the additive fit's sign agrees with the margin on 97–100% of the grid.
   - Interchange along the agreement-probe direction transfers **0%** of the differing decisions. So agreement is partly decodable (0.6–0.8) but not used.
4. **Decodability is not relation computation.** In T2, `ab` is linearly decodable from the post-MLP residual (R² 0.88–0.99), yet the margin depends only on `a + b`. This works because `ab = ((a+b)² − a² − b²)/2` is a linear function of non-relational features.
5. **Attention alone can compute the relation.** The attention-only TF2-AO solves T1 (val acc ≥ 0.9998) with no MLP: the cross-source interaction comes from softmax and LN across positions. In TF1, `sign(ab)` is already linearly decodable after attention, before the MLP, at 0.998.

## 2. Accuracy

Mean validation accuracy, with dev seeds 0–2 and held-out seeds 3–4 reported separately:

| task | TF1 dev / held-out | TF2 | TF2-AO | B-sum | B-lin | B-MLP | B-add (expl.) |
|---|---|---|---|---|---|---|---|
| T1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.599 | 0.479 | 0.999 | 0.482 |
| T2 | 1.000 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 1.000 | 1.000 | 0.999 | 1.000 |
| T3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.859 | 0.859 | 0.998 | 0.997 |
| T4 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.503 | 0.504 | 0.993 | 0.920 |
| T5a | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 1.000 | 1.000 | 0.997 | 0.998 |
| T5b | 0.984 / 0.983 | 0.999 / 0.998 | 0.993 / 0.991 | 0.813 | 0.813 | 0.984 | 0.991 |

Exact per-seed values for the held-out runs are in `logs/learned_train_heldout.json`. No architecture failed to learn any task, so the optimization-vs-representation question (§9) does not arise for the Transformers.

B-add's 0.92 on T4 is an optimization or smoothness limit, not a representational one. The task is LP-feasible as an additive rule, but needs steep jumps at `b = 0`.

## 3. Preregistered mechanism metrics

Relation units are those with `I_agree ≥ 0.2` and `D_sum ≥ 0.2`. Ablation is mean ablation. "Sign prediction" means: the unit's frozen-weight direction, times its agreement-conditional activation difference, compared with the sign of its actual ablation contribution.

| task / arch | n relation units (dev; held-out) | acc after ablating them | Δagree full → ablated | sign prediction hits | surrogate acc |
|---|---|---|---|---|---|
| T1 / TF1 | 22, 26, 23; 16, 19 | 0.746, 0.999, 0.749; 0.749, 0.750 | ≈ 21 → 7–11 | **61/71; 33/35** | 0.80–0.87 |
| T1 / TF2 | 39, 53, 49; 47, 50 | 1.000, 0.750, 0.995; 0.750, 0.749 | ≈ 21 → 10–17 | 100/141; 73/97 | 0.79–0.995 |
| T2 / TF1, TF2 | 0–2 | unchanged | ≈ 0 → ≈ 0 | — | unchanged |
| T3 / TF1 | 20, 20, 23; 17, 18 | ≥ 0.999 | ≈ −11 → −8 to −9 | 34/63; 23/35 (≈ chance) | 0.78–0.9997 |
| T3 / TF2 | 34, 36, 38; 33, 29 | ≥ 0.998 | ≈ −11 → −10 | 53/108; 33/62 (chance) | ≥ 0.998 |
| T4 / TF1 | 0; 0 | unchanged | ≈ 0 | — | — |
| T4 / TF2 | 0; 8, 6 | unchanged | ≈ 0 | 11/14 | unchanged |

**Grid metrics.**

| task | `D_sum` | Δagree | M-flip |
|---|---|---|---|
| T1 | 0.437 | 20.5–21.6 | 1.0 |
| T2 | ≤ 3.5e−5 | ≈ 0 | 0 |
| T3 (`p = ±1`) | 0.29 | ±10.4–11.6 (the sign follows p) | 0.5 |
| T4 | 0.45 | ≈ 0, but −20 when signed by S | 1.0 |

**Reading.**

- **T1.**
  - Identified relation units are causally used: in 7 of 10 models, ablating them drops accuracy to about 0.75, i.e. one quadrant pair is lost.
  - Their effect directions are predictable from frozen weights: 86% for TF1 dev and 94% for TF1 held-out.
  - In the three models where ablation leaves accuracy intact, the relation is computed **redundantly**. Attention already makes `sign(ab)` decodable at 0.998 before the MLP. Δagree still falls.
  - A sum-only surrogate costs 13–20 points of accuracy.
- **T3.**
  - Units pass the activation-level criterion, but are **not necessary**: ablation leaves accuracy ≥ 0.998.
  - Their direction is not predictable (chance-level hit rate).
  - The `I_agree` criterion flags units whose activation correlates with agreement in a model whose decision is additive given p. **The preregistered unit criterion is not sufficient for "used for agreement"**, and the ablation and interchange steps are what catch this.
- **T4.** There are almost no relation units, and the learned function is about 98% sign-additive.

## 4. Interchange: decodability vs use

This table restricts to matched pairs whose decisions differ (exploratory refinement, `learned_posthoc.py`). It reports the fraction of differing decisions that a probe-direction interchange at the final residual site transfers to the source.

| task | agreement-probe direction | random direction |
|---|---|---|
| T1 (all archs, all seeds) | **1.000** | 0.000 |
| T3 (all) | 0.000 | 0.000 |
| T4 (all) | 0.000 | ≤ 0.003 |
| T2 | (no differing pairs) | — |

The full-site patch at R transfers 1.0 in every model; this is the upper bound.

**Probe accuracy for `sign(ab)` at the final residual** (dev seed 0):

| task | probe accuracy |
|---|---|
| T1 | 0.999–1.0 |
| T2 | 0.59–0.80 |
| T3 | 0.65–0.79 |
| T4 | 0.60–0.70 |

Partial decodability coexists with zero causal use in T2–T4.

## 5. Transfer (exploratory, seeds 5–6)

Results are in `logs/transfer.txt`; the variants are residual width 8 and 32, 1 and 4 heads, and MLP width 8 and 64. If that log is present, its summary is added in `FINAL_RESEARCH_SYNTHESIS.md` §T.

## 6. Identification status

**Identified relative to the external generator labels, with selection never using the margin:**

- the source variables `a`, `b` (probes, R² up to 0.97–1.0 in T2; lower when the model compresses them);
- the agreement variable in T1, both decodable and causally used;
- a population of relation units in T1, with a set-level causal effect that is invariant under every applied gauge (`REPARAMETERIZATION_AND_BASELINES.md` §1).

**Not identified:**

- a unique unit-level implementation: indices and counts change under permutation and splitting, and the implementation is redundant across MLP and attention;
- the T1 relation feature as "the same mechanism as A". A's XNOR uses a designed `h(u+v) − h(u−v)` pairing; the learned units are not checked for that pairing, and attention-only models solve T1 without it.
