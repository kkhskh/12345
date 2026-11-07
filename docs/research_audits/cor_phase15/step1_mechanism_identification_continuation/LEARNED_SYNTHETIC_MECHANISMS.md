# Learned synthetic mechanisms (§9–§10), corrected after review

**Scope.** This program is exploratory. It does **not** validate COR in any pretrained language model.

**Preregistration.** `PREREGISTRATION_LEARNED_MODELS.md`, with these commits:

| commit | content |
|---|---|
| `32121da` | preregistration, before any training |
| `fad7f89` | clarifications D1–D4 |
| `c4bc553` | frozen analysis and dev results, before the held-out seeds were trained |

Deviations D5–D7 were declared after review.

**Code.**

- Preregistered: `src/learned.py`, `src/learned_analysis.py`
- Exploratory: `src/learned_posthoc.py`, `src/additive_lp.py`, `src/learned_review_checks.py` (post-review), `src/transfer.py`, `src/reparam.py`

**Compute.**

| run | wall time |
|---|---|
| dev training | 1,037 s |
| held-out training | ≈ 900 s |
| transfer | 992 s |
| post-review checks | ≈ 10 min |

All runs used 2 CPU cores.

## 1. Answer first

1. **T2 and T5a (Bayesian integration).** The learned margin is a function of the aggregate alone, but only on the training range.
   - `D_sum` is between 1e−6 and 3.5e−5 over 15 models and 24 transfer models.
   - `D_sum^K` is about 2e−4.
   - The graded test (T-RF′) finds 0 order reversals.
   - Outside [−3, 3], equal-sum pairs diverge (R6).
   - **No relation-sensitive computation emerges when it is not demanded.** This null is robust across seeds, widths, head counts and MLP widths.
2. **T1 (candidate-symmetric agreement, XNOR) is a learnability control.**
   - Every architecture learns it, except one transfer run: heads1/TF1 seed 5 reached 0.74, an optimization failure.
   - The attention-only model TF2-AO also learns it. Freezing its attention patterns drops it to chance (R3), so input-dependent routing is necessary there.
   - Because XNOR is not additively sign-representable (T-RF), any solution must contain a cross-source interaction. The graded test confirms it: 255–272 of 465 row pairs reverse, by up to 2.1 SD.
   - **Where that interaction is implemented is not identified** (§3–§4).
3. **T3 (conflict → prior)** is solved by an essentially relation-free function.
   - B-signs, linear on per-source signs plus the prior, scores **1.000**.
   - The graded reversals in the learned margins are tiny: ≤ 0.07 SD for TF1/TF2, ≤ 0.25 SD for TF2-AO.
   - So the margins are consistent with `h(f(a) + g(b))` up to small distortions.
4. **T4 (coherent-but-wrong).** The reviewed version said no relation was used. That was **wrong**.
   - The decision rule is additively representable on grids (LP).
   - But the learned margins show **28–337 reversing row pairs, up to 1.1 SD**, so the models implement it with a cross-source interaction.
   - The preregistered unit criterion missed this because it cannot see odd interactions such as `ab(a−b)` (R4 F5).
5. **T5b (K = 3, argmax agreement → prior)** is threshold logic (T-RF): B-signs scores 1.000, and B-add3 scores 0.979.
6. **Decodability is not use.** `ab` is linearly decodable in T2 (R² up to 0.99), yet the T2 margin is a function of `a + b`. Note, however, that `ab` and `sign(ab)` are themselves functions of non-relational features: `ab = ((a+b)² − a² − b²)/2` (R4 F2). So "relational" must be defined relative to a declared function class. Here that class is source-additive functions `h(f(a) + g(b))`.

## 2. Accuracy

Mean validation accuracy. Each Transformer cell gives dev seeds 0–2 / held-out seeds 3–4.

| task | TF1 | TF2 | TF2-AO | B-sum | B-lin | B-MLP | B-add (p in f, g) | B-add3 | B-signs |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.599 | 0.479 | 0.999 | 0.482 | 0.518 | 0.500 |
| T2 | 1.000 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 | 0.752 |
| T3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.859 | 0.859 | 0.998 | 0.997 | 0.996 | **1.000** |
| T4 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.503 | 0.504 | 0.993 | 0.920 | 0.930 | 0.751 |
| T5a | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 1.000 | 1.000 | 0.997 | 0.998 | 0.998 | 0.697 |
| T5b | 0.984 / 0.983 | 0.999 / 0.998 | 0.993 / 0.991 | 0.813 | 0.813 | 0.984 | 0.991 | 0.979 | **1.000** |

On T4, B-add trained 4× longer reaches 0.978 (R5), and an exact additive construction exists. B-add's lower score is an optimization limit, not a representational one.

## 3. The preregistered unit-level identification failed

The preregistered criterion was `I_agree ≥ 0.2` and `D_sum ≥ 0.2`, followed by mean ablation and a sign prediction. The review found four problems with it (R2, R4):

1. **It is not specific.**
   - It flags 50–83% of all hidden units in T1, and 45–72% in T3.
   - Ablating random unit sets of the same size gives similar effects (R4 F3).
   - The effect disappears at threshold 0.25.
2. **It is not the literal definition.**
   - It adds `ab` and `sign(ab)` as regressors against a two-way ANOVA baseline, rather than backfitting.
   - Under the literal definition, `ab` itself would count as non-relational.
   - This is declared as deviation D5.
3. **Its ablation results depend on the method** (post-review K3).

   | model | mean-ablate flagged | resample-ablate flagged |
   |---|---|---|
   | T1/TF1 | 0.75, 0.998, 0.75, 0.75, 0.75 | 0.50–0.69 |
   | T1/TF2 seed 0 | 1.000 | 0.995 |

   For T1/TF2 seed 0, ablating the **non**-flagged units is what drops accuracy, to 0.749 or 0.688.
4. **It misses odd interactions**, which is why it reported nothing for T4.

**Sign prediction.**

| | T1/TF1 | T1/TF2 | T3 |
|---|---|---|---|
| hits | 61/71 dev, **33/35 held-out** | 100/141, 73/97 (71–75%) | at chance |

The held-out TF1 rate is the one genuinely prospective, preregistered prediction in the program. It depends on how units are counted (unit splits), and it ignores the final-LN mean (R4 F7).

**Verdict.** The criterion is reported as a **failed identification procedure**. It is a counterexample to the idea that activation statistics select used units, and it is not evidence of an identified mechanism. The population-level gauge invariance reported earlier holds by construction: an exact split duplicates the activation (R4 F9, R5).

## 4. Interchange at the final residual is uninformative

The preregistered interchange along the `sign(ab)` probe at the final residual measures alignment with the readout, not use (R2 FATAL-2, R4 F4, R5 H3).

| task | \|cos(probe, readout)\| | readout-direction interchange transfer |
|---|---|---|
| T1 | 0.89–0.99 | 1.0 |
| T2 | 0.01–0.15 | — (no differing pairs) |
| T3 | ≤ 0.08 | 1.0 |
| T4 | ≤ 0.14 | 1.0 |

Upstream, the T1 probe transfers only 0–0.5. The reviewed claims "agreement is causally used in T1" and "decodable but not used in T3/T4", both based on this test, are **withdrawn**.

## 5. What survives

1. **The T2/T5a null**, on the training range: no relation emerges when it is not demanded. It holds across 5 seeds, 3 architectures and 6 transfer variants.
2. **T3 and T5b are threshold logic**: exact relation-free baselines exist. The T3 learned margins are near-additive.
3. **T1 and T4 learned margins contain a cross-source interaction** (T-RF′ reversals), but its *locus* is not identified.
4. **TF2-AO needs input-dependent attention routing to solve XNOR** (R3).
5. **The preregistered sign prediction** for T1/TF1 held out: 33/35.

## 6. Transfer (exploratory, seeds 5–6)

Variants were residual width 8 and 32, 1 and 4 heads, and MLP width 8 and 64. `logs/transfer.txt` has the numbers.

- **T2:** `D_sum` ≤ 2e−5 and 0–1 flagged units in all 24 runs. The null transfers.
- **T1:** learned in 23 of 24 runs (heads1/TF1 seed 5 reached 0.74). Mean-ablation effects vary from 0.69 to 0.999, which is consistent with §3: the criterion is unreliable.

## 7. Identification status

| | |
|---|---|
| **Identified**, relative to generator labels | the source variables (probes); the *existence* of a cross-source interaction in T1 and T4 (T-RF′); its absence in T2/T5a (on range) |
| **Not identified** | any unit-, direction- or site-level implementation of the T1 or T4 interaction; equivalence to the designed A pairing |
| **Residual-rotation gauge** | applicable exactly (R4 `c4`: logit change ≤ 8e−6, residual-site results unchanged under the conjugated rotation). The reviewed claim that it could not be applied was wrong. |
