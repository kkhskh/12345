# Source semantics, consistency definitions, and identification procedure

This document covers mandate sections §4, §5 and §10.

- Code:
  - `src/semantics.py` (the generator and the definitions)
  - `src/matched.py` (D-experiments)
  - `src/learned_analysis.py` (the §10 procedure)
- Logs:
  - `logs/matched.txt`
  - `logs/learned_analysis_*.txt`

## 1. The common generator (§4)

All semantic variables are defined by the data-generating process. **No variable is read from a model.**

| variable | binary tasks | K = 3 (T5) |
|---|---|---|
| source identity | position and identity embedding (A, B) | same |
| source content | `a, b ∈ ℝ` | `z_A, z_B ∈ ℝ³` |
| evidence strength | `\|a\|, \|b\|` | `μ_s` (generator), `‖z_s‖` |
| candidate support | `sign a, sign b` | `argmax z_s` |
| semantic agreement | `1[ab > 0]` | `1[argmax z_A = argmax z_B]` |
| task target | per task (T1–T5), fixed in the preregistration | same |
| model preference | recorded separately, never used to define the above | same |

`scenario_catalogue()` lists every §4 case as explicit generator settings:

- agreement for c, and agreement for c′;
- disagreement;
- unequal strengths;
- weak agreement, weak disagreement, strong disagreement and balanced disagreement;
- conflict with prior c, and conflict with prior c′;
- coherent-but-wrong (T4), where agreeing sources are wrong by construction;
- conflicting-but-correct (T2, `(2, −0.5)`), where the Bayes posterior follows the stronger source and is correct;
- ≥ 3 candidates (T5);
- distractors, as anchor and distractor content `w` in the residual (`MLP_NORMALIZATION_THEOREMS.md`, M2).

## 2. Five operational definitions (§5)

| | D1 semantic candidate agreement | D2 representation agreement | D3 computational compatibility | D4 causal compatibility | D5 agreement-sensitive computation |
|---|---|---|---|---|---|
| **domain** | source pairs | encoded vector pairs | source pairs, relative to one frozen mechanism | input pairs, relative to a mechanism and an intervention family | operators × an input distribution |
| **reference variables** | candidate set and generator semantics | an alignment matrix M | the mechanism's read directions, e.g. `sign(u'v')` (A, under cross-talk) | the intervention family, e.g. single-source ablation | a declared simpler aggregate, e.g. `a + b` or `Ŝ` |
| **coordinate dependence** | none given the candidate set | yes: changes under non-orthogonal basis change unless M is covariant | inherits the mechanism's coordinates | none beyond the intervention family | depends on the declared aggregate |
| **same value, different decision (computed)** | (1.5, 1.2) and (−1.5, −1.2): both agree. T3 targets c vs c′; C′ margins are +2.326 vs −2.326 | (1, 1) and (−1, −1): ⟨·,·⟩ = 1 for both; C′ margins are +1.890 vs −1.890 | (2, 0.5) and (−2, −0.5): A's margin is identical (0.997, by A1*); C′ margins are +1.996 vs −1.996 | in C′, 76% of agreeing cells and 31% of conflicting cells are D4-compatible: D4 ≠ D1 | not an input property |
| **preserves** | which candidate is jointly supported | magnitude-weighted alignment | exactly what the downstream gate reads | response stability under the declared interventions | whether a relation is *used* beyond the aggregate |
| **discards** | strength, and which candidate (the binary label is symmetric) | the candidate identity of the top choice (K ≥ 3, see below) | everything the mechanism ignores | the mechanism-independent meaning | the direction of the effect unless stated |
| **computational/causal meaning** | none by itself: a property of the world | only if a mechanism reads exactly that bilinear form (B reads it; see Program B) | yes, by definition, but only for that mechanism | yes, by definition, but only for that family | yes; this is the property testable by matched interventions (§11) |

**Same value, different decision, in every definition.** D1 and D2 both assign the same value to agreement for c and to agreement for c′. So no agreement score by itself determines the candidate preference. This is the formal content of F3 and F25, now shown for two definitions on a real circuit. Any decision theorem must pair the relation with a **candidate-directed** variable (support, or `Ŝ`).

**K = 3: D2 is not D1** (T5 generator, 50,000 draws, seed 21):

| quantity | value |
|---|---|
| P(⟨z_A, z_B⟩ < 0 \| argmax agree) | 0.080 |
| P(⟨z_A, z_B⟩ > 0 \| disagree) | 0.436 |
| accuracy of sign⟨z_A, z_B⟩ for agreement | 0.713 |
| accuracy of the centred dot product (what an LN-centred encoder yields) | 0.819 |
| accuracy of the candidate-relative margin agreement `min_s(z_s[k*] − max_{j≠k*} z_s[j]) > 0` | 1.000 (exact by construction) |

**Post-review note (R4 F10).**
- K = 3 agreement is defined on the *noisy observations* `z_s`. It differs from agreement of the latent per-source candidates `Y_s` on 34% of draws.
- `μ_s` and `Y_s` are generated but not returned by `sample_k`.
- The 1.000 row for candidate-relative margin agreement holds by definition.

Explicit example: `z_A = (1, −3, 0)` and `z_B = (1, 0, −3)` agree on candidate 0. Their dot product is +1, but the centred dot product is −0.333.

**Verdict on the definitions.**

- **D1** is the right notion of *what the sources say*.
- **D5** is the right notion of *what a mechanism does*.
- **D2** is a mechanism description, valid only when a mechanism actually computes that bilinear form. For K ≥ 3 it is not a proxy for D1.
- **D3** and **D4** are mechanism-relative by construction, and useful only as bridges.

We do not keep a single "consistency" score.

## 3. Identification procedure without outcome fitting (§10)

The procedure is preregistered in `PREREGISTRATION_LEARNED_MODELS.md` §7 and implemented in `src/learned_analysis.py`:

1. **Source variables come from the generator**, as listed above.
2. **Localisation.** Ridge and logistic probes are fit on 2,000 generator inputs and evaluated on 2,000 held-out inputs, with no margin involved. The labels are `a`, `b`, `a+b`, `ab`, `sign(ab)`, `⟨z_A, z_B⟩`, the centred dot product and argmax agreement. Representations are identified **relative to these external labels**, which is recorded.
3. **Decodability vs use.**
   - Distributed interchange along the probe direction, between matched-sum agree/conflict pairs.
   - A random-direction control (D3).
   - A full-site patch as an upper bound.
4. **Relation operators.** These are hidden units whose activation, on the generator grid, satisfies `I_agree ≥ 0.2` and `D_sum ≥ 0.2`. Both statistics are computed from activations alone.
5. **Confirmation.** Mean-ablation (generator mean), followed by a recomputed forward pass.
6. **Direction from frozen weights.** The predicted sign is `sign(W_out[u]·(γ_f ⊙ ΔW_U)) × sign(E[act | agree] − E[act | conflict])`. It is computed **before** the ablation is run and compared with the ablation's sign.
7. **Relation-blind surrogate.** Each identified unit is replaced by its bin-mean as a function of `a + b`.

**Where the readout is used.** Step 6 uses the readout weights `ΔW_U` and the final-LN gain. They are inspectable frozen parameters, not the evaluated margin. This is recorded as a dependence: unit *direction* is identified relative to the readout; unit *selection* is not.

**Post-review status.** Steps 3–5 failed in practice:
- Final-site interchange measures readout alignment.
- The unit criterion is non-specific, and it misses odd interactions.

The defensible identification tools are:
- the graded single-cancellation test (T-RF′), which detects the *existence* of a cross-source interaction behaviourally;
- in engineered circuits, the mediator sign test (T-MS).

See `LEARNED_SYNTHETIC_MECHANISMS.md` §3–§5.

**Fixed-checkpoint observability vs function-level uniqueness.** The procedure identifies units at a fixed checkpoint. `REPARAMETERIZATION_AND_BASELINES.md` tests which outputs survive gauge transformations. Unit indices do not survive permutation, and unit counts do not survive splitting. The probe R² and interchange effects in the residual stream are invariant under all applied gauges.

Results of the procedure are in `LEARNED_SYNTHETIC_MECHANISMS.md`.
