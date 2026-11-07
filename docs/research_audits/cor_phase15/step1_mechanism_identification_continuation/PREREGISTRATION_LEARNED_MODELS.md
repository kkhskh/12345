# Preregistration: small learned models (§9–§10, §15–§16)

This file is committed **before any model in this program is trained or evaluated**. The commit hash is recorded in `RESEARCH_LOG.md`. Anything changed after that commit is listed in §9 as a deviation.

The program is exploratory. **It does not validate COR in any pretrained language model.** No pretrained model and no sealed benchmark is used.

## 1. Generator (`src/semantics.py`)

Source semantics are fixed by the generator, not by any model readout.

- **Candidates.** K = 2 (`c = +1`, `c′ = −1`) for T1–T4. K = 3 for T5.
- **Binary sources.** `a, b ∈ ℝ` are signed evidence. Positive values support `c`.
  - Recorded variables:
    - identity (A or B), fixed by position and identity embedding;
    - content (the value);
    - strength `|a|`;
    - support `sign a`;
    - semantic agreement `1[ab > 0]`;
    - the task target.
  - The model's preference is recorded separately.
- **Training distribution.** A mixture:
  - 70%: `a, b ~ U[−3, 3]` independently;
  - 30%: weak evidence, `a, b ~ U[−0.5, 0.5]`.
- **Prior token P** (T3 only): `p ∈ {−1, +1}` uniformly. It is embedded like a source with its own identity.
- **Multi-candidate sources** (T5): `z_A, z_B ∈ ℝ³`, `z_s = μ·onehot(Y_s) + N(0, I)`.
  - `Y_s` equals the latent truth `Y` with probability 0.6 and is uniform otherwise.
  - `μ ~ U[0, 3]`, drawn independently per source.
  - Prior candidate `p ∈ {0, 1, 2}` uniformly.

## 2. Tasks (targets fixed now)

| task | target | relation demanded? |
|---|---|---|
| T1 explicit agreement | `y = c` iff `ab > 0` (XNOR) | yes |
| T2 ordinary integration (Bayes) | soft target `P(c) = σ(a + b)`. This is the exact posterior when `a, b` are independent log-likelihood ratios. | **no**; the Bayes margin is `a + b` |
| T3 conflict-aware | `y = sign a` if `ab > 0`, else `y = p` (defer to prior) | yes |
| T4 coherent-but-wrong | `y = −sign a` if `ab > 0`, else `y = sign(a + b)` | yes, reversed polarity |
| T5a multi-candidate Bayes | soft target `softmax(z_A + z_B)` | **no** |
| T5b multi-candidate argmax agreement | `y = argmax z_A` if `argmax z_A = argmax z_B`, else `y = p` | yes, candidate-relative |

## 3. Architectures (`src/learned.py`)

Tokens are `[A, B, (P), R]`. Each source token is `W_in·value + E_identity`, with `W_in` linear and learned; R carries only its identity embedding. The models are causal pre-norm Transformers, read out at R through a final LN and an unembedding.

- **TF1:** 1 layer, 2 heads, GELU MLP width 32, `d = 16`.
- **TF2:** 2 layers, otherwise as TF1.
- **TF2-AO:** 2 layers, attention only (no MLP). This is the structural negative control for MLP interaction.

Baselines, given the same inputs and the same data:

- **B-sum:** logistic regression on `a + b` (and on p where present). For T5, softmax on `z_A + z_B` plus a prior one-hot.
- **B-lin:** logistic regression on `(a, b, p)`.
- **B-MLP:** matched-capacity MLP on the raw inputs, 2 hidden layers of 32, GELU.
- **B-bil (T5):** bilinear-plus-linear score model, `logit_k = z_Aᵀ M_k z_B + linear + prior`.

## 4. Training (identical for all runs)

- Optimizer: Adam, learning rate 3e−3 with cosine decay.
- Batch size 512; **3000 steps**; no early stopping.
- Loss: cross-entropy against the target (soft targets for T2 and T5a).
- Initialization: normal, std 0.02 for attention, 0.1 for MLP and input weights.
- Seeds:
  - **dev seeds 0, 1, 2** are used to develop the analysis;
  - **held-out seeds 3, 4** are run only after the analysis code is frozen and committed (§8).

## 5. Evaluation splits

- Validation: generator seed 1000 (20,000 samples, training mixture).
- Test grids:
  - **G-full:** a 61 × 61 grid on `[−3, 3]²`;
  - **G-weak:** a 41 × 41 grid on `[−0.5, 0.5]²`;
  - **G-matched:** raw-sum-matched pairs, defined below.
- T5 test: generator seed 2000 (20,000 samples).

## 6. Primary metrics (fixed now)

- **M-acc.** Accuracy against the hard target on the validation set. For soft targets, compare to the argmax of the target.
- **M-Dsum, the sum-sufficiency deficit.** On G-full, `D_sum = E[(m − ĝ(a+b))²]/Var(m)`. Here `ĝ` is the bin-mean of the model margin m over 121 bins of `a + b`. `D_sum ≈ 0` means the margin is a function of the aggregate alone.
- **M-Δagree.** The matched-sum agreement contrast. Over raw-sum-matched pairs `(a, b)` and `(a′, b′)` with `a + b = a′ + b′ = S`, `ab > 0` and `a′b′ < 0`, Δagree is the mean of `m(a, b) − m(a′, b′)`. Pairs are constructed with `S ∈ {−2, −1, −0.5, 0.5, 1, 2}` and differences `Δ = a − b ∈ (0, |S|)` for agreement and `Δ ∈ (|S| + 0.5, |S| + 3)` for conflict, 50 pairs each.
- **M-flip.** For T3 and T5b, the fraction of matched pairs whose model decision differs. This is a decision-level, not magnitude-level, relation consequence.

## 7. Mechanism-identification procedure (§10), fixed now

All steps use generator labels and frozen activations. **No step selects a probe, direction, unit or site by the evaluated margin.**

1. **Sites.** For every layer: the residual at each position after attention and after the MLP, and the MLP hidden units at R.
2. **Source localization.** On 4,000 fresh generator inputs (seed 3000), fit ridge probes (λ = 1e−3), training on half and evaluating R² on the other half, for:
   - `a`, `b`, `a + b`, `ab`, and `s = sign(ab)` (logistic, accuracy);
   - for T5: `⟨z_A, z_B⟩` and argmax agreement.
3. **Decodability vs use.** At each site, do a distributed interchange along the probe direction of a variable V. Replace the component of the base run's activation along the unit probe vector `w_V` with the component from a source run, recompute downstream, and record the change in margin. Source runs are matched-sum pairs (V = agreement) or random. Report both decodability (R²) and interchange effect.
4. **Candidate relation operators.** For each MLP hidden unit at R:
   - compute `D_sum(unit)`, using unit activations only;
   - compute the agreement interaction share `I_agree(unit)`: the share of the unit's activation variance explained by `ab` beyond the best functions of `a + b` and of `a` and `b` separately, i.e. the additive fit `f1(a) + f2(b)` by backfitting.

   A unit is **relation-sensitive** if `I_agree ≥ 0.2` **and** `D_sum ≥ 0.2`. These thresholds are fixed now.
5. **Confirmation.** Mean-ablate the relation-sensitive units, using the generator-distribution mean, and recompute. Report the change in M-Δagree and M-flip, and the change in accuracy.
6. **Predicted direction.** Before ablation, predict the sign of each unit's direct effect on m from frozen weights: `(W_out[unit]·γ_f)·(W_U[c] − W_U[c′])`, ignoring the final-LN scale, which is positive. Also predict the unit's agreement-conditional activation difference. The predicted sign of the unit's contribution to Δagree is the product of the two. Report the hit rate.
7. **Relation-blind surrogate.** Replace each relation-sensitive unit's activation with its bin-mean as a function of `a + b`, fitted on unit activations, not on margin. Recompute M-acc, M-Dsum and M-Δagree.

**Reparameterization** (§10 and `REPARAMETERIZATION_AND_BASELINES.md`):

- Apply to the trained checkpoints:
  - QK gauge (`W_Q → W_Q M`, `W_K → W_K M^{−T}`, random well-conditioned M);
  - VO gauge;
  - MLP unit permutation;
  - unit splitting (`w_out/2` on two copies);
  - head permutation.
- Report which identification outputs change: unit indices, per-unit `I_agree`, probe directions in the Q/K space, and the interchange effects.
- Positive rescaling is **not** function-preserving for GELU, so it is not used.

## 8. Order of operations

1. Commit this file.
2. Implement the generator, training and analysis. Run the dev seeds; development of the analysis code is allowed.
3. Commit the analysis code with the dev results, then run held-out seeds 3 and 4 with **no code changes**.
4. Report held-out results separately.

## 9. Deviations

None at the time of the preregistration commit (32121da). The following clarifications were added after the dev-seed training began. Only training loss and validation accuracy had been seen at that point; no mechanism metric had been computed:

- **D1, T3 grids.** §5–§6 did not fix the prior token on the grids. Every T3 grid metric is computed separately for `p = +1` and `p = −1`, and both are reported.
- **D2, T5 analogues of the §6–§7 metrics.** These were left unspecified in the preregistration and are fixed here:
  - **Sum-invariance deficit:** `D_sum^K = E_x[Var_δ(logits | z_A+δ, z_B−δ)] / Var(logits)`, with `δ ~ N(0, I)` and 16 draws per base point over 2,000 base points.
  - **Unit relation share:** `I_agree^K` is the R² gain from adding `[⟨z_A, z_B⟩, 1[argmax agree]]` to the base features. The base features are degree-2 polynomials in `s = z_A + z_B` plus degree-2 polynomials in `z_A` and `z_B` separately.
  - **Unit sum deficit:** `D_sum^K(unit) = 1 − R²` of a degree-2 polynomial in `s`.
  - The thresholds are the same as for the binary tasks (0.2 and 0.2).
- **D3, interchange control.** Every probe-direction interchange is also run along a random unit direction (seed 999) as a negative control.
- **D4, initialization.** Before any run, the initialization in `learned.py` was corrected to the preregistered stds: 0.02 for attention, 0.1 for MLP and input weights.


## 10. Compute budget

The budget is CPU only (JAX 0.12, optax), at most about 2 CPU-hours for all training. Actual usage is recorded in `logs/learned_train.txt`.

**Deviations declared after review (R4 F8).** These were not listed before the review and are recorded here as deviations:

- **D5.** The step-4 statistic `I_agree` was implemented differently from the preregistered text. It uses:
  - the R² gain of `[ab, sign(ab)]` over `[additive fit, sum bin-mean fit]`;
  - the additive fit computed exactly as two-way ANOVA main effects on the product grid, instead of by backfitting.

  The literal definition would classify `ab` itself as non-relational. The criterion has been reported as a failed procedure (`LEARNED_SYNTHETIC_MECHANISMS.md` §3).
- **D6.** `learned_analysis.py` was edited during dev-seed analysis (last edit 20:36, while dev checkpoints existed). §8.2 permits developing analysis code on dev seeds. It was frozen at `c4bc553` before the held-out seeds were trained.
- **D7.** Two kinds of run happened before the preregistered initialization fix (D4):
  - a 300-step timing run (T1, seed 0, all three architectures) used the old initialization;
  - after the fix, a full-length smoke test was run (T1 and T3, TF2, seed 0).

  Neither run's checkpoints or results were used. The dev training run began about 4 minutes before the D1–D4 commit (git and file timestamps). Only losses and accuracies had been seen at that point.
