# Frozen protocol: latent reliability (mandate §4)

**Freeze commit.** The commit that adds this file.

**What existed before the freeze:**

- the math validation (`MATH_VALIDATION.md`);
- two pilot Transformer runs at size S (seed 100, T1 and T4) and a dev analysis of the S/T1 pilot, used to debug code;
- the M-size T1 pilot (seed 100), which was still training. Only its training loss had been seen (it falls below the own-question Bayes loss after about 1,800 steps).

**What did not exist before the freeze:**

- no model with seeds 0–2 had been trained;
- no confirmation-split data had been generated for any model.

**Honesty notes.**

- This protocol is preregistered **for this package only**. The hypotheses are new to this mandate and are *not* presented as preregistered before the earlier Phase 1.5 work.
- Old checkpoints in earlier packages are historical exploratory evidence and are not modified.

## 1. Generators and Bayes (frozen)

- **Code:** `src/bayes.py` at the freeze commit.
- **Parameters:**
  - primary setting K = 3, Q = 6, α = 0.8, α5 = 0.5;
  - the six sources are in groups {0,1}, {2,3}, {4,5};
  - the R prior is uniform.
- **Exact posterior:** `episode_posterior`.
- **Tasks:** T0–T5 as in `MATH_VALIDATION.md` §0 and §7.
  - T2 adds a confidence head, trained by MSE (weight 4) to the Bayes max posterior.
  - T3 adds an R head, trained by cross-entropy on the true R.
  - The extension K = 5 is run only if compute remains after the primary grid, and it is labelled as an extension.

## 2. Architectures and training (frozen)

**Transformer** (`src/nets.py`):

- bidirectional and pre-norm, GELU MLP;
- one token per (question, source) = value embedding + source embedding + question embedding;
- Q readout tokens and one episode token;
- the answer is read from readout token q; the R head (T3) is read from the episode token.

| size | d | layers | heads | MLP width | params |
|---|---|---|---|---|---|
| S | 16 | 2 | 2 | 64 | 6.8k |
| M | 64 | 3 | 4 | 256 | 151k |
| L | 128 | 4 | 4 | 512 | 794k |

**Baselines** (same data stream, same optimiser, same steps):

| baseline | input | params (M) |
|---|---|---|
| `mlp_full` | one-hot of the entire 6×6 report matrix, 3 hidden layers of width 256 | 166k |
| `mlp_hist` | per-question global vote histograms only | 143k |
| `deepsets` | per-question encoder φ(X_q), mean pool m, readout ρ(h_q, m, X_q) | 209k |

- `mlp_hist` has a Bayes ceiling of 0.636 on T1.
- `deepsets` has an explicit shared-summary site, `pool`.

**Training:**

- AdamW, lr 1e-3, warmup over min(200, steps/10) steps then cosine decay, weight decay 1e-4;
- batch 256, **6000 steps** (1.54M episodes);
- fresh samples from the generator each step;
- the training data RNG is a fixed function of (seed, task, K).

**Training grid** (run in this priority order; later items are dropped if compute runs out, and any drop is reported):

1. M: T1 and T4, seeds 0, 1, 2.
2. Baselines (`mlp_full`, `mlp_hist`, `deepsets`) at M width: T1 and T4, seeds 0, 1, 2; T0 and T5, seed 0.
3. M: T2 and T3, seeds 0, 1, 2.
4. S: T1 and T4, seeds 0, 1, 2.
5. M: T0 and T5, seeds 0, 1.
6. L: T1 and T4, seed 0, then seed 1.

**Compute recording.** Wall-clock seconds, parameter count and examples seen are recorded in each run's json.

**No size escalation.** Capacity is not increased beyond L, whatever the result.

**Optimization failure vs mechanism absence:**

- A run whose confirmation accuracy is below own + 0.5·(Bayes − own) is labelled **"did not learn the cross-question task"**.
- Mechanism analyses of such runs are reported as descriptive only.
- If the same (task, size) is learned by `mlp_full` or `deepsets` but not by the Transformer, the failure is attributed to Transformer optimisation at that size, not to the task.

## 3. Evaluation splits and seeds (frozen)

- **Dev split:** data RNG base 999000.
  - Used to select the site s* for each model: the layer whose readout-token linear probe best decodes the leave-one-out reliability posterior (§4 D1).
  - The selection is written to that model's dev json.
- **Confirmation split:** data RNG base 777000.
  - Generated only after this freeze.
  - All reported numbers and criteria use it, with s* taken from the dev json of the same model.
- **Pilot models** (seed 100) are used for code checks only. They are not part of any criterion.

## 4. Mediator candidates (frozen definitions; identified without model outputs)

All candidates live at the readout token of question q, at site s*. Sites are resid_pre0 … resid_pre{L−1} and resid_post{L−1}. A candidate is a subspace of dimension k:

- k = |R| − 1 for T1–T4 (2 for T1–T3, 5 for T4);
- k = 3 for T0 and T5.

| id | definition | uses |
|---|---|---|
| D1_probe | row space of a ridge probe (λ = 0.01·N) from readout activations to the centred log of the leave-one-out reliability posterior π_{−q}. For T0/T5 the target is the leave-one-out group agreement counts. | Bayes variable, never outputs |
| D2_contrast | top-k PCA of activation differences at readout q between on-manifold `cond` pairs | inputs only |
| D1_fit_q012 | as D1, fitted on questions 0–2 only | reuse test |
| hist_surrogate | row space of a probe to the other questions' global vote histograms (aggregate, non-relational) | equivalent-computation control |
| random | 8 random orthonormal k-dim bases | control |
| full_residual | identity (whole residual at readout q) | upper bound for this location |

The site × location sweep (cond pairs only) is a diagnostic, not a criterion. It covers:

- every site;
- locations "readout" and "q_tokens" (the readout token plus the 6 source tokens of question q), each with D1 refit and with the full residual;
- the episode token (full residual).

## 5. Interventions (frozen)

Pairs share the base episode E. The source E′ is built as follows (`analysis.pairs`):

| kind | construction | what it changes | on-manifold? |
|---|---|---|---|
| cond | X′_q = X_q; X′_{−q} resampled from P(X_{−q} \| X_q) via R′ ~ P(R \| X_q) | the reliability evidence in the other questions | yes, by construction |
| perm | X′_q = X_q; each other question's reports permuted across the 6 sources | relation only; every per-question histogram is identical (matched aggregate) | classified per pair: on if log P(E′) is within the 1st–99th percentile of natural episodes |
| splice | X′_q = X_q; X′_{−q} from an independent episode | context | classified as for perm |
| target | X′_{−q} = X_{−q}; X′_q = X_q + shift (mod K) | target values only; all agreement counts, hence P(R \| X), are unchanged | yes |

**Interchange metrics** (model-relative):

- **IIA:** the fraction of *informative* pairs (where the model's argmax at q differs between E and E′) for which the patched-E argmax equals the model's argmax on E′.
- **kl_recovered:** the median over pairs (with KL(m(E′) ‖ m(E)) > 1e−3) of 1 − KL(m(E′) ‖ patched) / KL(m(E′) ‖ m(E)).
- **change_rate:** the fraction of pairs where the patched argmax ≠ the base argmax. Used for `target`, where *low* is the prediction.

**Other tests:**

- **Necessity.** Resample-ablate the candidate subspace at **all** readout tokens using activations of independent episodes. drop_frac = (acc_clean − acc_ablated) / (acc_clean − own-question Bayes acc). It is undefined if the gap is < 0.02.
- **Restoration.** Take the corrupted episode (`splice`), restore the candidate at readout q from the clean episode, and measure kl_recovered toward the clean output.
- **Cross-head (T2).** Correlation and slope between the patched and the model change in confidence under cond/D1.
- **Content (output-free D2 coordinates).** Ridge R² (fit and test splits) of the D2 coordinates from:
  - (a) logit π_{−q};
  - (b) leave-one-out counts;
  - (c) other-question histograms;
  - (d) the target question's own one-hot reports.
- **Type B (attention).**
  - For every layer and head: R² between the readout-q attention mass on each group's sources *within question q* and P(group g reliable | X).
  - Pattern patching of the readout-q query row (all heads, and each head) from cond pairs, scored by IIA and kl_recovered.
- **Bypass.** D1 vs full residual at the same location and site; the readout location vs q_tokens; the episode token.
- **Controls.**
  - A random-init model of the same architecture is probed identically.
  - T0 and T5 models (no Bayes value of cross-question coherence) get the same battery.
  - T3 (R supervised) is a positive control, reported separately.

## 6. Criteria (frozen thresholds; confirmation split)

For one Transformer model:

| id | criterion | threshold |
|---|---|---|
| P0 | behavioural precondition | acc ≥ own + 0.5·(Bayes − own) |
| P1 | identification | probe R²(readout, π_{−q}) at s* ≥ 0.6 and ≥ random-init R² + 0.3 |
| P2 | sufficiency (on-manifold) | D1 cond IIA ≥ 0.7, ≥ random mean IIA + 0.3, kl_recovered ≥ 0.5 |
| P3 | matched aggregate | D1 perm IIA ≥ 0.6 |
| P4 | specificity | D1 target change_rate ≤ 0.10 |
| P5 | necessity | D1 drop_frac ≥ 0.5 and random-mean drop_frac ≤ 0.2 |
| P6 | cross-question reuse | D1_fit_q012 IIA on questions 3–5 ≥ 0.7 × D1 cond IIA |
| P7 | relational content | D2 content R²(max of π_{−q}, counts) ≥ R²(histograms) + 0.2 |

**Verdicts per (task, size), over the trained seeds:**

- **Type A (explicit low-dimensional reliability summary at the readout):** P0–P7 all pass in ≥ 2/3 of the seeds (≥ 2 of 3, or 2 of 2).
- **Type B evidence (operation-level / routing)**, reported whether or not Type A passes:
  - some layer's all-head readout pattern patch has IIA ≥ 0.7 and kl_recovered ≥ 0.5;
  - and some head's group-mass R² ≥ 0.5;
  - in ≥ 2/3 of the seeds.
- **Distributed but localized:** Type A fails, but the readout or q_tokens full residual at some site has IIA ≥ 0.9 while D1 fails P2. This means the information passes through those tokens but is not captured by a k-dimensional probe subspace.
- If none of these hold, the result is "no identified mediator **under these protocols**". It is never "no mechanism".

**Primary answer to the central question:**

- **YES** if the Type A verdict holds for M/T1 **and** M/T4, and P6 holds. T4 is required because there the relation must be used non-monotonically (`MATH_VALIDATION.md` §7).
- **PARTIAL** if it holds for exactly one of them, or only Type B / distributed evidence holds.
- **NO (scoped)** otherwise.
- Every answer is scoped to these generators, sizes and protocols.

## 7. Failure conditions and what is not allowed

- Thresholds, site selection rules, the pair constructions and k are not changed after confirmation data are generated.
- Code bugs found later may be fixed. Each fix is logged in `DEVIATIONS.md`, with the old and new results where they differ.
- A test that cannot be computed (for example, no informative pairs) counts as a fail for that criterion, not a pass.
- Results for T3 (supervised R) are never pooled with T1/T4 verdicts.
