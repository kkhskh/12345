# Learnable composition experiments (Problem A; development)

**Status.** These are development results. They were used to design the frozen protocol (`FROZEN_PROTOCOL_AND_CONFIRMATION.md`).
They are not confirmation evidence. The prediction about composition learners (P1) is confirmed on new seeds there.

## Question

Can a learner compute the eligibility relation from evidence that does not contain it, if nobody computes it first? The
relation is m_g, the number of other groups that are perfect (n_h = 5) and agree at the queried question. The learner must
reach the exact posterior p*(y) ∝ Σ_g RHO^{n_g} m_g L_g(y) (`MATHEMATICAL_RELATIONAL_OPERATION.md` §1).

## Conditions and readers

| condition | input to the reader (for the queried question q, for each of the three groups) |
|---|---|
| O (oracle reference) | one-hot count n_g (0..5), one-hot q-report pair, group id, **plus one-hot m_g** |
| R (raw ingredients) | the same, without m_g. No combined "perfect and agrees" flag exists anywhere |
| X (ordinary tokens) | the full episode token sequence; a Transformer from scratch (`scripts/x_train.py`) |
| Z (frozen states) | frozen P-count / T1 representations (`SELECTIVE_LEARNED_REPAIR.md`, `REPRESENTATION_ACCESSIBILITY.md`) |

Readers (`src/readers.py`) are fed the same per-group features in every case:

- **linear**: the concatenated features feed a linear map to 3 logits.
- **mlp**: concat → 128 → 128 → 3 (GELU).
- **attn**: per-group embedding (d = 40), one self-attention block across the 3 groups with feed-forward, mean pool, then
  linear to 3. **attn2** uses two blocks.
- **relnet**: pairwise φ(group g, group h) summed over h, plus ψ(g), then ρ (H = 64).

Training uses fresh generator samples (stream 19_000_000 + 1000·seed) with AdamW warmup-cosine at lr 1e-3. The target is
the generator label (cross-entropy) or the exact Bayes posterior (soft targets). Evaluation uses Investigation VII's
`eval_dev.npz` (stream 17_100_000), with the same metrics and regimes as Investigations VI and VII.

**The R condition is built so that the control families are null.** F0, FLOC and FID change only tokens that do not enter
the reader's features, so these readers return exactly zero on them (the "F0 0.000" column). That is a fact about the
design, not a finding.

## Results at the end of training (dev eval)

E is the response error on the F45 R1 edit: the model's centred-logit change minus Bayes' change. Bayes' own R1 change is
dm_B = +0.233. |E| R2∪R3 is the mean absolute error on the opposite-sign regimes.

| reader | cond | target | steps | seeds | KL to Bayes | E_R1 | dm_R1 | \|E\| R2∪R3 | F34 E |
|---|---|---|---|---|---|---|---|---|---|
| linear | O | label | 16k | 1 | 0.335 | −0.23 | 0.00 | 2.11 | −0.44 |
| linear | R | label | 16k | 1 | 0.335 | −0.23 | 0.00 | 2.11 | −0.44 |
| mlp | O | label | 16k | 2 | 0.0005 | +0.03 / +0.01 | +0.26 / +0.25 | 0.10–0.11 | ≈0 |
| mlp | R | label | 16k | 2 | 0.003–0.004 | **−0.46 / −0.42** | **−0.22 / −0.18** | 0.34–0.42 | ≈0 |
| mlp | R | label | 48k | 1 | 0.0005 | +0.08 | +0.31 | 0.12 | ≈0 |
| mlp | R | Bayes | 16k | 1 | 0.0002 | +0.01 | +0.24 | 0.05 | ≈0 |
| attn | O | label | 16k | 2 | 0.0002 | −0.02 / +0.00 | +0.21 / +0.23 | 0.05–0.09 | ≈0 |
| attn | R | label | 16k | 2 | 0.0006–0.0007 | −0.08 / −0.12 | +0.15 / +0.12 | 0.12–0.13 | ≈0 |
| attn | R | label | 48k | 1 | 0.0002 | −0.04 | +0.19 | 0.05 | ≈0 |
| attn | R | Bayes | 16k | 1 | 0.0003 | −0.06 | +0.17 | 0.08 | +0.03 |
| attn2 | O | label | 16k | 1 | 0.0002 | −0.02 | +0.22 | 0.05 | ≈0 |
| attn2 | R | label | 16k | 1 | 0.0003 | +0.00 | +0.23 | 0.06 | +0.01 |
| relnet | O | label | 16k | 2 | 0.0004–0.0005 | +0.03 / +0.05 | +0.26 / +0.29 | 0.08 | ≈0 |
| relnet | R | label | 16k | 2 | 0.0009–0.0011 | +0.07 / +0.14 | +0.31 / +0.37 | 0.17–0.19 | ≈0 |
| relnet | R | label | 48k | 1 | 0.0002 | −0.01 | +0.23 | 0.06 | +0.01 |
| relnet | R | Bayes | 16k | 1 | 0.0001 | +0.00 | +0.23 | 0.06 | ≈0 |

Source: `logs/dev_summary.json`, `checkpoints/compose/*.json`.

## Learning curves: E_R1 / KL at 1k, 4k, 16k (48k) steps

| run | 1k | 4k | 16k | 48k |
|---|---|---|---|---|
| attn O label s0 | +0.15 / 0.006 | −0.03 / 0.003 | −0.02 / 0.000 | |
| mlp O label s0 | +0.27 / 0.013 | +0.00 / 0.004 | +0.03 / 0.001 | |
| relnet O label s0 | +0.49 / 0.010 | +0.05 / 0.005 | +0.03 / 0.000 | |
| attn R label s0 | **−1.65** / 0.017 | −0.26 / 0.005 | −0.08 / 0.001 | |
| attn R label 48k s0 | −1.65 / 0.017 | −0.22 / 0.005 | −0.28 / 0.002 | −0.04 / 0.000 |
| relnet R label s0 | **−1.54** / 0.022 | −0.04 / 0.007 | +0.07 / 0.001 | |
| mlp R label s0 | **−1.65** / 0.039 | −1.44 / 0.018 | −0.46 / 0.003 | |
| mlp R label 48k s0 | −1.65 / 0.039 | −1.44 / 0.018 | −0.05 / 0.003 | +0.08 / 0.000 |
| mlp R Bayes s0 | −1.37 / 0.022 | −0.17 / 0.002 | +0.01 / 0.000 | |
| attn R Bayes s0 | −0.53 / 0.004 | −0.10 / 0.001 | −0.06 / 0.000 | |
| relnet R Bayes s0 | −1.11 / 0.012 | +0.05 / 0.001 | +0.00 / 0.000 | |

Note: the 16k column of a 48k run is not comparable with a 16k run, because the cosine schedule is three times longer.

## Findings (development)

1. **Learnability.** The relation can be learned from raw ingredients with ordinary labels. Every nonlinear reader eventually
   reaches Bayes on R1, R2 and R3 together: KL ≤ 0.0005, |E_R1| ≤ 0.08, |E| R2∪R3 ≤ 0.12. The relational readers (attention,
   pairwise) do it within 16k steps. The concatenation MLP has not done it at 16k (E −0.46 / −0.42; dm_R1 still has the
   wrong sign) and does it at 48k, or at 16k with Bayes soft targets.

2. **A separable phase comes first.** Every R learner passes through a phase at 1k steps where E_R1 ≈ −1.1 to −1.8. In
   that phase the R1 response has the wrong sign, as in the P-count discount: one more voter in a group is treated as
   evidence against other perfect groups. Learners with m given (O) do not go through this phase; their R1 error is
   *positive* at 1k. The cross-group conjunction is therefore learned *after* the separable, count-based part, and the
   relational architecture shortens that lag. Accuracy and KL are already good during the phase (0.017–0.039 KL), so
   ordinary aggregate metrics hide it.

3. **A linear reader cannot do it even with m available** (O and R identical, KL 0.335). This is an instance of
   Proposition C: a logit rule that is additive over groups gives the same response to a 4→5 edit in every regime. That
   rules out the R1/R2 sign difference, whatever the features. The relation needs a *multiplicative* group-level
   interaction, RHO^{n_g}·m_g·L_g, and not just access to m.

4. **Soft Bayes targets speed up optimization** (mlp: −0.46 → +0.01 at 16k). They add no information about the function
   class, because labels are drawn from the same posterior, so this is a variance-reduction effect.

5. **The family controls are uninformative in condition R by construction** (see above). The preservation question
   (F34 and the R2/R3 regimes) is informative: no successful reader damages F34 (|E| ≤ 0.03).

## Condition X (ordinary tokens, from scratch)

See `TRAINING_AND_OPTIMIZATION.md` §X. It reports a one-block and a three-block Transformer trained on the full token
sequence with labels for 16k steps.
