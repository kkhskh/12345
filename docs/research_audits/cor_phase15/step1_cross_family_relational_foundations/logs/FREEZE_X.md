# FREEZE X: prospective cross-family prediction X1 (interaction-order staircase)

Committed before any confirmation output is inspected. Seeds 3 and 4 started training at 17:20Z. Their checkpoint directories have only been listed by file count, and nothing they contain has been loaded.

## Mathematical argument

The proofs are in CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md; the exact numbers are in `logs/g_rules_exact.json` and `logs/g_order_ceilings.json`.

1. **Relational invisibility.** In GATE, I(g; x_s) = 0 for every single source s (exact). The trust variable is therefore invisible to every single-source statistic. The same holds in T4 for the configuration (g, h).
2. **Interaction order.** Write the Bayes log-posterior in log-linear interaction terms. GATE's Bayes rule is exactly of order 3:
   - log-odds = g·ℓ₂₃ + (1 − g)·ℓ₄₅, where g = [x₀ + x₁ ≡ 0];
   - the expected-NLL-optimal order-3 rule reaches KL 8e-9.

   The optimal rules of order 1 and 2 have exactly **zero gate-flip response**. Both reach accuracy 0.801, the same as the source-wise rule.
   - order 1: NLL 0.462;
   - order 2: NLL 0.431, essentially the gate-free mixture (0.4305), which uses within-pair agreement, the T4-like trust cue.
3. **Swap test.** The order-1 rule is exactly invariant to the swap x₃ ↔ x₅, because the swap preserves the multiset of evidence values. The gate-free (order-2) rule is not.
4. **Principle under test.** Gradient training acquires a target's lower-order interaction components before its higher-order ones; this is the staircase or simplicity-bias hypothesis. Combined with (1) and (2), it predicts a learning sequence for GATE:
   1. an order-1 (source-wise) phase with no relational use;
   2. pairwise-agreement use;
   3. the gate.

   The T4 line has related findings. Investigation VIII [C]: learners first adopt the separable, per-group rule before the cross-group conjunction. T4 from scratch failed in 0 of 3 seeds (latent_reliability), and T4's trust needs high-order, cross-question relations.

## Development evidence (disclosed)

**Coarse loss logs of seeds 0, 1 and 2** (single-batch, every 400 steps).
- At step 400 the losses were 0.448, 0.454 and 0.403; by step 1200 they reached the Bayes level (about 0.18).

**Development seed 1, re-run to step 400 with snapshots** (stream 99_000_000/99_100_000):
- From step 150 to step 400: accuracy 0.80, NLL 0.46–0.49 (the order-1 ceiling is 0.462).
- Gate-flip following was 0.04–0.12.
- The swap-agreement index A rose only from 0.03 to 0.15.

**Thresholds.** The thresholds below come from the exact references, not from fitting to these runs. Whether pairwise use comes before the gate (X1b) is not visible in the development data, which stop at step 400.

## Models and data

- **Models.** GATE seeds 3 and 4. They are new, independently trained models: from scratch, with their own initialisation and their own data streams. They use the identical recipe of seeds 0–2 via `scripts/g_train.py`, which reproduced seed 1's logged step-400 loss of 0.4543 exactly.
- **Snapshots.**
  - every 25 steps up to step 1500;
  - every 100 steps up to step 2000;
  - every 500 steps up to step 4000.
- **Evaluation stream.** 98_500_000, with 3000 episodes (18,000 questions). This stream is unused.
- **Statistical unit.** The model, n = 2. Episode bootstrap intervals are reported for A and G.

## Measures

- **G(t), gate-flip slope.** The flip redraws x₁ so that the gate switches; it is on-support and leaves the evidence unchanged.
  - order ≤ 2 rules: G = 0;
  - Bayes: G = 1.
- **A(t), agreement index.** The slope of the model's swap response on the gate-free rule's swap response, over swaps where that rule changes by more than 0.05.
  - order-1 rule: A = 0;
  - gate-free rule: A = 1.
- **Also recorded:** NLL to the exact posterior, accuracy, and KL against the Bayes, gate-free and source-wise rules.

## Predictions

A model is **competent** if its final KL is ≤ 0.02 and its final G is ≥ 0.9.

**X1a: order-1 phase first.** At least 4 consecutive snapshots (100 steps) at or before step 1500 with accuracy ≥ 0.75, A ≤ 0.3 and G ≤ 0.1.
- Competing explanation: relational and evidential components are learned jointly, so A or G is already large when accuracy first reaches 0.75.

**X1b: pairwise agreement before the gate.** Some snapshot with G ≤ 0.1 and A ≥ 0.5.
- Competing explanations: the gate is learned first or together with agreement; or the model skips order-2 use and jumps from order 1 to order 3.

**Verdict for each prediction:**
- **CONFIRMED** if both models are competent and both satisfy it;
- **FALSIFIED** if any competent model fails it;
- **INCONCLUSIVE** if fewer than 2 models are competent.

## Scorer

`scripts/score_X.py`, which is fail-closed. It refuses unless the frozen hashes match, every snapshot exists and training has finished.

Preflight (`logs/preflight_X.json`):
- **P0:** refuses on the current, incomplete state.
- **P1:** on a synthetic root built only from development seed 1, returns X1a CONFIRMED and X1b FALSIFIED, as constructed.
- **P2:** refuses when a snapshot is missing.

The hashes are in `logs/FREEZE_X_HASHES.txt`.

## Interpretation if falsified

The cause would be one of these:
- **the mathematics**, which cannot fail here because the references are exact;
- **the training principle**, i.e. simplicity bias does not order these components;
- **a model failing to train**, giving INCONCLUSIVE;
- **the measures**: G and A are slopes on the model's own centred log-probabilities, and their meaning is fixed by the exact references above.
