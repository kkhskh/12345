# Learning dynamics and the origin of the error

**Sources.**
- Investigation VI's archived transfer snapshots (steps 0, 250, 500, 1000, 2000, 4000), re-evaluated on the dev set with an
  input-route split: `scripts/dynamics_snapshots.py` → `logs/dynamics_snapshots_xP80.json`. Parents: P80, P81, P83, T70, T71,
  T73.
- Development repair trajectories: `checkpoints/dev/*.json`.
- Confirmation trajectories at steps 1000 and 4000: `checkpoints/conf/*.json`.

Status: development, except the confirmation trajectories, which are descriptive (the frozen evaluation was at step 4000 only).

## 1. When the wrong sign appears

**Step 0** (pretrained layers 0–1, fresh layer 2, inherited readout): R1 Δm ≈ 0 in every parent (|Δm| ≤ 0.012). The output is
near-uniform (KL 0.44–0.86). **The error is not present before layer 2 is trained.**

**Raw R1 response Δm_R1 over transfer:**

| parent | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| P80 | −0.05 | −0.09 | −0.09 | −0.44 | −0.66 |
| P81 | −0.20 | −0.44 | −0.64 | −0.53 | −0.48 |
| P83 | −0.06 | −0.29 | −0.25 | −0.53 | −0.69 |
| T70 | +0.00 | +0.07 | +0.15 | +0.16 | +0.13 |
| T71 | +0.05 | +0.19 | +0.16 | +0.13 | +0.14 |
| T73 | +0.20 | +0.31 | +0.27 | +0.24 | +0.24 |

In P-count parents the wrong sign appears within the first 250–500 steps. In P80 and P83 it **deepens** while KL falls
(P80: 0.41 → 0.10). In P81 it peaks at step 1000 (−0.64) and eases slightly by step 4000 (−0.48), while KL keeps falling
(review R2).
T1 parents get the right sign from the start.

## 2. Which route develops

Margin change from switching only the readout-token input copy (z_t), versus only the other 42 tokens. For R3, where both arms
discount correctly:

| parent | readout-token route @250 → @4000 | other-token route @250 → @4000 |
|---|---|---|
| T70 | −0.04 → −0.03 | **−1.15 → −1.96** |
| T71 | −0.09 → −0.14 | **−1.65 → −1.98** |
| P80 | **−0.08 → −1.41** | −0.00 → −0.35 |
| P81 | **−0.29 → −1.46** | −0.16 → −0.76 |
| P83 | **−0.02 → −1.82** | 0.00 → −0.15 |

**The two arms learn the legitimate discount in different ways.**
- **T1 parents** learn it almost immediately (by step 250) through the source tokens. Their frozen source states already expose
  "another group is perfect" linearly (probes).
- **P-count parents** learn it slowly, mainly through the readout token's own count-bearing state. That route is read by a
  query head and by the MLP, and in P-count it carries the same discount into R1:

  | parent | R1 readout-token route @250 → @4000 |
  |---|---|
  | P80 | −0.05 → −0.47 |
  | P81 | −0.18 → −0.35 |
  | P83 | −0.05 → −0.71 |

  The other-token route in P-count grows late and only partly.

**Reading.** The error is the side-effect of the route the P-count solution uses to learn the common, correct discount. On the
readout-token route, R1 and R3 move together. That is the separable pattern of Theorem 1. The cross-group information that would
separate them is not available there.

## 3. Causal changes to the training problem (same compute: 4000 steps, same streams)

**Mean R1 excess E over the 8 confirmation lineages:**

| condition | @1000 | @4000 | R2 / R3 at 4000 | KL at 4000 |
|---|---|---|---|---|
| natural sampling (C1) | −0.695 | −0.724 | +0.52 / −0.07 | 0.070 |
| balanced decision classes (W_bal) | +0.131 | +0.176 | **+1.46 / +1.24** | 0.149 |
| + m feature (injm) | **+0.208** | −0.029 | −0.16 / +0.10 | 0.039 |
| + n feature (injn) | −1.874 | −1.361 | −0.10 / −0.71 | 0.034 |

**Development trajectories of injm** (E_R1 at 250 / 500 / 1000 / 2000 / 4000):

| parent | E_R1 trajectory |
|---|---|
| P80 | −0.22 / +0.47 / +0.59 / +0.38 / +0.23 |
| P81 | −0.33 / +0.13 / −0.07 / −0.27 / −0.23 |
| P83 | −0.29 / −0.11 / +0.15 / +0.20 / +0.18 |

The repaired models over-credit R1 at intermediate steps and drift back towards a smaller positive value, or slightly below,
by step 4000.

**What the training interventions establish.**
1. **The R1 sign responds to loss composition.** The balancing classes are defined by the same relation, so this is not a
   relation-free baseline (review R3-5). Equalising the decision classes (R1-type examples about 12× more frequent)
   flips it. It does so by over-crediting perfect agreement everywhere, not by learning the interaction. Rarity alone is
   therefore *not* the explanation: balancing exposes the R1 class but moves the model along the separable family (Theorem 1)
   rather than off it.
2. **Supplying the interaction statistic m fixes the sign without that cost.** Supplying the separable statistic n makes the
   sign worse while improving KL. The binding constraint is the *form* of the available information (separable vs relational),
   not the amount of R1 data or the training signal.
3. **Gradient conflict was not measured directly.** A training gradient pointing one way does not establish what the optimiser
   converges to, so no gradient-level claim is made.

## 4. Error origin: strongest supported statement

The P-count R1 reversal arises during transfer. It is a learned, structured approximation error with three ingredients:
- The frozen P-count states present each group's agreement count separately, at the readout token, in an easily read form.
- They do not present the cross-group eligibility relation in a form the one-block downstream can read.
- A separable perfect-agreement discount fits about 97 % of decisions well. Under Theorem 1 it necessarily gives R1 the same
  sign as R3.

T1 states expose the cross-group relation at the source tokens, so the T1 learner takes a route where the relation is available.

## 5. Post-review additions (development)

- **Extra training (3× budget, P80).** R1 E −0.68 at 12,000 steps; the sign is still wrong.
- **Generic ingredient adapter (injna).** R1 becomes worse (P80) or is unchanged (P81).
- Both support the reading that the binding constraint is the *conjunction*, not the budget or the ingredients. With n = 1–2
  development parents, this is **not** confirmed.
