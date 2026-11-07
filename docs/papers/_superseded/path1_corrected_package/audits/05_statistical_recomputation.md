# Statistical recomputation

## Identity checked in source, not on archived arrays

The scoring convention in `src/othello_r2_fiber_spec.py` is

\[
Y = G - Q,
\]

where \(G\) is probability mass on squares that are legal on the intended target and illegal on the recipient, and \(Q\) is mass on squares that are legal on the recipient and illegal on the target. Sets come from the archived simulator legal lists (`legality_sets`), not from a reimplementation of Othello legality at scoring time.

Softmax is standard log-sum-exp. Jensen–Shannon uses natural-log KL after adding \(10^{-12}\) and renormalizing (`js_divergence` in `src/othello_r2_fiber.py`).

Unit tests in this repository check these identities on synthetic probabilities. They do not substitute for recomputation on saved 61-way logits.

## What was not recomputed here

Because Drive NPZ files are absent, this package did **not** independently recompute:

- SPEC \(\Delta_{\mathrm{specific}} = +0.391627\)
- Fixed-JS \(\Delta_{\mathrm{wrong}} = +0.02655\), \(\Delta_{\mathrm{random}} = +0.02297\)
- NST pooled means including \(\Delta_{\mathrm{wrong}} = 0.020057\)
- Pair-level bootstrap intervals from resampling the actual rows
- Probe-probability invariance on the saved 64×3 arrays
- Rank 124 / null 388 from the saved contrast matrix SVD

Those figures remain **session scientific record** or **operator-pasted Colab stdout**. They are reconciliation targets for a future Level 1 run, not products of this package.

## What is taken from operator-pasted stdout (this program)

FRESH Stage 5/6 and VNF Stage 5/6 were pasted by the operator from Colab. The numbers below are transcribed from those pastes. They are not a local NPZ replay, but they are primary session evidence rather than rounded Path 1 brief targets.

FRESH FULL natural-anchor (\(n=32\) triplets):

- \(A_T = 0.0284998707684192\), 95% CI \([0.01651597, 0.04061570]\), win rate 0.8125
- \(A_W = 0.01856145964513889\), 95% CI \([0.00717904, 0.03009127]\), win rate 0.5625
- both-positive \(13/32 = 0.40625\)
- \(C = 0.04706133041355809\)

VNF (\(n=32\)):

- \(I_T = 0.0008076023895302598\), 95% CI \([-0.000940300476329714, 0.0030170128325769344]\)
- \(I_W = 0.0009723259609861756\), 95% CI \([-8.623079802076993 \times 10^{-5}, 0.0022080467274865232]\)

NST pooled stdout (prior session paste, \(n=48\)):

- \(\Delta_{\mathrm{wrong}} = 0.020057007168831593\), CI \([0.01079050562509556, 0.02978427238954672]\)
- \(\Delta_{\mathrm{random}} = 0.031416531839834505\)
- \(\Delta_{\mathrm{mean}} = 0.021998457046033575\)
- \(\Delta_{\mathrm{editor}} = 0.05371273508264925\)
- \(Y_{\mathrm{learned}} = -0.5821099868511178\)
- \(Y_{\mathrm{editor}} = -0.6358227219337671\)

SEM mean-level identity (git-tracked):

- FULL − Editor A = 0.053713
- STAGE_ONLY − Editor A = 0.030099
- ratio = 0.5604

## Statistical unit

| Study | Unit in the preregistered bootstrap | Independence caveat |
|---|---|---|
| SPEC / fixed-JS / NST / SEM | Recipient-game clustered; 48 clusters for 48 pairs | One pair per recipient by design; wrong/random directions share the recipient |
| FRESH / VNF | Triplet; 32 clusters | T and W arms share the recipient; not 64 independent games |
| Multiple analyses of SPEC 48 | Same pairs | Multiplicity: later JS study is not a new sample |

Interventions that share a recipient are not independent games. The clustered bootstrap matches that design. It is not an independent confirmation of an already inspected dataset.

## Calibration is not interchangeable

Equal L2 (SPEC) produced large, unequal JS (correct ~0.279, random ~0.038). Equal JS (fixed-JS, NST) produced unequal residual L2 (correct ~5.30, random ~12.27). VNF used equal L2 with no JS dose selection. These are different estimands. The manuscript does not treat them as one matched strength.

## Coverage versus complete-case means

Fixed-JS required 44/48 complete calibrations. Session record: 48/48. If a baseline fails calibration on most examples, its mean must not be compared with another method’s full-sample mean. The NST gradient control was recorded as JS-ok on only 7/48 rows and is therefore **not** a fair full-sample comparator.
