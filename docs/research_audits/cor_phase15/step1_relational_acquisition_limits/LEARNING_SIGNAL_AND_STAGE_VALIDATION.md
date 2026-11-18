# Learning signal and stage validation (mandate section 6)

**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

Every XV measurement that failed its controls was rebuilt before the freeze, and each rebuilt measure was checked against exact references. Any measure that failed a reference check in development is reported here and was not used.

## 1. Gate acquisition: decision-level and confidence-invariant (D3, D4)

**F_G (gate-flip following)** (`src/xvidiag.py`):
- It is computed over gate flips (the last gate source redrawn so that g switches) that change the Bayes argmax, with the argmax unique before and after.
- It is the fraction of those flips on which the model's argmax moves to the new Bayes argmax.
- Any monotone sharpening or flattening of a fixed decision rule leaves its argmaxes, and therefore F_G, unchanged.

**Reference values** [num, V6, all four neural conditions]:
- F_G = 0 for the gate-free optimum at temperatures 0.5, 1 and 2;
- F_G = 1 for Bayes at the same three temperatures.

**XV's graded G, by contrast,** equals 1/T for Bayes tempered by T, so it is 2.0 or 0.5. It is kept as a secondary, confidence-dependent measure, together with G_conf = G / sB (sB is the model's overall sharpness relative to Bayes). G_conf is 1 for tempered Bayes and 0 for gate-free rules.

**Acquisition (frozen):** three consecutive diagnostic points with F_G ≥ 0.5 **and** KL ≤ 0.75 × the gate-free KL. The KL condition requires that at least a quarter of the gate's value has actually been realised, not merely that some decisions follow.

**Density:**
- 50-step diagnostics over the entire runs of Workstream A (15,000 or 4000 steps);
- 25-step diagnostics over the entire runs of Workstream B.

XV used 250-step spacing after step 2000. Because diagnostics run inline, density does not require dense parameter checkpoints.

## 2. Agreement use, and the floor-conditioned stage (D2, D3)

**A decision-level agreement measure was tried and rejected in development** [dev].
- The obvious analogue of F_G is following on pair-single swaps.
- It gave 1.0 for the order-1 rule as well, and no swap changes the gate-free optimum's argmax without also changing the order-1 optimum's.
- So no decision-level test separates agreement use from source-wise use with this edit **in the arity-3 conditions and WB-high**. Review A (finding 8) shows that in WB-low, 54 of 162 swaps would have allowed one. The rejection was generalised too far.

**A_conf** = XV's agreement index A_p × sB(gate-free optimum) / sB(model):
- It is constant across temperatures for the gate-free optimum (spread < 1e-6), and 0 for the order-1 rule and for Bayes [num, V6].
- The reference value is 1.0, or 1.116 at arity 2 with ν = 0.05.

**Floor-conditioned stage (frozen).** At least 4 consecutive points before acquisition with:
- F_G ≤ 0.1;
- A_conf ≥ 0.5 × reference;
- KL ≤ KL(order 1) − 0.5 · (KL(order 1) − KL(gate-free)), i.e. at least half the agreement benefit realised beyond the best source-wise rule.

It is evaluated **only in models that acquire the gate.** Note (Review A finding 3): it certifies that the gate is not yet *decision-visible* (F_G ≤ 0.1), not that it is unused. A graded partial gate can coexist with it. XV's problem, that a censored model sitting at the gate-free optimum automatically "showed the stage", therefore cannot recur.

## 3. Cue-conditioned probes (D1) (`scripts/xvi_probe.py`)

For one question per episode (3000 episodes), linear probes for g read the residual stream at the inputs of layers 1 and 2 and after layer 2, at the gate tokens, the evidence tokens and the readout.

**Natural probe and its baselines.**
- The natural probe is tested on held-out x₀ = 2.
- Exact baselines from the task law:
  - AGREE: the best prediction of g from [pair agrees];
  - EVID: the best prediction from the full evidence.
- In every condition EVID = AGREE: the evidence carries no information about g beyond agreement. The values are 0.741 at ν = 1/3 and 0.883 at ν = 0.05.

**Decorrelated probe.** It is trained and tested on gate-randomised inputs, where the gate sources are redrawn independently of the evidence regime.
- The evidence then carries **no** information about the visible g, so chance is exactly 0.5.
- Accuracy above 0.5 requires g computed from the gate tokens.
- This is the mandate's "data that separates the gate from agreement".

**Causal use.** Flip transfer of the residual at the evidence tokens or the readout, on Bayes-decisive flips.

**Validation on XV models** [expl, XV checkpoints only]:

| model | natural probe after layer 2 at the readout | EVID | decorrelated probe (sites carrying g) | causal use at the evidence tokens, layer-2 input |
|---|---|---|---|---|
| XV C_s201 at step 4000 (censored; at the gate-free optimum) | 0.75 | 0.741 | 0.48–0.51 (chance) | 0.29 = no-patch baseline |
| XV A_s201 at step 925 (acquired) | 1.00 | 0.883 | 0.96–1.00 | 1.00 |

So the natural probe of a gate-free model sits exactly at the cue baseline, which is what misled XV's first reading of D_s203. The decorrelated probe separates the two cases cleanly.

**Checkpoints near transitions.** The confirmation runs saved parameters every 500 steps. Probes near a transition use checkpoints regenerated by deterministic replay (`scripts/xvi_replay.py`), which was verified bitwise against an archived development snapshot (`logs/expl/replay_check_dev.txt`: max |diff| = 0.0).

## 4. Learning-signal estimator (D5) (`scripts/xvi_gradsig2.py`)

**Quantity.** μ = E[∇L(true batch) − ∇L(the same batch with the gate sources redrawn independently)]. All marginals, labels and evidence are kept, and only the dependence between the visible gate and the evidence regime is removed.

**Estimator.** The unbiased U = (‖Σ D_i‖² − Σ ‖D_i‖²) / (B(B − 1)) for ‖μ‖², with a jackknife standard error over B independent pairs.
- Its expectation is exactly 0 when μ = 0.
- XV's ‖mean D‖ / noise had expectation about 1 under the null; that was the defect.

**Controls on every checkpoint:**
- **null:** the difference of two independent gate-randomisations, so μ = 0 exactly;
- **planted:** the null plus a fixed vector with ‖v‖² = 10 null standard errors, which U must recover. This checks the arithmetic only, not sensitivity (Review B finding 2);
- **natural positive:** the true batch minus the same batch with its labels permuted.

**Interpretation.** U is the squared norm of the expected first-order gate-attributable gradient. A value below the detection limit at a plateau bounds that push. It does not show that the push is zero.

**Validation and results.** See §6.2.

## 5. Calibration and sharpening controls (D3)

Every trajectory point records:
- confidence (mean max-probability), entropy and sB;
- G, G_conf, A_p and A_conf;
- KL, NLL and accuracy;
- the optimizer quantities: learning rate, cumulative learning rate, gradient norm, update norm, parameter norm, and the Adam first- and second-moment norms.

A transition is called abrupt only when F_G, which is invariant to sharpening, changes within a few diagnostic intervals.

## 6. Results (after scoring)

### 6.1 Stages under floor-conditioned, confidence-normalised criteria

**[frozen P5]** STAGE_ROBUST. Every one of the 12 fresh acquirers showed at least 4 consecutive points of agreement use before acquiring the gate:
- WA fresh 4/4: longest runs of 9–54 points at 50-step spacing;
- WB low 4/4: 31–58 points at 25-step spacing;
- WB high 4/4: 16–65 points.

The restart acquirers (4/4) also met the criterion. They are not part of P5.

**Sensitivity [frozen grid].**
- The WB calls are unchanged under every variant.
- The WA fresh call drops to 3/4 when A_conf ≥ 0.7 is required, and to 2/4 when 75% of the agreement benefit is required. In condition C that benefit is only 0.023 nats, so the floor test operates on a small margin there.

**XV reanalysis [expl; downgraded after review].** The same definitions were applied to XV's snapshots (`logs/expl/xv_original_arm.json`):
- Every acquiring arity-3 model and every arity-2 A model (ν = 0.05) showed the stage.
- In XV's condition B (arity 2, ν = 1/3, a weak cue), only 1 of 4 passed (longest runs of 1, 4, 3 and 1 points).
- **But in this weak-cue condition the KL floor sits a few thousandths of a nat from the gate-free value.** A merely mistempered agreement rule fails it (Review B finding 8).
- **So whether XV-B skipped the stage is not established either way.** What is established: with a strong cue (ν = 0.05) or a slow gate (arity 3), the floor-conditioned stage is present in every acquirer [frozen P5 + expl].

### 6.2 Learning-signal estimator: validation and measurements [expl]

**Validation over 31 estimates on 27 checkpoints:**
- The estimator is unbiased by construction, and the null z values stay between −1.10 and +1.13.
- **But their SD is 0.55, not 1.** The jackknife SE is conservative by about a factor of 2, so the z values below understate significance (Review B finding 2).
- The planted signal is recovered at 0.89–1.11 (an arithmetic check only).
- The natural positive control gives z ≥ 18.7.
- **Detection limit on plateaus:** ‖μ‖ ≈ 0.001–0.004, about 1–3% of the full expected gradient (10.5% for D_s202).

**Measurements** (gate-attributable expected-gradient z; B = 48 pairs):

| model | initialisation | plateau, well before onset | about 250–500 steps before first response | at onset | after |
|---|---|---|---|---|---|
| XV C_s201 (censored) | −0.15, −0.81, −0.50 (3 replicates) | −0.17 (1000), −0.66 (2000), −0.36 / +0.25 / −0.73 (4000) | — | — | — |
| XV D_s201 (censored) | — | −0.23, +0.05, +0.15 | — | — | — |
| XVI restart D_s201, D_s202 (censored at total exposure 5.9) | — | −0.01, +0.75 | — | — | — |
| XVI restart C_s201 (first response 1200 after restart) | — | −0.53 (500 after restart) | — | — | — |
| XV C_s204 (first response 2250) | — | +0.03 (1000) | +1.89 (2000) | 14.9 (2250) | 57.2 (2500) |
| XV D_s203 (first response 2000) | — | +0.34 (1000) | **5.17** (1750) | 40.9 (2000) | 46.7 |
| XVI fresh s403 (first response 3000) | — | — | +0.43 (2500) | — | — |
| XVI fresh s404 (first response 3400) | — | +0.31 (1000) | +1.18 (3000) | — | — |
| XV B_s201, arity 2 (first response 425) | — | — | **4.40** (300) | 30.6 (450) | 53.2 |
| XV A_s201, arity 2 (first response 775) | — | — | **7.26** (625) | 28.5 (775) | 58.1 |

**Reading (corrected after review; ADJUDICATION W1):**
- On every plateau measured the gate-attributable expected gradient is **below the detection limit** (|z| ≤ 0.81 with conservative SEs). This covers 3 distinct censored models in 4 states, initialisation, and points 1000–2400 steps before the first response.
- It is **clearly present (z ≥ 4)** 150–250 steps before the first response in 3 of 5 measured transitions (B_s201, A_s201, D_s203).
- It is **marginally above every null value** at 250–500 steps before in 2 more (C_s204 @2000, z 1.89; s404 @3000, z 1.18).
- It grows by one to two orders of magnitude through the transition.
- **This does not foreshadow escape far in advance,** and in that sense it does not predict acquisition timing (question 21). Only 1–2 checkpoints precede each onset, and ordinary first-order escape from a saddle with exponential growth would also look like "zero" at earlier points.
- **The data do not distinguish first-order saddle escape from noise-driven or higher-order escape.** The earlier statement "no expected push toward the gate; escape must be stochastic or higher-order" is withdrawn.

### 6.3 Cue-conditioned probes around transitions [expl]

**Checkpoints.** Exact checkpoints were regenerated by deterministic replay; replays match the archived 500-step snapshots bitwise.

| checkpoint | F_G-type following | natural probe after layer 2, readout (EVID) | decorrelated probe after layer 2: evidence / readout | causal use, evidence tokens (no-patch baseline) |
|---|---|---|---|---|
| fresh s402 @800 | 0.00 | 0.69 (0.741) | 0.49 / 0.50 | 0.29 (0.29) |
| fresh s402 @1000 | 0.00 | 0.72 | 0.49 / 0.52 | 0.29 |
| fresh s402 @1050 | 0.30 | 0.74 | **0.67 / 0.58** | 0.50 |
| fresh s402 @1100 | 0.59 | 0.83 | 0.65 / 0.83 | 0.71 |
| fresh s402 @1200 | 0.99 | 0.99 | 0.89 / 0.98 | 1.00 |
| restart C_s201 @500–1100 | 0.00 | 0.74–0.75 (0.741) | 0.49–0.51 / 0.48–0.49 | 0.29 (0.29) |
| restart C_s201 @1150 | 0.10 | 0.76 | **0.62 / 0.57** | 0.35 (0.26) |
| restart C_s201 @1200 | 0.69 | 0.91 | 0.70 / 0.91 | 0.76 (0.08) |
| restart C_s201 @1300 | 1.00 | 1.00 | 0.90 / 1.00 | 0.98 (0.00) |
| restart D_s201 @4000 (censored) | 0.00 | 0.89 (0.883) | 0.49 / 0.51 | 0.06 (0.06) |
| restart D_s202 @4000 (censored) | 0.00 | 0.89 (0.883) | 0.48 / 0.51 | 0.06 (0.06) |
| WB high s501 @500–650 (arity 2) | 0.01–0.18 | 0.83–0.85 (0.883) | 0.35–0.40 / 0.41–0.46 (below chance; see caveat) | 0.08–0.22 |
| WB high s501 @700 | 0.50 | 0.87 | 0.71 / 0.64 | 0.49 |
| WB high s501 @800 | 0.99 | 1.00 | 1.00 / 0.99 | 0.95 |

**Answer to mandate question 18 [expl; corrected after review, ADJUDICATION W6].**
- **No evidence of gate information beyond the cue before use.**
- Pooled over slots, the natural probe never exceeds the cue baseline before behavioural acquisition, and the decorrelated probe rises only when following and causal use rise: s402 @1050, C_s201 @1150, WB s501 @700.
- **But these probes pool the six question slots, and acquisition is slot-local.**
- With the slot fixed (Review B finding 5), gate information and causal use are already present in one slot of WB_high_s501 at step **600**, and in C_s202 they appear together slot by slot.
- So the slot-level evidence also supports **co-emergence of representation and use**, at an earlier checkpoint than the pooled table shows.
- The censored models never form gate information beyond the cue (natural probe 0.89 against 0.883; decorrelated 0.48–0.51).

**Caveat (arity 2).**
- With the probe trained on x₀ ∈ {0, 1} and tested on x₀ = 2, a representation that encodes the gate **sources** but not g generalises *systematically wrong*. At arity 2 the decorrelated probe therefore falls **below** 0.5 before the transition (down to 0.03 at the gate tokens).
- So only values above 0.5 are read as evidence of g. **The artefact also occurs at arity 3** (Review B finding 6): C_s201 at the gate tokens before onset reads 0.44 and 0.38.
