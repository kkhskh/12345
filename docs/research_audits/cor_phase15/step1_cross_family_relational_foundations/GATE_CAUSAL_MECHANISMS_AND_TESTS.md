# GATE: verified prior findings, new causal experiments, and decision mechanisms

**Models.**
- **s0:** atlas continuation.
- **s1, s2:** relational-mechanism deepening.
- **s3, s4:** new in XIV; untouched until the X1 freeze; see CROSS_FAMILY_CONFIRMATION.md.

All five are trained **from scratch** with the identical recipe: 3 layers, 4 heads, d = 64, 4000 steps, and their own initialisation and data seed. They are independent training replications. They are not fine-tunes of one parent.

**Evidence labels.**
- **[repro]** independent reproduction, on a new stream, of a previously reported observation;
- **[expl]** exploratory, on models s0–s2, which XIV examined while designing its work;
- **[frozen]** the X1 confirmation.

**Scripts:** `scripts/g_mech.py`, `scripts/g_das.py`, `scripts/g_traj_eval.py`.
**Logs:** `logs/gate_mech/`.

## 1. Is the relation actually used? Behaviour against explicit alternatives [repro + num]

Stream 98_200_000; one edited question per episode; n = 2000.

| | s0 | s1 | s2 | exact gate-free rule | exact order-1 rule |
|---|---|---|---|---|---|
| KL(Bayes ‖ model) | 0.0002 | 0.0002 | 0.0002 | 0.249 | 0.280 |
| KL(gate-free ‖ model) | 0.580 | 0.577 | 0.583 | 0 | — |
| gate-flip slope G (Bayes = 1, every order-≤ 2 rule = 0) | 0.993 | 0.990 | 0.997 | 0 | 0 |
| mean \|Δ centred log-prob\|, gate flip | 2.45 | 2.44 | 2.46 | 0 | 0 |
| same, relation-preserving shift (x₀ + k, x₁ − k) | 0.067 | 0.053 | 0.038 | 0 | 0 |
| same, decoy toggle (g kept, [x₀ = x₁] changed) | 0.031 | 0.021 | 0.013 | 0 | 0 |
| decision changes under the shift, on Bayes-unique questions | **0.000** | **0.000** | **0.000** | — | — |
| decision changes under the shift, at exact Bayes ties (18% of questions) | 0.35 | 0.31 | 0.20 | — | — |

**Reading.**
- Every model follows the gate relation and not the two explicit relation-free alternatives.
- The models are exactly invariant, at the decision level, to edits that preserve the relation. The only changes happen at exact Bayes ties, where the posterior is 0.474 / 0.474 / 0.053 because the trusted pair disagrees.
- The response to the relation-preserving shift is 1.5–2.7% of the response to the gate flip.
- This reproduces, on a new stream, the deepening package's "flipping another question's gate changes nothing" and "Bayes-level accuracy". It extends them with explicit order-1 and order-2 alternatives and a relation-preserving control.

## 2. Where is the gate? Presence, access and use [repro + expl]

**Presence.** Logistic probes for g are trained on x₀ ∈ {0, 1} and tested on held-out x₀ = 2, so they measure abstraction rather than lookup. Residuals are taken at the gate tokens G, the mean of the evidence tokens E, and the readout R of the edited question.

**Use.** Flip transfer from residual patching: the residual of the gate-flipped donor is patched at that site and token set. The score is the fraction of Bayes-decisive flips reproduced (351 decisive cases per model).

| site | s0 probe / patch | s1 probe / patch | s2 probe / patch |
|---|---|---|---|
| layer-1 input, G | 0.46 / 0.00 | 0.23 / 0.00 | 0.15 / **1.00** |
| layer-1 input, E | 0.87 / **0.83** | 1.00 / **1.00** | 0.26 / 0.00 |
| layer-1 input, R | 0.23 / 0.19 | 0.36 / 0.00 | 0.34 / 0.00 |
| layer-2 input, G | 0.99 / 0.00 | 0.88 / 0.00 | 0.60 / 0.00 |
| layer-2 input, E | 0.97 / **0.03** | 1.00 / **0.94** | 1.00 / **1.00** |
| layer-2 input, R | 1.00 / **0.97** | 1.00 / **0.08** | 0.31 / 0.00 |
| after layer 2, R | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

**Reproduction.** The patching values reproduce the deepening package's table on a new stream:
- s0: E at the layer-1 input 0.83; R at the layer-2 input 0.97.
- s1: E at the layer-2 input 0.94 (previously 0.93).
- s2: the gate is computed in layer 1 from G (G at the layer-1 input 1.00), and E at the layer-2 input is 1.00.

**New: "decodable ≠ used" occurs in every model.**
- **s0:** g is decodable at E at the layer-2 input (0.97), but patching there transfers 0.03. The decision has already moved to the readout through layer-1 attention.
- **s1:** g is decodable at R at the layer-2 input (1.00), but patching R transfers 0.08. The decision is formed by layer-2 readout attention to E.
- **s2:** g is decodable at G after layer 1 (0.60–0.92), but G patches transfer 0.

This is the GATE analogue of the T4 findings N13 and C3, under the corrected toolkit. A probe identifies information present at a site, not the pathway the decision uses.

## 3. Mediation by the stored gate variable, with ideal references and controls [expl]

**Site.** For each model, the evidence-token residual site whose full patch transfers best: layer-1 input for s0 and s1, layer-2 input for s2.

**Ideal-mechanism reference.** A model that stores g in a single direction at E and decides from it gives:
- interchange along the g direction: 1;
- along a decoy direction: 0;
- along a random direction: 0;
- relation-preserving null donor: 0 decision change.

**(a) Fixed directions** (`g_mech.py`):

| interchange (4 evidence tokens) | s0 | s1 | s2 | ideal |
|---|---|---|---|---|
| full residual patch | 0.86 | 1.00 | 1.00 | 1 |
| 1-d class-mean (encoding) direction of g | 0.17 | 0.21 | 0.33 | 1 |
| 1-d value-conditional direction (g given the token's own value) | 0.27 | 0.27 | 0.36 | — |
| 1-d decoy-relation direction ([x₀ = x₁]) | 0.09 | 0.09 | 0.09 | 0 |
| variance-matched random direction (10 draws; exact variance match) | 0.00 (max 0.00) | 0.00 (max 0.00) | 0.00 (max 0.006) | 0 |
| relation-preserving null donor, g direction (decision change) | 0.050 | 0.037 | 0.026 | 0 |

**(b) Trained subspaces (DAS)**, shared across the 4 evidence tokens, trained on half the episodes and evaluated on the other half (185 decisive cases per model):

| k | s0 | s1 | s2 |
|---|---|---|---|
| 1 | 0.21 | 0.59 | 0.58 |
| 2 | 0.74 | **1.00** | **1.00** |
| 4 | 0.87 | 1.00 | 1.00 |
| 8 | 0.90 | 1.00 | 1.00 |

Controls on the trained subspaces:
- **Random orthonormal subspaces of the same k:** ≤ 0.11 (max over 5). These have 0.06–0.35 of the DAS subspace's variance; that ratio is reported, not matched.
- **The same subspace applied at G or R:** 0.00 in every model and every k.
- **Relation-preserving null donor:** 0.02–0.09 decision change. This equals the models' own tie-change rate (§1), and every change is at Bayes ties.

**Reading.**
- **Sufficiency.** The gate's effect on the decision is fully mediated by a low-dimensional subspace at the evidence tokens of the edited question: k = 2 in s1 and s2; k = 4–8 for 0.87–0.90 in s0.
- **Specificity.** Random, decoy, G-site and R-site controls all fail. The ideal references discriminate here: the null donor's ideal is 0, and the observed value equals the tie rate.
- **The gate is not carried by one linear direction**, unlike the T4 relational carrier (1–2 stable directions; ledger C4). Neither the 1-d encoding direction nor the best trained 1-d subspace suffices. The value-conditional directions are close to the encoding direction (cos 0.34–0.99) and do not suffice either.
- **[spec]** One reading is that the stored variable is the trust of each token rather than g itself: trust is g for tokens 2 and 3 and 1 − g for tokens 4 and 5. A shared subspace then needs one direction per role. This was not tested.


## 3b. The two new models s3 and s4 [expl, after X1 was scored]

`g_mech.py` was run on the final checkpoints, with the same stream as s0–s2. DAS was not run on these two models: the container restart at about 22:34Z killed the jobs, and they were not repeated for time.

| | s3 | s4 |
|---|---|---|
| KL(Bayes ‖ model) / gate-flip slope G | 0.000 / 0.994 | 0.000 / 0.996 |
| relation-preserving shift, mean \|Δ\| (gate flip 2.45) | 0.048 | 0.031 |
| patch transfer: G at the layer-1 input / E at the layer-2 input / R at the layer-2 input | 1.00 / 1.00 / 0.00 | 0.95 / 1.00 / 0.00 |
| full E patch / 1-d g direction / value-conditional / decoy / variance-matched random | 1.00 / 0.28 / 0.31 / 0.01 / 0.00 | 1.00 / 0.31 / 0.28 / 0.13 / 0.00 |
| null-donor decision change | 0.036 | 0.021 |

Both new models have **s2's layout**:
- the gate is computed in layer 1 at E from the gate tokens;
- the gate is stored at E at the layer-2 input;
- the decision is formed by layer-2 readout attention.

Across five models there are therefore three layouts: s0 (early-readout-like), s1, and s2/s3/s4. In every model:
- the gate is mediated by the evidence tokens;
- no single direction suffices;
- decoy and random controls fail;
- the shift response is small.

## 4. Decision mechanism by model (comparison levels, mandate §8.5)

| level | shared by s0–s2? | evidence |
|---|---|---|
| behaviour (Bayes-level, gate-following, relation-invariant) | yes, in all five models | §1, §3b |
| abstract computation (a trust variable derived from the relation, stored at the evidence tokens, then combined with the evidence at the readout) | yes | §2 patches, §3 DAS |
| neural architecture of the route | **no** | gate computed in layer 0 (s0, s1) or layer 1 (s2); decision moved to the readout at layer 1 (s0) or by layer-2 readout attention (s1, s2) |
| individual circuit components | **no** | the deepening package found different heads per seed (L0-h1/h2, L1-h3; L1-h2 + L2-h0, L2-h1 + L2-h3, L2-h1) |

## 5. Comparison with T4

| property | T4 | GATE | same or different |
|---|---|---|---|
| trust invisible to single sources | yes (Proposition 1) | yes (exact I = 0) | **same** (mathematical) |
| exact source-wise ceiling far below Bayes | KL 0.33, agreement with Bayes 0.61 | KL 0.28, accuracy 0.80 against 0.90 | **same** (mathematical) |
| trust stored at source / evidence tokens and combined at the readout | the mixture organisation (N19, N21); the early-readout organisation instead pools relational evidence into the readout | all 3 models store it at E. s0 then moves the gated evidence to the readout at layer 1, an early-readout-like layout | **partly same**. The T4 mixture/early dichotomy has an analogue in *where the combination happens*, not in *where trust is stored* |
| dimensionality of the sufficient carrier | 1–2 stable directions | ≥ 2 (k = 1 insufficient in all three) | **different** |
| decodable ≠ used | yes (N13; C3 with the variance caveat) | yes, in every model (§2) | **same** (methodological, H5) |
| combination operator (power-mean ρ) | fitted ρ differs by organisation; the reading is non-unique (XIII-B) | **not identifiable** for Bayes-optimal behaviour (Proposition 6, near-trivial) | **different**: depends on trust uncertainty, which GATE lacks (H4) |
| relational cue screened off by a deterministic relation | no; agreement *is* the trust evidence, non-monotone | yes (Proposition 5) | **different** |
| learnable from scratch | the M Transformer no (0/3); DeepSets yes (3/3) | yes (s0–s2; s3 and s4 in CROSS_FAMILY_CONFIRMATION.md) | **different** for the Transformer. The interaction-order explanation is **not** supported (Proposition 4, corrected); the source's own explanation is the non-monotone agreement map |

## 6. Not done

- **A structurally relevant training-time intervention in GATE** (blocking evidence-token attention to the gate tokens early in training) was not run, for compute reasons. The T4 early-block results therefore have no GATE counterpart in XIV.
- **A cross-task interchange** is impossible, because the inputs differ. The T4–GATE comparison is mathematical and at the level of intervention-response structure, as the mandate requires, not a mapping of subspaces across tasks.
