# Parameter- and optimizer-state transplants (mandate Workstream A)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels.**
- [proof]: follows from construction.
- [num]: numerical check.
- [dev]: development data on the matched pairs E801 / M801.
- [frozen]: confirmation on D = 821–823, scored by the frozen `score19.py`. See `FROZEN_CONFIRMATION_AND_STATISTICS.md`.

**What a transplant result is.** It is a property of the continuation map Φ from step-100 states to step-1211 organisations. It locates which coordinates of the step-100 state difference carry the selection. It is **not** a natural mediation analysis and it does not identify a mechanism (DESIGN §2 L3; pre-review A10).

## A1. Matched trajectories

For each pair, B (plain) and I (block: gof B = 1e9 on own0 for E or own2 for M, steps < 100) start from the same parent, the same fresh layer-2 / WA initialisation (seed L = W) and the same data order D.

Saved at step 100:
- θ (44 tensors);
- AdamW state: two counters, μ and ν;
- the data RNG and the loss history;
- run_config.json with the FX_* environment and code hashes;
- window_loss.json.

[num] Checked equal in B and I: the RNG state and the step counter. compose19 asserts both.

**Parameter-difference inventory** (`logs/dev/state_diff_*.json`; layer-0 cosine diagnostics in `logs/dev/s100diag_*.json`). [dev]
- **E801:** every parameter group differs, at 0.3–1.1 of B's own movement from initialisation. The largest differences are in layer 0 (ratio ≈ 0.8–1.1).
- **M801:** the largest differences are in L2_OV (0.91) and L2_QK (0.73). The upstream layers also differ substantially.
- **Moments:** relative differences of 0.3–1.95 in μ and 0.05–0.27 in ν.

**Optimizer-state hashes.** Every composed state's input and output SHA-256 is in its provenance.json. Every continuation records the hashes of its start state in run_config.json.

## A2. Full-state interchange (machinery)

- [proof] fullB and fullI are the native states (L1).
- [num] Branch continuation from a composed state is bitwise identical to the uninterrupted run (step-150 difference 0.0 in dev; step-111 difference 0.0 in pre-review B).
- [num] compose19's perturbed composition is identical across PYTHONHASHSEED values. It refuses to overwrite a different state.
- In confirmation, the scorer asserts on every pair that composed fullB equals native B bit for bit, in θ, moments and RNG, and that fullI's θ equals I's.

## A3. Parameters against optimizer history [dev]

| arm | E801 φ | M801 φ |
|---|---|---|
| paramI_optB (I's θ, B's moments) | 0.97 | 1.03 |
| paramB_optI (B's θ, I's moments) | −0.01 | 0.09 |
| B←I L0 with I's L0 moments | 0.91 | — |
| B←I L0 with B's moments | 0.74 | — |
| frz against frzp (identical θ_100, different moments) | −0.01 against 0.57 | −0.03 against 0.01 |

**Reading [dev].**
- At the full-state level the parameters carry the selection. The moments alone do nothing.
- The moments are not inert, though. Inside a composite they shift φ by 0.17 (layer-0 moments, E).
- The frz/frzp contrast shows a *natural-trajectory* moment effect on E. Both runs have exactly the same parameters at step 100. Only frz's held moments, combined with the shared bias-correction counter, produce oversized steps after release. Reviewer B measured 1.7–2.4× the baseline step length over steps 111–200. Those steps undo the partial switch that frzp keeps.

## A4–A6. Parameter-group transplants: sufficiency, necessity, interaction [dev]

**E801** (target layer = layer 0):

| groups from I into B (sufficiency) | φ | groups from B into I (necessity) | φ |
|---|---|---|---|
| emb | −0.00 | emb | 0.99 |
| **L0** | **0.91** | **L0** | **−0.00** |
| L1 | 0.01 | L1 | 0.96 |
| L2 | 0.01 | L2 | 0.96 |
| WA + final LN | −0.00 | WA + final LN | 1.00 |
| L0_QK | 0.10 | L0_QK | 0.95 |
| L0_OV | 0.00 | L0_OV | 0.95 |
| L0_MLP + LN | 0.01 | L0_MLP + LN | 0.92 |
| L0_QK + OV | 0.61 | | |
| L0_OV + MLP + LN | 0.35 | | |
| L0_QK + MLP + LN | 0.46 | | |

- On E, the layer-0 state is sufficient and necessary.
- No layer-0 tensor class is sufficient or necessary alone. Every pair of classes is partial. The layer-0 tensor classes act jointly: an **identified combination**, not a single dominant parameter group.
- The localisation to layer 0 is close to tautological, because the E route that is measured lives in layer 0 (pre-review A10). The informative part is the joint action of the sub-groups.

**M801** (target layer = layer 2):

| B←I | φ | I←B | φ |
|---|---|---|---|
| L0 | 0.22 | L0 | 0.28 (HYBRID) |
| L1 | 0.11 | L1 | 0.99 |
| L2 (the blocked layer) | 0.18 | L2 | **1.00** |
| emb, WA + final LN | 0.00, 0.00 | emb, WA + final LN | 1.00, 1.00 |
| **L0 + L1** | **0.97** | | |
| **L0 + L2** | **0.99** | **L0 + L2** | **0.12** |
| L0 + L1 + L2 | 1.00 | | |

- On M, the blocked layer's own state is **not necessary**.
- Layer 0 is necessary in the sense that restoring it removes most of the switch, but it is not sufficient alone.
- Layer 0 becomes sufficient together with **either** layer 1 or layer 2. The partner layers are substitutable.
- Interaction: no single layer is sufficient, and the pair {L0, partner} is.

## A7. Perturbation, interpolation and mismatch controls [dev]

| control | E801 φ | M801 φ |
|---|---|---|
| matched-norm random direction, all groups (2 seeds) | −0.00, −0.00 | 0.01 |
| matched-norm random direction, L0 only | −0.00 | — |
| interpolation θ_B + α(θ_I − θ_B) on L0, α = 0.25 / 0.50 | −0.01 / −0.01 | — |
| foreign donor: L0 from another baseline trajectory (D802) | −0.02 | — |

**Reading [dev].**
- The selection needs the *direction* of the I − B layer-0 difference, not its size: random directions of matched norm do nothing.
- It also needs *most* of the difference: half of it does nothing. So the continuation map is strongly non-linear (threshold-like) along the I − B segment.
- A mismatched but natural layer-0 state does not switch E, and the reciprocal I←B arms restore the B outcome. Symmetric "mismatch" therefore does not explain the E result.

**State compatibility** (`logs/dev/compat_dev.json`). Composites can be worse than both sources at step 100: B←I L0 on E has KL 0.540 and accuracy 0.418, against B 0.459/0.560 and I 0.536/0.443. All continue to competent models (KL ≤ 0.042).

## Frozen confirmation (D = 821–823; 3 pairs per base; 1 parent)

**Scoring.** Frozen scorer `logs/conf/score_c19.json`. All 6 pairs were evaluable and all models competent. Values are φ (pair 1 / 2 / 3); the band is the frozen verdict.

| arm | E | band | M | band |
|---|---|---|---|---|
| paramI_optB | 0.98 / 0.98 / 0.98 | SUPPORTED | 1.00 / 0.98 / 0.98 | SUPPORTED |
| paramB_optI | 0.01 / 0.01 / −0.00 | SUPPORTED (nil) | 0.04 / 0.19 / 0.02 | SUPPORTED (nil, margin 0.25) |
| B←I L0 | 0.98 / 0.97 / 0.99 | SUPPORTED (sufficient) | 0.26 / 0.33 / 0.38 | SUPPORTED (alone insufficient) |
| I←B L0 | 0.00 / 0.02 / 0.00 | SUPPORTED (necessary) | 0.32 / 0.43 / 0.75 | PARTIAL ("restoration reduces" holds on 2 of 3) |
| I←B L0_QK | 0.98 / 0.96 / 0.96 | SUPPORTED (not necessary alone) | — | |
| I←B L0_OV | 0.97 / 0.97 / 0.96 | SUPPORTED (not necessary alone) | — | |
| B←I L0+L1 | — | | 0.99 / 0.99 / 0.97 | SUPPORTED (sufficient) |
| B←I L0+L2 | — | | 0.99 / 0.98 / 0.97 | SUPPORTED (sufficient) |
| I←B L0+L2 | — | | 0.06 / 0.13 / 0.06 | SUPPORTED (restoration prevents) |
| randnorm (E: L0; M: L0+L2), norm-matched only | 0.00 / 0.01 / −0.00 | SUPPORTED (nil) | −0.08 / 0.00 / −0.00 | SUPPORTED (nil) |

**State compatibility of the composites at step 100** (`logs/conf/compat_c19.json`, stream 94 800 000).
- **E.** B←I L0 composites are worse than both sources (KL 0.528–0.536 and accuracy 0.39–0.41, against B 0.451–0.456 / 0.53–0.55 and I 0.526–0.529 / 0.43–0.44). They still continue to competent switched models.
- **M.** Every composite, and both sources, is functionally almost indistinguishable at step 100:
  - KL 0.425–0.445;
  - early-route necessity 0.17–0.31;
  - mixture-route necessity ≈ 0.

  **The M selection is therefore carried by a step-100 state difference that is not yet visible in the step-100 function.** It decides the later race.

**What is identified [frozen, state level].** The scope is one parent and 3 data orders per base.

- **E.**
  - The layer-0 parameter state at step 100 is sufficient and necessary for the switch.
  - Neither its QK part nor its OV part is necessary alone. [dev] No single tensor class is sufficient alone: the layer-0 tensor classes jointly carry the selection.
- **M.**
  - Neither layer 0 nor the blocked layer 2 is sufficient alone.
  - Layer 0 together with *either* layer 1 or layer 2 is sufficient.
  - Restoring layer 0 and layer 2 to B prevents the switch.
  - Restoring layer 0 alone reduces the switch on 2 of 3 pairs, but not on M823. [dev] Restoring layer 2 alone does nothing.
  - So the M selection is a **combination** involving layer 0 and at least one downstream partner. The partners are substitutable for sufficiency.
- **Both bases.**
  - The parameters, not the Adam moments, carry the selection at the full-state level.
  - A random direction of the same per-group size does nothing.

**What is not identified.**
- How the intervention produces these states: see the training-arm reports.
- Whether the composites' trajectories resemble natural ones: they are synthetic states (L3).
- Anything beyond uP160.
