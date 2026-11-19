# XIX confirmation design (draft for pre-experiment review)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Stage.** This is a draft for the stage-4 reviewers.
- Nothing in it is frozen until `FREEZE_XIX.json` exists and is committed.
- Development evidence is labelled [dev]. Its source is the E801 / M801 matched pairs: parent uP160, data order 801, development measurement stream 86 100 000.

## 1. Objects and notation

- **Training state at step k:** S_k = (θ_k, m_k, v_k, c_k, r_k).
  - θ: 44 parameter tensors;
  - m, v: AdamW first and second moments;
  - c: step counter, which fixes the learning-rate schedule;
  - r: data-stream RNG state.
  - optax AdamW keeps no other state (scale_by_adam count, add_decayed_weights, scale_by_learning_rate). Weight decay is 1e-4; the schedule is a deterministic function of c.
- **Matched pair.** B and I are trained from the same initialisation, data order and r for 100 steps.
  - I applies the validated route block during those 100 steps:
    - E base: gof B = 1e9 on own0 (layer 0, readout → own-question sources);
    - M base: own2 (layer 2).
  - Verified [dev]: r_100^B = r_100^I, c_100 = 100 for both, and only θ, m, v differ.
- **Continuation.** For any composed state S, Φ(S) is the step-1211 model obtained by *plain* training from S. Φ is deterministic: single-threaded XLA, bitwise reproduction verified, `logs/` branch check at step 150 = 0.0.
- **Outcome.** O(S) = (nec_E, nec_M, KL, acc, f) of Φ(S), measured by `src/xix_score.py`. The primary scalar is φ_nec(S) relative to the pair's anchors Φ(S_B) and Φ(S_I).

## 2. What is guaranteed by construction, and is therefore not evidence

**L1 (Markov mediation).** The window intervention affects the outcome only through S_100: Φ(S_100^I) is the intervention run.
- So "the full intervention state transplanted into a baseline trajectory reproduces the switch" is a **machinery check**, not a finding.
- The same holds for fullB, and for paramB_optI = fullB when the moments do not matter.
- Every full-state control is scored as **implementation validation only**. No hypothesis is attached to it.

**L2 (partition duality).** For any partition of the parameter groups into G and Gᶜ (with the moments following the parameters), B←I_G and I←B_{Gᶜ} are the *same* composed state.
- So "G is sufficient" and "Gᶜ is necessary for the B outcome" are one observation.
- Counting them separately would double-count. The scorer treats them as one unit.

**L3 (no natural mediation).** A composed state (θ_B with group G from θ_I) is generally not on any natural training trajectory.
- A transplant result answers "which coordinates of the state difference, when imposed, determine Φ". It does not answer how the intervention produced them.
- Interpretation is limited accordingly (mandate A2, A7). State compatibility is measured for every composed state: step-100 KL, accuracy and route necessity (`scripts/compat19.py`).
  - [dev] All E801 composed states lie between the two source states (KL 0.459–0.540 against sources 0.459 and 0.536).

**L4 (optimizer-state transplants are synthetic).**
- paramI_optB pairs I's parameters with B's moments; paramB_optI does the reverse. Neither state occurs naturally.
- Adam's per-coordinate step is m̂/(√v̂ + ε). Moments taken from a different trajectory change the *direction* of the first post-transplant steps only to the extent that m and g are misaligned; the bias-corrected moments decay with β1 = 0.9 (time constant ≈ 10 steps) and β2 = 0.999 (≈ 1000 steps).
- So a nil optimizer effect is expected *unless* the first tens of steps are decisive. This is a genuine empirical question, not a guaranteed result:
  - the moments differ by a relative 0.3–1.95 (mu) at step 100 (`logs/dev/state_diff_*.json`);
  - the E801 dev results show no optimizer effect, but that is one pair.

## 3. Competing explanations (mandate §1.2) and what each predicts

Each row says what each explanation predicts for each arm in a fresh matched pair. "switch" means φ ≥ 0.75 with the I-anchor causal class; "none" means φ ≤ 0.25 with the B class.

| | X1 information-dependent learning of the target layer | X2 route competition carried by non-target layers | X5 downstream adaptation | X6 optimizer history | X8 general disturbance |
|---|---|---|---|---|---|
| paramI_optB | switch | switch | switch | **none / partial** | switch |
| paramB_optI | none | none | none | **switch / partial** | none |
| B←I target layer | **switch** | none | none | — | none, unless the disturbance is concentrated in that layer |
| I←B target layer (necessity) | **none** | switch | switch | — | — |
| B←I downstream layers | none | partial | **switch** | — | — |
| matched-norm random direction in the target layer | none | none | none | none | **switch** |
| foreign baseline donor of the target layer | none | none | none | none | **switch (mismatch)** |
| training-time matched off-target disturbance | none | **switch if the competition is generic** | — | — | **switch** |

The explanations that remain after the transplant tests concern *how* the intervention produced the target-layer state:

| arm (training-time, steps < 100) | X3 value learning through T | X4 routing learning of T | content-available-but-not-learned | content-loss only |
|---|---|---|---|---|
| vsg (forward and routing intact; value gradient through T removed) | **switch** | none | **switch** | none |
| vrs (content rolled; gradient kept) | partial | none | none | **switch** |
| vrg (content rolled; value gradient stopped) | switch | none | switch | switch |
| gsg (T-side logit gradient removed; complement-side routing intact) | none | partial (half the signal) | none | none |
| frz target-layer QK (all routing learning of that layer frozen; non-edge-local) | none | **switch** | none | none |

[dev] vsg gave a **hybrid** on both bases: E φ 0.38, M φ 0.43. That pattern supports neither "value learning alone" nor "routing learning alone". It is the reason the factorial content × value-gradient arms (vsg / vrs / vrg) are all in the confirmation.

## 4. Development results that shape the design [dev; one matched pair per base, D = 801]

Full table: `logs/dev/dev_table19.json` (measure18, stream 86 100 000). φ is measured against the fullB / fullI anchors. Every run is competent.

| arm | E801 φ (class) | M801 φ (class) |
|---|---|---|
| paramI_optB / paramB_optI | 0.97 (M) / −0.01 (E) | 1.03 (E) / 0.09 (M) |
| B←I emb, L1, L2, WA+final_LN | ≈ 0 each | 0.00, 0.11, 0.18, 0.00 |
| I←B emb, L1, L2, WA+final_LN | ≈ 1 each | 1.00, 0.99, 1.00, 1.00 |
| **B←I L0** | **0.91 (MIXTURE)** | 0.22 (MIXTURE) |
| **I←B L0** | **−0.00 (EARLY)** | **0.28 (HYBRID)** |
| B←I L0+L1 / L0+L2 / L0+L1+L2 | — | **0.97 / 0.99 / 1.00** |
| I←B L0+L2 | — | 0.12 |
| B←I L0 with B's moments for L0 | 0.74 | — |
| B←I L0_QK / L0_OV / L0_MLP+LN | 0.10 / 0.00 / 0.01 | — |
| I←B L0_QK / L0_OV / L0_MLP+LN | 0.95 / 0.95 / 0.92 | — |
| B←I L0_QK+OV / L0_OV+MLP+LN | 0.61 (H) / 0.35 (H) | — |
| matched-norm random direction, all groups | −0.00 | 0.01 |
| matched-norm random direction, L0 only | −0.00 | — |
| training: vsg / vrs / vrg | 0.38 (H) / 0.99 / 0.99 | 0.43 (H) / 1.00 / 1.00 |
| training: frz target-layer QK | −0.01 | −0.04 |
| training: matched off-target control (E msk oq0 ratio 1.06; M gof oq2 b 1.0 ratio 1.08) | −0.01 | −0.17 |

**Development reading.** This is interpretation, to be tested.

- On E, the layer-0 parameter state at step 100 is sufficient and necessary for the switch. No single layer-0 tensor class is either; they act jointly.
- On M, the layer-0 state is necessary (restoring B's layer 0 leaves φ 0.28). It is not sufficient alone (0.22). It becomes sufficient together with *either* layer 1 or layer 2, so the partner layers are substitutable.
- Layer 2 (the target layer on M) is not necessary.
- Neither base shows an optimizer-state effect at the full-state level.
- Random directions of the same per-group norm do nothing.
- Freezing all routing learning of the target layer does nothing.
- Destroying forward content through T switches regardless of the value gradient (vrs = vrg).
- Removing only the value gradient through T gives a stable partial result (vsg).

## 5. Confirmation arms (`logs/conf_spec19_draft.json` → `scripts/conf_setup19.py`)

**Independent units.**
- Six fresh matched pairs: data orders D = 821, 822, 823 (never used before), on both bases.
- Parent: uP160 only, so there is **one independent parent**. No parent-level generalisation is claimed.
- Measurement: frozen stream 94 700 000 (never used before), with `scripts/measure19.py` (code and reference hashes in the design).

**Per pair: 15 continuations (100 → 1211, plain training).**
- **Transplant arms (9).** Composed from the pair's B and I step-100 states:
  - fullB and fullI: anchors and machinery check, no hypothesis;
  - paramI_optB and paramB_optI;
  - B←I L0 and I←B L0;
  - B←I L0+L1 and B←I L0+L2;
  - randnorm_L0: B plus a random direction with ‖·‖ = ‖θ_I − θ_B‖ in each layer-0 group, with B's moments.
- **Training arms (6).** Each continues its own step-100 window state:
  - vsg, vrs, vrg, gsg on T;
  - frz of the target-layer QK;
  - ctl: E msk oq0, M gof oq2 b = 1.0.

**Compute.** 48 step-100 runs (≈ 1 min each) and 90 continuations (≈ 12.5 min each on 2 lanes) ≈ 10 h.

## 6. Frozen predictions (draft)

Notation: [lo, hi] is the φ band. Scoring rules are in `scripts/score19.py` (SUPPORTED / PARTIAL / CONTRADICTED / INCONCLUSIVE, min 2 evaluable pairs per base).

| id | base | prediction | explanation it is able to reject | status before confirmation |
|---|---|---|---|---|
| P1_E_L0_sufficient | E | B←I L0 ∈ [0.75, 1.25] | X2/X5 (non-target or downstream carriers) | replication of 1 dev pair |
| P2_E_L0_necessary | E | I←B L0 ∈ [−0.25, 0.25] | "several alternative carriers" (non-unique sufficiency) | replication |
| P1_M_L0_insufficient | M | B←I L0 ∈ [−0.25, 0.50] | single-layer mediation on M | replication |
| P2_M_L0_necessary | M | I←B L0 ∈ [−0.25, 0.50] | "layer 2 (the blocked layer) carries the M switch" | replication |
| P1_M_L0L1_sufficient / P1_M_L0L2_sufficient | M | ∈ [0.75, 1.25] each | "a specific partner layer is required" | replication |
| P3_param_carries / P3_opt_alone_nil | both | paramI_optB ∈ [0.75, 1.25]; paramB_optI ∈ [−0.25, 0.25] | X6 (optimizer history) | replication |
| A7_randdir_L0_nil | both | randnorm_L0 ∈ [−0.25, 0.25] | "any layer-0 displacement of this size switches" | E: replication; M: **new** |
| P6_matched_ctl_nil | both | ctl ∈ [−0.25, 0.25], only for pairs whose step-100 distance ratio lies in [0.67, 1.5] | X8 (general disturbance at matched state displacement) | replication |
| P4_vrs_switch / P4_vrg_switch | both | ∈ [0.75, 1.25] | "the switch needs the value gradient to be removed" (vrs keeps it) | replication |
| P4_vsg_partial | both | vsg ∈ [0.25, 0.75] | both pure readings: X3 (value-learning-only predicts a switch) and content-only (predicts none) | replication |
| P5_frz_nil | both | frz ∈ [−0.25, 0.25] | X4 (routing-parameter learning at the target layer) | replication |
| P5_gsg_exceeds_frz | both | gsg − frz ∈ [0.25, 1.5] | "gsg acts by removing routing learning" (frz removes all of it) | **new on these units**; XVIII c18 gsg values are on other seeds |
| P7_E_L0_functional, P7_vrs_functional | E / both | functional class switches with the causal class (category BOTH, not CAUSAL_ONLY) | Claim-1-only (route reorganisation with an unchanged functional rule) | replication |

**Admissibility (mandate §3.3, §11).**
- None of these is guaranteed by construction (§2). The full-state anchors carry no hypothesis.
- All but two are **replications of single-pair development observations**. The mandate rejects predictions "already known from development". The reviewers must decide whether replicating on fresh independent units is admissible.
- The design's position: a single-pair observation is not knowledge of the population of matched pairs. Each prediction is reported as a replication, never as an independent discovery.
- P6 and P5 are the two predictions able to eliminate an XVIII-open explanation (general disturbance; routing learning) even if every replication holds.

## 7. Scoring, sensitivity, integrity

- **Primary:** φ_nec against the pair's fullB and fullI anchors. Pair evaluability: both anchors competent and causally pure with opposite classes. No behavioural gate.
- **Frozen sensitivity variants:** delta0, delta20, phicut70, phicut80, eps003, eps03, evalB, thr_pm05. These are reported but never replace the primary.
- **Fail-closed provenance:** code hash; reference hash; stream; step; exactly one record per run; checkpoint hash; transplant provenance (groups, opt, perturb seed, composed and source hashes); training-arm branch-source hashes.
- **Preflight passed:** `logs/scorer_validation_summary19.json` and `logs/scorer_planted19.json` (ALL PASS, including the matching gate and every sensitivity variant).
- **Integrity:**
  - The freeze is committed and pushed before any confirmation run.
  - Before scoring, only `COMPUTE_LOG` and file existence are monitored.
  - The matching gate reads only step-100 states, and only inside the scorer.

## 8. Not authorised / out of scope

- No head-level transplants. All 4 heads share each tensor; head separation is not validated.
- No second-parent transport: there is not enough compute after the 10-hour core design.
- No claim about *how* the intervention produces the layer-0 difference beyond the factorial arms.
