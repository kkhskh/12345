# XIX confirmation design, v2 (revised after pre-experiment reviews A, B and C)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged). This is v2.1, after review rounds 1 and 2. Nothing here is frozen until `FREEZE_XIX.json` is committed.

- The reviewed v1 is kept as `reviews/DESIGN_XIX_v1_reviewed.md`.
- How each objection was resolved is in `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.

**Evidence labels.**
- [proof]: follows from construction.
- [num]: numerical verification.
- [dev]: development data. These come from matched pairs E801 and M801 (parent uP160, data order 801), measured by measure18 on stream 86 100 000, re-measured by measure19 on 86 200 000 (`logs/dev/dev_bandcheck19.json`).

## 1. Objects

**Training state.** S_k = (θ_k, m_k, v_k, c_k, r_k).
- θ_k: 44 tensors.
- m_k, v_k: AdamW moments.
- c_k: two step counters.
- r_k: the data-stream RNG state.
- No other optimizer state exists [num, pre-review B §0].

**Matched pair.** B and I start from the same initialisation, data order and RNG.
- I applies the block (gof B = 1e9) to the target edges for steps < 100: E: own0, layer 0; M: own2, layer 2.
- At step 100, r and c are equal, and θ, m, v differ.

**Continuation.** Φ(S) is plain training from S to step 1211. It is deterministic [num]:
- branch continuations are bit-identical to uninterrupted runs (pre-review B: step-111 maximum difference 0.0);
- composed fullB is bit-identical to native B, which the scorer asserts on every confirmation pair.

**Outcome.** The scorer recomputes everything from the recorded numbers:
- nec_E, nec_M, KL and acc (measure19);
- φ_nec against the pair's Φ(fullB) / Φ(fullI) anchors;
- the causal class;
- the residual-profile concordance (RPC) score f.

f was formerly called the "functional score". It correlates 0.91 with causal necessity and its references are classed causally (pre-review C3). **It is not evidence of a changed functional computation (Claim 2).** Claim 2 is reserved for the stage-6 test battery. The strongest statement this confirmation can support is "causal-route change with a concordant residual profile".

## 2. What is guaranteed, and what counts as one observation

- **L1 [proof].** The window intervention acts only through S_100. So Φ(fullI) = the intervention run. fullB and fullI are machinery anchors and carry no hypothesis.
  - paramB_optI is **not** a machinery check. It is P3's test.
- **L2 [proof].** For a partition (G, G') with the moments following the parameters, B←I_G and I←B_G' are the same state.
  - So the dual readings in §5 are one observation. For example, M_P1_L0_alone_insufficient is also "the non-L0 state of I does not, on its own, prevent the switch".
  - No two confirmation arms compose the same state. The scorer reports duplicate composed hashes per pair.
- **L3.** Composites are synthetic states.
  - [dev] They are **not** always between the sources. B←I L0 on E has step-100 KL 0.540 and accuracy 0.418, worse than both sources (B 0.459/0.560, I 0.536/0.443).
  - Step-100 KL, accuracy and route necessity of every composite are reported (`scripts/compat19.py`). A transplant outcome is a property of the state function Φ, not a natural mediation.
- **L4.** Mixed parameter/moment states are synthetic.
  - The first-moment influence decays with β1 = 0.9 (≈ 10 steps). The second-moment influence decays with β2 = 0.999. Through the bias-corrected denominator, that influence lasts ≈ 100+ steps under warm-up.
  - [dev] Relative moment differences at step 100: mu 0.3–1.95, nu 0.05–0.27.
  - P3 is an equivalence test with a 0.25 margin. [dev] It does detect moment effects: B←I L0 with B's L0 moments gives 0.74 against 0.91, and frz against frzp (identical θ_100, different moments) gives −0.01 against 0.57 on E.

## 3. Competing explanations

**State level** (where the selection is carried at step 100):
- X1: target-layer state;
- X2: non-target or competitor-layer state;
- X5: downstream-layer state;
- X6: optimizer moments;
- X7: an interaction of groups;
- X8: general disturbance.
  - X8a: state displacement of the same size in a random direction;
  - X8b: loss disturbance / slowed learning in the window.

**Process level** (how the window produces that state):
- X3: value-pathway learning through T (V and O);
- X4a: routing-parameter learning of the target layer;
- X4b: the routing (QK) state itself;
- C: forward-content loss.

**Claim boundary (pre-review A10).**
- Transplants localise the selection-carrying state. They do not identify mechanism.
- On E the localisation to layer 0 is **close to tautological**, because the measured E route lives in layer 0. The informative E results are the sub-group restorations, the moments, and the training-time arms.
- [dev] On M, layer 2 (the blocked layer) is **not** necessary. This contradicts X1 in its "blocked-layer" form.
- Tensor-class claims (QK / OV / MLP) are development-only, except for the two E restorations below.

## 4. Arms

**Units.**
- Six fresh matched pairs: D = 821, 822, 823 on E (L = W = 21) and on M (L = W = 23).
- **One independent parent (uP160).** Data orders are the independent units: 3 per base, never pooled across bases, and D801 never pooled.
- Frozen stream: 94 700 000.

**Step-100 window states (54).** Per pair:
- B, I;
- vsg, vsgo, vrs, frz, frzp, pnz, lbl.

All with FX_NOFALLBACK = 1, FX_WLOSS = 1 and run_config.json.

**Continuations (99).**

| arm | bases | definition | graph audit (`logs/graph_table19.csv`) |
|---|---|---|---|
| fullB, fullI | E, M | anchors | — |
| paramI_optB, paramB_optI | E, M | I's θ with B's moments, and the reverse | synthetic (L4) |
| B←I L0, I←B L0 | E, M | layer-0 groups (QK, OV, MLP, LN) with their moments | — |
| I←B L0_QK, I←B L0_OV | E | restore one layer-0 tensor class | tests X4b / state-level X3 |
| B←I L0+L1, B←I L0+L2, I←B L0+L2 | M | dev-located combination and its reciprocal | — |
| randnorm_L0 (E), randnorm_L0+L2 (M) | | B + a random direction with ‖·‖ = ‖θ_I − θ_B‖ per group, B's moments; seed 19000 + D (+500 on M); **norm-matched only** | X8a, state-level |
| vsg | E, M | stop-gradient on the value vectors read through T. WO and downstream readers still learn T content | forward identical; routing gradient = plain |
| vsgo | E, M | vsg plus WO receives no gradient from the T term. Downstream readers of T content cannot be stopped without changing the forward [limitation] | forward identical; routing = plain; WO changed |
| vrs | E, M | T's values rolled across episodes; gradient kept but uninformative (A5) | forward changed |
| frz | E, M | target-layer QK values and moments held; the shared counter keeps running, so the moments are under-scaled at release (B1 catch-up) | — |
| frzp | E, M | target-layer QK values held, moments update | — |
| pnz | E, M | per-step Gaussian noise on all tensors, σ_k = ‖θ_I − θ_B‖_k / √(n_k·100) from the pair's own B/I states | non-edge, row-neutral |
| lbl | E, M | a fraction 0.5 of the window's targets replaced by uniform labels (own RNG; data stream untouched). Forward and edges are unchanged, but **every gradient changes** (the targets change), and its step-100 route state partly resembles the block's (A R2) | non-edge; audit row `lbl` uses the noised labels |

**Mandate B3 categories of the training arms.**
- Natural forward perturbation, with its natural gradients: vrs.
- Synthetic gradient interventions (forward unchanged, part of the chain rule removed): vsg, vsgo.
- Parameter freezes: frzp. frz is a freeze plus an optimizer-state manipulation, because its moments are held.
- Non-targeted disturbances: pnz (parameter noise; optimizer state untouched) and lbl (target noise).
- Synthetic states: transplants.
- pnz and lbl never resume inside the window: a mid-window crash restarts the run from step 0, and every resume is logged in resume_log.json (B R2-1).

**Removed after review.**
- msk oq0 / gof oq2 (same-row controls, B3/A8): dev results kept as development evidence only.
- vrg (no discriminating hypothesis, A5): XVIII and dev evidence only.
- gsg: XVIII evidence only; P5's gsg contrast withdrawn (A6).
- Foreign-donor arm: dev only. On E it gives φ −0.02, so symmetric mismatch does not switch E.

**Control matching (predeclared, step-100 states only, fail closed).**
- pnz: total parameter-distance ratio to I in [0.67, 1.5]. [dev] E 1.01.
- lbl: step-100 KL excess ratio (KL − KL_B)/(KL_I − KL_B) ≥ 0.5. [dev] E 0.88.
  - The gate fails if lbl's own excess is ≤ 0.
  - If the block's excess is < 0.005, loss matching is **not applicable**. lbl is then usable only if its own excess is ≥ 0.005, i.e. a disturbance at least as large as the block's (A R2-4, C R2-5).
  - [dev] On M the block's excess is 0.012 and lbl's is 0.085.
- If fewer than 2 pairs per base pass a gate, the corresponding X8 form is reported as **unresolved** on that base. No re-tuned control is substituted.
- [dev, A3 bridge] Step-100 diagnostics (`logs/dev/s100diag_E801.json`). These are reported, not used as mediation:
  - lbl's layer-0 displacement has cosine 0.76 with the block's and leaves the E route non-functional at step 100 (nec_E 0.027), yet it does not switch E;
  - pnz's displacement has cosine 0.01 with the block's.
  - M801 (`logs/dev/s100diag_M801.json`): the switching arms vsgo and vrs have layer-0 displacement cosines of 0.88 and 0.92 with the block's. The non-switching arms frz, frzp and pnz have 0.13, 0.13 and 0.00; lbl has 0.28. But vsg, a hybrid, has 0.87, and on E lbl (no switch) has 0.76. **The cosine does not predict the outcome on its own.**

**Compute.** ≈ 54 × 1.5 min + 99 × 12.5 min / 2 lanes ≈ 11 h.

## 5. Frozen predictions (35; one base each; scored by `scripts/score19.py` v2)

**Scoring rules.**
- **Bands** (`band`): SUPPORTED if every evaluable pair lies in [lo, hi] *and*, where stated, the arm has the anchor's causal class. CONTRADICTED if a strict majority lie outside [lo − 0.1, hi + 0.1] or fail the class condition. INCONCLUSIVE if fewer than 2 evaluable pairs.
- **Category** (`category`): uses the pair categories (BOTH against CAUSAL_ONLY/NEITHER) on pairs whose anchors are RPC-concordant.
- **Elimination** of an explanation requires SUPPORTED. PARTIAL eliminates nothing.
- **Families.** The two P3 bands are one family per base, the two M partner bands are one family, and P5 frz/frzp are one family.

| id | prediction | dev value with measure19 (dev band) | status |
|---|---|---|---|
| E_P1_L0_sufficient | `phi['BfromI_L0']` ∈ [0.75, 1.25], class = I-anchor's | E801: 0.96 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P2_L0_necessary | `phi['IfromB_L0']` ∈ [-0.25, 0.25], class = B-anchor's | E801: 0.01 (SUPPORTED) | replication of a 1-pair dev observation |
| E_A4_L0QK_restore_insufficient | `phi['IfromB_L0_QK']` ∈ [0.75, 1.25], class = I-anchor's | E801: 0.97 (SUPPORTED) | replication of a 1-pair dev observation |
| E_A4_L0OV_restore_insufficient | `phi['IfromB_L0_OV']` ∈ [0.75, 1.25], class = I-anchor's | E801: 0.97 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P3_params_carry | `phi['paramI_optB']` ∈ [0.75, 1.25], class = I-anchor's | E801: 0.97 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P3_moments_alone_nil | `phi['paramB_optI']` ∈ [-0.25, 0.25], class = B-anchor's | E801: -0.0 (SUPPORTED) | replication of a 1-pair dev observation |
| E_A7_normmatched_L0_nil | `phi['randnorm_L0']` ∈ [-0.25, 0.25], class = B-anchor's | E801: -0.01 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P6_pnz_nil | `phi['pnz']` ∈ [-0.25, 0.25], class = B-anchor's | E801: 0.0 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P6_lbl_nil | `phi['lbl']` ∈ [-0.25, 0.25], class = B-anchor's | E801: -0.0 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P4_vsgo_switch | `phi['vsgo']` ∈ [0.75, 1.25], class = I-anchor's | E801: 0.83 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P4_vsgo_switch_sens070 | `phi['vsgo']` ∈ [0.7, 1.25], class = I-anchor's | E801: 0.83 (SUPPORTED) | SENSITIVITY READING of E_P4_vsgo_switch (never replaces it; pre-review A R2) |
| E_P4_vsg_partial | `phi['vsg']` ∈ [0.25, 0.75] | E801: 0.35 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P4_vsgo_exceeds_vsg | `phi['vsgo'] - phi['vsg']` ∈ [0.2, 1.5] | E801: 0.48 (SUPPORTED) | replication of a 1-pair dev observation (paired contrast) |
| E_P4_vrs_switch | `phi['vrs']` ∈ [0.75, 1.25], class = I-anchor's | E801: 1.0 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P5_frz_nil | `phi['frz']` ∈ [-0.25, 0.25], class = B-anchor's | E801: -0.01 (SUPPORTED) | replication of a 1-pair dev observation; moment-artefact check, eliminates nothing |
| E_P5_frzp_partial | `phi['frzp']` ∈ [0.25, 0.75] | E801: 0.57 (SUPPORTED) | replication of a 1-pair dev observation |
| E_P7_L0_concordant | category BOTH on pairs with a causal switch (contradicted by a majority CAUSAL_ONLY) | E801: BOTH (SUPPORTED) | replication of a 1-pair dev observation; same-run RPC concordance check, eliminates nothing |
| E_P7_vrs_concordant | category BOTH on pairs with a causal switch (contradicted by a majority CAUSAL_ONLY) | E801: BOTH (SUPPORTED) | replication of a 1-pair dev observation; same-run RPC concordance check, eliminates nothing |
| M_P1_L0_alone_insufficient | `phi['BfromI_L0']` ∈ [-0.25, 0.5] | M801: 0.25 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P2_L0_restore_reduces | `phi['IfromB_L0']` ∈ [-0.25, 0.5] | M801: 0.3 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P2_L0L2_restore_prevents | `phi['IfromB_L0L2']` ∈ [-0.25, 0.25], class = B-anchor's | M801: 0.13 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P1_L0L1_sufficient | `phi['BfromI_L0L1']` ∈ [0.75, 1.25], class = I-anchor's | M801: 0.96 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P1_L0L2_sufficient | `phi['BfromI_L0L2']` ∈ [0.75, 1.25], class = I-anchor's | M801: 0.98 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P3_params_carry | `phi['paramI_optB']` ∈ [0.75, 1.25], class = I-anchor's | M801: 1.02 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P3_moments_alone_nil | `phi['paramB_optI']` ∈ [-0.25, 0.25], class = B-anchor's | M801: 0.09 (SUPPORTED) | replication of a 1-pair dev observation |
| M_A7_normmatched_L0L2_nil | `phi['randnorm_L0L2']` ∈ [-0.25, 0.25], class = B-anchor's | — (MISSING (no dev run)) | new (no dev run) |
| M_P6_pnz_nil | `phi['pnz']` ∈ [-0.25, 0.25], class = B-anchor's | M801: -0.01 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P6_lbl_nil | `phi['lbl']` ∈ [-0.25, 0.25], class = B-anchor's | M801: 0.04 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P4_vsgo_switch | `phi['vsgo']` ∈ [0.75, 1.25], class = I-anchor's | M801: 1.0 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P4_vsg_partial | `phi['vsg']` ∈ [0.25, 0.75] | M801: 0.44 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P4_vsgo_exceeds_vsg | `phi['vsgo'] - phi['vsg']` ∈ [0.2, 1.5] | M801: 0.55 (SUPPORTED) | replication of a 1-pair dev observation (paired contrast) |
| M_P4_vrs_switch | `phi['vrs']` ∈ [0.75, 1.25], class = I-anchor's | M801: 1.0 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P5_frz_nil | `phi['frz']` ∈ [-0.25, 0.25], class = B-anchor's | M801: -0.03 (SUPPORTED) | replication of a 1-pair dev observation; moment-artefact check, eliminates nothing |
| M_P5_frzp_nil | `phi['frzp']` ∈ [-0.25, 0.25], class = B-anchor's | M801: 0.01 (SUPPORTED) | replication of a 1-pair dev observation |
| M_P7_vsgo_concordant | category BOTH on pairs with a causal switch (contradicted by a majority CAUSAL_ONLY) | M801: BOTH (SUPPORTED) | replication of a 1-pair dev observation; same-run RPC concordance check, eliminates nothing |

**What each family can eliminate if SUPPORTED, per base, and which XVIII-open item it addresses:**
- **P1/P2 (transplants):** X2, X5 (E); single-layer and blocked-layer forms of X1 (M); X7 is documented, not excluded.
- **A4:** restoration (necessity only). "Restoring B's L0 QK state alone, or OV state alone, prevents the E switch" — rejected if SUPPORTED. It says nothing about sufficiency (A R2-7).
- **P3:** X6 at the full-state level, with a 0.25 equivalence margin.
- **A7 + P6-pnz (one family, X8a):** random displacement, either at step 100 with matched per-group norm or during training as a matched random walk. Counted once per base (A R2-5).
- **P6-lbl (X8b):** loss disturbance / slowed early-route acquisition (dev E: the route is not yet functional at step 100, yet the model does not switch).
- **P4:** read only together: vsgo SUPPORTED, vsg PARTIAL and *vsgo − vsg ≥ 0.2* SUPPORTED.
  - This would show that removing V+O learning through T is **sufficient** for a full switch, with the forward and the routing gradient intact. Forward-content loss and routing-gradient removal are therefore **not necessary** (XVIII confound 1).
  - It is **not** a unique attribution. vrs (content loss) is also sufficient, and frzp is partial on E (A R2-1).
  - vsgo still lowers T's step-100 attention share: on M by about two-thirds as much as the block, on E by about a quarter. "Routing-gradient removal is not necessary" therefore does **not** mean "no routing change is involved" (A R3 advisory).
  - A PARTIAL on E_P4_vsgo_switch is likely even if the dev value is typical (dev 0.83 against an edge of 0.75; about even odds that all 3 pairs land in band). It is predeclared to mean "not established". The 0.70 reading (E_P4_vsgo_switch_sens070) is reported as a sensitivity reading and never replaces the primary.
- **P5:** frz SUPPORTED-nil and frzp partial (E) / nil (M) would show that the frz nil is a moment artefact, and that freezing routing-parameter learning is partially effective on E only. X4a is therefore **not** eliminated on E.
- **P7:** a same-run RPC concordance check. RPC correlates 0.91 with the causal measure, so P7 **eliminates nothing**.
  - It is scored only on pairs with a causal switch, and contradicted only by CAUSAL_ONLY.
  - **Disclosure:** the E P7 arm was changed from vsgo to vrs *after* the dev re-measurement showed vsgo's RPC to be AMBIGUOUS (A R2-6, C R2-1).

**Admissibility (§11; pre-review A11).**
- Every hypothesis except M_A7 is a *frozen replication of a 1-pair development observation, on 3 fresh pairs per base, 1 parent*.
- They are reported as such, never as independent discoveries, and never pooled with D801.
- None is guaranteed by construction (§2).

## 6. Scoring, sensitivity, uncertainty, integrity

**Primary.**
- Competence: KL ≤ 0.06 and acc ≥ 0.80.
  - **This threshold was set after looking at the validation-stream and dev KL distributions** (preflight calibration, before any confirmation data). uP160 models reach at most 0.042. The cut is not a clean gap: models of other parents lie at 0.047–0.057 (C R2-6).
  - The XVIII rule 0.045 is kept as the comp_strict variant.
  - A hypothesis whose arm is non-competent on more than half of its base's pairs is scored NONCOMPETENT.
- Pair evaluability: both anchors competent, pure and opposite. No behavioural gate.

**Frozen variants.** delta0, delta20, phicut70, phicut80, eps003, eps03, lin, evalB, thr_pm05, thr_strict, comp_strict, comp_loose. Every variant recomputes evaluability.

**Uncertainty** (`scripts/stats19.py`):
- per-pair values;
- per-base range, mean and descriptive t-interval;
- k/n in band with an exact one-sided 95% Clopper–Pearson lower bound (3/3 → 0.37);
- no p-values, no pooling.

**Fail-closed provenance.**
- Code, dependency, reference and harness hashes.
- run_config of every run (argv seeds, window environment, no fallback, branch source and hashes).
- Transplant provenance, including alpha and the per-pair perturbation seed.
- Native B/I window settings.
- fullB bit-identity.
- Matching gates fail closed.

**Preflight.**
- `logs/scorer_planted19.json`: 77 checks pass. These include:
  - 22 file-based defects, on E and M pairs, perturbation seeds and pnz references, each failing for its own reason;
  - the kl100 gate on real step-100 states, covering a valid ratio, the not-applicable case and a control with no disturbance.
- Predeclared low-power bands (C R2):
  - E_P4_vsgo_switch and E_P4_vsg_partial: dev values sit 0.08 and 0.10 from their edges;
  - the P7 checks: about a 23% chance of INCONCLUSIVE.
- `logs/scorer_validation_summary19.json`.

**Integrity.**
- Freeze committed and pushed before any confirmation run.
- Before scoring, only COMPUTE_LOG and file existence are monitored.
- The gates read only step-100 states, inside the scorer.

## 7. Not authorised

- Head-level transplants.
- Second-parent transport.
- Any claim of a changed functional computation from RPC.
- Any reading of a PARTIAL as an elimination.
