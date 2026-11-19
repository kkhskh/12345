# Pre-experiment identifiability reviews and adjudication (mandate §3.4, §13)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Who reviewed.** Three independent agent reviewers. These are agent reviews, **not human peer review**.

**Reports** (verbatim, in `reviews/`):
- `PRE_REVIEW_A.md`: mathematics and identifiability;
- `PRE_REVIEW_B.md`: computational graph and optimizer;
- `PRE_REVIEW_C.md`: measurement and scoring.

**What they reviewed.** The reviewed design is `reviews/DESIGN_XIX_v1_reviewed.md`. The revised design is `DESIGN_XIX.md` (v2).

## Round 1 classifications

All three classified the design **"Approved with mandatory corrections"**. None rejected it as non-identifying.

A states the result exactly: the design identifies the explanations **at the level of the step-100 state, not at the level of mechanism**. That limit is now written into the design's claim boundary (§3).

## Mandatory items and resolutions

| # | item | resolution | where |
|---|---|---|---|
| A1 | L1 listed paramB_optI as a machinery check; no bit-identity check | Clause removed. The scorer asserts that composed fullB equals native B bit for bit (θ, moments, rng) and that fullI's θ equals I's | DESIGN §2; `score19.py` provenance; planted "fullB not identical" |
| A2 | "the scorer treats dual states as one unit" was not implemented | The scorer reports duplicate composed-state hashes per pair. Dual readings are stated in §2/§5 | `score19.py` `duplicate_composed_states` |
| A3 | "composites lie between sources" was false | Corrected with the B←I L0 numbers. `compat19.py` reports every composite | DESIGN §2 L3 |
| A4 | L4 time scale; P3 margin | ν time scale and figures added. P3 is an equivalence test with margin 0.25, and the dev moment effects are quoted | DESIGN §2 L4 |
| A5 | vrs cannot separate X3 from content loss; vrg non-discriminating | vrs relabelled; the vrg arm and its hypothesis are removed. X3 now rests on vsgo vs vsg | DESIGN §4/§5 |
| A6 | frz rejects X4a only; gsg contrast needs monotonicity | gsg contrast withdrawn; frz and frzp form one family. [dev] With frzp the E result is partial (0.57), so X4a is **not** eliminated on E | DESIGN §5 |
| A7 | randnorm norm-matched only; redundant on M | Labelled "norm-matched only". On M it is replaced by randnorm on {L0, L2}. Step-100 KL reported | DESIGN §4 |
| A8 / B3 | same-row controls confounded (they raise T's share; on M the share ends above B; other-question content is task-relevant); P6 pooled | Both same-row controls are removed from the confirmation. They are replaced by two row-neutral, non-edge controls: **pnz** (parameter random walk matched per tensor to the block's displacement) and **lbl** (label noise matched in step-100 loss disturbance). P6 is split by base. The audit table now has rows for ctl, frz, frzp, pnz and vsgo, with per-component update and moment columns | `xix_train.py` pnz/lbl; `graph_table19.csv`; `s100diag19.py` |
| A9 | foreign-donor row without an arm | Run in development (E: φ −0.02); dev-only. The reciprocal I←B arms carry the mismatch argument | DESIGN §4 |
| A10 | claim boundary | State-level only. E localisation is near-tautological. M contradicts the blocked-layer X1. X5/X7 are explicit. Tensor-class claims are dev-only except the two E restorations | DESIGN §3 |
| A11 / C | labelling of replications | Every hypothesis is labelled a frozen replication of a 1-pair dev observation on 3 pairs per base, 1 parent, except M_A7, which is new | DESIGN §5 |
| A12 / C1 | pooled "both" hypotheses | Every hypothesis is per base. The scorer refuses "both" (fail closed) | planted "pooled hypothesis fails closed" |
| A13 / C2 | P7 could not fail | Category hypotheses: SUPPORTED if all BOTH; CONTRADICTED if a strict majority CAUSAL_ONLY/NEITHER. Requires RPC-concordant anchors | planted H_cat_* |
| A14 | dev record incomplete | All development runs are finished and tabulated: `dev_table19.json`, `dev_bandcheck19.json` | logs/dev |
| B1 | frz catch-up artefact | **frzp** added (values held, moments update). [dev] The catch-up is real: frz −0.01, frzp 0.57 on E | DESIGN §4 |
| B2 | vsg leaves WO learning | Relabelled. **vsgo** added (V and WO get no gradient from T). The downstream-reader limitation is stated | DESIGN §4; audit row vsgo |
| B4 | randnorm not reproducible; one seed | Sorted iteration (verified identical across PYTHONHASHSEED). Per-pair seed 19000 + D (+500 on M). compose refuses to overwrite a differing state. The composed-state hash is recorded at run start (run_config branch_src) and checked by the scorer | `compose19.py`; planted "composed state changed after the run" |
| B5 | audit table incomplete | Graph audit v2: the actual training step per mode, per-component update / mu / nu columns, and new modes | `graph_audit19.json` (v1 kept as `graph_audit19_v1.json`) |
| C3 | the functional score is not independent and not Claim-2 evidence | Renamed RPC. Claim 2 deferred to the stage-6 battery. Strongest permitted statement: "causal-route change with concordant residual profile" | DESIGN §1, §7 |
| C4 | bands ignore the causal class | `need_class` on every switch/nil band | planted H_mid_class |
| C5 | fail-closed checks incomplete | run_config.json from every run (argv, FX_* environment, harness and dependency hashes, branch source and hashes). FX_NOFALLBACK enforced. measure19 records the hashes of every measurement-path module, including the Bayes posterior (genlib/relpairs), and reference caches include the code hash. 19 file-based planted defects, each failing for its own reason | `score19.py`, `measure19.py`, `xix_train.py` |
| C6 | matching gate failed open | A missing or unreadable state fails the gate | planted "missing matching result fails closed" |
| C7 | the scorer trusted record labels | Labels are recomputed from the numbers | planted "record labels ignored" |
| C8 | competence threshold | Primary KL ≤ 0.06 (validation: competent uP160 models ≤ 0.042; degraded parents ≥ 0.063), with comp_strict (0.045) and comp_loose variants and planted boundary cases | DESIGN §6 |
| C9 | NaN → silent PARTIAL | NaN drops the pair with a reason | planted "NaN phi dropped" |
| C12 | uncertainty unspecified | `stats19.py` (per-pair values, range, descriptive t-interval, Clopper–Pearson lower bounds, no pooling, no p-values) | DESIGN §6 |

## Advisory items adopted

- **A1:** vrg swapped for I←B L0_QK and I←B L0_OV on E.
- **A2:** dev re-measured with measure19 on 86 200 000. Every dev value lies in its band, with these exceptions:
  - M_A7 has no dev run;
  - the original E vsgo concordance prediction was ambiguous in dev (RPC AMBIGUOUS), so P7 on E uses vrs instead.
- **A3:** step-100 L0-displacement cosine and distance diagnostics added.
- **A4:** I←B L0+L2 added on M.
- **A5:** renamed "restore reduces".
- **A6:** gate failure means "X8 unresolved".
- **A7:** NaN handling.
- **C10:** a 'lin' φ variant.
- **C13:** thr_strict variant.
- **C14:** HYBRID categories keep their RPC label.

## Round 2 (reviews appended to the same files)

All three again returned **"Approved with mandatory corrections"**. The remaining items are labels, scorer rules, instrumentation and an audit row. None changes the arms.

| # | item | resolution |
|---|---|---|
| A R2-1 | the vsg and vsgo bands touch; "+O" attribution | Paired hypotheses `*_P4_vsgo_exceeds_vsg`, `phi['vsgo'] - phi['vsg'] ∈ [0.2, 1.5]`, per base. P4 is reworded as sufficiency, not unique attribution. M801 step-100 diagnostics computed (`s100diag_M801.json`) |
| A R2-2 | mandate B3 categories | Every training arm is categorised (DESIGN §4) |
| A R2-3 / B R2-2 | lbl audit row; wrong "gradients unchanged" | Graph audit v3 has an `lbl` row computed with the noised labels. Every gradient changes, and the description is corrected |
| A R2-4 / C R2-5 | kl100 gate on M's tiny denominator | The gate fails if the control's excess is ≤ 0. If the block's excess is < 0.005, loss matching is not applicable and the control needs an excess of at least 0.005. Planted tests run on real step-100 states |
| A R2-5 | X8a counted twice | A7 and P6-pnz form one family |
| A R2-6 / C R2-1 | P7 swap undisclosed; NEITHER counted as a contradiction | Swap disclosed. P7 is scored only on causal-switch pairs, is contradicted only by CAUSAL_ONLY, and is labelled "eliminates nothing" |
| A R2-7 | A4 wording | Necessity only |
| B R2-1 | pnz/lbl mid-window resume not reproducible | A mid-window resume is discarded and the run restarts at step 0. Every resume is logged. Verified bit-identical to an uninterrupted run |
| C R2-2 | the class condition on M_P1_L0_alone_insufficient is fragile | Class condition removed |
| C R2-3 | silent paths | Double-quoted arm names are detected. Gate errors are reported separately (fail closed). No-files mode prints "SCORED_NO_FILES (not a frozen verdict)" |
| C R2-4 | planted coverage | The file world now includes M pairs, a perturbation-seed transplant and a pnz arm with its reference states, plus 3 new defects. kl100 gate tests run on real states. 77 checks pass |
| C R2-6 | competence justification; mostly-noncompetent arms | The justification is corrected: the threshold was set after looking at the distributions, and the gap is not clean. Band NONCOMPETENT when more than half of the pairs drop for competence |
| C R2-7 | stats19 route components and measurement floor | Implemented |
| advisory | pnz wait timeout; vsgo 0.70 reading; low-power bands predeclared | Done |

## Round 3

A verification-only pass by the same reviewers on their round-2 items.

| reviewer | round-3 classification | remaining (advisory only) |
|---|---|---|
| A (identifiability) | **Approved** | One sentence was added to the §5 P4 reading: vsgo still lowers T's share, so "no routing change is involved" is not claimed |
| B (graph/optimizer) | **Approved** | Round-2 advisories R2.5–R2.8 (wording and housekeeping) |
| C (measurement) | **Approved** | Some simple equality provenance checks are not exercised by planted tests; arm-name validation covers single-quoted names only (all frozen expressions use single quotes); stats19 does not refuse a no-files score |

**Authorisation.** All three reviewers approved the design after two correction rounds. Every mandatory item from rounds 1 and 2 is resolved in code and documents. The confirmation is authorised as specified in DESIGN_XIX.md v2.1.

**Mechanisms the reviewers rejected or narrowed before any confirmation** (mandate Q25):
- gsg as "removal of route learning": it removes only the T-side half of the routing gradient.
- vsg as "removal of value learning": WO still learns.
- frz's nil as evidence against routing learning: it is a moment catch-up artefact.
- The same-row msk/gof controls as non-targeted.
- vrs as separating value learning from content loss.
- The functional score as Claim-2 evidence.
- Transplants as mechanism identification: they give state-level localisation only.
