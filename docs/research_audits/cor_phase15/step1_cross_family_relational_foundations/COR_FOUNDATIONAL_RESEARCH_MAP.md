# COR foundational research map (after Investigation XIV)

**Status.** COR Phase 1.5, Step 1, at a provisional ≈ 25%, which is unchanged. No thesis is selected, and the original obstruction/discrepancy (O/D) framework is not validated. Listing an area here does not mean it has been resolved.

**Status codes:**
- **explored**: substantial frozen or causal work exists;
- **touched**: some analysis, no confirmation;
- **untested**: no experiments.

## Territory

| area | status | what is known (ledger / XIV pointers) | highest-value next question |
|---|---|---|---|
| **T4** (latent reliability with a colluder) | explored | Exact sufficiency (M1–M7). Source-wise invisibility of trust (M11). Two neural organisations, both with heterogeneous circuits (N19–N21). Corrected causal toolkit (C1–C7). Initialisation and early-training sensitivity on one parent (N22–N25). | Stop local refinement. Use T4 only as a reference for cross-family claims. |
| **GATE** (a modular relation among non-answer sources gates which evidence pair counts) | explored in XIV | Exact order-3 decision. Order ≤ 2 rules provably ignore the gate. The combination operator is non-identifiable. Three original and two new from-scratch models. Gate stored at evidence tokens and needing a ≥ 2-dimensional subspace. "Decodable ≠ used" in every original model. Interaction-order learning test X1 (CROSS_FAMILY_CONFIRMATION.md). | Does the learning order (X1) hold under changed gate order (a gate on 3 sources, which makes the decision order 4) and under noisy gates (trust uncertainty, which makes ρ identifiable)? |
| **TRANS** (two-hop latent transport across questions) | touched (atlas continuation, deepening) | σ is always represented; τ only with auxiliary supervision. Composition was not learned on the full task. The likelihood factorises into AB and BC parts [P]. | Is TRANS's failure an interaction-order or data-efficiency barrier? Its decision needs the composition of two cross-question relations. |
| **ADAPT** (changing reliable group) | touched (deepening) | Agreement counts are insufficient and position matters [P]. A T1-curriculum ADAPT model tracks changes [C]. | Does relational invisibility hold? Does trust evidence decay with position? |
| **Unknown groups / hidden pairing** (`perm = True`) | touched (atlas) | Pairing must be inferred: a cross-pair relation over sources. Mechanisms were not studied. | Interaction-order ceilings with pairing unknown. |
| **Exact against partial copying** (ρ, β) | explored mathematically (XIII) | Per-group statistics are sufficient for C = 1 at every β and insufficient for C = 2 with shared copying (M4). | Do networks trained at β < 1 differ in organisation? Untested. |
| **Alternative architectures** | touched | DeepSets go global off-support (continuation). Relational readers learn the eligibility relation (N11). Concatenation MLPs fail at equal budget. | Is the interaction-order learning sequence architecture-independent? |
| **Initialisation and organisation selection** | explored on 2 parents | A pooled within-parent W effect is not identified (N23). | Deferred. Do not expand without a candidate statistic. |
| **Early-training route sensitivity** | explored on 1 parent | Off-route sham and location-matched noise were added in XIV (N25, T4_FINAL_CAUSAL_CONTROLS.md). | Does a structurally relevant early block in GATE (blocking evidence-token attention to the gate tokens) delay or redirect gate learning? Not run. |
| **Original O/D questions** | untested as a framework | Unweighted discrepancy is Bayes-order-compatible only under a weighted-voting coincidence, and fails for copying and zero-weight sources (atlas continuation and deepening). No obstruction operator has been tested on any neural system. | O/D remains a conjecture. No XIV result bears on it directly. |

## Untested claims about the original COR framework

1. That a discrepancy or consistency measure across sources tracks decision quality in learned systems. Only the Bayes-level counterexamples exist.
2. That an obstruction operator identifies when evidence cannot be reconciled. No implementation or test exists.
3. That relational trust computations in Transformers instantiate O/D quantities. No mapping exists. XIV's principle (relational invisibility plus interaction order) is a property of the generators, not of O/D.

## Recommended order of next directions

These are judgements, not results.

1. **Interaction-order learning dynamics across families** (GATE variants with gate order 2, 3 and 4; TRANS). This is the only XIV principle that makes quantitative learning predictions from exact mathematics.
2. **Trust uncertainty**: a noisy-gate GATE, in which ρ becomes identifiable. This tests whether T4's two combination operators reappear once trust is uncertain, the one T4-specific phenomenon with a mathematical explanation.
3. **Presence versus use with the corrected toolkit** in TRANS, where σ is present but composition fails.
