# What the original COR O/D mathematics implies about pathway selection

**Status.**
- COR Phase 1.5, Step 1 (provisional ≈ 25%, unchanged). The original O/D framework remains unvalidated.
- This assessment was written and committed **before any XVIII confirmation run** (mandate §8). It uses only the original definitions, as recovered in XVII (`step1_relational_operator_reuse/ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md`, with Review A's corrections F1–F4).

## 1. The original objects

- **The complex.** An analyst chooses a complex: vertices (for example layer, token position and component), edges, and the stalks and maps ρ.
- **A state.** A state is a section s.
- **Obstruction.** O = ‖ds‖²_W, a weighted squared coboundary.
- **Discrepancy.** D = min over g ∈ ker d of ‖s − g‖²_M, the squared distance to the nearest globally consistent section.
- **Probes.** The probes P, and the weights W and M, are chosen by the analyst.
- **Corrected Proposition S′** (XVII, Review A). Any source-level O_c or D_c with single-source germs is a quadratic form, hence an order-≤2 score of the inputs. No order-≤2 score reproduces the GATE Bayes decisions.

**What the definitions do not contain:**
- **No time:** nothing depends on the training step, so no training dynamics.
- **No optimiser, loss gradient or parameter update rule.**
- **No notion of which of several sections a learning process will reach.**
- **No intervention semantics:** nothing says what happens to O or D when an edge of the *network* is masked during training.

## 2. The four questions in mandate §8

| question | what the original mathematics gives | outcome class |
|---|---|---|
| **Information-pathway obstruction.** Does O/D say anything about blocking an attention pathway? | O is defined on a complex the analyst chooses. Blocking network edges changes the network, not the complex. One *could* build a complex over (layer, token) vertices with attention-weighted edges and compute O before and after a test-time block. But nothing in the definitions says O must rise or fall, by how much, or what follows from the change. Any such statement would need a new axiom linking ρ or W to the network's attention weights. | **O3** |
| **Available against required relational evidence.** | D compares a state with the nearest consistent section. It does not compare what a site *has* with what a decision *needs*. "Required evidence" is a Bayes-sufficiency notion: in T4 the within-pair agreement counts across questions (ledger M1). The original D does not define it. At task level, source-level D is an order-≤2 score. In T4 the trust variable has no order-1 component (ledger M11), and the known correspondence of unweighted discrepancy with the Bayes ordering holds only under a weighted-voting coincidence, failing for copying sources (ledger §3, item 12). | **O2** (limited, task-level, already known to fail for the T4 colluder) |
| **Competition between alternative neural computations.** | The definitions contain no dynamics, so they cannot express competition during learning. Two trained organisations could each be given O and D values after the fact, but no theorem orders them, or says which one training reaches. | **O3** |
| **Change of organisation after a pathway is restricted.** | This needs a map from (training intervention, initial state) to (final organisation). Nothing in O/D supplies one. | **O3** |

## 3. Verdict

- **O3, no applicable prediction**, for the mechanism XVIII studies: training-time selection between competent organisations after a temporary pathway restriction.
- **O2, limited correspondence**, only for a task-level relation between source agreement and Bayes decisions. That relation is already known to fail for the T4 colluder, and it does not involve the neural intervention.
- **P6 is therefore not run.** No neural compute is spent on rescuing O/D (mandate §8).

## 4. What would be needed, and what is out of scope

**Needed.** For the original mathematics to make a prediction here, COR would need at least:
- **a dynamical component:** a rule for how the section, or the network realising it, evolves under gradient descent;
- **a link between intervention and complex:** a statement of which complex edges a network-edge mask removes;
- **a selection principle:** which consistent section a process reaches, given where it starts and what is blocked.

None of these exists in the original definitions. Each would be new mathematics.

**Out of scope.** The project document `claude/od_repair_proposal.md` (written in another session, not reviewed) proposes a *repaired* O/D: a declared trust complex, a signed higher-order discrepancy, and a gluing-gap obstruction.
- Its own assessment states that the repaired task-level objects become Bayesian factor-graph quantities.
- Its "factor-deletion bridge" concerns **test-time** deletion or marginalisation, not training-time selection.
- It is not the original O/D, and under mandate §8 it is not presented here as a COR prediction.
