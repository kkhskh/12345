# Beyond latent reliability: two other relation families (mandate §11)

**Status.**
- Each task was trained once (seed 0), on the same 3-layer Transformer, for 4000 steps. That budget was cut from 6000 before training (`logs/COMPUTE_LOG.txt`).
- The tasks were designed before training. All mechanism analyses are post hoc.
- One seed per task: nothing here is a law.

**Code and logs.**
- Tasks and exact Bayes: `src/tasks_ext.py`. Training: `scripts/run_job.py`, with `gate_s0` and `trans_s0` in `checkpoints/`.
- Analysis: `scripts/beyond_analysis.py` and `scripts/gate_mechanism.py`.
- Logs: `logs/beyond_analysis.json`, `logs/beyond_analysis_gate.json`, `logs/gate_mechanism.json`.

## 1. Why these two families

Both relations are defined independently of the answer and independently of any agreement count. Each tests a property that latent reliability does not have:

| family | relation | what is new relative to latent reliability |
|---|---|---|
| **GATE** | per question, a *modular constraint* x₀ + x₁ ≡ 0 (mod K) between two non-answer sources decides which evidence pair is informative | the relation is not agreement; it is within-question; nothing is shared across questions; the relation's sources carry no answer information at all |
| **TRANS** | latent permutations σ, τ shared by the episode. Channel A reports Y, B reports σ(Y), C reports τσ(Y). Each question shows only AB, only BC, or only C | decoding C-only questions needs the **composition** (τσ)⁻¹. That is identified only by chaining A–B and B–C co-occurrences across different questions: a two-hop cross-question relation |

## 2. GATE: a clean decision-relevant relational mechanism

**Behaviour.** Accuracy on 6000 episodes (stream 9,200,000):

| reference | accuracy |
|---|---|
| model | **0.900** |
| exact Bayes | 0.899 |
| best rule that ignores the gate relation | 0.799 |

KL to Bayes is 2·10⁻⁴.

**Decision-level intervention.** Flip the gate in question q by re-drawing x₁, keeping every evidence source. This is an on-support counterfactual: the flipped episode is a natural GATE input.
- The scored set is the 4,377 cases where the Bayes argmax changes and both Bayes argmaxes are unique.
- On that set the model follows the Bayes switch in **1.000** of cases.
- [Correction] A pre-check without tie exclusion gave 0.64. That figure was driven by ties: after a flip, the newly trusted pair may disagree internally, which leaves Bayes indifferent between two candidates.

**Routing (last layer).** Attention mass of readout q on q's sources:

| head | gate open: sources 2, 3 | gate open: sources 4, 5 | gate closed: sources 2, 3 | gate closed: sources 4, 5 |
|---|---|---|---|---|
| 0 | **0.94** | 0.03 | 0.02 | **0.95** |
| 1 | 0.26 | 0.08 | 0.10 | 0.25 |
| 2, 3 | ≤ 0.06 | ≤ 0.06 | ≤ 0.06 | ≤ 0.06 |

Head 0 routes to the informative pair according to the relation.

**Where the relation lives** (`gate_mechanism.py`). Activations from the flipped run are patched into the base run. Entries are the fraction of scored cases in which the patched output takes the flipped Bayes answer:

| patched positions | at layer-1 input | at layer-2 input |
|---|---|---|
| gate tokens (sources 0, 1) | 0.00 | 0.00 |
| evidence tokens (sources 2–5) | **0.85** | 0.03 |
| readout token q | 0.15 | **0.97** |
| evidence + readout | 1.00 | 1.00 |

**Reconstruction.** This is a three-stage causal path, supported by interventions at each stage:
1. **Layer 0.** The evidence tokens of question q read the gate tokens and encode the gate status. By the layer-1 input, the gate tokens themselves carry nothing that matters (0.00).
2. **Layer 1.** The readout token q reads the gate status from the evidence tokens. At the layer-2 input it is carried by the readout token (0.97), that is, on the *query* side.
3. **Layer 2.** Head 0 routes the readout's attention to the informative pair (0.94–0.95).

**Comparison with latent reliability.**
- *Same operation:* reliability-gated (here, constraint-gated) routing to an evidence pair in the last layer. In GATE the routing is query-side (readout token). In class A′ latent-reliability seeds, the atlas located the tag on the key side; seed 2 split it between key and query.
- *Different:* the gating variable is computed within the question, from a non-agreement relation, in two early-layer hops.
- So "routing the readout's attention by a relation-derived variable" recurs across two relation families in this architecture, with different variable placement. One seed per task; not a law.

## 3. TRANS: composition not learned (negative result)

**Accuracy by question type** (6000 episodes, stream 9,300,000):

| question type | model | exact Bayes | one-hop reference (no composition) | own-question only |
|---|---|---|---|---|
| AB (A, B present) | 0.933 | 0.952 | 0.952 | 0.902 |
| BC (B, C present; needs σ⁻¹) | 0.629 | 0.653 | 0.627 | 0.333 |
| C-only (needs (τσ)⁻¹) | **0.336** | **0.487** | 0.336 | 0.329 |

**Findings.**
- The model learns the **one-hop** cross-question relation: on BC questions it reaches the one-hop reference (0.629 vs 0.627), far above own-question (0.333). So it infers σ from AB co-occurrences in other questions.
- It does **not** learn the composition. C-only accuracy is at chance (0.336) and identical to the one-hop reference. Bayes gets 0.487, so the information is present: this is a **neural failure, not a statistical one**, at this budget and seed.
- Input interventions agree. Masking the C reports of the BC questions, or the A reports of the AB questions, leaves model C-only accuracy at chance (0.343, 0.330). Bayes drops to the prior (0.336, 0.344).
- All four heads attend 0.75–0.87 to other questions on BC and C-only questions. So cross-question reading happens, but the second hop is not composed.

**Limits.**
- One seed, 4000 steps, and one architecture.
- The atlas found that T4 needs a curriculum (T1 → T4) or explicit pooling. TRANS may need an analogous curriculum, for example pretraining σ-only episodes. That was not tried.
- The failure is reported as "not learned under this protocol", not "unlearnable".

## 4. What carries over and what does not

| property (from latent reliability) | GATE | TRANS |
|---|---|---|
| model ≈ exact Bayes on the training distribution | yes (KL 2·10⁻⁴) | no (C-only at chance) |
| decision-level counterfactuals follow Bayes | yes (1.000, unique-argmax cases) | not tested beyond masking |
| last-layer relation-gated routing | yes (head 0, query-side variable) | cross-question attention present; no composition |
| relational variable localised | yes: evidence tokens (layer-1 input) → readout token (layer-2 input) | — |
| a generator-free "discrepancy" statistic would suffice | no: the gate relation is a modular constraint, not agreement | no: the relation is a latent map |

**Implication for COR.** In GATE, the relation that determines the decision is not an agreement or discrepancy relation of any kind. So any theory framed in terms of consistency or discrepancy among sources (O/D) cannot express it. This is a second, independent counterexample to the generality of the original O/D framing, alongside COR_TEN §2 for latent reliability.
