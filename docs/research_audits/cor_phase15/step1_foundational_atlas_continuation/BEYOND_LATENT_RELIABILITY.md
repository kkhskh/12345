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

**Last-layer attention (descriptive).** Attention mass of readout q on q's sources:

| head | gate open: sources 2, 3 | gate open: sources 4, 5 | gate closed: sources 2, 3 | gate closed: sources 4, 5 |
|---|---|---|---|---|
| 0 | 0.94 | 0.03 | 0.02 | 0.95 |
| 1 | 0.26 | 0.08 | 0.10 | 0.25 |
| 2, 3 | ≤ 0.06 | ≤ 0.06 | ≤ 0.06 | ≤ 0.06 |

**[R3] This last-layer pattern is causally unnecessary** (reviewer R3-F1, verified in `logs/r3_checks.json`). Zeroing *all* layer-2 head outputs leaves agreement with Bayes at **1.000** on unique-argmax cases, with accuracy 0.900. By contrast:
- zeroing layer-1 heads drops agreement to 0.978;
- zeroing layer-0 heads drops it to 0.756.

The relation-specific routing in head 0 is a correlate of the gate state, not the route by which it reaches the decision.

**Where the relation lives** (`gate_mechanism.py`). Activations from the flipped run are patched into the base run. Entries are the fraction of scored cases in which the patched output takes the flipped Bayes answer:

| patched positions | at layer-1 input | at layer-2 input |
|---|---|---|
| gate tokens (sources 0, 1) | 0.00 | 0.00 |
| evidence tokens (sources 2–5) | **0.85** | 0.03 |
| readout token q | 0.15 | **0.97** |
| evidence + readout | 1.00 | 1.00 |

**Reconstruction, corrected [R3].**
1. **Layer 0.** Question q's evidence tokens encode the gate status, read from the gate tokens. Layer-0 heads are necessary: 0.756 without them.
2. **Layer 1.** The readout token q integrates gate status and evidence. At the layer-2 input, the *full* readout residual carries the decision (0.97). A reviewer check shows that patching the readout only into the layer-2 *query* gives 0.003, so it is not a routing variable.
3. **After layer 1.** The decision reaches the logits through the readout's residual path (the layer-2 MLP and the skip connection), not through layer-2 attention.

The pre-review claims "three-stage path ending in head-0 routing", "query-side routing variable" and "same operation as latent reliability" are **withdrawn**.

**The flip result adds no new information.** It is implied by the model matching Bayes wherever the argmax is unique (R3-F2). Its value is only as a check that tie handling was the cause of the 0.64.

**Comparison with latent reliability, corrected.**
- In class A′ latent-reliability seeds, the last-layer routing pattern *is* causally used: freezing it removes the cond effect, and frozen-pattern tag ablation removes 0.02 against 0.64 with live patterns.
- In GATE it is not.
- So "relation-gated last-layer routing" does **not** recur as a causal operation across the two families. What recurs is only that a relation-derived variable is computed early and carried at the evidence tokens or readout token.

## 3. TRANS: composition not learned (negative result)

**Accuracy by question type** (6000 episodes, stream 9,300,000):

| question type | model | exact Bayes | one-hop reference (no composition) | own-question only |
|---|---|---|---|---|
| AB (A, B present) | 0.933 | 0.952 | 0.952 | 0.902 |
| BC (B, C present; needs σ⁻¹) | 0.629 | 0.653 | 0.627 | 0.333 |
| C-only (needs (τσ)⁻¹) | **0.336** | **0.487** | 0.336 | 0.329 |

**Findings, corrected [R3].**
- The model uses σ: BC accuracy matches the one-hop reference (0.629 vs 0.627), far above own-question (0.333).
  - Its BC cross-entropy (0.669) is worse than the one-hop reference (0.635), so it does not fully reach even the σ-only solution (reviewer R3-F4).
- **It makes no use of C reports anywhere.**
  - Masking C in all BC questions moves its BC accuracy only from 0.629 to 0.626 (reviewer).
  - C-only accuracy is at chance (0.336). Its exact equality with the one-hop reference is a coincidence.
- Whether τ is *represented* but not composed was not tested. A τ probe would decide it.
- Bayes reaches 0.487 on C-only questions (verified by brute-force enumeration, to 9·10⁻¹⁶). So the information is present per episode, *given the generative family*.
- [R3] The pre-review "neural failure, not a statistical one" is withdrawn. Two facts make it unsupported:
  - the Bayes reference knows the generative family, whereas the network must learn it;
  - the training loss shows a late abrupt transition (0.79–0.81 until step 2400, then 0.59–0.66 by step 4000) under a cosine schedule cut from 6000 to 4000 steps.
- The run was likely stopped mid-learning. The correct statement is "composition not learned in a truncated 4000-step run".
- The cross-question attention figures (0.75–0.87) are close to the uniform-attention baseline (about 0.70), and are not evidence of targeted reading (R3-F7).

**Limits.**
- One seed, 4000 steps, and one architecture.
- The atlas found that T4 needs a curriculum (T1 → T4) or explicit pooling. TRANS may need an analogous curriculum, for example pretraining σ-only episodes. That was not tried.
- The failure is reported as "not learned under this protocol", not "unlearnable".

## 4. What carries over and what does not

| property (from latent reliability) | GATE | TRANS |
|---|---|---|
| model ≈ exact Bayes on the training distribution | yes (KL 2·10⁻⁴) | no (C-only at chance; truncated run) |
| decision-level counterfactuals follow Bayes | yes (1.000, unique-argmax cases) | not tested beyond masking |
| last-layer relation-gated routing **causally used** | **no** [R3]: the pattern is gate-dependent, but zeroing layer-2 heads changes nothing | attention near uniform; no composition |
| relational variable localised | yes: evidence tokens (layer-1 input) → full readout residual (layer-2 input), not the query | — |
| a generator-free "discrepancy" statistic would suffice | no: the gate relation is a modular constraint, not agreement | no: the relation is a latent map |

**Implication for COR (corrected [R3-F3]).**
- GATE's relation x₀ + x₁ ≡ 0 is equality after a fixed recoding (x₀ = −x₁). So it *is* an agreement relation under per-source alphabets, and the pre-review "not agreement of any kind" is withdrawn.
- What O/D in its answer-discrepancy form cannot express is that agreement among **non-answer** sources gates *which* answer sources count.
- A generalised agreement statistic with known recodings and gating could express it. That would again be a semantics-laden (Bayesian) model, not a generator-free discrepancy.
