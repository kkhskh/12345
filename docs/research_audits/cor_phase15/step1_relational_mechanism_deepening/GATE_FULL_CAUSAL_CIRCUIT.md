# GATE: full causal circuit, three seeds (Workstream B)

Task (continuation `tasks_ext.sample_gate`): per question, sources 0, 1 satisfy x₀ + x₁ ≡ 0 (mod 3) ("gate open") w.p. ½;
open ⇒ sources 2, 3 report Y w.p. 0.9 and 4, 5 are noise; closed ⇒ the reverse. Bayes depends on X_q only through the gate
state and the two evidence pairs; nothing is shared across questions.
Models: seed 0 (continuation), seeds 1 and 2 (this package, identical recipe, 4000 steps). Code: `scripts/gate_circuit.py`
(patches, ablations, controls, representation test), `gate_redundancy.py`, `gate_headpairs.py`, forward `src/fwd3.py`
(verified equal to the model's forward: max |Δ| 1.7·10⁻⁶). Logs: `logs/gate/`.
Positions for question q: gate tokens G (sources 0, 1), evidence tokens E (sources 2–5), readout R.
Patches transfer activations from a gate-flipped run (x₁ redrawn; evidence kept; on-support) into the base run; score =
fraction of scored cases (Bayes argmax changes, both unique) whose patched answer is the flipped Bayes answer.

## 1. Behaviour and controls (all seeds)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| accuracy (Bayes 0.902) | 0.899 | 0.903 | 0.900 |
| negative control: flipping ANOTHER question's gate changes q's answer | 0.017 | 0.016 | 0.009 |
| negative control: patching readout q from that run | 0.014 | 0.010 | 0.003 |

## 2. Where the gate relation is computed and how it reaches the decision

Flip-transfer of single-site patches (selected; full tables in `circuit_s*.json`):

| patched component | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| layer-0 attention output at G | 0.00 | 0.00 | 0.00 |
| layer-0 attention output at E | **0.83** | **1.00** | 0.00 |
| layer-0 MLP output at E | 0.66 | 0.98 | 0.00 |
| layer-0 head at E carrying it | h2 0.47, h1 0.28 | **h1 1.00** | — |
| layer-1 attention output at E | 0.00 | 0.00 | **1.00** (head 3: 1.00) |
| layer-1 MLP output at E | 0.00 | 0.11 | 1.00 |
| residual at E, layer-2 input | 0.04 | **0.93** | **1.00** |
| layer-1 attention output at R | **0.97** (head 2: 0.66) | 0.07 | 0.00 |
| residual at R, layer-2 input | **0.97** | 0.07 | 0.00 |
| layer-2 attention output at R | 0.08 | **1.00** | **1.00** |
| layer-2 MLP output at R | 0.00 | 0.00 | 0.07 |

Mean-ablation (unique-argmax Bayes agreement; 1.000 = no effect; matched controls = other heads of the same layer):

| ablated | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| the stage-1 head(s) | L0-h2 0.886, L0-h1 0.961 | **L0-h1 0.700** | **L1-h3 0.664** |
| other heads of that layer | 1.000 | 1.000 | 1.000 |
| layer-2 attention at the readouts | 1.000 | 0.939 | **0.663** |
| the decisive layer-2 head alone | L2-h0 1.000 | L2-h1, L2-h3 1.000 each | **L2-h1 0.663** |
| jointly necessary pair (each alone ≤ 0.01 drop) | **L1-h2 + L2-h0 → 0.883** | **L2-h1 + L2-h3 → 0.936** | none (single heads already necessary) |
| layer-0 MLP (all positions) | 0.951 | 0.861 | 0.808 |

Conditional sufficiency with the layer-1 readout attention removed in both runs: patching the layer-2 attention at R gives
1.000 in all seeds, carried by L2-h0 (seed 0), L2-h1 0.67 / L2-h3 0.31 (seed 1), L2-h1 1.00 (seed 2).

Representation test (B2; answer-neutral questions — both evidence pairs agree on the same value, so the gate does not affect
the Bayes answer — linear probe for the gate state, majority ≈ 0.51):

| site | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| evidence tokens, layer-1 input | 0.73 | 0.92 | 0.47 |
| evidence tokens, layer-2 input | **1.00** | **1.00** | **1.00** |
| readout, layer-2 input | 0.91 | 1.00 | 0.48 |

## 3. Reconstruction (per seed)

- **Common to all three seeds (interventionally supported).** (i) The gate relation is computed by attention **at the
  evidence tokens**, reading the gate tokens (patching the gate tokens' own outputs transfers nothing); a single head does it
  (L0-h1 in seed 1, L1-h3 in seed 2; L0-h2 with L0-h1 in seed 0) and its ablation, unlike matched heads, breaks the decision.
  (ii) By the layer-2 input the gate state is linearly present at the evidence tokens **as a stored variable** — decodable
  at 1.00 even where it does not affect the answer (B2). (iii) The decision is formed at the readout token by **attention
  from the readout to the gate-tagged evidence tokens** (layer-1 or layer-2 attention at R carries the full flip in every
  seed); the readout's MLPs never carry it (≤ 0.07).
- **Seed-specific.** Which layer computes the gate (0 vs 1), which layer's readout attention carries the decision (1 vs 2),
  and whether a backup exists: seed 0 has a redundant pair across layers (primary L1-h2, backup L2-h0); seed 1 a redundant
  pair within layer 2 (L2-h1, L2-h3); seed 2 none (L2-h1 alone is necessary: unique-argmax Bayes agreement 0.663 after ablating it).
- **Why the salient last-layer attention was dispensable in seed 0 (question 11).** Because it is a backup route: the
  primary route (layer-1 head 2 at the readout) carries the decision, and the layer-2 head becomes decisive only when the
  primary route is removed (conditional transfer 1.000). The continuation's "last-layer attention is causally unnecessary"
  was a seed-0 property, not a GATE property: in seed 2 the last-layer head is the only route.

## 4. Frozen predictions (Q3-B) — outcomes

| | prediction | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| Q3-B1 | Bayes-level accuracy ≥ 0.89 | 0.903 | 0.900 | **pass** |
| Q3-B2 | layer-2 attention at readouts dispensable (agreement ≥ 0.98) | 0.939 | 0.663 | **FAIL** |
| Q3-B3 | a redundant head pair in layers 1–2 | L2-h1 + L2-h3 | none | **FAIL** (seed 2) |
| Q3-B4 | layer-0 attention at E transfers ≥ 0.5, at G ≤ 0.05 | 1.00 / 0.00 | 0.00 / 0.00 | **FAIL** (seed 2) |

The failed predictions were extrapolations of seed 0's specific layout; they are kept as failures. What replicated is the
abstract structure in §3 (common to all seeds), which was not itself frozen and is therefore post hoc across seeds.

## 5. Comparison with T4 reliability routing (B5; detail in RELATIONAL_COMPUTATIONAL_CLASSES.md)

Shared: a relation-derived variable is computed and stored at the **evidence (source) tokens**, and the decision is formed by
**readout attention to those tokens** (pattern-mediated in T4 A′ seeds: frozen-pattern tag ablation 0.02; in GATE the
decisive head's output carries the flip). Different: T4's variable is cross-question (episode-level reliability), GATE's is
within-question; T4 A′ places the variable on the key side of a last-layer head consistently, GATE's decisive layer varies
by seed; GATE shows explicit backup routes in 2 of 3 seeds.
Not established: that the GATE evidence-token variable and the T4 tag are the *same* abstract variable under any
intervention — no cross-task interchange is possible (different inputs), so the comparison is structural, not causal.

## 6. Generalisation variants (B4) — not run

Larger K, other constraints, noisy or multiple gates, changing gates: not run in this package (compute); listed in
OPEN_FOUNDATIONAL_RESEARCH_MAP.md.
