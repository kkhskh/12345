# Main research result

## The integrated result (§14)

A Transformer can use the relation between independently specified evidence sources to make a candidate decision. That use is identifiable, but only by internal interventions. Behaviour cannot identify it.

The chain below holds for the strongest mechanism, C′ (conditional rereading through Q-composition):

```
source semantics fixed by generator
  → representations identified
  → relation feature characterised
  → downstream operation identified
  → causal intervention moves that operation
  → candidate-specific response follows
  → survives controls and baselines
```

It then hits a hard limit: for every candidate-directed agreement rule, there is a decision-equivalent circuit with no relation computation at all (T-RF).

The same limit governs the learned models:

- **T1** (candidate-symmetric XNOR) demands a cross-source interaction. Small Transformers learn one there, and it is causally identifiable.
- **T2 and T5a** (Bayesian integration), and **T3, T4 and T5b** (tasks *designed* to require "reconciliation"), do not demand one. The learned models show no causally used relation feature on these tasks, and relation-free baselines match them.

## Arrow-by-arrow record

| arrow | proof | computational evidence | assumptions | counterexamples / limits | not established |
|---|---|---|---|---|---|
| 1. Source semantics are fixed independently | the generator (`semantics.py`); no variable is read from any model | §4 scenario catalogue | the generator is the ground truth | — | that any pretrained model has such sources |
| 2. Representations identified | C′ closed form: R carries `(x, y)` exactly on e1, e2 (§1 of `CPRIME_…`) | closed form vs forward pass to 1e−14 | engineered basis | in learned models, only relative to probes (R² 0.3–1.0) | basis-free identification in learned models |
| 3. Relation feature characterised | `q_R = h(c1(u+v)) − h(c1(u−v))`; `sign q_R = sign(ab)` under (N1)–(N4) (A1*) | 0/120,000 sign violations; derived cones for every perturbation, 0 violations | (N1)–(N4); otherwise cones | (A0) false; magnitude is not separable (P4) | — |
| 4. Downstream operation identified | block-2 query gain `G = λq̂/√2`, scores `±Gμ/σ` (closed form) | C2: `q_R` carries 100% of TE, `q_B` 0%, values 1.8% | single gate head, no rereader | redundant copies break single-site necessity (C3) | uniqueness of the mediator |
| 5. Intervention moves the operation | C2-OFF proof: the `q_R` patch is off-manifold | **on-manifold:** matched relation-blind pairs (X4), e.g. (1, 1) vs (3, −0.238), equal `m_blind = 0.822` | reachability (T-R) | strong agreement has no matched partner (T-R) | — |
| 6. Candidate-specific response | C′1*: source-vs-prior selection is set by `sign G` and `G` versus the log value-mass ratio | opposite decisions in X4 (+1.890 vs −2.630); certified regions | prior votes c′; weights positive | dead bands are unavoidable (T-CO): agreement floor ≈ 0.12, conflict band ≈ 0.6% | — |
| 7. Survives controls and baselines | T-SS (sum-only reader is identical on matched pairs); T-B-TM, T-C4 | sum-only Δm = 0 exactly; competing writers, rereaders and bypasses map out where it fails | C4 assumptions | B-add and T-RF reproduce the **decisions** | **any behavioural or predictive advantage of the relation account** |

## What the result is, and is not

**Positive (survives).**

1. Corrected, certified operating theorems for the engineered mechanisms: A1*, C′1*, B-TM, C4. These are decision-level theorems with signed, inspectable conditions.
2. An on-manifold identification criterion that separates "uses the relation" from "uses the aggregate": exact matching of the relation-blind pathway (X4), within the reachable set characterised by T-R.
3. In learned models: when the objective is candidate-symmetric agreement (T1), the relation is computed, causally used, directionally predictable from frozen weights (86–94% for TF1), and gauge-robust at the population level.

**Negative (also survives).**

1. Theorem **T-RF**: candidate-directed agreement rules ("follow agreeing sources, otherwise defer") are threshold logic. Behaviour alone never shows relation use. C′ has a decision-equivalent relation-free twin.
2. The learned models do not develop relation computation for reconciliation tasks. They solve them additively.
3. For K ≥ 3, the bilinear relation computed by a single attention score cannot equal candidate-relative agreement (**T-B-IMP**). The routing it produces suppresses 8.4% of genuinely agreeing pairs.
4. Every agreement-gated sign rule needs a dead band (**T-CO**). The historical unrestricted C′1 claim had an impossible form.
5. Decodability of `ab` or `sign(ab)` coexists with zero causal use: in T2, `ab` has R² up to 0.99 while the margin depends only on `a + b`.

**What this does not add beyond known theory.**

- The mechanisms are ordinary nonlinear gating and Q-composition.
- The identification criteria are path-specific and interchange interventions.
- T-RF is the classical linear-threshold / XOR distinction.

The contribution is the **precise placement of the "consistency" question**:

- It has computational content only for candidate-symmetric relations.
- For candidate-directed rules, it is a statement about internal implementation, checkable only by matched internal interventions.
- It carries no predictive advantage over relation-free accounts.

This document does not propose closing Step 1. `OPEN_RESEARCH_OBLIGATIONS.md` lists the work that remains.
