# Math Map V0: Causal Obstruction-Response Theory

## 0. Status

This is a V0 architecture/math atlas, not final empirical evidence. Its purpose is to preserve the broad project direction: a local-to-global obstruction-response framework for AI decision systems, with empirical tests organized as a falsification ladder.

Current project state:

- IOI was Experiment 1.
- IOI result: obstruction-response collapsed to attribution patching.
- IOI interpretation: attribution-positive / obstruction-negative; useful negative control.
- Experiment 2 is margin-matched factual conflict.
- Factual conflict tests whether obstruction detects internal incompatibility when final answer margin is approximately controlled.
- Results so far: GPT-2 small AUC 0.6219, partial corr 0.1793; GPT-2 medium AUC 0.6095, partial corr 0.1008; Pythia-410m AUC 0.6036, partial corr -0.0236; Pythia-1b AUC 0.6818, partial corr 0.2211.
- Pythia-1b is the first promising smoke result. It passes obstruction signal thresholds but fails the data-size gate.
- Next empirical gate: implement multi-token factual answer scoring, rerun Pythia-1b factual conflict, require matched facts >= 100, AUC > 0.65, and partial corr > 0.20 before component-level patching.

## 1. Central Thesis

AI decisions can be studied as local-to-global consistency events. For a candidate contrast, distributed local evidence across internal sites may glue into a stable global decision or exhibit measurable obstruction.

This is a candidate-specific, model-specific, task-specific framework for testing internal compatibility. It is not a universal claim that all cognition is literally a global section.

## 2. Main Objects

Let `M` be a model, `x` an input, and `c = (a, b)` a candidate contrast.

Candidate margin:

```text
m_c(x) = z_a(x) - z_b(x)
```

Computation complex:

```text
K_x = (V_x, E_x)
```

A vertex is an internal site:

```text
v = (layer, token/time/position, component type)
```

For candidate contrast `c`, each vertex has a candidate stalk:

```text
F_c(v)
```

The local decision germ at `v` is:

```text
s_c(v; x) = P^c_v phi_v(x)
```

Compatibility maps compare evidence across sites:

```text
rho^c_{u -> v}: F_c(u) -> F_c(v)
```

The candidate coboundary is:

```text
(d_c s_c)(e: u -> v) = rho^c_{u -> v} s_c(u; x) - s_c(v; x)
```

Compatible global sections are:

```text
H^0(K_x; F_c) = ker d_c
```

The corrected distance-to-global-section obstruction is:

```text
D_c(x) = min_{g in H^0(K_x; F_c)} ||s_c(x) - g||^2_M
```

The computable Laplacian residual obstruction is:

```text
O_c(x) = ||d_c s_c(x)||^2_W
       = sum_{e:u->v} w_e ||rho^c_{u->v} s_c(u; x) - s_c(v; x)||^2
```

Normalized obstruction:

```text
O_tilde_c(x) = O_c(x) / (||s_c(x)||^2_M + epsilon)
```

Fragility:

```text
Frag_c(x) = O_tilde_c(x) / (|m_c(x)| + epsilon)
```

Response score:

```text
R_c(u, delta; x) =
  alpha[-Delta O_c(u, delta; x)]
  + beta[Delta m_c(u, delta; x)]
  - gamma[Delta Cost_u(delta; x)]
```

Practical score:

```text
S_u = - predicted_Delta_O_u
      + beta * predicted_Delta_m_u
      - gamma * KL_or_Fisher_cost_u
```

Important correction: obstruction should not be defined as `[d s]` in `H^1` when `s` is a globally defined 0-cochain. That class is zero because `d s` is exact. Relative cohomology is relevant only for partial-section extension problems, where `U subset K` and one asks whether a partial section on `U` extends globally.

## 3. Closed-Loop Empirical Test

The closed-loop test is:

```text
local evidence
-> obstruction
-> predicted site/intervention
-> patch/ablate/steer
-> measured Delta obstruction
-> measured Delta margin/stability
-> repair/flip/stabilization
```

A positive result should show that obstruction predicts interventions that change compatibility and decision behavior beyond attribution-only baselines.

## 4. Two-Layer Atlas

Layer A: External model manifold.

This layer includes all models, including closed frontier models. Its geometry is observable only: outputs, benchmarks, modalities, context, price, latency, refusal behavior, tool behavior, and API logprobs if available. It does not claim hidden architecture, parameter count, training data, activations, or routing for closed systems unless explicitly public in source notes.

Layer B: Internal obstruction manifold.

This layer includes open/local or otherwise instrumentable models only. It supports activation capture, patching, steering, SAE analysis, transport fitting, obstruction metrics, and causal response tests. For open/open-weight models, mechanistic feasibility remains conditional on hardware, license, tokenizer/logit access, activation access, and patching feasibility.

## 5. Mathematical Scope vs Empirical Scope

Framework scope:

- broad model-family atlas
- candidate contrasts across text, multimodal, retrieval, agentic, ranking, control, and generative systems
- external-observable model geometry separated from internal mechanistic obstruction geometry

Empirical scope now:

- Transformer LMs first
- IOI negative control
- margin-matched factual conflict
- later MCQA, ambiguity/conflict, hallucination, and SAE feature-space experiments

Claim scope:

- candidate-specific
- model-specific
- task-specific
- falsifiable by attribution, margin, and intervention baselines

## 6. Theorem Targets

Energy-distance equivalence: under linear transports and positive weights, the Laplacian residual energy is equivalent to a metric distance from the space of compatible global sections, up to conditioning constants.

Minimum-energy repair under linear response: when interventions induce local linear changes in germs and costs are quadratic, the best repair direction solves a regularized least-squares response problem.

Obstruction-margin stability: for candidate contrasts with controlled margin, residual incompatibility can predict instability only if it contributes information beyond margin and attribution baselines.

Non-identifiability: different choices of stalks, projections, transports, and gauges may yield equivalent observed obstructions, so obstruction values are not unique without a measurement convention.

Local decision boundary under frozen attention/fixed activation region: inside a fixed local computation regime, margins and obstruction-response estimates may admit local linear or quadratic approximations.

Relative extension obstruction for partial sections: for `U subset K`, a partial section on `U` may fail to extend to a compatible global section; relative obstruction is meaningful in that extension problem, not as `[d s]` for a globally defined 0-cochain.

## 7. Failure Modes

- Obstruction collapses to attribution.
- Obstruction tracks margin only.
- Transport maps fail to predict compatibility across sites.
- Scalar germs are too weak to capture meaningful incompatibility.
- Patching changes margin without reducing obstruction.
- Obstruction reduction does not predict repair or stabilization.
- Closed systems only allow external geometry.
- Model-space clusters are empirical strata, not literal loss basins unless training trajectory data exists.

## 8. Current Empirical Ladder

1. IOI scalar residual obstruction: failed.
2. IOI multi-token/scalar variants: failed or attribution-dominated.
3. IOI component patching: attribution dominated.
4. Factual GPT-2 small: weak.
5. Factual GPT-2 medium: weak/no-go.
6. Factual Pythia-410m: no-go.
7. Factual Pythia-1b: promising smoke result but underpowered.

## 9. Next Empirical Gate

Do not do expensive component patching until multi-token factual scoring raises matched examples and Pythia-1b still passes:

- matched facts >= 100
- `AUC(O_conflict > O_clean) > 0.65`
- `partial corr(O, conflict_label | margin) > 0.20`

Only after this gate should the project move to component-level factual obstruction and patching.
