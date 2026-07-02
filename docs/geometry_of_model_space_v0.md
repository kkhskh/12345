# Geometry of Model Space V0

## 1. Goal

Build a high-dimensional model feature space `Phi(M)`, then create multiple 3D projections for different questions. The goal is not to assert a single true geometry of model space. It is to maintain several task-relevant empirical geometries while separating external-observable model geometry from internal mechanistic obstruction geometry.

## 2. Feature Vector Phi(M)

`Phi(M)` is a mixed continuous/categorical feature vector. Missing or unverified values should remain missing or marked `TODO_VERIFY`.

Capability:

- general benchmark score
- coding
- math
- science
- multilingual
- agentic/tool-use
- multimodal
- long-context

Modality:

- text in/out
- image in/out
- audio in/out
- video in/out
- document/PDF
- robotics/control

Openness/transparency:

- closed/API/open-weight/local
- license
- model card
- system card
- FMTI/developer transparency if available
- training recipe public?
- weights public?
- checkpoints public?

Cost/speed/deployment:

- input price
- output price
- latency
- throughput
- local deployability
- quantization support
- hardware burden

Architecture:

- decoder-only
- encoder-only
- encoder-decoder
- MoE
- multimodal fusion
- diffusion
- recurrent/state-space
- hybrid/tool system

Context/tool/reasoning:

- context window
- function calling
- tool use
- web/search/RAG
- computer use
- reasoning mode
- structured outputs

Mechanistic:

- logits access
- activation access
- patching feasible
- SAE feasible
- checkpoint availability
- transport `R^2`
- obstruction AUC
- partial corr(`O`, label | margin)
- attribution-collapse score
- obstruction-response lift over attribution
- KL/cost
- repair/stabilization score

## 3. Projection Families

Projection A: External capability manifold.

- Input features: benchmark, modality, context, tool-use, cost/speed, and observable behavior.
- Intended use: compare all systems, including closed APIs, at the behavioral and deployment level.
- Models included: all models/systems with external observations.
- Limitations: cannot infer hidden architecture, training data, parameters, activations, routing, or mechanisms for closed systems.

Projection B: Access/transparency/deployability manifold.

- Input features: access type, weights, license, model card/system card, local deployability, price, latency, hardware burden.
- Intended use: select systems for feasible empirical work.
- Models included: closed, API, open-weight, local, media, and agent systems.
- Limitations: transparency scores are source-dependent and can become stale.

Projection C: Mechanistic access manifold.

- Input features: logits access, activation access, patching feasibility, SAE feasibility, checkpoint availability, hardware burden, license.
- Intended use: identify models where internal obstruction tests can actually be run.
- Models included: open/local and partially instrumentable systems, plus closed systems as external-only reference points.
- Limitations: access does not guarantee meaningful causal response; hardware and tooling can dominate feasibility.

Projection D: Internal obstruction manifold.

- Input features: transport diagnostics, obstruction AUC, margin-controlled partial correlation, attribution-collapse score, obstruction-response lift, KL/cost, repair/stabilization score.
- Intended use: compare measured obstruction-response behavior across tasks and models.
- Models included: only models with internal measurements.
- Limitations: currently sparse; should not be projected as a global model atlas until enough measured rows exist.

Projection E: Task-response geometry.

- Input features: task family, candidate contrast type, margin regime, obstruction metrics, response metrics, intervention cost, repair/stability.
- Intended use: compare obstruction-response behavior across tasks within or across models.
- Models included: measured model-task pairs.
- Limitations: task design can dominate geometry; margin matching and baselines are required.

## 4. Basins / Strata

Here "basin" means empirical cluster or stratum in feature space. It does not mean a literal training-loss basin unless training trajectory data exists.

### 1. Closed Frontier API Basin

Observable signature: high external capability, API/UI access, broad modalities or tool interfaces, hidden internals.

Mathematical interpretation: external model manifold only.

Obstruction-response meaning: obstruction can be tested behaviorally only through outputs, margins/logprobs if available, stability, and perturbation response.

Internal feasibility: normally none without special research access.

Likely failure modes: hidden system wrappers, changing model versions, missing logprobs, refusal/scaffold effects.

### 2. Closed RAG/Agent System Basin

Observable signature: search, retrieval, tools, memory, planner, or agent wrapper around hidden model calls.

Mathematical interpretation: composite system graph with model, retrieval, tool, and scaffold sites.

Obstruction-response meaning: decompose model decision from retrieval/scaffold/tool decision before assigning obstruction.

Internal feasibility: external-only unless local scaffold or research access exists.

Likely failure modes: retriever failure mistaken for model obstruction, tool policy discontinuities, non-determinism.

### 3. Open Frontier MoE Basin

Observable signature: open/open-weight high-capability systems with public MoE-like architecture or routing claims from source notes.

Mathematical interpretation: computation complex includes router/expert nodes where public and accessible.

Obstruction-response meaning: candidate evidence may be incompatible across residual stream, router, experts, and merged outputs.

Internal feasibility: possible but often hardware-heavy.

Likely failure modes: routing discontinuities, inaccessible expert activations, costly patching.

### 4. Open Dense Generalist Basin

Observable signature: open-weight dense LMs with text or multimodal capabilities.

Mathematical interpretation: layer-position computation complex with residual, attention, and MLP sites.

Obstruction-response meaning: direct internal obstruction experiments are feasible when activation/logit access and patching are available.

Internal feasibility: medium to high, conditional on hardware and license.

Likely failure modes: scalar germs too weak, attribution collapse, margin confounds.

### 5. Research-Control Basin

Observable signature: controlled suites, multiple checkpoints, smaller models, or unusually strong research instrumentation.

Mathematical interpretation: same internal complex as the family, with better trajectory/checkpoint comparisons.

Obstruction-response meaning: strongest setting for falsification and transport diagnostics.

Internal feasibility: high relative to other basins.

Likely failure modes: results may not scale; benchmark capability may be lower.

### 6. Multimodal Open Basin

Observable signature: open or open-weight models with text plus image/audio/video/document paths.

Mathematical interpretation: modality-specific complexes plus fusion sites.

Obstruction-response meaning: cross-modal incompatibility can be tested when modality-local evidence conflicts with fused answer evidence.

Internal feasibility: conditional on activation access across encoders, projectors, fusion layers, and decoders.

Likely failure modes: one modality dominates; patching interfaces are fragile; evaluation can be subjective.

### 7. Generative Media Basin

Observable signature: image, video, or audio generation/editing systems.

Mathematical interpretation: denoising or media-token trajectory complexes.

Obstruction-response meaning: usually external prompt/conditioning consistency unless open weights allow latent or activation interventions.

Internal feasibility: mostly external-only for closed APIs.

Likely failure modes: stochastic decoding, subjective metrics, hidden safety filters.

### 8. Safety/Classifier Basin

Observable signature: moderation, guard, classifier, or policy models.

Mathematical interpretation: classification complex with candidate label contrasts.

Obstruction-response meaning: incompatibility between local evidence and policy label can be margin-controlled.

Internal feasibility: possible for open classifiers, external-only for closed moderation APIs.

Likely failure modes: policy thresholds, calibration artifacts, label ambiguity.

### 9. Tool/Planner/Memory Hybrid Basin

Observable signature: explicit memory, tool calls, planners, controllers, or multi-module systems.

Mathematical interpretation: heterogeneous computation complex across learned and non-learned modules.

Obstruction-response meaning: system-level obstruction may arise between local module evidence and final system action.

Internal feasibility: depends on local control of each subsystem.

Likely failure modes: assigning obstruction to the wrong module, discontinuous control policies, hidden state.

## 5. Differential-Geometric Framing

Model space is a stratified empirical manifold, not guaranteed smooth. Coordinates are mixed continuous/categorical features, and projections depend on preprocessing choices.

Metrics can be induced by feature scaling for atlas work, or by Fisher/KL where a meaningful probabilistic interface exists. Response fields may be defined over task/model coordinates by measuring how interventions change obstruction, margin, cost, and stability.

Singular regimes include routing boundaries, refusal thresholds, tool-selection discontinuities, decision-boundary crossings, and rank drops in transport or feature maps.

## 6. Algebraic-Topological Framing

Each model family induces a computation complex over internal or system sites. Sheaves over sites assign candidate-specific evidence spaces to vertices and compatibility maps to edges. Global sections represent compatible decision evidence. Obstruction is measured as metric distance or residual, not as `[d s]` in `H^1` for a globally defined 0-cochain.

Persistent or topological summaries may be useful diagnostics, but they should not be treated as evidence for the theory unless tied directly to causal response: predicted intervention, measured obstruction reduction, and measured margin/stability change.
