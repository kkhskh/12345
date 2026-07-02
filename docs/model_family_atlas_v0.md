# Model Family Atlas V0

This atlas maps model families to the obstruction-response objects `K`, `F`, `s`, `rho`, `O`, and `R`. It is an architecture-level map, not a claim that every family is already empirically validated.

Decoder-only Transformers are the current empirical focus. Other families preserve the broad scope of the theory and identify later falsification paths.

## Decoder-Only Text Transformers

### Definition

Autoregressive language models that predict next-token distributions from prior tokens.

### Representative systems

GPT-2 small, GPT-2 medium, EleutherAI Pythia-410m, EleutherAI Pythia-1b, and other open/open-weight decoder LMs. Closed decoder-only systems are external-only unless public instrumentation exists.

### Decision type

Next-token or sequence continuation decisions.

### Candidate contrast c = (a, b)

Two answer tokens, token sequences, classes, completions, or decoded candidates.

### Computation complex K_x

Layer-by-position graph over residual stream, attention, MLP, and optional head/component sites.

### Vertices V_x

`(layer, token position, component type)` such as residual pre/post, attention head output, MLP output, or logit lens site.

### Edges E_x

Residual stream flow, attention-mediated dependencies, MLP updates, and transport links between adjacent or causally connected sites.

### Stalks F_c(v)

Candidate-specific evidence subspaces, scalar margins, low-rank residual projections, logit-lens coordinates, or SAE features.

### Local germs s_c(v; x)

Projected local evidence for candidate `a` over `b`.

### Transport maps rho^c_{u->v}

Linear or regularized maps predicting downstream candidate evidence from upstream evidence.

### Obstruction O_c(x)

Weighted residual incompatibility across the computation complex.

### Natural interventions

Activation patching, component ablation, steering vectors, causal tracing, transport-guided repairs, SAE feature patching.

### Observable metrics

Margins, logprobs, accuracy, calibration, refusal behavior, answer stability, KL cost, obstruction AUC, partial correlation conditional on margin.

### External-only experiments

Prompt contrast tests, margin-matched behavioral tests, multi-token answer scoring, consistency under paraphrase.

### Internal mechanistic experiments

Activation capture, residual decomposition, attention/MLP patching, transport fitting, causal response tests.

### Failure modes

Obstruction collapses to attribution; margin explains the signal; scalar germs are too weak; patching changes the answer without reducing obstruction.

### Current project relevance

Current empirical focus. IOI was a useful negative control. Margin-matched factual conflict is the current smoke-test path.

## Encoder-Only Transformers

### Definition

Bidirectional Transformer encoders used for classification, tagging, retrieval, or representation learning.

### Representative systems

BERT-family and RoBERTa-family models. Specific rows require TODO_VERIFY source grounding before publication.

### Decision type

Class label, span label, token label, embedding similarity, or retrieval score.

### Candidate contrast c = (a, b)

Two labels, spans, classes, or retrieval candidates.

### Computation complex K_x

Layer-by-token graph with bidirectional attention and classifier/readout sites.

### Vertices V_x

Token representations, pooled representation, attention heads, MLP blocks, classifier head.

### Edges E_x

Bidirectional attention dependencies and layer transitions.

### Stalks F_c(v)

Candidate-label evidence subspaces or classifier-head projections.

### Local germs s_c(v; x)

Local support for label `a` versus label `b`.

### Transport maps rho^c_{u->v}

Maps across layers, tokens, and pooled readout.

### Obstruction O_c(x)

Residual mismatch between local label evidence and downstream class evidence.

### Natural interventions

Token representation patching, head ablation, feature steering, counterfactual span replacement.

### Observable metrics

Accuracy, logit margin, confidence, calibration, robustness under perturbation.

### External-only experiments

Contrastive classification and paraphrase consistency.

### Internal mechanistic experiments

Layerwise probes, patching, transport diagnostics, representation similarity.

### Failure modes

Probe artifacts, shortcut features, label leakage, obstruction tracking confidence only.

### Current project relevance

Later candidate for classification-style obstruction tests after decoder LM gates.

## Encoder-Decoder Transformers

### Definition

Sequence-to-sequence models with separate encoder and decoder stacks.

### Representative systems

T5-style and translation/summarization models. Specific model metadata is TODO_VERIFY.

### Decision type

Generated output sequence conditional on encoded input.

### Candidate contrast c = (a, b)

Two output tokens, answer strings, translations, summaries, or labels.

### Computation complex K_x

Encoder token graph, decoder token graph, and cross-attention links.

### Vertices V_x

Encoder sites, decoder sites, cross-attention heads, output readout sites.

### Edges E_x

Encoder flow, decoder causal flow, cross-attention dependencies.

### Stalks F_c(v)

Candidate-specific evidence in encoder, decoder, and cross-attention subspaces.

### Local germs s_c(v; x)

Projected evidence for output candidate contrast.

### Transport maps rho^c_{u->v}

Within-stack and cross-stack compatibility maps.

### Obstruction O_c(x)

Mismatch between encoded evidence, decoder state, and output candidate support.

### Natural interventions

Encoder patching, decoder patching, cross-attention ablation, source-span edits.

### Observable metrics

Sequence logprob, exact match, BLEU/ROUGE-like task metrics where appropriate, answer stability.

### External-only experiments

Counterfactual input edits, controlled translation or summarization contrasts.

### Internal mechanistic experiments

Cross-attention transport fitting, encoder-decoder patching, causal mediation.

### Failure modes

Decoder prior dominates encoder evidence; transport across stacks fails; sequence metrics obscure local decisions.

### Current project relevance

Later extension for seq2seq tasks.

## Vision Transformers

### Definition

Transformer models over image patches or visual tokens for classification, detection, or vision-language bridging.

### Representative systems

ViT-family models. Specific model metadata is TODO_VERIFY.

### Decision type

Image class, region class, embedding match, or visual answer support.

### Candidate contrast c = (a, b)

Two classes, regions, captions, or visual concepts.

### Computation complex K_x

Layer-by-patch graph with class token and patch-token sites.

### Vertices V_x

Patch tokens, class token, attention heads, MLP blocks.

### Edges E_x

Attention between patches and layer transitions.

### Stalks F_c(v)

Class-contrast visual evidence subspaces.

### Local germs s_c(v; x)

Patch-local evidence for candidate classes or concepts.

### Transport maps rho^c_{u->v}

Patch-to-class-token and layerwise evidence maps.

### Obstruction O_c(x)

Incompatibility between local patch evidence and global class evidence.

### Natural interventions

Patch masking, activation patching, concept steering, attention/head ablation.

### Observable metrics

Class margins, confidence, robustness, localization stability.

### External-only experiments

Occlusion, counterfactual image edits, prompt-free perturbation tests.

### Internal mechanistic experiments

Patch-token tracing, attention map interventions, representation patching.

### Failure modes

Texture shortcuts, localization artifacts, obstruction tracking confidence only.

### Current project relevance

Model-family breadth; not current empirical focus.

## Multimodal Transformers

### Definition

Models combining text with image, audio, video, document, or other modalities.

### Representative systems

Closed and open multimodal systems from provider and open-weight families. Closed systems are external-only unless instrumentation is available.

### Decision type

Cross-modal answer, caption, classification, grounding, or multimodal generation.

### Candidate contrast c = (a, b)

Two answers, grounded entities, regions, captions, or multimodal actions.

### Computation complex K_x

Modality-specific token graphs plus fusion/cross-attention sites.

### Vertices V_x

Text tokens, image/video/audio tokens, fusion layers, modality projectors, decoder sites.

### Edges E_x

Within-modality flow and cross-modal fusion dependencies.

### Stalks F_c(v)

Candidate evidence in each modality and fused representation.

### Local germs s_c(v; x)

Local support for candidate contrast from each modality.

### Transport maps rho^c_{u->v}

Modality-to-fusion and fusion-to-output maps.

### Obstruction O_c(x)

Mismatch between modality-local evidence and fused answer evidence.

### Natural interventions

Modality masking, region/audio segment patching, text prompt patching, fusion-layer patching.

### Observable metrics

Answer margin, grounding consistency, cross-modal robustness, refusal/stability.

### External-only experiments

Image-text conflict tests, audio-text conflict tests, document-answer consistency.

### Internal mechanistic experiments

Conditional on open weights and activation access: fusion transport fitting and cross-modal patching.

### Failure modes

One modality dominates; closed systems hide internal geometry; benchmark scores do not isolate compatibility.

### Current project relevance

Important later family for broad atlas; not current empirical focus.

## Mixture-of-Experts Transformers

### Definition

Transformers with router-selected expert subnetworks.

### Representative systems

Open MoE families and closed MoE-like systems where architecture is public. Hidden routing must not be claimed for closed models without source grounding.

### Decision type

Next-token, sequence, or multimodal decisions with routed computation.

### Candidate contrast c = (a, b)

Candidate token, answer, route-conditioned answer, or class contrast.

### Computation complex K_x

Layer-position graph augmented with router and expert nodes.

### Vertices V_x

Residual sites, router logits, selected experts, expert outputs, merge sites.

### Edges E_x

Residual flow, router-to-expert assignments, expert-to-merge links.

### Stalks F_c(v)

Candidate evidence and route/expert evidence.

### Local germs s_c(v; x)

Local candidate support and route-conditioned evidence.

### Transport maps rho^c_{u->v}

Maps across residual stream, router choices, experts, and merged output.

### Obstruction O_c(x)

Candidate-evidence incompatibility plus possible router/expert obstruction.

### Natural interventions

Expert ablation, route patching, residual patching, expert-output steering.

### Observable metrics

Margin, route stability if public, latency/cost, answer stability.

### External-only experiments

Prompt sensitivity and output stability across conflict cases.

### Internal mechanistic experiments

Conditional on routing and activation access: router/expert transport fitting and route-response interventions.

### Failure modes

Routing discontinuities, sparse expert activation, hardware burden, non-public routing.

### Current project relevance

Important for atlas breadth; internal work only where access permits.

## Retrieval-Augmented Generation Systems

### Definition

Systems combining a model with retrieval, search, documents, or external memory.

### Representative systems

RAG products, search-grounded QA systems, local RAG stacks.

### Decision type

Answer generation with retrieval-conditioned evidence.

### Candidate contrast c = (a, b)

Two answers, sources, citations, or retrieved evidence sets.

### Computation complex K_x

Composite graph over query, retriever, retrieved documents, reranker, generator, and citation/output sites.

### Vertices V_x

Query embeddings, retrieved chunks, reranker scores, generator token sites, citation sites.

### Edges E_x

Retrieval links, evidence selection, context injection, generation flow.

### Stalks F_c(v)

Candidate answer evidence at retrieval and generation stages.

### Local germs s_c(v; x)

Local support for candidate from retrieved evidence or generated state.

### Transport maps rho^c_{u->v}

Evidence propagation maps from retrieval to generation and output.

### Obstruction O_c(x)

Mismatch between retrieved evidence, selected context, and final answer support.

### Natural interventions

Document injection/removal, retrieval patching, reranker edits, context ablation, generator patching where local.

### Observable metrics

Answer accuracy, citation support, retrieval recall, faithfulness, margin/logprob if available.

### External-only experiments

Controlled corpus conflicts, citation perturbations, retrieval recall tests.

### Internal mechanistic experiments

Only in local/instrumentable RAG stacks; decompose model decision from scaffold, tool, and retrieval decisions.

### Failure modes

Retriever failure mistaken for model obstruction; generator ignores retrieved evidence; scaffold behavior dominates.

### Current project relevance

Later external/internal split case. Requires decomposition of model decision versus retrieval/scaffold decision.

## Tool-Using / Agentic LLM Systems

### Definition

Systems where an LLM selects tools, actions, plans, or computer-use steps.

### Representative systems

Coding agents, research agents, browser/computer-use agents, function-calling systems.

### Decision type

Tool call, plan step, action, answer, or termination decision.

### Candidate contrast c = (a, b)

Two tools, actions, plans, answers, or continue/stop decisions.

### Computation complex K_x

Hierarchical graph over model calls, tool observations, memory, planner/controller states, and final outputs.

### Vertices V_x

Prompt states, tool-call decisions, observations, memory entries, planner states, model-token sites if accessible.

### Edges E_x

Call sequence, tool result dependencies, memory updates, controller transitions.

### Stalks F_c(v)

Evidence for action or answer candidates at each system stage.

### Local germs s_c(v; x)

Local support for candidate action or answer.

### Transport maps rho^c_{u->v}

Maps across scaffold states and model calls.

### Obstruction O_c(x)

Mismatch between plan evidence, tool observations, memory, and final action/answer.

### Natural interventions

Tool result edits, memory edits, plan patching, prompt steering, model-call patching if local.

### Observable metrics

Task success, tool accuracy, action stability, cost, latency, refusal/error rate.

### External-only experiments

Tool-result conflict tests, plan perturbation, controlled environment tasks.

### Internal mechanistic experiments

Only where model and scaffold are instrumentable; decompose scaffold/tool decisions from model decisions.

### Failure modes

System-level obstruction hides model-level obstruction; discontinuous tool policies; non-determinism.

### Current project relevance

Broad project direction; not current empirical focus.

## Sparse-Autoencoder Feature-Space Models

### Definition

Feature-space representations induced by sparse autoencoders or related dictionary models over activations.

### Representative systems

SAE artifacts for open models where available. Specific artifact claims require TODO_VERIFY source grounding.

### Decision type

Feature-mediated token, concept, or behavior decision.

### Candidate contrast c = (a, b)

Two features, concepts, answers, or classes.

### Computation complex K_x

Feature graph over layers, positions, and SAE latents.

### Vertices V_x

SAE features, residual sites, decoder directions, feature interaction sites.

### Edges E_x

Feature activation dependencies, layer transitions, decoder-induced links.

### Stalks F_c(v)

Candidate-specific feature coordinates.

### Local germs s_c(v; x)

Feature-local evidence for candidate contrast.

### Transport maps rho^c_{u->v}

Feature-to-feature or feature-to-residual maps.

### Obstruction O_c(x)

Incompatibility among feature-level evidence streams.

### Natural interventions

Feature clamping, feature patching, dictionary-direction steering, residual reconstruction patching.

### Observable metrics

Feature activation, reconstruction error, margin, KL cost, repair/stability.

### External-only experiments

Behavioral contrast tests paired with feature availability checks.

### Internal mechanistic experiments

SAE feature obstruction and response tests. This may be crucial if residual scalar obstruction is weak.

### Failure modes

Dictionary artifacts, feature splitting, poor reconstruction, non-causal features.

### Current project relevance

Later feature-space version and possible route beyond weak scalar germs.

## State-Space / Recurrent Sequence Models

### Definition

Sequence models with recurrent or state-space dynamics rather than standard attention-only computation.

### Representative systems

State-space and recurrent model families. Specific entries are TODO_VERIFY.

### Decision type

Token, sequence, class, or control output.

### Candidate contrast c = (a, b)

Two tokens, answers, states, or actions.

### Computation complex K_x

Time-indexed recurrent/state graph.

### Vertices V_x

Hidden states, update modules, readout sites.

### Edges E_x

Temporal transitions and readout links.

### Stalks F_c(v)

Candidate evidence in hidden state coordinates.

### Local germs s_c(v; x)

Projected local evidence at each time.

### Transport maps rho^c_{u->v}

State-transition-induced evidence maps.

### Obstruction O_c(x)

Mismatch between temporally propagated evidence and readout evidence.

### Natural interventions

State patching, input perturbation, hidden-state steering, readout edits.

### Observable metrics

Margin, sequence accuracy, stability under perturbation.

### External-only experiments

Controlled sequence conflict tests.

### Internal mechanistic experiments

State capture, transition fitting, recurrent patching where accessible.

### Failure modes

Long-horizon credit assignment, state compression, transport mismatch.

### Current project relevance

Future family for testing whether attention-specific assumptions are unnecessary.

## Diffusion Models

### Definition

Generative models that iteratively denoise latent or pixel/media states.

### Representative systems

Image, video, and audio diffusion systems. Closed media APIs are external-only unless open weights are available.

### Decision type

Generation trajectory, prompt adherence, concept inclusion, or edit selection.

### Candidate contrast c = (a, b)

Two concepts, image attributes, edit targets, or prompt interpretations.

### Computation complex K_x

Timestep-by-layer graph over denoising states and conditioning paths.

### Vertices V_x

Denoising blocks, attention sites, conditioning tokens, latent sites.

### Edges E_x

Timestep transitions, conditioning links, U-Net/Transformer block flow.

### Stalks F_c(v)

Concept-contrast evidence in latent or conditioning space.

### Local germs s_c(v; x)

Local support for concept `a` versus `b`.

### Transport maps rho^c_{u->v}

Timestep and conditioning evidence maps.

### Obstruction O_c(x)

Mismatch between prompt/conditioning evidence and denoising trajectory evidence.

### Natural interventions

Prompt edits, attention control, latent patching, timestep steering.

### Observable metrics

Prompt adherence, classifier/embedding scores, human preference, edit consistency.

### External-only experiments

Prompt conflict and edit consistency tests.

### Internal mechanistic experiments

Mostly external-only unless open weights, latent access, and patching are available.

### Failure modes

Evaluation subjectivity, stochasticity, conditioning dominance, closed-media opacity.

### Current project relevance

Broad atlas family; not current empirical focus.

## Autoregressive Image/Video/Audio Models

### Definition

Models that generate media tokens autoregressively.

### Representative systems

Open or closed media-token generators. Specific metadata is TODO_VERIFY.

### Decision type

Next media token, segment, frame, or audio unit.

### Candidate contrast c = (a, b)

Two token continuations, concepts, frames, or acoustic alternatives.

### Computation complex K_x

Layer-by-time/media-token graph.

### Vertices V_x

Media tokens, text-conditioning tokens, attention/MLP sites, output heads.

### Edges E_x

Causal token flow and conditioning links.

### Stalks F_c(v)

Candidate media-token or concept evidence.

### Local germs s_c(v; x)

Projected local evidence for media candidate contrast.

### Transport maps rho^c_{u->v}

Layer/time transport maps.

### Obstruction O_c(x)

Incompatibility across conditioning, local media state, and output continuation.

### Natural interventions

Token patching, conditioning edits, latent/state steering.

### Observable metrics

Likelihood if available, embedding similarity, prompt adherence, stability.

### External-only experiments

Controlled prompt conflict and continuation consistency.

### Internal mechanistic experiments

Conditional on open weights and activation access.

### Failure modes

Large token spaces, stochastic decoding, subjective metrics.

### Current project relevance

Future extension.

## RL Policies / World-Model Agents

### Definition

Policies or world-model systems that select actions based on observations, memory, and objectives.

### Representative systems

Embodied agents, game agents, robotics policies, simulator-trained policies. Specific entries are TODO_VERIFY.

### Decision type

Action, plan, value estimate, or belief update.

### Candidate contrast c = (a, b)

Two actions, plans, states, or value hypotheses.

### Computation complex K_x

Observation encoder, memory/state, policy head, value head, planner/world-model graph.

### Vertices V_x

Observation tokens, hidden states, action logits, value estimates, planner nodes.

### Edges E_x

Temporal transitions, policy/value dependencies, planning rollouts.

### Stalks F_c(v)

Action-contrast or state-hypothesis evidence.

### Local germs s_c(v; x)

Local support for action or plan contrast.

### Transport maps rho^c_{u->v}

State-transition and policy-readout maps.

### Obstruction O_c(x)

Mismatch between observation evidence, world model, value, and action.

### Natural interventions

Observation edits, state patching, policy logit steering, memory interventions.

### Observable metrics

Return, action stability, safety violations, value calibration.

### External-only experiments

Controlled environment perturbations and counterfactual observations.

### Internal mechanistic experiments

Possible in local policies with state access.

### Failure modes

Distribution shift, delayed reward, policy stochasticity, environment confounding.

### Current project relevance

Future decision-system generalization.

## Recommendation/Ranking Systems

### Definition

Systems that rank items, ads, documents, or users.

### Representative systems

Retrievers, rankers, recommendation stacks. Specific production metadata is TODO_VERIFY.

### Decision type

Pairwise preference, ranking order, click probability, or allocation.

### Candidate contrast c = (a, b)

Two items, documents, ads, or ranked slates.

### Computation complex K_x

User/item feature graph, candidate scoring modules, reranker, policy constraints.

### Vertices V_x

Feature embeddings, interaction layers, score heads, reranker modules.

### Edges E_x

Feature interactions and ranking pipeline dependencies.

### Stalks F_c(v)

Pairwise preference evidence.

### Local germs s_c(v; x)

Local support for item `a` over item `b`.

### Transport maps rho^c_{u->v}

Feature-to-score and score-to-rank maps.

### Obstruction O_c(x)

Mismatch between feature evidence, score evidence, and final ranking.

### Natural interventions

Feature ablation, counterfactual item edits, score calibration, reranker patching.

### Observable metrics

Ranking metrics, calibration, pairwise stability, fairness/risk metrics.

### External-only experiments

Counterfactual feature and slate perturbations.

### Internal mechanistic experiments

Possible only in instrumentable systems.

### Failure modes

Feedback loops, hidden business rules, confounding, non-stationary preferences.

### Current project relevance

Broadens beyond generative AI; not current empirical focus.

## Graph Neural Networks

### Definition

Models operating over graph-structured data via message passing or graph attention.

### Representative systems

Graph classifiers, link predictors, molecular/property models. Specific entries are TODO_VERIFY.

### Decision type

Node label, graph label, edge/link prediction, or property estimate.

### Candidate contrast c = (a, b)

Two labels, links, graph properties, or actions.

### Computation complex K_x

Input graph crossed with message-passing layers.

### Vertices V_x

Graph nodes, edges, layer states, readout sites.

### Edges E_x

Message-passing edges, layer transitions, pooling links.

### Stalks F_c(v)

Candidate evidence on graph nodes/edges/readouts.

### Local germs s_c(v; x)

Local support for candidate contrast.

### Transport maps rho^c_{u->v}

Message-passing compatibility maps.

### Obstruction O_c(x)

Incompatibility between local graph evidence and global prediction.

### Natural interventions

Node/edge masking, message patching, subgraph counterfactuals.

### Observable metrics

Accuracy, calibration, robustness, explanation consistency.

### External-only experiments

Subgraph perturbations and controlled graph conflicts.

### Internal mechanistic experiments

Message-level patching and transport fitting in open models.

### Failure modes

Oversmoothing, spurious subgraphs, graph distribution shift.

### Current project relevance

Mathematical fit for complexes; future non-LM testbed.

## Hybrid Systems with Memory, Tools, Planners, Controllers

### Definition

Composite AI systems combining learned models with memory, tools, planners, controllers, retrievers, or rules.

### Representative systems

Production agents, local agent stacks, robotics controllers, RAG-plus-tool systems.

### Decision type

Composite decision: answer, action, plan, retrieval, tool call, memory update, or controller command.

### Candidate contrast c = (a, b)

Two system-level actions, outputs, plans, or memory states.

### Computation complex K_x

Heterogeneous graph over learned modules, tools, memory, planners, controllers, and output stages.

### Vertices V_x

Model calls, memory cells, retrieved documents, planner states, controller states, tool observations.

### Edges E_x

Dataflow, control flow, tool-call dependencies, memory updates.

### Stalks F_c(v)

Candidate evidence in each module's native representation.

### Local germs s_c(v; x)

Local support for a candidate system decision.

### Transport maps rho^c_{u->v}

Interface maps between modules, often empirical and task-specific.

### Obstruction O_c(x)

System-level incompatibility across module evidence and final behavior.

### Natural interventions

Memory edits, tool-result edits, planner interventions, model patching where accessible, controller constraints.

### Observable metrics

Task success, cost, latency, failure rate, stability, safety outcomes.

### External-only experiments

End-to-end perturbations, environment counterfactuals, tool/memory conflict tests.

### Internal mechanistic experiments

Only for instrumentable components; must decompose model, scaffold, tool, retrieval, planner, and controller decisions.

### Failure modes

Attribution to wrong subsystem, discontinuous policies, hidden state, non-determinism, opaque closed modules.

### Current project relevance

Long-term architecture-level target; keeps the theory from narrowing to IOI or factual conflict.
