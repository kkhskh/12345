"""Typed structures for the T0 model-family and model-space atlas.

This module is intentionally lightweight: it uses only the Python standard
library and performs no network access. The structures are meant to keep the
math-map/model-atlas artifact machine-readable while leaving factual model
metadata marked for external verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class AccessType(str, Enum):
    """Observable access category for a model or system."""

    CLOSED_API_UI = "closed_api_ui"
    CLOSED_SYSTEM_WRAPPER = "closed_system_wrapper"
    OPEN_WEIGHT_API = "open_weight_api"
    OPEN_WEIGHT_LOCAL = "open_weight_local"
    RESEARCH_CONTROL_SUITE = "research_control_suite"
    MEDIA_API = "media_api"
    TOOL_RAG_AGENT_SYSTEM = "tool_rag_agent_system"
    UNKNOWN = "unknown"


class MechanisticAccess(str, Enum):
    """Mechanistic feasibility category."""

    E = "external_only"
    P = "partial_mechanistic_access"
    M = "internal_mechanistic_possible"
    M_STAR = "possible_but_hardware_heavy"
    M_PLUS_PLUS = "unusually_good_for_mechanistic_research"
    U = "unknown"


class ModelFamily(str, Enum):
    """Broad architecture or system family."""

    DECODER_ONLY_TRANSFORMER = "decoder_only_transformer"
    ENCODER_ONLY_TRANSFORMER = "encoder_only_transformer"
    ENCODER_DECODER_TRANSFORMER = "encoder_decoder_transformer"
    VISION_TRANSFORMER = "vision_transformer"
    MULTIMODAL_TRANSFORMER = "multimodal_transformer"
    MOE_TRANSFORMER = "moe_transformer"
    RAG_SYSTEM = "retrieval_augmented_generation_system"
    TOOL_AGENT_SYSTEM = "tool_using_agentic_llm_system"
    SAE_FEATURE_SPACE = "sparse_autoencoder_feature_space"
    STATE_SPACE_RECURRENT = "state_space_recurrent_sequence_model"
    DIFFUSION = "diffusion_model"
    AUTOREGRESSIVE_MEDIA = "autoregressive_image_video_audio_model"
    RL_WORLD_MODEL_AGENT = "rl_policy_world_model_agent"
    RECOMMENDER_RANKER = "recommendation_ranking_system"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    HYBRID_SYSTEM = "hybrid_memory_tool_planner_controller_system"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    """Input/output modality flags."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TOOL = "tool"
    ROBOTICS_CONTROL = "robotics_control"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class ProjectionType(str, Enum):
    """Projection families for model-space geometry."""

    EXTERNAL_CAPABILITY = "external_capability"
    ACCESS_TRANSPARENCY_DEPLOYABILITY = "access_transparency_deployability"
    MECHANISTIC_ACCESS = "mechanistic_access"
    INTERNAL_OBSTRUCTION = "internal_obstruction"
    TASK_RESPONSE = "task_response"


@dataclass(frozen=True)
class ComputationComplexSpec:
    """Specification of the computation complex K_x."""

    name: str
    vertices: list[str]
    edges: list[tuple[str, str]]
    vertex_description: str
    edge_description: str
    notes: str = ""

    def validate(self) -> list[str]:
        """Return validation warnings for missing or inconsistent fields."""

        warnings: list[str] = []
        if not self.name:
            warnings.append("complex name is empty")
        if not self.vertices:
            warnings.append("complex has no vertices")
        vertex_set = set(self.vertices)
        for source, target in self.edges:
            if source not in vertex_set:
                warnings.append(f"edge source {source!r} is not in vertices")
            if target not in vertex_set:
                warnings.append(f"edge target {target!r} is not in vertices")
        return warnings


@dataclass(frozen=True)
class GermSpec:
    """Specification of a local decision germ s_c(v; x)."""

    name: str
    stalk_dimension: str
    activation_source: str
    projection: str
    candidate_specific: bool = True
    notes: str = ""


@dataclass(frozen=True)
class TransportSpec:
    """Specification of compatibility maps rho^c_{u->v}."""

    name: str
    map_type: str
    fit_data: str
    regularization: str
    diagnostics: list[str]
    notes: str = ""


@dataclass(frozen=True)
class ObstructionSpec:
    """Formulas and cautions for obstruction metrics."""

    name: str
    raw_formula: str
    normalized_formula: str
    fragility_formula: str
    invalid_definitions_to_avoid: list[str]
    notes: str = ""


@dataclass(frozen=True)
class ResponseSpec:
    """Specification of obstruction-response scoring and measurements."""

    name: str
    score_formula: str
    intervention_types: list[str]
    predicted_quantities: list[str]
    observed_quantities: list[str]
    cost_terms: list[str]
    notes: str = ""


@dataclass(frozen=True)
class ModelFamilyMap:
    """Map from a model family to obstruction-response objects."""

    family: ModelFamily
    description: str
    representative_models: list[str]
    decision_types: list[str]
    candidate_contrasts: list[str]
    complex_spec: ComputationComplexSpec
    germ_specs: list[GermSpec]
    transport_specs: list[TransportSpec]
    obstruction_spec: ObstructionSpec
    response_spec: ResponseSpec
    external_experiments: list[str]
    internal_experiments: list[str]
    failure_modes: list[str]
    current_relevance: str

    def validate(self) -> list[str]:
        """Return validation warnings for required atlas fields."""

        warnings = self.complex_spec.validate()
        if not self.germ_specs:
            warnings.append(f"{self.family.value} has no germ specs")
        if not self.transport_specs:
            warnings.append(f"{self.family.value} has no transport specs")
        if not self.failure_modes:
            warnings.append(f"{self.family.value} has no failure modes")
        return warnings


@dataclass(frozen=True)
class ModelAtlasEntry:
    """One row in the model atlas."""

    model_id: str
    canonical_name: str
    provider: str
    family: ModelFamily
    access_type: AccessType
    mechanistic_access: MechanisticAccess
    modalities: list[Modality]
    architecture_public: str
    parameter_count_public: str
    context_window_public: str
    license: str
    source_notes: str
    verification_status: str
    atlas_role: str
    known_unknowns: list[str]


@dataclass(frozen=True)
class ModelSpaceFeatureSchema:
    """Feature schema for high-dimensional model-space projections."""

    feature_blocks: dict[str, list[str]]
    projection_types: list[ProjectionType]
    required_columns: list[str]
    optional_columns: list[str]


def default_obstruction_spec() -> ObstructionSpec:
    """Return the corrected default obstruction formulas."""

    return ObstructionSpec(
        name="candidate_laplacian_residual_obstruction",
        raw_formula=(
            "O_c(x) = ||d_c s_c(x)||^2_W = "
            "sum_{e:u->v} w_e ||rho^c_{u->v} s_c(u; x) - s_c(v; x)||^2"
        ),
        normalized_formula="O_tilde_c(x) = O_c(x) / (||s_c(x)||^2_M + epsilon)",
        fragility_formula="Frag_c(x) = O_tilde_c(x) / (|m_c(x)| + epsilon)",
        invalid_definitions_to_avoid=[
            "Do not define obstruction as [d s] in H^1 for a globally defined 0-cochain.",
            "Do not treat exact coboundaries as nonzero cohomology classes.",
            "Use relative cohomology only for partial-section extension problems U subset K.",
        ],
        notes="D_c(x) is the corrected distance-to-global-section target; O_c(x) is the computable residual.",
    )


def default_response_spec() -> ResponseSpec:
    """Return the default response scoring specification."""

    return ResponseSpec(
        name="candidate_obstruction_response",
        score_formula=(
            "S_u = - predicted_Delta_O_u + beta * predicted_Delta_m_u "
            "- gamma * KL_or_Fisher_cost_u"
        ),
        intervention_types=["patch", "ablate", "steer", "feature_patch", "route_patch"],
        predicted_quantities=["predicted_Delta_O", "predicted_Delta_m", "predicted_cost"],
        observed_quantities=["measured_Delta_O", "measured_Delta_m", "stability", "repair_or_flip"],
        cost_terms=["KL", "Fisher", "activation_norm", "intervention_size"],
        notes="Response is meaningful only when tested against attribution and margin baselines.",
    )


def decoder_transformer_family_map() -> ModelFamilyMap:
    """Return the current empirical-focus family map."""

    complex_spec = ComputationComplexSpec(
        name="layer_position_component_complex",
        vertices=["residual_stream", "attention_head", "mlp", "logit_readout"],
        edges=[
            ("residual_stream", "attention_head"),
            ("attention_head", "residual_stream"),
            ("residual_stream", "mlp"),
            ("mlp", "residual_stream"),
            ("residual_stream", "logit_readout"),
        ],
        vertex_description="Sites are layer, token position, and component type.",
        edge_description="Edges follow residual flow and component contribution paths.",
        notes="Current IOI and factual-conflict experiments live here.",
    )
    return ModelFamilyMap(
        family=ModelFamily.DECODER_ONLY_TRANSFORMER,
        description="Autoregressive text Transformer over causal token positions.",
        representative_models=["gpt2-small", "gpt2-medium", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b"],
        decision_types=["next-token", "multi-token answer", "candidate contrast"],
        candidate_contrasts=["token a vs token b", "answer string a vs answer string b"],
        complex_spec=complex_spec,
        germ_specs=[
            GermSpec(
                name="candidate_residual_germ",
                stalk_dimension="scalar or low-rank candidate subspace",
                activation_source="residual stream / component output",
                projection="candidate logit direction or fitted projection P^c_v",
                notes="Scalar germs have already shown attribution-collapse risk.",
            )
        ],
        transport_specs=[
            TransportSpec(
                name="layerwise_candidate_transport",
                map_type="regularized linear map",
                fit_data="held-out clean/conflict activations",
                regularization="ridge or low-rank constraint",
                diagnostics=["transport_R2", "heldout_residual", "condition_number"],
            )
        ],
        obstruction_spec=default_obstruction_spec(),
        response_spec=default_response_spec(),
        external_experiments=["margin-matched factual conflict", "multi-token answer scoring"],
        internal_experiments=["activation patching", "component obstruction", "transport fitting"],
        failure_modes=["attribution collapse", "margin-only signal", "weak scalar germs"],
        current_relevance="Primary empirical focus; Pythia-1b factual conflict is promising but underpowered.",
    )


def moe_transformer_family_map() -> ModelFamilyMap:
    """Return a family map for mixture-of-experts Transformers."""

    return ModelFamilyMap(
        family=ModelFamily.MOE_TRANSFORMER,
        description="Transformer with router-selected expert subnetworks.",
        representative_models=["TODO_VERIFY open MoE rows from atlas seed"],
        decision_types=["next-token", "sequence", "tool/action if scaffolded"],
        candidate_contrasts=["candidate answer", "candidate route-conditioned answer"],
        complex_spec=ComputationComplexSpec(
            name="router_expert_component_complex",
            vertices=["residual_stream", "router", "expert", "merge", "logit_readout"],
            edges=[
                ("residual_stream", "router"),
                ("router", "expert"),
                ("expert", "merge"),
                ("merge", "residual_stream"),
                ("residual_stream", "logit_readout"),
            ],
            vertex_description="Residual, router, expert, merge, and readout sites.",
            edge_description="Edges include routing and expert merge dependencies.",
        ),
        germ_specs=[
            GermSpec(
                name="route_conditioned_candidate_germ",
                stalk_dimension="candidate evidence plus optional route evidence",
                activation_source="router logits, expert outputs, residual stream",
                projection="candidate projection and route-conditioned projection",
            )
        ],
        transport_specs=[
            TransportSpec(
                name="router_expert_transport",
                map_type="piecewise or route-conditioned linear map",
                fit_data="examples with observable routes/experts",
                regularization="route-stratified ridge",
                diagnostics=["route_stability", "transport_R2", "expert_residual"],
            )
        ],
        obstruction_spec=default_obstruction_spec(),
        response_spec=default_response_spec(),
        external_experiments=["prompt stability", "margin-matched conflict tests"],
        internal_experiments=["route patching", "expert ablation", "expert-output patching"],
        failure_modes=["routing discontinuity", "hardware-heavy patching", "hidden routing in closed models"],
        current_relevance="Atlas breadth; internal experiments only where routing and activations are accessible.",
    )


def rag_agent_family_map() -> ModelFamilyMap:
    """Return a family map for RAG and agentic systems."""

    return ModelFamilyMap(
        family=ModelFamily.TOOL_AGENT_SYSTEM,
        description="Composite systems with retrieval, tools, memory, planners, or controllers.",
        representative_models=["TODO_VERIFY RAG and agent rows from atlas seed"],
        decision_types=["tool call", "retrieved evidence selection", "answer", "plan step"],
        candidate_contrasts=["tool a vs tool b", "answer a vs answer b", "source a vs source b"],
        complex_spec=ComputationComplexSpec(
            name="system_scaffold_complex",
            vertices=["query", "retriever", "tool", "memory", "model_call", "output"],
            edges=[
                ("query", "retriever"),
                ("retriever", "model_call"),
                ("query", "tool"),
                ("tool", "model_call"),
                ("memory", "model_call"),
                ("model_call", "output"),
            ],
            vertex_description="System modules and model-call states.",
            edge_description="Edges represent dataflow and control flow.",
        ),
        germ_specs=[
            GermSpec(
                name="system_decision_germ",
                stalk_dimension="module-native candidate evidence",
                activation_source="retrieval scores, tool observations, model outputs, memory state",
                projection="task-specific candidate projection",
            )
        ],
        transport_specs=[
            TransportSpec(
                name="module_interface_transport",
                map_type="empirical interface map",
                fit_data="controlled system traces",
                regularization="task-specific",
                diagnostics=["trace_coverage", "interface_residual", "decision_stability"],
            )
        ],
        obstruction_spec=default_obstruction_spec(),
        response_spec=default_response_spec(),
        external_experiments=["controlled retrieval conflict", "tool-result perturbation", "memory conflict"],
        internal_experiments=["local scaffold patching", "retrieval patching", "model-call patching when open"],
        failure_modes=["retriever failure mistaken for model obstruction", "scaffold dominates", "non-determinism"],
        current_relevance="Future system-level extension; requires decomposition of model vs scaffold/tool/retrieval decisions.",
    )


def sae_feature_family_map() -> ModelFamilyMap:
    """Return a family map for sparse-autoencoder feature-space obstruction."""

    return ModelFamilyMap(
        family=ModelFamily.SAE_FEATURE_SPACE,
        description="Feature-space obstruction over sparse-autoencoder latents.",
        representative_models=["TODO_VERIFY SAE artifacts for open models"],
        decision_types=["feature-mediated token decision", "concept contrast", "behavior contrast"],
        candidate_contrasts=["feature/concept a vs feature/concept b", "answer a vs answer b"],
        complex_spec=ComputationComplexSpec(
            name="sae_feature_complex",
            vertices=["residual_site", "sae_feature", "decoder_direction", "logit_readout"],
            edges=[
                ("residual_site", "sae_feature"),
                ("sae_feature", "decoder_direction"),
                ("decoder_direction", "residual_site"),
                ("residual_site", "logit_readout"),
            ],
            vertex_description="Residual and SAE feature sites.",
            edge_description="Edges represent encode/decode and readout dependencies.",
        ),
        germ_specs=[
            GermSpec(
                name="candidate_feature_germ",
                stalk_dimension="sparse feature coordinates",
                activation_source="SAE latents and reconstructed residual directions",
                projection="candidate feature/logit direction",
            )
        ],
        transport_specs=[
            TransportSpec(
                name="feature_transport",
                map_type="feature-to-feature or feature-to-residual map",
                fit_data="activation and SAE latent datasets",
                regularization="sparsity-aware ridge",
                diagnostics=["reconstruction_error", "feature_stability", "transport_R2"],
            )
        ],
        obstruction_spec=default_obstruction_spec(),
        response_spec=default_response_spec(),
        external_experiments=["behavioral contrast paired with SAE availability"],
        internal_experiments=["feature patching", "feature clamping", "residual reconstruction patching"],
        failure_modes=["dictionary artifact", "feature splitting", "non-causal feature", "poor reconstruction"],
        current_relevance="Later feature-space route; may be crucial if residual scalar obstruction remains weak.",
    )


def default_model_feature_schema() -> ModelSpaceFeatureSchema:
    """Return the default model-space feature schema."""

    feature_blocks = {
        "capability": [
            "general_benchmark_score",
            "coding_score",
            "math_score",
            "science_score",
            "multilingual_score",
            "agentic_tool_use_score",
            "multimodal_score",
            "long_context_score",
        ],
        "modality": [
            "modality_text",
            "modality_image",
            "modality_audio",
            "modality_video",
            "modality_document",
            "modality_tool",
            "modality_robotics_control",
        ],
        "openness_transparency": [
            "access_type",
            "license",
            "model_card_available",
            "system_card_available",
            "training_recipe_public",
            "weights_available",
            "checkpoints_available",
        ],
        "cost_speed_deployment": [
            "input_price_per_million_tokens",
            "output_price_per_million_tokens",
            "latency_ms",
            "throughput_tokens_per_second",
            "local_deployable",
            "quantization_support",
            "hardware_burden",
        ],
        "architecture": [
            "family",
            "architecture_decoder_only",
            "architecture_encoder_only",
            "architecture_encoder_decoder",
            "architecture_moe",
            "architecture_multimodal_fusion",
            "architecture_diffusion",
            "architecture_recurrent_state_space",
            "architecture_hybrid_tool_system",
        ],
        "context_tool_reasoning": [
            "context_window_tokens",
            "function_calling",
            "tool_use",
            "web_search_rag",
            "computer_use",
            "reasoning_mode",
            "structured_outputs",
        ],
        "mechanistic": [
            "logits_available",
            "activations_available",
            "patching_feasible",
            "sae_feasible",
            "checkpoint_availability",
            "transport_r2",
        ],
        "obstruction_metrics": [
            "obstruction_auc_factual_conflict",
            "obstruction_partial_corr_margin_controlled",
            "attribution_collapse_score",
            "obstruction_response_lift",
            "kl_cost",
            "repair_stabilization_score",
        ],
    }
    return ModelSpaceFeatureSchema(
        feature_blocks=feature_blocks,
        projection_types=list(ProjectionType),
        required_columns=[
            "model_id",
            "canonical_name",
            "provider",
            "family",
            "access_type",
            "mechanistic_access",
            "source_notes",
            "verification_status",
        ],
        optional_columns=sorted({column for columns in feature_blocks.values() for column in columns}),
    )


def validate_model_entry(entry: ModelAtlasEntry) -> list[str]:
    """Return validation warnings for a model atlas entry."""

    warnings: list[str] = []
    if not entry.model_id:
        warnings.append("model_id is required")
    if not entry.canonical_name:
        warnings.append("canonical_name is required")
    if not entry.provider:
        warnings.append(f"{entry.model_id}: provider is required")
    if not entry.modalities:
        warnings.append(f"{entry.model_id}: at least one modality should be specified")
    if entry.access_type in {AccessType.CLOSED_API_UI, AccessType.CLOSED_SYSTEM_WRAPPER, AccessType.MEDIA_API}:
        if entry.mechanistic_access not in {MechanisticAccess.E, MechanisticAccess.U, MechanisticAccess.P}:
            warnings.append(f"{entry.model_id}: closed systems should not claim full mechanistic access")
    if entry.verification_status == "TODO_VERIFY" and "TODO" not in entry.source_notes:
        warnings.append(f"{entry.model_id}: TODO_VERIFY row should keep source_notes conservative")
    return warnings


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_enum_to_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _enum_to_value(item) for key, item in value.items()}
    return value


def atlas_entry_to_dict(entry: ModelAtlasEntry) -> dict[str, Any]:
    """Convert a model atlas entry to a JSON-serializable dictionary."""

    return _enum_to_value(asdict(entry))


def schema_to_json_dict(schema: ModelSpaceFeatureSchema) -> dict[str, Any]:
    """Convert a feature schema to a JSON-serializable dictionary."""

    return _enum_to_value(asdict(schema))


def _default_family_maps() -> list[ModelFamilyMap]:
    return [
        decoder_transformer_family_map(),
        moe_transformer_family_map(),
        rag_agent_family_map(),
        sae_feature_family_map(),
    ]


if __name__ == "__main__":
    for family_map in _default_family_maps():
        warnings = family_map.validate()
        status = "ok" if not warnings else f"{len(warnings)} warning(s)"
        print(
            f"{family_map.family.value}: "
            f"{len(family_map.complex_spec.vertices)} vertex types, "
            f"{len(family_map.complex_spec.edges)} edge types, "
            f"{status}"
        )
