"""Builds a knowledge-graph subgraph from a computed spectral feature bundle.

The dataset, channel, and brain-region nodes referenced here are emitted as
low-confidence placeholders (matching the convention used by
``neural_search.graph.builder``); merge this subgraph with the dataset's
primary subgraph via ``neural_search.graph.builder.merge_graphs`` to resolve
them against the richer canonical nodes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)
from neural_search.spectral.schemas import SpectralEstimate, SpectralFeatureBundle

EXTRACTOR_NAME = "neural_search.spectral.kg"
EXTRACTOR_VERSION = "v0.1.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dataset_node_id(dataset_id: str) -> str:
    parts = dataset_id.split(":")
    if len(parts) >= 3 and parts[0] == "dataset":
        return make_node_id("dataset", *parts[1:])
    return make_node_id("dataset", dataset_id)


def _evidence(source_id: str, source_field: str, evidence_text: str, confidence: float) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"evidence:{source_id}:{source_field}",
        source_type="spectral_feature_bundle",
        source_id=source_id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=confidence,
        extractor_name=EXTRACTOR_NAME,
        extractor_version=EXTRACTOR_VERSION,
    )


def _add_node(nodes: dict[str, KnowledgeGraphNode], node: KnowledgeGraphNode) -> None:
    nodes.setdefault(node.node_id, node)


def _add_edge(
    edges: dict[str, KnowledgeGraphEdge],
    source_node_id: str,
    edge_type: str,
    target_node_id: str,
    *,
    confidence: float,
    evidence: list[GraphEvidence],
    properties: dict | None = None,
) -> None:
    edge = KnowledgeGraphEdge(
        edge_id=make_edge_id(source_node_id, edge_type, target_node_id),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence=confidence,
        evidence=evidence,
        properties=properties or {},
    )
    edges[edge.edge_id] = edge


def _placeholder_node(node_id: str, node_type: str, label: str, source_id: str) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        source_ids=[source_id],
        properties={"placeholder": True},
        confidence=0.35,
        created_at=_now(),
    )


def _estimate_nodes_and_edges(
    nodes: dict[str, KnowledgeGraphNode],
    edges: dict[str, KnowledgeGraphEdge],
    bundle: SpectralFeatureBundle,
    bundle_node_id: str,
    run_node_id: str,
    dataset_node_id: str,
    estimate: SpectralEstimate,
) -> None:
    estimate_node_id = make_node_id("spectral_estimate", estimate.estimate_id)
    evidence = [_evidence(estimate.estimate_id, "spectral_estimate", str(estimate.estimate_id), 0.9)]

    _add_node(
        nodes,
        KnowledgeGraphNode(
            node_id=estimate_node_id,
            node_type="spectral_estimate",
            label=f"Spectral estimate {estimate.estimate_id}",
            source_ids=[estimate.estimate_id],
            properties={
                "dataset_id": estimate.dataset_id,
                "n_frequency_bins": estimate.n_frequency_bins,
                "fit_r_squared": estimate.fit_r_squared,
                "fit_error": estimate.fit_error,
            },
            evidence=evidence,
            confidence=0.9,
            created_at=estimate.created_at,
        ),
    )
    _add_edge(edges, dataset_node_id, "dataset_has_spectral_estimate", estimate_node_id, confidence=0.9, evidence=evidence)
    _add_edge(edges, bundle_node_id, "dataset_has_spectral_estimate", estimate_node_id, confidence=0.9, evidence=evidence)
    _add_edge(edges, estimate_node_id, "spectral_estimate_generated_by_run", run_node_id, confidence=0.95, evidence=evidence)

    aperiodic_node_id = make_node_id("aperiodic_component", estimate.estimate_id)
    _add_node(
        nodes,
        KnowledgeGraphNode(
            node_id=aperiodic_node_id,
            node_type="aperiodic_component",
            label=f"Aperiodic component {estimate.estimate_id}",
            source_ids=[estimate.estimate_id],
            properties={
                "offset": estimate.aperiodic_offset,
                "exponent": estimate.aperiodic_exponent,
                "knee_hz": estimate.aperiodic_knee_hz,
                "mode": estimate.run_config.aperiodic_mode,
            },
            evidence=evidence,
            confidence=estimate.fit_r_squared,
            created_at=estimate.created_at,
        ),
    )
    _add_edge(edges, estimate_node_id, "spectral_estimate_has_aperiodic_component", aperiodic_node_id, confidence=estimate.fit_r_squared, evidence=evidence)
    _add_edge(edges, aperiodic_node_id, "aperiodic_component_estimated_by_method", run_node_id, confidence=0.95, evidence=evidence)

    for index, peak in enumerate(estimate.peaks):
        peak_node_id = make_node_id("periodic_peak", estimate.estimate_id, str(index))
        _add_node(
            nodes,
            KnowledgeGraphNode(
                node_id=peak_node_id,
                node_type="periodic_peak",
                label=f"Peak @ {peak.center_frequency_hz:.1f} Hz",
                source_ids=[estimate.estimate_id],
                properties={
                    "center_frequency_hz": peak.center_frequency_hz,
                    "power": peak.power,
                    "bandwidth_hz": peak.bandwidth_hz,
                    "band_label": peak.band_label,
                },
                evidence=evidence,
                confidence=0.7,
                created_at=estimate.created_at,
            ),
        )
        _add_edge(edges, estimate_node_id, "spectral_estimate_has_periodic_peak", peak_node_id, confidence=0.7, evidence=evidence)
        _add_edge(edges, peak_node_id, "periodic_peak_estimated_by_method", run_node_id, confidence=0.9, evidence=evidence)

    if estimate.qc is not None:
        qc_node_id = make_node_id("spectral_qc_assessment", estimate.qc.qc_id)
        _add_node(
            nodes,
            KnowledgeGraphNode(
                node_id=qc_node_id,
                node_type="spectral_qc_assessment",
                label=f"QC: {estimate.qc.status}",
                source_ids=[estimate.estimate_id],
                properties={"status": estimate.qc.status, "flags": estimate.qc.flags, "notes": estimate.qc.notes},
                evidence=evidence,
                confidence=1.0,
                created_at=estimate.qc.created_at,
            ),
        )
        _add_edge(edges, estimate_node_id, "spectral_estimate_has_qc_assessment", qc_node_id, confidence=1.0, evidence=evidence)

    if estimate.region_id:
        region_node_id = make_node_id("brain_region", estimate.region_id)
        _add_node(nodes, _placeholder_node(region_node_id, "brain_region", estimate.region_id, estimate.estimate_id))
        _add_edge(edges, estimate_node_id, "spectral_estimate_measured_in_region", region_node_id, confidence=0.6, evidence=evidence)

    if estimate.task_state_id:
        state_node_id = make_node_id("task_state_epoch", estimate.task_state_id)
        _add_node(nodes, _placeholder_node(state_node_id, "task_state_epoch", estimate.task_state_id, estimate.estimate_id))
        _add_edge(edges, estimate_node_id, "spectral_estimate_measured_during_state", state_node_id, confidence=0.6, evidence=evidence)

    if estimate.channel_id:
        channel_node_id = make_node_id("channel", estimate.channel_id)
        _add_node(nodes, _placeholder_node(channel_node_id, "channel", estimate.channel_id, estimate.estimate_id))
        _add_edge(edges, estimate_node_id, "spectral_estimate_measured_from_channel", channel_node_id, confidence=0.6, evidence=evidence)


def build_spectral_subgraph(bundle: SpectralFeatureBundle) -> KnowledgeGraph:
    """Build a knowledge-graph subgraph for one computed spectral feature
    bundle: the bundle, its run config, every estimate, their aperiodic
    components, periodic peaks, and QC assessments."""

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    dataset_node_id_value = _dataset_node_id(bundle.dataset_id)
    _add_node(nodes, _placeholder_node(dataset_node_id_value, "dataset", bundle.dataset_id, bundle.dataset_id))

    bundle_node_id = make_node_id("spectral_feature_bundle", bundle.bundle_id)
    bundle_evidence = [_evidence(bundle.bundle_id, "spectral_feature_bundle", bundle.bundle_id, 0.9)]
    _add_node(
        nodes,
        KnowledgeGraphNode(
            node_id=bundle_node_id,
            node_type="spectral_feature_bundle",
            label=f"Spectral feature bundle {bundle.bundle_id}",
            source_ids=[bundle.dataset_id],
            properties={
                "overall_qc_status": bundle.overall_qc_status,
                "n_estimates": len(bundle.estimates),
                "eligibility_support_level": bundle.eligibility.support_level,
                "interpretation_cautions": bundle.interpretation_cautions,
            },
            evidence=bundle_evidence,
            confidence=bundle.eligibility.confidence,
            created_at=bundle.created_at,
        ),
    )
    _add_edge(edges, dataset_node_id_value, "dataset_has_spectral_feature_bundle", bundle_node_id, confidence=bundle.eligibility.confidence, evidence=bundle_evidence)

    if bundle.eligibility.support_level in ("high", "medium"):
        _add_edge(edges, dataset_node_id_value, "dataset_supports_aperiodic_reanalysis", bundle_node_id, confidence=bundle.eligibility.confidence, evidence=bundle_evidence)
        _add_edge(edges, dataset_node_id_value, "dataset_reanalyzable_by_pipeline", bundle_node_id, confidence=bundle.eligibility.confidence, evidence=bundle_evidence, properties={"pipeline": "aperiodic_spectral_parameterization"})
    if bundle.eligibility.missing_fields:
        _add_edge(
            edges,
            dataset_node_id_value,
            "dataset_missing_aperiodic_requirement",
            bundle_node_id,
            confidence=bundle.eligibility.confidence,
            evidence=bundle_evidence,
            properties={"missing_fields": bundle.eligibility.missing_fields},
        )

    run_node_id = make_node_id("spectral_run", bundle.run_config.run_id)
    _add_node(
        nodes,
        KnowledgeGraphNode(
            node_id=run_node_id,
            node_type="spectral_run",
            label=f"Spectral run {bundle.run_config.run_id} ({bundle.run_config.backend})",
            source_ids=[bundle.dataset_id],
            properties=bundle.run_config.model_dump(mode="json"),
            evidence=bundle_evidence,
            confidence=1.0,
            created_at=bundle.run_config.created_at,
        ),
    )
    for estimate in bundle.estimates:
        _estimate_nodes_and_edges(nodes, edges, bundle, bundle_node_id, run_node_id, dataset_node_id_value, estimate)

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "graph_version": "v0.1.0",
            "builder": EXTRACTOR_NAME,
            "bundle_id": bundle.bundle_id,
            "dataset_id": bundle.dataset_id,
            "estimate_count": len(bundle.estimates),
        },
    )
