from neural_search.graph.schema import SUPPORTED_EDGE_TYPES, SUPPORTED_NODE_TYPES
from neural_search.spectral.kg import build_spectral_subgraph
from neural_search.spectral.schemas import (
    AperiodicEligibility,
    PeriodicPeak,
    SpectralEstimate,
    SpectralFeatureBundle,
    SpectralQCAssessment,
    SpectralRunConfig,
)


def _bundle(*, support_level: str = "high", missing_fields: list[str] | None = None) -> SpectralFeatureBundle:
    run_config = SpectralRunConfig(run_id="run:dandi:1", backend="mock", freq_range_hz=(2.0, 40.0))
    qc = SpectralQCAssessment(qc_id="qc:estimate:1", status="pass", flags=[])
    estimate = SpectralEstimate(
        estimate_id="estimate:1",
        dataset_id="dataset:dandi:000001",
        run_config=run_config,
        channel_id="ch0",
        region_id="V1",
        task_state_id="resting_state",
        aperiodic_offset=-2.0,
        aperiodic_exponent=2.0,
        fit_r_squared=0.95,
        fit_error=0.05,
        n_frequency_bins=80,
        peaks=[PeriodicPeak(center_frequency_hz=10.0, power=0.5, bandwidth_hz=2.0)],
        qc=qc,
    )
    eligibility = AperiodicEligibility(
        dataset_id="dataset:dandi:000001",
        support_level=support_level,
        confidence=0.85,
        missing_fields=missing_fields or [],
    )
    return SpectralFeatureBundle(
        bundle_id="bundle:dandi:000001",
        dataset_id="dataset:dandi:000001",
        eligibility=eligibility,
        run_config=run_config,
        estimates=[estimate],
    )


def test_build_spectral_subgraph_produces_valid_graph():
    graph = build_spectral_subgraph(_bundle())

    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    for node in graph.nodes.values():
        assert node.node_type in SUPPORTED_NODE_TYPES
    for edge in graph.edges.values():
        assert edge.edge_type in SUPPORTED_EDGE_TYPES
        assert edge.source_node_id in graph.nodes
        assert edge.target_node_id in graph.nodes


def test_build_spectral_subgraph_includes_expected_node_types():
    graph = build_spectral_subgraph(_bundle())
    node_types = {node.node_type for node in graph.nodes.values()}

    assert {
        "dataset",
        "spectral_feature_bundle",
        "spectral_run",
        "spectral_estimate",
        "aperiodic_component",
        "periodic_peak",
        "spectral_qc_assessment",
        "brain_region",
        "task_state_epoch",
        "channel",
    } <= node_types


def test_build_spectral_subgraph_emits_support_edges_only_when_eligible():
    eligible_graph = build_spectral_subgraph(_bundle(support_level="high"))
    eligible_edge_types = {edge.edge_type for edge in eligible_graph.edges.values()}
    assert "dataset_supports_aperiodic_reanalysis" in eligible_edge_types

    ineligible_graph = build_spectral_subgraph(_bundle(support_level="unsupported"))
    ineligible_edge_types = {edge.edge_type for edge in ineligible_graph.edges.values()}
    assert "dataset_supports_aperiodic_reanalysis" not in ineligible_edge_types


def test_build_spectral_subgraph_flags_missing_requirements():
    graph = build_spectral_subgraph(_bundle(support_level="medium", missing_fields=["channel_or_probe_metadata"]))
    edge_types = {edge.edge_type for edge in graph.edges.values()}

    assert "dataset_missing_aperiodic_requirement" in edge_types
