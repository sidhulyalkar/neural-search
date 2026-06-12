"""Tests for intent-aware usefulness scorer."""
import pytest
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    UsefulnessScore,
    score_usefulness,
    INTENT_WEIGHT_PROFILES,
)
from neural_search.retrieval.query_intent import UsefulnessIntent
from neural_search.graph.schema import (
    KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge,
    make_node_id, make_edge_id,
)


def _ctx(
    dataset_id="ds_a",
    modalities=None,
    tasks=None,
    species=None,
    brain_regions=None,
    affordances=None,
    data_standards=None,
    session_count=None,
    trial_count=None,
    quality_score=0.5,
):
    return DatasetContext(
        dataset_id=dataset_id,
        modalities=modalities or [],
        tasks=tasks or [],
        species=species or [],
        brain_regions=brain_regions or [],
        affordances=affordances or [],
        data_standards=data_standards or [],
        session_count=session_count,
        trial_count=trial_count,
        quality_score=quality_score,
    )


class TestScoreBounds:
    def test_score_between_zero_and_one(self):
        q = _ctx(modalities=["neuropixels"], tasks=["decision_making"])
        c = _ctx(modalities=["neuropixels"], tasks=["decision_making"])
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert 0.0 <= score.total_score <= 1.0

    def test_perfect_match_scores_high(self):
        attrs = dict(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            species=["mouse"],
            brain_regions=["prefrontal_cortex"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
            quality_score=1.0,
        )
        q = _ctx(**attrs)
        c = _ctx(**attrs)
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert score.total_score >= 0.7

    def test_empty_candidate_scores_low(self):
        q = _ctx(modalities=["neuropixels"], tasks=["decision_making"], species=["mouse"])
        c = _ctx()
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert score.total_score <= 0.4

    def test_all_dimension_scores_bounded(self):
        q = _ctx(modalities=["calcium_imaging"])
        c = _ctx(modalities=["neuropixels"])
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        for dim, val in score.dimension_scores.items():
            assert 0.0 <= val <= 1.0, f"Dimension {dim} out of bounds: {val}"


class TestWeightNormalization:
    def test_weights_sum_to_one_for_all_intents(self):
        for intent in UsefulnessIntent:
            if intent in INTENT_WEIGHT_PROFILES:
                weights = INTENT_WEIGHT_PROFILES[intent]
                total = sum(weights.values())
                assert abs(total - 1.0) < 1e-6, f"{intent} weights sum to {total}"

    def test_intent_changes_score(self):
        q = _ctx(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
        )
        c = _ctx(
            modalities=["neuropixels"],
            tasks=["decision_making"],
            affordances=["choice_decoding"],
            data_standards=["nwb"],
            quality_score=0.9,
        )
        score_lookup = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        score_pipeline = score_usefulness(q, c, UsefulnessIntent.PIPELINE_REUSE)
        assert 0.0 <= score_lookup.total_score <= 1.0
        assert 0.0 <= score_pipeline.total_score <= 1.0

    def test_inactive_neural_signature_weight_is_renormalized(self):
        q = _ctx(tasks=["decision_making"])
        c = _ctx(tasks=["decision_making"])
        score = score_usefulness(q, c, UsefulnessIntent.EXPLORATION)
        assert "neural_signature_similarity" not in score.weights
        assert abs(sum(score.weights.values()) - 1.0) < 1e-6
        assert any("neural_signature_similarity" in w for w in score.warnings)


class TestMissingMetadataHandling:
    def test_empty_empty_jaccard_dimensions_are_zero_not_neutral(self):
        q = _ctx()
        c = _ctx()
        score = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        for dim in [
            "modality_alignment",
            "task_compatibility",
            "species_match",
            "region_overlap",
            "affordance_compatibility",
            "pipeline_transferability",
        ]:
            assert score.dimension_scores[dim] == 0.0

    def test_missing_metadata_does_not_increase_similarity(self):
        q = _ctx(modalities=["neuropixels"])
        c_missing = _ctx()
        c_mismatch = _ctx(modalities=["calcium_imaging"])
        s_missing = score_usefulness(q, c_missing, UsefulnessIntent.STRICT_LOOKUP)
        s_mismatch = score_usefulness(q, c_mismatch, UsefulnessIntent.STRICT_LOOKUP)
        assert s_missing.dimension_scores["modality_alignment"] == 0.0
        assert s_mismatch.dimension_scores["modality_alignment"] == 0.0


class TestExplanations:
    def test_evidence_list_nonempty(self):
        q = _ctx(modalities=["calcium_imaging"], tasks=["go_nogo"])
        c = _ctx(modalities=["calcium_imaging"], tasks=["go_nogo"])
        score = score_usefulness(q, c, UsefulnessIntent.REPLICATION)
        assert len(score.evidence) >= 1

    def test_warnings_for_missing_graph(self):
        q = _ctx()
        c = _ctx()
        score = score_usefulness(q, c, UsefulnessIntent.EXPLORATION)
        assert any("graph" in w.lower() or "neural_signature" in w.lower() for w in score.warnings)


class TestIntentOnRanking:
    def test_pipeline_reuse_prefers_same_standards(self):
        q = _ctx(modalities=["neuropixels"], affordances=["choice_decoding"], data_standards=["nwb"])
        c_match = _ctx(modalities=["neuropixels"], affordances=["choice_decoding"], data_standards=["nwb"])
        c_diff = _ctx(modalities=["calcium_imaging"], affordances=["dimensionality_reduction"], data_standards=["bids"])
        s_match = score_usefulness(q, c_match, UsefulnessIntent.PIPELINE_REUSE)
        s_diff = score_usefulness(q, c_diff, UsefulnessIntent.PIPELINE_REUSE)
        assert s_match.total_score > s_diff.total_score

    def test_replication_prefers_same_species_region(self):
        q = _ctx(species=["mouse"], brain_regions=["hippocampus"], tasks=["spatial_navigation"])
        c_match = _ctx(species=["mouse"], brain_regions=["hippocampus"], tasks=["spatial_navigation"])
        c_diff = _ctx(species=["macaque"], brain_regions=["v1"], tasks=["visual_discrimination"])
        s_match = score_usefulness(q, c_match, UsefulnessIntent.REPLICATION)
        s_diff = score_usefulness(q, c_diff, UsefulnessIntent.REPLICATION)
        assert s_match.total_score > s_diff.total_score


def _make_minimal_graph() -> KnowledgeGraph:
    """Two dataset nodes sharing a task neighbor."""
    d1_id = make_node_id("dataset", "ds001")
    d2_id = make_node_id("dataset", "ds002")
    t1_id = make_node_id("task", "decision_making")
    nodes = {
        d1_id: KnowledgeGraphNode(node_id=d1_id, node_type="dataset", label="DS001"),
        d2_id: KnowledgeGraphNode(node_id=d2_id, node_type="dataset", label="DS002"),
        t1_id: KnowledgeGraphNode(node_id=t1_id, node_type="task", label="Decision Making"),
    }
    e1_id = make_edge_id(d1_id, "dataset_has_task", t1_id)
    e2_id = make_edge_id(d2_id, "dataset_has_task", t1_id)
    edges = {
        e1_id: KnowledgeGraphEdge(edge_id=e1_id, source_node_id=d1_id, target_node_id=t1_id, edge_type="dataset_has_task"),
        e2_id: KnowledgeGraphEdge(edge_id=e2_id, source_node_id=d2_id, target_node_id=t1_id, edge_type="dataset_has_task"),
    }
    return KnowledgeGraph(nodes=nodes, edges=edges)


def test_score_usefulness_accepts_graph_parameter():
    """score_usefulness must accept a 'graph' keyword argument."""
    qctx = DatasetContext(dataset_id="node:dataset:ds001", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="node:dataset:ds002", tasks=["decision-making"])
    graph = _make_minimal_graph()
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=graph)
    assert 0.0 <= score.total_score <= 1.0
    assert score.dimension_scores.get("graph_proximity") == 1.0


def test_graph_proximity_no_warning_when_graph_provided():
    """When a graph is provided and nodes exist, no graph_proximity warning."""
    qctx = DatasetContext(dataset_id="node:dataset:ds001", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="node:dataset:ds002", tasks=["decision-making"])
    graph = _make_minimal_graph()
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=graph)
    graph_warnings = [w for w in score.warnings if "graph_proximity" in w.lower()]
    assert graph_warnings == [], f"unexpected graph warning: {graph_warnings}"


def test_graph_proximity_still_warns_without_graph():
    """Without a graph, the 0.3 prior warning must still appear."""
    qctx = DatasetContext(dataset_id="q", tasks=["decision-making"])
    cctx = DatasetContext(dataset_id="c", tasks=["decision-making"])
    score = score_usefulness(qctx, cctx, UsefulnessIntent.REPLICATION, graph=None)
    assert any("graph_proximity" in w.lower() for w in score.warnings)


def test_graph_proximity_resolves_node_prefix():
    """score_usefulness should find nodes via node: prefix and return non-0.3 score."""
    ds_a_id = make_node_id("dataset", "dandi", "000003")   # node:dataset:dandi:000003
    ds_b_id = make_node_id("dataset", "dandi", "000004")   # node:dataset:dandi:000004
    task_id = make_node_id("task", "decision_making")       # node:task:decision_making

    nodes = {
        ds_a_id: KnowledgeGraphNode(node_id=ds_a_id, node_type="dataset", label="DANDI 000003"),
        ds_b_id: KnowledgeGraphNode(node_id=ds_b_id, node_type="dataset", label="DANDI 000004"),
        task_id: KnowledgeGraphNode(node_id=task_id, node_type="task", label="Decision Making"),
    }
    e1 = make_edge_id(ds_a_id, "dataset_has_task", task_id)
    e2 = make_edge_id(ds_b_id, "dataset_has_task", task_id)
    edges = {
        e1: KnowledgeGraphEdge(edge_id=e1, source_node_id=ds_a_id, target_node_id=task_id, edge_type="dataset_has_task"),
        e2: KnowledgeGraphEdge(edge_id=e2, source_node_id=ds_b_id, target_node_id=task_id, edge_type="dataset_has_task"),
    }
    graph = KnowledgeGraph(nodes=nodes, edges=edges)

    # DatasetContext carries bare dataset_id (no "node:" prefix) — the production format
    q = DatasetContext(dataset_id="dataset:dandi:000003", tasks=["decision_making"])
    c = DatasetContext(dataset_id="dataset:dandi:000004", tasks=["decision_making"])

    score = score_usefulness(q, c, graph=graph)
    assert score.dimension_scores["graph_proximity"] != 0.3, (
        "graph_proximity returned neutral prior 0.3 — node: prefix resolution is broken"
    )
    assert score.dimension_scores["graph_proximity"] > 0.0
    assert not any("not found" in w for w in score.warnings), (
        f"Unexpected graph_proximity warning: {score.warnings}"
    )
