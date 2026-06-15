"""Tests for graph-derived usefulness signals."""
import pytest

from neural_search.retrieval.graph_usefulness import (
    affordance_overlap,
    complementarity_score,
    graph_usefulness_features,
    normalized_metapath_score,
    pipeline_overlap,
)


def _ds(affordances=None, data_standards=None, dataset_id="ds_x"):
    return {
        "dataset_id": dataset_id,
        "affordances": affordances or [],
        "data_standards": data_standards or [],
    }


def _graph(nodes=None, edges=None):
    return {"nodes": nodes or {}, "edges": edges or {}}


class TestAffordanceOverlap:
    def test_identical_returns_one(self):
        a = _ds(affordances=["choice_decoding", "q_learning"])
        assert affordance_overlap(a, a) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self):
        a = _ds(affordances=["choice_decoding"])
        b = _ds(affordances=["calcium_imaging"])
        assert affordance_overlap(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _ds(affordances=["a", "b", "c"])
        b = _ds(affordances=["b", "c", "d"])
        result = affordance_overlap(a, b)
        assert 0.0 < result < 1.0

    def test_empty_both_returns_zero(self):
        assert affordance_overlap(_ds(), _ds()) == pytest.approx(0.0)


class TestPipelineOverlap:
    def test_same_standard_returns_one(self):
        a = _ds(data_standards=["nwb"])
        assert pipeline_overlap(a, a) == pytest.approx(1.0)

    def test_different_standard_returns_zero(self):
        a = _ds(data_standards=["nwb"])
        b = _ds(data_standards=["bids"])
        assert pipeline_overlap(a, b) == pytest.approx(0.0)


class TestComplementarity:
    def test_complementary_affordances_score_high(self):
        a = _ds(affordances=["spike_sorting", "stimulus_response_modeling"])
        b = _ds(affordances=["calcium_imaging", "dimensionality_reduction"])
        score = complementarity_score(a, b)
        assert 0.0 <= score <= 1.0

    def test_identical_affordances_score_lower_than_different(self):
        a = _ds(affordances=["choice_decoding"])
        score_identical = complementarity_score(a, a)
        score_complement = complementarity_score(a, _ds(affordances=["q_learning"]))
        assert score_complement >= score_identical

    def test_both_empty_returns_zero(self):
        assert complementarity_score(_ds(), _ds()) == pytest.approx(0.0)

    def test_score_bounded(self):
        a = _ds(affordances=["a", "b"])
        b = _ds(affordances=["b", "c", "d"])
        assert 0.0 <= complementarity_score(a, b) <= 1.0


class TestNormalizedMetapathScore:
    def _build_hub_graph(self):
        nodes = {f"ds_{i}": {"node_id": f"ds_{i}", "node_type": "dataset", "label": f"DS{i}"} for i in range(12)}
        nodes["hub"] = {"node_id": "hub", "node_type": "task", "label": "Hub Task"}
        nodes["rare"] = {"node_id": "rare", "node_type": "task", "label": "Rare Task"}
        edges = {}
        for i in range(10):
            eid = f"e_hub_{i}"
            edges[eid] = {
                "edge_id": eid,
                "source_node_id": f"ds_{i}",
                "target_node_id": "hub",
                "edge_type": "dataset_has_task",
                "confidence": 1.0,
            }
        for i in range(2):
            eid = f"e_rare_{i}"
            edges[eid] = {
                "edge_id": eid,
                "source_node_id": f"ds_{i}",
                "target_node_id": "rare",
                "edge_type": "dataset_has_task",
                "confidence": 1.0,
            }
        return _graph(nodes=nodes, edges=edges)

    def test_hub_normalization(self):
        graph = self._build_hub_graph()
        # ds_0 and ds_1 share BOTH hub and rare; ds_0 and ds_5 share only hub
        score_shared_rare = normalized_metapath_score(graph, "ds_0", "ds_1", "dataset_has_task")
        score_hub_only = normalized_metapath_score(graph, "ds_0", "ds_5", "dataset_has_task")
        assert score_shared_rare > score_hub_only

    def test_missing_nodes_returns_zero(self):
        graph = _graph()
        score = normalized_metapath_score(graph, "nonexistent_a", "nonexistent_b", "dataset_has_task")
        assert score == pytest.approx(0.0)

    def test_score_bounded(self):
        graph = self._build_hub_graph()
        score = normalized_metapath_score(graph, "ds_0", "ds_1", "dataset_has_task")
        assert 0.0 <= score <= 1.0

    def test_none_graph_returns_zero(self):
        assert normalized_metapath_score(None, "a", "b", "dataset_has_task") == pytest.approx(0.0)


class TestGraphUsefulnessFeatures:
    def test_returns_dict_with_expected_keys(self):
        graph = _graph()
        q = _ds(affordances=["choice_decoding"])
        c = _ds(affordances=["choice_decoding"])
        features = graph_usefulness_features(q, c, graph)
        assert "affordance_overlap" in features
        assert "pipeline_overlap" in features
        assert "complementarity" in features
        assert "metapath_score" in features

    def test_missing_graph_does_not_raise(self):
        features = graph_usefulness_features(_ds(), _ds(), None)
        assert isinstance(features, dict)

    def test_all_values_bounded(self):
        graph = _graph()
        q = _ds(affordances=["a", "b"], data_standards=["nwb"])
        c = _ds(affordances=["b", "c"], data_standards=["bids"])
        features = graph_usefulness_features(q, c, graph)
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
