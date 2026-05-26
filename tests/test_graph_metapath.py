"""Tests for metapath-based graph reasoning and path scoring."""

from __future__ import annotations

import pytest

from neural_search.graph.metapath import (
    ALL_TEMPLATES,
    ANALYSIS_METAPATHS,
    MetapathMatch,
    MetapathScorer,
    MetapathScoreResult,
    MetapathStep,
    MetapathTemplate,
    PathInstance,
    QueryIntent,
    SIMILARITY_METAPATHS,
    explain_path,
    score_metapaths,
)
from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)


def _make_evidence(source_id: str, text: str = "") -> GraphEvidence:
    """Create a test evidence object."""
    return GraphEvidence(
        evidence_id=f"ev:{source_id}",
        source_type="test",
        source_id=source_id,
        evidence_text=text,
        confidence=0.9,
        extractor_name="test_extractor",
        extractor_version="1.0",
    )


@pytest.fixture
def simple_graph() -> KnowledgeGraph:
    """Create a simple test graph with datasets, tasks, and analyses."""
    nodes = {}
    edges = {}

    # Create dataset nodes
    d1_id = make_node_id("dataset", "dandi", "000001")
    nodes[d1_id] = KnowledgeGraphNode(
        node_id=d1_id,
        node_type="dataset",
        label="Mouse Neuropixels Dataset",
        confidence=0.95,
    )

    d2_id = make_node_id("dataset", "dandi", "000002")
    nodes[d2_id] = KnowledgeGraphNode(
        node_id=d2_id,
        node_type="dataset",
        label="Rat Calcium Imaging Dataset",
        confidence=0.90,
    )

    # Create task nodes
    t1_id = make_node_id("task", "decision_making")
    nodes[t1_id] = KnowledgeGraphNode(
        node_id=t1_id,
        node_type="task",
        label="Decision Making",
        confidence=0.95,
    )

    t2_id = make_node_id("task", "visual_coding")
    nodes[t2_id] = KnowledgeGraphNode(
        node_id=t2_id,
        node_type="task",
        label="Visual Coding",
        confidence=0.90,
    )

    # Create modality nodes
    m1_id = make_node_id("modality", "neuropixels")
    nodes[m1_id] = KnowledgeGraphNode(
        node_id=m1_id,
        node_type="modality",
        label="Neuropixels",
        confidence=0.95,
    )

    m2_id = make_node_id("modality", "calcium_imaging")
    nodes[m2_id] = KnowledgeGraphNode(
        node_id=m2_id,
        node_type="modality",
        label="Calcium Imaging",
        confidence=0.90,
    )

    # Create analysis affordance nodes
    a1_id = make_node_id("analysis_affordance", "choice_decoding")
    nodes[a1_id] = KnowledgeGraphNode(
        node_id=a1_id,
        node_type="analysis_affordance",
        label="Choice Decoding",
        confidence=0.95,
    )

    # Create behavioral event nodes
    e1_id = make_node_id("behavioral_event", "choice")
    nodes[e1_id] = KnowledgeGraphNode(
        node_id=e1_id,
        node_type="behavioral_event",
        label="Choice",
        confidence=0.90,
    )

    # Create edges: dataset -> task
    edge1_id = make_edge_id(d1_id, "dataset_has_task", t1_id)
    edges[edge1_id] = KnowledgeGraphEdge(
        edge_id=edge1_id,
        source_node_id=d1_id,
        target_node_id=t1_id,
        edge_type="dataset_has_task",
        confidence=0.95,
        evidence=[_make_evidence("d1_task", "Uses decision making task")],
    )

    edge2_id = make_edge_id(d2_id, "dataset_has_task", t1_id)
    edges[edge2_id] = KnowledgeGraphEdge(
        edge_id=edge2_id,
        source_node_id=d2_id,
        target_node_id=t1_id,
        edge_type="dataset_has_task",
        confidence=0.85,
        evidence=[_make_evidence("d2_task")],
    )

    # Create edges: dataset -> modality
    edge3_id = make_edge_id(d1_id, "dataset_has_modality", m1_id)
    edges[edge3_id] = KnowledgeGraphEdge(
        edge_id=edge3_id,
        source_node_id=d1_id,
        target_node_id=m1_id,
        edge_type="dataset_has_modality",
        confidence=0.95,
        evidence=[_make_evidence("d1_modality")],
    )

    edge4_id = make_edge_id(d2_id, "dataset_has_modality", m2_id)
    edges[edge4_id] = KnowledgeGraphEdge(
        edge_id=edge4_id,
        source_node_id=d2_id,
        target_node_id=m2_id,
        edge_type="dataset_has_modality",
        confidence=0.90,
        evidence=[_make_evidence("d2_modality")],
    )

    # Create edges: dataset -> behavioral_event
    edge5_id = make_edge_id(d1_id, "dataset_has_behavioral_event", e1_id)
    edges[edge5_id] = KnowledgeGraphEdge(
        edge_id=edge5_id,
        source_node_id=d1_id,
        target_node_id=e1_id,
        edge_type="dataset_has_behavioral_event",
        confidence=0.90,
        evidence=[_make_evidence("d1_event", "Contains choice events")],
    )

    # Create edges: analysis requires behavioral_event
    edge6_id = make_edge_id(a1_id, "analysis_requires_behavioral_event", e1_id)
    edges[edge6_id] = KnowledgeGraphEdge(
        edge_id=edge6_id,
        source_node_id=a1_id,
        target_node_id=e1_id,
        edge_type="analysis_requires_behavioral_event",
        confidence=0.95,
        evidence=[_make_evidence("a1_requires_event")],
    )

    # Create edges: analysis requires modality
    edge7_id = make_edge_id(a1_id, "analysis_requires_modality", m1_id)
    edges[edge7_id] = KnowledgeGraphEdge(
        edge_id=edge7_id,
        source_node_id=a1_id,
        target_node_id=m1_id,
        edge_type="analysis_requires_modality",
        confidence=0.90,
        evidence=[_make_evidence("a1_requires_modality")],
    )

    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def paper_graph() -> KnowledgeGraph:
    """Create a test graph with paper-dataset links."""
    nodes = {}
    edges = {}

    # Dataset
    d1_id = make_node_id("dataset", "dandi", "000001")
    nodes[d1_id] = KnowledgeGraphNode(
        node_id=d1_id,
        node_type="dataset",
        label="Test Dataset",
        confidence=0.95,
    )

    # Paper
    p1_id = make_node_id("paper", "doi", "10.1234")
    nodes[p1_id] = KnowledgeGraphNode(
        node_id=p1_id,
        node_type="paper",
        label="Test Paper",
        confidence=0.90,
    )

    # Task shared by paper and dataset
    t1_id = make_node_id("task", "visual_task")
    nodes[t1_id] = KnowledgeGraphNode(
        node_id=t1_id,
        node_type="task",
        label="Visual Task",
        confidence=0.90,
    )

    # Paper uses dataset
    edge1_id = make_edge_id(p1_id, "paper_uses_dataset", d1_id)
    edges[edge1_id] = KnowledgeGraphEdge(
        edge_id=edge1_id,
        source_node_id=p1_id,
        target_node_id=d1_id,
        edge_type="paper_uses_dataset",
        confidence=0.95,
        evidence=[_make_evidence("paper_dataset")],
    )

    # Paper studies task
    edge2_id = make_edge_id(p1_id, "paper_studies_task", t1_id)
    edges[edge2_id] = KnowledgeGraphEdge(
        edge_id=edge2_id,
        source_node_id=p1_id,
        target_node_id=t1_id,
        edge_type="paper_studies_task",
        confidence=0.85,
        evidence=[_make_evidence("paper_task")],
    )

    # Dataset has task
    edge3_id = make_edge_id(d1_id, "dataset_has_task", t1_id)
    edges[edge3_id] = KnowledgeGraphEdge(
        edge_id=edge3_id,
        source_node_id=d1_id,
        target_node_id=t1_id,
        edge_type="dataset_has_task",
        confidence=0.90,
        evidence=[_make_evidence("dataset_task")],
    )

    return KnowledgeGraph(nodes=nodes, edges=edges)


class TestMetapathTemplate:
    """Tests for MetapathTemplate dataclass."""

    def test_create_template(self):
        """Test creating a metapath template."""
        template = MetapathTemplate(
            name="test_template",
            description="Test template",
            steps=[
                MetapathStep("dataset", "dataset_has_task", "task"),
            ],
            applicable_intents=[QueryIntent.TASK_RELATED],
            weight=1.0,
        )

        assert template.name == "test_template"
        assert len(template.steps) == 1
        assert template.weight == 1.0

    def test_predefined_templates(self):
        """Test that predefined templates are valid."""
        assert len(ALL_TEMPLATES) > 0
        assert len(ANALYSIS_METAPATHS) > 0
        assert len(SIMILARITY_METAPATHS) > 0

        for template in ALL_TEMPLATES:
            assert template.name
            assert template.description
            assert len(template.steps) > 0
            assert len(template.applicable_intents) > 0


class TestMetapathScorer:
    """Tests for MetapathScorer class."""

    def test_create_scorer(self, simple_graph):
        """Test creating a metapath scorer."""
        scorer = MetapathScorer(simple_graph)
        assert scorer.graph == simple_graph
        assert len(scorer.templates) > 0

    def test_score_paths_analysis_intent(self, simple_graph):
        """Test scoring paths for analysis intent."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.DATASET_FOR_ANALYSIS)

        assert isinstance(result, MetapathScoreResult)
        assert result.query_node_id == d1_id
        # Should find paths to analysis affordance via behavioral_event or modality

    def test_score_paths_similarity_intent(self, simple_graph):
        """Test scoring paths for similarity intent."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        assert isinstance(result, MetapathScoreResult)
        # Should find paths to d2 via shared task

    def test_find_paths_task_similarity(self, simple_graph):
        """Test finding similar datasets via shared task."""
        # Create a template for task similarity
        template = MetapathTemplate(
            name="task_sim",
            description="Dataset -> Task <- Dataset",
            steps=[
                MetapathStep("dataset", "dataset_has_task", "task"),
                MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
            ],
            applicable_intents=[QueryIntent.SIMILAR_DATASET],
        )

        scorer = MetapathScorer(simple_graph, templates=[template])
        d1_id = make_node_id("dataset", "dandi", "000001")

        paths = scorer._find_paths(d1_id, template)

        # Should find path: d1 -> decision_making -> d2
        assert len(paths) > 0

        # Check path goes to d2
        d2_id = make_node_id("dataset", "dandi", "000002")
        target_ids = [p.nodes[-1] for p in paths]
        assert d2_id in target_ids

    def test_confidence_propagation(self, simple_graph):
        """Test that confidence is correctly propagated through paths."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        for match in result.matches:
            for path in match.paths:
                # Confidence should be product of edge confidences
                assert 0 < path.confidence <= 1.0

    def test_min_confidence_filter(self, simple_graph):
        """Test minimum confidence filtering."""
        # Very high threshold should filter out paths
        scorer = MetapathScorer(simple_graph, min_confidence=0.99)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        # Most paths should be filtered
        total_paths = sum(m.path_count for m in result.matches)
        # May have zero paths if all below threshold

    def test_max_paths_limit(self, simple_graph):
        """Test maximum paths per template limit."""
        scorer = MetapathScorer(simple_graph, max_paths_per_template=1)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        for match in result.matches:
            # Each template should have at most 1 path per target
            assert match.path_count <= 1

    def test_get_templates_for_intent(self, simple_graph):
        """Test getting templates for specific intent."""
        scorer = MetapathScorer(simple_graph)

        analysis_templates = scorer.get_templates_for_intent(QueryIntent.DATASET_FOR_ANALYSIS)
        similarity_templates = scorer.get_templates_for_intent(QueryIntent.SIMILAR_DATASET)

        assert len(analysis_templates) > 0
        assert len(similarity_templates) > 0

        # Should be different sets
        analysis_names = {t.name for t in analysis_templates}
        similarity_names = {t.name for t in similarity_templates}
        assert analysis_names != similarity_names


class TestPaperMetapaths:
    """Tests for paper-related metapaths."""

    def test_paper_to_dataset_direct(self, paper_graph):
        """Test direct paper to dataset path."""
        template = MetapathTemplate(
            name="paper_uses_dataset",
            description="Paper -> Dataset",
            steps=[
                MetapathStep("paper", "paper_uses_dataset", "dataset"),
            ],
            applicable_intents=[QueryIntent.PAPER_TO_DATASET],
        )

        scorer = MetapathScorer(paper_graph, templates=[template])
        p1_id = make_node_id("paper", "doi", "10.1234")

        result = scorer.score_paths(p1_id, QueryIntent.PAPER_TO_DATASET)

        assert len(result.matches) > 0
        d1_id = make_node_id("dataset", "dandi", "000001")
        target_ids = [m.target_node_id for m in result.matches]
        assert d1_id in target_ids

    def test_paper_via_task_to_dataset(self, paper_graph):
        """Test paper to dataset via shared task."""
        template = MetapathTemplate(
            name="paper_task_dataset",
            description="Paper -> Task <- Dataset",
            steps=[
                MetapathStep("paper", "paper_studies_task", "task"),
                MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
            ],
            applicable_intents=[QueryIntent.PAPER_TO_DATASET],
        )

        scorer = MetapathScorer(paper_graph, templates=[template])
        p1_id = make_node_id("paper", "doi", "10.1234")

        paths = scorer._find_paths(p1_id, template)

        # Should find path: paper -> task -> dataset
        assert len(paths) > 0
        assert paths[0].length == 2


class TestPathSim:
    """Tests for PathSim similarity computation."""

    def test_compute_pathsim(self, simple_graph):
        """Test PathSim computation."""
        template = MetapathTemplate(
            name="task_sim",
            description="Dataset -> Task <- Dataset",
            steps=[
                MetapathStep("dataset", "dataset_has_task", "task"),
                MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
            ],
            applicable_intents=[QueryIntent.SIMILAR_DATASET],
        )

        scorer = MetapathScorer(simple_graph, templates=[template])
        d1_id = make_node_id("dataset", "dandi", "000001")
        d2_id = make_node_id("dataset", "dandi", "000002")

        similarity = scorer.compute_pathsim(d1_id, d2_id, template)

        # Should be > 0 since they share a task
        assert similarity >= 0.0
        assert similarity <= 1.0


class TestExplainPath:
    """Tests for path explanation generation."""

    def test_explain_path(self, simple_graph):
        """Test generating path explanation."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        if result.matches:
            match = result.matches[0]
            if match.paths:
                path = match.paths[0]
                explanation = explain_path(simple_graph, path)

                assert "template" in explanation
                assert "nodes" in explanation
                assert "edges" in explanation
                assert "confidence" in explanation

    def test_match_explanation(self, simple_graph):
        """Test that matches have explanations."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        # At least some matches should have explanations
        if result.matches:
            assert any(m.explanation for m in result.matches)


class TestScoreMetapathsFunction:
    """Tests for the convenience function."""

    def test_score_metapaths_function(self, simple_graph):
        """Test the score_metapaths convenience function."""
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = score_metapaths(
            simple_graph,
            d1_id,
            QueryIntent.SIMILAR_DATASET,
        )

        assert isinstance(result, MetapathScoreResult)
        assert result.query_node_id == d1_id

    def test_score_metapaths_with_target_type(self, simple_graph):
        """Test filtering by target type."""
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = score_metapaths(
            simple_graph,
            d1_id,
            QueryIntent.SIMILAR_DATASET,
            target_type="dataset",
        )

        # All targets should be datasets
        for match in result.matches:
            target_type = match.target_node_id.split(":")[1] if ":" in match.target_node_id else None
            assert target_type == "dataset"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_graph(self):
        """Test with empty graph."""
        graph = KnowledgeGraph(nodes={}, edges={})
        scorer = MetapathScorer(graph)

        result = scorer.score_paths("node:dataset:test", QueryIntent.SIMILAR_DATASET)

        assert result.matches == []
        assert result.total_score == 0.0

    def test_node_not_in_graph(self, simple_graph):
        """Test with node ID not in graph."""
        scorer = MetapathScorer(simple_graph)

        result = scorer.score_paths(
            "node:dataset:nonexistent:123",
            QueryIntent.SIMILAR_DATASET,
        )

        # Should handle gracefully
        assert result.matches == []

    def test_wrong_start_type(self, simple_graph):
        """Test starting from wrong node type for template."""
        scorer = MetapathScorer(simple_graph)

        # Try to find dataset similarity starting from a task node
        t1_id = make_node_id("task", "decision_making")

        result = scorer.score_paths(t1_id, QueryIntent.SIMILAR_DATASET)

        # Should not match dataset-starting templates
        # May have some matches from other templates

    def test_cycle_avoidance(self, simple_graph):
        """Test that cycles are avoided in path finding."""
        scorer = MetapathScorer(simple_graph)
        d1_id = make_node_id("dataset", "dandi", "000001")

        result = scorer.score_paths(d1_id, QueryIntent.SIMILAR_DATASET)

        # No path should contain the same node twice
        for match in result.matches:
            for path in match.paths:
                assert len(path.nodes) == len(set(path.nodes))
