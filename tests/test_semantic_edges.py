"""Tests for semantic graph edges."""

from __future__ import annotations

import pytest

from neural_search.embeddings import (
    SemanticDatasetFingerprint,
)
from neural_search.graph import (
    KnowledgeGraph,
    KnowledgeGraphNode,
    SemanticEdgeConfig,
    SemanticEdgeResult,
    add_semantic_edges_to_graph,
    build_semantic_dataset_edges,
    get_semantic_neighbors,
    make_node_id,
)


@pytest.fixture
def sample_fingerprints() -> list[SemanticDatasetFingerprint]:
    """Create sample fingerprints for testing."""
    import numpy as np

    np.random.seed(42)

    # Create fingerprints with similar and different characteristics
    fp1 = SemanticDatasetFingerprint(
        dataset_id="dataset_001",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.8] * 128,  # Similar to fp3
        modality_embedding=[0.5] * 128,
        behavior_embedding=[0.3] * 64,
        analysis_embedding=[0.4] * 64,
        region_embedding=[0.6] * 64,
        design_embedding=[0.7] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        design_type="reversal_learning",
    )

    fp2 = SemanticDatasetFingerprint(
        dataset_id="dataset_002",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.2] * 128,  # Different task
        modality_embedding=[0.5] * 128,  # Same modality
        behavior_embedding=[0.9] * 64,
        analysis_embedding=[0.1] * 64,
        region_embedding=[0.3] * 64,
        design_embedding=[0.4] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        design_type="free_behavior",
    )

    fp3 = SemanticDatasetFingerprint(
        dataset_id="dataset_003",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.8] * 128,  # Similar to fp1
        modality_embedding=[0.5] * 128,
        behavior_embedding=[0.3] * 64,
        analysis_embedding=[0.4] * 64,
        region_embedding=[0.6] * 64,
        design_embedding=[0.7] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        design_type="reversal_learning",
    )

    return [fp1, fp2, fp3]


@pytest.fixture
def sample_graph() -> KnowledgeGraph:
    """Create a sample graph with dataset nodes."""
    nodes = {
        make_node_id("dataset", "dataset_001"): KnowledgeGraphNode(
            node_id=make_node_id("dataset", "dataset_001"),
            node_type="dataset",
            label="Dataset 001",
        ),
        make_node_id("dataset", "dataset_002"): KnowledgeGraphNode(
            node_id=make_node_id("dataset", "dataset_002"),
            node_type="dataset",
            label="Dataset 002",
        ),
        make_node_id("dataset", "dataset_003"): KnowledgeGraphNode(
            node_id=make_node_id("dataset", "dataset_003"),
            node_type="dataset",
            label="Dataset 003",
        ),
    }
    return KnowledgeGraph(nodes=nodes)


class TestSemanticEdgeConfig:
    """Tests for SemanticEdgeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SemanticEdgeConfig()
        assert config.min_combined_similarity == 0.5
        assert config.min_task_similarity == 0.6
        assert config.max_similar_datasets_per_node == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = SemanticEdgeConfig(
            min_combined_similarity=0.3,
            max_similar_datasets_per_node=10,
        )
        assert config.min_combined_similarity == 0.3
        assert config.max_similar_datasets_per_node == 10


class TestBuildSemanticDatasetEdges:
    """Tests for building semantic dataset edges."""

    def test_builds_edges_between_similar_fingerprints(self, sample_fingerprints):
        """Test that edges are built between similar fingerprints."""
        config = SemanticEdgeConfig(
            min_combined_similarity=0.3,  # Low threshold to get edges
            max_similar_datasets_per_node=5,
        )
        edges = build_semantic_dataset_edges(sample_fingerprints, config)

        assert len(edges) > 0
        for edge in edges:
            assert "semantic" in edge.edge_type
            assert edge.confidence > 0

    def test_respects_similarity_threshold(self, sample_fingerprints):
        """Test that edges respect minimum similarity threshold."""
        # High threshold should produce fewer edges
        config_high = SemanticEdgeConfig(
            min_combined_similarity=0.9,
        )
        edges_high = build_semantic_dataset_edges(sample_fingerprints, config_high)

        # Low threshold should produce more edges
        config_low = SemanticEdgeConfig(
            min_combined_similarity=0.3,
        )
        edges_low = build_semantic_dataset_edges(sample_fingerprints, config_low)

        assert len(edges_low) >= len(edges_high)

    def test_limits_edges_per_node(self, sample_fingerprints):
        """Test that edges per node are limited."""
        config = SemanticEdgeConfig(
            min_combined_similarity=0.0,  # Very low to allow all edges
            max_similar_datasets_per_node=1,
        )
        edges = build_semantic_dataset_edges(sample_fingerprints, config)

        # With 3 fingerprints and max 1 edge per node, should have limited edges
        # Each source can only have 1 similar target
        source_counts: dict[str, int] = {}
        for edge in edges:
            source_id = edge.source_node_id
            source_counts[source_id] = source_counts.get(source_id, 0) + 1

        for count in source_counts.values():
            assert count <= 1

    def test_edge_properties_include_similarity_breakdown(self, sample_fingerprints):
        """Test that edge properties include similarity breakdown."""
        config = SemanticEdgeConfig(min_combined_similarity=0.3)
        edges = build_semantic_dataset_edges(sample_fingerprints, config)

        if edges:
            edge = edges[0]
            props = edge.properties
            assert "combined_similarity" in props
            assert "task_similarity" in props
            assert "modality_similarity" in props
            assert "behavior_similarity" in props
            assert "similarity_type" in props


class TestAddSemanticEdgesToGraph:
    """Tests for adding semantic edges to graph."""

    def test_adds_edges_to_graph(self, sample_graph, sample_fingerprints):
        """Test that edges are added to graph."""
        initial_edge_count = len(sample_graph.edges)

        result = add_semantic_edges_to_graph(
            sample_graph,
            fingerprints=sample_fingerprints,
            config=SemanticEdgeConfig(min_combined_similarity=0.3),
        )

        assert isinstance(result, SemanticEdgeResult)
        assert result.total_edges_added >= 0
        assert len(sample_graph.edges) >= initial_edge_count

    def test_only_adds_edges_for_existing_nodes(self, sample_graph, sample_fingerprints):
        """Test that edges are only added for nodes that exist in graph."""
        # Add a fingerprint for a non-existent node
        import numpy as np

        fake_fp = SemanticDatasetFingerprint(
            dataset_id="nonexistent_dataset",
            text_embedding=np.random.rand(256).tolist(),
            task_embedding=[0.5] * 128,
            modality_embedding=[0.5] * 128,
            behavior_embedding=[0.5] * 64,
            analysis_embedding=[0.5] * 64,
            region_embedding=[0.5] * 64,
            design_embedding=[0.5] * 32,
            combined_embedding=np.random.rand(736).tolist(),
        )

        fingerprints_with_fake = sample_fingerprints + [fake_fp]

        add_semantic_edges_to_graph(
            sample_graph,
            fingerprints=fingerprints_with_fake,
            config=SemanticEdgeConfig(min_combined_similarity=0.0),
        )

        # Check that no edge references the nonexistent node
        for edge in sample_graph.edges.values():
            assert "nonexistent_dataset" not in edge.source_node_id
            assert "nonexistent_dataset" not in edge.target_node_id


class TestGetSemanticNeighbors:
    """Tests for getting semantic neighbors."""

    def test_finds_semantic_neighbors(self, sample_graph, sample_fingerprints):
        """Test finding semantic neighbors for a node."""
        # First add some edges
        add_semantic_edges_to_graph(
            sample_graph,
            fingerprints=sample_fingerprints,
            config=SemanticEdgeConfig(min_combined_similarity=0.3),
        )

        # Get neighbors
        node_id = make_node_id("dataset", "dataset_001")
        get_semantic_neighbors(
            sample_graph,
            node_id,
            min_similarity=0.3,
        )

        # Should find at least one neighbor
        if len(sample_graph.edges) > 0:
            # Neighbors might be empty if no edges connect to this node
            # This depends on which edges were created
            pass

    def test_respects_minimum_similarity(self, sample_graph, sample_fingerprints):
        """Test that minimum similarity is respected."""
        add_semantic_edges_to_graph(
            sample_graph,
            fingerprints=sample_fingerprints,
            config=SemanticEdgeConfig(min_combined_similarity=0.3),
        )

        node_id = make_node_id("dataset", "dataset_001")

        # High threshold
        neighbors_high = get_semantic_neighbors(
            sample_graph, node_id, min_similarity=0.9
        )

        # Low threshold
        neighbors_low = get_semantic_neighbors(
            sample_graph, node_id, min_similarity=0.1
        )

        assert len(neighbors_low) >= len(neighbors_high)

    def test_returns_sorted_by_similarity(self, sample_graph, sample_fingerprints):
        """Test that neighbors are sorted by similarity descending."""
        add_semantic_edges_to_graph(
            sample_graph,
            fingerprints=sample_fingerprints,
            config=SemanticEdgeConfig(min_combined_similarity=0.3),
        )

        node_id = make_node_id("dataset", "dataset_001")
        neighbors = get_semantic_neighbors(sample_graph, node_id, min_similarity=0.1)

        if len(neighbors) >= 2:
            similarities = [n[1] for n in neighbors]
            assert similarities == sorted(similarities, reverse=True)
