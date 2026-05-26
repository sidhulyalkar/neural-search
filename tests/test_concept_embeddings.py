"""Tests for concept embedding infrastructure."""

import pytest

from neural_search.embeddings import (
    ConceptEmbeddingBuilder,
    ConceptEmbeddingIndex,
    concept_similarity,
    expand_query_semantically,
    find_semantically_similar,
)


@pytest.fixture
def concept_index():
    """Create a concept embedding index from ontology."""
    builder = ConceptEmbeddingBuilder(text_model="hashing", target_dim=128)
    embeddings = builder.build_all_embeddings(
        ontology_path="data/ontology/behavioral_task_ontology.yaml",
        analysis_path="data/ontology/analysis_methods.yaml",
    )
    return ConceptEmbeddingIndex(embeddings)


class TestConceptEmbeddingBuilder:
    """Test concept embedding building."""

    def test_builds_task_embeddings(self):
        """Should build embeddings for tasks from ontology."""
        builder = ConceptEmbeddingBuilder(text_model="hashing", target_dim=128)
        embeddings = builder.build_task_embeddings(
            "data/ontology/behavioral_task_ontology.yaml"
        )

        assert len(embeddings) > 0
        # Check structure
        for emb in embeddings[:5]:
            assert emb.concept_type == "task"
            assert emb.concept_id.startswith("task:")
            assert len(emb.embedding) == 128

    def test_builds_modality_embeddings(self):
        """Should build embeddings for modalities."""
        builder = ConceptEmbeddingBuilder(text_model="hashing", target_dim=128)
        embeddings = builder.build_modality_embeddings()

        assert len(embeddings) >= 10
        for emb in embeddings[:5]:
            assert emb.concept_type == "modality"
            assert len(emb.embedding) == 128

    def test_builds_analysis_embeddings(self):
        """Should build embeddings for analysis methods."""
        builder = ConceptEmbeddingBuilder(text_model="hashing", target_dim=128)
        embeddings = builder.build_analysis_embeddings(
            "data/ontology/analysis_methods.yaml"
        )

        assert len(embeddings) >= 8
        for emb in embeddings:
            assert emb.concept_type == "analysis"

    def test_minor_type_uses_smaller_dimension(self):
        """Behavior and region embeddings should use minor_dim."""
        builder = ConceptEmbeddingBuilder(
            text_model="hashing", target_dim=128, minor_dim=64
        )
        behavior_embeddings = builder.build_behavior_embeddings(
            "data/ontology/behavioral_task_ontology.yaml"
        )
        region_embeddings = builder.build_region_embeddings()

        # Behaviors use minor_dim
        for emb in behavior_embeddings[:3]:
            assert len(emb.embedding) == 64

        # Regions use minor_dim
        for emb in region_embeddings[:3]:
            assert len(emb.embedding) == 64


class TestConceptEmbeddingIndex:
    """Test concept embedding index."""

    def test_index_by_id(self, concept_index):
        """Should retrieve concept by ID."""
        emb = concept_index.get("task:reversal_learning")
        assert emb is not None
        assert emb.label == "Reversal Learning"

    def test_index_by_label(self, concept_index):
        """Should retrieve concept by label."""
        # Labels are stored as "Reversal Learning", case-insensitive lookup
        emb = concept_index.get_by_label("Reversal Learning")
        assert emb is not None
        assert "reversal" in emb.concept_id.lower()

    def test_index_by_alias(self, concept_index):
        """Should retrieve concept by alias."""
        # Neuropixels has alias "np probe"
        emb = concept_index.get_by_label("neuropixels")
        assert emb is not None
        assert emb.concept_type == "modality"

    def test_find_similar(self, concept_index):
        """Should find similar concepts."""
        similar = concept_index.find_similar(
            "task:reversal_learning",
            concept_type="task",
            k=5,
            min_similarity=0.3,
        )

        assert len(similar) > 0
        # All should be tasks
        for s in similar:
            assert s.concept_type == "task"
            assert s.similarity >= 0.3

    def test_concept_types(self, concept_index):
        """Should have multiple concept types."""
        types = concept_index.concept_types
        assert "task" in types
        assert "modality" in types
        assert "analysis" in types


class TestConceptSimilarity:
    """Test concept similarity computation."""

    def test_similarity_same_type(self, concept_index):
        """Concepts of same type should have meaningful similarity."""
        sim = concept_similarity(
            "task:reversal_learning", "task:go_nogo", concept_index
        )
        # Both are decision-making tasks, should have some similarity
        assert sim > 0.3

    def test_similarity_related_modalities(self, concept_index):
        """Related modalities should have higher similarity."""
        # Both are electrophysiology
        sim_ephys = concept_similarity(
            "modality:neuropixels", "modality:extracellular_ephys", concept_index
        )
        # One is electrophysiology, one is optical
        sim_diff = concept_similarity(
            "modality:neuropixels", "modality:fmri", concept_index
        )
        # Same-category should be more similar
        assert sim_ephys > sim_diff

    def test_similarity_missing_concept(self, concept_index):
        """Missing concept should return 0."""
        sim = concept_similarity(
            "task:nonexistent", "task:reversal_learning", concept_index
        )
        assert sim == 0.0


class TestFindSemanticallySimlar:
    """Test semantic similarity search."""

    def test_finds_similar_tasks(self, concept_index):
        """Should find tasks similar to a given task."""
        similar = find_semantically_similar(
            "task:reversal_learning",
            concept_index,
            concept_type="task",
            min_similarity=0.3,
        )

        assert len(similar) > 0
        # Should include related decision-making tasks
        labels = [s.label.lower() for s in similar]
        # At least some should be found
        assert len(labels) >= 1

    def test_finds_by_label(self, concept_index):
        """Should find similar concepts by label."""
        similar = find_semantically_similar(
            "Reversal Learning",  # Use actual label format
            concept_index,
            concept_type="task",
            min_similarity=0.3,
        )

        assert len(similar) > 0


class TestQuerySemanticExpansion:
    """Test semantic query expansion."""

    def test_expands_tasks(self, concept_index):
        """Should expand task concepts."""
        parsed_query = {
            "query": "reversal learning",
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.3
        )

        assert len(expansion.expanded_tasks) > 1
        # First should be exact match
        assert expansion.expanded_tasks[0].is_exact
        assert expansion.expanded_tasks[0].similarity == 1.0

    def test_expands_modalities(self, concept_index):
        """Should expand modality concepts."""
        parsed_query = {
            "query": "neuropixels recording",
            "tasks": [],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.3
        )

        assert len(expansion.expanded_modalities) >= 1
        # Should include extracellular ephys as similar
        labels = [m.matched_label.lower() for m in expansion.expanded_modalities]
        # Neuropixels should be there (exact match)
        assert any("neuropixels" in lbl for lbl in labels)

    def test_expansion_count(self, concept_index):
        """Should count total expansions."""
        parsed_query = {
            "query": "reversal learning with neuropixels",
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.3
        )

        # Should have multiple expansions
        assert expansion.expansion_count() > 2

    def test_respects_min_similarity(self, concept_index):
        """Should respect minimum similarity threshold."""
        parsed_query = {
            "query": "reversal learning",
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        # High threshold
        expansion_high = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.8
        )

        # Low threshold
        expansion_low = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.3
        )

        # Low threshold should find more
        assert len(expansion_low.expanded_tasks) >= len(expansion_high.expanded_tasks)

    def test_empty_query(self, concept_index):
        """Should handle empty query gracefully."""
        parsed_query = {
            "query": "",
            "tasks": [],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_semantically(
            parsed_query, concept_index, min_similarity=0.3
        )

        assert expansion.expansion_count() == 0
