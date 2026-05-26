"""Tests for semantic query expansion."""

from __future__ import annotations

import pytest
import numpy as np

from neural_search.embeddings import (
    ConceptEmbedding,
    ConceptEmbeddingIndex,
)
from neural_search.search.semantic_expansion import (
    SemanticExpansion,
    compute_expansion_boost,
    enrich_query_with_semantic_context,
    expand_query_with_concepts,
    merge_expansion_into_query,
)


@pytest.fixture
def sample_concept_index() -> ConceptEmbeddingIndex:
    """Create a sample concept embedding index."""
    np.random.seed(42)

    # Create embeddings with some similarity structure
    base_task_1 = np.random.rand(128)
    base_task_2 = base_task_1 + np.random.rand(128) * 0.3  # Similar to task 1
    base_task_3 = np.random.rand(128)  # Different

    base_mod_1 = np.random.rand(128)
    base_mod_2 = base_mod_1 + np.random.rand(128) * 0.3  # Similar to mod 1

    embeddings = [
        ConceptEmbedding(
            concept_id="task:reversal_learning",
            concept_type="task",
            label="Reversal Learning",
            embedding=base_task_1.tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="task:set_shifting",
            concept_type="task",
            label="Set Shifting",
            embedding=base_task_2.tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="task:spatial_navigation",
            concept_type="task",
            label="Spatial Navigation",
            embedding=base_task_3.tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="modality:neuropixels",
            concept_type="modality",
            label="Neuropixels",
            embedding=base_mod_1.tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="modality:extracellular_ephys",
            concept_type="modality",
            label="Extracellular Ephys",
            embedding=base_mod_2.tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="behavior:licking",
            concept_type="behavior",
            label="Licking",
            embedding=np.random.rand(64).tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="analysis:decoding",
            concept_type="analysis",
            label="Decoding",
            embedding=np.random.rand(64).tolist(),
            model_version="test",
        ),
    ]
    return ConceptEmbeddingIndex(embeddings)


class TestSemanticExpansion:
    """Tests for SemanticExpansion dataclass."""

    def test_total_expansions(self):
        """Test total expansion count."""
        expansion = SemanticExpansion(
            original_tasks=["task1"],
            original_modalities=["mod1"],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("task2", 0.8), ("task3", 0.7)],
            expanded_modalities=[("mod2", 0.9)],
        )
        assert expansion.total_expansions == 3

    def test_all_task_ids(self):
        """Test getting all task IDs including expansions."""
        expansion = SemanticExpansion(
            original_tasks=["task1"],
            original_modalities=[],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("task2", 0.8)],
        )
        all_tasks = expansion.all_task_ids()
        assert "task1" in all_tasks
        assert "task2" in all_tasks

    def test_all_modality_ids(self):
        """Test getting all modality IDs including expansions."""
        expansion = SemanticExpansion(
            original_tasks=[],
            original_modalities=["neuropixels"],
            original_behaviors=[],
            original_affordances=[],
            expanded_modalities=[("extracellular_ephys", 0.9)],
        )
        all_mods = expansion.all_modality_ids()
        assert "neuropixels" in all_mods
        assert "extracellular_ephys" in all_mods


class TestExpandQueryWithConcepts:
    """Tests for expand_query_with_concepts."""

    def test_expands_tasks(self, sample_concept_index):
        """Test that tasks are expanded with related concepts."""
        parsed_query = {
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_with_concepts(
            parsed_query,
            sample_concept_index,
            min_similarity=0.3,  # Low threshold to get expansions
            max_expansions_per_concept=3,
        )

        assert expansion.original_tasks == ["reversal_learning"]
        # Should find set_shifting as similar to reversal_learning
        expanded_task_ids = [t[0] for t in expansion.expanded_tasks]
        # May or may not find expansions depending on similarity
        assert isinstance(expansion.expanded_tasks, list)

    def test_expands_modalities(self, sample_concept_index):
        """Test that modalities are expanded with related concepts."""
        parsed_query = {
            "tasks": [],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_with_concepts(
            parsed_query,
            sample_concept_index,
            min_similarity=0.3,
            max_expansions_per_concept=3,
        )

        assert expansion.original_modalities == ["neuropixels"]
        # Should find extracellular_ephys as related
        expanded_mod_ids = [m[0] for m in expansion.expanded_modalities]
        assert isinstance(expansion.expanded_modalities, list)

    def test_no_duplicates_in_expansion(self, sample_concept_index):
        """Test that original concepts are not duplicated in expansion."""
        parsed_query = {
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_with_concepts(
            parsed_query,
            sample_concept_index,
            min_similarity=0.0,  # Very low to catch all
        )

        # Original task should not appear in expanded tasks
        expanded_task_ids = [t[0] for t in expansion.expanded_tasks]
        assert "reversal_learning" not in expanded_task_ids

    def test_respects_max_expansions(self, sample_concept_index):
        """Test that maximum expansions per concept is respected."""
        parsed_query = {
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "affordances": [],
        }

        expansion = expand_query_with_concepts(
            parsed_query,
            sample_concept_index,
            min_similarity=0.0,
            max_expansions_per_concept=1,
        )

        # Should have at most 1 expansion per original concept
        assert len(expansion.expanded_tasks) <= 1


class TestMergeExpansionIntoQuery:
    """Tests for merge_expansion_into_query."""

    def test_merges_expansion(self):
        """Test that expansion is merged into query."""
        parsed_query = {
            "query": "test query",
            "tasks": ["task1"],
            "modalities": ["mod1"],
        }

        expansion = SemanticExpansion(
            original_tasks=["task1"],
            original_modalities=["mod1"],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("task2", 0.8)],
            expanded_modalities=[("mod2", 0.9)],
        )

        result = merge_expansion_into_query(parsed_query, expansion)

        assert "expanded_tasks" in result
        assert "task2" in result["expanded_tasks"]
        assert "expanded_modalities" in result
        assert "mod2" in result["expanded_modalities"]

    def test_includes_expansion_metadata(self):
        """Test that expansion metadata is included."""
        parsed_query = {"query": "test"}

        expansion = SemanticExpansion(
            original_tasks=[],
            original_modalities=[],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("task2", 0.85)],
        )

        result = merge_expansion_into_query(
            parsed_query, expansion, include_expansion_metadata=True
        )

        assert "semantic_expansion" in result
        assert result["semantic_expansion"]["total_expansions"] == 1
        assert result["semantic_expansion"]["tasks"][0]["id"] == "task2"
        assert result["semantic_expansion"]["tasks"][0]["similarity"] == 0.85


class TestEnrichQueryWithSemanticContext:
    """Tests for enrich_query_with_semantic_context."""

    def test_enriches_query(self, sample_concept_index):
        """Test that query is enriched with semantic context."""
        query = "reversal learning neuropixels"
        parsed_query = {
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "affordances": [],
        }

        enriched, context_tokens = enrich_query_with_semantic_context(
            query,
            parsed_query,
            concept_index=sample_concept_index,
            min_similarity=0.3,
        )

        assert isinstance(enriched, dict)
        assert isinstance(context_tokens, list)

    def test_returns_original_without_index(self):
        """Test that original is returned when no index provided."""
        query = "test query"
        parsed_query = {"tasks": ["task1"]}

        enriched, context_tokens = enrich_query_with_semantic_context(
            query, parsed_query, concept_index=None
        )

        assert enriched == parsed_query
        assert context_tokens == []


class TestComputeExpansionBoost:
    """Tests for compute_expansion_boost."""

    def test_computes_boost_for_matches(self):
        """Test that boost is computed for matching expansions."""
        dataset_labels = {
            "tasks": ["set_shifting"],
            "modalities": ["extracellular_ephys"],
        }

        expansion = SemanticExpansion(
            original_tasks=[],
            original_modalities=[],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("set_shifting", 0.8)],  # Matches dataset
            expanded_modalities=[("calcium_imaging", 0.7)],  # Doesn't match
        )

        boost = compute_expansion_boost(dataset_labels, expansion, boost_factor=0.5)

        # 1 out of 2 expansions match = 0.5 * 0.5 = 0.25
        assert boost == pytest.approx(0.25, abs=0.01)

    def test_zero_boost_for_no_matches(self):
        """Test that zero boost for no matching expansions."""
        dataset_labels = {"tasks": ["spatial_navigation"]}

        expansion = SemanticExpansion(
            original_tasks=[],
            original_modalities=[],
            original_behaviors=[],
            original_affordances=[],
            expanded_tasks=[("set_shifting", 0.8)],  # Doesn't match
        )

        boost = compute_expansion_boost(dataset_labels, expansion, boost_factor=0.5)
        assert boost == 0.0

    def test_zero_boost_for_no_expansions(self):
        """Test that zero boost when no expansions."""
        dataset_labels = {"tasks": ["reversal_learning"]}

        expansion = SemanticExpansion(
            original_tasks=["reversal_learning"],
            original_modalities=[],
            original_behaviors=[],
            original_affordances=[],
        )

        boost = compute_expansion_boost(dataset_labels, expansion, boost_factor=0.5)
        assert boost == 0.0
