"""Tests for semantic scoring in search."""

from __future__ import annotations

import numpy as np
import pytest

from neural_search.embeddings import (
    ConceptEmbedding,
    ConceptEmbeddingIndex,
    SemanticDatasetFingerprint,
)
from neural_search.search.semantic_scoring import (
    SemanticScoreResult,
    SemanticSearchIndex,
    _extract_label_core,
    compute_semantic_score_for_result,
    load_semantic_index,
)


@pytest.fixture
def sample_fingerprints() -> dict[str, SemanticDatasetFingerprint]:
    """Create sample fingerprints for testing."""
    np.random.seed(42)

    fp1 = SemanticDatasetFingerprint(
        dataset_id="dataset_001",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.8] * 128,
        modality_embedding=[0.5] * 128,
        behavior_embedding=[0.3] * 64,
        analysis_embedding=[0.4] * 64,
        region_embedding=[0.6] * 64,
        design_embedding=[0.7] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        task_labels=["reversal_learning", "decision_making"],
        modality_labels=["neuropixels", "extracellular_ephys"],
        behavior_labels=["licking", "choice"],
        analysis_affordance_ids=["decoding"],
        design_type="reversal_learning",
    )

    fp2 = SemanticDatasetFingerprint(
        dataset_id="dataset_002",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.2] * 128,
        modality_embedding=[0.5] * 128,
        behavior_embedding=[0.9] * 64,
        analysis_embedding=[0.1] * 64,
        region_embedding=[0.3] * 64,
        design_embedding=[0.4] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        task_labels=["spatial_navigation"],
        modality_labels=["calcium_imaging"],
        behavior_labels=["running", "position"],
        analysis_affordance_ids=["tuning_curve"],
        design_type="free_behavior",
    )

    fp3 = SemanticDatasetFingerprint(
        dataset_id="dataset_003",
        text_embedding=np.random.rand(256).tolist(),
        task_embedding=[0.8] * 128,
        modality_embedding=[0.5] * 128,
        behavior_embedding=[0.3] * 64,
        analysis_embedding=[0.4] * 64,
        region_embedding=[0.6] * 64,
        design_embedding=[0.7] * 32,
        combined_embedding=np.random.rand(736).tolist(),
        task_labels=["reversal_learning"],  # Same as fp1
        modality_labels=["neuropixels"],  # Same as fp1
        behavior_labels=["licking"],
        analysis_affordance_ids=["decoding"],
        design_type="reversal_learning",
    )

    return {
        "dataset_001": fp1,
        "dataset_002": fp2,
        "dataset_003": fp3,
    }


@pytest.fixture
def sample_concept_index() -> ConceptEmbeddingIndex:
    """Create a sample concept embedding index."""
    np.random.seed(42)

    embeddings = [
        ConceptEmbedding(
            concept_id="task:reversal_learning",
            concept_type="task",
            label="Reversal Learning",
            embedding=np.random.rand(128).tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="task:decision_making",
            concept_type="task",
            label="Decision Making",
            embedding=np.random.rand(128).tolist(),
            model_version="test",
        ),
        ConceptEmbedding(
            concept_id="modality:neuropixels",
            concept_type="modality",
            label="Neuropixels",
            embedding=np.random.rand(128).tolist(),
            model_version="test",
        ),
    ]
    return ConceptEmbeddingIndex(embeddings)


@pytest.fixture
def sample_search_index(
    sample_fingerprints, sample_concept_index
) -> SemanticSearchIndex:
    """Create a sample semantic search index."""
    return SemanticSearchIndex(
        fingerprints=sample_fingerprints,
        concept_index=sample_concept_index,
    )


class TestExtractLabelCore:
    """Tests for _extract_label_core helper."""

    def test_extracts_from_prefixed_label(self):
        """Test extracting core from prefixed labels."""
        label = "label:task:reversal_learning:neural_search.corpus"
        assert _extract_label_core(label) == "reversal_learning"

    def test_extracts_from_modality_label(self):
        """Test extracting core from modality labels."""
        label = "label:modality:neuropixels:neural_search.file_inspection"
        assert _extract_label_core(label) == "neuropixels"

    def test_handles_simple_label(self):
        """Test handling simple labels without prefix."""
        label = "reversal_learning"
        result = _extract_label_core(label)
        assert result == "reversal learning"


class TestSemanticSearchIndex:
    """Tests for SemanticSearchIndex."""

    def test_get_fingerprint(self, sample_search_index):
        """Test getting fingerprint by ID."""
        fp = sample_search_index.get_fingerprint("dataset_001")
        assert fp is not None
        assert fp.dataset_id == "dataset_001"

    def test_get_fingerprint_not_found(self, sample_search_index):
        """Test getting non-existent fingerprint."""
        fp = sample_search_index.get_fingerprint("nonexistent")
        assert fp is None


class TestComputeSemanticScoreForResult:
    """Tests for compute_semantic_score_for_result."""

    def test_computes_score(self, sample_search_index):
        """Test computing semantic score."""
        parsed_query = {
            "query": "reversal learning neuropixels",
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "analysis": [],
            "affordances": [],
        }

        result = compute_semantic_score_for_result(
            sample_search_index,
            "dataset_001",
            parsed_query,
        )

        assert isinstance(result, SemanticScoreResult)
        assert 0 <= result.score <= 1
        assert result.task_relevance > 0.5  # Should match task
        assert result.modality_relevance > 0.5  # Should match modality

    def test_higher_score_for_matching_dataset(self, sample_search_index):
        """Test that matching datasets get higher scores."""
        parsed_query = {
            "query": "reversal learning neuropixels",
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "analysis": [],
            "affordances": [],
        }

        # dataset_001 and dataset_003 have reversal_learning + neuropixels
        # dataset_002 has spatial_navigation + calcium_imaging
        result_match = compute_semantic_score_for_result(
            sample_search_index, "dataset_001", parsed_query
        )
        result_mismatch = compute_semantic_score_for_result(
            sample_search_index, "dataset_002", parsed_query
        )

        assert result_match.score > result_mismatch.score

    def test_handles_missing_fingerprint(self, sample_search_index):
        """Test handling of missing fingerprint."""
        parsed_query = {
            "query": "test query",
            "tasks": [],
            "modalities": [],
            "behaviors": [],
            "analysis": [],
            "affordances": [],
        }

        result = compute_semantic_score_for_result(
            sample_search_index,
            "nonexistent_dataset",
            parsed_query,
        )

        assert result.score == 0.0
        assert "No semantic fingerprint" in result.explanation

    def test_returns_similar_datasets(self, sample_search_index):
        """Test that similar datasets are returned."""
        parsed_query = {
            "query": "reversal learning",
            "tasks": ["reversal_learning"],
            "modalities": [],
            "behaviors": [],
            "analysis": [],
            "affordances": [],
        }

        result = compute_semantic_score_for_result(
            sample_search_index,
            "dataset_001",
            parsed_query,
        )

        # dataset_003 should be similar to dataset_001
        assert isinstance(result.similar_datasets, list)

    def test_neutral_score_for_empty_query(self, sample_search_index):
        """Test neutral scores when query has no constraints."""
        parsed_query = {
            "query": "",
            "tasks": [],
            "modalities": [],
            "behaviors": [],
            "analysis": [],
            "affordances": [],
        }

        result = compute_semantic_score_for_result(
            sample_search_index,
            "dataset_001",
            parsed_query,
        )

        # With no constraints, relevance should be neutral (0.5)
        assert result.task_relevance == 0.5
        assert result.modality_relevance == 0.5


class TestLoadSemanticIndex:
    """Tests for load_semantic_index."""

    def test_returns_none_for_missing_files(self, tmp_path):
        """Test that None is returned when files don't exist."""
        index = load_semantic_index(
            fingerprints_path=tmp_path / "nonexistent.jsonl",
        )
        assert index is None

    def test_loads_from_valid_files(self, sample_fingerprints, tmp_path):
        """Test loading from valid files."""
        from neural_search.embeddings import write_semantic_fingerprints

        # Write test fingerprints
        fp_path = tmp_path / "fingerprints.jsonl"
        write_semantic_fingerprints(list(sample_fingerprints.values()), fp_path)

        # Load index
        index = load_semantic_index(fingerprints_path=fp_path)

        assert index is not None
        assert len(index.fingerprints) == 3
        assert "dataset_001" in index.fingerprints
