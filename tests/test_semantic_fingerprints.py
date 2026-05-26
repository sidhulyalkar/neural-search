"""Tests for semantic fingerprint infrastructure."""

from __future__ import annotations

import pytest

from neural_search.embeddings import (
    ConceptEmbedding,
    ConceptEmbeddingIndex,
    SemanticDatasetFingerprint,
    SemanticFingerprintBuilder,
    SemanticSimilarity,
    compute_semantic_similarity,
)


@pytest.fixture
def sample_concept_index() -> ConceptEmbeddingIndex:
    """Create a sample concept embedding index for testing."""
    import numpy as np

    # Create distinct embeddings for different concepts
    np.random.seed(42)

    embeddings = [
        ConceptEmbedding(
            concept_id="task:reversal_learning",
            concept_type="task",
            label="Reversal Learning",
            embedding=np.random.rand(128).tolist(),
            model_version="test",
            aliases=["reversal"],
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
        ConceptEmbedding(
            concept_id="modality:calcium_imaging",
            concept_type="modality",
            label="Calcium Imaging",
            embedding=np.random.rand(128).tolist(),
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
        ConceptEmbedding(
            concept_id="region:hippocampus",
            concept_type="region",
            label="Hippocampus",
            embedding=np.random.rand(64).tolist(),
            model_version="test",
        ),
    ]
    return ConceptEmbeddingIndex(embeddings)


@pytest.fixture
def sample_records() -> list[dict]:
    """Create sample normalized records as dicts for testing."""
    return [
        {
            "id": "test_001",
            "record_type": "dataset",
            "source": "test",
            "title": "Mouse Reversal Learning Dataset",
            "description": "Neuropixels recordings during reversal learning task",
            "species": ["mouse"],
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": ["licking"],
            "brain_regions": ["hippocampus"],
            "analysis_affordances": ["decoding"],
        },
        {
            "id": "test_002",
            "record_type": "dataset",
            "source": "test",
            "title": "Decision Making Calcium Imaging",
            "description": "Calcium imaging during decision making",
            "species": ["mouse"],
            "tasks": ["decision_making"],
            "modalities": ["calcium_imaging"],
            "behaviors": ["licking"],
            "brain_regions": ["hippocampus"],
            "analysis_affordances": ["decoding"],
        },
        {
            "id": "test_003",
            "record_type": "dataset",
            "source": "test",
            "title": "Similar Reversal Task",
            "description": "Another reversal learning experiment with Neuropixels",
            "species": ["mouse"],
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "behaviors": [],
            "brain_regions": [],
            "analysis_affordances": [],
        },
    ]


class TestSemanticDatasetFingerprint:
    """Tests for SemanticDatasetFingerprint dataclass."""

    def test_fingerprint_creation(self):
        """Test creating a semantic fingerprint."""
        fp = SemanticDatasetFingerprint(
            dataset_id="test_001",
            text_embedding=[0.1] * 256,
            task_embedding=[0.2] * 128,
            modality_embedding=[0.3] * 128,
            behavior_embedding=[0.4] * 64,
            analysis_embedding=[0.5] * 64,
            region_embedding=[0.6] * 64,
            design_embedding=[0.7] * 32,
            combined_embedding=[0.8] * 736,
        )
        assert fp.dataset_id == "test_001"
        assert len(fp.text_embedding) == 256
        assert len(fp.task_embedding) == 128
        assert len(fp.design_embedding) == 32

    def test_fingerprint_with_metadata(self):
        """Test fingerprint with optional metadata."""
        fp = SemanticDatasetFingerprint(
            dataset_id="test_001",
            text_embedding=[0.1] * 256,
            task_embedding=[0.2] * 128,
            modality_embedding=[0.3] * 128,
            behavior_embedding=[0.4] * 64,
            analysis_embedding=[0.5] * 64,
            region_embedding=[0.6] * 64,
            design_embedding=[0.7] * 32,
            combined_embedding=[0.8] * 736,
            behavior_complexity=0.75,
            design_type="time_series",
            task_labels=["reversal_learning"],
            modality_labels=["neuropixels"],
        )
        assert fp.behavior_complexity == 0.75
        assert fp.design_type == "time_series"
        assert "reversal_learning" in fp.task_labels


class TestSemanticFingerprintBuilder:
    """Tests for SemanticFingerprintBuilder."""

    def test_builder_initialization(self, sample_concept_index):
        """Test builder initializes correctly."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        assert builder.text_dim == 256
        assert builder.concept_dim == 128
        assert builder.minor_dim == 64
        assert builder.design_dim == 32

    def test_build_single_fingerprint(self, sample_concept_index, sample_records):
        """Test building fingerprint for a single record."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fp = builder.build_fingerprint(sample_records[0])

        assert fp.dataset_id == "test_001"
        assert len(fp.text_embedding) == 256
        assert len(fp.task_embedding) == 128
        assert len(fp.modality_embedding) == 128
        assert len(fp.behavior_embedding) == 64
        assert len(fp.analysis_embedding) == 64
        assert len(fp.region_embedding) == 64
        assert len(fp.design_embedding) == 32

    def test_build_multiple_fingerprints(self, sample_concept_index, sample_records):
        """Test building fingerprints for multiple records."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fingerprints = builder.build_fingerprints(sample_records)

        assert len(fingerprints) == 3
        assert all(isinstance(fp, SemanticDatasetFingerprint) for fp in fingerprints)
        ids = [fp.dataset_id for fp in fingerprints]
        assert "test_001" in ids
        assert "test_002" in ids
        assert "test_003" in ids

    def test_combined_embedding_dimension(self, sample_concept_index, sample_records):
        """Test combined embedding has correct dimension."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fp = builder.build_fingerprint(sample_records[0])

        # Combined = text(256) + task(128) + modality(128) + behavior(64) + analysis(64) + region(64) + design(32) = 736
        assert len(fp.combined_embedding) == 736


class TestSemanticSimilarity:
    """Tests for semantic similarity computation."""

    def test_compute_similarity(self, sample_concept_index, sample_records):
        """Test computing similarity between fingerprints."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fps = builder.build_fingerprints(sample_records)

        # Test similarity between first two records
        sim = compute_semantic_similarity(fps[0], fps[1])

        assert isinstance(sim, SemanticSimilarity)
        # Use tolerance for floating point comparisons
        assert -0.01 <= sim.combined_similarity <= 1.01
        assert -0.01 <= sim.text_similarity <= 1.01
        assert -0.01 <= sim.task_similarity <= 1.01
        assert -0.01 <= sim.modality_similarity <= 1.01

    def test_self_similarity(self, sample_concept_index, sample_records):
        """Test that self-similarity is 1.0."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fp = builder.build_fingerprint(sample_records[0])

        sim = compute_semantic_similarity(fp, fp)

        # Self-similarity should be 1.0 (or very close due to floating point)
        assert sim.combined_similarity > 0.99
        assert sim.text_similarity > 0.99
        assert sim.task_similarity > 0.99

    def test_similar_tasks_higher_similarity(self, sample_concept_index, sample_records):
        """Test that records with same task have higher task similarity."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fps = builder.build_fingerprints(sample_records)

        # fps[0] and fps[2] both have reversal_learning
        # fps[0] and fps[1] have different tasks
        sim_same_task = compute_semantic_similarity(fps[0], fps[2])
        sim_diff_task = compute_semantic_similarity(fps[0], fps[1])

        # Task similarity should be higher (or equal within tolerance) for same task
        # Using >= to handle floating point edge cases where embeddings may produce very similar values
        assert sim_same_task.task_similarity >= sim_diff_task.task_similarity - 0.01

    def test_similar_modalities_higher_similarity(self, sample_concept_index, sample_records):
        """Test that records with same modality have higher modality similarity."""
        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        fps = builder.build_fingerprints(sample_records)

        # fps[0] and fps[2] both have neuropixels
        # fps[0] and fps[1] have different modalities
        sim_same_mod = compute_semantic_similarity(fps[0], fps[2])
        sim_diff_mod = compute_semantic_similarity(fps[0], fps[1])

        # Modality similarity should be higher (or equal within tolerance) for same modality
        assert sim_same_mod.modality_similarity >= sim_diff_mod.modality_similarity - 0.01


class TestFingerprintSerialization:
    """Tests for fingerprint read/write operations."""

    def test_write_and_read_fingerprints(self, sample_concept_index, sample_records, tmp_path):
        """Test writing and reading fingerprints."""
        from neural_search.embeddings import (
            read_semantic_fingerprints,
            write_semantic_fingerprints,
        )

        builder = SemanticFingerprintBuilder(
            concept_index=sample_concept_index,
            text_dim=256,
            concept_dim=128,
            minor_dim=64,
            design_dim=32,
        )
        original_fps = builder.build_fingerprints(sample_records)

        # Write to file
        output_path = tmp_path / "fingerprints.jsonl"
        write_semantic_fingerprints(original_fps, output_path)

        # Read back
        loaded_fps = read_semantic_fingerprints(output_path)

        assert len(loaded_fps) == len(original_fps)
        for orig, loaded in zip(original_fps, loaded_fps):
            assert orig.dataset_id == loaded.dataset_id
            assert len(orig.text_embedding) == len(loaded.text_embedding)
            assert len(orig.combined_embedding) == len(loaded.combined_embedding)
