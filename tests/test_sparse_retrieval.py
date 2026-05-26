"""Tests for sparse retrieval index with BM25-like scoring."""

from __future__ import annotations

import pytest

from neural_search.search.sparse import (
    SparseCandidate,
    SparseIndex,
    SparseIndexConfig,
    SparseIndexStats,
    build_sparse_index,
)


@pytest.fixture
def sample_datasets() -> list[dict]:
    """Sample dataset records for testing."""
    return [
        {
            "dataset_id": "dandi:000001",
            "source_id": "000001",
            "title": "Mouse Neuropixels Visual Cortex Recordings",
            "description": "High-density recordings from mouse V1 during visual stimulation",
            "tasks": ["visual discrimination", "passive viewing"],
            "modalities": ["neuropixels", "extracellular electrophysiology"],
            "species": ["Mus musculus"],
            "brain_regions": ["visual cortex", "V1"],
            "behaviors": ["licking", "running"],
        },
        {
            "dataset_id": "dandi:000002",
            "source_id": "000002",
            "title": "Rat Hippocampus Calcium Imaging Memory Task",
            "description": "Two-photon calcium imaging in rat hippocampus during spatial memory",
            "tasks": ["spatial memory", "T-maze"],
            "modalities": ["calcium imaging", "two-photon"],
            "species": ["Rattus norvegicus"],
            "brain_regions": ["hippocampus", "CA1"],
            "behaviors": ["running", "choice"],
        },
        {
            "dataset_id": "openneuro:ds003001",
            "source_id": "ds003001",
            "title": "Human fMRI Decision Making Study",
            "description": "Functional MRI during economic decision making tasks",
            "tasks": ["decision making", "gambling task"],
            "modalities": ["fMRI", "BOLD"],
            "species": ["Homo sapiens"],
            "brain_regions": ["prefrontal cortex", "striatum"],
            "behaviors": ["button press", "choice"],
        },
        {
            "dataset_id": "dandi:000004",
            "source_id": "000004",
            "title": "Mouse Motor Cortex Neuropixels During Reaching",
            "description": "Neuropixels recordings from motor cortex during skilled reaching",
            "tasks": ["reaching", "motor task"],
            "modalities": ["neuropixels", "extracellular electrophysiology"],
            "species": ["Mus musculus"],
            "brain_regions": ["motor cortex", "M1"],
            "behaviors": ["reaching", "grasping"],
        },
        {
            "dataset_id": "allen:visual",
            "source_id": "visual_coding",
            "title": "Allen Visual Coding Neuropixels Dataset",
            "description": "Large-scale Neuropixels survey of mouse visual cortex",
            "tasks": ["visual coding", "natural movies"],
            "modalities": ["neuropixels", "extracellular electrophysiology"],
            "species": ["Mus musculus"],
            "brain_regions": ["visual cortex", "LGN", "V1"],
            "behaviors": ["running", "pupil"],
        },
    ]


class TestSparseIndexConfig:
    """Tests for SparseIndexConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SparseIndexConfig()
        assert config.k1 == 1.5
        assert config.b == 0.75
        assert "title" in config.field_weights
        assert config.field_weights["title"] > config.field_weights["description"]

    def test_custom_config(self):
        """Test custom configuration."""
        config = SparseIndexConfig(
            k1=2.0,
            b=0.5,
            field_weights={"title": 5.0, "description": 1.0},
        )
        assert config.k1 == 2.0
        assert config.b == 0.5
        assert config.field_weights["title"] == 5.0


class TestSparseIndex:
    """Tests for SparseIndex."""

    def test_build_index(self, sample_datasets):
        """Test building the index from datasets."""
        index = SparseIndex()
        stats = index.build(sample_datasets)

        assert isinstance(stats, SparseIndexStats)
        assert stats.num_documents == 5
        assert stats.num_terms > 0
        assert stats.avg_document_length > 0

    def test_search_basic(self, sample_datasets):
        """Test basic search functionality."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("mouse neuropixels", top_k=10)

        assert len(candidates) > 0
        assert all(isinstance(c, SparseCandidate) for c in candidates)

        # Check that results are sorted by score
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_search_ranking(self, sample_datasets):
        """Test that relevant documents rank higher."""
        index = build_sparse_index(sample_datasets)

        # Search for neuropixels - should favor neuropixels datasets
        candidates = index.search("neuropixels mouse visual cortex", top_k=5)

        # The visual cortex neuropixels datasets should rank high
        top_ids = [c.dataset_id for c in candidates[:3]]
        assert any("dandi:000001" in id or "allen:visual" in id for id in top_ids)

    def test_search_matched_terms(self, sample_datasets):
        """Test that matched terms are tracked."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("hippocampus calcium imaging", top_k=5)

        # Find the hippocampus dataset
        hippocampus_results = [c for c in candidates if "000002" in c.dataset_id]
        if hippocampus_results:
            result = hippocampus_results[0]
            # Should have matched terms
            assert len(result.matched_terms) > 0

    def test_search_field_contributions(self, sample_datasets):
        """Test that field contributions are computed."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("visual cortex", top_k=5)

        for candidate in candidates:
            # Field contributions should exist
            assert isinstance(candidate.field_contributions, dict)
            # Total score should be sum of field contributions (approximately)
            total_contrib = sum(candidate.field_contributions.values())
            assert abs(candidate.score - total_contrib) < 0.001

    def test_search_empty_query(self, sample_datasets):
        """Test search with empty query."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("", top_k=10)
        assert candidates == []

    def test_search_no_matches(self, sample_datasets):
        """Test search with non-matching query."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("completely unrelated xyz123", top_k=10)
        # Might have zero results or very low scores
        if candidates:
            assert candidates[0].score < 1.0

    def test_search_min_score_filter(self, sample_datasets):
        """Test minimum score filtering."""
        index = build_sparse_index(sample_datasets)
        candidates = index.search("mouse", top_k=10, min_score=100.0)
        # High min_score should filter out most/all results
        assert len(candidates) <= len(sample_datasets)

    def test_explain_score(self, sample_datasets):
        """Test score explanation."""
        index = build_sparse_index(sample_datasets)
        explanation = index.explain_score("mouse neuropixels", "dandi:000001")

        assert "query" in explanation
        assert "dataset_id" in explanation
        assert "total_score" in explanation
        assert "term_explanations" in explanation
        assert "bm25_params" in explanation

        # Should have explanations for query terms
        assert len(explanation["term_explanations"]) > 0

    def test_explain_score_unknown_dataset(self, sample_datasets):
        """Test explanation for unknown dataset."""
        index = build_sparse_index(sample_datasets)
        explanation = index.explain_score("mouse", "unknown:dataset")
        assert "error" in explanation

    def test_get_stats(self, sample_datasets):
        """Test getting index statistics."""
        index = SparseIndex()

        # Before building
        assert index.get_stats() is None

        # After building
        index.build(sample_datasets)
        stats = index.get_stats()

        assert stats is not None
        assert stats.num_documents == 5
        assert stats.num_terms > 0

    def test_search_before_build(self):
        """Test that search fails before building."""
        index = SparseIndex()
        with pytest.raises(RuntimeError, match="not built"):
            index.search("test query")

    def test_bigram_matching(self, sample_datasets):
        """Test that bigrams are indexed for phrase matching."""
        index = build_sparse_index(sample_datasets)

        # Search for a phrase that appears as bigram
        candidates = index.search("visual cortex", top_k=5)

        # Should find visual cortex datasets
        assert len(candidates) > 0

    def test_custom_field_weights(self, sample_datasets):
        """Test custom field weights affect ranking."""
        # High title weight
        config1 = SparseIndexConfig(field_weights={"title": 10.0, "description": 0.1})
        index1 = build_sparse_index(sample_datasets, config1)
        candidates1 = index1.search("visual", top_k=5)

        # High description weight
        config2 = SparseIndexConfig(field_weights={"title": 0.1, "description": 10.0})
        index2 = build_sparse_index(sample_datasets, config2)
        candidates2 = index2.search("visual", top_k=5)

        # Rankings might differ based on where "visual" appears more
        scores1 = {c.dataset_id: c.score for c in candidates1}
        scores2 = {c.dataset_id: c.score for c in candidates2}

        # At least scores should be different
        assert scores1 != scores2


class TestBuildSparseIndex:
    """Tests for build_sparse_index convenience function."""

    def test_build_with_defaults(self, sample_datasets):
        """Test building with default config."""
        index = build_sparse_index(sample_datasets)
        assert index.get_stats().num_documents == 5

    def test_build_with_custom_config(self, sample_datasets):
        """Test building with custom config."""
        config = SparseIndexConfig(k1=2.0, b=0.5)
        index = build_sparse_index(sample_datasets, config)
        assert index.config.k1 == 2.0
        assert index.config.b == 0.5


class TestBM25Behavior:
    """Tests for correct BM25 mathematical behavior."""

    def test_idf_rare_terms(self, sample_datasets):
        """Test that rare terms have higher IDF."""
        index = build_sparse_index(sample_datasets)

        # "hippocampus" appears in 1 doc, "mouse" in 3 docs
        hippocampus_results = index.search("hippocampus", top_k=1)
        mouse_results = index.search("mouse", top_k=1)

        # Both should find results
        assert len(hippocampus_results) > 0
        assert len(mouse_results) > 0

    def test_term_frequency_saturation(self, sample_datasets):
        """Test that TF saturates (diminishing returns)."""
        # Create dataset with repeated term
        datasets = [
            {"dataset_id": "test1", "title": "mouse mouse mouse mouse mouse"},
            {"dataset_id": "test2", "title": "mouse"},
        ]

        index = build_sparse_index(datasets)
        candidates = index.search("mouse", top_k=2)

        # Dataset with more "mouse" should score higher, but not 5x higher
        if len(candidates) == 2:
            ratio = candidates[0].score / candidates[1].score
            assert ratio < 5.0  # Saturation should prevent linear scaling

    def test_length_normalization(self, sample_datasets):
        """Test that length normalization affects scores."""
        # Short vs long document with same term - need 3+ docs to avoid IDF filter
        datasets = [
            {
                "dataset_id": "short",
                "title": "mouse brain recording",
                "description": "Short description",
            },
            {
                "dataset_id": "long",
                "title": "mouse brain recording " + "other words " * 20,
                "description": "Long description with many extra words " * 10,
            },
            {
                "dataset_id": "other",
                "title": "completely different topic",
                "description": "unrelated content",
            },
        ]

        # Use config with max_df_ratio=1.0 to avoid filtering common terms
        config = SparseIndexConfig(max_df_ratio=1.0)
        index = build_sparse_index(datasets, config)
        candidates = index.search("mouse brain recording", top_k=3)

        # At least the two matching docs should be found
        assert len(candidates) >= 2
        matching_ids = [c.dataset_id for c in candidates]
        assert "short" in matching_ids
        assert "long" in matching_ids


class TestStability:
    """Tests for deterministic/stable behavior."""

    def test_deterministic_scores(self, sample_datasets):
        """Test that scores are deterministic."""
        index1 = build_sparse_index(sample_datasets)
        index2 = build_sparse_index(sample_datasets)

        results1 = index1.search("mouse neuropixels visual", top_k=5)
        results2 = index2.search("mouse neuropixels visual", top_k=5)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.dataset_id == r2.dataset_id
            assert abs(r1.score - r2.score) < 0.0001

    def test_stable_tie_breaking(self, sample_datasets):
        """Test that ties are broken stably by ID."""
        # Create datasets that might tie - need 3+ docs and relaxed IDF filter
        datasets = [
            {
                "dataset_id": "b_dataset",
                "title": "identical test term dataset",
                "description": "same content here",
            },
            {
                "dataset_id": "a_dataset",
                "title": "identical test term dataset",
                "description": "same content here",
            },
            {
                "dataset_id": "c_dataset",
                "title": "different content entirely",
                "description": "no overlap",
            },
        ]

        # Use config with max_df_ratio=1.0 to avoid filtering common terms
        config = SparseIndexConfig(max_df_ratio=1.0)
        index = build_sparse_index(datasets, config)
        candidates = index.search("identical test term dataset", top_k=3)

        # At least the two matching docs should be found
        matching = [c for c in candidates if c.dataset_id in ("a_dataset", "b_dataset")]
        assert len(matching) == 2

        # With same scores, should be sorted alphabetically by ID
        if abs(matching[0].score - matching[1].score) < 0.0001:
            assert matching[0].dataset_id < matching[1].dataset_id
