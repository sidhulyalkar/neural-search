"""Tests for embedding providers and model comparison."""

import pytest

from neural_search.embeddings.providers import (
    EmbeddingRecord,
    HashingEmbeddingProvider,
    check_provider_availability,
    get_provider,
    list_available_providers,
)
from neural_search.embeddings.model_comparison import (
    EmbeddingIndex,
    ModelComparisonReport,
    compare_embedding_models,
    cosine_similarity,
    generate_comparison_report_markdown,
)


class TestHashingEmbeddingProvider:
    """Tests for the hashing embedding provider."""

    def test_provider_properties(self):
        """Test provider metadata properties."""
        provider = HashingEmbeddingProvider(dimensions=64)
        assert provider.provider_name == "hashing"
        assert provider.dimension == 64
        assert provider.normalize is True
        assert "hashing" in provider.model_name

    def test_embed_text_is_deterministic(self):
        """Test that same text produces same embedding."""
        provider = HashingEmbeddingProvider()
        text = "neural activity in visual cortex"

        emb1 = provider.embed_text(text)
        emb2 = provider.embed_text(text)

        assert emb1 == emb2

    def test_embed_text_is_normalized(self):
        """Test that embeddings are L2-normalized."""
        import math

        provider = HashingEmbeddingProvider(normalize_embeddings=True)
        embedding = provider.embed_text("some text here")

        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 0.001 or norm == 0  # Either normalized or zero

    def test_embed_batch(self):
        """Test batch embedding."""
        provider = HashingEmbeddingProvider()
        texts = ["text one", "text two", "text three"]

        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == provider.dimension for e in embeddings)

    def test_different_text_different_embedding(self):
        """Test that different text produces different embeddings."""
        provider = HashingEmbeddingProvider()

        emb1 = provider.embed_text("calcium imaging mouse V1")
        emb2 = provider.embed_text("human fMRI resting state")

        assert emb1 != emb2


class TestEmbeddingRecord:
    """Tests for the EmbeddingRecord schema."""

    def test_create_embedding_record(self):
        """Test creating an embedding record."""
        provider = HashingEmbeddingProvider()
        record = provider.create_embedding_record(
            entity_id="dataset:dandi:000026",
            field="text_card",
            text="Neuropixels recordings from mouse visual cortex",
        )

        assert record.entity_id == "dataset:dandi:000026"
        assert record.field == "text_card"
        assert record.provider == "hashing"
        assert len(record.vector) == provider.dimension
        assert record.vector_hash
        assert record.text_hash

    def test_vector_hash_is_deterministic(self):
        """Test that vector hash is deterministic."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        hash1 = EmbeddingRecord.compute_vector_hash(vector)
        hash2 = EmbeddingRecord.compute_vector_hash(vector)
        assert hash1 == hash2

    def test_text_hash_is_deterministic(self):
        """Test that text hash is deterministic."""
        text = "some scientific text"
        hash1 = EmbeddingRecord.compute_text_hash(text)
        hash2 = EmbeddingRecord.compute_text_hash(text)
        assert hash1 == hash2


class TestProviderRegistry:
    """Tests for the provider registry functions."""

    def test_list_available_providers(self):
        """Test listing providers."""
        providers = list_available_providers()
        assert "hashing" in providers
        assert len(providers) >= 1

    def test_check_provider_availability(self):
        """Test checking availability."""
        availability = check_provider_availability()
        assert availability["hashing"] is True
        # sentence-transformer may or may not be available

    def test_get_hashing_provider(self):
        """Test getting hashing provider by name."""
        provider = get_provider("hashing")
        assert provider.provider_name == "hashing"

    def test_get_auto_provider(self):
        """Test auto provider selection."""
        provider = get_provider("auto")
        # Should get either sentence-transformer or hashing
        assert provider.provider_name in ["hashing", "sentence-transformer"]

    def test_get_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_model")


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2)) < 0.001

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(v1, v2) + 1.0) < 0.001

    def test_dimension_mismatch_raises(self):
        """Test that mismatched dimensions raise error."""
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestEmbeddingIndex:
    """Tests for the in-memory embedding index."""

    def test_add_and_search(self):
        """Test adding entities and searching."""
        provider = HashingEmbeddingProvider()
        index = EmbeddingIndex(provider=provider)

        index.add("doc1", "calcium imaging in visual cortex")
        index.add("doc2", "electrophysiology in motor cortex")
        index.add("doc3", "calcium imaging in prefrontal cortex")

        results = index.search("calcium imaging", top_k=2)

        assert len(results) == 2
        # Results should include calcium imaging docs
        result_ids = [r[0] for r in results]
        assert "doc1" in result_ids or "doc3" in result_ids

    def test_add_batch(self):
        """Test batch adding entities."""
        provider = HashingEmbeddingProvider()
        index = EmbeddingIndex(provider=provider)

        entities = [
            ("doc1", "text one"),
            ("doc2", "text two"),
            ("doc3", "text three"),
        ]
        index.add_batch(entities)

        assert len(index.ids) == 3
        assert len(index.embeddings) == 3

    def test_empty_index_search(self):
        """Test searching empty index returns empty."""
        provider = HashingEmbeddingProvider()
        index = EmbeddingIndex(provider=provider)

        results = index.search("any query")
        assert results == []


class TestModelComparison:
    """Tests for model comparison functionality."""

    def test_compare_hashing_model(self):
        """Test comparing with hashing model only."""
        corpus = [
            {"dataset_id": "d1", "text_card": "calcium imaging mouse visual cortex"},
            {"dataset_id": "d2", "text_card": "electrophysiology rat motor cortex"},
            {"dataset_id": "d3", "text_card": "fMRI human decision making"},
        ]
        queries = ["calcium imaging", "motor cortex"]
        labels = {
            "calcium imaging": {"d1"},
            "motor cortex": {"d2"},
        }

        report = compare_embedding_models(
            queries=queries,
            corpus=corpus,
            relevance_labels=labels,
            models=["hashing"],
        )

        assert "hashing" in report.models
        assert "hashing" in report.model_metrics
        assert report.query_count == 2
        assert report.corpus_size == 3

    def test_comparison_report_has_required_fields(self):
        """Test that comparison report has all required fields."""
        corpus = [
            {"dataset_id": "d1", "text_card": "text one"},
            {"dataset_id": "d2", "text_card": "text two"},
        ]
        queries = ["query"]
        labels = {"query": {"d1"}}

        report = compare_embedding_models(
            queries=queries,
            corpus=corpus,
            relevance_labels=labels,
            models=["hashing"],
        )

        assert report.generated_at
        assert report.best_precision_model
        assert report.best_mrr_model

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        corpus = [
            {"dataset_id": "d1", "text_card": "text one"},
        ]
        queries = ["query"]
        labels = {"query": {"d1"}}

        report = compare_embedding_models(
            queries=queries,
            corpus=corpus,
            relevance_labels=labels,
            models=["hashing"],
        )

        markdown = generate_comparison_report_markdown(report)

        assert "# Embedding Model Comparison Report" in markdown
        assert "hashing" in markdown
        assert "P@5" in markdown
