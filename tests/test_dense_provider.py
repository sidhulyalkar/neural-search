"""Tests for DenseEmbeddingProvider (BGE-large-en-v1.5)."""


def test_dense_provider_import():
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    assert DenseEmbeddingProvider is not None


def test_dense_provider_metadata():
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    assert DenseEmbeddingProvider.MODEL_NAME == "BAAI/bge-large-en-v1.5"
    assert DenseEmbeddingProvider.DIMENSION == 1024


def test_dense_provider_registry():
    from neural_search.embeddings.providers import list_available_providers
    assert "bge-large" in list_available_providers()


def test_dense_provider_get_provider_bge(monkeypatch):
    """get_provider('bge-large') returns DenseEmbeddingProvider without loading model."""
    import numpy as np

    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.zeros((len(texts), 1024), dtype=np.float32)
        def get_sentence_embedding_dimension(self):
            return 1024

    monkeypatch.setattr(
        "neural_search.embeddings.dense_provider.DenseEmbeddingProvider._load_model",
        lambda self: setattr(self, "_model", _FakeModel()),
    )
    from neural_search.embeddings.providers import get_provider
    p = get_provider("bge-large")
    assert isinstance(p, DenseEmbeddingProvider)
    assert p.dimension == 1024


def test_embed_batch_shape(monkeypatch):
    """embed_batch returns list of 1024-dim vectors."""
    import numpy as np

    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.ones((len(texts), 1024), dtype=np.float32)
        def get_sentence_embedding_dimension(self):
            return 1024

    monkeypatch.setattr(
        "neural_search.embeddings.dense_provider.DenseEmbeddingProvider._load_model",
        lambda self: setattr(self, "_model", _FakeModel()),
    )
    p = DenseEmbeddingProvider()
    vecs = p.embed_batch(["text1", "text2", "text3"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 1024


def test_embed_corpus_batch_empty(monkeypatch):
    """embed_corpus_batch([]) returns (0, 1024) array, not (0,)."""
    import numpy as np

    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(
        "neural_search.embeddings.dense_provider.DenseEmbeddingProvider._load_model",
        lambda self: setattr(self, "_model", _FakeModel()),
    )
    p = DenseEmbeddingProvider()
    result = p.embed_corpus_batch([])
    assert result.shape == (0, 1024), f"Expected (0, 1024), got {result.shape}"
