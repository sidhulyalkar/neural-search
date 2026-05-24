import builtins

import pytest

from neural_search.embeddings import (
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    cosine_similarity,
)


def test_hashing_embedding_provider_is_deterministic_and_normalized():
    provider = HashingEmbeddingProvider(dimensions=16)

    first = provider.embed_text("Go NoGo calcium imaging")
    second = provider.embed_batch(["Go NoGo calcium imaging"])[0]

    assert first == second
    assert len(first) == 16
    assert provider.dimension == 16
    assert provider.provider_name == "hashing"
    assert provider.normalize is True
    assert round(sum(value * value for value in first), 6) == 1.0


def test_cosine_similarity_maps_to_retrieval_signal():
    provider = HashingEmbeddingProvider(dimensions=32)
    query, related, unrelated = provider.embed_texts(
        [
            "human ECoG BCI reaching",
            "ECoG reaching BCI human motor control",
            "mouse calcium visual decision",
        ]
    )

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


class _FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return [[float(len(text)), 1.0, 0.5] for text in texts]


def test_sentence_transformer_provider_accepts_injected_model():
    provider = SentenceTransformerEmbeddingProvider(
        model_name="fake-model",
        model=_FakeSentenceTransformer(),
    )

    vectors = provider.embed_batch(["abc", "abcd"])

    assert provider.provider_name == "sentence-transformer"
    assert provider.model_name == "fake-model"
    assert provider.dimension == 3
    assert vectors == [[3.0, 1.0, 0.5], [4.0, 1.0, 0.5]]


def test_sentence_transformer_provider_has_clear_optional_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing test dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Install neural-search\\[embeddings\\]"):
        SentenceTransformerEmbeddingProvider()
