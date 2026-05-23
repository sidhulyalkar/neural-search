from neural_search.embeddings import HashingEmbeddingProvider, cosine_similarity


def test_hashing_embedding_provider_is_deterministic_and_normalized():
    provider = HashingEmbeddingProvider(dimensions=16)

    first = provider.embed_texts(["Go NoGo calcium imaging"])[0]
    second = provider.embed_texts(["Go NoGo calcium imaging"])[0]

    assert first == second
    assert len(first) == 16
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
