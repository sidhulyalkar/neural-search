"""Tests for NVIDIA NeMo Retriever embedding and reranking NIM support."""

from __future__ import annotations

import json

import httpx

from neural_search.inference import (
    EmbeddingRequest,
    InferenceCapability,
    InferenceRegistry,
    InferenceService,
    ModelProfile,
    ProviderSettings,
    RerankRequest,
)


def test_nim_embedding_endpoint_and_manifest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content)
        assert payload["input_type"] == "query"
        assert payload["truncate"] == "END"
        return httpx.Response(
            200,
            json={
                "model": "nvidia/test-embed",
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )

    registry = InferenceRegistry(
        providers={
            "nim_embeddings": ProviderSettings(
                name="nim_embeddings",
                kind="nim",
                base_url="http://embed.local",
            )
        },
        models={
            "embeddings": ModelProfile(
                name="embeddings",
                provider="nim_embeddings",
                model="nvidia/test-embed",
                capabilities={InferenceCapability.EMBEDDING},
            )
        },
    )
    service = InferenceService(
        registry,
        transports={"nim_embeddings": httpx.MockTransport(handler)},
    )
    result = service.embed(
        EmbeddingRequest(
            inputs=["query text", "second text"],
            input_type="query",
            metadata={"input_revision": "corpus@v1"},
        )
    )
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.manifest.capability == InferenceCapability.EMBEDDING
    assert result.manifest.input_revision == "corpus@v1"
    service.close()


def test_nim_ranking_endpoint_preserves_passage_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ranking"
        payload = json.loads(request.content)
        assert payload["query"] == {"text": "spike sorting"}
        assert payload["passages"][1] == {"text": "Kilosort clustering"}
        return httpx.Response(
            200,
            json={
                "model": "nvidia/test-rerank",
                "rankings": [
                    {"index": 1, "logit": 3.2},
                    {"index": 0, "logit": 0.4},
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

    registry = InferenceRegistry(
        providers={
            "nim_reranker": ProviderSettings(
                name="nim_reranker",
                kind="nim",
                base_url="http://rerank.local",
            )
        },
        models={
            "reranker": ModelProfile(
                name="reranker",
                provider="nim_reranker",
                model="nvidia/test-rerank",
                capabilities={InferenceCapability.RERANKING},
            )
        },
    )
    service = InferenceService(
        registry,
        transports={"nim_reranker": httpx.MockTransport(handler)},
    )
    result = service.rerank(
        RerankRequest(
            query="spike sorting",
            passages=["generic image processing", "Kilosort clustering"],
        )
    )
    assert result.rankings[0].index == 1
    assert result.rankings[0].passage == "Kilosort clustering"
    assert result.rankings[0].score == 3.2
    assert result.manifest.capability == InferenceCapability.RERANKING
    service.close()
