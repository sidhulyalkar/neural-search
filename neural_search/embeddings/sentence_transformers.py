"""Optional sentence-transformers embedding provider."""

from __future__ import annotations


class SentenceTransformerProvider:
    """Provider backed by sentence-transformers when the extra is installed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install neural-search[embeddings] to use SentenceTransformerProvider."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]
