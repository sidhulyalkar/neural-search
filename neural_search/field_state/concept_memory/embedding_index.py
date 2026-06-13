"""Optional embedding support for concept memory.

If the project's embedding provider is available it will be used; otherwise
a no-op fallback is provided so callers always get ConceptEmbeddingRecord
objects with embedding=None.
"""

from __future__ import annotations

import hashlib
from typing import Any

from neural_search.field_state.concept_memory.schema import (
    ConceptEmbeddingRecord,
    ConceptNode,
)


def build_concept_text(concept: ConceptNode) -> str:
    """Build a text representation of a concept for embedding/indexing.

    Combines: canonical_name, aliases (joined), description, tags (joined).
    """
    parts: list[str] = [concept.canonical_name]
    if concept.aliases:
        parts.append(", ".join(concept.aliases))
    if concept.description:
        parts.append(concept.description)
    if concept.tags:
        parts.append(", ".join(concept.tags))
    return " | ".join(parts)


def try_load_embedder() -> Any | None:
    """Attempt to import the project embedding provider.

    Returns the embedder object if available, else None.
    Never raises — missing provider is a silent no-op.
    """
    try:
        import importlib
        emb_module = importlib.import_module("neural_search.retrieval.embedding")
        return getattr(emb_module, "embedder", emb_module)
    except Exception:  # noqa: BLE001
        return None


def build_embedding_records(
    concepts: list[ConceptNode],
    embedder: Any | None = None,
) -> list[ConceptEmbeddingRecord]:
    """Build a ConceptEmbeddingRecord for each concept.

    If embedder is None or the embedding call fails, embedding is set to None
    and embedding_model is set to "none".
    source_hash is the SHA-256 hex digest of the concept text.
    """
    records: list[ConceptEmbeddingRecord] = []

    for concept in concepts:
        text = build_concept_text(concept)
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        embedding: list[float] | None = None
        embedding_model = "none"

        if embedder is not None:
            try:
                raw = embedder.embed(text)
                if isinstance(raw, list) and raw:
                    embedding = [float(v) for v in raw]
                    model_name = getattr(embedder, "model_name", None)
                    embedding_model = str(model_name) if model_name else "unknown"
            except Exception:  # noqa: BLE001
                embedding = None
                embedding_model = "none"

        records.append(
            ConceptEmbeddingRecord(
                concept_id=concept.concept_id,
                text=text,
                embedding_model=embedding_model,
                embedding=embedding,
                source_hash=source_hash,
            )
        )

    return records
