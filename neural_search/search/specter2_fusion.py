"""SPECTER2 score fusion for hybrid retrieval.

Loads cached SPECTER2 embeddings (built by scripts/eval/run_specter2_comparison.py)
and adds a weighted cosine-similarity signal to each scored result.

This is a post-scoring pass: neural_search ontology/semantic scores are computed
first, then the SPECTER2 signal is fused as an additive term capped at max_weight.
SPECTER2 tends to outperform BGE-large on abstract-heavy queries; BGE-large wins
on short metadata-heavy queries. Fusion improves coverage of both.

Configuration (in retrieval.yaml / retrieval_config):
    specter2:
      enabled: false
      embeddings_path: data/embeddings/specter2_corpus.jsonl
      weight: 0.20        # additive cap on score contribution
      model: allenai/specter2_base

Environment: SPECTER2 model requires ~2GB download on first use.
Run scripts/eval/run_specter2_comparison.py --build-embeddings first.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_WEIGHT = 0.20
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings/specter2_corpus.jsonl")
DEFAULT_MODEL = "allenai/specter2_base"


@lru_cache(maxsize=1)
def _load_embeddings(path: str) -> tuple[list[str], list[list[float]]]:
    """Load SPECTER2 embeddings cache; returns (ids, vectors)."""
    p = Path(path)
    if not p.exists():
        log.warning("SPECTER2 embeddings not found: %s", p)
        return [], []
    ids: list[str] = []
    vecs: list[list[float]] = []
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(str(rec["dataset_id"]))
            vecs.append(rec["embedding"])
    log.info("Loaded %d SPECTER2 embeddings from %s", len(ids), p)
    return ids, vecs


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _embed_query(query: str, model_name: str) -> list[float] | None:
    """Embed query with SPECTER2; returns None if model unavailable."""
    try:
        from neural_search.embeddings.providers import SPECTER2Provider
        provider = SPECTER2Provider(model_name)
        vec = provider.embed(query)
        return vec.tolist()
    except Exception as exc:
        log.debug("SPECTER2 embed failed: %s", exc)
        return None


def augment_with_specter2(
    results: list[Any],
    query: str,
    config: dict[str, Any],
) -> None:
    """Add SPECTER2 cosine similarity score to each result in-place.

    Args:
        results: List of SearchResult objects (mutated in-place).
        query: The search query string.
        config: The specter2 config block from retrieval config.
    """
    if not results:
        return

    embeddings_path = str(config.get("embeddings_path", DEFAULT_EMBEDDINGS_PATH))
    weight = float(config.get("weight", DEFAULT_WEIGHT))
    model_name = str(config.get("model", DEFAULT_MODEL))

    try:
        ids, vecs = _load_embeddings(embeddings_path)
    except Exception as exc:
        log.warning("Failed to load SPECTER2 embeddings: %s", exc)
        return

    if not ids:
        return

    id_to_vec: dict[str, list[float]] = dict(zip(ids, vecs))

    q_vec = _embed_query(query, model_name)
    if q_vec is None:
        return

    augmented = 0
    for result in results:
        did = str(result.dataset_id)
        d_vec = id_to_vec.get(did)
        if d_vec is None:
            result.score_breakdown["specter2_score"] = 0.0
            continue

        sim = _cosine(q_vec, d_vec)
        contribution = round(min(sim * weight, weight), 4)
        result.score_breakdown["specter2_score"] = round(sim, 4)
        result.score_breakdown["specter2_contribution"] = contribution

        new_score = min(result.score + contribution * 100, 100.0)
        result.score = round(new_score, 2)
        if "final_score" in result.score_breakdown:
            result.score_breakdown["final_score"] = round(
                result.score_breakdown["final_score"] + contribution, 4
            )
        augmented += 1

    log.debug("SPECTER2: augmented %d/%d results", augmented, len(results))
