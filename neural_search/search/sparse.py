"""Sparse retrieval index with BM25-like scoring for high-recall candidate generation.

This module implements a deterministic, field-weighted BM25 index over normalized
dataset records. It provides lexical candidate generation that complements semantic
and ontology-based retrieval.

Mathematical foundation:
    BM25(d, q) = sum_{t in q} IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d|/avgdl))

    Where:
    - tf: term frequency in document
    - |d|: document length
    - avgdl: average document length in corpus
    - k1: term frequency saturation parameter (typically 1.2-2.0)
    - b: length normalization parameter (typically 0.75)
    - IDF(t): log((N - n_t + 0.5) / (n_t + 0.5)) where N is corpus size, n_t is doc freq

Field weighting extends this to multiple fields with configurable boosts:
    BM25_field(d, q) = sum_{f in fields} w_f * BM25(d_f, q)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from neural_search.ontology import normalize_text


@dataclass
class SparseIndexConfig:
    """Configuration for BM25 sparse index."""

    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization

    # Field weights (higher = more important)
    field_weights: dict[str, float] = field(default_factory=lambda: {
        "title": 3.0,
        "description": 1.5,
        "tasks": 2.5,
        "modalities": 2.0,
        "species": 2.0,
        "brain_regions": 1.8,
        "behaviors": 2.0,
        "analysis_goals": 1.5,
        "data_standards": 1.0,
        "source_id": 0.5,
    })

    # Minimum document frequency for term inclusion
    min_df: int = 1

    # Maximum document frequency ratio for term inclusion (IDF filter)
    max_df_ratio: float = 0.95


@dataclass
class SparseCandidate:
    """A candidate document from sparse retrieval."""

    dataset_id: str
    score: float
    matched_terms: list[str]
    field_contributions: dict[str, float]
    term_scores: dict[str, float]


@dataclass
class SparseIndexStats:
    """Statistics about the sparse index."""

    num_documents: int
    num_terms: int
    avg_document_length: float
    field_avg_lengths: dict[str, float]


class SparseIndex:
    """BM25-based sparse retrieval index for scientific datasets.

    Builds an inverted index over normalized dataset records with field-specific
    term frequencies and configurable BM25 parameters.

    Example:
        >>> config = SparseIndexConfig(k1=1.5, b=0.75)
        >>> index = SparseIndex(config)
        >>> index.build(datasets)
        >>> candidates = index.search("mouse neuropixels decision making", top_k=20)
    """

    def __init__(self, config: SparseIndexConfig | None = None):
        self.config = config or SparseIndexConfig()

        # Inverted index: term -> {doc_id -> {field -> tf}}
        self._inverted_index: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Document info: doc_id -> {field -> length}
        self._doc_lengths: dict[str, dict[str, int]] = {}

        # Corpus statistics
        self._doc_ids: list[str] = []
        self._num_docs: int = 0
        self._doc_freq: dict[str, int] = Counter()  # term -> num docs containing term
        self._field_total_lengths: dict[str, int] = defaultdict(int)
        self._field_doc_counts: dict[str, int] = defaultdict(int)

        # Precomputed IDF values
        self._idf_cache: dict[str, float] = {}

        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into normalized terms."""
        if not text:
            return []
        normalized = normalize_text(text)
        # Split on whitespace and filter empty tokens
        tokens = [t.strip() for t in re.split(r'\s+', normalized) if t.strip()]
        # Also include bigrams for phrase matching
        bigrams = [
            f"{tokens[i]}_{tokens[i+1]}"
            for i in range(len(tokens) - 1)
        ]
        return tokens + bigrams

    def _get_field_text(self, dataset: Mapping[str, Any], field_name: str) -> str:
        """Extract text content from a dataset field."""
        value = dataset.get(field_name)
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            # Handle list of strings or list of dicts with 'label' or 'id' keys
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping):
                    parts.append(str(item.get("label", item.get("id", ""))))
            return " ".join(parts)
        if isinstance(value, Mapping):
            # Handle dict with text content
            return str(value.get("text", value.get("label", value.get("id", ""))))
        return str(value)

    def _get_dataset_id(self, dataset: Mapping[str, Any]) -> str:
        """Extract dataset ID from various possible fields."""
        for key in ("dataset_id", "id", "source_id"):
            if key in dataset and dataset[key]:
                return str(dataset[key])
        return str(hash(str(dataset)))

    def build(self, datasets: Sequence[Mapping[str, Any]]) -> SparseIndexStats:
        """Build the sparse index from a collection of datasets.

        Args:
            datasets: Sequence of dataset records with metadata fields.

        Returns:
            SparseIndexStats with index statistics.
        """
        # Clear any existing index
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._doc_ids.clear()
        self._doc_freq.clear()
        self._field_total_lengths.clear()
        self._field_doc_counts.clear()
        self._idf_cache.clear()

        # First pass: build inverted index and collect statistics
        for dataset in datasets:
            doc_id = self._get_dataset_id(dataset)
            self._doc_ids.append(doc_id)
            self._doc_lengths[doc_id] = {}

            doc_terms: set[str] = set()

            for field_name, weight in self.config.field_weights.items():
                if weight <= 0:
                    continue

                text = self._get_field_text(dataset, field_name)
                tokens = self._tokenize(text)

                if not tokens:
                    self._doc_lengths[doc_id][field_name] = 0
                    continue

                # Count term frequencies for this field
                term_counts = Counter(tokens)
                self._doc_lengths[doc_id][field_name] = len(tokens)
                self._field_total_lengths[field_name] += len(tokens)
                self._field_doc_counts[field_name] += 1

                for term, count in term_counts.items():
                    self._inverted_index[term][doc_id][field_name] = count
                    doc_terms.add(term)

            # Update document frequencies
            for term in doc_terms:
                self._doc_freq[term] += 1

        self._num_docs = len(self._doc_ids)

        # Precompute IDF values
        self._compute_idf_cache()

        self._built = True

        # Compute statistics
        field_avg_lengths = {
            f: self._field_total_lengths[f] / max(self._field_doc_counts[f], 1)
            for f in self.config.field_weights
        }

        total_length = sum(self._field_total_lengths.values())
        avg_doc_length = total_length / max(self._num_docs, 1)

        return SparseIndexStats(
            num_documents=self._num_docs,
            num_terms=len(self._inverted_index),
            avg_document_length=avg_doc_length,
            field_avg_lengths=field_avg_lengths,
        )

    def _compute_idf_cache(self) -> None:
        """Precompute IDF values for all terms."""
        for term, df in self._doc_freq.items():
            # Filter by min_df and max_df_ratio
            if df < self.config.min_df:
                self._idf_cache[term] = 0.0
                continue
            if df / max(self._num_docs, 1) > self.config.max_df_ratio:
                self._idf_cache[term] = 0.0
                continue

            # Robertson-Sparck Jones IDF formula
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf_cache[term] = max(idf, 0.0)

    def _bm25_term_score(
        self,
        tf: int,
        idf: float,
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        """Compute BM25 score for a single term in a single field."""
        if tf == 0 or idf <= 0:
            return 0.0

        k1 = self.config.k1
        b = self.config.b

        # Length normalization
        length_norm = 1 - b + b * (doc_length / max(avg_doc_length, 1))

        # BM25 term score
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * length_norm

        return idf * (numerator / denominator)

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[SparseCandidate]:
        """Search the index for matching documents.

        Args:
            query: Search query string.
            top_k: Maximum number of candidates to return.
            min_score: Minimum score threshold for inclusion.

        Returns:
            List of SparseCandidate objects sorted by score descending.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Compute average field lengths
        field_avg_lengths = {
            f: self._field_total_lengths[f] / max(self._field_doc_counts[f], 1)
            for f in self.config.field_weights
        }

        # Accumulate scores per document
        doc_scores: dict[str, float] = defaultdict(float)
        doc_matched_terms: dict[str, set[str]] = defaultdict(set)
        doc_field_contributions: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        doc_term_scores: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for term in set(query_terms):  # Deduplicate query terms
            idf = self._idf_cache.get(term, 0.0)
            if idf <= 0:
                continue

            if term not in self._inverted_index:
                continue

            for doc_id, field_tfs in self._inverted_index[term].items():
                term_total_score = 0.0

                for field_name, tf in field_tfs.items():
                    weight = self.config.field_weights.get(field_name, 0.0)
                    if weight <= 0:
                        continue

                    doc_length = self._doc_lengths.get(doc_id, {}).get(field_name, 0)
                    avg_length = field_avg_lengths.get(field_name, 1.0)

                    field_score = self._bm25_term_score(tf, idf, doc_length, avg_length)
                    weighted_score = weight * field_score

                    term_total_score += weighted_score
                    doc_field_contributions[doc_id][field_name] += weighted_score

                if term_total_score > 0:
                    doc_scores[doc_id] += term_total_score
                    doc_matched_terms[doc_id].add(term)
                    doc_term_scores[doc_id][term] = term_total_score

        # Build candidate list
        candidates: list[SparseCandidate] = []
        for doc_id, score in doc_scores.items():
            if score < min_score:
                continue

            candidates.append(SparseCandidate(
                dataset_id=doc_id,
                score=score,
                matched_terms=sorted(doc_matched_terms[doc_id]),
                field_contributions=dict(doc_field_contributions[doc_id]),
                term_scores=dict(doc_term_scores[doc_id]),
            ))

        # Sort by score descending, then by dataset_id for stability
        candidates.sort(key=lambda c: (-c.score, c.dataset_id))

        return candidates[:top_k]

    def get_stats(self) -> SparseIndexStats | None:
        """Get index statistics if built."""
        if not self._built:
            return None

        field_avg_lengths = {
            f: self._field_total_lengths[f] / max(self._field_doc_counts[f], 1)
            for f in self.config.field_weights
        }

        total_length = sum(self._field_total_lengths.values())
        avg_doc_length = total_length / max(self._num_docs, 1)

        return SparseIndexStats(
            num_documents=self._num_docs,
            num_terms=len(self._inverted_index),
            avg_document_length=avg_doc_length,
            field_avg_lengths=field_avg_lengths,
        )

    def explain_score(
        self,
        query: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Explain the BM25 score for a specific query-document pair.

        Args:
            query: Search query string.
            dataset_id: ID of the dataset to explain.

        Returns:
            Dictionary with detailed score explanation.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_terms = self._tokenize(query)
        if not query_terms:
            return {"error": "Empty query"}

        if dataset_id not in self._doc_lengths:
            return {"error": f"Dataset {dataset_id} not in index"}

        field_avg_lengths = {
            f: self._field_total_lengths[f] / max(self._field_doc_counts[f], 1)
            for f in self.config.field_weights
        }

        term_explanations: list[dict[str, Any]] = []
        total_score = 0.0

        for term in set(query_terms):
            idf = self._idf_cache.get(term, 0.0)
            df = self._doc_freq.get(term, 0)

            term_info: dict[str, Any] = {
                "term": term,
                "idf": round(idf, 4),
                "document_frequency": df,
                "field_scores": {},
                "total_term_score": 0.0,
            }

            if term in self._inverted_index and dataset_id in self._inverted_index[term]:
                field_tfs = self._inverted_index[term][dataset_id]

                for field_name, tf in field_tfs.items():
                    weight = self.config.field_weights.get(field_name, 0.0)
                    doc_length = self._doc_lengths.get(dataset_id, {}).get(field_name, 0)
                    avg_length = field_avg_lengths.get(field_name, 1.0)

                    raw_score = self._bm25_term_score(tf, idf, doc_length, avg_length)
                    weighted_score = weight * raw_score

                    term_info["field_scores"][field_name] = {
                        "tf": tf,
                        "doc_length": doc_length,
                        "avg_length": round(avg_length, 2),
                        "field_weight": weight,
                        "raw_bm25": round(raw_score, 4),
                        "weighted_score": round(weighted_score, 4),
                    }
                    term_info["total_term_score"] += weighted_score

            term_info["total_term_score"] = round(term_info["total_term_score"], 4)
            total_score += term_info["total_term_score"]
            term_explanations.append(term_info)

        return {
            "query": query,
            "dataset_id": dataset_id,
            "query_terms": list(set(query_terms)),
            "total_score": round(total_score, 4),
            "bm25_params": {"k1": self.config.k1, "b": self.config.b},
            "term_explanations": term_explanations,
        }


def build_sparse_index(
    datasets: Sequence[Mapping[str, Any]],
    config: SparseIndexConfig | None = None,
) -> SparseIndex:
    """Convenience function to build a sparse index from datasets.

    Args:
        datasets: Sequence of dataset records.
        config: Optional index configuration.

    Returns:
        Built SparseIndex ready for search.
    """
    index = SparseIndex(config)
    index.build(datasets)
    return index
