"""Hybrid search engine combining keyword, ontology, and vector search."""

import time
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import (
    DatasetRecord,
    SearchQuery,
    SearchResult,
    SearchResultItem,
)
from ontology import OntologyMatcher, get_ontology

from .ranker import HybridRanker
from .embeddings import EmbeddingService


class SearchEngine:
    """
    Hybrid search engine for neural datasets.

    Combines:
    - Keyword search (text matching)
    - Ontology matching (synonym expansion)
    - Vector search (semantic similarity)
    - Metadata filtering
    - Readiness weighting
    """

    def __init__(
        self,
        datasets: Optional[list[DatasetRecord]] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.datasets: dict[str, DatasetRecord] = {}
        self.matcher = OntologyMatcher(get_ontology())
        self.ranker = HybridRanker()
        self.embedding_service = embedding_service or EmbeddingService()

        if datasets:
            self.index_datasets(datasets)

    def index_datasets(self, datasets: list[DatasetRecord]) -> None:
        """Index a batch of datasets."""
        for dataset in datasets:
            self.datasets[dataset.id] = dataset

    def index_dataset(self, dataset: DatasetRecord) -> None:
        """Index a single dataset."""
        self.datasets[dataset.id] = dataset

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Execute a hybrid search.

        Args:
            query: Search query with filters.

        Returns:
            SearchResult with ranked results.
        """
        start_time = time.time()

        # Get candidates (apply filters)
        candidates = self._filter_candidates(query)

        if not candidates:
            return SearchResult(
                query=query.query,
                total_count=0,
                results=[],
                facets=self._compute_facets([]),
                search_time_ms=(time.time() - start_time) * 1000,
            )

        # Score candidates
        scored_results = []
        for dataset in candidates:
            score, why_matched = self._score_dataset(dataset, query)
            if score > 0:
                warnings = self._get_warnings(dataset)
                suggestions = self._get_suggestions(dataset, query)

                scored_results.append(
                    SearchResultItem(
                        dataset=dataset,
                        score=score,
                        why_matched=why_matched,
                        warnings=warnings,
                        suggested_next_actions=suggestions,
                    )
                )

        # Sort by score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # Apply pagination
        total_count = len(scored_results)
        paginated = scored_results[query.offset : query.offset + query.limit]

        # Compute facets
        facets = self._compute_facets(candidates)

        return SearchResult(
            query=query.query,
            total_count=total_count,
            results=paginated,
            facets=facets,
            search_time_ms=(time.time() - start_time) * 1000,
        )

    def _filter_candidates(self, query: SearchQuery) -> list[DatasetRecord]:
        """Apply metadata filters to get candidate datasets."""
        candidates = list(self.datasets.values())

        # Task filter
        if query.task_filter:
            task_set = set(t.lower() for t in query.task_filter)
            candidates = [
                d
                for d in candidates
                if any(t.lower() in task_set for t in d.tasks)
            ]

        # Modality filter
        if query.modality_filter:
            mod_set = set(m.lower() for m in query.modality_filter)
            candidates = [
                d
                for d in candidates
                if any(m.lower() in mod_set for m in d.modalities)
            ]

        # Species filter
        if query.species_filter:
            species_set = set(s.lower() for s in query.species_filter)
            candidates = [
                d
                for d in candidates
                if any(s.lower() in species_set for s in d.species)
            ]

        # Source filter
        if query.source_filter:
            source_set = set(query.source_filter)
            candidates = [d for d in candidates if d.source in source_set]

        return candidates

    def _score_dataset(
        self, dataset: DatasetRecord, query: SearchQuery
    ) -> tuple[float, list[str]]:
        """
        Score a dataset against the query.

        Returns (score, reasons).
        """
        score = 0.0
        reasons: list[str] = []
        query_lower = query.query.lower()

        # Keyword matching in title
        if query_lower in dataset.title.lower():
            score += 0.4
            reasons.append(f"Title contains '{query.query}'")

        # Keyword matching in description
        if dataset.description and query_lower in dataset.description.lower():
            score += 0.2
            reasons.append(f"Description contains '{query.query}'")

        # Ontology matching
        ontology_matches = self.matcher.find_all_matches(query.query)
        for match in ontology_matches:
            # Check if dataset has this task
            if match.task.id in [t.lower() for t in dataset.tasks]:
                score += 0.3 * match.confidence
                reasons.append(
                    f"Task match: {match.task.label} ({match.match_type})"
                )
            # Check in description
            elif dataset.description:
                for syn in match.task.synonyms:
                    if syn.lower() in dataset.description.lower():
                        score += 0.15 * match.confidence
                        reasons.append(
                            f"Task synonym '{syn}' found in description"
                        )
                        break

        # Modality relevance
        if ontology_matches:
            task_modalities = set()
            for m in ontology_matches:
                task_modalities.update(m.task.relevant_modalities)

            dataset_modalities = set(m.lower() for m in dataset.modalities)
            overlap = task_modalities & dataset_modalities
            if overlap:
                score += 0.1 * len(overlap)
                reasons.append(f"Has relevant modalities: {', '.join(overlap)}")

        # Boost for NWB files
        if dataset.nwb_count > 0:
            score += 0.05
            reasons.append(f"Has {dataset.nwb_count} NWB files")

        return score, reasons

    def _get_warnings(self, dataset: DatasetRecord) -> list[str]:
        """Get warnings about a dataset."""
        warnings = []

        if not dataset.description:
            warnings.append("Missing description")
        if not dataset.species:
            warnings.append("Species not specified")
        if not dataset.doi:
            warnings.append("No DOI available")
        if dataset.nwb_count == 0 and dataset.source.value == "dandi":
            warnings.append("No NWB files detected")

        return warnings

    def _get_suggestions(
        self, dataset: DatasetRecord, query: SearchQuery
    ) -> list[str]:
        """Get suggested next actions for a dataset."""
        suggestions = []

        suggestions.append("View dataset card")

        if dataset.nwb_count > 0:
            suggestions.append("Generate starter notebook")

        if dataset.url:
            suggestions.append("View on source repository")

        # Task-specific suggestions
        matches = self.matcher.find_all_matches(query.query)
        if matches:
            analyses = matches[0].task.suggested_analyses[:2]
            for analysis in analyses:
                suggestions.append(f"Try: {analysis.replace('_', ' ')}")

        return suggestions[:5]

    def _compute_facets(
        self, datasets: list[DatasetRecord]
    ) -> dict[str, dict[str, int]]:
        """Compute facet counts for filtering."""
        facets: dict[str, dict[str, int]] = {
            "source": {},
            "species": {},
            "modalities": {},
            "tasks": {},
        }

        for dataset in datasets:
            # Source facet
            source = dataset.source.value
            facets["source"][source] = facets["source"].get(source, 0) + 1

            # Species facet
            for species in dataset.species:
                facets["species"][species] = (
                    facets["species"].get(species, 0) + 1
                )

            # Modalities facet
            for mod in dataset.modalities:
                facets["modalities"][mod] = (
                    facets["modalities"].get(mod, 0) + 1
                )

            # Tasks facet
            for task in dataset.tasks:
                facets["tasks"][task] = facets["tasks"].get(task, 0) + 1

        return facets
