"""Literature layer: paper ingest, finding extraction, normalization, KG, and search."""

from neural_search.literature.normalizer import (
    deduplicate_findings,
    normalize_finding,
    normalize_regions,
    normalize_species,
    normalize_tasks,
)
from neural_search.literature.search import FindingResult, PaperResult, search_findings, search_papers

__all__ = [
    "FindingResult",
    "PaperResult",
    "search_findings",
    "search_papers",
    "normalize_finding",
    "normalize_species",
    "normalize_regions",
    "normalize_tasks",
    "deduplicate_findings",
]
