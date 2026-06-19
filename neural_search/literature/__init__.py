"""Literature layer: paper ingest, finding extraction, normalization, KG, and search."""

from neural_search.literature.normalizer import (
    deduplicate_findings,
    normalize_cell_types,
    normalize_finding,
    normalize_molecules,
    normalize_regions,
    normalize_species,
    normalize_tasks,
)
from neural_search.literature.relationship_builder import (
    build_consensus_summaries,
    build_cross_finding_edges,
    build_region_cooccurrence_edges,
)
from neural_search.literature.search import (
    FindingResult,
    PaperResult,
    search_findings,
    search_papers,
)

__all__ = [
    "FindingResult",
    "PaperResult",
    "search_findings",
    "search_papers",
    "normalize_finding",
    "normalize_species",
    "normalize_regions",
    "normalize_tasks",
    "normalize_cell_types",
    "normalize_molecules",
    "deduplicate_findings",
    "build_cross_finding_edges",
    "build_region_cooccurrence_edges",
    "build_consensus_summaries",
]
