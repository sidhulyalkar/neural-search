"""Ingestion helpers and data source connectors.

Connectors:
- demo_seed: Demo fixture data for testing
- curated: Manually curated seed sources from YAML
- openneuro: OpenNeuro BIDS dataset connector
- openalex: OpenAlex paper linking connector
- registry: Adapter registry for corpus expansion adapters
"""

from neural_search.ingestion.curated import (
    CuratedSource,
    Priority,
    SourceType,
    curated_to_dataset_create,
    curated_to_paper_stub,
    load_by_priority,
    load_curated_datasets,
    load_curated_papers,
    load_curated_sources,
    load_high_priority,
    summarize_curated_sources,
)
from neural_search.ingestion.demo_seed import build_demo_seed, seed_demo_database
from neural_search.ingestion.registry import list_adapters, register, run_adapter

__all__ = [
    # Adapter registry
    "register",
    "run_adapter",
    "list_adapters",
    # Demo seed
    "build_demo_seed",
    "seed_demo_database",
    # Curated sources
    "CuratedSource",
    "Priority",
    "SourceType",
    "load_curated_sources",
    "load_curated_datasets",
    "load_curated_papers",
    "load_by_priority",
    "load_high_priority",
    "curated_to_dataset_create",
    "curated_to_paper_stub",
    "summarize_curated_sources",
]

