"""Ingestion helpers and data source connectors.

Connectors:
- demo_seed: Demo fixture data for testing
- openneuro: OpenNeuro BIDS dataset connector
- openalex: OpenAlex paper linking connector
"""

from neural_search.ingestion.demo_seed import build_demo_seed, seed_demo_database

__all__ = [
    "build_demo_seed",
    "seed_demo_database",
]

