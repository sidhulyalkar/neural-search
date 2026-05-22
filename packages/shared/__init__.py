"""
Neural Search Shared Package

Core schemas and utilities shared across all packages.
"""

from .schemas import (
    DatasetRecord,
    DatasetSource,
    PaperRecord,
    ExtractionResult,
    ExtractionLabel,
    SearchQuery,
    SearchResult,
    DatasetCard,
    AnalysisReadiness,
)

__all__ = [
    "DatasetRecord",
    "DatasetSource",
    "PaperRecord",
    "ExtractionResult",
    "ExtractionLabel",
    "SearchQuery",
    "SearchResult",
    "DatasetCard",
    "AnalysisReadiness",
]
