"""
Neural Search Ingestion Package

Connectors for ingesting data from external sources:
- DANDI Archive
- OpenNeuro
- OpenAlex
"""

from .base import DataSourceConnector
from .dandi import DandiConnector
from .openneuro import OpenNeuroConnector
from .openalex import OpenAlexConnector

__all__ = [
    "DataSourceConnector",
    "DandiConnector",
    "OpenNeuroConnector",
    "OpenAlexConnector",
]
