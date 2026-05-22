"""Base connector interface for data sources."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import DatasetRecord


class DataSourceConnector(ABC):
    """Abstract base class for data source connectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass

    @abstractmethod
    async def search(
        self,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> list[DatasetRecord]:
        """
        Search for datasets matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching dataset records.
        """
        pass

    @abstractmethod
    async def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """
        Fetch a specific dataset by ID.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            DatasetRecord if found, None otherwise.
        """
        pass

    @abstractmethod
    async def list_datasets(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> AsyncIterator[DatasetRecord]:
        """
        Iterate over all datasets.

        Args:
            offset: Starting offset.
            limit: Batch size.

        Yields:
            DatasetRecord for each dataset.
        """
        pass

    async def ingest_all(
        self,
        limit: Optional[int] = None,
    ) -> list[DatasetRecord]:
        """
        Ingest all datasets from this source.

        Args:
            limit: Maximum number of datasets to ingest.

        Returns:
            List of ingested dataset records.
        """
        records = []
        count = 0
        async for record in self.list_datasets():
            records.append(record)
            count += 1
            if limit and count >= limit:
                break
        return records
