"""DANDI Archive connector."""

from typing import AsyncIterator, Optional

import httpx

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import DatasetRecord, DatasetSource, DataStandard, DatasetAsset

from .base import DataSourceConnector


DANDI_API_URL = "https://api.dandiarchive.org/api"


class DandiConnector(DataSourceConnector):
    """
    Connector for the DANDI Archive.

    Uses the DANDI REST API to fetch Dandiset metadata.
    """

    def __init__(self, api_url: str = DANDI_API_URL):
        self.api_url = api_url
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def source_name(self) -> str:
        return "dandi"

    async def search(
        self,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> list[DatasetRecord]:
        """Search DANDI for datasets matching query."""
        params = {"page_size": limit}
        if query:
            params["search"] = query

        response = await self._client.get(
            f"{self.api_url}/dandisets/",
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("results", []):
            record = self._parse_dandiset(item)
            if record:
                results.append(record)

        return results

    async def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Fetch a specific Dandiset by ID."""
        try:
            response = await self._client.get(
                f"{self.api_url}/dandisets/{dataset_id}/",
            )
            response.raise_for_status()
            return self._parse_dandiset(response.json())
        except httpx.HTTPStatusError:
            return None

    async def list_datasets(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> AsyncIterator[DatasetRecord]:
        """Iterate over all Dandisets."""
        page = 1
        while True:
            response = await self._client.get(
                f"{self.api_url}/dandisets/",
                params={"page": page, "page_size": limit},
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                record = self._parse_dandiset(item)
                if record:
                    yield record

            if not data.get("next"):
                break
            page += 1

    async def get_assets(self, dataset_id: str) -> list[DatasetAsset]:
        """Get assets for a Dandiset."""
        assets = []
        page = 1

        while True:
            response = await self._client.get(
                f"{self.api_url}/dandisets/{dataset_id}/versions/draft/assets/",
                params={"page": page, "page_size": 100},
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                path = item.get("path", "")
                assets.append(
                    DatasetAsset(
                        path=path,
                        size_bytes=item.get("size"),
                        content_type=item.get("contentType"),
                        is_nwb=path.endswith(".nwb"),
                    )
                )

            if not data.get("next"):
                break
            page += 1

        return assets

    def _parse_dandiset(self, data: dict) -> Optional[DatasetRecord]:
        """Parse DANDI API response into DatasetRecord."""
        try:
            dandiset_id = data.get("identifier", "")
            most_recent = data.get("most_recent_published_version") or data.get("draft_version", {})
            metadata = most_recent.get("metadata", {}) or {}

            # Extract contributors
            contributors = []
            for contrib in metadata.get("contributor", []):
                if isinstance(contrib, dict):
                    name = contrib.get("name", "")
                    if name:
                        contributors.append(name)

            # Extract species
            species = []
            for s in metadata.get("species", []) or []:
                if isinstance(s, dict):
                    species.append(s.get("name", str(s)))
                else:
                    species.append(str(s))

            # Count NWB files
            asset_summary = most_recent.get("asset_count", 0)

            return DatasetRecord(
                id=f"dandi:{dandiset_id}",
                source=DatasetSource.DANDI,
                source_id=dandiset_id,
                title=metadata.get("name", f"Dandiset {dandiset_id}"),
                description=metadata.get("description"),
                contributors=contributors,
                species=species,
                license=metadata.get("license", [None])[0] if metadata.get("license") else None,
                doi=metadata.get("doi"),
                url=f"https://dandiarchive.org/dandiset/{dandiset_id}",
                version=most_recent.get("version"),
                data_standard=DataStandard.NWB,
                modalities=[],
                brain_regions=[],
                tasks=[],
                raw_metadata=data,
                nwb_count=asset_summary,
            )
        except Exception:
            return None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
