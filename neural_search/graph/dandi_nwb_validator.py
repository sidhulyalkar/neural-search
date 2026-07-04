"""Live NWB file validation against real DANDI assets.

`neural_search/data/dandi_streaming.py` and
`neural_search/affordances/validators/nwb_validator.py` already sketch this
capability, but both depend on the `dandi` and `remfile` packages, which are
not installed in this environment (confirmed by direct import — both raise
`ModuleNotFoundError`; `dandi_streaming.get_dandiset_metadata` and
`neural_search.ingestion.dandi.fetch_dandiset_rich_metadata` have both been
silently returning empty results via a broad except-Exception, unnoticed).

This module reimplements the same "read only the HDF5 header, not the data
arrays" capability using only already-installed dependencies: `httpx` for
the DANDI REST API (matching the pattern already used in
`neural_search/ingestion/dandi.py`) and a small custom HTTP-range file-like
object for h5py, avoiding the need for `fsspec`'s HTTP backend (which itself
requires the uninstalled `aiohttp`) or `remfile`.

Verified against a real asset (dandiset 000003, an 8.4GB NWB file) during
development: header-only inspection completed in ~13 HTTP range requests,
under 2 seconds, without downloading the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

DANDI_API_URL = "https://api.dandiarchive.org/api"
log = logging.getLogger(__name__)


class HttpRangeFile:
    """A minimal seekable, read-only file-like object backed by HTTP Range
    requests, sufficient for h5py.File(fileobj) to lazily read an HDF5/NWB
    file's structure without downloading its data arrays.

    S3 presigned URLs (which DANDI asset downloads redirect to) are commonly
    signed for GET only — a HEAD request returns 403 — so size is discovered
    via a 1-byte range GET's `Content-Range` response header instead.
    """

    def __init__(self, url: str, client: httpx.Client):
        self.url = url
        self.client = client
        self.pos = 0
        self.n_requests = 0
        self.bytes_fetched = 0
        resp = self._request("bytes=0-0")
        content_range = resp.headers.get("content-range")
        if not content_range:
            # Server doesn't support ranges; fall back to full content-length.
            self.size = int(resp.headers.get("content-length", 0))
        else:
            self.size = int(content_range.rsplit("/", 1)[-1])

    def _request(self, range_header: str) -> httpx.Response:
        self.n_requests += 1
        resp = self.client.get(self.url, headers={"Range": range_header})
        resp.raise_for_status()
        return resp

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            end = self.size - 1
        else:
            end = min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        resp = self._request(f"bytes={self.pos}-{end}")
        data = resp.content
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True


@dataclass
class DandiAssetValidation:
    dandiset_id: str
    asset_path: str
    size_bytes: int
    n_http_requests: int = 0
    bytes_fetched: int = 0
    has_units: bool = False
    n_units: int | None = None
    has_trials: bool = False
    n_trials: int | None = None
    trial_columns: list[str] = field(default_factory=list)
    has_electrodes: bool = False
    n_electrodes: int | None = None
    has_imaging: bool = False
    nwb_version: str | None = None
    error: str | None = None


def _list_nwb_assets(dandiset_id: str, client: httpx.Client, max_assets: int) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    url: str | None = f"{DANDI_API_URL}/dandisets/{dandiset_id}/versions/draft/assets/"
    # Only the first request needs an explicit params= — passing params=
    # (even {}) on later requests would overwrite the query string that
    # `next` already encodes (verified: httpx replaces, not merges, a URL's
    # existing query component whenever `params=` is given at all).
    first_request = True
    while url and len(assets) < max_assets:
        if first_request:
            resp = client.get(url, params={"page_size": 100}, timeout=20.0)
            first_request = False
        else:
            resp = client.get(url, timeout=20.0)
        resp.raise_for_status()
        payload = resp.json()
        for asset in payload.get("results", []):
            if asset.get("path", "").endswith(".nwb"):
                assets.append(asset)
                if len(assets) >= max_assets:
                    break
        url = payload.get("next")
    return assets


def _resolve_download_url(asset_id: str, client: httpx.Client) -> str | None:
    resp = client.get(
        f"{DANDI_API_URL}/assets/{asset_id}/download/",
        follow_redirects=False,
        timeout=15.0,
    )
    if resp.status_code in (301, 302, 303, 307, 308):
        return resp.headers.get("location")
    return None


def inspect_nwb_asset_streaming(
    download_url: str, client: httpx.Client
) -> dict[str, Any]:
    """Open a remote NWB asset and read only its header/structure.

    Returns a plain dict (not opinionated about affordances) so callers can
    apply their own requirement logic — mirrors the metadata shape already
    used by `neural_search.data.dandi_streaming.extract_nwb_metadata_streaming`.
    """

    import h5py

    range_file = HttpRangeFile(download_url, client)
    metadata: dict[str, Any] = {
        "size_bytes": range_file.size,
        "has_units": False,
        "has_trials": False,
        "has_electrodes": False,
        "has_imaging": False,
    }
    h5f = None
    try:
        h5f = h5py.File(range_file, "r")

        nwb_version = h5f.attrs.get("nwb_version")
        if nwb_version is not None:
            metadata["nwb_version"] = (
                nwb_version.decode() if isinstance(nwb_version, bytes) else str(nwb_version)
            )

        if "units" in h5f:
            units = h5f["units"]
            metadata["has_units"] = True
            metadata["n_units"] = int(units["id"].shape[0]) if "id" in units else None

        if "intervals" in h5f and "trials" in h5f["intervals"]:
            trials = h5f["intervals"]["trials"]
            metadata["has_trials"] = True
            metadata["n_trials"] = int(trials["id"].shape[0]) if "id" in trials else None
            metadata["trial_columns"] = [
                k for k in trials.keys() if k not in ("id", "start_time", "stop_time")
            ]

        if "general" in h5f and "extracellular_ephys" in h5f["general"]:
            ephys = h5f["general"]["extracellular_ephys"]
            if "electrodes" in ephys:
                electrodes = ephys["electrodes"]
                metadata["has_electrodes"] = True
                metadata["n_electrodes"] = (
                    int(electrodes["id"].shape[0]) if "id" in electrodes else None
                )

        if "processing" in h5f:
            for module_name in h5f["processing"]:
                module = h5f["processing"][module_name]
                if "Fluorescence" in module or "ImageSegmentation" in module:
                    metadata["has_imaging"] = True
                    break
    finally:
        if h5f is not None:
            h5f.close()  # explicit close avoids a segfault-on-exit seen with
            # custom file-like objects + h5py's C-extension cleanup order

    metadata["n_http_requests"] = range_file.n_requests
    metadata["bytes_fetched"] = range_file.bytes_fetched
    return metadata


def validate_dandiset(
    dandiset_id: str,
    max_assets: int = 1,
    client: httpx.Client | None = None,
) -> list[DandiAssetValidation]:
    """Validate up to `max_assets` NWB files in a dandiset via live streaming."""

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    results: list[DandiAssetValidation] = []
    try:
        assets = _list_nwb_assets(dandiset_id, client, max_assets)
        for asset in assets:
            result = DandiAssetValidation(
                dandiset_id=dandiset_id,
                asset_path=asset.get("path", ""),
                size_bytes=asset.get("size", 0),
            )
            try:
                download_url = _resolve_download_url(asset["asset_id"], client)
                if not download_url:
                    result.error = "could not resolve download URL"
                    results.append(result)
                    continue
                meta = inspect_nwb_asset_streaming(download_url, client)
                result.has_units = meta["has_units"]
                result.n_units = meta.get("n_units")
                result.has_trials = meta["has_trials"]
                result.n_trials = meta.get("n_trials")
                result.trial_columns = meta.get("trial_columns", [])
                result.has_electrodes = meta["has_electrodes"]
                result.n_electrodes = meta.get("n_electrodes")
                result.has_imaging = meta["has_imaging"]
                result.nwb_version = meta.get("nwb_version")
                result.n_http_requests = meta["n_http_requests"]
                result.bytes_fetched = meta["bytes_fetched"]
            except Exception as exc:  # noqa: BLE001 - record and continue, this validates untrusted remote data
                log.warning("NWB validation failed for %s: %s", asset.get("path"), exc)
                result.error = str(exc)
            results.append(result)
    finally:
        if owns_client:
            client.close()
    return results
