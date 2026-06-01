"""DANDI streaming access for NWB files.

This module provides streaming access to DANDI NWB files without
downloading entire datasets. Uses remfile for efficient byte-range
requests.

Usage:
    from neural_search.data.dandi_streaming import (
        stream_nwb_file,
        list_dandiset_assets,
        get_dandiset_metadata,
        extract_signature_streaming,
    )

    # Get dandiset metadata (no download)
    meta = get_dandiset_metadata("000003")

    # List NWB assets in a dandiset
    assets = list_dandiset_assets("000003")

    # Stream an NWB file and extract signature
    signature = extract_signature_streaming("000003", assets[0]["asset_id"])

Requirements:
    pip install dandi remfile pynwb h5py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DANDIAsset:
    """Represents an NWB asset in DANDI."""

    dandiset_id: str
    asset_id: str
    path: str
    size_bytes: int
    content_url: str | None = None

    @property
    def dataset_id(self) -> str:
        return f"dandi:{self.dandiset_id}"


def get_dandiset_metadata(dandiset_id: str, version: str = "draft") -> dict[str, Any]:
    """Get dandiset metadata from DANDI API.

    Args:
        dandiset_id: DANDI ID (e.g., "000003")
        version: Version to fetch ("draft" or specific version)

    Returns:
        Dandiset metadata dictionary
    """
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id, version)
        return dandiset.get_raw_metadata()


def list_dandiset_assets(
    dandiset_id: str,
    version: str = "draft",
    max_assets: int | None = None,
) -> list[DANDIAsset]:
    """List NWB assets in a dandiset.

    Args:
        dandiset_id: DANDI ID (e.g., "000003")
        version: Version to list
        max_assets: Maximum number of assets to return

    Returns:
        List of DANDIAsset objects
    """
    from dandi.dandiapi import DandiAPIClient

    assets = []

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id, version)

        for i, asset in enumerate(dandiset.get_assets()):
            if max_assets and i >= max_assets:
                break

            if not asset.path.endswith(".nwb"):
                continue

            # Get download URL
            try:
                content_url = asset.get_content_url(follow_redirects=1, strip_query=True)
            except Exception:
                content_url = None

            assets.append(DANDIAsset(
                dandiset_id=dandiset_id,
                asset_id=asset.identifier,
                path=asset.path,
                size_bytes=asset.size,
                content_url=content_url,
            ))

    return assets


def stream_nwb_file(asset: DANDIAsset):
    """Open an NWB file for streaming access.

    Args:
        asset: DANDIAsset to stream

    Returns:
        pynwb.NWBFile object (opened for reading)

    Example:
        asset = list_dandiset_assets("000003")[0]
        with stream_nwb_file(asset) as nwbfile:
            print(nwbfile.units)
    """
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    if not asset.content_url:
        raise ValueError(f"No content URL for asset {asset.asset_id}")

    # Create streaming file handle
    remote_file = remfile.File(asset.content_url)
    h5_file = h5py.File(remote_file, "r")

    # Open with pynwb
    io = NWBHDF5IO(file=h5_file, load_namespaces=True)
    return io.read()


def extract_nwb_metadata_streaming(asset: DANDIAsset) -> dict[str, Any]:
    """Extract metadata from NWB file via streaming.

    Only reads the metadata/headers, not the actual data arrays.
    Much faster than downloading the whole file.

    Args:
        asset: DANDIAsset to read

    Returns:
        Dictionary with extracted metadata
    """
    import h5py
    import remfile

    if not asset.content_url:
        raise ValueError(f"No content URL for asset {asset.asset_id}")

    metadata: dict[str, Any] = {
        "asset_id": asset.asset_id,
        "dandiset_id": asset.dandiset_id,
        "path": asset.path,
        "size_bytes": asset.size_bytes,
    }

    try:
        remote_file = remfile.File(asset.content_url)
        with h5py.File(remote_file, "r") as f:
            # Check for units table
            if "units" in f:
                units = f["units"]
                metadata["has_units"] = True
                metadata["n_units"] = units["id"].shape[0] if "id" in units else 0

                # Check for spike times
                if "spike_times" in units:
                    metadata["has_spike_times"] = True

                # Check for brain regions
                if "location" in units:
                    try:
                        locations = units["location"][:]
                        if hasattr(locations[0], "decode"):
                            locations = [loc.decode() for loc in locations]
                        metadata["brain_regions"] = list(set(locations))
                    except Exception:
                        pass
            else:
                metadata["has_units"] = False

            # Check for trials
            if "intervals" in f and "trials" in f["intervals"]:
                trials = f["intervals"]["trials"]
                metadata["has_trials"] = True
                metadata["n_trials"] = trials["id"].shape[0] if "id" in trials else 0

                # Check trial columns
                metadata["trial_columns"] = list(trials.keys())
            else:
                metadata["has_trials"] = False

            # Check for electrodes
            if "general" in f and "extracellular_ephys" in f["general"]:
                ephys = f["general"]["extracellular_ephys"]
                if "electrodes" in ephys:
                    electrodes = ephys["electrodes"]
                    metadata["has_electrodes"] = True
                    metadata["n_electrodes"] = electrodes["id"].shape[0] if "id" in electrodes else 0

            # Check for imaging
            if "processing" in f:
                for mod_name in f["processing"]:
                    mod = f["processing"][mod_name]
                    if "ImageSegmentation" in mod or "Fluorescence" in mod:
                        metadata["has_imaging"] = True
                        break

            # Session info
            if "session_description" in f:
                try:
                    desc = f["session_description"][()]
                    if isinstance(desc, bytes):
                        desc = desc.decode()
                    metadata["session_description"] = desc[:500]
                except Exception:
                    pass

    except Exception as e:
        metadata["streaming_error"] = str(e)
        logger.warning(f"Error streaming {asset.path}: {e}")

    return metadata


def extract_signature_streaming(
    dandiset_id: str,
    asset_id: str,
) -> "NeuralSignatureV1":
    """Extract a NeuralSignatureV1 from a streaming NWB file.

    Args:
        dandiset_id: DANDI ID
        asset_id: Asset identifier

    Returns:
        NeuralSignatureV1 with extracted features
    """
    from neural_search.core.neural_signature import (
        NeuralSignatureV1,
        RecordingModality,
        SignatureQuality,
        TrialStats,
    )

    # Get asset info
    assets = list_dandiset_assets(dandiset_id, max_assets=100)
    asset = next((a for a in assets if a.asset_id == asset_id), None)

    if not asset:
        raise ValueError(f"Asset {asset_id} not found in dandiset {dandiset_id}")

    # Extract metadata via streaming
    meta = extract_nwb_metadata_streaming(asset)

    # Determine modality
    modality = RecordingModality.UNKNOWN
    if meta.get("has_units"):
        modality = RecordingModality.EPHYS
    elif meta.get("has_imaging"):
        modality = RecordingModality.CALCIUM_IMAGING

    # Build trial stats
    trial_stats = None
    if meta.get("has_trials"):
        trial_stats = TrialStats(
            n_trials=meta.get("n_trials"),
            event_types=meta.get("trial_columns", []),
        )

    return NeuralSignatureV1(
        dataset_id=f"dandi:{dandiset_id}",
        asset_id=asset_id,
        modality=modality,
        n_units=meta.get("n_units"),
        n_electrodes=meta.get("n_electrodes"),
        brain_regions=meta.get("brain_regions", []),
        trial_stats=trial_stats,
        quality=SignatureQuality.HIGH,
        extractor_notes=[
            f"Extracted via streaming from {asset.path}",
            f"File size: {asset.size_bytes / 1e6:.1f} MB",
        ],
    )


def scan_dandiset_for_affordances(
    dandiset_id: str,
    max_assets: int = 10,
) -> list[dict[str, Any]]:
    """Scan a dandiset to assess affordance support.

    Streams metadata from multiple NWB files to understand
    what analyses the dataset supports.

    Args:
        dandiset_id: DANDI ID
        max_assets: Maximum assets to scan

    Returns:
        List of asset metadata dictionaries
    """
    logger.info(f"Scanning dandiset {dandiset_id}...")

    assets = list_dandiset_assets(dandiset_id, max_assets=max_assets)
    logger.info(f"Found {len(assets)} NWB assets")

    results = []
    for i, asset in enumerate(assets):
        logger.info(f"  [{i+1}/{len(assets)}] {asset.path}")
        meta = extract_nwb_metadata_streaming(asset)
        results.append(meta)

    return results


# Convenience function for batch processing
def scan_multiple_dandisets(
    dandiset_ids: list[str],
    assets_per_dandiset: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Scan multiple dandisets for metadata.

    Args:
        dandiset_ids: List of DANDI IDs
        assets_per_dandiset: Assets to scan per dandiset

    Returns:
        Dictionary mapping dandiset_id to list of asset metadata
    """
    results = {}

    for dandiset_id in dandiset_ids:
        try:
            results[dandiset_id] = scan_dandiset_for_affordances(
                dandiset_id,
                max_assets=assets_per_dandiset,
            )
        except Exception as e:
            logger.error(f"Failed to scan {dandiset_id}: {e}")
            results[dandiset_id] = []

    return results
