"""Source resolvers: adapters from dataset identity to ``DatasetIntrospectionV1``.

Resolvers never claim more than they can support. The metadata-only
resolver works for every dataset regardless of source and never inspects
file bytes; it only reads whatever the dataset record and dataset card
already carry (modalities, experimental structure, listed assets).

The DANDI/NWB and OpenNeuro/BIDS resolvers below read real file-level
evidence (streamed NWB headers, local BIDS sidecar/events files) and mark
what they find as ``file_derived``. Both degrade to the metadata-only
resolver on any failure -- missing fixture, network error, missing optional
dependency -- so a scene can always be produced.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_search.experimentglancer.schemas import ClockKind, DatasetIntrospectionV1

log = logging.getLogger(__name__)

# Local BIDS fixtures shipped with this repo (see
# data/corpus/fixtures/real_v07/bids/ds003505/ for the one currently present).
_BIDS_FIXTURES_ROOT = Path("data/corpus/fixtures/real_v07/bids")

_ASSET_MODALITY_TO_LAYER_HINT: dict[str, str] = {
    "video": "video.frames",
    "behavior_video": "video.frames",
    "pose": "behavior.pose",
    "calcium_imaging": "neural.calcium",
    "two_photon": "neural.calcium",
    "ophys": "neural.calcium",
    "electrophysiology": "neural.spikes",
    "ephys": "neural.spikes",
    "neuropixels": "neural.spikes",
    "units": "neural.spikes",
}


def _clock_for_data_standard(data_standard: str | None) -> ClockKind:
    standard = (data_standard or "").upper()
    if standard == "NWB":
        return "nwb_time_seconds"
    if standard == "BIDS":
        return "bids_events_seconds"
    return "metadata_only"


def resolve_metadata_only(
    dataset: Mapping[str, Any],
    dataset_card: Mapping[str, Any] | None = None,
) -> DatasetIntrospectionV1:
    """Build a ``DatasetIntrospectionV1`` from metadata alone.

    This is deliberately conservative: it reports what the dataset card
    *claims* (trial/event column names, listed assets, modalities) as
    ``metadata_inferred`` evidence. It never marks anything ``file_derived``
    -- that requires an NWB/BIDS resolver that actually reads bytes.
    """

    card = dataset_card or {}
    dataset_id = str(dataset.get("dataset_id") or dataset.get("id") or card.get("dataset_id") or "unknown")
    source = dataset.get("source") or card.get("source")
    data_standard = dataset.get("data_standard") or card.get("data_standard")

    modalities = list(
        dict.fromkeys([*dataset.get("modalities", []), *card.get("modalities", [])])
    )

    experimental_structure = card.get("experimental_structure") or {}
    trial_event_structure = list(experimental_structure.get("trial_event_structure", []))
    sessions_value = experimental_structure.get("sessions")
    subjects_value = experimental_structure.get("subjects")

    trial_columns = (
        [c for c in trial_event_structure if "trial" in c]
        if any("trial" in c for c in trial_event_structure)
        else (["trial_id"] if sessions_value or subjects_value else [])
    )
    event_columns = list(trial_event_structure)

    neural_data = card.get("neural_data") or {}
    assets = list(neural_data.get("available_assets", []))

    detected_layers: list[str] = []
    missing_layer_requirements: list[str] = []
    for asset in assets:
        modality = str(asset.get("modality") or "").lower()
        layer_hint = _ASSET_MODALITY_TO_LAYER_HINT.get(modality)
        if layer_hint and layer_hint not in detected_layers:
            detected_layers.append(layer_hint)

    if any(m in modalities for m in ("electrophysiology", "ephys", "neuropixels")) and not any(
        str(a.get("modality", "")).lower() in ("units", "electrophysiology", "ephys", "neuropixels")
        for a in assets
    ):
        missing_layer_requirements.append("spike-sorted units asset not listed in dataset metadata")

    return DatasetIntrospectionV1(
        dataset_id=dataset_id,
        source=source,
        resolver="metadata_only",
        assets=assets,
        clocks=[_clock_for_data_standard(data_standard)],
        sessions=[str(sessions_value)] if sessions_value else [],
        subjects=[str(subjects_value)] if subjects_value else [],
        trial_columns=sorted(set(trial_columns)),
        event_columns=sorted(set(event_columns)),
        available_modalities=modalities,
        detected_layers=detected_layers,
        missing_layer_requirements=missing_layer_requirements,
        source_warnings=["Metadata-only resolver: no file bytes were inspected."],
    )


def _dandiset_id_from(dataset: Mapping[str, Any], dataset_card: Mapping[str, Any]) -> str | None:
    source_id = dataset.get("source_id") or dataset_card.get("source_id")
    if source_id:
        return str(source_id)
    dataset_id = str(dataset.get("dataset_id") or dataset.get("id") or "")
    match = re.search(r"(\d{6})", dataset_id)
    return match.group(1) if match else None


def resolve_dandi_nwb(
    dataset: Mapping[str, Any],
    dataset_card: Mapping[str, Any] | None = None,
    *,
    max_assets: int = 3,
) -> DatasetIntrospectionV1:
    """Lightweight DANDI/NWB introspection via streaming metadata.

    Never downloads full files -- only streams NWB headers via
    ``neural_search.data.dandi_streaming``. This does real network I/O and
    needs optional heavy dependencies (``dandi``, ``remfile``, ``pynwb``,
    ``h5py``), so callers should only reach for it when file-derived
    evidence is explicitly requested (see ``resolve_dataset_introspection``'s
    ``deep`` flag) -- never as a default/automatic path.
    """

    card = dataset_card or {}
    fallback = resolve_metadata_only(dataset, card)
    dandiset_id = _dandiset_id_from(dataset, card)
    if dandiset_id is None:
        return fallback

    try:
        from neural_search.data.dandi_streaming import (
            extract_nwb_metadata_streaming,
            list_dandiset_assets,
        )

        assets = list_dandiset_assets(dandiset_id, max_assets=max_assets)
    except Exception as exc:  # noqa: BLE001 - network/optional-dependency failures degrade gracefully
        log.warning("DANDI streaming introspection unavailable for %s: %s", dandiset_id, exc)
        fallback.source_warnings.append(
            f"DANDI streaming introspection unavailable ({exc}); using metadata-only introspection."
        )
        return fallback

    if not assets:
        fallback.source_warnings.append(
            f"No NWB assets listed for dandiset {dandiset_id}; using metadata-only introspection."
        )
        return fallback

    detected_layers: list[str] = []
    trial_columns: set[str] = set()
    missing: list[str] = []
    warnings: list[str] = []
    asset_records: list[dict[str, Any]] = []

    for asset in assets:
        asset_records.append({"id": asset.asset_id, "path": asset.path, "size_bytes": asset.size_bytes})
        try:
            meta = extract_nwb_metadata_streaming(asset)
        except Exception as exc:  # noqa: BLE001 - a single bad asset shouldn't fail the whole scene
            warnings.append(f"Could not stream metadata for {asset.path}: {exc}")
            continue

        if meta.get("streaming_error"):
            warnings.append(f"{asset.path}: {meta['streaming_error']}")
            continue

        if meta.get("has_units"):
            if "neural.spikes" not in detected_layers:
                detected_layers.append("neural.spikes")
            if not meta.get("has_spike_times"):
                missing.append(f"{asset.path}: units table has no spike_times column")
        if meta.get("has_imaging"):
            if "neural.calcium" not in detected_layers:
                detected_layers.append("neural.calcium")
        if meta.get("has_trials"):
            if "timeline.trials" not in detected_layers:
                detected_layers.append("timeline.trials")
            trial_columns.update(meta.get("trial_columns", []))

    if not detected_layers:
        fallback.source_warnings.append(
            f"Streamed {len(assets)} NWB asset(s) for dandiset {dandiset_id} but found no "
            "units/trials/imaging structure; using metadata-only introspection."
        )
        return fallback

    return DatasetIntrospectionV1(
        dataset_id=fallback.dataset_id,
        source="dandi",
        resolver="dandi_nwb_streaming",
        assets=asset_records,
        clocks=["nwb_time_seconds"],
        sessions=fallback.sessions,
        subjects=fallback.subjects,
        trial_columns=sorted(trial_columns),
        event_columns=sorted(trial_columns),
        available_modalities=fallback.available_modalities,
        detected_layers=detected_layers,
        missing_layer_requirements=missing,
        source_warnings=warnings,
    )


def _openneuro_accession_from(dataset: Mapping[str, Any], dataset_card: Mapping[str, Any]) -> str | None:
    source_id = dataset.get("source_id") or dataset_card.get("source_id")
    if source_id and re.fullmatch(r"ds\d+", str(source_id), re.IGNORECASE):
        return str(source_id).lower()
    dataset_id = str(dataset.get("dataset_id") or dataset.get("id") or "")
    match = re.search(r"(ds\d{6})", dataset_id, re.IGNORECASE)
    return match.group(1).lower() if match else None


def resolve_openneuro_bids_local(
    dataset: Mapping[str, Any],
    dataset_card: Mapping[str, Any] | None = None,
    *,
    fixtures_root: Path = _BIDS_FIXTURES_ROOT,
) -> DatasetIntrospectionV1:
    """Introspect a locally-available BIDS fixture directory.

    Only reads files this repo already ships under
    ``data/corpus/fixtures/real_v07/bids/<accession>/``; it never fetches
    anything over the network. Falls back to metadata-only when no local
    fixture exists for the dataset.
    """

    card = dataset_card or {}
    fallback = resolve_metadata_only(dataset, card)
    accession = _openneuro_accession_from(dataset, card)
    if accession is None:
        return fallback

    dataset_dir = fixtures_root / accession
    if not dataset_dir.is_dir():
        fallback.source_warnings.append(
            f"No local BIDS fixture found at {dataset_dir}; using metadata-only introspection."
        )
        return fallback

    event_columns: set[str] = set()
    trial_columns: set[str] = set()
    subjects: set[str] = set()
    detected_layers: list[str] = []
    assets: list[dict[str, Any]] = []

    for events_path in sorted(dataset_dir.rglob("*_events.tsv")):
        assets.append({"path": str(events_path.relative_to(fixtures_root)), "asset_type": "events"})
        header = events_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        event_columns.update(header)
        if "trial_type" in header:
            trial_columns.add("trial_type")
        if "timeline.events" not in detected_layers:
            detected_layers.append("timeline.events")

    if trial_columns and "timeline.trials" not in detected_layers:
        detected_layers.append("timeline.trials")

    for channels_path in sorted(dataset_dir.rglob("*_channels.tsv")):
        assets.append({"path": str(channels_path.relative_to(fixtures_root)), "asset_type": "channels"})
        if "neural.lfp" not in detected_layers:
            detected_layers.append("neural.lfp")

    for sidecar_path in sorted(dataset_dir.rglob("*_*eg.json")):
        assets.append({"path": str(sidecar_path.relative_to(fixtures_root)), "asset_type": "sidecar"})

    for sub_dir in sorted(dataset_dir.glob("sub-*")):
        if sub_dir.is_dir():
            subjects.add(sub_dir.name)

    participants_path = dataset_dir / "participants.tsv"
    if participants_path.exists():
        lines = participants_path.read_text(encoding="utf-8").splitlines()[1:]
        subjects.update(line.split("\t")[0] for line in lines if line.strip())

    if not detected_layers:
        fallback.source_warnings.append(
            f"BIDS fixture at {dataset_dir} had no recognizable events/channels files; "
            "using metadata-only introspection."
        )
        return fallback

    return DatasetIntrospectionV1(
        dataset_id=fallback.dataset_id,
        source="openneuro",
        resolver="openneuro_bids_local",
        assets=assets,
        clocks=["bids_events_seconds"],
        sessions=[],
        subjects=sorted(subjects),
        trial_columns=sorted(trial_columns),
        event_columns=sorted(event_columns),
        available_modalities=fallback.available_modalities,
        detected_layers=detected_layers,
        missing_layer_requirements=[],
        source_warnings=[],
    )


def resolve_dataset_introspection(
    dataset: Mapping[str, Any],
    dataset_card: Mapping[str, Any] | None = None,
    *,
    deep: bool = False,
) -> DatasetIntrospectionV1:
    """Pick the best resolver for a dataset's source and always return a scene-ready result.

    The OpenNeuro/BIDS local resolver is safe to run by default -- it only
    reads local fixture files this repo ships, so it's always attempted for
    ``source == "openneuro"``. The DANDI/NWB streaming resolver does real
    network I/O, so it only runs when ``deep=True`` is explicitly requested;
    otherwise DANDI datasets get the metadata-only resolver.
    """

    card = dataset_card or {}
    source = str(dataset.get("source") or card.get("source") or "").lower()
    if source == "openneuro":
        return resolve_openneuro_bids_local(dataset, card)
    if source == "dandi" and deep:
        return resolve_dandi_nwb(dataset, card)
    return resolve_metadata_only(dataset, card)
