"""Tiny deterministic NWB/BIDS metadata inspectors.

The inspectors intentionally operate on metadata sidecars and small fixture files.
Full NWB/BIDS downloads are out of scope for CI, but the claims emitted here use the
same schema expected from richer local extractors.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neural_search.file_inspection.claims import FileInspectionClaim, make_claim_id

NWB_EXTRACTOR = "neural_search.file_inspection.nwb_fixture"
BIDS_EXTRACTOR = "neural_search.file_inspection.bids_fixture"


def _claim(
    *,
    dataset_id: str,
    claim_type: str,
    field: str,
    value: Any,
    evidence: str,
    source_path: str | Path,
    extractor: str,
    confidence: float = 0.9,
) -> FileInspectionClaim:
    value_text = str(value)
    return FileInspectionClaim(
        claim_id=make_claim_id(dataset_id, field, value_text, source_path),
        dataset_id=dataset_id,
        claim_type=claim_type,  # type: ignore[arg-type]
        field=field,
        value=value_text,
        confidence=confidence,
        evidence=evidence,
        source_path=str(source_path),
        extractor=extractor,
        timestamp="2026-05-24T00:00:00+00:00",
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def inspect_nwb_metadata(path: str | Path, dataset_id: str) -> list[FileInspectionClaim]:
    """Inspect a tiny NWB metadata export and return conservative claims."""

    source_path = Path(path)
    if not source_path.exists():
        return [
            _claim(
                dataset_id=dataset_id,
                claim_type="warning",
                field="missing_file",
                value=source_path,
                evidence="NWB metadata sidecar was not found.",
                source_path=source_path,
                extractor=NWB_EXTRACTOR,
                confidence=1.0,
            )
        ]

    payload = _load_json(source_path)
    nwb = payload.get("nwb", payload)
    claims: list[FileInspectionClaim] = [
        _claim(
            dataset_id=dataset_id,
            claim_type="label",
            field="data_standards",
            value="NWB",
            evidence="NWB metadata sidecar was present.",
            source_path=source_path,
            extractor=NWB_EXTRACTOR,
            confidence=0.98,
        )
    ]

    trials = nwb.get("trials") or {}
    if int(trials.get("count", 0) or 0) > 0:
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="usability",
                field="has_trials",
                value="true",
                evidence=f"trials table count={trials.get('count')}",
                source_path=source_path,
                extractor=NWB_EXTRACTOR,
                confidence=0.95,
            )
        )
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="analysis_affordance",
                field="analysis_affordances",
                value="event_aligned_analysis",
                evidence="trials table supports event-aligned analyses.",
                source_path=source_path,
                extractor=NWB_EXTRACTOR,
                confidence=0.82,
            )
        )

    units = nwb.get("units") or {}
    if int(units.get("count", 0) or 0) > 0:
        claims.extend(
            [
                _claim(
                    dataset_id=dataset_id,
                    claim_type="metadata_presence",
                    field="units",
                    value=str(units.get("count")),
                    evidence=f"units table count={units.get('count')}",
                    source_path=source_path,
                    extractor=NWB_EXTRACTOR,
                    confidence=0.95,
                ),
                _claim(
                    dataset_id=dataset_id,
                    claim_type="analysis_affordance",
                    field="analysis_affordances",
                    value="spike_train_analysis",
                    evidence="units table supports spike-train analyses.",
                    source_path=source_path,
                    extractor=NWB_EXTRACTOR,
                    confidence=0.84,
                ),
            ]
        )

    subject = nwb.get("subject") or {}
    if subject.get("species"):
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="label",
                field="species",
                value=subject["species"],
                evidence="subject.species present in NWB metadata.",
                source_path=source_path,
                extractor=NWB_EXTRACTOR,
                confidence=0.92,
            )
        )

    for field, output_field in [
        ("devices", "recording_devices"),
        ("processing_modules", "processing_modules"),
        ("intervals", "intervals"),
        ("sessions", "sessions"),
        ("acquisition", "acquisition_groups"),
    ]:
        for value in nwb.get(field, []) or []:
            claims.append(
                _claim(
                    dataset_id=dataset_id,
                    claim_type="metadata_presence",
                    field=output_field,
                    value=value,
                    evidence=f"{field} contains {value}",
                    source_path=source_path,
                    extractor=NWB_EXTRACTOR,
                    confidence=0.88,
                )
            )
            lowered = str(value).casefold()
            if "neuropixels" in lowered or "electricalseries" in lowered:
                claims.append(
                    _claim(
                        dataset_id=dataset_id,
                        claim_type="label",
                        field="modalities",
                        value="neuropixels",
                        evidence=f"{field} contains {value}",
                        source_path=source_path,
                        extractor=NWB_EXTRACTOR,
                        confidence=0.85,
                    )
                )
    return claims


def inspect_bids_directory(path: str | Path, dataset_id: str) -> list[FileInspectionClaim]:
    """Inspect a tiny BIDS fixture directory and return conservative claims."""

    root = Path(path)
    if not root.exists():
        return [
            _claim(
                dataset_id=dataset_id,
                claim_type="warning",
                field="missing_file",
                value=root,
                evidence="BIDS directory was not found.",
                source_path=root,
                extractor=BIDS_EXTRACTOR,
                confidence=1.0,
            )
        ]

    claims: list[FileInspectionClaim] = []
    description = root / "dataset_description.json"
    if description.exists():
        payload = _load_json(description)
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="label",
                field="data_standards",
                value="BIDS",
                evidence="dataset_description.json present.",
                source_path=description,
                extractor=BIDS_EXTRACTOR,
                confidence=0.98,
            )
        )
        for value in payload.get("DatasetType", [] if isinstance(payload.get("DatasetType"), list) else [payload.get("DatasetType")]):
            if value:
                claims.append(
                    _claim(
                        dataset_id=dataset_id,
                        claim_type="metadata_presence",
                        field="dataset_type",
                        value=value,
                        evidence="DatasetType present in dataset_description.json.",
                        source_path=description,
                        extractor=BIDS_EXTRACTOR,
                        confidence=0.86,
                    )
                )
        text = json.dumps(payload).casefold()
        if "eeg" in text:
            claims.append(
                _claim(
                    dataset_id=dataset_id,
                    claim_type="label",
                    field="modalities",
                    value="eeg",
                    evidence="dataset description mentions EEG.",
                    source_path=description,
                    extractor=BIDS_EXTRACTOR,
                    confidence=0.86,
                )
            )

    participants = root / "participants.tsv"
    if participants.exists():
        rows = _read_tsv(participants)
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="metadata_presence",
                field="participants",
                value=len(rows),
                evidence=f"participants.tsv contains {len(rows)} rows.",
                source_path=participants,
                extractor=BIDS_EXTRACTOR,
                confidence=0.9,
            )
        )
        if rows:
            species_values = {row.get("species", "").strip() for row in rows if row.get("species")}
            if not species_values:
                species_values = {"human"}
            for species in species_values:
                claims.append(
                    _claim(
                        dataset_id=dataset_id,
                        claim_type="label",
                        field="species",
                        value=species,
                        evidence="participants.tsv provides participant-level metadata.",
                        source_path=participants,
                        extractor=BIDS_EXTRACTOR,
                        confidence=0.84,
                    )
                )

    event_files = sorted(root.glob("**/*_events.tsv"))
    for events_path in event_files:
        rows = _read_tsv(events_path)
        if not rows:
            continue
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="usability",
                field="has_event_timestamps",
                value="true",
                evidence=f"{events_path.name} contains event rows.",
                source_path=events_path,
                extractor=BIDS_EXTRACTOR,
                confidence=0.92,
            )
        )
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="analysis_affordance",
                field="analysis_affordances",
                value="event_aligned_analysis",
                evidence=f"{events_path.name} supports event-aligned analysis.",
                source_path=events_path,
                extractor=BIDS_EXTRACTOR,
                confidence=0.82,
            )
        )
        trial_types = {row.get("trial_type", "").strip() for row in rows if row.get("trial_type")}
        for trial_type in sorted(trial_types):
            claims.append(
                _claim(
                    dataset_id=dataset_id,
                    claim_type="label",
                    field="behavioral_events",
                    value=trial_type,
                    evidence=f"events.tsv trial_type includes {trial_type}.",
                    source_path=events_path,
                    extractor=BIDS_EXTRACTOR,
                    confidence=0.80,
                )
            )

    for channels_path in sorted(root.glob("**/*_channels.tsv")):
        rows = _read_tsv(channels_path)
        channel_types = {row.get("type", "").strip().lower() for row in rows if row.get("type")}
        if "eeg" in channel_types:
            claims.append(
                _claim(
                    dataset_id=dataset_id,
                    claim_type="label",
                    field="modalities",
                    value="eeg",
                    evidence=f"{channels_path.name} includes EEG channels.",
                    source_path=channels_path,
                    extractor=BIDS_EXTRACTOR,
                    confidence=0.9,
                )
            )

    if (root / "derivatives").exists():
        claims.append(
            _claim(
                dataset_id=dataset_id,
                claim_type="metadata_presence",
                field="derivatives",
                value="present",
                evidence="BIDS derivatives directory present.",
                source_path=root / "derivatives",
                extractor=BIDS_EXTRACTOR,
                confidence=0.8,
            )
        )
    return claims


def inspect_dataset_files(
    inspection_paths: Iterable[str | Path],
    dataset_id: str,
) -> list[FileInspectionClaim]:
    """Inspect known fixture file types for one dataset."""

    claims: list[FileInspectionClaim] = []
    for raw_path in inspection_paths:
        path = Path(raw_path)
        name = path.name.casefold()
        if path.is_dir() or name == "dataset_description.json":
            root = path if path.is_dir() else path.parent
            claims.extend(inspect_bids_directory(root, dataset_id))
        elif name.endswith(".nwb.json") or "nwb" in name:
            claims.extend(inspect_nwb_metadata(path, dataset_id))
        elif path.exists() and path.suffix == ".json":
            claims.extend(inspect_nwb_metadata(path, dataset_id))
        else:
            claims.append(
                _claim(
                    dataset_id=dataset_id,
                    claim_type="warning",
                    field="unsupported_path",
                    value=path,
                    evidence="No deterministic inspector is registered for this path.",
                    source_path=path,
                    extractor="neural_search.file_inspection.dispatch",
                    confidence=1.0,
                )
            )
    return claims


def write_claims_jsonl(
    claims: Iterable[FileInspectionClaim],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(claim.model_dump_json())
            handle.write("\n")
    return output
