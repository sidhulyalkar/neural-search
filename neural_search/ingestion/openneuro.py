"""OpenNeuro connector for ingesting BIDS datasets."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)
from neural_search.ingestion.registry import register
from neural_search.normalized import (
    evidence_label_from_extraction,
    stable_normalized_id,
)
from neural_search.schemas import NormalizedDatasetRecord, UsabilityFlags

OPENNEURO_API_URL = "https://openneuro.org/crn/graphql"

logger = logging.getLogger(__name__)


def _paginated_query() -> str:
    return """
    query SearchDatasets($first: Int, $after: String) {
        datasets(first: $first, after: $after, filterBy: {public: true}) {
            edges {
                cursor
                node {
                    id
                    name
                    created
                    public
                    latestSnapshot {
                        tag
                        created
                        size
                        readme
                        summary {
                            subjects
                            tasks
                            modalities
                        }
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """


def _search_query() -> str:
    return """
    query SearchDatasets($modality: String, $first: Int) {
        datasets(first: $first, modality: $modality, filterBy: {public: true}) {
            edges {
                node {
                    id
                    name
                    created
                    public
                    latestSnapshot {
                        tag
                        created
                        size
                        readme
                        summary {
                            subjects
                            tasks
                            modalities
                        }
                    }
                }
            }
        }
    }
    """


def fetch_openneuro(modality: str | None, limit: int) -> dict[str, Any]:
    """Fetch datasets from OpenNeuro, optionally filtered by modality.

    Args:
        modality: BIDS modality to filter by (e.g., 'eeg', 'func', 'anat', 'meg', 'ieeg')
                  Pass None to get all public datasets.
        limit: Maximum number of datasets to fetch.
    """
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.post(
            OPENNEURO_API_URL,
            json={
                "query": _search_query(),
                "variables": {"modality": modality, "first": limit},
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data


def fetch_all_openneuro(
    *,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch all public OpenNeuro datasets using GraphQL cursor pagination."""
    all_records: list[dict[str, Any]] = []
    after: str | None = None

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        while True:
            resp = client.post(
                OPENNEURO_API_URL,
                json={
                    "query": _paginated_query(),
                    "variables": {"first": page_size, "after": after},
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("errors"):
                # OpenNeuro returns partial results alongside errors (e.g. a single
                # broken latestSnapshot). Log and continue; don't abort the crawl.
                logger.warning("OpenNeuro GraphQL partial error: %s", data["errors"][:2])

            datasets_data = data.get("data", {}).get("datasets", {})
            edges = datasets_data.get("edges", [])
            page_info = datasets_data.get("pageInfo", {})

            for edge in edges:
                edge = edge or {}
                node = edge.get("node") or {}
                if node.get("id"):
                    all_records.append(normalize_openneuro_dataset(node))
                    if max_records is not None and len(all_records) >= max_records:
                        return all_records

            logger.info("OpenNeuro harvest: %d records, hasNextPage=%s", len(all_records), page_info.get("hasNextPage"))

            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")

    return all_records


async def search_datasets(
    query: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Search OpenNeuro for datasets.

    Args:
        query: Search query string.
        limit: Maximum number of results.

    Returns:
        List of dataset records.
    """
    data = fetch_openneuro(query, limit)
    return records_from_response(data, limit)


# Map OpenNeuro vocabulary terms to canonical modality IDs
_OPENNEURO_MODALITY_MAP: dict[str, str] = {
    "mri": "fmri",
    "fmri": "fmri",
    "bold": "fmri",
    "functional mri": "fmri",
    "t1w": "structural_mri",
    "structural mri": "structural_mri",
    "dwi": "diffusion_mri",
    "diffusion": "diffusion_mri",
    "eeg": "eeg",
    "ecog": "ecog",
    "ieeg": "ieeg",
    "meg": "meg",
    "pet": "pet",
    "ct": "ct",
    "nirs": "fNIRS",
    "fnirs": "fNIRS",
    "beh": "behavioral",
    "behavioral": "behavioral",
    "motion": "motion_capture",
    "eye_tracking": "eye_tracking",
    "eyetracking": "eye_tracking",
}

# Map OpenNeuro species names to canonical IDs
_OPENNEURO_SPECIES_MAP: dict[str, str] = {
    "homo sapiens": "human",
    "human": "human",
    "mus musculus": "mouse",
    "mouse": "mouse",
    "rattus norvegicus": "rat",
    "rat": "rat",
    "macaca mulatta": "macaque",
    "rhesus macaque": "macaque",
    "macaque": "macaque",
}


def _map_openneuro_modalities(raw_modalities: list[str]) -> list[str]:
    """Normalize OpenNeuro modality strings to canonical IDs."""
    result: list[str] = []
    seen: set[str] = set()
    for m in raw_modalities:
        canonical = _OPENNEURO_MODALITY_MAP.get(m.lower().strip())
        if canonical and canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
        elif m.lower().strip() not in seen:
            # Keep original if not in map (may still be useful)
            result.append(m.lower().strip())
            seen.add(m.lower().strip())
    return result


# Map BIDS task names AND ontology task IDs to brain regions likely recruited.
# Keys are lowercased/stripped/cleaned strings; values are lists of canonical region IDs.
_OPENNEURO_TASK_REGION_MAP: dict[str, list[str]] = {
    # Episodic memory / spatial navigation
    "memory": ["hippocampus"],
    "memorytask": ["hippocampus"],
    "encoding": ["hippocampus"],
    "retrieval": ["hippocampus"],
    "autobiographicalmemory": ["hippocampus", "mPFC"],
    "prospection": ["hippocampus", "mPFC"],
    "spatial": ["hippocampus", "entorhinal_cortex"],
    "navigation": ["hippocampus", "entorhinal_cortex"],
    "spatialnavigation": ["hippocampus", "entorhinal_cortex"],
    "virtualnavigation": ["hippocampus", "entorhinal_cortex"],
    "contextualfear": ["hippocampus", "amygdala"],
    # Working memory / executive
    "workingmemory": ["dlPFC", "hippocampus"],
    "working_memory": ["dlPFC", "hippocampus"],
    "nback": ["dlPFC"],
    "stroop": ["ACC", "dlPFC"],
    "stroop_task": ["ACC", "dlPFC"],
    "taskswitch": ["dlPFC", "ACC"],
    "flanker": ["ACC"],
    "flanker_task": ["ACC"],
    "inhibition": ["dlPFC", "ACC"],
    "stop_signal_task": ["dlPFC", "ACC"],
    "stopsignaltask": ["dlPFC", "ACC"],
    "go_nogo": ["ACC", "dlPFC"],
    "gonogo": ["ACC", "dlPFC"],
    # Reward / value
    "reward": ["nucleus_accumbens", "OFC"],
    "monetary": ["nucleus_accumbens", "OFC"],
    "gambling": ["OFC", "nucleus_accumbens"],
    "decision": ["OFC", "dlPFC"],
    "rewardlearning": ["striatum", "OFC"],
    "monetaryincentivedelay": ["nucleus_accumbens"],
    "mid": ["nucleus_accumbens"],
    "value_based_decision_making": ["OFC", "striatum"],
    "valuebaseddecisionmaking": ["OFC", "striatum"],
    "delay_discounting": ["OFC", "striatum"],
    "delaydiscounting": ["OFC", "striatum"],
    "probability_discounting": ["OFC", "striatum"],
    "foraging": ["striatum", "OFC"],
    # Motor
    "motor": ["motor_cortex"],
    "motor_imagery": ["motor_cortex", "premotor_cortex"],
    "motorimagery": ["motor_cortex", "premotor_cortex"],
    "fingertapping": ["motor_cortex", "cerebellum"],
    "tapping": ["motor_cortex"],
    "movement": ["motor_cortex", "premotor_cortex"],
    "sequencelearning": ["motor_cortex", "striatum"],
    "handmotor": ["motor_cortex"],
    "reaching": ["motor_cortex", "posterior_parietal_cortex"],
    "locomotion": ["motor_cortex", "cerebellum"],
    # Visual / perception
    "visual": ["visual_cortex"],
    "faceperception": ["temporal_cortex"],
    "objectrecognition": ["temporal_cortex"],
    "object_recognition": ["temporal_cortex"],
    "facerecognition": ["temporal_cortex"],
    "face_processing": ["temporal_cortex", "amygdala"],
    "faceprocessing": ["temporal_cortex", "amygdala"],
    "faces": ["temporal_cortex", "amygdala"],
    "objects": ["temporal_cortex"],
    "multisensory_integration": ["temporal_cortex", "parietal_cortex"],
    "oddball": ["temporal_cortex", "ACC"],
    "mismatch_negativity": ["temporal_cortex"],
    "missmatchnegativity": ["temporal_cortex"],
    "natural_movie_viewing": ["visual_cortex", "temporal_cortex"],
    "movieviewing": ["visual_cortex", "temporal_cortex"],
    # Auditory
    "auditory": ["auditory_cortex"],
    "auditory_processing": ["auditory_cortex"],
    "auditoryprocessing": ["auditory_cortex"],
    "sentencelistening": ["auditory_cortex", "temporal_cortex"],
    # Language / speech
    "language": ["temporal_cortex"],
    "language_comprehension": ["temporal_cortex"],
    "languagecomprehension": ["temporal_cortex"],
    "speech": ["temporal_cortex", "broca_area"],
    "words": ["temporal_cortex"],
    "reading": ["temporal_cortex"],
    "semantic": ["temporal_cortex"],
    # Emotion / social
    "emotion": ["amygdala"],
    "fear": ["amygdala"],
    "emotionregulation": ["amygdala", "mPFC"],
    "social": ["mPFC", "temporal_cortex"],
    "social_interaction": ["mPFC", "temporal_cortex"],
    "socialinteraction": ["mPFC", "temporal_cortex"],
    "theory_of_mind": ["mPFC", "temporal_cortex"],
    "theoryofmind": ["mPFC", "temporal_cortex"],
    # Attention
    "attention": ["parietal_cortex", "ACC"],
    "spatialattention": ["parietal_cortex"],
    "pupil_arousal": ["locus_coeruleus"],
    # Pain / interoception
    "pain": ["insula", "ACC"],
    "pain_task": ["insula", "ACC"],
    "paintask": ["insula", "ACC"],
    "heat": ["insula"],
    "interoception": ["insula"],
}


def _map_openneuro_tasks_to_regions(tasks_raw: list[str]) -> list[str]:
    """Map OpenNeuro/BIDS task names to canonical brain region IDs."""
    canonical: set[str] = set()
    for task in tasks_raw:
        key = str(task).lower().strip().replace("-", "").replace("_", "").replace(" ", "")
        # Try exact match on cleaned key
        if key in _OPENNEURO_TASK_REGION_MAP:
            canonical.update(_OPENNEURO_TASK_REGION_MAP[key])
            continue
        # Try raw cleaned key directly
        raw_key = str(task).lower().strip()
        if raw_key in _OPENNEURO_TASK_REGION_MAP:
            canonical.update(_OPENNEURO_TASK_REGION_MAP[raw_key])
            continue
        # Substring match for compound task names
        for map_key, regions in _OPENNEURO_TASK_REGION_MAP.items():
            if map_key in raw_key or map_key in key:
                canonical.update(regions)
                break
    return sorted(canonical)


def _map_openneuro_species(raw_species: list[str] | str | None) -> list[str]:
    """Normalize OpenNeuro species strings to canonical IDs."""
    if not raw_species:
        return []
    if isinstance(raw_species, str):
        raw_species = [raw_species]
    result: list[str] = []
    seen: set[str] = set()
    for s in raw_species:
        canonical = _OPENNEURO_SPECIES_MAP.get(s.lower().strip())
        if canonical and canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def normalize_openneuro_dataset(node: dict[str, Any]) -> dict[str, Any]:
    snapshot = node.get("latestSnapshot", {}) or {}
    summary = snapshot.get("summary", {}) or {}
    on_metadata = node.get("metadata", {}) or {}
    title_raw = node.get("name") or node.get("id")
    snap_desc = snapshot.get("description") or {}
    # Prefer snapshot description name as title when it's more informative
    snap_name = snap_desc.get("Name") if isinstance(snap_desc, dict) else None
    title = (snap_name or title_raw or node.get("id"))
    description = node.get("description") or snapshot.get("readme")

    # Merge modalities from multiple sources, preferring summary > metadata
    raw_modalities_summary = [str(v).lower() for v in (summary.get("modalities") or [])]
    raw_modalities_meta = [str(v).lower() for v in (on_metadata.get("modalities") or [])]
    raw_modalities = raw_modalities_summary or raw_modalities_meta
    modalities = _map_openneuro_modalities(raw_modalities)

    # Species from metadata
    raw_species = on_metadata.get("species")
    meta_species = _map_openneuro_species(raw_species)

    # DOI from associatedPaperDOI / openneuroPaperDOI
    doi_sources: list[str] = []
    for doi_key in ("associatedPaperDOI", "openneuroPaperDOI"):
        doi_val = on_metadata.get(doi_key)
        if doi_val and isinstance(doi_val, str) and doi_val.strip():
            doi_sources.append(doi_val.strip())

    text = " ".join(str(part) for part in [title, description, summary, modalities] if part)
    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={"summary": summary, "modalities": modalities, "standard": "BIDS"},
        linked_paper_abstracts=[],
    )

    # Merge species: metadata-derived + extractor
    species_ids: list[str] = list(dict.fromkeys(
        meta_species + [item.id for item in extraction.species]
    ))

    tasks_raw = list(on_metadata.get("tasksCompleted") or summary.get("tasks") or [])
    tasks = sorted({*tasks_raw, *(item.id for item in extraction.tasks)})
    task_regions = _map_openneuro_tasks_to_regions(tasks_raw)
    extractor_regions = [item.id for item in extraction.brain_regions]
    brain_regions = list(dict.fromkeys(extractor_regions + task_regions))
    return {
        "source": "openneuro",
        "source_id": node["id"],
        "title": title,
        "description": description,
        "url": f"https://openneuro.org/datasets/{node['id']}",
        "license": snap_desc.get("License") if isinstance(snap_desc, dict) else None,
        "species": species_ids,
        "modalities": sorted({*modalities, *(item.id for item in extraction.modalities)}),
        "brain_regions": brain_regions,
        "tasks": tasks,
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": ["BIDS"],
        "has_behavior": bool(extraction.behaviors) or "events.tsv" in text.casefold(),
        "has_trials": any(term in text.casefold() for term in ["trial", "events.tsv", "task"]),
        "has_raw_data": True,
        "has_processed_data": "derivative" in text.casefold() or bool(on_metadata.get("dataProcessed")),
        "linked_paper_dois": doi_sources,
        "metadata_json": {
            "raw_source": "openneuro",
            "subjects": summary.get("subjects"),
            "snapshot_tag": snapshot.get("tag"),
            "size_bytes": snapshot.get("size"),
            "created": node.get("created"),
            "senior_author": on_metadata.get("seniorAuthor"),
            "study_design": on_metadata.get("studyDesign"),
            "study_domain": on_metadata.get("studyDomain"),
            "ages": on_metadata.get("ages"),
        },
    }


def normalize_openneuro_record(
    node: dict[str, Any],
    raw_payload_path: str | None = None,
) -> NormalizedDatasetRecord:
    """Normalize a raw OpenNeuro node into the v0.3 provenance-aware schema."""

    legacy = normalize_openneuro_dataset(node)
    metadata = legacy.get("metadata_json", {})
    extraction = extract_dataset_labels(
        title=legacy.get("title"),
        description=legacy.get("description"),
        file_paths=[],
        source_metadata={**metadata, "standard": "BIDS"},
        linked_paper_abstracts=[],
    )
    source_value = " ".join(
        str(part) for part in [legacy.get("title"), legacy.get("description"), metadata] if part
    )

    # Use legacy-normalized species (already mapped from OpenNeuro vocab + extraction)
    legacy_species = legacy.get("species") or []
    legacy_modalities = legacy.get("modalities") or []

    # Build EvidenceLabel lists; for legacy-mapped labels use structured source
    def _make_labels(ids: list[str], label_type: str, source_field: str) -> list:
        seen: set[str] = set()
        result = []
        for label_id in ids:
            if label_id in seen:
                continue
            seen.add(label_id)
            from neural_search.schemas import EvidenceLabel
            result.append(EvidenceLabel(
                id=label_id,
                label=label_id.replace("_", " "),
                label_type=label_type,
                confidence=0.75,
                evidence_text=f"openneuro_metadata:{label_id}",
                source_field=source_field,
                source_value=source_value[:100],
                extractor_name="openneuro_normalizer",
                extractor_version="v0.9.0",
            ))
        return result

    # DOI-based paper links
    from neural_search.ingestion.doi_utils import dois_to_paper_ids
    linked_papers = dois_to_paper_ids(legacy.get("linked_paper_dois") or [])

    return NormalizedDatasetRecord(
        dataset_id=stable_normalized_id("dataset", "openneuro", legacy["source_id"]),
        source="openneuro",
        source_id=legacy["source_id"],
        title=legacy["title"],
        description=legacy.get("description"),
        url=legacy.get("url"),
        raw_payload_path=raw_payload_path,
        species=_make_labels(legacy_species, "species", "openneuro_metadata"),
        modalities=_make_labels(legacy_modalities, "modality", "openneuro_summary"),
        brain_regions=[
            evidence_label_from_extraction(
                label, "brain_region", source_field="summary", source_value=source_value
            )
            for label in extraction.brain_regions
        ],
        tasks=[
            evidence_label_from_extraction(
                label, "task", source_field="summary", source_value=source_value
            )
            for label in extraction.tasks
        ],
        behavioral_events=[
            evidence_label_from_extraction(
                label, "behavioral_event", source_field="summary", source_value=source_value
            )
            for label in extraction.behaviors
        ],
        data_standards=[
            evidence_label_from_extraction(
                label, "data_standard", source_field="summary", source_value=source_value
            )
            for label in extraction.data_standards
        ],
        usability_flags=UsabilityFlags(
            has_trials=legacy.get("has_trials"),
            has_behavior=legacy.get("has_behavior"),
            has_neural_data=bool(legacy_modalities),
            has_raw_data=legacy.get("has_raw_data"),
            has_processed_data=legacy.get("has_processed_data"),
            has_standard_format=True,
        ),
        linked_papers=linked_papers,
        missing_fields=extraction.missing_fields,
    )


def records_from_response(data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    edges = data.get("data", {}).get("datasets", {}).get("edges", [])

    for edge in edges:
        node = edge.get("node", {})
        if node.get("id"):
            results.append(normalize_openneuro_dataset(node))

    return results[:limit]


@register("openneuro")
def fetch_openneuro_records(limit: int = 2000) -> list[dict[str, Any]]:
    """Registry adapter for full OpenNeuro cursor pagination."""
    return fetch_all_openneuro(max_records=limit)


async def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    """
    Fetch a specific dataset by ID.

    Args:
        dataset_id: OpenNeuro dataset ID (e.g., 'ds000001').

    Returns:
        Dataset record or None.
    """
    graphql_query = """
    query GetDataset($id: ID!) {
        dataset(id: $id) {
            id
            name
            description
            created
            public
            latestSnapshot {
                tag
                created
                size
                readme
                summary {
                    subjects
                    tasks
                    modalities
                }
            }
        }
    }
    """

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENNEURO_API_URL,
            json={
                "query": graphql_query,
                "variables": {"id": dataset_id},
            },
        )
        response.raise_for_status()
        data = response.json()

    node = data.get("data", {}).get("dataset")
    if not node:
        return None

    snapshot = node.get("latestSnapshot", {}) or {}
    summary = snapshot.get("summary", {}) or {}

    return {
        "source": "openneuro",
        "source_id": node["id"],
        "title": node.get("name", node["id"]),
        "description": node.get("description"),
        "url": f"https://openneuro.org/datasets/{node['id']}",
        "data_standards": ["BIDS"],
        "modalities": summary.get("modalities", []),
        "tasks": summary.get("tasks", []),
        "metadata_json": {
            "subjects": summary.get("subjects"),
            "snapshot_tag": snapshot.get("tag"),
            "size_bytes": snapshot.get("size"),
            "readme": snapshot.get("readme"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.openneuro")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        payload = fetch_openneuro(args.query, args.limit)
        if args.save or args.save_raw:
            raw_path = save_raw_response("openneuro", args.query, payload)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))
        records = records_from_response(payload, args.limit)
        print_normalized_records(records)
        if args.dry_run or not args.save:
            return 0
        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("openneuro", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
