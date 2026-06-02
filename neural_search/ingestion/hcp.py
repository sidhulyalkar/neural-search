"""Human Connectome Project (HCP) ingestion adapter.

HCP provides high-resolution human MRI/MEG/EEG data. Access requires:
1. ConnectomeDB account: https://db.humanconnectome.org/
2. Data use agreement acceptance
3. Generated credentials for S3 bucket access

This adapter reads from a curated metadata manifest (no auth required for metadata).
Full data access needs credentials.

Set HCP_USERNAME and HCP_PASSWORD env vars, or pass credentials to fetch_hcp().
"""
from __future__ import annotations

import logging
from typing import Any

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

HCP_DATASETS: list[dict[str, Any]] = [
    {
        "source_id": "HCP-Young-Adult",
        "title": "Human Connectome Project Young Adult (HCP-YA)",
        "description": (
            "High-resolution structural and functional MRI, resting-state fMRI, "
            "task fMRI, diffusion MRI, and MEG data from 1200 healthy young adults (22-35 years). "
            "Participants completed multiple cognitive tasks and resting-state paradigms."
        ),
        "url": "https://db.humanconnectome.org/data/projects/HCP_1200",
        "n_subjects": 1200,
        "modalities": ["fmri", "meg", "dwi"],
        "tasks": ["working_memory", "motor", "language", "social_cognition", "emotion"],
        "species": ["human"],
        "license": "HCP Open Access",
        "auth_required": True,
    },
    {
        "source_id": "HCP-Aging",
        "title": "Human Connectome Project Aging (HCP-A)",
        "description": (
            "MRI and behavioral data from 1200+ adults across the lifespan (36-100 years). "
            "Multiband MRI, resting-state fMRI, task fMRI, T1w, T2w, dMRI."
        ),
        "url": "https://db.humanconnectome.org/data/projects/HCP_Aging",
        "n_subjects": 1200,
        "modalities": ["fmri", "dwi"],
        "tasks": ["resting_state", "working_memory"],
        "species": ["human"],
        "license": "HCP Open Access",
        "auth_required": True,
    },
    {
        "source_id": "HCP-Development",
        "title": "Human Connectome Project Development (HCP-D)",
        "description": (
            "Lifespan human connectome data from 1350 healthy participants ages 5-21. "
            "Structural MRI, resting-state fMRI, task fMRI, dMRI."
        ),
        "url": "https://db.humanconnectome.org/data/projects/HCP_Development",
        "n_subjects": 1350,
        "modalities": ["fmri", "dwi"],
        "tasks": ["resting_state", "emotion", "language"],
        "species": ["human"],
        "license": "HCP Open Access",
        "auth_required": True,
    },
]


def normalize_hcp_dataset(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an HCP dataset entry."""
    source_id = raw["source_id"]
    title = raw["title"]
    description = raw["description"]

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    return {
        "source": "hcp",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": raw.get("url"),
        "license": raw.get("license", "HCP Open Access"),
        "species": raw.get("species") or [item.id for item in extraction.species] or ["human"],
        "modalities": raw.get("modalities") or sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": raw.get("tasks") or [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": ["BIDS", "HCP-MMP"],
        "subject_count": raw.get("n_subjects"),
        "has_behavior": True,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "hcp",
            "auth_required": raw.get("auth_required", True),
            "access_note": (
                "Requires ConnectomeDB registration and data use agreement. "
                "See https://db.humanconnectome.org/"
            ),
        },
    }


@register("hcp")
def fetch_hcp(limit: int = 10) -> list[dict[str, Any]]:
    """Return HCP dataset metadata (from curated manifest — no auth needed for metadata)."""
    records = [normalize_hcp_dataset(d) for d in HCP_DATASETS[:limit]]
    logger.info("HCP: returning %d datasets (metadata only; auth required for data access)", len(records))
    return records
