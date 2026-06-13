"""Tests for Brain Image Library ingestion."""
from __future__ import annotations

import httpx
import respx

from neural_search.ingestion.brain_image_library import (
    BIL_API,
    fetch_brain_image_library,
    normalize_bil_record,
)


BIL_RECORD = {
    "Metadata": "2.0",
    "Submission": {
        "submission_uuid": "96ba8210cceceeb7",
        "doi": "https://doi.org/10.35077/rut",
    },
    "Dataset": [
        {
            "bildirectory": "/bil/data/96/ba/96ba8210cceceeb7/0539072223",
            "title": "Mouse cortex two-photon whole brain image",
            "rights": "Creative Commons Attribution 4.0 International",
            "rightsuri": "https://creativecommons.org/licenses/by/4.0/",
            "rightsidentifier": "CC-BY-4.0",
            "generalmodality": "population imaging",
            "technique": "enhancer virus labeling two photon",
            "abstract": "Serial two-photon images of whole mouse brain cortex.",
            "doi": "https://doi.org/10.35077/rut",
            "dataset_size": "17GB",
        }
    ],
    "Specimen": [
        {
            "species": "Mouse",
            "organname": "Brain",
            "locations": "cortex",
        }
    ],
    "Instrument": [
        {
            "microscopetype": "Two Photon",
            "microscopemanufacturerandmodel": "TissueVision TissueCyte 1000",
        }
    ],
    "Assets": [
        {
            "bildid": "rut",
            "bildoi": "https://doi.org/10.35077/rut",
            "manifestfile": "https://download.brainimagelibrary.org/inventory/rut.json",
        }
    ],
    "Image": [
        {
            "representation": "neurite",
            "files": "tiff",
            "gbytes": "17.0",
        }
    ],
}


def test_normalize_bil_record_extracts_core_fields():
    rec = normalize_bil_record(BIL_RECORD)

    assert rec["source"] == "brain_image_library"
    assert rec["source_id"] == "rut"
    assert rec["source_type"] == "canonical_dataset"
    assert rec["record_type"] == "dataset_collection"
    assert rec["license"] == "CC-BY-4.0"
    assert rec["doi"] == "https://doi.org/10.35077/rut"
    assert "mouse" in rec["species"]
    assert "microscopy" in rec["modalities"]
    assert "two_photon" in rec["modalities"]
    assert "morphology" in rec["modalities"]
    assert "cortex" in rec["brain_regions"]
    assert "whole_brain" in rec["brain_regions"]
    assert rec["assets"][0]["record_type"] == "child_asset"
    assert rec["assets"][0]["asset_type"] == "manifest"
    assert "brain_image_analysis" in rec["analysis_affordances"]


@respx.mock
def test_fetch_brain_image_library_query_retrieve_flow():
    respx.get(f"{BIL_API}/query/fulltext").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": "true",
                "endpoint": "Fulltext/text",
                "message": "GET success",
                "bildids": ["rut"],
            },
        )
    )
    respx.get(f"{BIL_API}/retrieve").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": "true",
                "endpoint": "Retrieve/bildid",
                "message": "GET success",
                "retjson": [BIL_RECORD],
            },
        )
    )

    records = fetch_brain_image_library(limit=1)

    assert len(records) == 1
    assert records[0]["source_id"] == "rut"


def test_brain_image_library_registry_adapter(monkeypatch):
    import neural_search.ingestion.brain_image_library as bil
    from neural_search.ingestion.registry import run_adapter

    monkeypatch.setattr(
        bil,
        "_query_ids",
        lambda client, endpoint, params: ["rut"],
    )
    monkeypatch.setattr(
        bil,
        "_retrieve_metadata",
        lambda client, bildid: [BIL_RECORD],
    )

    records = run_adapter("brain_image_library", limit=1)

    assert records[0]["source"] == "brain_image_library"
