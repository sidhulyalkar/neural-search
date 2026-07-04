"""Tests for the live OpenNeuro BIDS structure validator.

A real network call against ds000117 was made once during development to
confirm the GraphQL query shape (documented in the module docstring); these
tests mock the transport so the suite doesn't depend on network access.
"""

from __future__ import annotations

import httpx
import respx

from neural_search.graph.openneuro_bids_validator import (
    OPENNEURO_API_URL,
    validate_openneuro_dataset,
)


@respx.mock
def test_validate_openneuro_dataset_parses_real_shape():
    respx.post(OPENNEURO_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "dataset": {
                        "id": "ds000117",
                        "latestSnapshot": {
                            "tag": "1.0.5",
                            "files": [
                                {"filename": "dataset_description.json", "size": 100},
                                {"filename": "participants.tsv", "size": 200},
                                {"filename": "README", "size": 50},
                            ],
                            "summary": {
                                "modalities": ["mri", "meg"],
                                "tasks": ["facerecognition"],
                                "subjects": ["01", "02", "03"],
                            },
                        },
                    }
                }
            },
        )
    )
    result = validate_openneuro_dataset("ds000117")
    assert result.error is None
    assert result.has_dataset_description is True
    assert result.has_participants_tsv is True
    assert result.modalities == ["mri", "meg"]
    assert result.tasks == ["facerecognition"]
    assert result.n_subjects == 3


@respx.mock
def test_validate_openneuro_dataset_handles_missing_dataset():
    respx.post(OPENNEURO_API_URL).mock(
        return_value=httpx.Response(200, json={"data": {"dataset": None}})
    )
    result = validate_openneuro_dataset("ds999999")
    assert result.error == "dataset not found"


@respx.mock
def test_validate_openneuro_dataset_handles_graphql_errors():
    respx.post(OPENNEURO_API_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "boom"}]}
        )
    )
    result = validate_openneuro_dataset("ds000117")
    assert result.error is not None
    assert "boom" in result.error


@respx.mock
def test_validate_openneuro_dataset_handles_transport_error():
    respx.post(OPENNEURO_API_URL).mock(side_effect=httpx.ConnectTimeout("timeout"))
    result = validate_openneuro_dataset("ds000117")
    assert result.error is not None
