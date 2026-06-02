from __future__ import annotations
import json
import pytest
import httpx
import respx
from neural_search.ingestion.dandi import fetch_all_dandisets, normalize_dandiset

DANDI_API = "https://api.dandiarchive.org/api"

STUB_DANDISET = {
    "identifier": "000001",
    "draft_version": {"name": "Mouse Visual Cortex Neuropixels", "metadata": {"description": "Neuropixels recordings from mouse V1."}},
}

def _make_page(results: list, next_url: str | None) -> dict:
    return {"count": 10, "next": next_url, "previous": None, "results": results}


@respx.mock
def test_fetch_all_dandisets_single_page() -> None:
    respx.get(f"{DANDI_API}/dandisets/").mock(
        return_value=httpx.Response(200, json=_make_page([STUB_DANDISET], None))
    )
    records = fetch_all_dandisets(page_size=100)
    assert len(records) == 1
    assert records[0]["source"] == "dandi"
    assert records[0]["source_id"] == "000001"


@respx.mock
def test_fetch_all_dandisets_two_pages() -> None:
    page1_url = f"{DANDI_API}/dandisets/?page=1&page_size=2"
    page2_url = f"{DANDI_API}/dandisets/?page=2&page_size=2"
    respx.get(page1_url).mock(
        return_value=httpx.Response(200, json=_make_page([STUB_DANDISET], page2_url))
    )
    stub2 = dict(STUB_DANDISET, identifier="000002")
    respx.get(page2_url).mock(
        return_value=httpx.Response(200, json=_make_page([stub2], None))
    )
    records = fetch_all_dandisets(start_url=page1_url, page_size=2)
    assert len(records) == 2
    assert {r["source_id"] for r in records} == {"000001", "000002"}


@respx.mock
def test_fetch_all_dandisets_respects_max_records() -> None:
    stubs = [dict(STUB_DANDISET, identifier=str(i).zfill(6)) for i in range(5)]
    respx.get(f"{DANDI_API}/dandisets/").mock(
        return_value=httpx.Response(200, json=_make_page(stubs, None))
    )
    records = fetch_all_dandisets(max_records=3, page_size=100)
    assert len(records) == 3
