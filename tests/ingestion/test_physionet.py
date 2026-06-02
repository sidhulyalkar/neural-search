from __future__ import annotations
import pytest
import httpx
import respx
from neural_search.ingestion.physionet import (
    _parse_dataset_links,
    normalize_physionet_dataset,
    fetch_physionet,
)

SAMPLE_LISTING_HTML = """
<html><body>
<ul class="database-list">
  <li><a href="/content/chbmit/1.0.0/">CHB-MIT Scalp EEG</a></li>
  <li><a href="/content/eegmmidb/1.0.0/">EEG Motor Imagery</a></li>
  <li><a href="/content/noneuro/1.0.0/">Non-Neuro Dataset</a></li>
</ul>
</body></html>
"""

SAMPLE_DETAIL_HTML = """
<html><body>
<h1 class="project-title">CHB-MIT Scalp EEG Database</h1>
<div class="project-description">
  Scalp EEG recordings from pediatric patients with intractable epilepsy.
  Collected at Boston Children's Hospital.
</div>
<ul class="subject-list">
  <li>EEG</li><li>Seizure</li><li>Pediatric</li>
</ul>
</body></html>
"""


def test_parse_dataset_links_extracts_content_paths() -> None:
    links = _parse_dataset_links(SAMPLE_LISTING_HTML)
    assert "/content/chbmit/1.0.0/" in links
    assert "/content/eegmmidb/1.0.0/" in links
    assert "/content/noneuro/1.0.0/" in links
    assert len(links) == 3


def test_normalize_physionet_dataset_basic() -> None:
    raw = {
        "slug": "chbmit",
        "version": "1.0.0",
        "title": "CHB-MIT Scalp EEG Database",
        "description": "Scalp EEG recordings from pediatric patients with intractable epilepsy.",
        "topics": ["EEG", "Seizure"],
    }
    rec = normalize_physionet_dataset(raw)
    assert rec["source"] == "physionet"
    assert rec["source_id"] == "chbmit"
    assert "eeg" in rec["modalities"]


@respx.mock
def test_fetch_physionet_returns_neuro_records() -> None:
    respx.get("https://physionet.org/content/").mock(
        return_value=httpx.Response(200, text=SAMPLE_LISTING_HTML)
    )
    # Mock two neuro detail pages
    respx.get("https://physionet.org/content/chbmit/1.0.0/").mock(
        return_value=httpx.Response(200, text=SAMPLE_DETAIL_HTML)
    )
    respx.get("https://physionet.org/content/eegmmidb/1.0.0/").mock(
        return_value=httpx.Response(200, text=SAMPLE_DETAIL_HTML)
    )
    respx.get("https://physionet.org/content/noneuro/1.0.0/").mock(
        return_value=httpx.Response(200, text="<html><body><h1>Non-Neuro</h1></body></html>")
    )
    records = fetch_physionet(limit=50)
    # All returned records must have source="physionet" and at least one is accepted
    assert len(records) >= 1
    assert all(r["source"] == "physionet" for r in records)
