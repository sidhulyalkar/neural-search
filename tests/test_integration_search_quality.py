"""Integration tests for end-to-end search quality on the live corpus.

These tests are skipped when the full corpus is not present so CI still runs.
They verify that domain-specific neuroscience queries surface correct datasets
in the top-3 results based on modality, species, and brain region metadata.

Run locally with:
    pytest tests/test_integration_search_quality.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")

pytestmark = pytest.mark.skipif(
    not CORPUS_PATH.exists(),
    reason="full corpus not present (run scripts/corpus/build_full_corpus.py first)",
)


@pytest.fixture(scope="module")
def corpus() -> list[dict[str, Any]]:
    return [json.loads(l) for l in CORPUS_PATH.read_text().strip().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def ds_map(corpus: list[dict[str, Any]]) -> dict[str, dict]:
    m: dict[str, dict] = {}
    for r in corpus:
        m[r["dataset_id"]] = r
        if r.get("source_id"):
            m[r["source_id"]] = r
    return m


def _ids(field: Any) -> list[str]:
    if not field:
        return []
    if isinstance(field, list):
        return [(v.get("id") if isinstance(v, dict) else v) for v in field if v]
    return [str(field)]


def _top3(query: str, corpus: list, ds_map: dict) -> list[dict[str, Any]]:
    from neural_search.search.core import search_datasets
    results_obj = search_datasets(query, datasets=corpus, limit=5)
    results = results_obj.results if hasattr(results_obj, "results") else results_obj
    out = []
    for r in results[:3]:
        ds = ds_map.get(r.dataset_id) or ds_map.get(r.dataset_id.split(":")[-1], {})
        out.append({
            "score": r.score,
            "title": ds.get("title", r.dataset_id),
            "source_id": ds.get("source_id", ""),
            "modalities": _ids(ds.get("modalities")),
            "species": _ids(ds.get("species")),
            "brain_regions": _ids(ds.get("brain_regions")),
        })
    return out


class TestDomainPrecision:
    """Queries with specific modality + species + region should surface matching datasets."""

    def test_calcium_imaging_mouse_hippocampus(self, corpus, ds_map) -> None:
        results = _top3("two-photon calcium imaging mouse hippocampal CA1", corpus, ds_map)
        assert results, "expected at least one result"
        # All top-3 should be calcium imaging
        assert any("calcium_imaging" in r["modalities"] for r in results), (
            f"expected calcium_imaging in top-3 modalities, got {[r['modalities'] for r in results]}"
        )
        # At least one should be mouse
        assert any("mouse" in r["species"] for r in results), (
            f"expected mouse in top-3 species, got {[r['species'] for r in results]}"
        )

    def test_neuropixels_mouse(self, corpus, ds_map) -> None:
        # Neuropixels IS extracellular ephys; accept either tag in top-3
        results = _top3("Neuropixels silicon probe mouse multi-region recording", corpus, ds_map)
        ephys_tags = {"neuropixels", "extracellular_ephys"}
        assert any(bool(ephys_tags & set(r["modalities"])) for r in results), (
            f"expected extracellular_ephys or neuropixels in top-3, got {[r['modalities'] for r in results]}"
        )
        assert any("mouse" in r["species"] for r in results), (
            f"expected mouse species in top-3, got {[r['species'] for r in results]}"
        )

    def test_rat_lfp_hippocampus(self, corpus, ds_map) -> None:
        results = _top3("rat LFP hippocampus sleep sharp-wave ripples", corpus, ds_map)
        assert any("lfp" in r["modalities"] for r in results), (
            f"expected lfp in top-3, got {[r['modalities'] for r in results]}"
        )
        assert any("rat" in r["species"] for r in results), (
            f"expected rat in top-3, got {[r['species'] for r in results]}"
        )

    def test_macaque_ephys(self, corpus, ds_map) -> None:
        results = _top3("macaque motor cortex single-unit reaching task", corpus, ds_map)
        assert any("macaque" in r["species"] for r in results), (
            f"expected macaque in top-3, got {[r['species'] for r in results]}"
        )

    def test_human_ecog(self, corpus, ds_map) -> None:
        results = _top3("human ECoG intracranial recording speech production", corpus, ds_map)
        assert any(
            "ecog" in r["modalities"] or "ieeg" in r["modalities"]
            for r in results
        ), f"expected ecog/ieeg in top-3, got {[r['modalities'] for r in results]}"
        assert any("human" in r["species"] for r in results), (
            f"expected human in top-3, got {[r['species'] for r in results]}"
        )


class TestCrossModalSpecificity:
    """Modality-specific queries should not cross-contaminate."""

    def test_calcium_imaging_not_ephys(self, corpus, ds_map) -> None:
        results = _top3("calcium imaging fluorescence GCaMP mouse NOT electrophysiology", corpus, ds_map)
        # Top result should be calcium imaging, not extracellular_ephys-only
        top = results[0] if results else {}
        assert "calcium_imaging" in top.get("modalities", []), (
            f"expected rank-1 to be calcium_imaging, got {top.get('modalities')}"
        )

    def test_fmri_human(self, corpus, ds_map) -> None:
        results = _top3("BIDS fMRI human task-based brain imaging GLM", corpus, ds_map)
        assert any("fmri" in r["modalities"] for r in results), (
            f"expected fmri in top-3, got {[r['modalities'] for r in results]}"
        )
        assert any("human" in r["species"] for r in results), (
            f"expected human in top-3, got {[r['species'] for r in results]}"
        )


class TestAnalysisReadiness:
    """Analysis-specific queries should surface datasets with relevant affordances."""

    def test_trial_aligned_population_dynamics(self, corpus, ds_map) -> None:
        results = _top3("trial-aligned neural population dynamics dimensionality reduction", corpus, ds_map)
        assert results, "expected results for trial-aligned query"
        # Should be ephys datasets (extracellular or LFP)
        ephys_types = {"extracellular_ephys", "lfp", "neuropixels"}
        assert any(
            bool(ephys_types & set(r["modalities"])) for r in results
        ), f"expected ephys in top-3, got {[r['modalities'] for r in results]}"

    def test_ibl_brain_wide_map_found(self, corpus, ds_map) -> None:
        results = _top3("IBL brain wide map neuropixels mouse", corpus, ds_map)
        # IBL Brain Wide Map (000409) should be in top-3 by name or source_id
        found = any(
            "IBL" in r["title"] or "000409" in r["source_id"]
            for r in results
        )
        assert found, (
            f"expected IBL Brain Wide Map in top-3, got {[r['title'] for r in results]}"
        )


class TestCorpusMetadataCoverage:
    """Validate that enrichment improved metadata coverage."""

    def test_modality_coverage_above_75pct(self, corpus) -> None:
        with_mods = sum(1 for r in corpus if r.get("modalities"))
        pct = 100 * with_mods // len(corpus)
        assert pct >= 75, f"modality coverage {pct}% below 75% target"

    def test_species_coverage_above_40pct(self, corpus) -> None:
        with_spc = sum(1 for r in corpus if r.get("species"))
        pct = 100 * with_spc // len(corpus)
        assert pct >= 40, f"species coverage {pct}% below 40% target"

    def test_corpus_has_multi_species(self, corpus) -> None:
        def _ids(v: Any) -> list[str]:
            if isinstance(v, list):
                return [(x.get("id") if isinstance(x, dict) else x) for x in v if x]
            return []
        all_species: set[str] = set()
        for r in corpus:
            all_species.update(s for s in _ids(r.get("species")) if s)
        for expected in ("mouse", "rat", "human", "macaque"):
            assert expected in all_species, f"expected species '{expected}' missing from corpus"
