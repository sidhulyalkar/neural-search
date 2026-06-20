"""Tests for Obsidian vault export scripts."""
from __future__ import annotations

import subprocess
import sys

from neural_search.eval.evidence import DatasetEvidence, QuerySpec
from neural_search.obsidian.templates import (
    dataset_card_body,
    dataset_card_frontmatter,
    query_card_body,
    query_card_frontmatter,
)


class TestDatasetCardTemplate:
    def _make_ev(self) -> DatasetEvidence:
        return DatasetEvidence(
            record_id="dandi:000004",
            source="dandi",
            title="Human ephys",
            description="A human single-neuron dataset",
            species=["human"],
            modalities=["extracellular_ephys"],
            data_levels=["raw"],
            tasks=[],
            regions=[],
            license="CC-BY-4.0",
            doi=None,
            url="https://dandiarchive.org/dandiset/000004",
            raw_data_available=True,
            metadata_completeness=0.7,
        )

    def test_frontmatter_has_required_fields(self):
        ev = self._make_ev()
        fm = dataset_card_frontmatter(ev)
        assert fm["type"] == "dataset"
        assert fm["dataset_id"] == "dandi:000004"
        assert fm["source"] == "dandi"
        assert "dataset" in fm["tags"]

    def test_body_contains_title(self):
        ev = self._make_ev()
        body = dataset_card_body(ev)
        assert "Human ephys" in body


class TestQueryCardTemplate:
    def _make_spec(self) -> QuerySpec:
        return QuerySpec(
            query_id="q_0001",
            query_text="human fMRI reward",
            intent="META_ANALYSIS",
            scientific_goal="Find datasets for meta-analysis.",
            required_modalities=["fmri"],
            required_species=["human"],
            hard_negatives=["resting-state fMRI"],
        )

    def test_frontmatter_has_query_id(self):
        spec = self._make_spec()
        fm = query_card_frontmatter(spec)
        assert fm["query_id"] == "q_0001"
        assert fm["type"] == "query"

    def test_body_contains_hard_negatives(self):
        spec = self._make_spec()
        body = query_card_body(spec)
        assert "resting-state fMRI" in body


class TestInitVaultScript:
    def test_creates_vault_folders(self, tmp_path):
        vault = tmp_path / "vault"
        result = subprocess.run(
            [sys.executable, "scripts/obsidian/init_vault.py", "--vault", str(vault)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (vault / "00_Project").exists()
        assert (vault / "05_Annotations" / "Human Audits").exists()
        assert (vault / "99_Templates").exists()
