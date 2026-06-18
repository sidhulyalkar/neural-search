"""Tests for neural_search.literature.normalizer."""

from __future__ import annotations

import pytest
from neural_search.literature.normalizer import (
    compute_quality_flags,
    deduplicate_findings,
    normalize_finding,
    normalize_regions,
    normalize_species,
    normalize_tasks,
)


class TestNormalizeSpecies:
    def test_humans_to_human(self):
        result, changed = normalize_species(["humans"])
        assert result == ["human"]
        assert changed

    def test_mice_to_mouse(self):
        result, changed = normalize_species(["mice"])
        assert result == ["mouse"]
        assert changed

    def test_rats_to_rat(self):
        result, changed = normalize_species(["rats"])
        assert result == ["rat"]
        assert changed

    def test_monkeys_to_monkey(self):
        result, changed = normalize_species(["monkeys"])
        assert result == ["monkey"]
        assert changed

    def test_macaques_to_macaque(self):
        result, changed = normalize_species(["macaques"])
        assert result == ["macaque"]
        assert changed

    def test_unknown_species_passthrough(self):
        result, changed = normalize_species(["zebrafish"])
        assert result == ["zebrafish"]
        # zebrafish is in canonical but maps to itself, changed may vary
        # main check: passthrough doesn't crash

    def test_deduplication_after_normalization(self):
        result, changed = normalize_species(["humans", "human"])
        assert result == ["human"]
        assert changed

    def test_empty_list(self):
        result, changed = normalize_species([])
        assert result == []
        assert not changed

    def test_mixed_list(self):
        result, changed = normalize_species(["humans", "mice", "macaques"])
        assert result == ["human", "mouse", "macaque"]
        assert changed


class TestNormalizeRegions:
    def test_underscore_to_space(self):
        result, generic, changed = normalize_regions(["prefrontal_cortex"])
        assert "prefrontal cortex" in result
        assert changed

    def test_generic_brain_removed(self):
        result, generic, changed = normalize_regions(["brain"])
        assert result == []
        assert "brain" in generic
        assert changed

    def test_generic_cortex_removed(self):
        result, generic, changed = normalize_regions(["cortex"])
        assert result == []
        assert "cortex" in generic

    def test_v1_expanded(self):
        result, generic, changed = normalize_regions(["v1"])
        assert "primary visual cortex" in result
        assert changed

    def test_pfc_expanded(self):
        result, generic, changed = normalize_regions(["pfc"])
        assert "prefrontal cortex" in result
        assert changed

    def test_specific_region_kept(self):
        result, generic, changed = normalize_regions(["hippocampus"])
        assert "hippocampus" in result
        assert not generic

    def test_empty_list(self):
        result, generic, changed = normalize_regions([])
        assert result == []
        assert generic == []

    def test_mixed_generic_and_specific(self):
        result, generic, changed = normalize_regions(["brain", "hippocampus", "amygdala"])
        assert "hippocampus" in result
        assert "amygdala" in result
        assert "brain" in generic
        assert "brain" not in result

    def test_dedup_after_alias_expansion(self):
        result, generic, changed = normalize_regions(["hippocampus", "hippocampus"])
        assert result.count("hippocampus") == 1


class TestNormalizeTasks:
    def test_spatial_navigation_variant(self):
        result, changed = normalize_tasks(["spatial navigation"])
        assert "navigation" in result
        assert changed

    def test_decision_hyphen_normalized(self):
        result, changed = normalize_tasks(["decision-making"])
        assert "decision making" in result
        assert changed

    def test_working_memory_variant(self):
        result, changed = normalize_tasks(["working memory task"])
        assert "working memory" in result
        assert changed

    def test_unknown_task_passthrough(self):
        result, changed = normalize_tasks(["novelty detection"])
        assert "novelty detection" in result

    def test_dedup_after_normalization(self):
        result, changed = normalize_tasks(["spatial navigation", "navigation"])
        assert result.count("navigation") == 1


class TestComputeQualityFlags:
    def _base_finding(self, **kwargs):
        base = {
            "finding_text": "Alpha stimulation modulated target visibility.",
            "regions": ["occipital"],
            "species": ["human"],
            "confidence": 0.9,
        }
        base.update(kwargs)
        return base

    def test_clean_finding_no_flags(self):
        flags = compute_quality_flags(self._base_finding(), generic_regions=[])
        assert flags == []

    def test_no_region_flag(self):
        flags = compute_quality_flags(self._base_finding(regions=[]), generic_regions=[])
        assert "no_region" in flags

    def test_generic_region_only_flag(self):
        flags = compute_quality_flags(self._base_finding(regions=[]), generic_regions=["brain"])
        assert "generic_region_only" in flags
        assert "no_region" not in flags

    def test_no_species_flag(self):
        flags = compute_quality_flags(self._base_finding(species=[]), generic_regions=[])
        assert "no_species" in flags

    def test_low_confidence_flag(self):
        flags = compute_quality_flags(self._base_finding(confidence=0.6), generic_regions=[])
        assert "low_confidence" in flags

    def test_meta_language_flag(self):
        flags = compute_quality_flags(
            self._base_finding(finding_text="This study demonstrates that..."),
            generic_regions=[],
        )
        assert "meta_language" in flags

    def test_very_short_finding_flag(self):
        flags = compute_quality_flags(
            self._base_finding(finding_text="Signal increases."),
            generic_regions=[],
        )
        assert "very_short_finding" in flags


class TestNormalizeFinding:
    def _raw_finding(self, **kwargs):
        base = {
            "paper_id": "paper:openalex:W1234",
            "paper_doi": None,
            "finding_id": "paper:openalex:W1234:f0",
            "finding_text": "Alpha power modulates memory encoding in the hippocampus.",
            "result_direction": "increase",
            "regions": ["hippocampus"],
            "species": ["humans"],
            "modalities": ["EEG"],
            "tasks": ["working memory task"],
            "cell_types": [],
            "molecules": [],
            "confidence": 0.9,
            "extraction_model": "qwen2.5:7b",
            "extracted_at": "2026-06-17T00:00:00+00:00",
        }
        base.update(kwargs)
        return base

    def test_species_normalized(self):
        result = normalize_finding(self._raw_finding())
        assert result["species"] == ["human"]
        assert result["_normalized"]["species_original"] == ["humans"]

    def test_task_normalized(self):
        result = normalize_finding(self._raw_finding())
        assert "working memory" in result["tasks"]

    def test_original_preserved_in_normalized(self):
        result = normalize_finding(self._raw_finding(species=["mice"]))
        assert result["_normalized"]["species_original"] == ["mice"]
        assert result["species"] == ["mouse"]

    def test_clean_finding_no_normalized_key(self):
        result = normalize_finding(
            self._raw_finding(species=["human"], tasks=["memory"])
        )
        assert "_normalized" not in result

    def test_underscore_region_cleaned(self):
        result = normalize_finding(self._raw_finding(regions=["prefrontal_cortex"]))
        assert "prefrontal cortex" in result["regions"]

    def test_generic_region_flagged(self):
        result = normalize_finding(self._raw_finding(regions=["brain"]))
        assert result["regions"] == []
        assert "generic_region_only" in result.get("quality_flags", [])

    def test_no_mutation_of_input(self):
        raw = self._raw_finding()
        original_species = raw["species"][:]
        normalize_finding(raw)
        assert raw["species"] == original_species


class TestDeduplicateFindings:
    def _finding(self, paper_id: str, text: str, idx: int = 0) -> dict:
        return {
            "paper_id": paper_id,
            "finding_id": f"{paper_id}:f{idx}",
            "finding_text": text,
            "result_direction": "increase",
        }

    def test_removes_exact_duplicates_within_paper(self):
        findings = [
            self._finding("p1", "Alpha increases."),
            self._finding("p1", "Alpha increases."),
            self._finding("p1", "Beta decreases."),
        ]
        deduped, n_removed = deduplicate_findings(findings)
        assert n_removed == 1
        assert len(deduped) == 2

    def test_keeps_same_text_across_different_papers(self):
        findings = [
            self._finding("p1", "Alpha increases."),
            self._finding("p2", "Alpha increases."),
        ]
        deduped, n_removed = deduplicate_findings(findings)
        assert n_removed == 0
        assert len(deduped) == 2

    def test_case_insensitive_dedup(self):
        findings = [
            self._finding("p1", "Alpha increases."),
            self._finding("p1", "alpha increases."),
        ]
        deduped, n_removed = deduplicate_findings(findings)
        assert n_removed == 1

    def test_empty_list(self):
        deduped, n_removed = deduplicate_findings([])
        assert deduped == []
        assert n_removed == 0
