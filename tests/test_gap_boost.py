"""Tests for CoverageGapBooster — rarity-based retrieval scoring."""
from __future__ import annotations

import math

import pytest

from neural_search.coverage.gap_boost import MAX_BOOST, CoverageGapBooster


@pytest.fixture
def booster() -> CoverageGapBooster:
    return CoverageGapBooster(
        region_counts={"hippocampus": 808, "barrel_cortex": 9, "retina": 87, "ca1": 432},
        modality_counts={"fmri": 2473, "calcium_imaging": 849, "neuropixels": 311},
        species_counts={"mouse": 1200, "macaque": 45, "zebrafish": 8},
        total_datasets=7176,
    )


class TestRarityScore:
    def test_common_value_has_low_rarity(self, booster: CoverageGapBooster) -> None:
        fmri_rarity = booster.rarity("modalities", "fmri")
        ca1_rarity = booster.rarity("brain_regions", "ca1")
        assert fmri_rarity < 0.3
        assert ca1_rarity < 0.5

    def test_rare_value_has_high_rarity(self, booster: CoverageGapBooster) -> None:
        barrel_rarity = booster.rarity("brain_regions", "barrel_cortex")
        zebrafish_rarity = booster.rarity("species", "zebrafish")
        assert barrel_rarity > 0.7
        assert zebrafish_rarity > 0.7

    def test_zero_count_gives_max_rarity(self, booster: CoverageGapBooster) -> None:
        rarity = booster.rarity("brain_regions", "nonexistent_region")
        expected = 1.0 - math.log(0 + 1) / math.log(7176 + 1)
        assert abs(rarity - expected) < 1e-6

    def test_rarity_ordering(self, booster: CoverageGapBooster) -> None:
        r_barrel = booster.rarity("brain_regions", "barrel_cortex")
        r_ca1 = booster.rarity("brain_regions", "ca1")
        r_hippo = booster.rarity("brain_regions", "hippocampus")
        assert r_barrel > r_ca1 > r_hippo


class TestScoreMethod:
    def test_empty_inputs_return_zero(self, booster: CoverageGapBooster) -> None:
        assert booster.score() == 0.0
        assert booster.score(region_ids=set()) == 0.0
        assert booster.score(region_ids=None, modality_ids=set(), species_ids=None) == 0.0

    def test_rare_region_modality_pair_scores_high(self, booster: CoverageGapBooster) -> None:
        boost = booster.score(region_ids={"barrel_cortex"}, modality_ids={"fmri"})
        assert boost > 0.0
        assert boost <= MAX_BOOST

    def test_common_pair_scores_low(self, booster: CoverageGapBooster) -> None:
        boost_rare = booster.score(region_ids={"barrel_cortex"}, modality_ids={"fmri"})
        boost_common = booster.score(region_ids={"hippocampus"}, modality_ids={"calcium_imaging"})
        assert boost_rare > boost_common

    def test_score_capped_at_max_boost(self, booster: CoverageGapBooster) -> None:
        boost = booster.score(
            region_ids={"nonexistent"}, modality_ids={"nonexistent"}, species_ids={"nonexistent"}
        )
        assert boost <= MAX_BOOST

    def test_score_nonnegative(self, booster: CoverageGapBooster) -> None:
        boost = booster.score(region_ids={"hippocampus"}, modality_ids={"fmri"})
        assert boost >= 0.0

    def test_multiple_regions_averages_rarity(self, booster: CoverageGapBooster) -> None:
        boost_rare_only = booster.score(region_ids={"barrel_cortex"})
        boost_mixed = booster.score(region_ids={"barrel_cortex", "hippocampus"})
        # Mixed should be between hippocampus-only and barrel_cortex-only
        boost_common_only = booster.score(region_ids={"hippocampus"})
        assert boost_common_only <= boost_mixed <= boost_rare_only


class TestEmptyBooster:
    def test_empty_booster_returns_zero(self) -> None:
        empty = CoverageGapBooster._empty()
        assert empty.score(region_ids={"ca1"}) == 0.0
        assert empty.score(modality_ids={"fmri"}, species_ids={"mouse"}) == 0.0

    def test_from_db_fallback_when_no_db(self, tmp_path) -> None:
        booster = CoverageGapBooster.from_db(tmp_path / "nonexistent.duckdb")
        assert booster.score(region_ids={"hippocampus"}) == 0.0


class TestCoverageStats:
    def test_stats_has_expected_keys(self, booster: CoverageGapBooster) -> None:
        stats = booster.coverage_stats()
        assert "total_datasets" in stats
        assert "regions_tracked" in stats
        assert "rarest_regions" in stats

    def test_rarest_regions_are_actually_rare(self, booster: CoverageGapBooster) -> None:
        stats = booster.coverage_stats()
        for entry in stats["rarest_regions"]:
            assert entry["n_datasets"] < 10
            assert entry["rarity"] > 0.5

    def test_from_live_db(self) -> None:
        from pathlib import Path
        db = Path("data/coverage/ledger.duckdb")
        if not db.exists():
            pytest.skip("Live DuckDB ledger not available")
        booster = CoverageGapBooster.from_db(db)
        assert booster._total > 1000
        # hippocampus should be common
        hippo_rarity = booster.rarity("brain_regions", "hippocampus")
        assert hippo_rarity < 0.5
