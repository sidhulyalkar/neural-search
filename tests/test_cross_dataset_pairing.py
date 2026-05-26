"""Tests for cross-dataset pairing scorer."""

from __future__ import annotations

import json

import pytest

from neural_search.evaluation.cross_dataset_pairing import (
    DatasetPair,
    PairingFeatures,
    classify_pair_type,
    compute_compatibility_score,
    compute_pairing_features,
    generate_pairing_json,
    generate_pairing_markdown,
    generate_rationale,
    generate_use_cases,
    score_dataset_pair,
)


class TestPairingFeatures:
    """Test feature computation."""

    def test_identical_datasets(self):
        """Identical datasets should have perfect overlap."""
        ds = {
            "tasks": ["go_nogo", "reversal_learning"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
            "brain_regions": ["pfc", "striatum"],
            "behaviors": ["lick", "choice"],
        }
        features = compute_pairing_features(ds, ds)
        assert features.task_similarity == 1.0
        assert features.modality_compatibility == 1.0
        assert features.species_relationship == 1.0

    def test_completely_different_datasets(self):
        """Completely different datasets should have low overlap."""
        ds_a = {
            "tasks": ["go_nogo"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
            "brain_regions": ["pfc"],
        }
        ds_b = {
            "tasks": ["speech_production"],
            "modalities": ["fmri"],
            "species": ["human"],
            "brain_regions": ["auditory_cortex"],
        }
        features = compute_pairing_features(ds_a, ds_b)
        assert features.task_similarity == 0.0
        assert features.species_relationship == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give intermediate scores."""
        ds_a = {
            "tasks": ["go_nogo", "reversal_learning"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
        }
        ds_b = {
            "tasks": ["go_nogo", "visual_decision_making"],
            "modalities": ["calcium_imaging"],
            "species": ["mouse"],
        }
        features = compute_pairing_features(ds_a, ds_b)
        # Jaccard of {go_nogo, reversal} and {go_nogo, visual_dm} = 1/3
        assert 0.2 < features.task_similarity < 0.5
        assert features.species_relationship == 1.0

    def test_novelty_bonus_different_sources(self):
        """Different sources should get novelty bonus."""
        ds_a = {"source": "dandi", "tasks": []}
        ds_b = {"source": "openneuro", "tasks": []}
        features = compute_pairing_features(ds_a, ds_b)
        assert features.novelty_bonus > 0

    def test_no_novelty_same_source(self):
        """Same source should not get novelty bonus."""
        ds_a = {"source": "dandi", "tasks": []}
        ds_b = {"source": "dandi", "tasks": []}
        features = compute_pairing_features(ds_a, ds_b)
        assert features.novelty_bonus == 0


class TestPairClassification:
    """Test pair type classification."""

    def test_same_task_different_region(self):
        """Classify same task, different region."""
        features = PairingFeatures(
            task_similarity=0.8,
            region_overlap=0.1,
            species_relationship=1.0,
        )
        ds_a = {"tasks": ["go_nogo"], "brain_regions": ["pfc"]}
        ds_b = {"tasks": ["go_nogo"], "brain_regions": ["striatum"]}
        pair_type = classify_pair_type(features, ds_a, ds_b)
        assert pair_type == "same_task_different_region"

    def test_same_task_different_species(self):
        """Classify same task, different species."""
        features = PairingFeatures(
            task_similarity=0.8,
            species_relationship=0.0,
            region_overlap=0.5,
        )
        ds_a = {"tasks": ["go_nogo"], "species": ["mouse"]}
        ds_b = {"tasks": ["go_nogo"], "species": ["macaque"]}
        pair_type = classify_pair_type(features, ds_a, ds_b)
        assert pair_type == "same_task_different_species"

    def test_shared_affordance(self):
        """Classify by shared affordance."""
        features = PairingFeatures(
            task_similarity=0.2,
            shared_affordances=0.5,
        )
        ds_a = {}
        ds_b = {}
        pair_type = classify_pair_type(features, ds_a, ds_b)
        assert pair_type == "shared_affordance"


class TestCompatibilityScore:
    """Test compatibility score computation."""

    def test_perfect_match(self):
        """Perfect match should have high score."""
        features = PairingFeatures(
            task_similarity=1.0,
            modality_compatibility=1.0,
            species_relationship=1.0,
            region_overlap=1.0,
            shared_events=1.0,
            shared_affordances=1.0,
            provenance_confidence=1.0,
            novelty_bonus=0.1,
            incompatibility_penalty=0.0,
        )
        score = compute_compatibility_score(features)
        assert score > 0.9

    def test_no_match(self):
        """No match should have low score."""
        features = PairingFeatures(
            task_similarity=0.0,
            modality_compatibility=0.0,
            species_relationship=0.0,
            region_overlap=0.0,
            shared_events=0.0,
            shared_affordances=0.0,
            provenance_confidence=0.0,
            novelty_bonus=0.0,
            incompatibility_penalty=0.2,
        )
        score = compute_compatibility_score(features)
        assert score < 0.2

    def test_score_bounded(self):
        """Score should be bounded [0, 1]."""
        features = PairingFeatures(
            task_similarity=2.0,  # Invalid but test bounds
            novelty_bonus=1.0,
        )
        score = compute_compatibility_score(features)
        assert 0.0 <= score <= 1.0


class TestRationaleAndUseCases:
    """Test rationale and use case generation."""

    def test_rationale_includes_high_similarity(self):
        """Rationale should mention high similarity."""
        features = PairingFeatures(task_similarity=0.8)
        rationale = generate_rationale(features, "same_task_different_region")
        assert any("task" in r.lower() for r in rationale)

    def test_use_cases_for_cross_species(self):
        """Cross-species should have appropriate use cases."""
        use_cases = generate_use_cases("same_task_different_species", PairingFeatures())
        assert any("species" in uc.lower() for uc in use_cases)

    def test_use_cases_for_shared_affordance(self):
        """Shared affordance should have method validation use cases."""
        use_cases = generate_use_cases("shared_affordance", PairingFeatures())
        assert any("method" in uc.lower() or "validation" in uc.lower() for uc in use_cases)


class TestPairScoring:
    """Test end-to-end pair scoring."""

    def test_score_dataset_pair(self):
        """Score a pair of datasets."""
        ds_a = {
            "id": "ds001",
            "tasks": ["go_nogo"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
            "brain_regions": ["pfc"],
        }
        ds_b = {
            "id": "ds002",
            "tasks": ["go_nogo"],
            "modalities": ["calcium_imaging"],
            "species": ["mouse"],
            "brain_regions": ["striatum"],
        }
        pair = score_dataset_pair(ds_a, ds_b)
        assert pair.dataset_a_id == "ds001"
        assert pair.dataset_b_id == "ds002"
        assert 0.0 <= pair.compatibility_score <= 1.0
        assert len(pair.rationale) > 0


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def mock_pair(self):
        """Create mock pair."""
        return DatasetPair(
            dataset_a_id="ds001",
            dataset_b_id="ds002",
            pair_type="same_task_different_region",
            compatibility_score=0.75,
            features=PairingFeatures(task_similarity=0.8, region_overlap=0.1),
            rationale=["High task similarity", "Different brain regions"],
            use_cases=["Compare neural representations"],
        )

    def test_markdown_has_sections(self, mock_pair):
        """Markdown should have expected sections."""
        from neural_search.evaluation.cross_dataset_pairing import (
            CrossDatasetPairingReport,
        )

        report = CrossDatasetPairingReport(
            generated_at="2026-01-01T00:00:00",
            total_datasets=10,
            total_pairs_evaluated=45,
            top_pairs=[mock_pair],
            pairs_by_type={"same_task_different_region": [mock_pair]},
            summary={"mean_compatibility": 0.5},
        )
        md = generate_pairing_markdown(report)
        assert "# Cross-Dataset Pairing Report" in md
        assert "## Summary" in md
        assert "## Top Dataset Pairs" in md

    def test_json_valid(self, mock_pair):
        """JSON should be valid."""
        from neural_search.evaluation.cross_dataset_pairing import (
            CrossDatasetPairingReport,
        )

        report = CrossDatasetPairingReport(
            generated_at="2026-01-01T00:00:00",
            total_datasets=10,
            total_pairs_evaluated=45,
            top_pairs=[mock_pair],
            pairs_by_type={"same_task_different_region": [mock_pair]},
            summary={"mean_compatibility": 0.5},
        )
        json_str = generate_pairing_json(report)
        data = json.loads(json_str)
        assert data["total_pairs_evaluated"] == 45
        assert len(data["top_pairs"]) == 1
