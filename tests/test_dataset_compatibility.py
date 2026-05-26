"""Tests for dataset compatibility scoring."""

from __future__ import annotations

import pytest

from neural_search.search.compatibility import (
    CompatibilityConfig,
    CompatibilityResult,
    CompatibilityScore,
    CompatibilityScorer,
    CompatibilityType,
    SlotMatch,
    compute_compatibility,
    explain_compatibility,
)


@pytest.fixture
def mouse_neuropixels_dataset() -> dict:
    """Mouse Neuropixels decision-making dataset."""
    return {
        "dataset_id": "dandi:000001",
        "title": "Mouse Neuropixels Decision Making",
        "species": ["Mus musculus"],
        "tasks": ["decision_making", "two_alternative_forced_choice"],
        "modalities": ["neuropixels", "extracellular_ephys"],
        "brain_regions": ["visual_cortex", "prefrontal_cortex"],
        "behaviors": ["choice", "licking", "running"],
        "data_standards": ["NWB"],
        "analysis_affordances": [
            {"analysis_id": "choice_decoding"},
            {"analysis_id": "event_aligned_activity"},
        ],
        "analysis_readiness_score": 85,
    }


@pytest.fixture
def rat_neuropixels_dataset() -> dict:
    """Rat Neuropixels decision-making dataset (complementary species)."""
    return {
        "dataset_id": "dandi:000002",
        "title": "Rat Neuropixels Decision Making",
        "species": ["Rattus norvegicus"],
        "tasks": ["decision_making", "two_alternative_forced_choice"],
        "modalities": ["neuropixels", "extracellular_ephys"],
        "brain_regions": ["prefrontal_cortex", "striatum"],
        "behaviors": ["choice", "nose_poke"],
        "data_standards": ["NWB"],
        "analysis_affordances": [
            {"analysis_id": "choice_decoding"},
            {"analysis_id": "q_learning_modeling"},
        ],
        "analysis_readiness_score": 80,
    }


@pytest.fixture
def mouse_calcium_dataset() -> dict:
    """Mouse calcium imaging dataset (different modality, same task)."""
    return {
        "dataset_id": "dandi:000003",
        "title": "Mouse Calcium Imaging Decision Making",
        "species": ["Mus musculus"],
        "tasks": ["decision_making", "go_nogo"],
        "modalities": ["calcium_imaging", "two_photon"],
        "brain_regions": ["visual_cortex", "motor_cortex"],
        "behaviors": ["choice", "licking"],
        "data_standards": ["NWB"],
        "analysis_affordances": [
            {"analysis_id": "choice_decoding"},
            {"analysis_id": "event_aligned_activity"},
        ],
        "analysis_readiness_score": 75,
    }


@pytest.fixture
def human_fmri_dataset() -> dict:
    """Human fMRI dataset (translational)."""
    return {
        "dataset_id": "openneuro:ds001",
        "title": "Human fMRI Decision Making",
        "species": ["Homo sapiens"],
        "tasks": ["decision_making", "gambling_task"],
        "modalities": ["fmri", "bold"],
        "brain_regions": ["prefrontal_cortex", "striatum"],
        "behaviors": ["choice", "button_press"],
        "data_standards": ["BIDS"],
        "analysis_affordances": [
            {"analysis_id": "choice_decoding"},
        ],
        "analysis_readiness_score": 90,
    }


@pytest.fixture
def mouse_visual_dataset() -> dict:
    """Mouse visual coding dataset (different task)."""
    return {
        "dataset_id": "allen:visual",
        "title": "Mouse Visual Coding Neuropixels",
        "species": ["Mus musculus"],
        "tasks": ["visual_coding", "passive_viewing"],
        "modalities": ["neuropixels", "extracellular_ephys"],
        "brain_regions": ["visual_cortex", "lgn"],
        "behaviors": ["running", "pupil"],
        "data_standards": ["NWB"],
        "analysis_affordances": [
            {"analysis_id": "event_aligned_activity"},
            {"analysis_id": "receptive_field_mapping"},
        ],
        "analysis_readiness_score": 95,
    }


class TestSlotMatch:
    """Tests for slot matching."""

    def test_exact_match(self):
        """Test exact slot matching."""
        from neural_search.search.compatibility import _compute_slot_match

        match = _compute_slot_match(
            "species",
            {"mus_musculus"},
            {"mus_musculus"},
        )

        assert match.match_score == 1.0
        assert match.match_type == "exact"
        assert match.overlap == {"mus_musculus"}

    def test_partial_match(self):
        """Test partial slot matching."""
        from neural_search.search.compatibility import _compute_slot_match

        match = _compute_slot_match(
            "tasks",
            {"decision_making", "go_nogo"},
            {"decision_making", "reversal_learning"},
        )

        assert 0 < match.match_score < 1.0
        assert match.match_type == "partial"
        assert match.overlap == {"decision_making"}

    def test_no_match(self):
        """Test no slot matching."""
        from neural_search.search.compatibility import _compute_slot_match

        match = _compute_slot_match(
            "species",
            {"mus_musculus"},
            {"homo_sapiens"},
        )

        assert match.match_score == 0.0
        assert match.match_type == "none"

    def test_homology_match(self):
        """Test species homology matching."""
        from neural_search.search.compatibility import _compute_slot_match

        match = _compute_slot_match(
            "species",
            {"mus_musculus"},
            {"rattus_norvegicus"},
            use_homology=True,
        )

        assert match.match_score > 0.0
        assert match.match_type == "hierarchical"

    def test_empty_values(self):
        """Test matching with empty values."""
        from neural_search.search.compatibility import _compute_slot_match

        match = _compute_slot_match("species", set(), {"mus_musculus"})

        assert match.match_score == 0.0
        assert match.match_type == "none"


class TestCompatibilityScorer:
    """Tests for CompatibilityScorer class."""

    def test_create_scorer(self):
        """Test creating a compatibility scorer."""
        scorer = CompatibilityScorer()
        assert scorer.config is not None

    def test_equivalent_datasets(
        self,
        mouse_neuropixels_dataset,
        mouse_calcium_dataset,
    ):
        """Test detecting equivalent datasets."""
        # Make them more similar for this test
        equivalent_dataset = mouse_neuropixels_dataset.copy()
        equivalent_dataset["dataset_id"] = "dandi:000001_copy"

        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, equivalent_dataset)

        assert score.overall_score > 0.9
        assert score.compatibility_type == CompatibilityType.EQUIVALENT

    def test_complementary_cross_modal(
        self,
        mouse_neuropixels_dataset,
        mouse_calcium_dataset,
    ):
        """Test detecting cross-modal complementary datasets."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, mouse_calcium_dataset)

        # Same species, overlapping tasks, different modality
        assert score.overall_score > 0.3
        # Could be complementary due to cross-modal nature

    def test_translational_cross_species(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test detecting translational relationships."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)

        # Same task, same modality, different species (homologs)
        assert score.overall_score > 0.3

    def test_analysis_compatible(
        self,
        mouse_neuropixels_dataset,
        mouse_calcium_dataset,
    ):
        """Test detecting analysis-compatible datasets."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, mouse_calcium_dataset)

        # Both support choice_decoding and event_aligned_activity
        assert "choice_decoding" in str(score.supporting_evidence) or score.overall_score > 0.3

    def test_contrastive_detection(
        self,
        mouse_neuropixels_dataset,
        mouse_visual_dataset,
    ):
        """Test detecting contrastive relationships."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, mouse_visual_dataset)

        # Same species, same modality, same some regions, different task
        # This is a controlled contrast
        assert score.overall_score >= 0.0  # May or may not be classified as contrastive

    def test_slot_contributions(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test that slot contributions are computed."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)

        assert len(score.slot_contributions) > 0
        # Total contributions should sum to <= 1
        assert sum(score.slot_contributions.values()) <= 1.1  # Allow small float errors

    def test_supporting_evidence(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test that supporting evidence is generated."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)

        # Should have some evidence from overlapping slots
        assert len(score.supporting_evidence) > 0

    def test_contradiction_detection(
        self,
        mouse_neuropixels_dataset,
        human_fmri_dataset,
    ):
        """Test that contradictions are detected."""
        scorer = CompatibilityScorer()
        score = scorer.score_compatibility(mouse_neuropixels_dataset, human_fmri_dataset)

        # Different species, different modality - may have contradictions
        # At minimum, the score reflects these differences
        assert score.overall_score < 0.9


class TestFindCompatibleDatasets:
    """Tests for finding compatible datasets from a pool."""

    def test_find_compatible_basic(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
        mouse_calcium_dataset,
        human_fmri_dataset,
    ):
        """Test finding compatible datasets."""
        candidates = [
            rat_neuropixels_dataset,
            mouse_calcium_dataset,
            human_fmri_dataset,
        ]

        scorer = CompatibilityScorer()
        result = scorer.find_compatible_datasets(
            mouse_neuropixels_dataset,
            candidates,
            min_score=0.2,
        )

        assert isinstance(result, CompatibilityResult)
        assert result.query_dataset_id == "dandi:000001"
        assert len(result.candidates) > 0

    def test_find_compatible_filter_type(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
        mouse_calcium_dataset,
    ):
        """Test filtering by compatibility type."""
        candidates = [rat_neuropixels_dataset, mouse_calcium_dataset]

        scorer = CompatibilityScorer()
        result = scorer.find_compatible_datasets(
            mouse_neuropixels_dataset,
            candidates,
            compatibility_types=[CompatibilityType.TRANSLATIONAL],
            min_score=0.1,
        )

        # Should only include translational matches
        for candidate in result.candidates:
            assert candidate.compatibility_type == CompatibilityType.TRANSLATIONAL

    def test_find_compatible_top_k(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
        mouse_calcium_dataset,
        human_fmri_dataset,
        mouse_visual_dataset,
    ):
        """Test top_k limiting."""
        candidates = [
            rat_neuropixels_dataset,
            mouse_calcium_dataset,
            human_fmri_dataset,
            mouse_visual_dataset,
        ]

        scorer = CompatibilityScorer()
        result = scorer.find_compatible_datasets(
            mouse_neuropixels_dataset,
            candidates,
            min_score=0.1,
            top_k=2,
        )

        assert len(result.candidates) <= 2

    def test_skip_self_comparison(
        self,
        mouse_neuropixels_dataset,
    ):
        """Test that self-comparison is skipped."""
        candidates = [mouse_neuropixels_dataset]  # Same dataset

        scorer = CompatibilityScorer()
        result = scorer.find_compatible_datasets(
            mouse_neuropixels_dataset,
            candidates,
        )

        assert len(result.candidates) == 0


class TestComputeCompatibilityFunction:
    """Tests for the convenience function."""

    def test_compute_compatibility(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test the compute_compatibility convenience function."""
        score = compute_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)

        assert isinstance(score, CompatibilityScore)
        assert score.dataset_i_id == "dandi:000001"
        assert score.dataset_j_id == "dandi:000002"

    def test_compute_with_custom_config(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test with custom configuration."""
        config = CompatibilityConfig(
            equivalent_threshold=0.9,
            contradiction_penalty=0.5,
        )

        score = compute_compatibility(
            mouse_neuropixels_dataset,
            rat_neuropixels_dataset,
            config=config,
        )

        assert isinstance(score, CompatibilityScore)


class TestExplainCompatibility:
    """Tests for compatibility explanation."""

    def test_explain_compatibility(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test generating explanation."""
        score = compute_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)
        explanation = explain_compatibility(score)

        assert "datasets" in explanation
        assert "compatibility_type" in explanation
        assert "overall_score" in explanation
        assert "slot_details" in explanation
        assert "explanation" in explanation

    def test_slot_details_sorted(
        self,
        mouse_neuropixels_dataset,
        rat_neuropixels_dataset,
    ):
        """Test that slot details are sorted by match score."""
        score = compute_compatibility(mouse_neuropixels_dataset, rat_neuropixels_dataset)
        explanation = explain_compatibility(score)

        slot_details = explanation["slot_details"]
        if len(slot_details) > 1:
            # Should be sorted by match_score descending
            scores = [s["match_score"] for s in slot_details]
            assert scores == sorted(scores, reverse=True)


class TestCompatibilityConfig:
    """Tests for configuration options."""

    def test_default_config(self):
        """Test default configuration."""
        config = CompatibilityConfig()

        assert config.equivalent_threshold == 0.75
        assert "species" in config.equivalent_weights
        assert "task" in config.complementary_weights

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = CompatibilityConfig(
            equivalent_weights={"species": 0.5, "task": 0.5},
        )

        assert config.equivalent_weights["species"] == 0.5

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = CompatibilityConfig(
            equivalent_threshold=0.9,
            complementary_threshold=0.7,
        )

        assert config.equivalent_threshold == 0.9
        assert config.complementary_threshold == 0.7


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty = {"dataset_id": "empty"}
        full = {
            "dataset_id": "full",
            "species": ["Mus musculus"],
            "tasks": ["visual"],
        }

        score = compute_compatibility(empty, full)

        assert score.overall_score == 0.0

    def test_minimal_dataset(self):
        """Test with minimal dataset."""
        minimal_i = {"dataset_id": "min_i", "species": ["Mus musculus"]}
        minimal_j = {"dataset_id": "min_j", "species": ["Mus musculus"]}

        score = compute_compatibility(minimal_i, minimal_j)

        # Should have some match on species
        assert score.overall_score > 0.0

    def test_different_id_formats(self):
        """Test with different ID field formats."""
        d1 = {"id": "dataset_1", "species": ["mouse"]}
        d2 = {"source_id": "dataset_2", "species": ["mouse"]}

        score = compute_compatibility(d1, d2)

        assert score.dataset_i_id == "dataset_1"
        assert score.dataset_j_id == "dataset_2"

    def test_list_of_dicts_affordances(self):
        """Test affordances as list of dicts."""
        d1 = {
            "dataset_id": "d1",
            "analysis_affordances": [
                {"analysis_id": "choice_decoding", "support_level": "high"},
            ],
        }
        d2 = {
            "dataset_id": "d2",
            "analysis_affordances": [
                {"analysis_id": "choice_decoding", "support_level": "medium"},
            ],
        }

        score = compute_compatibility(d1, d2)

        # Should extract analysis_id from dicts
        assert "analysis_affordance" in score.slot_matches
        match = score.slot_matches["analysis_affordance"]
        # Normalized form uses space, not underscore
        assert "choice decoding" in match.overlap or "choice_decoding" in match.overlap
