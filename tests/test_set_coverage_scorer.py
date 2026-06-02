"""Tests for set-coverage scorer."""
import pytest
from neural_search.retrieval.set_coverage_scorer import (
    SetCoverageScorer,
    SetCoverageResult,
    SetConstraints,
)


def test_import():
    from neural_search.retrieval.set_coverage_scorer import SetCoverageScorer
    assert SetCoverageScorer is not None


def _make_dataset(dataset_id, modalities, species, regions, affordances, usefulness):
    return {
        "dataset_id": dataset_id,
        "modalities": modalities,
        "species": species,
        "brain_regions": regions,
        "affordances": affordances,
        "usefulness_score": usefulness,
    }


def test_score_single_dataset():
    scorer = SetCoverageScorer()
    constraints = SetConstraints(
        required_modalities=["fmri"],
        required_species=["human"],
    )
    datasets = [_make_dataset("ds1", ["fmri"], ["human"], ["cortex"], ["decoding"], 0.7)]
    result = scorer.score_set(datasets, constraints)
    assert isinstance(result, SetCoverageResult)
    assert result.total_score > 0.0


def test_penalizes_hard_negative():
    scorer = SetCoverageScorer()
    constraints = SetConstraints(hard_negative_modalities=["eeg"])
    datasets = [
        _make_dataset("ds1", ["fmri"], ["human"], ["cortex"], [], 0.8),
        _make_dataset("ds2", ["eeg"], ["human"], ["cortex"], [], 0.9),
    ]
    result = scorer.score_set(datasets, constraints)
    violations = result.hard_negative_violations
    assert "ds2" in violations


def test_rewards_modality_diversity():
    scorer = SetCoverageScorer()
    constraints = SetConstraints()
    diverse = [
        _make_dataset("ds1", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds2", ["neuropixels"], ["mouse"], [], [], 0.6),
        _make_dataset("ds3", ["eeg"], ["human"], [], [], 0.6),
    ]
    uniform = [
        _make_dataset("ds4", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds5", ["fmri"], ["human"], [], [], 0.6),
        _make_dataset("ds6", ["fmri"], ["human"], [], [], 0.6),
    ]
    r_diverse = scorer.score_set(diverse, constraints)
    r_uniform = scorer.score_set(uniform, constraints)
    assert r_diverse.coverage_bonus > r_uniform.coverage_bonus


def test_empty_dataset_list():
    scorer = SetCoverageScorer()
    result = scorer.score_set([], SetConstraints())
    assert result.total_score == 0.0
