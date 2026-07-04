"""Tests for the methodology registry overlay schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neural_search.graph.method_registry_builder import load_method_registry
from neural_search.kg.schemas.method_registry import (
    MethodAnalysisLink,
    MethodRegistry,
    known_analysis_families,
    known_taxonomy_method_ids,
)


def _valid_link(**overrides):
    payload = {
        "analysis_family": "time_frequency",
        "taxonomy_method_ids": ["fft"],
        "rationale": "FFT computes a time-frequency representation.",
    }
    payload.update(overrides)
    return MethodAnalysisLink(**payload)


def test_valid_link_round_trips():
    link = _valid_link()
    assert link.analysis_family == "time_frequency"
    assert link.taxonomy_method_ids == ["fft"]
    assert link.confidence == 0.75
    assert link.requires_human_review is False


def test_rejects_unknown_analysis_family():
    with pytest.raises(ValidationError):
        _valid_link(analysis_family="not_a_real_analysis_family")


def test_rejects_unknown_taxonomy_method_id():
    with pytest.raises(ValidationError):
        _valid_link(taxonomy_method_ids=["not_a_real_method"])


def test_rejects_empty_rationale():
    with pytest.raises(ValidationError):
        _valid_link(rationale="   ")


def test_rejects_unknown_cross_ref_affordance_id():
    with pytest.raises(ValidationError):
        _valid_link(cross_ref_affordance_id="not_a_real_affordance")


def test_accepts_known_cross_ref_affordance_id():
    link = _valid_link(cross_ref_affordance_id="encoding_modeling")
    assert link.cross_ref_affordance_id == "encoding_modeling"


def test_registry_rejects_duplicate_analysis_family():
    with pytest.raises(ValidationError):
        MethodRegistry(links=[_valid_link(), _valid_link()])


def test_known_analysis_families_is_non_empty():
    families = known_analysis_families()
    assert "time_frequency" in families
    assert "decoding" in families
    assert len(families) >= 20


def test_known_taxonomy_method_ids_is_non_empty():
    ids = known_taxonomy_method_ids()
    assert "fft" in ids
    assert "dcm" in ids
    assert len(ids) >= 25


def test_real_method_registry_yaml_loads_without_error():
    registry = load_method_registry()
    assert isinstance(registry, MethodRegistry)
    assert len(registry.links) >= 10
    families = {link.analysis_family for link in registry.links}
    assert len(families) == len(registry.links)
