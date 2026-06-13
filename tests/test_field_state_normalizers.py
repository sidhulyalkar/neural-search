"""Tests for species, modality, brain region, and raw/processed data normalizers."""

from __future__ import annotations

import pytest

from neural_search.field_state.normalizers import (
    extract_modalities_from_text,
    extract_processed_data_evidence,
    extract_raw_data_evidence,
    extract_regions_from_text,
    extract_species_from_text,
    modality_satisfies_affordance,
    normalize_brain_region,
    normalize_modality,
    normalize_species,
    regions_are_equivalent,
    species_are_interchangeable,
)


class TestSpeciesNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("mouse", "mouse"),
        ("mice", "mouse"),
        ("Mus musculus", "mouse"),
        ("rat", "rat"),
        ("Rattus norvegicus", "rat"),
        ("human", "human"),
        ("participants", "human"),
        ("patients", "human"),
        ("macaque", "macaque"),
        ("monkey", "macaque"),
        ("NHP", "non_human_primate"),
        ("zebrafish", "zebrafish"),
        ("Drosophila", "drosophila"),
        ("fruit fly", "drosophila"),
    ])
    def test_canonical_species(self, raw: str, expected: str) -> None:
        result = normalize_species(raw)
        assert result == expected, f"normalize_species({raw!r}) → {result!r}, expected {expected!r}"

    def test_unknown_species_returns_none(self) -> None:
        assert normalize_species("zebrafinch") is None
        assert normalize_species("") is None

    def test_extract_species_from_text(self) -> None:
        text = "Recordings from mice and rats in a navigation task"
        results = extract_species_from_text(text)
        canonicals = [c for c, _ in results]
        assert "mouse" in canonicals
        assert "rat" in canonicals

    def test_extract_species_deduplicates(self) -> None:
        text = "mouse mice Mus musculus recordings"
        results = extract_species_from_text(text)
        canonicals = [c for c, _ in results]
        assert canonicals.count("mouse") == 1


class TestModalityNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("neuropixels", "extracellular_ephys"),
        ("spike sorting", "extracellular_ephys"),
        ("LFP", "lfp"),
        ("local field potential", "lfp"),
        ("two-photon", "calcium_imaging"),
        ("GCaMP", "calcium_imaging"),
        ("calcium imaging", "calcium_imaging"),
        ("fMRI", "fmri"),
        ("BOLD", "fmri"),
        ("EEG", "eeg"),
        ("patch clamp", "intracellular_ephys"),
        ("MEG", "meg"),
        ("ECoG", "ecog"),
    ])
    def test_canonical_modality(self, raw: str, expected: str) -> None:
        result = normalize_modality(raw)
        assert result == expected

    def test_unknown_modality_returns_none(self) -> None:
        assert normalize_modality("unknown_recording") is None

    def test_extract_modalities_from_description(self) -> None:
        text = "Two-photon calcium imaging and LFP recordings in mice"
        results = extract_modalities_from_text(text)
        canonicals = [c for c, _ in results]
        assert "calcium_imaging" in canonicals
        assert "lfp" in canonicals


class TestBrainRegionNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("CA1", "hippocampus_ca1"),
        ("CA3", "hippocampus_ca3"),
        ("dentate gyrus", "dentate_gyrus"),
        ("DG", "dentate_gyrus"),
        ("hippocampus", "hippocampus"),
        ("MEC", "entorhinal_cortex"),
        ("V1", "visual_cortex"),
        ("VISp", "visual_cortex"),
        ("LIP", "lateral_intraparietal"),
        ("MT", "mt_v5"),
        ("V5", "mt_v5"),
        ("PFC", "prefrontal_cortex"),
        ("mPFC", "prefrontal_cortex"),
        ("dlPFC", "prefrontal_cortex"),
        ("striatum", "striatum"),
        ("caudate", "striatum"),
        ("putamen", "striatum"),
    ])
    def test_canonical_region(self, raw: str, expected: str) -> None:
        result = normalize_brain_region(raw)
        assert result == expected

    def test_unknown_region_returns_none(self) -> None:
        assert normalize_brain_region("zona incerta") is None

    def test_extract_regions_from_text(self) -> None:
        text = "Neuropixels probe in CA1 and MEC during navigation"
        results = extract_regions_from_text(text)
        canonicals = [c for c, _ in results]
        assert "hippocampus_ca1" in canonicals
        assert "entorhinal_cortex" in canonicals


class TestRawVsProcessedExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("Raw AP-band data from Neuropixels probes", True),
        ("raw voltage traces", True),
        ("continuous data from silicon probes", True),
        (".ap.bin files included", True),
        ("spike times only — no raw waveforms", False),
        ("preprocessed and analysis-ready", None),
        ("", None),
    ])
    def test_extract_raw_data(self, text: str, expected) -> None:
        available, evidence = extract_raw_data_evidence(text)
        assert available == expected, f"extract_raw_data_evidence({text!r}) → {available!r}, expected {expected!r}"

    def test_raw_returns_matched_phrases(self) -> None:
        _, evidence = extract_raw_data_evidence("Contains raw AP data and .ap.bin files")
        assert len(evidence) >= 1

    def test_negation_overrides_positive_keywords(self) -> None:
        available, _ = extract_raw_data_evidence("This dataset has no raw recordings, only spike times")
        assert available is False

    @pytest.mark.parametrize("text,expected", [
        ("spike times only from sorted units", True),
        ("preprocessed ROI traces", True),
        ("derivatives only", True),
        ("raw continuous data", None),
    ])
    def test_extract_processed_data(self, text: str, expected) -> None:
        available, _ = extract_processed_data_evidence(text)
        assert available == expected


class TestGuardrails:
    def test_calcium_does_not_satisfy_spike_affordance(self) -> None:
        assert not modality_satisfies_affordance("calcium_imaging", "extracellular_ephys")
        assert not modality_satisfies_affordance("calcium_imaging", "spike_sorting")

    def test_fmri_does_not_satisfy_spike_affordance(self) -> None:
        assert not modality_satisfies_affordance("fmri", "extracellular_ephys")
        assert not modality_satisfies_affordance("fmri", "lfp")

    def test_ephys_satisfies_spike_affordance(self) -> None:
        assert modality_satisfies_affordance("extracellular_ephys", "extracellular_ephys")

    def test_human_mouse_not_interchangeable(self) -> None:
        assert not species_are_interchangeable("human", "mouse")
        assert not species_are_interchangeable("human", "rat")

    def test_mouse_rat_interchangeable_by_default(self) -> None:
        # Mouse and rat are different but not explicitly blocked
        assert species_are_interchangeable("mouse", "rat")

    def test_mt_v5_not_equivalent_to_lip(self) -> None:
        assert not regions_are_equivalent("mt_v5", "lateral_intraparietal")

    def test_pfc_not_equivalent_to_lip(self) -> None:
        assert not regions_are_equivalent("prefrontal_cortex", "lateral_intraparietal")

    def test_entorhinal_not_equivalent_to_hippocampus(self) -> None:
        assert not regions_are_equivalent("entorhinal_cortex", "hippocampus")

    def test_ca1_not_blocked_from_ca1(self) -> None:
        assert regions_are_equivalent("hippocampus_ca1", "hippocampus_ca1")
