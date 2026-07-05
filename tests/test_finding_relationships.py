"""Tests for neural_search.literature.relationship_builder."""

from __future__ import annotations

import json
from pathlib import Path

from neural_search.literature.normalizer import (
    normalize_cell_types,
    normalize_molecules,
)
from neural_search.literature.relationship_builder import (
    build_consensus_summaries,
    build_cross_finding_edges,
    build_qualified_consensus_summaries,
    build_region_cooccurrence_edges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_findings(findings: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for f in findings:
            fh.write(json.dumps(f) + "\n")


def _base_finding(
    paper_id: str,
    finding_id: str,
    regions: list[str],
    direction: str,
    tasks: list[str] | None = None,
) -> dict:
    return {
        "paper_id": paper_id,
        "finding_id": finding_id,
        "finding_text": f"Finding in {regions}",
        "result_direction": direction,
        "regions": regions,
        "species": ["human"],
        "tasks": tasks or [],
        "cell_types": [],
        "molecules": [],
        "confidence": 0.9,
    }


# ---------------------------------------------------------------------------
# Cross-finding edge tests
# ---------------------------------------------------------------------------


class TestBuildCrossFindingEdges:
    def test_supports_edge_same_direction_different_papers(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        supports = [e for e in edges if e.edge_type == "supports"]
        assert len(supports) >= 1
        assert supports[0].shared_regions == ["hippocampus"]

    def test_contradicts_edge_opposite_directions(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["amygdala"], "increase"),
            _base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        contradicts = [e for e in edges if e.edge_type == "contradicts"]
        assert len(contradicts) >= 1

    def test_same_paper_not_compared(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
            _base_finding("p1", "p1:f1", ["hippocampus"], "decrease"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        assert len(edges) == 0

    def test_no_region_overlap_produces_no_edge(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
            _base_finding("p2", "p2:f0", ["cerebellum"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f, min_shared_regions=1)
        assert len(edges) == 0

    def test_empty_findings(self, tmp_path):
        f = tmp_path / "findings.jsonl"
        _write_findings([], f)
        edges = build_cross_finding_edges(f)
        assert edges == []

    def test_correlation_direction_not_supports(self, tmp_path):
        # "correlation" direction should not create supports edges
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "correlation"),
            _base_finding("p2", "p2:f0", ["hippocampus"], "correlation"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        supports = [e for e in edges if e.edge_type == "supports"]
        assert len(supports) == 0

    def test_negated_finding_does_not_create_supports_edge(self, tmp_path):
        findings = [
            {**_base_finding("p1", "p1:f0", ["hippocampus"], "increase"), "negation": True},
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        supports = [e for e in edges if e.edge_type == "supports"]
        assert len(supports) == 0

    def test_non_negated_findings_still_create_supports_edge(self, tmp_path):
        findings = [
            {**_base_finding("p1", "p1:f0", ["hippocampus"], "increase"), "negation": False},
            {**_base_finding("p2", "p2:f0", ["hippocampus"], "increase"), "negation": False},
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        supports = [e for e in edges if e.edge_type == "supports"]
        assert len(supports) >= 1

    def test_contradicts_edge_has_opposite_direction_subtype_by_default(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["amygdala"], "increase"),
            _base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        contradicts = [e for e in edges if e.edge_type == "contradicts"]
        assert len(contradicts) == 1
        assert contradicts[0].contradiction_subtype == "opposite_direction"

    def test_negated_finding_yields_direct_refutation_subtype(self, tmp_path):
        findings = [
            {**_base_finding("p1", "p1:f0", ["amygdala"], "increase"), "negation": True},
            _base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        contradicts = [e for e in edges if e.edge_type == "contradicts"]
        assert len(contradicts) == 1
        assert contradicts[0].contradiction_subtype == "direct_refutation"

    def test_supports_edge_has_no_contradiction_subtype(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        supports = [e for e in edges if e.edge_type == "supports"]
        assert supports[0].contradiction_subtype is None

    def test_mismatched_frequency_band_blocks_false_contradiction(self, tmp_path):
        # "theta increases" vs "gamma decreases" in the same region are two
        # different signals, not opposing evidence about the same phenomenon.
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["amygdala"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
                "frequency_band": ["gamma"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        assert len([e for e in edges if e.edge_type == "contradicts"]) == 0

    def test_matching_frequency_band_still_contradicts(self, tmp_path):
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["amygdala"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
                "frequency_band": ["theta"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        contradicts = [e for e in edges if e.edge_type == "contradicts"]
        assert len(contradicts) == 1

    def test_one_sided_frequency_band_does_not_block_contradiction(self, tmp_path):
        # Only one finding has frequency_band populated — not enough
        # information to claim a mismatch, so today's region-only behavior
        # still applies.
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["amygdala"], "increase"),
                "frequency_band": ["theta"],
            },
            _base_finding("p2", "p2:f0", ["amygdala"], "decrease"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f)
        assert len([e for e in edges if e.edge_type == "contradicts"]) == 1

    def test_max_edges_cap_respected(self, tmp_path):
        # 10 findings from different papers all sharing a region → many pairs
        findings = [
            _base_finding(f"p{i}", f"p{i}:f0", ["hippocampus"], "increase")
            for i in range(10)
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_cross_finding_edges(f, max_edges=5)
        assert len(edges) <= 5


# ---------------------------------------------------------------------------
# Region co-occurrence edge tests
# ---------------------------------------------------------------------------


class TestBuildRegionCooccurrenceEdges:
    def test_cooccurrence_detected(self, tmp_path):
        findings = [
            {**_base_finding("p1", "p1:f0", ["hippocampus", "amygdala"], "increase")},
            {**_base_finding("p2", "p2:f0", ["hippocampus", "amygdala"], "increase")},
            {**_base_finding("p3", "p3:f0", ["hippocampus", "amygdala"], "decrease")},
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_region_cooccurrence_edges(f, min_cooccurrences=2)
        assert len(edges) == 1
        assert edges[0].region_a == "amygdala"
        assert edges[0].region_b == "hippocampus"
        assert edges[0].n_findings == 3

    def test_single_region_finding_excluded(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_region_cooccurrence_edges(f, min_cooccurrences=1)
        assert len(edges) == 0

    def test_min_cooccurrences_threshold(self, tmp_path):
        findings = [
            {**_base_finding("p1", "p1:f0", ["amygdala", "thalamus"], "increase")},
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_region_cooccurrence_edges(f, min_cooccurrences=2)
        assert len(edges) == 0

    def test_edges_sorted_by_frequency(self, tmp_path):
        findings = [
            {**_base_finding(f"p{i}", f"p{i}:f0", ["amygdala", "hippocampus"], "increase")}
            for i in range(5)
        ] + [
            {**_base_finding(f"q{i}", f"q{i}:f0", ["cortex", "thalamus"], "decrease")}
            for i in range(2)
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        edges = build_region_cooccurrence_edges(f, min_cooccurrences=2)
        assert edges[0].n_findings >= edges[-1].n_findings


# ---------------------------------------------------------------------------
# Consensus summary tests
# ---------------------------------------------------------------------------


class TestBuildConsensusSummaries:
    def test_consensus_record_created(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase", tasks=["navigation"]),
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase", tasks=["navigation"]),
            _base_finding("p3", "p3:f0", ["hippocampus"], "increase", tasks=["navigation"]),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_consensus_summaries(f, min_papers=2)
        assert len(records) >= 1
        top = records[0]
        assert top.region == "hippocampus"
        assert top.direction == "increase"
        assert top.n_papers >= 2
        assert top.facet_fields == []
        assert top.specificity_tier == "base"
        assert top.qualifier_values == {}

    def test_contested_finding_has_lower_strength(self, tmp_path):
        # 3 increase vs 2 decrease → strength for increase = 3/5 = 0.6
        findings = [
            _base_finding(f"p{i}", f"p{i}:f0", ["amygdala"], "increase")
            for i in range(3)
        ] + [
            _base_finding(f"q{i}", f"q{i}:f0", ["amygdala"], "decrease")
            for i in range(2)
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_consensus_summaries(f, min_papers=2)
        increase_records = [r for r in records if r.direction == "increase" and r.region == "amygdala"]
        if increase_records:
            assert increase_records[0].consensus_strength < 1.0

    def test_min_papers_threshold(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["cerebellum"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_consensus_summaries(f, min_papers=2)
        assert len(records) == 0


class TestBuildQualifiedConsensusSummaries:
    def test_qualified_record_created_for_shared_frequency_band(self, tmp_path):
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_qualified_consensus_summaries(f, min_papers=2)
        band_records = [r for r in records if r.facet_fields == ["frequency_band"]]
        assert len(band_records) == 1
        assert band_records[0].qualifier_values == {"frequency_band": "theta"}
        assert band_records[0].specificity_tier == "qualified"
        assert band_records[0].n_papers == 2

    def test_findings_without_qualifier_field_excluded_from_that_tier(self, tmp_path):
        findings = [
            _base_finding("p1", "p1:f0", ["hippocampus"], "increase"),  # no frequency_band
            _base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_qualified_consensus_summaries(f, min_papers=2, qualifier_fields=("frequency_band",))
        assert records == []

    def test_does_not_duplicate_base_tier_records(self, tmp_path):
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_qualified_consensus_summaries(f, min_papers=2)
        assert all(r.specificity_tier == "qualified" for r in records)

    def test_different_qualifier_values_split_into_separate_records(self, tmp_path):
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
            },
            {
                **_base_finding("p3", "p3:f0", ["hippocampus"], "increase"),
                "frequency_band": ["gamma"],
            },
            {
                **_base_finding("p4", "p4:f0", ["hippocampus"], "increase"),
                "frequency_band": ["gamma"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_qualified_consensus_summaries(
            f, min_papers=2, qualifier_fields=("frequency_band",)
        )
        band_values = {r.qualifier_values["frequency_band"] for r in records}
        assert band_values == {"theta", "gamma"}

    def test_multiple_qualifier_fields_each_produce_own_tier(self, tmp_path):
        findings = [
            {
                **_base_finding("p1", "p1:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
                "injury_model": ["alzheimer_app"],
            },
            {
                **_base_finding("p2", "p2:f0", ["hippocampus"], "increase"),
                "frequency_band": ["theta"],
                "injury_model": ["alzheimer_app"],
            },
        ]
        f = tmp_path / "findings.jsonl"
        _write_findings(findings, f)
        records = build_qualified_consensus_summaries(
            f, min_papers=2, qualifier_fields=("frequency_band", "injury_model")
        )
        facets = {tuple(r.facet_fields) for r in records}
        assert facets == {("frequency_band",), ("injury_model",)}


# ---------------------------------------------------------------------------
# Cell type and molecule normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeCellTypes:
    def test_neurons_plural(self):
        result, changed = normalize_cell_types(["neurons"])
        assert result == ["neuron"]
        assert changed

    def test_pyramidal_neurons_canonical(self):
        result, changed = normalize_cell_types(["pyramidal neurons"])
        assert result == ["pyramidal cell"]
        assert changed

    def test_pv_interneurons_expanded(self):
        result, changed = normalize_cell_types(["pv interneurons"])
        assert "parvalbumin interneuron" in result

    def test_place_cells_unchanged(self):
        result, _ = normalize_cell_types(["place cells"])
        assert result == ["place cell"]

    def test_dedup(self):
        result, _ = normalize_cell_types(["neurons", "neuron"])
        assert result.count("neuron") == 1

    def test_unknown_passthrough(self):
        result, changed = normalize_cell_types(["chandelier cells"])
        assert result == ["chandelier cells"]


class TestNormalizeMolecules:
    def test_ca2plus_to_calcium(self):
        result, changed = normalize_molecules(["Ca2+"])
        assert "calcium" in result
        assert changed

    def test_5ht_to_serotonin(self):
        result, changed = normalize_molecules(["5-HT"])
        assert "serotonin" in result

    def test_alpha_synuclein_variants(self):
        r1, _ = normalize_molecules(["α-synuclein"])
        r2, _ = normalize_molecules(["alpha-synuclein"])
        assert r1 == r2 == ["alpha-synuclein"]

    def test_nmda_canonical(self):
        result, _ = normalize_molecules(["NMDA receptors"])
        assert "NMDA receptor" in result

    def test_gaba_case_insensitive(self):
        result, _ = normalize_molecules(["gaba"])
        assert "GABA" in result

    def test_unknown_molecule_passthrough(self):
        result, _ = normalize_molecules(["myelin basic protein"])
        assert result == ["myelin basic protein"]
